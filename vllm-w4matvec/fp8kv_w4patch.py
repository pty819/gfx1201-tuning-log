"""Route M==1 W4A8 linear layers through the Triton matvec (fp8kv_w4mv2)."""
import os

def _install():
    if os.environ.get("FP8KV_W4MV", "0") != "1":
        return
    import sys, torch
    try:
        import radiance_mxfp4 as _rm
        from fp8kv_w4mv2 import run as _run
    except Exception as e:
        sys.stderr.write(f"[w4mv] import fail: {e}\n")
        return
    if getattr(_rm, "WPERM", False):
        sys.stderr.write("[w4mv] WPERM=1, standing down\n")
        return
    _hits = [0]
    _orig_aw = None
    _wrap_target = None

    def apply_weights(self, layer, x, bias=None):
        try:
            if _hits[0] == 0:
                _hits[0] = -1
                w0 = layer.weight; s0 = layer.weight_scale
                sys.stderr.write(f"[w4mv] first call: x={tuple(x.shape)}/{x.dtype} "
                                 f"w={tuple(w0.shape)}/{w0.dtype} s={tuple(s0.shape)}/{s0.dtype} "
                                 f"K%256={(w0.shape[1]*2)%256} N%16={w0.shape[0]%16}\n")
            if (torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == 1
                    and x.dtype == torch.bfloat16):
                w = layer.weight
                s = layer.weight_scale
                N = w.shape[0]
                K = w.shape[1] * 2
                if K % 256 == 0 and N % 16 == 0 and s.shape[0] == K // 32:
                    out = torch.empty((1, N), device=x.device, dtype=torch.bfloat16)
                    _run(x, w.view(torch.uint32), s, out,
                         BLOCK_N=16, BLOCK_J=32, warps=8, stages=4)
                    if _hits[0] == 0:
                        _hits[0] = 1
                        sys.stderr.write(f"[w4mv] ENGAGED N={N} K={K}\n")
                    if bias is not None:
                        out = out + bias
                    return out
        except Exception as e:
            sys.stderr.write(f"[w4mv] fast path failed ({e!r}), falling back\n")
        return _orig_aw(self, layer, x, bias)

    import threading, time
    def _poll():
        import sys as _s
        for i in range(1500):
            try:
                import radiance_mxfp4 as _rm2
                _f = _rm2.mxfp4_linear
                _body = _f._backend_fns.get("cuda") or _f._init_fn  # steal the registered impl
                break
            except Exception:
                time.sleep(0.2)
        else:
            _s.stderr.write("[w4mv] radiance module never appeared\n")
            return

        @torch.library.impl("radiance::mxfp4_linear", "CUDA")
        def _patched_op(x, weight, weight_scale, wref):
            try:
                if (torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == 1
                        and x.dtype == torch.bfloat16):
                    N = weight.shape[0]
                    K = weight.shape[1] * 2
                    # shape gate: only small tiles where the fork's M=1 kernel is worst
                    # (fused qkv 12.6MB @41GB/s, o 8.4MB); big MLP GEMMs stay on the fork
                    # (its M=4-band path does 376GB/s there, our matvec ~186).
                    if (N * K // 2 <= 16 * 1024 * 1024
                            and K % 256 == 0 and N % 16 == 0
                            and weight_scale.shape[0] == K // 32):
                        out = torch.empty((1, N), device=x.device, dtype=torch.bfloat16)
                        _run(x, weight.view(torch.uint32), weight_scale, out,
                             BLOCK_N=16, BLOCK_J=32, warps=8, stages=4)
                        if _hits[0] == 0:
                            _hits[0] = 1
                            _s.stderr.write(f"[w4mv] OP-PATH ENGAGED N={N} K={K}\n")
                        return out
            except Exception as e:
                _s.stderr.write(f"[w4mv] op fast path failed ({e!r})\n")
            assert _body is not None, "no raw body"
            return _body(x, weight, weight_scale, wref)

        _s.stderr.write("[w4mv] op impl overridden\n")
    threading.Thread(target=_poll, daemon=True).start()
    sys.stderr.write("[w4mv] installed\n")

_install()
