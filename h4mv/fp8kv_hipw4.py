"""Route M<=8 W4A8 linear layers through the HIP h4mv kernel (libh4mv.so).

Intercepts radiance::mxfp4_linear at the op level (proven mechanics from
fp8kv_w4patch.py): steal _backend_fns["cuda"] before re-registering, fall back
to the original body for anything we don't handle (M>8, odd shapes, fp8 x).

- Lazy repack: weight_scale [K/32, N] -> [N, K/32] contiguous, cached by
  data_ptr. Never done while the stream is capturing (graph safety).
- x arrives bf16 [M, K] at the op boundary; h4mv consumes it natively.
- Output bf16 [M, N], same contract as the fork kernel (bias handled outside).
"""
import os

def _install():
    if os.environ.get("FP8KV_HIPW4", "0") != "1":
        return
    import sys, ctypes, threading, time
    import torch

    try:
        lib = ctypes.CDLL("/opt/fp8kv/libh4mv.so")
        lib.h4mv_launch.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3 + [ctypes.c_void_p]
        lib.h4mv_launch.restype = None
    except Exception as e:
        sys.stderr.write(f"[hipw4] lib load fail: {e}\n")
        return

    _MAXM = int(os.environ.get("FP8KV_HIPW4_MAX_M", "4"))
    _repack_cache = {}
    _stats = {"fast": 0, "slow": 0, "logged": False, "calls": 0}

    def _st_for(s, device):
        key = (s.data_ptr(), s.shape)
        ent = _repack_cache.get(key)
        if ent is not None:
            return ent
        if torch.cuda.is_current_stream_capturing():
            return None                      # never allocate/launch aux ops during capture
        st = s.t().contiguous()
        if len(_repack_cache) > 512:
            _repack_cache.clear()
        _repack_cache[key] = st
        return st

    def _poll():
        body = None
        for i in range(1500):
            try:
                import radiance_mxfp4 as rm
                if getattr(rm, "WPERM", False):
                    sys.stderr.write("[hipw4] WPERM=1, standing down\n")
                    return
                f = rm.mxfp4_linear
                body = f._backend_fns.get("cuda") or f._init_fn
                break
            except Exception:
                time.sleep(0.2)
        if body is None:
            sys.stderr.write("[hipw4] radiance module never appeared\n")
            return

        @torch.library.impl("radiance::mxfp4_linear", "CUDA")
        def _patched(x, weight, weight_scale, wref):
            try:
                if _stats["calls"] < 3:
                    _stats["calls"] += 1
                    sys.stderr.write(f"[hipw4] call#{_stats['calls']} x={tuple(x.shape)}/{x.dtype}"
                                     f"/c={x.is_contiguous()} w={tuple(weight.shape)}/{weight.dtype}"
                                     f" s={tuple(weight_scale.shape)} cap={torch.cuda.is_current_stream_capturing()}\n")
                if (torch.is_tensor(x) and x.dim() == 2 and x.is_contiguous()
                        and 1 <= x.shape[0] <= _MAXM and x.dtype == torch.bfloat16):
                    N = weight.shape[0]
                    K = weight.shape[1] * 2
                    if (K % 32 == 0 and N >= 8
                            and weight_scale.shape[0] == K // 32
                            and weight.data_ptr() % 16 == 0):
                        st = _st_for(weight_scale, x.device)
                        if st is not None:
                            M = x.shape[0]
                            out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
                            lib.h4mv_launch(weight.data_ptr(), st.data_ptr(), x.data_ptr(),
                                            out.data_ptr(), N, K, M,
                                            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
                            _stats["fast"] += 1
                            if not _stats["logged"]:
                                _stats["logged"] = True
                                sys.stderr.write(f"[hipw4] ENGAGED N={N} K={K} M={M} "
                                                 f"(x={tuple(x.shape)} w={tuple(weight.shape)} "
                                                 f"s={tuple(weight_scale.shape)})\n")
                            return out
            except Exception as e:
                sys.stderr.write(f"[hipw4] fast path failed ({e!r}), falling back\n")
            _stats["slow"] += 1
            return body(x, weight, weight_scale, wref)

        sys.stderr.write("[hipw4] op impl overridden\n")

    threading.Thread(target=_poll, daemon=True).start()
    sys.stderr.write("[hipw4] installed\n")

_install()
