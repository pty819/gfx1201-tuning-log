"""Route fp8 per-tensor KV decode attention through the split-KV Triton kernel."""
import os
import sys

def _install():
    try:
        import torch
        import triton
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        from vllm.v1.kv_cache_interface import KVQuantMode
        import vllm.v1.attention.ops.triton_unified_attention as _tua
        import triton.language as _tl
        import threading

        @triton.jit
        def _fast_cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: _tl.constexpr):
            # drop-in for _cast_kv_tile: fp8->bf16 direct (fast cvt), scale in Q's domain.
            # exact when scale == 1.0; per-element bf16 rounding otherwise (was f32 exact).
            if KV_QUANT_MODE == 1:
                if Q.dtype.is_fp8():
                    return data.to(Q.dtype)
                return data.to(Q.dtype) * _tl.load(tensor_scale).to(Q.dtype)
            return data.to(Q.dtype)
        _tua._cast_kv_tile = _fast_cast_kv_tile
        _counts = {"unified": 0, "impl_forward": 0}
        _orig_ua = _tua.unified_attention
        def _counting_ua(*a, **kw):
            _counts["unified"] += 1
            return _orig_ua(*a, **kw)
        _tua.unified_attention = _counting_ua
        def _report():
            import time as _t
            while True:
                _t.sleep(15)
                sys.stderr.write(f"[fp8kv-override] counters: {_counts}\n")
        threading.Thread(target=_report, daemon=True).start()
        sys.path.insert(0, "/opt/fp8kv")
        from fp8kv_v3 import run as _run_mine
    except Exception as e:
        sys.stderr.write(f"[fp8kv-override] install skipped: {e}\n")
        return

    NSPLIT, TILE, WARPS = 8, 32, 4
    _orig = TritonAttentionImpl.forward
    _scratch = {}
    _engaged = [0]

    def _forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                 output_scale=None, output_block_scale=None):
        try:
            fast_ok = (
                attn_metadata is not None
                and self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
                and attn_metadata.max_query_len == 1
                and self.head_size == 128
                and self.num_kv_heads > 0
                and not self.use_td
                and self.sliding_window[0] < 0
                and self.sinks is None
                and self.alibi_slopes is None
                and self.logits_soft_cap == 0
            )
        except AttributeError:
            fast_ok = False
        if _engaged[0] == 0:
            _engaged[0] = -1
            try:
                sys.stderr.write(f"[fp8kv-override] diag: mode={self._kv_quant_mode!r} "
                                 f"maxq={attn_metadata.max_query_len} hs={self.head_size} "
                                 f"sw={self.sliding_window} td={self.use_td} "
                                 f"sinks={self.sinks is None} alibi={self.alibi_slopes is None} "
                                 f"cap={self.logits_soft_cap} fast_ok={fast_ok}\n")
            except Exception as e:
                sys.stderr.write(f"[fp8kv-override] diag-fail {e}\n")
        if not fast_ok:
            return _orig(self, layer, query, key, value, kv_cache, attn_metadata,
                         output, output_scale, output_block_scale)

        _counts["impl_forward"] += 1
        n = attn_metadata.num_actual_tokens
        hs = self.head_size
        kc = kv_cache.transpose(1, 2)
        key_cache, value_cache = kc.split(hs, dim=-1)
        key_cache = key_cache.view(self.fp8_dtype)
        value_cache = value_cache.view(self.fp8_dtype)

        NS = attn_metadata.query_start_loc.shape[0] - 1
        HKV = self.num_kv_heads
        BM = 16
        need = max(NS, 64) * HKV * NSPLIT * BM
        scr = _scratch.get("buf")
        if scr is None or scr[0].numel() < need * hs or scr[0].device != query.device:
            scr = (torch.empty(need * hs, device=query.device, dtype=torch.float32),
                   torch.empty(need, device=query.device, dtype=torch.float32),
                   torch.empty(need, device=query.device, dtype=torch.float32))
            _scratch["buf"] = scr

        if _engaged[0] == 0:
            _engaged[0] = 1
            sys.stderr.write(f"[fp8kv-override] ENGAGED: NS={NS} HKV={HKV} hs={hs} "
                             f"GQA={self.num_queries_per_kv}\n")
        _run_mine(query[:n], key_cache, value_cache, output[:n],
                  attn_metadata.query_start_loc, attn_metadata.seq_lens,
                  attn_metadata.block_table,
                  layer._k_scale, layer._v_scale, self.scale,
                  TILE=TILE, warps=WARPS, nsplit=NSPLIT, scratch=scr)
        return output

    TritonAttentionImpl.forward = _forward

    try:
        from vllm.model_executor.layers.attention.attention import Attention as _Attn
        _orig_pw = _Attn.process_weights_after_loading
        def _pw(self, act_dtype):
            sys.stderr.write(f"[fp8kv-override] layer impl = {self.impl.__class__.__name__} "
                             f"(VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND')})\n")
            return _orig_pw(self, act_dtype)
        _Attn.process_weights_after_loading = _pw
    except Exception as e:
        sys.stderr.write(f"[fp8kv-override] pw-probe fail {e}\n")

    if os.environ.get("FP8KV_FORCE_TRITON"):
        try:
            import vllm.v1.attention.selector as _sel
            from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend as _TB
            _sel.get_attn_backend = lambda *a, **kw: _TB
            sys.stderr.write("[fp8kv-override] selector forced to TritonAttentionBackend\n")
        except Exception as e:
            sys.stderr.write(f"[fp8kv-override] force-triton fail {e}\n")

    sys.stderr.write("[fp8kv-override] installed\n")

_install()
