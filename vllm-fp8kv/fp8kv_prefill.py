"""Flash-prefill kernel v2 for fp8-KV per-tensor, head_dim=128, GQA, single-seq.

v1 lessons (measured on gfx1201 / Triton 3.7.1):
  - tl.dot DOES lower to WMMA (128 wmma instrs in ISA) - compute path is fine
  - but 4860us/layer vs ~300-450us floor: tl.trans(K) shuffles every tile,
    tl.exp (not exp2 form), causal mask applied to EVERY kv tile, spills at
    num_stages>=3 (13440us).
v2: K loaded directly as [D, BN] (no transpose), exp2 canonical form
    (log2e folded into the score scale), two-phase kv loop (full tiles
    unmasked, diagonal tiles masked), BM/BN/warps tuning, stages locked to 2.
v2.1 (gfx1201 2026-09-15): default tile 128/128/8/2 was the slowest of the
    measured grid (~3360us/layer @2048). 64/32/2/2 is ~1687us, still WMMA,
    same numerics. stages>=3 still spills.

Numerics identical to the decode kernel: fp8->bf16 direct cast (exact),
k_scale folded into the score scale, v_scale at the output, f32 softmax.
Single-sequence prefill only; everything else falls through the chain.
No host<->device syncs (q0 = num_computed_tokens, qlen = num_actual_tokens).
Env gate: FP8KV_PREFILL=1.
"""
import os
import sys

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _prefill_attn(
    q_ptr, k_ptr, v_ptr, o_ptr,
    block_table_ptr, bt_stride,
    sm_scale, k_scale_ptr, v_scale_ptr,
    sk0: tl.int64, sk1: tl.int64, sk2: tl.int64,
    sv0: tl.int64, sv1: tl.int64, sv2: tl.int64,
    q0, qlen,
    HQ: tl.constexpr, GQA: tl.constexpr,
    D: tl.constexpr, PBLK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
    K_FP8: tl.constexpr,
):
    pid_m = tl.program_id(0)
    kh = tl.program_id(1)
    rows = tl.arange(0, BM)
    BMT: tl.constexpr = BM // GQA
    tok = pid_m * BMT + rows // GQA
    g = rows % GQA
    row_ok = tok < qlen

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BN)
    qp = (q_ptr + tok.to(tl.int64)[:, None] * (HQ * D)
          + (kh * GQA + g)[:, None] * D + offs_d[None, :])
    Q = tl.load(qp, mask=row_ok[:, None], other=0.0)
    if K_FP8:
        qk_scale = sm_scale * tl.load(k_scale_ptr) * LOG2E
    else:
        qk_scale = sm_scale * LOG2E

    hi = q0 + tl.minimum(qlen, (pid_m + 1) * BMT)      # last kv needed by this tile
    nfull = ((q0 + pid_m * BMT) // BN) * BN            # kv below tile start: fully visible

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    bt_row = block_table_ptr

    # ---- phase 1: full tiles, no causal mask ----
    for t0 in range(0, nfull, BN):
        t = t0 + offs_n
        blk = tl.load(bt_row + t // PBLK).to(tl.int64)
        inb = (t % PBLK).to(tl.int64)
        Kt = tl.load(k_ptr + blk[None, :] * sk0 + inb[None, :] * sk1 + kh * sk2 + offs_d[:, None])
        Vt = tl.load(v_ptr + blk[:, None] * sv0 + inb[:, None] * sv1 + kh * sv2 + offs_d[None, :])
        if K_FP8:
            Kd = Kt.to(tl.bfloat16)
            Vd = Vt.to(tl.bfloat16)
        else:
            Kd = Kt
            Vd = Vt
        S = tl.dot(Q, Kd) * qk_scale
        m_new = tl.maximum(m_i, tl.max(S, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(S - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), Vd)
        m_i = m_new

    # ---- phase 2: diagonal tiles, causal mask ----
    for t0 in range(nfull, hi, BN):
        t = t0 + offs_n
        tmask = t < hi
        blk = tl.load(bt_row + t // PBLK, mask=tmask, other=0).to(tl.int64)
        inb = (t % PBLK).to(tl.int64)
        Kt = tl.load(k_ptr + blk[None, :] * sk0 + inb[None, :] * sk1 + kh * sk2 + offs_d[:, None],
                     mask=tmask[None, :], other=0.0)
        Vt = tl.load(v_ptr + blk[:, None] * sv0 + inb[:, None] * sv1 + kh * sv2 + offs_d[None, :],
                     mask=tmask[:, None], other=0.0)
        if K_FP8:
            Kd = Kt.to(tl.bfloat16)
            Vd = Vt.to(tl.bfloat16)
        else:
            Kd = Kt
            Vd = Vt
        S = tl.dot(Q, Kd) * qk_scale
        causal = t[None, :] <= (q0 + tok)[:, None]
        S = tl.where(causal & tmask[None, :], S, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(S, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(S - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), Vd)
        m_i = m_new

    vs = tl.load(v_scale_ptr) if K_FP8 else 1.0
    o = acc * vs / l_i[:, None]
    op = (o_ptr + tok.to(tl.int64)[:, None] * (HQ * D)
          + (kh * GQA + g)[:, None] * D + offs_d[None, :])
    tl.store(op, o.to(tl.bfloat16), mask=row_ok[:, None])


def run(q, k_cache, v_cache, out, bt, kscale, vscale, sm_scale,
        q0, qlen, BM=64, BN=32, warps=2, stages=2):
    HKV = k_cache.shape[2]
    HQ = q.shape[1]
    GQA = HQ // HKV
    BMT = BM // GQA
    grid = (triton.cdiv(qlen, BMT), HKV)
    if not torch.is_tensor(vscale):
        vscale = torch.tensor([float(vscale)], device=q.device, dtype=torch.float32)
    if not torch.is_tensor(kscale):
        kscale = torch.tensor([float(kscale)], device=q.device, dtype=torch.float32)
    _prefill_attn[grid](
        q, k_cache, v_cache, out,
        bt, bt.stride(0),
        sm_scale, kscale, vscale,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        q0, qlen,
        HQ=HQ, GQA=GQA, D=q.shape[2], PBLK=k_cache.shape[1],
        BM=BM, BN=BN, K_FP8=(k_cache.dtype != torch.bfloat16),
        num_warps=warps, num_stages=stages)


def _install():
    if os.environ.get("FP8KV_PREFILL", "0") != "1":
        return
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    from vllm.v1.kv_cache_interface import KVQuantMode

    _next = TritonAttentionImpl.forward
    _engaged = [0]

    def _forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                 output_scale=None, output_block_scale=None):
        try:
            ok = (
                attn_metadata is not None
                and self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
                and attn_metadata.max_query_len >= 64
                and attn_metadata.query_start_loc.shape[0] == 2
                and self.head_size == 128
                and self.num_kv_heads > 0
                and not self.use_td
                and self.sliding_window[0] < 0
                and self.sinks is None
                and self.alibi_slopes is None
                and self.logits_soft_cap == 0
                and not getattr(attn_metadata, "use_cascade", False)
            )
        except AttributeError:
            ok = False
        if ok:
            try:
                n = attn_metadata.num_actual_tokens
                # this vLLM: no num_computed_tokens; seq_len is full KV length
                q0 = int(attn_metadata.max_seq_len) - int(n)
                hs = self.head_size
                if 0 <= q0 and n >= 64:
                    kc = kv_cache.transpose(1, 2)
                    key_cache, value_cache = kc.split(hs, dim=-1)
                    key_cache = key_cache.view(self.fp8_dtype)
                    value_cache = value_cache.view(self.fp8_dtype)
                    run(query[:n], key_cache, value_cache, output[:n],
                        attn_metadata.block_table,
                        layer._k_scale, layer._v_scale, self.scale, q0, n)
                    if _engaged[0] == 0:
                        _engaged[0] = 1
                        sys.stderr.write(f"[fp8kv-prefill] ENGAGED qlen={n} q0={q0}\n")
                    return output
            except Exception as e:
                sys.stderr.write(f"[fp8kv-prefill] fast path failed ({e!r}), falling back\n")
        return _next(self, layer, query, key, value, kv_cache, attn_metadata,
                     output, output_scale, output_block_scale)

    TritonAttentionImpl.forward = _forward
    sys.stderr.write("[fp8kv-prefill] installed\n")


_install()
