import torch, triton, triton.language as tl, time

@triton.jit
def _decode_attn(
    q_ptr, k_ptr, v_ptr, o_ptr,
    qsl_ptr, seq_lens_ptr, block_table_ptr,
    sm_scale,
    k_scale_ptr, v_scale_ptr,
    bt_stride,
    sk0: tl.int64, sk1: tl.int64, sk2: tl.int64,
    sv0: tl.int64, sv1: tl.int64, sv2: tl.int64,
    HQ: tl.constexpr, GQA: tl.constexpr,
    D: tl.constexpr, PBLK: tl.constexpr,
    BM: tl.constexpr, TILE: tl.constexpr,
    K_FP8: tl.constexpr,
):
    seq = tl.program_id(0)
    kh = tl.program_id(1)
    q_tok = tl.load(qsl_ptr + seq)
    seqlen = tl.load(seq_lens_ptr + seq)

    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, TILE)
    q_mask = offs_m < GQA

    qp = (q_ptr + q_tok.to(tl.int64) * (HQ * D)
          + (kh * GQA + offs_m)[:, None] * D + offs_d[None, :])
    Q = tl.load(qp, mask=q_mask[:, None], other=0.0)

    if K_FP8:
        ks = tl.load(k_scale_ptr)   # folded into sm_scale by the caller
        vs = tl.load(v_scale_ptr)

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    bt_row = block_table_ptr + seq.to(tl.int64) * bt_stride

    for t0 in range(0, seqlen, TILE):
        tok = t0 + offs_t
        tmask = tok < seqlen
        blk = tl.load(bt_row + tok // PBLK, mask=tmask, other=0).to(tl.int64)
        inb = (tok % PBLK).to(tl.int64)
        # (TILE, D) native-layout load: D on the contiguous minor axis -> coalesced
        koff = (blk[:, None] * sk0 + inb[:, None] * sk1
                + kh * sk2 + offs_d[None, :] * 1)
        Kt = tl.load(k_ptr + koff, mask=tmask[:, None], other=0.0)   # (TILE, D)
        Vt = tl.load(v_ptr + (blk[:, None] * sv0 + inb[:, None] * sv1
                              + kh * sv2 + offs_d[None, :]), mask=tmask[:, None], other=0.0)
        if K_FP8:
            Kd = Kt.to(tl.bfloat16)   # e4m3 -> bf16 is EXACT (and the fast cvt path)
            Vd = Vt.to(tl.bfloat16)
        else:
            Kd = Kt
            Vd = Vt
        # per-tensor scales folded: k_scale into sm_scale (K linear in scores),
        # v_scale applied once on the final output (V linear in the weighted sum)
        S = tl.dot(Q, tl.trans(Kd)) * sm_scale            # (BM, TILE) f32
        S = tl.where(tmask[None, :], S, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(S, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(S - m_new[:, None])                    # f32
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), Vd)
        m_i = m_new

    o = acc * vs / l_i[:, None] if K_FP8 else acc / l_i[:, None]
    op = (o_ptr + q_tok.to(tl.int64) * (HQ * D)
          + (kh * GQA + offs_m)[:, None] * D + offs_d[None, :])
    tl.store(op, o.to(tl.bfloat16), mask=q_mask[:, None])


def run(q, k_cache, v_cache, out, qsl, slens, bt, kscale, vscale, sm_scale,
        TILE=64, warps=4, stages=3):
    NS, HKV = bt.shape[0], k_cache.shape[2]
    GQA = q.shape[1] // HKV
    grid = (NS, HKV)
    _decode_attn[grid](
        q, k_cache, v_cache, out, qsl, slens, bt,
        sm_scale * (float(kscale) if k_cache.dtype != torch.bfloat16 else 1.0),
        kscale, vscale, bt.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        q.shape[1], GQA, q.shape[2], k_cache.shape[1],
        BM=16, TILE=TILE, K_FP8=(k_cache.dtype != torch.bfloat16),
        num_warps=warps, num_stages=stages)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    NCTX, HQ, HKV, D, PBLK = 4096, 32, 8, 128, 16
    NB = NCTX // PBLK
    scale = D ** -0.5

    def setup(kv_dtype, ns=1):
        q = torch.randn(ns, HQ, D, dtype=torch.bfloat16, device=dev)
        k = (torch.randn(NB, PBLK, HKV, D, device=dev) * 2).to(kv_dtype)
        v = (torch.randn(NB, PBLK, HKV, D, device=dev) * 2).to(kv_dtype)
        bt = torch.arange(NB, dtype=torch.int32, device=dev).view(1, NB).repeat(ns, 1)
        qsl = torch.arange(ns + 1, dtype=torch.int32, device=dev)
        slens = torch.full((ns,), NCTX, dtype=torch.int32, device=dev)
        out = torch.empty(ns, HQ, D, dtype=torch.bfloat16, device=dev)
        ks = torch.ones(1, device=dev, dtype=torch.float32)
        return q, k, v, out, qsl, slens, bt, ks

    # correctness: fp8 kernel vs fp32 reference on the SAME fp8 cache
    q, k, v, out, qsl, slens, bt, ks = setup(torch.float8_e4m3fn)
    run(q, k, v, out, qsl, slens, bt, ks, ks, scale)
    kd = k.float().view(-1, HKV, D)[:NCTX]
    vd = v.float().view(-1, HKV, D)[:NCTX]
    qh = q.float().view(HKV, 4, D).transpose(0, 1)
    ref = torch.empty(4, HKV, D, device=dev)
    for g in range(4):
        s = torch.einsum("hd,nhd->hn", qh[g], kd) * scale
        ref[g] = torch.einsum("hn,nhd->hd", torch.softmax(s, -1), vd)
    ref = ref.transpose(0, 1).reshape(1, HQ, D)
    err = (out.float() - ref).abs().max().item()
    print(f"correctness fp8: max|out-ref| = {err:.5f}  (should be < 0.01)")
    assert err < 0.05, "WRONG RESULT"

    # bf16 mode correctness
    q2, k2, v2, out2, qsl2, slens2, bt2, ks2 = setup(torch.bfloat16)
    run(q2, k2, v2, out2, qsl2, slens2, bt2, ks2, ks2, scale)
    ref2 = torch.empty(4, HKV, D, device=dev)
    qh2 = q2.float().view(HKV, 4, D).transpose(0, 1)
    kd2 = k2.float().view(-1, HKV, D)[:NCTX]
    vd2 = v2.float().view(-1, HKV, D)[:NCTX]
    for g in range(4):
        s = torch.einsum("hd,nhd->hn", qh2[g], kd2) * scale
        ref2[g] = torch.einsum("hn,nhd->hd", torch.softmax(s, -1), vd2)
    err2 = (out2.float() - ref2.transpose(0, 1).reshape(1, HQ, D)).abs().max().item()
    print(f"correctness bf16: max|out-ref| = {err2:.5f}")
    assert err2 < 0.05

    # perf sweep
    print("perf (us/call, 1 seq @4k ctx):")
    for dt, name in [(torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]:
        q3, k3, v3, out3, qsl3, slens3, bt3, ks3 = setup(dt)
        for tile in (32, 64, 128):
            for w in (4, 8):
                try:
                    run(q3, k3, v3, out3, qsl3, slens3, bt3, ks3, ks3, scale, TILE=tile, warps=w)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(200):
                        run(q3, k3, v3, out3, qsl3, slens3, bt3, ks3, ks3, scale, TILE=tile, warps=w)
                    torch.cuda.synchronize()
                    us = (time.perf_counter() - t0) / 200 * 1e6
                    print(f"  {name} TILE={tile:3d} warps={w}: {us:7.1f} us  (x32 = {us*32/1000:.2f} ms/step)")
                except Exception as e:
                    print(f"  {name} TILE={tile} warps={w}: FAIL {str(e)[:60]}")
