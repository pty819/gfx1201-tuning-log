import torch, triton, triton.language as tl, time

@triton.jit
def _decode_attn_split(
    q_ptr, k_ptr, v_ptr,
    part_o_ptr, part_m_ptr, part_l_ptr,
    qsl_ptr, seq_lens_ptr, block_table_ptr,
    sm_scale, k_scale_ptr,
    bt_stride,
    sk0: tl.int64, sk1: tl.int64, sk2: tl.int64,
    sv0: tl.int64, sv1: tl.int64, sv2: tl.int64,
    HQ: tl.constexpr, GQA: tl.constexpr,
    D: tl.constexpr, PBLK: tl.constexpr,
    BM: tl.constexpr, TILE: tl.constexpr,
    K_FP8: tl.constexpr, NSPLIT: tl.constexpr,
):
    seq = tl.program_id(0)
    kh = tl.program_id(1)
    sp = tl.program_id(2)
    q_tok = tl.load(qsl_ptr + seq)
    seqlen = tl.load(seq_lens_ptr + seq)

    span = tl.cdiv(seqlen, NSPLIT)
    lo = sp * span
    hi = tl.minimum(seqlen, lo + span)
    if lo >= hi:
        # still must write neutral partials for the reducer
        b0 = ((seq * tl.num_programs(1) + kh) * NSPLIT + sp) * BM
        tl.store(part_m_ptr + b0 + tl.arange(0, BM), tl.full([BM], float("-inf"), tl.float32))
        tl.store(part_l_ptr + b0 + tl.arange(0, BM), tl.zeros([BM], tl.float32))
        po = part_o_ptr + (b0 + tl.arange(0, BM))[:, None] * D + tl.arange(0, D)[None, :]
        tl.store(po, tl.zeros([BM, D], tl.float32))
        return

    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, TILE)
    q_mask = offs_m < GQA

    qp = (q_ptr + q_tok.to(tl.int64) * (HQ * D)
          + (kh * GQA + offs_m)[:, None] * D + offs_d[None, :])
    Q = tl.load(qp, mask=q_mask[:, None], other=0.0)
    # k_scale folded into the score scale on-device (HIP-graph safe, no host sync)
    if K_FP8:
        sm_scale = sm_scale * tl.load(k_scale_ptr)

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    bt_row = block_table_ptr + seq.to(tl.int64) * bt_stride

    for t0 in range(lo, hi, TILE):
        tok = t0 + offs_t
        tmask = tok < hi
        blk = tl.load(bt_row + tok // PBLK, mask=tmask, other=0).to(tl.int64)
        inb = (tok % PBLK).to(tl.int64)
        koff = (blk[:, None] * sk0 + inb[:, None] * sk1 + kh * sk2 + offs_d[None, :])
        Kt = tl.load(k_ptr + koff, mask=tmask[:, None], other=0.0)
        Vt = tl.load(v_ptr + (blk[:, None] * sv0 + inb[:, None] * sv1 + kh * sv2 + offs_d[None, :]),
                     mask=tmask[:, None], other=0.0)
        if K_FP8:
            Kd = Kt.to(tl.bfloat16)
            Vd = Vt.to(tl.bfloat16)
        else:
            Kd = Kt
            Vd = Vt
        S = tl.dot(Q, tl.trans(Kd)) * sm_scale
        S = tl.where(tmask[None, :], S, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(S, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(S - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), Vd)
        m_i = m_new

    b0 = ((seq * tl.num_programs(1) + kh) * NSPLIT + sp) * BM
    tl.store(part_m_ptr + b0 + offs_m, m_i)
    tl.store(part_l_ptr + b0 + offs_m, l_i)
    po = part_o_ptr + (b0 + offs_m)[:, None] * D + offs_d[None, :]
    tl.store(po, acc, mask=q_mask[:, None])


@triton.jit
def _decode_attn_reduce(
    part_o_ptr, part_m_ptr, part_l_ptr, o_ptr, v_scale_ptr,
    HQ: tl.constexpr, GQA: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, NSPLIT: tl.constexpr, K_FP8: tl.constexpr,
):
    seq = tl.program_id(0)
    kh = tl.program_id(1)
    base = (seq * tl.num_programs(1) + kh) * NSPLIT
    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    offs_s = tl.arange(0, NSPLIT)
    q_mask = offs_m < GQA

    base = (seq * tl.num_programs(1) + kh) * NSPLIT * BM
    moffs = offs_m[None, :] + offs_s[:, None] * BM       # (NSPLIT, BM)
    ms = tl.load(part_m_ptr + base + moffs)              # (NSPLIT, BM)
    m_g = tl.max(ms, 0)                                  # (BM,)
    ls = tl.load(part_l_ptr + base + moffs)
    l_g = tl.sum(ls * tl.exp(ms - m_g[None, :]), 0)      # (BM,)
    acc = tl.zeros([BM, D], tl.float32)
    for s in range(NSPLIT):
        ms_ = tl.load(part_m_ptr + base + s * BM + offs_m)      # (BM,)
        w_ = tl.exp(ms_ - m_g)                                  # (BM,)
        p_ = tl.load(part_o_ptr + (base + s * BM) * D + offs_m[:, None] * D + offs_d[None, :],
                     mask=q_mask[:, None], other=0.0)
        acc += p_ * w_[:, None]
    vs = tl.load(v_scale_ptr) if K_FP8 else 1.0
    o = acc * vs / l_g[:, None]
    op = o_ptr + seq.to(tl.int64) * (HQ * D) + (kh * GQA + offs_m)[:, None] * D + offs_d[None, :]
    tl.store(op, o.to(tl.bfloat16), mask=q_mask[:, None])


def run(q, k_cache, v_cache, out, qsl, slens, bt, kscale, vscale, sm_scale,
        TILE=32, warps=4, stages=3, nsplit=8, scratch=None):
    NS, HKV = bt.shape[0], k_cache.shape[2]
    GQA = q.shape[1] // HKV
    BM, DH = 16, q.shape[2]
    if scratch is None:
        scratch = (torch.empty(NS * HKV * nsplit * BM * DH, device=q.device, dtype=torch.float32),
                   torch.empty(NS * HKV * nsplit * BM, device=q.device, dtype=torch.float32),
                   torch.empty(NS * HKV * nsplit * BM, device=q.device, dtype=torch.float32))
    po, pm, pl = scratch
    if not torch.is_tensor(vscale):
        vscale = torch.tensor([float(vscale)], device=q.device, dtype=torch.float32)
    if not torch.is_tensor(kscale):
        kscale = torch.tensor([float(kscale)], device=q.device, dtype=torch.float32)
    _decode_attn_split[(NS, HKV, nsplit)](
        q, k_cache, v_cache, po, pm, pl, qsl, slens, bt, sm_scale, kscale, bt.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        q.shape[1], GQA, q.shape[2], k_cache.shape[1],
        BM=BM, TILE=TILE, K_FP8=(k_cache.dtype != torch.bfloat16), NSPLIT=nsplit,
        num_warps=warps, num_stages=stages)
    _decode_attn_reduce[(NS, HKV)](
        po, pm, pl, out, vscale, q.shape[1], GQA, q.shape[2],
        BM=BM, NSPLIT=nsplit, K_FP8=(k_cache.dtype != torch.bfloat16), num_warps=4)

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
        return q, k, v, out, qsl, slens, bt, (q, torch.empty(0))[:1] + (None,)

    # correctness fp8 (vscale=1 case) against f32 reference
    q, k, v, out, qsl, slens, bt, _ = setup(torch.float8_e4m3fn)
    run(q, k, v, out, qsl, slens, bt, 1.0, 1.0, scale)
    kd = k.float().view(-1, HKV, D)[:NCTX]; vd = v.float().view(-1, HKV, D)[:NCTX]
    qh = q.float().view(HKV, 4, D).transpose(0, 1)
    ref = torch.empty(4, HKV, D, device=dev)
    for g in range(4):
        s = torch.einsum("hd,nhd->hn", qh[g], kd) * scale
        ref[g] = torch.einsum("hn,nhd->hd", torch.softmax(s, -1), vd)
    err = (out.float() - ref.transpose(0, 1).reshape(1, HQ, D)).abs().max().item()
    print(f"correctness fp8 split: max err = {err:.5f}"); assert err < 0.05
    # scales folded correctly? test ks=0.5 vs=2.0
    run(q, k, v, out, qsl, slens, bt, 0.5, 2.0, scale)
    kd2 = k.float().view(-1, HKV, D)[:NCTX] * 0.5; vd2 = v.float().view(-1, HKV, D)[:NCTX] * 2.0
    ref2 = torch.empty(4, HKV, D, device=dev)
    for g in range(4):
        s = torch.einsum("hd,nhd->hn", qh[g], kd2) * scale
        ref2[g] = torch.einsum("hn,nhd->hd", torch.softmax(s, -1), vd2)
    err2 = (out.float() - ref2.transpose(0, 1).reshape(1, HQ, D)).abs().max().item()
    print(f"correctness scaled (ks=.5,vs=2): max err = {err2:.5f}"); assert err2 < 0.05

    print("perf (us/call, 1 seq @4k ctx):")
    for dt, name in [(torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]:
        q3, k3, v3, out3, qsl3, slens3, bt3, _ = setup(dt)
        for nsplit in (4, 8, 16):
            for tile, w in ((32, 4), (64, 4), (64, 8)):
                try:
                    run(q3, k3, v3, out3, qsl3, slens3, bt3, 1.0, 1.0, scale,
                        TILE=tile, warps=w, nsplit=nsplit)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(200):
                        run(q3, k3, v3, out3, qsl3, slens3, bt3, 1.0, 1.0, scale,
                            TILE=tile, warps=w, nsplit=nsplit)
                    torch.cuda.synchronize()
                    us = (time.perf_counter() - t0) / 200 * 1e6
                    print(f"  {name} split={nsplit:2d} TILE={tile:3d} w={w}: {us:7.1f} us  (x32 = {us*32/1000:.2f} ms/step)")
                except Exception as e:
                    print(f"  {name} split={nsplit} TILE={tile} w={w}: FAIL {str(e)[:50]}")
