"""M=1 W4 matvec v4: exponent-folded scales + bf16 MMA (tl.dot), padded M=16."""
import torch, triton, triton.language as tl, time


@triton.jit
def _w4_matvec_mma(x_ptr, w_ptr, s_ptr, o_ptr, K, N,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    offs_m = tl.arange(0, 16)
    acc = tl.zeros([16, BLOCK_N], tl.float32)
    G: tl.constexpr = BLOCK_K // 32
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_j = k0 // 8 + tl.arange(0, BLOCK_K // 8)
        # W tile u32 -> nibbles -> bf16 with folded scale exponent
        wq = tl.load(w_ptr + offs_n[:, None] * (K // 8) + offs_j[None, :],
                     mask=nmask[:, None], other=0)                              # (BN, BK/8) u32
        v = tl.reshape(((wq[:, :, None] >> ((tl.arange(0, 8) * 4)[None, None, :].to(tl.uint32)))
                        & 0xF).to(tl.uint32), (BLOCK_N, BLOCK_K))                # (BN, BK)
        ee = (v & 0x6) >> 1
        sg = tl.load(s_ptr + (k0 // 32 + tl.arange(0, G))[None, :] * N + offs_n[:, None],
                     mask=nmask[:, None], other=127)                            # (BN, G)
        # fold: broadcast sg over 32 k's -> (BN, BK)
        sg_k = tl.reshape(tl.broadcast_to(sg[:, :, None], (BLOCK_N, G, 32)), (BLOCK_N, BLOCK_K))
        mant = tl.where(ee == 0, tl.zeros_like(v), (v & 0x1) << 6)
        b16 = (((v & 0x8) << 12) | ((ee + sg_k - 1) << 7) | mant).to(tl.uint16)
        W = b16.to(tl.bfloat16, bitcast=True)
        W = tl.where((v & 0x7) == 0, 0.0, W)
        # x padded to (16, BK): row0 = chunk, rows 1-15 zero
        xk = tl.load(x_ptr + offs_k).to(tl.bfloat16)
        xp = tl.where(offs_m[:, None] == 0, xk[None, :], 0.0).to(tl.bfloat16)
        acc += tl.dot(xp, tl.trans(W))     # (16, BN) f32
    out = tl.sum(acc, 0)                    # rows 1-15 are zero -> picks row 0
    tl.store(o_ptr + offs_n, out.to(tl.bfloat16), mask=nmask)


def run(x, w32, s, out, BLOCK_N=64, BLOCK_K=256, warps=4, stages=3):
    N, K8 = w32.shape
    K = K8 * 8
    _w4_matvec_mma[(triton.cdiv(N, BLOCK_N),)](x, w32, s, out, K, N,
                                    BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                    num_warps=warps, num_stages=stages)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                         -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=dev)

    def make(N, K):
        w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
        s = torch.randint(124, 131, (K // 32, N), dtype=torch.uint8, device=dev)
        x = (torch.randn(1, K, device=dev) * 0.5).to(torch.bfloat16)
        return w, w.view(torch.uint32), s, x

    def ref(w, s, x):
        N, Kh = w.shape; K = Kh * 2
        codes = torch.stack([w & 0x0F, w >> 4], -1).reshape(N, K).long()
        sc = torch.pow(2.0, s.float() - 127.0).T.repeat_interleave(32, dim=1)
        return (x.float() @ (E2M1[codes] * sc).T)

    print("== correctness ==")
    for (N, K) in [(256, 4096), (1024, 4096), (96, 12288)]:
        w, w32, s, x = make(N, K)
        out = torch.empty(1, N, device=dev, dtype=torch.bfloat16)
        run(x, w32, s, out)
        r = ref(w, s, x)
        rel = ((out.float() - r).norm() / r.norm()).item()
        print(f"  N={N} K={K}: rel={rel:.5f}")
        assert rel < 0.01

    print("== perf (8-buffer rotation) ==")
    NREP = 8
    for (N, K, tag) in [(4096, 4096, "qkv-ish"), (1024, 4096, "k/v"),
                        (12288, 4096, "gate/up"), (4096, 12288, "down")]:
        pool = []
        for _ in range(NREP):
            w, w32, sc, x = make(N, K)
            out = torch.empty(1, N, device=dev, dtype=torch.bfloat16)
            pool.append((w32, sc, x, out))
        best = None
        for bn in (16, 32, 64, 128):
            for bk in (128, 256, 512):
                for wp in (4, 8):
                    for st in (2, 3, 4):
                        try:
                            for i in range(NREP):
                                w32, sc, x, out = pool[i]
                                run(x, w32, sc, out, BLOCK_N=bn, BLOCK_K=bk, warps=wp, stages=st)
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            R = 40
                            for it in range(R * NREP):
                                w32, sc, x, out = pool[it % NREP]
                                run(x, w32, sc, out, BLOCK_N=bn, BLOCK_K=bk, warps=wp, stages=st)
                            torch.cuda.synchronize()
                            us = (time.perf_counter() - t0) / (R * NREP) * 1e6
                            if best is None or us < best[0]:
                                best = (us, bn, bk, wp, st)
                        except Exception:
                            pass
        wb = N * K / 2
        print(f"  {tag:9s} N={N:5d} K={K:5d}: {best[0]:7.1f} us  {wb/best[0]/1e3:6.0f} GB/s  "
              f"(BN={best[1]} BK={best[2]} w={best[3]} st={best[4]})")
