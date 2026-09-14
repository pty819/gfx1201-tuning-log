"""W4 matvec v5: scale via duplicated (BN,J) gather (no reshape-broadcast) + exponent fold."""
import torch, triton, triton.language as tl, time


@triton.jit
def _w4_matvec5(x_ptr, w_ptr, s_ptr, o_ptr, K, N,
                BLOCK_N: tl.constexpr, BLOCK_J: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    sh = (tl.arange(0, 8) * 4)[None, None, :].to(tl.uint32)
    G: tl.constexpr = BLOCK_J // 4
    acc = tl.zeros([BLOCK_N, BLOCK_J * 8], tl.float32)
    for j0 in range(0, K // 8, BLOCK_J):
        offs_j = j0 + tl.arange(0, BLOCK_J)
        wq = tl.load(w_ptr + offs_n[:, None] * (K // 8) + offs_j[None, :])       # (BN,J) u32
        # sT is [N, K/32] contiguous: coalesced (BN,G) tile, one broadcast for the fold
        sg = tl.load(s_ptr + offs_n[:, None] * (K // 32)
                     + (j0 // 4 + tl.arange(0, G))[None, :])                     # (BN,G) u8  (group = 8*j0/32)
        sgj = tl.reshape(tl.broadcast_to(sg[:, :, None], (BLOCK_N, G, 4)), (BLOCK_N, BLOCK_J))
        v = ((wq[:, :, None] >> sh) & 0xF).to(tl.uint32)                         # (BN,J,8)
        ee = (v & 0x6) >> 1
        mant = tl.where(ee == 0, tl.zeros_like(v), (v & 0x1) << 6)
        b16 = (((v & 0x8) << 12) | ((ee + sgj[:, :, None].to(tl.uint32) - 1) << 7) | mant).to(tl.uint16)
        w = b16.to(tl.bfloat16, bitcast=True)
        w = tl.where((v & 0x7) == 0, 0.0, w)
        xk = tl.load(x_ptr + 8 * offs_j[:, None] + tl.arange(0, 8)[None, :]).to(tl.float32)  # (J,8)
        acc += tl.reshape(w.to(tl.float32) * xk[None, :, :], (BLOCK_N, BLOCK_J * 8))
    tl.store(o_ptr + offs_n, tl.sum(acc, 1).to(tl.bfloat16))


def run(x, w32, s, out, BLOCK_N=16, BLOCK_J=32, warps=8, stages=4):
    N, K8 = w32.shape
    K = K8 * 8
    _w4_matvec5[(triton.cdiv(N, BLOCK_N),)](x, w32, s, out, K, N,
                                            BLOCK_N=BLOCK_N, BLOCK_J=BLOCK_J,
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
        sT = s.t().contiguous()
        out = torch.empty(1, N, device=dev, dtype=torch.bfloat16)
        run(x, w32, sT, out)
        r = ref(w, s, x)
        rel = ((out.float() - r).norm() / r.norm()).item()
        print(f"  N={N} K={K}: rel={rel:.5f}")
        assert rel < 0.01

    print("== perf (8-buf rotation) ==")
    NREP = 8
    for (N, K, tag) in [(6144, 4096, "qkv"), (24576, 4096, "gate+up"), (4096, 12288, "down"),
                        (4096, 4096, "o")]:
        pool = []
        for _ in range(NREP):
            w, w32, sc, x = make(N, K)
            sT = sc.t().contiguous()          # [N, K/32] coalesced for the kernel
            out = torch.empty(1, N, device=dev, dtype=torch.bfloat16)
            pool.append((w32, sT, x, out))
        best = None
        for bn in (8, 16):
            for bj in (32, 64):
                for wp in (4, 8):
                    for st in (4, 5, 6, 8):
                        try:
                            for i in range(NREP):
                                w32, sc, x, out = pool[i]
                                run(x, w32, sc, out, BLOCK_N=bn, BLOCK_J=bj, warps=wp, stages=st)
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            R = 40
                            for it in range(R * NREP):
                                w32, sc, x, out = pool[it % NREP]
                                run(x, w32, sc, out, BLOCK_N=bn, BLOCK_J=bj, warps=wp, stages=st)
                            torch.cuda.synchronize()
                            us = (time.perf_counter() - t0) / (R * NREP) * 1e6
                            if best is None or us < best[0]:
                                best = (us, bn, bj, wp, st)
                        except Exception:
                            pass
        wb = N * K / 2
        print(f"  {tag:9s} N={N:5d} K={K:5d}: {best[0]:7.1f} us  {wb/best[0]/1e3:6.0f} GB/s  "
              f"(BN={best[1]} BJ={best[2]} w={best[3]} st={best[4]})")
        # fixed ladder config for cross-check vs L3
        t0 = time.perf_counter(); R = 40
        for it in range(R * NREP):
            w32, sc, x, out = pool[it % NREP]
            run(x, w32, sc, out, BLOCK_N=16, BLOCK_J=32, warps=8, stages=4)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / (R * NREP) * 1e6
        print(f"    @ladder-cfg(16,32,8,4): {us:7.1f} us  {wb/us/1e3:6.0f} GB/s")
