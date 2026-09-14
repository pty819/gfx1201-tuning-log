"""M=1 W4(MXFP4) x bf16 matvec: replaces the fork's slow decode-band GEMM at M==1.
Weight: [N, K/2] uint8 raw checkpoint order (low nibble = even k). Scales: [K/32, N] e8m0."""
import torch, triton, triton.language as tl


@triton.jit
def _w4_matvec(x_ptr, w_ptr, s_ptr, o_ptr, K, N,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    HALF_K: tl.constexpr = BLOCK_K // 2
    G: tl.constexpr = BLOCK_K // 32
    acc = tl.zeros([BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_j = k0 // 2 + tl.arange(0, HALF_K)
        wb = tl.load(w_ptr + offs_n[:, None] * (K // 2) + offs_j[None, :])   # (BN, HK) u8
        offs_g = k0 // 32 + tl.arange(0, G)
        sg = tl.load(s_ptr + offs_g[None, :] * N + offs_n[:, None])          # (BN, G) u8
        sc = (sg.to(tl.uint32) << 23).to(tl.float32, bitcast=True)           # 2^(e-127)
        sc_h = tl.reshape(tl.broadcast_to(sc[:, :, None], (BLOCK_N, G, 16)),
                          (BLOCK_N, HALF_K))
        x_lo = tl.load(x_ptr + 2 * offs_j).to(tl.float32)
        x_hi = tl.load(x_ptr + 2 * offs_j + 1).to(tl.float32)
        vlo = (wb & 0x0F).to(tl.uint32)
        vhi = (wb >> 4).to(tl.uint32)
        e = (vlo & 0x6) >> 1
        m = (vlo & 0x1).to(tl.float32)
        mag = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2(e.to(tl.float32) - 1.0))
        wlo = tl.where((vlo & 0x8) != 0, -mag, mag)
        e2 = (vhi & 0x6) >> 1
        m2 = (vhi & 0x1).to(tl.float32)
        mag2 = tl.where(e2 == 0, m2 * 0.5, (1.0 + m2 * 0.5) * tl.exp2(e2.to(tl.float32) - 1.0))
        whi = tl.where((vhi & 0x8) != 0, -mag2, mag2)
        acc += tl.sum(wlo * sc_h * x_lo[None, :], 1)
        acc += tl.sum(whi * sc_h * x_hi[None, :], 1)
    tl.store(o_ptr + offs_n, acc.to(tl.bfloat16))


def run(x, w, s, out, BLOCK_N=16, BLOCK_K=256, warps=4):
    N, Kh = w.shape
    K = Kh * 2
    _w4_matvec[(N // BLOCK_N,)](x, w, s, out, K, N,
                                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                num_warps=warps)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    dev = "cuda"
    E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                         -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=dev)

    def make(N, K):
        w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
        s = torch.randint(124, 131, (K // 32, N), dtype=torch.uint8, device=dev)
        x = (torch.randn(1, K, device=dev) * 0.5).to(torch.bfloat16)
        return w, s, x

    def ref(w, s, x):
        N, Kh = w.shape; K = Kh * 2
        codes = torch.stack([w & 0x0F, w >> 4], -1).reshape(N, K).long()
        sc = torch.pow(2.0, s.float() - 127.0).T.repeat_interleave(32, dim=1)
        return (x.float() @ (E2M1[codes] * sc).T)

    print("== correctness ==")
    for (N, K) in [(256, 4096), (1024, 4096), (96, 12288)]:
        w, s, x = make(N, K)
        out = torch.empty(1, N, device=dev, dtype=torch.bfloat16)
        run(x, w, s, out)
        r = ref(w, s, x)
        rel = ((out.float() - r).norm() / r.norm()).item()
        mx = (out.float() - r).abs().max().item()
        print(f"  N={N} K={K}: rel={rel:.5f} max={mx:.5f}")
        assert rel < 0.01, "WRONG"

    print("== perf (us / GB/s vs weights) ==")
    for (N, K, tag) in [(4096, 4096, "qkv-ish"), (1024, 4096, "k/v"),
                        (12288, 4096, "gate/up"), (4096, 12288, "down")]:
        w, s, x = make(N, K)
        out = torch.empty(1, N, device=dev, dtype=torch.bfloat16)
        best = None
        for bn in (16, 32, 64):
            for bk in (128, 256, 512):
                for wp in (2, 4, 8):
                    try:
                        run(x, w, s, out, BLOCK_N=bn, BLOCK_K=bk, warps=wp)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(300):
                            run(x, w, s, out, BLOCK_N=bn, BLOCK_K=bk, warps=wp)
                        torch.cuda.synchronize()
                        us = (time.perf_counter() - t0) / 300 * 1e6
                        if best is None or us < best[0]:
                            best = (us, bn, bk, wp)
                    except Exception:
                        pass
        wb = N * K / 2
        print(f"  {tag:9s} N={N:5d} K={K:5d}: {best[0]:7.1f} us  {wb/best[0]/1e3:6.0f} GB/s  "
              f"(BN={best[1]} BK={best[2]} w={best[3]})")
