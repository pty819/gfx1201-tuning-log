"""M=1 W4 matvec v3: row-streaming — one program owns rows, reads each row contiguously."""
import torch, triton, triton.language as tl, time


@triton.jit
def _w4_matvec_rows(x_ptr, w_ptr, s_ptr, o_ptr, K, N, ROWS_PER_PROG,
                    BLOCK_J: tl.constexpr):
    pid = tl.program_id(0)
    sh = (tl.arange(0, 8) * 4)[None, :].to(tl.uint32)
    ar8 = tl.arange(0, 8)[None, :]
    GJ: tl.constexpr = BLOCK_J // 4     # scale groups per chunk (32k = 4 u32 cols)
    for r in range(ROWS_PER_PROG):
        row = pid * ROWS_PER_PROG + r
        acc = 0.0
        for j0 in range(0, K // 8, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)
            wq = tl.load(w_ptr + row * (K // 8) + offs_j)                    # (BJ,) u32 contiguous
            v = ((wq[:, None] >> sh) & 0xF).to(tl.uint32)                    # (BJ, 8)
            ee = (v & 0x6) >> 1
            mant = tl.where(ee == 0, tl.zeros_like(v), (v & 0x1) << 6)
            b16 = (((v & 0x8) << 12) | ((ee + 126) << 7) | mant).to(tl.uint16)
            w = b16.to(tl.bfloat16, bitcast=True).to(tl.float32)
            w = tl.where((v & 0x7) == 0, 0.0, w)
            xk = tl.load(x_ptr + 8 * offs_j[:, None] + ar8).to(tl.float32)   # (BJ, 8)
            sg = tl.load(s_ptr + ((8 * j0) // 32 + tl.arange(0, GJ)) * N + row)   # (GJ,)
            sc = ((sg.to(tl.uint32) << 23).to(tl.float32, bitcast=True))
            sc_j = tl.reshape(tl.broadcast_to(sc[:, None], (GJ, 4)), (BLOCK_J,))
            acc += tl.sum(w * xk * sc_j[:, None])
        tl.store(o_ptr + row, acc.to(tl.bfloat16))


def run(x, w32, s, out, BLOCK_J=256, rpp=1, warps=4, stages=3):
    N, K8 = w32.shape
    K = K8 * 8
    grid = (N // rpp,)
    _w4_matvec_rows[grid](x, w32, s, out, K, N, rpp,
                          BLOCK_J=BLOCK_J, num_warps=warps, num_stages=stages)


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
        for bj in (64, 128, 256, 512):
            for rpp in (1, 2, 4):
                for wp in (1, 2, 4, 8):
                    for st in (2, 4):
                        try:
                            for i in range(NREP):
                                w32, sc, x, out = pool[i]
                                run(x, w32, sc, out, BLOCK_J=bj, rpp=rpp, warps=wp, stages=st)
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            R = 40
                            for it in range(R * NREP):
                                w32, sc, x, out = pool[it % NREP]
                                run(x, w32, sc, out, BLOCK_J=bj, rpp=rpp, warps=wp, stages=st)
                            torch.cuda.synchronize()
                            us = (time.perf_counter() - t0) / (R * NREP) * 1e6
                            if best is None or us < best[0]:
                                best = (us, bj, rpp, wp, st)
                        except Exception:
                            pass
        wb = N * K / 2
        print(f"  {tag:9s} N={N:5d} K={K:5d}: {best[0]:7.1f} us  {wb/best[0]/1e3:6.0f} GB/s  "
              f"(BJ={best[1]} rpp={best[2]} w={best[3]} st={best[4]})")
