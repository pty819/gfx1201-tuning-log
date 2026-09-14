"""Bisection ladder L0-L4: localize what eats the W4 matvec bandwidth. L2-rotation methodology."""
import torch, triton, triton.language as tl, time

@triton.jit
def _L0(w_ptr, o_ptr, K, N, BLOCK_N: tl.constexpr, BLOCK_J: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], tl.float32)
    for j0 in range(0, K // 8, BLOCK_J):
        offs_j = j0 + tl.arange(0, BLOCK_J)
        wq = tl.load(w_ptr + offs_n[:, None] * (K // 8) + offs_j[None, :])
        acc += tl.sum(wq.to(tl.float32), 1)
    tl.store(o_ptr + offs_n, acc)

@triton.jit
def _ladder(w_ptr, x_ptr, o_ptr, K, N, BLOCK_N: tl.constexpr, BLOCK_J: tl.constexpr,
            LEVEL: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    sh = (tl.arange(0, 8) * 4)[None, None, :].to(tl.uint32)
    acc = tl.zeros([BLOCK_N, BLOCK_J * 8], tl.float32)
    for j0 in range(0, K // 8, BLOCK_J):
        offs_j = j0 + tl.arange(0, BLOCK_J)
        wq = tl.load(w_ptr + offs_n[:, None] * (K // 8) + offs_j[None, :])   # (BN,J) u32
        if LEVEL == 1:      # + nibble unpack
            v = ((wq[:, :, None] >> sh) & 0xF).to(tl.float32)
            acc += tl.reshape(v, (BLOCK_N, BLOCK_J * 8))
        elif LEVEL == 2:    # + e2m1->bf16 bit concat
            v = ((wq[:, :, None] >> sh) & 0xF).to(tl.uint32)
            b16 = ((v & 0x8) << 12 | 126 << 7 | (v & 0x1) << 6).to(tl.uint16)
            w = b16.to(tl.bfloat16, bitcast=True).to(tl.float32)
            acc += tl.reshape(w, (BLOCK_N, BLOCK_J * 8))
        elif LEVEL == 3:    # + x multiply
            v = ((wq[:, :, None] >> sh) & 0xF).to(tl.uint32)
            b16 = ((v & 0x8) << 12 | 126 << 7 | (v & 0x1) << 6).to(tl.uint16)
            w = b16.to(tl.bfloat16, bitcast=True).to(tl.float32)
            xk = tl.load(x_ptr + 8 * offs_j[:, None] + tl.arange(0, 8)[None, :]).to(tl.float32)
            acc += tl.reshape(w * xk[None, :, :], (BLOCK_N, BLOCK_J * 8))
    tl.store(o_ptr + offs_n, tl.sum(acc, 1))

def run0(w32, o, bn, bj, wp, st):
    N, K8 = w32.shape; K = K8 * 8
    _L0[(N // bn,)](w32, o, K, N, BLOCK_N=bn, BLOCK_J=bj, num_warps=wp, num_stages=st)

def runL(w32, x, o, level, bn, bj, wp, st):
    N, K8 = w32.shape; K = K8 * 8
    _ladder[(N // bn,)](w32, x, o, K, N, BLOCK_N=bn, BLOCK_J=bj, LEVEL=level,
                        num_warps=wp, num_stages=st)

def bench(fn, NREP=8, R=40):
    for i in range(NREP): fn(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in range(R * NREP): fn(it % NREP)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (R * NREP) * 1e6

if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    NREP = 8
    print(f"{'shape':>14s} " + " ".join(f"{l:>8s}" for l in ["L0", "L1", "L2", "L3", "L4"]))
    for (N, K, tag) in [(6144, 4096, "qkv"), (12288, 4096, "gate/up"), (4096, 12288, "down")]:
        pools = []
        for _ in range(NREP):
            w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
            x = (torch.randn(1, K, device=dev) * 0.5).to(torch.bfloat16)
            o = torch.empty(N, device=dev, dtype=torch.float32)
            pools.append((w.view(torch.uint32), x, o))
        wb = N * K / 2
        row = []
        row.append(bench(lambda i: run0(pools[i][0], pools[i][2], 16, 32, 8, 4)))
        for lv in (1, 2, 3):
            row.append(bench(lambda i, lv=lv: runL(pools[i][0], pools[i][1], pools[i][2], lv, 16, 32, 8, 4)))
        # L4 = real v2 kernel
        import sys; sys.path.insert(0, "/work")
        from fp8kv_w4mv2 import _w4_matvec as _real
        s_pool = [torch.randint(124, 131, (K // 32, N), dtype=torch.uint8, device=dev) for _ in range(NREP)]
        o16_pool = [torch.empty(1, N, device=dev, dtype=torch.bfloat16) for _ in range(NREP)]
        def run4(i):
            w32, x, _ = pools[i]
            _real[(N // 16,)](x, w32, s_pool[i], o16_pool[i], K, N,
                              BLOCK_N=16, BLOCK_J=32, num_warps=8, num_stages=4)
        row.append(bench(run4))
        print(f"{tag + f' {N}x{K}':>14s} " + " ".join(
            f"{wb/us/1e3:5.0f}B/s" if False else f"{wb/us/1e3:7.0f}" for us in row))
        print(f"{'':>14s} " + " ".join(f"{us:7.1f}us" for us in row))
