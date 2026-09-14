import torch, triton, triton.language as tl, time

@triton.jit
def _dummy(w_ptr, o_ptr, K, N, BLOCK_N: tl.constexpr, BLOCK_J: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], tl.float32)
    for j0 in range(0, K // 8, BLOCK_J):
        offs_j = j0 + tl.arange(0, BLOCK_J)
        wq = tl.load(w_ptr + offs_n[:, None] * (K // 8) + offs_j[None, :])
        acc += tl.sum(wq.to(tl.float32), 1)
    tl.store(o_ptr + offs_n, acc)

def bench(N, K, tag):
    w32 = torch.randint(0, 2**32, (N, K // 8), dtype=torch.uint32, device="cuda")
    o = torch.empty(N, device="cuda", dtype=torch.float32)
    best = None
    for bn in (8, 16, 32, 64):
        for bj in (32, 64, 128):
            for wp in (4, 8):
                try:
                    _dummy[(N // bn,)](w32, o, K, N, BLOCK_N=bn, BLOCK_J=bj, num_warps=wp)
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    for _ in range(300):
                        _dummy[(N // bn,)](w32, o, K, N, BLOCK_N=bn, BLOCK_J=bj, num_warps=wp)
                    torch.cuda.synchronize()
                    us = (time.perf_counter() - t0) / 300 * 1e6
                    if best is None or us < best[0]: best = (us, bn, bj, wp)
                except Exception: pass
    wb = N * K / 2
    print(f"  {tag:9s} N={N:5d}: pure-stream {best[0]:6.1f} us = {wb/best[0]/1e3:5.0f} GB/s (BN={best[1]} BJ={best[2]} w={best[3]})")

bench(4096, 4096, "qkv-ish")
bench(12288, 4096, "gate/up")
bench(4096, 12288, "down")
