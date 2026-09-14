import torch, triton, triton.language as tl, time

@triton.jit
def _probe(k_ptr, o_ptr, NITER: tl.constexpr, TILE: tl.constexpr, D: tl.constexpr,
           MODE: tl.constexpr):
    # stream NITER tiles of (TILE, D), convert per MODE, reduce-sum, store 1 f32
    offs_t = tl.arange(0, TILE)
    offs_d = tl.arange(0, D)
    acc = tl.zeros([TILE, D], tl.float32)
    for i in range(NITER):
        p = (i * TILE * D + offs_t[:, None] * D + offs_d[None, :])
        if MODE == 0:      # bf16 load, no convert
            x = tl.load(k_ptr + p).to(tl.float32)
        elif MODE == 1:    # fp8 load, convert f32
            x = tl.load(k_ptr + p).to(tl.float32)
        elif MODE == 2:    # fp8 load, convert direct to bf16
            x = tl.load(k_ptr + p).to(tl.bfloat16).to(tl.float32)
        elif MODE == 3:    # fp8 as raw uint8 (test pure load bandwidth)
            x = tl.load(k_ptr.to(tl.pointer_type(tl.uint8)) + p).to(tl.float32)
        else:              # fp8 via manual bit expansion (e4m3 -> f32)
            u = tl.load(k_ptr.to(tl.pointer_type(tl.uint8)) + p).to(tl.uint32)
            bits = (((u & 0x80) << 24) | ((u & 0x78) << 20) | ((u & 0x07) << 20) | 0x3C000000)
            x = bits.to(tl.float32, bitcast=True)
        acc += x
    tl.store(o_ptr, tl.sum(acc))

def bench(t, mode, tile=64, D=128, niter=64, warps=8):
    N = tile * D * niter
    o = torch.zeros(1, device="cuda", dtype=torch.float32)
    grid = (1,)
    _probe[grid](t, o, NITER=niter, TILE=tile, D=D, MODE=mode, num_warps=warps)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50): _probe[grid](t, o, NITER=niter, TILE=tile, D=D, MODE=mode, num_warps=warps)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / 50 * 1e6
    gbs = (N * t.element_size()) / (us * 1e-6) / 1e9
    return us, gbs, o.item()

dev = "cuda"
N = 64 * 128 * 64
bf = torch.randn(N, device=dev, dtype=torch.bfloat16)
fp = bf.to(torch.float8_e4m3fn)
print(f"tile=64x128 x64iter ({N*1:,} elems), 1 program, warps=8")
for mode, name in [(0, "bf16 load          "), (1, "fp8 -> f32         "), (2, "fp8 -> bf16        "),
                   (3, "fp8 raw u8         "), (4, "fp8 bitcast-math   ")]:
    t = bf if mode == 0 else fp
    us, gbs, s = bench(t, mode)
    print(f"  {name}: {us:8.1f} us  {gbs:7.1f} GB/s   checksum={s:.1f}")
