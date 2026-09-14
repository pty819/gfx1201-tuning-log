# -*- coding: utf-8 -*-
# h4mv production kernel test: correctness probes (M=1..8, all k positions)
# + L2-rotation bandwidth on the three real shapes.
import ctypes, json
import torch

torch.cuda.init()
lib = ctypes.CDLL("/opt/fp8kv/libh4mv.so")
lib.h4mv_launch.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3 + [ctypes.c_void_p]
lib.h4mv_launch.restype = None

def launch(W, St, X, Y, N, K, M):
    lib.h4mv_launch(W.data_ptr(), St.data_ptr(), X.data_ptr(), Y.data_ptr(),
                    N, K, M, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))

E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

def reference(W, S, X, chunk=2048):
    N, KW = W.shape; K = KW*2; M = X.shape[0]
    scale = torch.exp2(S.float() - 127.0).t()          # [N, K/32]
    lut = torch.tensor(E2M1_LUT, dtype=torch.float32, device="cuda")
    xf = X.float()
    out = torch.empty(M, N, dtype=torch.float32, device="cuda")
    for r0 in range(0, N, chunk):
        r1 = min(r0+chunk, N)
        w = torch.empty(r1-r0, K, dtype=torch.float32, device="cuda")
        w[:, 0::2] = lut[(W[r0:r1] & 0xF).long()]; w[:, 1::2] = lut[(W[r0:r1] >> 4).long()]
        ws = w.view(r1-r0, K//32, 32) * scale[r0:r1].view(r1-r0, K//32, 1)
        out[:, r0:r1] = xf @ ws.view(r1-r0, K).t()
        del w, ws
    return out

def bf16_pair(v):  # exact bf16 rounding for reference output
    return v.to(torch.bfloat16).float()

# ---- probes: one-hot positions, M rows, real scales -----------------------
def probe():
    N, K, M = 4, 32, 8
    St = torch.full((N, 1), 127, dtype=torch.uint8, device="cuda")  # scale=1... shape [N,K/32]
    fails = 0
    for k in range(32):
        W = torch.zeros(N, K//2, dtype=torch.uint8, device="cuda")
        byte, nib = k // 2, k % 2
        W[:, byte] = 2 if nib == 0 else (2 << 4)      # weight 1.0 everywhere in row? set all rows
        X = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda"); X[:, k] = 1.0
        Y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        launch(W, St, X, Y, N, K, M); torch.cuda.synchronize()
        if not torch.allclose(Y.float(), torch.full((M, N), 1.0, device="cuda"), atol=1e-3):
            print(f"  probe FAIL k={k}: Y[0]={Y[0].tolist()}"); fails += 1
    print(f"probes (32 positions x M=8): {'ALL OK' if fails == 0 else str(fails)+' FAILS'}")
    return fails

def full_case(N, K, M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W  = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda", generator=g)
    S  = torch.randint(122, 133, (K//32, N), dtype=torch.uint8, device="cuda", generator=g)
    St = S.t().contiguous()
    X  = torch.randn(M, K, dtype=torch.float32, device="cuda", generator=g).to(torch.bfloat16)
    return W, S, St, X

def bench(name, N, K, M, nb=12, iters=25):
    torch.cuda.synchronize()
    free_b, _ = torch.cuda.mem_get_info()
    per = N*(K//2) + N*(K//32) + M*2*K + M*2*N
    nb = max(nb, -(-(96 << 20) // per))
    if nb * per + (1 << 26) > free_b:
        nb = max(4, int((free_b - (1 << 26)) // per))
    bufs = [full_case(N, K, M, 1000+i) for i in range(nb)]
    Y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    W, S, St, X = bufs[0]
    launch(W, St, X, Y, N, K, M); torch.cuda.synchronize()
    ref = reference(W, S, X)
    err = ((Y.float() - ref).abs() / ref.abs().clamp_min(ref.abs().max())).max().item()
    for b in bufs: launch(b[0], b[2], b[3], Y, N, K, M)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        for W, S, St, X in bufs: launch(W, St, X, Y, N, K, M)
    e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / iters
    gbps = nb * (N*(K//2) + N*(K//32)) / (ms/1e3) / 1e9
    ok = "OK " if err < 5e-3 else "BAD"
    print(f"[{name} M={M}] N={N} K={K} nb={nb}: {ms*1e3:7.1f}us  {gbps:5.0f} GB/s  rel={err:.1e} [{ok}]")
    del bufs, Y, ref; torch.cuda.empty_cache()
    return gbps

if __name__ == "__main__":
    free, _ = torch.cuda.mem_get_info()
    print(f"[mem] free={free//1048576}MiB")
    if probe() == 0:
        try:
            cfg = json.load(open("/models/hy-mt2-7b-awq2-mxfp4/config.json"))
            h, inter = cfg["hidden_size"], cfg["intermediate_size"]
            nq, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
            SHAPES = {"qkv": ((nq+2*nkv)*128, h), "gate_up": (2*inter, h), "down": (h, inter)}
        except Exception as e:
            print(f"[shapes] config unreadable: {e}"); SHAPES = {}
        for name, (N, K) in SHAPES.items():
            for M in (1, 4, 8):
                bench(name, N, K, M)
