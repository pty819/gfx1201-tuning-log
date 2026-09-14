# -*- coding: utf-8 -*-
# W4 HIP demo kernels (v1 / v2): correctness vs torch reference + L2-rotation bandwidth.
# Run INSIDE the vllm-radiance container (torch + ROCm 7.14 + gfx1201).
import ctypes, json
import torch

torch.cuda.init()
free, total = torch.cuda.mem_get_info()
print(f"[mem] free={free//1048576}MiB / total={total//1048576}MiB")

lib = ctypes.CDLL("/tmp/w4demo/libw4demo.so")
for fn in ("w4matvec_v1_launch", "w4matvec_v2_launch"):
    getattr(lib, fn).argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    getattr(lib, fn).restype = None
lib.get_warpsize_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

ws = torch.zeros(1, dtype=torch.int32, device="cuda")
lib.get_warpsize_launch(ctypes.c_void_p(ws.data_ptr()),
                        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
torch.cuda.synchronize()
print(f"[env] device warpSize = {ws.item()}")

def launch(ver, W, St, X, Y, N, K):
    st = torch.cuda.current_stream().cuda_stream
    fn = lib.w4matvec_v1_launch if ver == 1 else lib.w4matvec_v2_launch
    fn(W.data_ptr(), St.data_ptr(), X.data_ptr(), Y.data_ptr(), N, K, ctypes.c_void_p(st))

try:
    cfg = json.load(open("/models/hy-mt2-7b-awq2-mxfp4/config.json"))
    h = cfg["hidden_size"]; inter = cfg["intermediate_size"]
    nq = cfg.get("num_attention_heads"); nkv = cfg.get("num_key_value_heads")
    hd = cfg.get("head_dim") or (h // nq)
    SHAPES = {"qkv": ((nq + 2*nkv)*hd, h), "gate_up": (2*inter, h), "down": (h, inter)}
    print(f"[shapes] h={h} inter={inter} nq={nq} nkv={nkv} hd={hd}")
except Exception as e:
    SHAPES = {"qkv": (6144, 4096), "gate_up": (2*14336, 4096), "down": (4096, 14336)}
    print(f"[shapes] config unreadable ({e}); defaults")

E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

def make_case(N, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W  = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda", generator=g)
    S  = torch.randint(122, 133, (K//32, N), dtype=torch.uint8, device="cuda", generator=g)
    St = S.t().contiguous()                                   # [N, K/32]
    Xb = torch.randn(K, dtype=torch.float32, device="cuda", generator=g).to(torch.bfloat16)
    Xh = Xb.to(torch.float16)                                 # same values, f16 for v2
    return W, S, St, Xb, Xh

def reference(W, S, X, chunk=2048):
    N, KW = W.shape; K = KW*2
    lut = torch.tensor(E2M1_LUT, dtype=torch.float32, device="cuda")
    scale = torch.exp2(S.float() - 127.0).t()
    xf = X.float()
    out = torch.empty(N, dtype=torch.float32, device="cuda")
    for r0 in range(0, N, chunk):
        r1 = min(r0+chunk, N)
        w = torch.empty(r1-r0, K, dtype=torch.float32, device="cuda")
        w[:, 0::2] = lut[(W[r0:r1] & 0xF).long()]; w[:, 1::2] = lut[(W[r0:r1] >> 4).long()]
        w = w.view(r1-r0, K//32, 32) * scale[r0:r1].view(r1-r0, K//32, 1)
        out[r0:r1] = (w.view(r1-r0, K) * xf).sum(dim=1)
    return out

def bench(ver, name, N, K, nb=12, iters=25):
    torch.cuda.synchronize()
    free_b, _ = torch.cuda.mem_get_info()
    bufs = []
    per = (N*(K//2) + N*(K//32) + 4*K + 8*N)
    nb = max(nb, -(-(96 << 20) // per))            # rotation set must exceed 64MB L2
    need = nb*per + (1<<26)
    if need > free_b:
        nb = max(4, int((free_b - (1<<26)) // per))
        print(f"[bench:{name}] VRAM tight, nb -> {nb}")
    for i in range(nb):
        bufs.append(make_case(N, K, 1000+i))
    Y = torch.empty(N, dtype=torch.float32, device="cuda")
    W, S, St, Xb, Xh = bufs[0]
    X = Xb if ver == 1 else Xh
    launch(ver, W, St, X, Y, N, K); torch.cuda.synchronize()
    ref = reference(W, S, X)
    rel = ((Y - ref).abs() / ref.abs().clamp_min(ref.abs().max())).max().item()
    for b in bufs:
        launch(ver, b[0], b[2], (b[3] if ver == 1 else b[4]), Y, N, K)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        for W, S, St, Xb, Xh in bufs:
            launch(ver, W, St, (Xb if ver == 1 else Xh), Y, N, K)
    e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / iters
    byt = (N*(K//2) + N*(K//32))
    gbps = nb * byt / (ms/1e3) / 1e9
    ok = "OK " if rel < 1e-4 else "BAD"
    print(f"[v{ver}:{name}] N={N} K={K} nb={nb}: {ms*1e3:7.1f} us/op  {gbps:5.0f} GB/s  rel={rel:.1e} [{ok}]")
    del bufs, Y, ref; torch.cuda.empty_cache()
    return gbps

if __name__ == "__main__":
    print("anchors: Triton matvec 163-193 | fork M=1 GEMM 125-205 | llama.cpp 294 | peak ~450")
    res = {}
    for ver in (1, 2):
        for name, (N, K) in SHAPES.items():
            res.setdefault(name, {})[ver] = bench(ver, name, N, K)
    print("\n=== summary (DRAM GB/s, weights+scales) ===")
    for name, d in res.items():
        print(f"  {name:8s} v1={d[1]:5.0f}  v2={d[2]:5.0f}")
