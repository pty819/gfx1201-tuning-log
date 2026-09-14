# -*- coding: utf-8 -*-
import ctypes, torch
lib = ctypes.CDLL("/tmp/w4demo/libw4demo.so")
for fn in ("w4matvec_v1_launch", "w4matvec_v2_launch"):
    getattr(lib, fn).argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

def launch(ver, W, St, X, Y, N, K):
    st = torch.cuda.current_stream().cuda_stream
    (lib.w4matvec_v1_launch if ver == 1 else lib.w4matvec_v2_launch)(
        W.data_ptr(), St.data_ptr(), X.data_ptr(), Y.data_ptr(), N, K, ctypes.c_void_p(st))

N, K = 1, 32                       # ONE group, single row: isolate pairing
St = torch.full((1, N), 127, dtype=torch.uint8, device="cuda").t().contiguous()
Y = torch.empty(N, dtype=torch.float32, device="cuda")
print("probe: weight=1.0 (nibble 2) at single k, x=one-hot, expect Y=1.0")
bad = []
for k in range(32):
    W = torch.zeros(N, K//2, dtype=torch.uint8, device="cuda")
    byte, nib = k // 2, k % 2
    W[0, byte] = 2 if nib == 0 else (2 << 4)
    Xb = torch.zeros(K, dtype=torch.bfloat16, device="cuda"); Xb[k] = 1.0
    Xh = Xb.to(torch.float16)
    for ver, X in ((1, Xb), (2, Xh)):
        launch(ver, W, St, X, Y, N, K); torch.cuda.synchronize()
        y = Y[0].item()
        if abs(y - 1.0) > 1e-6: bad.append((k, ver, y))
if not bad:
    print("all 64 probes (32k x 2 kernels) exact")
else:
    for k, ver, y in bad: print(f"  k={k:2d} v{ver}: Y={y:.6f}")

# full-row compare
torch.manual_seed(0)
N, K = 8, 4096
W  = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
S  = torch.full((K//32, N), 127, dtype=torch.uint8, device="cuda")
St = S.t().contiguous()
Xb = torch.arange(K, dtype=torch.float32, device="cuda").remainder(7).sub(3).to(torch.bfloat16)
Xh = Xb.to(torch.float16)
Y  = torch.empty(N, dtype=torch.float32, device="cuda")
lut = torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32, device="cuda")
w = torch.empty(N, K, dtype=torch.float32, device="cuda")
w[:, 0::2] = lut[(W & 0xF).long()]; w[:, 1::2] = lut[(W >> 4).long()]
for ver, X in ((1, Xb), (2, Xh)):
    launch(ver, W, St, X, Y, N, K); torch.cuda.synchronize()
    ref = (w * X.float()).sum(1)
    print(f"v{ver} full-row: maxerr={(Y-ref).abs().max().item():.2e}  ratio0={(Y[0]/ref[0]).item():.4f}")
