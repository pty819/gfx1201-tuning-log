import sys, re
sys.path.insert(0, "/work")
import torch, triton
import triton.language as tl
from fp8kv_w4mv5 import _w4_matvec5
from w4mv_diag import _ladder

def dump(kern, name, gridkw):
    try:
        ck = kern.warmup(*gridkw["args"], **gridkw["consts"], grid=(1,))
        asm = ck.asm.get("amdgcn", "")
    except Exception as e:
        print(f"{name}: warmup path failed ({e!r}), trying compile cache")
        asm = ""
    if not asm:
        print(f"{name}: no asm")
        return
    stats = {
        "vgpr_max": (re.search(r"\.max_vgpr[s]?\s*[:=]?\s*(\d+)", asm) or
                     re.search(r"vgpr[s]?_count\s*[:=]?\s*(\d+)", asm) or [None,"?"])[1],
        "sgpr": (re.search(r"\.max_sgpr[s]?\s*[:=]?\s*(\d+)", asm) or [None,"?"])[1],
        "load_x4": len(re.findall(r"global_load_dwordx4", asm)),
        "load_x2": len(re.findall(r"global_load_dwordx2", asm)),
        "load_dword": len(re.findall(r"global_load_dword\b", asm)),
        "load_ubyte": len(re.findall(r"global_load_ubyte", asm)),
        "ds_read": len(re.findall(r"ds_read", asm)),
        "ds_write": len(re.findall(r"ds_write", asm)),
        "v_and": len(re.findall(r"\bv_and\b", asm)),
        "v_lshl": len(re.findall(r"\bv_lshl\b", asm) + re.findall(r"\bv_lshlrev\b", asm)),
        "v_add": len(re.findall(r"\bv_add\b", asm) + re.findall(r"\bv_add_u32\b", asm)),
        "v_cvt": len(re.findall(r"\bv_cvt\b", asm)),
        "v_fma": len(re.findall(r"\bv_fma|v_mac\b", asm)),
        "s_load": len(re.findall(r"\bs_load_dword", asm)),
        "total_lines": asm.count("\n"),
    }
    print(f"== {name} ==")
    for k, v in stats.items():
        print(f"  {k:11s}: {v}")

N, K = 6144, 4096
dev = "cuda"
w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
w32 = w.view(torch.uint32)
sT = torch.randint(124, 131, (N, K // 32), dtype=torch.uint8, device=dev)
x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
o16 = torch.empty(1, N, device=dev, dtype=torch.bfloat16)
of32 = torch.empty(N, device=dev, dtype=torch.float32)

dump(_w4_matvec5, "v6 full (scale fold, sT)",
     dict(args=(x, w32, sT, o16, K, N), consts=dict(BLOCK_N=16, BLOCK_J=32, num_warps=8, num_stages=4)))
dump(_ladder, "L3 (no scales)",
     dict(args=(w32, x, of32, K, N), consts=dict(BLOCK_N=16, BLOCK_J=32, LEVEL=3, num_warps=8, num_stages=4)))
