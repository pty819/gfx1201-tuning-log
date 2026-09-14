import torch, time
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

torch.manual_seed(0)
dev = "cuda"
NCTX, HQ, HKV, D, BLK = 4096, 32, 8, 128, 16
NB = (NCTX + BLK - 1) // BLK
scale = D ** -0.5

def build(kv_dtype):
    q = torch.randn(1, HQ, D, dtype=torch.bfloat16, device=dev)
    k = (torch.randn(NB, BLK, HKV, D, device=dev) * 2).to(kv_dtype)
    v = (torch.randn(NB, BLK, HKV, D, device=dev) * 2).to(kv_dtype)
    bt = torch.arange(NB, dtype=torch.int32, device=dev).view(1, NB)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    seqk = torch.tensor([NCTX], dtype=torch.int32, device=dev)
    out = torch.empty(1, HQ, D, dtype=torch.bfloat16, device=dev)
    mode = KVQuantMode.NONE if kv_dtype == torch.bfloat16 else KVQuantMode.FP8_PER_TENSOR
    ds = torch.ones(1, NCTX, device=dev, dtype=torch.float32)
    return dict(q=q, k=k, v=v, out=out, cu_seqlens_q=cu_q, max_seqlen_q=1,
                seqused_k=seqk, max_seqlen_k=NCTX, softmax_scale=scale, causal=True,
                window_size=(-1, -1), block_table=bt, softcap=0.0,
                q_descale=None, k_descale=ds, v_descale=ds, kv_quant_mode=mode), k, v, q

def bench(kv_dtype, n=100):
    args, k, v, q = build(kv_dtype)
    for _ in range(5): unified_attention(**args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): unified_attention(**args)
    torch.cuda.synchronize()
    per_call = (time.perf_counter() - t0) / n * 1000
    print(f"{str(kv_dtype):24s}: {per_call*1000:8.1f} us/call  -> x32层 = {per_call*32:6.2f} ms/step")
    return args["out"].clone(), k, v, q

print("== stock unified_attention, decode 1tok @4k ctx, Hy-MT2 geometry (32Q/8KV/D128) ==")
o_bf, k_bf, v_bf, q = bench(torch.bfloat16)
o_fp, k_fp, v_fp, _ = bench(torch.float8_e4m3fn)

# reference: manual attention on dequantized fp8 cache
kd = k_fp.float().view(NB*BLK, HKV, D)[:NCTX]
vd = v_fp.float().view(NB*BLK, HKV, D)[:NCTX]
qh = q.float().view(HKV, 4, D).transpose(0, 1)  # (4, 8, 128)
ref = torch.empty(4, HKV, D, device=dev)
for g in range(4):
    s = torch.einsum("hd,nhd->hn", qh[g], kd) * scale
    p = torch.softmax(s, dim=-1)
    ref[g] = torch.einsum("hn,nhd->hd", p, vd)
ref = ref.transpose(0, 1).reshape(1, HQ, D).to(torch.bfloat16)
d_fp = (o_fp.float() - ref.float()).abs().max().item()
d_bf = (o_bf.float() - ref.float()).abs().max().item()
print(f"max|out-ref|: fp8={d_fp:.4f}  bf16={d_bf:.4f}  (fp8量化误差基线)")
