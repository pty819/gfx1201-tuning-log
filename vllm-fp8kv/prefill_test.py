# -*- coding: utf-8 -*-
# Standalone test for fp8kv_prefill: correctness vs f32 reference + perf.
import sys, time
sys.path.insert(0, "/opt/fp8kv")
import torch
from fp8kv_prefill import run

torch.manual_seed(0)
dev = "cuda"
HQ, HKV, D, PBLK, GQA = 32, 8, 128, 16, 4
scale = D ** -0.5

def make(ctx, q0, qlen, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    NB = (ctx + PBLK - 1) // PBLK          # real vLLM always allocates ceil blocks
    k = (torch.randn(NB, PBLK, HKV, D, device=dev, generator=g) * 2).to(torch.float8_e4m3fn)
    v = (torch.randn(NB, PBLK, HKV, D, device=dev, generator=g) * 2).to(torch.float8_e4m3fn)
    q = torch.randn(qlen, HQ, D, dtype=torch.bfloat16, device=dev, generator=g)
    bt = torch.arange(NB, dtype=torch.int32, device=dev).view(1, NB)
    out = torch.empty(qlen, HQ, D, dtype=torch.bfloat16, device=dev)
    return q, k, v, bt, out

def reference(q, k, v, q0, ks=1.0, vs=1.0):
    qlen = q.shape[0]
    ctx_kv = k.shape[0] * k.shape[1]
    kd = k.float().view(ctx_kv, HKV, D) * ks
    vd = v.float().view(ctx_kv, HKV, D) * vs
    qf = q.float()                                  # [qlen, HQ, D]
    ref = torch.empty_like(qf)
    for h in range(HQ):
        kh = h // GQA
        s = torch.einsum("md,nd->mn", qf[:, h], kd[:, kh]) * scale
        rows = torch.arange(qlen, device=dev) + q0
        mask = torch.arange(ctx_kv, device=dev)[None, :] > rows[:, None]
        s = s.masked_fill(mask, float("-inf"))
        p = torch.softmax(s, -1)
        ref[:, h] = torch.einsum("mn,nd->md", p, vd[:, kh])
    return ref

def check(name, ctx, q0, qlen, ks=1.0, vs=1.0):
    q, k, v, bt, out = make(ctx, q0, qlen, seed=q0 + qlen)
    kt = torch.tensor([ks], device=dev, dtype=torch.float32)
    vt = torch.tensor([vs], device=dev, dtype=torch.float32)
    run(q, k, v, out, bt, kt, vt, scale, q0, qlen)
    torch.cuda.synchronize()
    ref = reference(q, k, v, q0, ks, vs)
    err = (out.float() - ref).abs().max().item()
    print(f"  {name}: max err = {err:.5f}  {'OK' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, name

def perf_one(ctx, q0, qlen, BM, BN, w, st):
    q, k, v, bt, out = make(ctx, q0, qlen, seed=1)
    one = torch.ones(1, device=dev, dtype=torch.float32)
    try:
        run(q, k, v, out, bt, one, one, scale, q0, qlen, BM=BM, BN=BN, warps=w, stages=st)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            run(q, k, v, out, bt, one, one, scale, q0, qlen, BM=BM, BN=BN, warps=w, stages=st)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 50 * 1e6
        print(f"RESULT ctx={ctx} q0={q0} qlen={qlen} BM={BM} BN={BN} w={w} st={st}: "
              f"{us:7.1f} us/layer (x32={us*32/1000:.2f} ms/chunk, stock 48.8)")
    except Exception as e:
        print(f"RESULT ctx={ctx} BM={BM} BN={BN} w={w} st={st}: FAIL {str(e)[:60]}")


def perf(ctx, q0, qlen):
    q, k, v, bt, out = make(ctx, q0, qlen, seed=1)
    one = torch.ones(1, device=dev, dtype=torch.float32)
    for BM, BN, w, st in ((128, 128, 8, 2), (128, 64, 8, 2), (64, 128, 8, 2),
                          (64, 128, 4, 2), (64, 64, 4, 2), (128, 128, 4, 2)):
        try:
            run(q, k, v, out, bt, one, one, scale, q0, qlen, BM=BM, BN=BN, warps=w, stages=st)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                run(q, k, v, out, bt, one, one, scale, q0, qlen, BM=BM, BN=BN, warps=w, stages=st)
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) / 50 * 1e6
            print(f"  ctx={ctx} q0={q0} qlen={qlen} BM={BM} BN={BN} w={w} st={st}: "
                  f"{us:7.1f} us/layer  (x32 = {us*32/1000:.2f} ms/chunk; stock was 48.8ms)")
        except Exception as e:
            print(f"  BM={BM} BN={BN} w={w} st={st}: FAIL {str(e)[:60]}")

if __name__ == "__main__":
    import os as _os
    cfg = _os.environ.get("PF_CFG")
    if cfg:
        BM, BN, w, st = (int(x) for x in cfg.split(","))
        ctx, q0, qlen = (int(x) for x in _os.environ.get("PF_SHAPE", "2048,0,2048").split(","))
        perf_one(ctx, q0, qlen, BM, BN, w, st)
        sys.exit(0)
    print("=== correctness ===")
    check("first chunk causal (q0=0, qlen=2048, ctx=2048)", 2048, 0, 2048)
    check("continuation (q0=2048, qlen=850, ctx=2898)", 2898, 2048, 850)
    check("small tail (q0=3840, qlen=256, ctx=4096)", 4096, 3840, 256)
    check("scales ks=0.5 vs=2.0", 2048, 0, 2048, ks=0.5, vs=2.0)
    print("all correctness checks passed")
