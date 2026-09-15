# 07 · Prefill 注意力（2026-09-15）

decode 和 W4 GEMM 收官之后，长 prompt 的墙钟仍被 prefill 咬住。profiler（{doc}`perf-log` 09-15 stock 条）把 GPU 时间拆开：注意力 56%、GEMM 39%。GEMM 已经是 fork 的 fp8 WMMA；要动的是注意力。

## 不是 r4d 的问题

`r4d 0.5.0` 只编了 `attn_prefill/decode_h256_gqa6_{bf16kv,fp8kv}`。Hy-MT2 是 **h128 / GQA=4**，`r4d.select()` 返回空，从第一天起 prefill 就走上游 Triton。banner 上的 `RADIANCE_USE_R4D ON` 只管开关，不管几何。

所以「手写 WMMA 打不过 r4d」不成立——这条路径上没有 r4d 注意力。对照物是 **vLLM `unified_attention`**（引擎内 1525 µs/layer）和后来自写的 Triton v2。

## 自写 Triton v2 → v2.1

`vllm-fp8kv/fp8kv_prefill.py`。v1 的教训已经写在文件头：`tl.dot` 会降成 gfx12 WMMA；慢在 `tl.trans(K)`、`tl.exp`、每块都套因果 mask、`num_stages>=3` spill。v2 去掉转置、改 `exp2`、全可见砖 / 斜砖两段循环、stages 锁 2。

v2.1 没改算法，只改 launch 配置。隔离网格里生产默认 `BM=128 BN=128 warps=8 stages=2` 是最慢的之一（3360 µs @2048）。最快且 4-case 过关的是 **`64/32/2/2`（1680 µs）**。

挂进 serving：`PYTHONPATH=/opt/fp8kv` + `FP8KV_PREFILL=1`，`sitecustomize` import。monkey-patch `TritonAttentionImpl.forward`。vLLM 0.28 的 metadata **没有** `num_computed_tokens`，要用 `q0 = max_seq_len - qlen`。cascade / SWA / 多序列直接回落。成功时 EngineCore 打 `[fp8kv-prefill] ENGAGED qlen=… q0=…`。

e2e：~2050 tok TTFT 395 → **345 ms**（~5940 tok/s）；decode 仍 ~83 tok/s。

## 手写 HIP 为什么停

同一几何写过 gfx12 WMMA 的 HIP 内核（staging `pf_hip.cu`，未进生产）。正确性过了，2048 token **2323 µs**，慢于 Triton v2.1 的 1680。

原因不是「HIP 发不出 WMMA」。两边 ISA 都是 WMMA。差在占用：HIP 版 **VGPR=167**，RDNA4 每个 SIMD 大约 256 个 VGPR，`floor(256/167)=1`，同时只能蹲 1 条 wave。Triton 把寄存器编得更紧，同样的 MMA 被延迟藏住。

要再打过 Triton，先把 VGPR 压到 ≤128（2 波）最好 ≤64（4 波）。继续改 LDS 形状没有意义。

## 还剩什么

- 按 `(qlen, ctx)` autotune tile，不要一个 64/32 打所有 chunk。
- 给 r4d 加 h128/gqa4 编译变体——那才是真用 r4d 注意力。
- 多序列 / cascade / SWA 仍走 stock。
