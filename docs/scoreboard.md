# 成绩单

只放**现行有效**的数字。历史对照、失败实验、微基准网格见 {doc}`perf-log`。同口径才能横比：4k ctx 并发用 `bench/bench1.py`；prefill e2e 用 `bench/bench_e2e.py`（uuid 打掉前缀缓存，`max_tokens=1`）。

## 端到端 decode（4k ctx）

2026-09-14，`bench/bench1.py`。

| 配置 | 1 路 | 4 路 | 8 路 | KV 池 |
|---|---|---|---|---|
| llama.cpp Vulkan + q8 KV | 68.4 | 125 | 136 | 96k |
| vLLM bf16 KV | 49.4 | 139 | 210 | 33.8k |
| vLLM fp8 KV（stock 注意力，病态） | 29.6 | — | — | 67.7k |
| vLLM fp8 KV + decode kernel | 49.9 | 148 | 223 | 80,960 |
| **现行：+ h4mv HIP W4 GEMM** | **80.1** | **199.9** | **223.9** | 80,960 |

单路步长 20.04 → 12.48 ms。llama.cpp 退居零依赖备份。

## 端到端 prefill（2026-09-15）

单请求，`bench/bench_e2e.py` 中位 5 次。服务日志需出现 `[fp8kv-prefill] ENGAGED`。

| 形状 | TTFT | tok/s | 对照 |
|---|---|---|---|
| ~860 tok first chunk | **143 ms** | **6030** | 未挂 kernel：~150 ms / 5760 |
| ~2050 tok first chunk | **345 ms** | **5940** | 未挂 kernel：395 ms / 5200 |

32 层粗算：860 tok ≈ 4.5 ms/层，2050 tok ≈ 10.8 ms/层。墙钟含引擎调度，大于纯 kernel。

## 注意力微基准（隔离 kernel，µs/layer）

Hy-MT2 几何：HQ=32 HKV=8 D=128 GQA=4 PBLK=16。

| 内核 | 2048 first | 850 续写 ctx=2898 | 正确性 |
|---|---|---|---|
| 引擎内 stock `unified_attention`（profiler） | 1525 | — | 生产旧路径 |
| 自写 Triton v2 默认 128/128/8/2 | 3360 | 2975 | OK |
| **现行 Triton v2.1 64/32/2/2** | **1680** | **1627** | max err 0.016–0.033 |
| 手写 HIP WMMA v4 NW=8 | 2323 | 2376 | ALL OK；VGPR=167，SIMD 驻留 1 波 |

手写 HIP 正确但打不过现行 Triton，已停。详见 {doc}`07-prefill`。

## Decode 质量门

短 prompt ~180 + 生成 64：墙钟 769 ms ≈ **83 tok/s**（2026-09-15，prefill kernel 上线后）。与 80.1 同档，prefill 路径不碰 `qlen=1`。
