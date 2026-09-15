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

墙钟含引擎调度，不是纯 kernel。同方法 A/B 只有上表两行（挂 / 未挂 kernel）。

## 注意力微基准（µs/layer）

Hy-MT2 几何：HQ=32 HKV=8 D=128 GQA=4 PBLK=16。
隔离测与引擎内 profiler **不是同一套方法**，不要把 1525 填进 850 列再和 405 除。

| 内核 | 850 first（隔离） | 2048 first（隔离） | 850 续写 ctx=2898（隔离） |
|---|---|---|---|
| 自写 Triton v2 默认 128/128/8/2 | — | 3360 | 2975 |
| **现行 Triton v2.1 64/32/2/2** | **405** | **1680** | 1627 |
| 手写 HIP WMMA v4 NW=8 | — | 2323 | 2376 |

引擎内 stock `kernel_unified_attention.kd`：一次 prefill step 上 **1525 µs/层 × 32 = 48.8 ms**（`prof_prefill.txt`）。该 dump **没有 qlen**，不能标成 850 或 2048。和自写 kernel 的同方法对照用上面的 e2e，不用这张表。HIP 正确但 VGPR=167，已停。详见 {doc}`07-prefill`。

## Decode 冒烟（不是 80.1 同口径）

短 prompt ~180 + 生成 64：墙钟 769 ms，64/0.769 ≈ 83 tok/s。这是 **含 prefill 的整段墙钟**，不是 `bench1.py` 4k ctx 的生成速率 80.1。只能说明 prefill kernel 上线后 decode 没崩（`qlen=1` 不走 prefill 快路径）。
