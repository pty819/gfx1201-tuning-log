# 实测日志

每条测量一个小节，字段固定。不要把过期数字改写进 {doc}`scoreboard` 而不在这里留底。

字段：

- **日期** / **层**（微基准 / 引擎 profiler / e2e）
- **形状**（qlen、ctx、并发、几何）
- **方法**（脚本、次数、中位还是均值、防缓存）
- **改动**（相对哪条基线改了什么）
- **数字**（表）
- **备注**（门控、是否挂上、已知偏差）

---

## 2026-09-14 · decode 内核 + h4mv 收官

层
: e2e decode

形状
: 4k ctx；1/4/8 路

方法
: `bench/bench1.py`，P/T 两相，`ignore_eos`，nonce 防前缀缓存

改动
: fp8-KV flash-decode Triton + h4mv HIP W4 GEMM 上线

数字
: 单路 80.1 / 4 路 199.9 / 8 路 223.9 tok/s；步长 12.48 ms；KV 池 80,960

备注
: 过程见 {doc}`03-fp8kv-pathology-and-kernel`、{doc}`06-w4gemm-hip-roadmap`

---

## 2026-09-15 · 引擎内 stock prefill 注意力

层
: 引擎 profiler

形状
: 引擎内一次 prefill step。`prof_prefill.txt` **未写 qlen**。`prefill_test.py` 里的 `stock 48.8` 是写死的注释，不是这次测出来的。

方法
: `profiling/fp8kv_prof_prefill.py` / `prof_prefill.txt`，1 个 EngineCore step（含 profiler 税，墙钟 666ms）

改动
: 无（当时 `FP8KV_PREFILL` 未开）

数字
: `kernel_unified_attention.kd` **1525 µs/layer × 32 = 48.8 ms**，占该表 GPU 56%；GEMM 34.3 ms（39%）

备注
: 口头「stock 1525µs」= 这条 kernel 时间。不能当成 850 tok 或 2048 tok 的隔离基准。无 profiler 的服务墙钟用 e2e 条（~860 tok stock ≈ 150 ms）。

---

## 2026-09-15 · 手写 HIP WMMA prefill v4

层
: 微基准

形状
: HQ=32 HKV=8 D=128 GQA=4；official 4-case（2048 first / 850 续写 / 64 small / ks=0.5 vs=2.0）

方法
: `pf_hip_test.py`，正确性 vs f32 einsum；perf 100 次均值

改动
: gfx12 `__builtin_amdgcn_wmma_*`，O 路径 operand-swap，Q 从 global 取，NW=8，K/V 软件流水

数字
: 正确性 ALL OK（max err 0.020–0.034）；2048 = **2323 µs/layer**；850 续写 = 2376 µs

备注
: 元数据 `.vgpr_count=167` → SIMD 同时只能驻留 1 条 wave32。Triton 同几何、同样发 WMMA，隔离测 1680 µs。停手，不再改 LDS 形状。r4d 注意力仍只有 h256/gqa6，这条几何碰不到 r4d。

---

## 2026-09-15 · Triton v2 tile 网格

层
: 微基准

形状
: 2048 first；850 续写 ctx=2898

方法
: `vllm-fp8kv/prefill_test.py` / 临时网格脚本，warmup 后 50 次均值

改动
: 只改 `BM, BN, warps, stages`，kernel 正文不动

数字（2048 first，µs/layer）：

| BM | BN | warps | stages | µs |
|---|---|---|---|---|
| 128 | 128 | 8 | 2 | **3360**（当时生产默认，网格最差之一） |
| 128 | 64 | 8 | 2 | 2241 |
| 64 | 64 | 4 | 2 | 2029 |
| 64 | 64 | 4 | 1 | 1868 |
| 64 | 32 | 4 | 2 | 1795 |
| 32 | 32 | 2 | 2 | 1767 |
| **64** | **32** | **2** | **2** | **1687** |
| 64 | 16 | 4 | 2 | 1834 |
| 64 | 256 | 8 | 2 | FAIL LDS 98KB > 64KB |

正确性：上表能跑通的配置 4-case 全过，max err 与 64/32/2/2 同为 0.016–0.033。

备注
: v2 算法（无 `tl.trans(K)`、`exp2`、两段因果、`stages=2`）已经定稿。`stages>=3` 会 spill（v1 测过 13.4 ms）。生产默认改为 **64/32/2/2**。

---

## 2026-09-15 · 隔离 kernel 确认现行默认

层
: 微基准

形状
: 850 / 2048 / 4096 续写

方法
: 改默认后 `run()` 无参启动，50–80 次

数字
:

| 形状 | µs/layer |
|---|---|
| qlen=850 q0=0 ctx=850 | 405 |
| qlen=2048 q0=0 ctx=2048 | 1680 |
| qlen=850 q0=2048 ctx=2898 | 1627 |
| qlen=2048 q0=2048 ctx=4096 | 4483 |

备注
: 405 µs @850 已明显快于引擎内旧 stock 1525 µs。4483 µs 的 4k 续写是 KV 长度 4096、query 2048，计算量大约 ×2。

---

## 2026-09-15 · e2e，kernel 未挂上（负对照）

层
: e2e prefill

形状
: ~860 / ~2050 tok，单请求

方法
: `bench/bench_e2e.py`，uuid nonce，`max_tokens=1`，中位 5 次

改动
: 容器已带新 tile 代码，但 `hasattr(attn_metadata, "num_computed_tokens")` 在 vLLM 0.28 为假，快路径从未进入

数字
: ~860 tok → 150 ms / 5760 tok/s；~2050 tok → **395 ms / 5200 tok/s**；decode 短 prompt 64 tok 墙钟 769 ms ≈ 83 tok/s

备注
: 日志只有 `[fp8kv-prefill] installed`，没有 `ENGAGED`。这组数字 = 引擎 stock 注意力 + 新代码未生效。

---

## 2026-09-15 · e2e，kernel 挂上

层
: e2e prefill + decode 冒烟

形状
: 同上

方法
: 同上。门改为 `q0 = max_seq_len - qlen`，排除 cascade。重启后日志 `(EngineCore) [fp8kv-prefill] ENGAGED qlen=111 q0=0`

改动
: 快路径真正接管 `max_query_len>=64` 的单序列 prefill

数字
:

| 形状 | 中位 TTFT | tok/s | vs 未挂上 |
|---|---|---|---|
| ~860 tok | **143 ms** | **6030** | −7 ms |
| ~2050 tok | **345 ms** | **5940** | −50 ms |
| decode 180+64 | 769 ms ≈ 83 tok/s | — | 不变 |

备注
: 2048 档省 50 ms，和隔离 kernel 3360→1680 µs × 32 层 ≈ 54 ms 对得上。decode 不走这条 kernel。
