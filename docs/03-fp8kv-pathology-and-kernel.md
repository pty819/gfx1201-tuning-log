# 03 · fp8 KV 病灶链与自写 decode 内核（成功案例）

2026-09-14 完成：为 vllm-radiance 写了 gfx1201 的 fp8-KV decode attention Triton 内核，
**fp8 KV 从"不可用"变成"最优配置"**。

## 病灶发现（09-13）

`--kv-cache-dtype fp8` 后 decode 反而暴跌：4k ctx **33.8 ms/步 vs bf16 KV 20.2 ms/步**（fp8 白付 +13.6ms/步，40% decode 损失）；
短上下文两者同 ~19ms → 开销随 KV 长度线性涨（带宽型症状）。当时的临时方案是切回 `--kv-cache-dtype auto`（bf16），
代价 KV 池 67.7k→33.8k tokens。

## 归因过程（三层，前两次推断都被推翻）

1. **默认后端是 `RocmAttentionImpl`**（rocm.py 优先级表第一个）——fp8 KV 慢 68% 的元凶在这条路。
2. **r4d 手写注意力从未生效**：r4d.so 按固定几何编译（`r4d.ATTN_HEAD_DIM/ATTN_GQA/ATTN_BLOCK_SIZE` = 256 / 6 / 16），
   Hy-MT2 是 h128/gqa4（本仓隔离测试与 `fp8kv_prefill.py` 的 PBLK=16）——`r4d.select()` 返回 None → **静默回退上游 stock Triton attention（不崩，只打印拒绝原因）**。
   serve banner `RADIANCE_USE_R4D ON` 只是开关标志，有误导。r4d 入口全量 dump 证实解码器注意力只有
   `attn_prefill/decode_h256_gqa6_{bf16kv,fp8kv}` 一族 4 个入口（另有 Gemma4 vision h72、GDN 族——都不是给本模型的）。
3. **stock Triton 的 fp8 分支慢**：fp8 KV 走 per-tensor + inline descale 进统一 kernel；SM89+ 硬门只对 `is_cuda()` 生效（ROCm 放行）。
   慢机制 = **Triton AMD 后端 fp8→f32 转换是慢速模拟（171µs vs fp8→bf16 直转 23.9µs，9×）**，疑因 fp8 1 字节标量加载无法向量化。
   当时的对照：fp8 KV prefill 4647 ≈ bf16 4648 tok/s——只说明 **fp8 vs bf16 的转换税没在 prefill 吞吐上表现出来**，
   推不出「prefill 注意力已经够快」。09-15 profiler 里 stock prefill 注意力仍占该步 GPU 的 56%，见 {doc}`07-prefill`。

另：`VLLM_ATTENTION_BACKEND` env 在 vLLM 0.28 已失效，必须用 `--attention-backend TRITON_ATTN` 参数（短名，全模块路径会被拒）。

## 内核设计（`vllm-fp8kv/fp8kv_v3.py`）

flash-decode split-KV 两段式：

- **主 kernel** grid = (seq, kv_head, 8 splits)，TILE=32 warps=4；每 split 算 partial (o, m, l)；
- **reduce kernel** 合并 8 路 partial——partial m/l 按 **BM=16 行向量**存（`tl.max(m_i)` 存标量的版本 err 0.84，是本项目最深的正确性坑）；
- K/V 按 (TILE, D) 原生布局加载（D 连续轴合并访存）+ `tl.trans`，不要学 stock 的转置加载；
- **fp8→bf16 位级直转**（e4m3→bf16 精确，无精度损失）；
- **per-tensor scale 数学折叠**：k_scale 乘进 sm_scale、v_scale 乘在 reduce 输出——数学等价、图安全；
  kernel 内 `tl.load` 读标量，杜绝 host 侧 `float()` 同步破坏 CUDA graph 捕获；
- 正确性：vs f32 参考 err≈0.002；scale 折叠用 ks=.5/vs=2 构造用例验证精确。

独立微基准 78.5µs/层 = stock 2D 路径 796µs 的 **1/10**。serving 图内 stock Triton fp8 实际 47.1 t/s
（3D 分段路径比微基准的 2D 好），所以图内净增益：对 stock-Triton +6%，对默认 ROCM_ATTN fp8 **+68%**。

## 集成方式（不打镜像）

- volume-mount `~/fp8kv-dev:/opt/fp8kv` + `-e PYTHONPATH=/opt/fp8kv` → `sitecustomize.py` 自动 import；
- monkey-patch `TritonAttentionImpl.forward`：条件 `max_query_len==1 且 FP8_PER_TENSOR 且 head_dim==128` 走自研 kernel，否则回落 stock（SWA/sinks/alibi/per-token-head scale 安全回落）；
- v1 注意力经 `torch.ops.vllm.unified_attention_with_output` custom op 运行时动态调 `self.impl.forward`——类级 patch 有效，且 decode 图回放不进 Python（计数器证实），照常工作；
- **prefill 同款病**用同一招修：patch `_tua._cast_kv_tile`（bf16 直转 + bf16 域乘 scale，scale=1.0 时精确；jit 函数的模块属性替换必须在首次 trace 前生效）；
- 全部代码在 `vllm-fp8kv/fp8kv_override.py`，带调用计数器与 `FP8KV_FORCE_TRITON` 选择器。

## 端到端成绩（4k ctx，**09-14 decode 内核落地、h4mv 之前**）

| 指标 | 数值 |
|---|---|
| 单路 | **49.9 t/s** |
| 4 路 | **147.9 t/s** |
| 8 路 | **223.4 t/s** |
| prefill | 4385 tok/s（vs 当时 ROCM_ATTN 4647，-6%；prefill kernel 是 09-15 的事） |
| KV 池 | 80,960 tokens（9.88× @8k） |

并发全面超当时的 bf16 KV 配置（139/210），池子是 bf16 的 2 倍，翻译质量无退化（bench_tr2 md5 对照）。
现行数字（h4mv + prefill tile）见 {doc}`scoreboard`。

## 深挖 perf（profiling/fp8kv_profhook.py）

进程内剖析 = sitecustomize 注入的 `EngineCore.step_with_batch_queue` 包装器（busy loop 包 step；
镜像裁了 /start_profile 路由，离线 LLM 子进程又不吃 sitecustomize——只有 serving 进程吃）。
这是 **49.9 t/s / h4mv 之前** 的 decode 剖析。60 步 GPU-busy 16.19ms：GEMM ~82%、
自研 decode attention 0.57ms（3.5%，**decode 注意力病灶**到此解决）、
fp8 激活量化 0.39ms、int2 投机头 0.62ms/步；墙钟 20.04 − GPU-busy ≈ **3.9ms** 调度间隙。
eager 29.5 vs graph 49.8 t/s——Python 派发 ~17ms/步被 CUDA graph 隐藏。
h4mv 之后胶水只剩 0.4ms，见 {doc}`06-w4gemm-hip-roadmap`。

prefill 注意力的后续战役（tile、HIP 停手、e2e）见 {doc}`07-prefill` 与 {doc}`perf-log`。
