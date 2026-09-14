# gfx1201 调优实录（RX 9070 GRE · vLLM / llama.cpp）

> 一张 RDNA4 消费卡（AMD RX 9070 GRE 12GB, gfx1201）上，把 7B 翻译模型 Hy-MT2-7B 的推理服务从
> "vLLM 单路 29.6 tok/s" 推到 "单路 49.9 / 8 路聚合 223 tok/s" 的完整战役记录。
> 包含两次成功的自写 Triton 内核、一次被数学证明不可行的尝试、以及大量防重蹈的负结论。
>
> 时间跨度：2026-09-02 → 2026-09-14。所有数字均为实测（测量方法见 [docs/05-playbook.md](docs/05-playbook.md)）。

## 硬件与软件栈

| 项 | 值 |
|---|---|
| GPU | AMD RX 9070 GRE 12GB（RDNA4, gfx1201，无 HSA_OVERRIDE） |
| CPU | Ryzen 9700X（Zen 5, AVX-512 + BF16） |
| 系统 | CachyOS，GPU 功耗档锁 high |
| llama.cpp | 自编 HIP + ZenDNN + Vulkan 一体树 `~/llama.cpp/build-amd`（09-12 更新至 790cf51，含 gfx1201 专属 FA 调优） |
| vLLM | `magiccodingman/vllm-radi` image（v1.0.16；vLLM 0.28.0 / radiance v0.9.3-dev / r4d 0.5.0 / ROCm 7.14，Docker 专打 gfx1201；源码仓私有） |
| 模型 | Hy-MT2-7B（32 层 / hidden 4096 / 32Q+8KV 头 / head_dim 128，GQA=4），自量化 Quark-AWQ-MXFP4 W4A8（4.66GB）；llama.cpp 侧用 Q4_K_M GGUF |
| ROCm 工具链 | 容器内 ROCm 7.14；宿主 `~/rocm10`（TheRock 产线）用于编 llama.cpp |

该 fork 的快车道（手写内核）只有三样几何无关的东西对所有模型开放：FP8 GEMM、Quark-MXFP4 W4A8 GEMM
（`radiance_mxfp4_fp8.so`，RDNA4 原生 FP8 WMMA：手写 325 TF vs 通用 Triton 43 TF）、skinny GEMM / all-reduce。
**手写注意力 R4D 按固定几何编译，只伺候 head_dim=256/GQA=6 一族**——这是后面所有故事的开端。

## 时间线总览

| 日期 | 事件 | 结果 |
|---|---|---|
| 09-02 | llama.cpp HIP 首版构建 | 可用 |
| 09-04 | HIP + ZenDNN + Vulkan 一体树（9cc3394） | tg128：HIP 74.4 / Vulkan 77.6 t/s |
| 09-08 | llama.cpp 值守 :8080（Q4_K_M + q8 KV + unified 池） | 1路71 / 4路180 / 8路210 t/s |
| 09-12 | vLLM 全链路：FP8 试运行→删；Quark-AWQ-MXFP4 v1→v2（OPUS-100 校准修重复词退化）；llama.cpp 拉新 166 commits | 生产量化版定稿 4.66GB |
| 09-13 | 三家 4 并发对比、并发梯度 crossover、ngram 投机实测、**fp8 KV decode 病态发现**（4k ctx 33.8 vs bf16 20.2 ms/步） | 临时切 bf16 KV |
| 09-13/14 | 源码侦察：R4D 几何不匹配从未接管 → stock Triton fp8→f32 慢转换是根因；定 Triton 路线可行性 | 只需写一个 decode kernel |
| 09-14 | **自写 fp8-KV flash-decode split-KV Triton 内核落地** | fp8 KV 变最优配置：49.9 / 148 / 223 t/s，KV 池 80,960 tokens |
| 09-14 | **W4A8 matvec 四版尝试 → 二分阶梯 + ISA 定罪 → 数学证明 Triton 不可行** | 负结论归档，回退 fork GEMM |
| 09-14 | 常驻服务切换：vLLM 最优配置值守 :8080，llama.cpp 退居备用 | 收官 |

## 最终成绩单（4k ctx，`bench/bench1.py` 同口径）

| 配置 | 1 路 | 4 路 | 8 路 | prefill | KV 池 |
|---|---|---|---|---|---|
| llama.cpp Vulkan + q8 KV | **68.4** | 125 | 136 | ~2.8k tok/s | 96k tokens |
| vLLM bf16 KV | 49.4 | 139 | 210 | **~5.1k** tok/s | 33.8k |
| vLLM fp8 KV（stock，病态） | 29.6 | — | — | ~4.6k | 67.7k |
| **vLLM fp8 KV + 自写 kernel（现行）** | 49.9 | **148** | **223.4** | 4.4k tok/s | **80,960** |

选型结论：**prefill 重的负载永远 vLLM；≤3 路 decode 用 llama.cpp（单路仍领先 37%）；≥4 路用 vLLM + 自写 fp8 KV 内核**。
单路 decode 的剩余差距在 W4 GEMM matvec（llama.cpp 294 GB/s vs vLLM 侧 ~180-205 GB/s），Triton 语言层已证明闭合不了，唯 HIP/布局路线——见 [docs/04](docs/04-w4a8-matvec-wall.md)。

## 两次内核攻坚

### 1. fp8-KV flash-decode（成功，`vllm-fp8kv/`）

**病灶链**（前面几次推断都被推翻，最终归因三层）：

1. 默认注意力后端是 `RocmAttentionImpl`，fp8 KV 慢 68% 的元凶在这条路；
2. r4d 手写内核因几何不匹配（要 h256/gqa6，本模型 h128/gqa4）**从第一天起就没启用过**——serve banner `RADIANCE_USE_R4D ON` 只是开关标志，有误导；
3. `--attention-backend TRITON_ATTN` 强切后 fp8 仍慢：**Triton AMD 后端的 fp8→f32 转换是慢速模拟（171µs vs fp8→bf16 直转 23.9µs，9×）**，疑因 fp8 1 字节标量加载无法向量化。

**解法**：flash-decode split-KV 两段式 Triton 内核（主 kernel grid=seq×kv_head×8 split + reduce kernel），
关键点：fp8→bf16 位级直转（精确）+ per-tensor scale 数学折叠（k_scale 乘进 sm_scale、v_scale 乘在 reduce 输出——
graph-safe：kernel 内 `tl.load` 标量，杜绝 host `float()` 同步）；partial m/l 按 BM=16 行向量存（不是标量，踩过坑）。
独立微基准 78.5µs/层 = stock 2D 路径 796µs 的 1/10。prefill 的同款慢转换用 monkey-patch `_tua._cast_kv_tile` 修掉。

**集成方式**（不打镜像）：volume-mount 本目录 + `-e PYTHONPATH=/opt/fp8kv`，`sitecustomize.py` 自动 import →
monkey-patch `TritonAttentionImpl.forward`（条件 max_query_len==1 且 FP8_PER_TENSOR 且 h128 走自研 kernel，否则回落 stock）。
v1 注意力经 `torch.ops.vllm.unified_attention_with_output` custom op 动态调 `self.impl.forward`，类级 patch 有效且 decode 图回放照常工作。

### 2. W4A8 matvec（负结论，`vllm-w4matvec/`）

四版内核（tile → u32 位拼接 → 行流 → MMA）全部正确但 L2-轮转实测天花板 163-193 GB/s，与 fork 自带 GEMM 打平。
二分阶梯（L0 纯加载 → L4 完整 kernel）把代价定位到 scale 应用（~30% 税），八个优化假设逐一排除；
**ISA 实锤：vgpr 89 vs 60（+29 寄存器 → wave/CU 8→5，占空度 -38%）**——Triton 在"权重+scale 双加载流"下有结构性代码生成缺陷。
决定性账本：即使 scale 免费（L3 速率），步长 16.8ms ≈ 60 t/s，仍追不上 llama.cpp 68.4——**数学上不可能，到此为止**。
llama.cpp 的 294 GB/s 证明空间存在，钥匙在 HIP/显式布局协同设计。

## 目录索引

```
docs/            五篇阶段详录（栈与量化 / llama.cpp / fp8KV 内核 / W4 matvec / 手册）
vllm-fp8kv/      注意力内核全套：kernel v1/v3、注入 override、sitecustomize、最优 serve 脚本
vllm-w4matvec/   matvec 五版、op 级拦截 patch、二分阶梯、ISA 统计工具
profiling/       EngineCore 进程内 profiler 钩子、微基准、两份实测 profile 输出
bench/           通用并发基准（P/T 两相、ignore_eos）、真实翻译 md5 等价校验
llamacpp/        llama.cpp 启动包装、Vulkan 值守/回切脚本、env wrapper
```

## 方法论沉淀（可复用的招）

- **L2 轮转法打假**：9070 有 64MB L2，单缓冲微基准会跑出 700+ GB/s 的 L2 假象；必须 8 缓冲轮转测真实 DRAM 带宽。
- **性能二分阶梯**：L0 纯加载 → L1 解包 → L2 位拼接 → L3 乘激活 → L4 加 scale，逐级看带宽衰减，把"慢"钉到具体一步。
- **ISA 定罪**：`kernel.warmup()` 后读 `asm["amdgcn"]`，vgpr 数直接换算 wave/CU 占空度（注意 `\b` 正则匹配不到 `v_and_b32` 这类带后缀助记符）。
- **进程内 profiler**：fork 裁掉了 /start_profile 路由，就 monkey-patch `EngineCore.step_with_batch_queue`（busy loop 包住）自己采 torch.profiler；离线 `vLLM.LLM` 子进程不吃 sitecustomize，要 serving 进程才吃。
- **md5 等价校验**：改动前后的内核跑同一组真实翻译（4 段不重复），输出 md5 必须一致——投机解码、cast 补丁都用这招验无损。
- **op 级拦截**：torch.compile 图内的算子（如 `radiance::mxfp4_linear`）Python 侧只有 `torch.library.impl` 覆盖有效；CustomOpDef 没有 `.fn` 属性，要偷 `_backend_fns["cuda"]` 否则递归。

## 未竟之路

- **HIP 原生 W4 GEMM**（数周级）：唯一能闭合单路差距的路线，参考 `radiance_mxfp4_fp8.hip` + vLLM Marlin + r4d 现成的 `gemm_w4a16_nt_m64`。
- **给 fork 作者提 issue** 要 head_dim=128/GQA=4 的 r4d 编译变体（入口命名机制天生支持多几何，对作者可能只是加一行编译目标）。
- fp8 KV 的 per-token-head scale 模式、SWA/sinks/alibi、投机解码路径、head_dim≠128 的覆盖。
- INT4-W4A16 普通 AWQ checkpoint 的原生加速：该 fork 无加载路径，捷径排序 = requant→MXFP4（零工作量）> Triton v1（几天）> HIP（数周）。

---

*测量环境注意：bench 前必须停掉另一套服务（vLLM 占 92% 显存会让 llama.cpp load fail，报错像 GGUF 坏了）；9070 功耗档锁 high；两套服务切换见 `llamacpp/vk_launch.sh` 与 `llamacpp/switch_to_vllm.sh`。*
