# 06 · W4 GEMM：HIP demo 实测与生产化路线图

> **2026-09-14 当晚生产化完成（P1-P3 + P5 全过）**：`h4mv/` 内核上线，单路 **80.1 t/s**。
> 终态见文末"生产化结果"一节；以下路线图保留作过程记录。

2026-09-14 spike 完成的 **demo 级 HIP W4 matvec**（`w4-hip-demo/`），把"Triton 数学上不可行"
的墙直接翻过去了。本篇 = demo 结果 + "真要啃 W4 GEMM 到底要干啥"的完整清单。

## Demo 成绩（gfx1201，L2 轮转法，模型真实形状）

| 内核 | qkv (6144×4096) | gate_up (28672×4096) | down (4096×14336) | 正确性 |
|---|---|---|---|---|
| v1 逐元素位拼接→f32 fmaf | 213* | 237 | 218 | rel ~1e-7 |
| **v2 LDS 字节LUT + HMUL2 + dot2** | (448*)* | **400** | **397** | rel ~1e-7，64 位置探针全对 |

\* qkv 形状小（12.6MB/缓冲），VRAM 紧只转得起 4 缓冲=52MB，蹭了 64MB L2，数字虚高；gate_up/down 轮转集 248/118MB 是真实 DRAM 数字。

对照锚点：Triton matvec 天花板 163-193 / fork M=1 GEMM 125-205 / **llama.cpp 294** / 理论峰 ~450。

**结论：~150 行的 demo 内核以 397-400 GB/s 流权重，超 llama.cpp 36%、超 Triton 墙 ~2.1×。**
按此速率回填步长账本：每层 GEMM 510µs → ~312µs，单路步长 ~13.7ms ≈ 73 t/s——**光 GEMM 一项
即可越过 llama.cpp 的 68.4，还没算引擎瘦身（3.9ms 间隙 + 0.62ms 投机头）那另一半**。

## 内核设计（两版共用的骨架）

- **一个 wave 一行输出**：无 LDS 激活暂存、无 `__syncthreads` 热路径，wave 内 shfl 归约。
- **布局红利**：一个 `uint4`（16B）= 32 个权重 = **恰好一个 e8m0 scale 组**——scale 流量只有权重的
  1/16，且相邻 lane 读相邻 scale 字节（协同）。scale 需预转置成 `[N, K/32]`（产线 = load-time repack）。
- v1：每 nibble 位拼接 e2m1→f32（指数折叠 scale：`E = e + d + 126`）+ fmaf；~230 VALU ops/16B → 213-237 GB/s（ALU-bound）。
- v2（快 1.9×）：**256 项 LDS LUT 以权重字节为索引** → f16x2 对（零和次正规全部吃进表里，
  运行时零特判）+ 一次 packed f16 乘折 scale（`2^d` 广播；零安全、e2m1×2^d 在 f16 精确）+
  `__builtin_amdgcn_fdot2`（f16 乘积精确、f32 累加）；~75 VALU ops/16B → 存储带宽受限。

## Demo 里踩的三个坑（产线项目的真实学费）

1. **gfx1201 的 hipcc 默认编 wave32**（ISA 里 `.amdhsa_wavefront_size32`，`warpSize=32`）。
   内核必须 wave-size 无关：lane 数学全用 `warpSize`，launcher 多开 block 让多余 wave 提前退出。
   按 wave64 写死的 `__shfl_xor(…, 64)` 会静默丢一半部分和。
2. **e2m1 的两个位级特例**（Triton 时期就踩过，HIP 又踩一遍）：零判据是 `(v & 7) == 0`
   （`0b1000` 是 -0）；次正规（e=0,m=1）值 0.5 的尾数必须为 0。
3. **W 与 X 的配对在翻译时最容易错**：组内第 j 个权重 u32（8 权重）必须配它自己的 8 个 x
   （`xv[j]`），b 字节配 `xb[b]` 字对。两次 bug 都是这一层——用 32 位置单热探针
   （`probe_w4hip.py`）十分钟钉死。

## 真要啃 W4 GEMM：生产化清单（按依赖排序）

**P1 内核生产化（3-5 天专注）**——把 demo 变成引擎能用的算子：
- 激活输入改 **fp8 e4m3**（W4A8 真实路径，x 流量再减半；RDNA4 有硬件 cvt，逐对转 f16 后走 v2 同款 dot2）
- per-tensor 激活 scale 折进 epilogue（数学等价，等价验证）
- K 尾部处理（K%32≠0 的形状）、M≤8 小批量（每 lane 同 x 多行累加，权重只读一遍）
- CUDA graph 兼容（纯 kernel launch 天然可捕获；host 侧不得有查询分支）
- 数值边界：e8m0 的 d 超出 [-14,15] 时 clamp 或回退（真实校准分布远在界内，但要防御）

**P2 load-time repack（1 天）**——checkpoint 加载时把 scale 转置成 `[N, K/32]`（连续协同），
可选：W/scale 交错布局合成单流。llama.cpp 的 repack 同思路，我们已验证必要性（Triton 侧的教训）。

**P3 集成（1-2 天）**——脚手架已验证过（`vllm-w4matvec/fp8kv_w4patch.py` 的 op 级拦截跑通过
ENGAGED）：`torch.library.impl` 覆盖 fork 的 `radiance::mxfp4_linear`，M≤阈值 走自研、
其余回落 fork 自带 GEMM（prefill/skinny M6-64 本来就有快车道，**不需要写通用 GEMM**）。
生效验证：coverage counter + 启动日志。

**P4 引擎瘦身（独立轨道，与内核并行）**——单路要兑现还必须吃掉 ~4.5ms/步非 GEMM 开销：
int2 投机头（若确认无产出，关掉白赚 3%）+ 3.9ms 调度间隙解剖（async 输出处理、调度策略）。

**P5 验证阶梯（贯穿）**：32 位置探针 → 全形状 rel<1e-5 → 真实权重 golden md5（bench_tr2）→
进程内 profile（profhook 确认 GEMM 时间下降）→ bench1 全矩阵（1/4/8 路 × fp8 KV）。

## 风险与未定项

- 微基准 ≠ 引擎内：x 的 L2 命中模式、多 kernel 竞争、launch 开销都会打折；P3 后必须以 profhook 实测为准。
- qkv 形状的真实 DRAM 数字待服务空窗期补测（需要 >96MB 轮转集，现役服务占着显存）。
- fork 若发新版自带更快的 M=1 GEMM（或作者愿意给），先测再写。
- fp8 激活的 e4m3×f16 dot2 精度：乘积仍精确（e4m3 ⊂ f16），累加 f32，预期无损，但要用真实权重 md5 验证。

## 捷径提醒（动手前先过一遍性价比）

1. 给 fork 作者提 issue 要改进 M=1 路径（零工作量，看运气）。
2. 本 demo 已把最大不确定性（Triton 之外是否真有 294+）消灭——HIP 路线从"信念"变成"已测得 400"。
3. （写这段时）若只是要"这台机器单路最快"，llama.cpp 68.4 仍是零成本方案。**当晚 P1–P3 做完后作废**：h4mv 单路 80.1，已经反超。

## 生产化结果（2026-09-14 晚，P1-P3+P5 完成）

**h4mv 生产内核**（`h4mv/`）：demo v2 骨架 + 三处生产化改造——

1. **`V_DOT2_F32_BF16` inline asm**（RDNA3 引入、gfx1201 延续；clang 无 builtin，inline asm 实测发出）：
   x 原生 bf16 零转换、LUT 直出 bf16x2 对（e2m1 在 bf16 精确）、每 2 个 MAC 一条指令、乘积精确、f32 累加——比 f32-FMA 路线快 40%（265→362-401 GB/s），比 demo 的 f16 路线还少一次 x 域转换。
2. **模板 M 特化**（M=1..8 编译期展开）：运行时 M 的谓词展开只有 265/110 GB/s（M=1/4），模板化后 362/401/391（M=1）。**M=8 断崖（132-169）= x 流量撞 L2 带宽墙**（每 16B 权重喂 512B x）→ 分派策略 M≤4 走 h4mv、M≥6 回落 fork skinny（其设计区间）。
3. **组级 scale 一次乘**：`2^(b-127)` 的 f32 位型恰是 `srow[g] << 23`——一个移位、每组一次乘。

集成（`fp8kv_hipw4.py`）：op 级拦截（沿用验证过的偷 `_backend_fns["cuda"]` 机制）+ 懒 repack
（scale 转置缓存 by data_ptr，图捕获期间不分配）+ CUDA graph 兼容（纯 kernel launch 被录进
decode 图，M=1/2/4 档全部走 h4mv）。

**端到端成绩（4k ctx，bench1 同口径，对齐历史基线）**：

| 并发 | 基线 | h4mv | 变化 |
|---|---|---|---|
| 1 路 | 49.9 | **80.1** | **+60%**（超 llama.cpp 68.4 达 17%；真实翻译 86.5 t/s） |
| 4 路 | 147.9 | **199.9** | +35% |
| 8 路 | 223.4 | 223.9 | 持平（M=8 → fork skinny，设计使然） |

当时 prefill 4379 tok/s（M=2048 GEMM 回落 fork，**尚未换 prefill 注意力**）。单路步长 20.04 → 12.48ms。
09-15 prefill tile 之后的 e2e 见 {doc}`scoreboard`（~2050 tok / 345 ms / ~5940 tok/s）。

**质量与稳定性**：h4mv 三轮真实翻译 md5 完全一致；fork 基线反而三轮两个 md5（自身含不确定性）。
译文人工验收干净（术语准确、无退化）。回退验证：`FP8KV_HIPW4=0` 重启即回 49.7 t/s 基线。

## 终局 perf 分解与降精度调研（09-14 深夜，80.1 t/s 配置）

60 步 torch.profiler（`profiling/prof_h4mv.txt`）：步长 12.47ms，GPU-busy 12.07ms，**引擎胶水只剩 0.4ms**（旧 3.9ms 间隙已消失）。

| 成分 | ms/步 | 占比 | 备注 |
|---|---|---|---|
| h4mv W4 GEMM（128 层） | 9.37 | 78% | **引擎内 384 GB/s** = 微基准全额兑现（450 峰值的 85%） |
| 自研 attention（split+reduce） | 1.32 | 11% | 长上下文才放大 |
| int2 draft/lm_head + topk 采样链 | 0.88 | 7% | lm_head=tied embedding（vocab 128167），fork 已量化 int2；采样融合可省 ~0.2ms |
| KV 写 + norm + 量化 | ~0.4 | 3% | |
| 胶水 | ~0.4 | 3% | |

**没有浪费型瓶颈了**；内核打磨余量 ≤1ms。唯一数量级杠杆是权重位数。

**降精度调研结论**（RDNA3/4 指令参考 + QuaRot/SpinQuant/QuIP# 文献）：
- RDNA4 **无 fp8/fp4 标量 dot**——算力全在 WMMA（M≥16，prefill 已吃 325TF）；decode 物理够不着
- VALU 有 `V_DOT4/8_I32_IU4`，但 int4 与 MXFP4 同为 4bit 带宽，decode 带宽受限 → **零收益**（e2m1 非整数网格本也用不了）
- 激活降精度在 decode 无意义（M≤4 仅 32KB）；lm_head fork 已 int2
- **有效路径只有权重 <4bit**：W3=-25% 字节 → 单路 ~+20%（~95 t/s），需旋转+GPTQ 级校准（AMD Quark 官方有 QuaRot 教程，R2/R3/R4 旋转）；W2=-50% → +45%（~115）但需 QuIP# E8 codebook、7B 质量有险。h4mv 改 int3 只是 LUT 换码表，骨架全复用
- KV fp8→fp4+旋转：长上下文选项（16k+ 回本，池再 ×2）

**决策（用户拍板）：W3 等降精度项目暂不开工**，当前 80.1 t/s 收官。

## Prefill 真凶画像（09-15，`profiling/prof_prefill.txt`）

用 uptime/慢步触发的 prefill 专用 profiler（`fp8kv_prof_prefill.py`，v4：该引擎空闲时**不 tick step**，
启动热身/捕获全绕过 step() 直接调模型——所以钩子装好后第一个 >0.2s 的步必是真实请求的 prefill chunk，
无需任何武装）。`prof_prefill.txt` **没有写入该步的 qlen**。能确定的是：

| GPU 项 | 时间 | 占该表 GPU | 判读 |
|---|---|---|---|
| stock `kernel_unified_attention.kd`（32 次，**1525 µs/次**） | 48.8ms | **56%** | prefill 注意力是这一步的 GPU 大头；decode 战役没动这条路径 |
| fork W4A8 GEMM（128 个 `radiance_mxfp4_fp8_gemm_folded`） | 34.3ms | 39% | 已在 fp8 WMMA 量级，不是这条 prefill 的主因 |
| fp8 量化 + silu/norm/KV 写 | ~4.5ms | 5% | 边角料 |

该 dump 的墙钟 666ms 含 profiler；`hipDeviceSynchronize` 一项就是 89ms CPU，**不能**当成无 profiler 的服务墙钟。
无 profiler 的同口径 e2e 见 {doc}`perf-log`（stock ~860 tok ≈ 150ms，不是 468ms）。

**当时的结论**（kernel 还没换）：prefill 4379 tok/s 的锅不在 GEMM，而在 stock 注意力 + CPU 间隙。
flash-prefill 内核 09-15 已上线，见 {doc}`07-prefill`。GEMM 换格式对 prefill 仍然没必要。
