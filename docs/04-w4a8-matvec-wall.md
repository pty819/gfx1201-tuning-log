# 04 · W4A8 matvec：四面碰壁与结构性定罪（负结论）

2026-09-14 的第二个内核项目，目标是闭合单路 decode 差距（vLLM 49.9 vs llama.cpp 68.4 t/s）。
**结论：Triton 语言层数学上不可能做到，已用数据证明并回退。** 本篇是防重蹈档案。

## 背景与目标

单路 decode 时步长分解里 GEMM 占 ~82%，而 W4A8 的 M=1 matvec 是纯带宽操作。
llama.cpp 在同一张卡上把 Q4 权重流做到 **294 GB/s**，vLLM 侧（fork 的 `radiance_mxfp4_fp8` GEMM 与通用 Triton）
只有 ~125-205 GB/s——这就是单路差距的全部来源。目标：写一个更快的 Triton W4 matvec。

权重格式（Quark AWQ-MXFP4，WPERM=0 原始序）：weights `[N, K/2]` uint8（每字节两个 e2m1），
scales `[K/32, N]` e8m0（scale = 2^(x-127)）。

## 四版内核（vllm-w4matvec/，全部数值正确）

| 版本 | 思路 |
|---|---|
| w4mv (v1) | 朴素 tile 版 |
| w4mv2/v3 | u32 加载 + **e2m1→bf16 位拼接直转**（`bits = s<<15 \| (ee+126)<<7 \| m<<6`；**subnormal 修正：ee=0 时尾数必须为 0**，否则 0.75≠0.5） |
| w4mv4 | 行流式变体 |
| w4mv5 | dup-gather / 预转置协同加载 sT + **指数折叠**（`E_field = ee + sg - 1` 把 e8m0 scale 折进 bf16 指数位，精确且免乘法） |

**性能天花板：L2-轮转实测 163-193 GB/s**——与 fork 自带 GEMM 的 M=1 路径（125-205 GB/s）打平，没有增量。

> 测量方法警告：单缓冲微基准会跑出 700+ GB/s 的假象（9070 有 **64MB L2**，数据全在缓存里），
> 必须 8 缓冲轮转逼它去 DRAM。
>
> 归档一个重要误判修正：曾把"variant1 = 309µs"读成坏掉的 M=1 路径（41-80GB/s）、把模板参数 `<8,128,4,1>` 错当 M 得出
> "M=4 兄弟 335GB/s"。图内 re-profile 证明 variant1 实际是 50MB 融合 gate+up @164GB/s，fork 的 M=1 GEMM 全家
> 125-205GB/s——**与自研 matvec 本来就是平手**。

## 二分阶梯（w4mv_diag.py，L2 轮转法）

逐级叠加操作，看带宽在哪一级崩：

| 阶梯 | qkv / gate+up / down (GB/s) | 增量代价 |
|---|---|---|
| L0 纯加载 | 295 / 340 / 205 | 访存模式天花板 |
| L1 +解包 | 289 / 339 / 219 | 免费 |
| L2 +位拼接 | 259 / 314 / 206 | ~5% |
| L3 +乘激活 | 232 / 282 / 197 | ~10% |
| **L4 +scale** | **172 / 186 / 148** | **scale 税 ~30%** |

八个优化假设**逐一系统排除**（全部无效）：f32 乘改指数折叠、reshape-broadcast 改重复 gather、
scale 预转置协同加载、tile 全谱扫描（BJ 8-128 / BN 8-64）、warps 2-8、stages 2-8（st=6 反而掉到 170）、
MMA 路线、行流式。

## ISA 定罪（isa_dump.py）

读 `kernel.warmup()` 后的 `asm["amdgcn"]`：

- **vgpr：89（v6 完整版） vs 60（L3 无 scale 版）**——多 29 个寄存器把 wave/CU 从 ~8 压到 ~5（**占空度 -38%**），代码量 +38%；
- 但小 tile 降寄存器也不恢复带宽 → 不是单纯的寄存器压力问题；
- 判定：**Triton 流水线/代码生成在"权重 + scale 双加载流"下有结构性缺陷，语言层无解**——
  这正是手写 HIP 用显式异步拷贝、手工软件流水才能处理的那类问题。

（工具坑：`\b` 正则匹配不到 `v_and_b32` 这类带后缀助记符，统计 ISA 要用更宽的模式。）

## 决定性的账本（为什么停手）

即使 scale 完全免费（按 L3 速率跑）：

- 每层 GEMM ≈ 410µs vs fork 现状 510µs；
- 步长 20.0 → 16.8ms ≈ **60 t/s，仍追不上 llama.cpp 的 68.4**。

**Triton matvec 路线数学上闭合不了单路差距。** llama.cpp 的 294 GB/s 证明空间存在，
钥匙在 HIP / 显式布局协同设计（寄存器内顺带解包、权重流式布局），数周级工程。

## 集成实验（fp8kv_w4patch.py，已回退）

op 级拦截 `torch.library.impl("radiance::mxfp4_linear", "CUDA")`：
CustomOpDef 没有 `.fn` 属性、模块属性是定义对象本身（直接调 = 递归爆栈），正确姿势是先偷 `_backend_fns["cuda"]` 存原函数。
（编译图内 scheme.apply_weights / kernel 类的 apply_weights 都不被 Python 调用，**只有 op 级拦截有效**。）
拦截本身成功（ENGAGED 日志确认），但净效果 47.2 < 49.9 t/s——matvec 不比被替换者快，回退。
`FP8KV_W4MV=1` 可重启实验。

---

> **09-14 后记**：墙只属于 Triton。同日用 ~150 行 HIP demo 内核实测 **397-400 GB/s 真实 DRAM**（warp-per-row + LDS 字节-LUT + HMUL2 折 scale + dot2），超 llama.cpp 的 294。demo 代码与生产化路线见 {doc}`06-w4gemm-hip-roadmap`。论文级的教训：gfx1201 的 hipcc 默认编 wave32，`__shfl_xor(…,64)` 会静默丢一半部分和。
