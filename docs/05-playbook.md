---
layout: default
title: "05 · 运行手册与未竟之路"
---

# 05 · 运行手册与未竟之路

## 现行服务拓扑（2026-09-14 收官态）

**常驻 = vLLM 最优配置**：`http://<host>:8080/v1`，双别名 `Hy-MT2-7B` + `hy-mt2-7b`。

```bash
bash ~/fp8kv-dev/serve-h4mv.sh          # = h4mv/serve-h4mv.sh（09-14 起含 h4mv HIP W4 GEMM）
```

要点：`--attention-backend TRITON_ATTN --kv-cache-dtype fp8 --port 8080`，
`-v ~/fp8kv-dev:/opt/fp8kv -e PYTHONPATH=/opt/fp8kv`（内核注入），全部 RADIANCE env，
`--max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.92 --enable-prefix-caching`。

生效自检（`llamacpp/switch_to_vllm.sh` 已自动做）：

```bash
podman logs hy-mt2-vllm 2>&1 | grep -E "KV cache size|128/128" | tail -2
# 预期 KV 池 ~80,960 tokens（9.88× @8k）+ 128/128 W4A8 快车道
```

**备用 = llama.cpp Vulkan**（单路 decode 更快 / 池更大）：

```bash
pkill -f llama-[s]erver; podman rm -f hy-mt2-vllm; bash /tmp/vk_launch.sh 98304
```

## 负载选型表

| 负载形态 | 选择 | 数字依据（4k ctx） |
|---|---|---|
| prefill 重（长文档进出） | vLLM | ~5.1k vs ~2.8k tok/s |
| decode（任意并发） | **vLLM + h4mv** | 单路 80.1 / 4 路 199.9 / 8 路 223.9 全面领先（09-14 起） |
| ≥4 路 decode | vLLM + fp8 KV kernel | 8 路 223.4 vs 136 t/s |
| 要最大 KV 池 | llama.cpp 96k 或 vLLM fp8 81k | bf16 KV 只有 33.8k |
| 重复结构文本（列表/代码/改写） | llama.cpp + ngram 投机 | 2.9×，无损 |

## 基准工具

- `bench/bench1.py <base> <model> <conc>`——通用并发基准（P/T 两相、ignore_eos、nonce 防前缀缓存）；
- `bench/bench_tr2.py`——真实翻译 4 段不重复 + **md5 输出等价校验**（golden md5 在无投机基线下采集）；
- 微基准 `profiling/mb_stock.py`、`profiling/micro_isolate.py`——**必须多缓冲轮转**（64MB L2 假象）。

测量纪律：bench 前停掉另一套服务（显存撞车会以 "failed to load model" 伪装成 GGUF 损坏）；
GPU 功耗档锁 high；同口径对比（不同池深/测法的历史数字不可直接横比）。

## 复盘：这条路线图上每个岔口的判定

1. **fp8 KV 病态** → 不是"fp8 在 AMD 上不行"，是 stock Triton 的 fp8→f32 慢转换；自写 kernel 后 fp8 是最优解。
2. **单路差距** → 已钉死在 W4 matvec 带宽（llama.cpp 294 GB/s vs 双方 ~180-205）；Triton 写不出来（vgpr/占空度结构性缺陷 + L3 上限算账不过关），唯 HIP。
3. **投机解码** → ngram 对真实翻译零收益（2.9× 只存在于重复文本）。
4. **多路扩展** → vLLM 每步边际成本 ~2.6ms/路 < llama.cpp ~5.8ms/路——高起步低边际 vs 低起步高边际，交叉点 3-4 路。
5. **引擎胶水税** → eager 29.5 vs graph 49.8 t/s；墙钟-GPU busy ≈ 3.9ms 调度间隙。想再快要么更深地吃掉这 3.9ms，要么换调度器。

## 未竟之路（按性价比排序）

1. **给 fork 作者提 issue 要 h128/gqa4 的 r4d 编译变体**——入口命名机制（`attn_decode_h256_gqa6_fp8kv`）天生支持多几何，
   对作者可能只是加一行编译目标；拿到即免费替换自研 Triton kernel（上限更高）。
2. **requant 现有 AWQ → MXFP4**（零内核工作量，骑现成 W4A8 快车道）——普通 AWQ checkpoint 的最省捷径。
3. **Triton v1 的 INT4-W4A16**（几天级，预期 2-3×）——但先要补 quark W4A16 的 loader scheme，这是真正的缺口。
4. **HIP 原生 W4 GEMM**（数周级）——唯一能闭合单路差距的路线；参考 `radiance_mxfp4_fp8.hip`（fork 仓库）+
   vLLM Marlin repack + r4d 现成 `gemm_w4a16_nt_m64`（`import r4d` 前需 `LD_LIBRARY_PATH=/opt/rocm/lib`）。
5. fp8 KV kernel 的未覆盖面：per-token-head scale、SWA/sinks/alibi、投机解码路径、head_dim≠128。

## 环境速查（宿主 82）

- 容器：`hy-mt2-vllm`（podman，`--network host` 必须）；日志 `podman logs hy-mt2-vllm`；
- llama.cpp：`~/llama.cpp/build-amd/bin/llama-server`，env wrapper `~/llama-amd.sh`；
- 开发目录：`~/fp8kv-dev/`（本仓 `vllm-fp8kv/` + `vllm-w4matvec/` + `profiling/` 的原件，含 `ref/` = fork 源码镜像拷贝，未入本仓）；
- 模型：`~/models/hy-mt2-7b-awq2-mxfp4`（4.66GB）+ `~/models/Hy-MT2-7B-Q4_K_M.gguf`；BF16 源已删，重新量化需重新下载；
- GPU 功耗档：`/sys/class/drm/card1/device/power_dpm_force_performance_level`（high，恢复改 auto）。
