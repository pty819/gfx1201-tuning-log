# 05 · 运行手册与未竟之路

## 现行服务拓扑（2026-09-15）

**常驻 = vLLM 最优配置**：`http://<host>:8080/v1`，双别名 `Hy-MT2-7B` + `hy-mt2-7b`。

```bash
bash ~/fp8kv-dev/serve-h4mv.sh          # = h4mv/serve-h4mv.sh（09-14 起含 h4mv HIP W4 GEMM）
```

要点：`--attention-backend TRITON_ATTN --kv-cache-dtype fp8 --port 8080`，
`-v ~/fp8kv-dev:/opt/fp8kv -e PYTHONPATH=/opt/fp8kv -e FP8KV_HIPW4=1 -e FP8KV_PREFILL=1`，
全部 RADIANCE env，
`--max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.92 --enable-prefix-caching`。

生效自检（`llamacpp/switch_to_vllm.sh` 已自动做）：

```bash
podman logs hy-mt2-vllm 2>&1 | grep -E "KV cache size|128/128|fp8kv-prefill" | tail -6
# 预期 KV 池 ~80,960 tokens + 128/128 W4A8 快车道
# 以及 [fp8kv-prefill] installed / ENGAGED
```

**备用 = llama.cpp Vulkan**（KV 池更大 96k vs 81k；单路 decode **不再**更快，68.4 vs 80.1）：

```bash
pkill -f llama-[s]erver; podman rm -f hy-mt2-vllm; bash /tmp/vk_launch.sh 98304
```

## 负载选型表

| 负载形态 | 选择 | 数字依据 |
|---|---|---|
| prefill 重（长文档进出） | **vLLM + fp8kv_prefill** | e2e ~6k tok/s（09-15，~2050 tok / 345 ms） |
| decode（任意并发） | **vLLM + h4mv** | 单路 80.1 / 4 路 199.9 / 8 路 223.9（09-14） |
| 要最大 KV 池 | llama.cpp 96k 或 vLLM fp8 81k | bf16 KV 只有 33.8k |
| 重复结构文本（列表/代码/改写） | llama.cpp + ngram 投机 | 2.9×，无损 |

## 基准工具

- `bench/bench1.py <base> <model> <conc>`——通用并发基准（P/T 两相、ignore_eos、nonce 防前缀缓存）；
- `bench/bench_e2e.py`——prefill TTFT（uuid 防缓存，`max_tokens=1`）+ 短 decode 冒烟；
- `bench/bench_tr2.py`——真实翻译 4 段不重复 + **md5 输出等价校验**；
- 微基准 `vllm-fp8kv/prefill_test.py`、`profiling/mb_stock.py`——注意力 / 带宽；L2 假象必须多缓冲轮转。

测量纪律：bench 前停掉另一套服务（显存撞车会以 "failed to load model" 伪装成 GGUF 损坏）；
GPU 功耗档锁 high；同口径对比（不同池深/测法的历史数字不可直接横比）。

## 复盘：这条路线图上每个岔口的判定

1. **fp8 KV 病态（decode）** → 不是"fp8 在 AMD 上不行"，是 stock 的 fp8→f32 慢转换；自写 decode kernel 后 fp8 KV 可用。prefill 注意力慢是另一件事，见 {doc}`07-prefill`。
2. **单路差距** → 09-14 上午钉在 W4 matvec 带宽；Triton 写不出来。当晚 h4mv HIP 落地后 **80.1 > llama.cpp 68.4**，这条差距已闭合。
3. **投机解码** → ngram 对真实翻译零收益（2.9× 只存在于重复文本）。
4. **多路扩展** → 交叉点 3-4 路是 h4mv 之前的口径；现行 1 路已经用 vLLM。
5. **引擎胶水税** → h4mv 之前墙钟−GPU busy ≈ 3.9ms（49.9 t/s 配置）。80.1 配置上同一 profiler 只剩 **0.4ms**。

## 未竟之路（按性价比排序）

1. **给 fork 作者提 issue 要 h128/gqa4 的 r4d 编译变体**——prefill/decode 注意力都还在自写 Triton 上。
2. prefill tile 按 `(qlen, ctx)` autotune；多序列 / cascade / SWA 仍走 stock。
3. **requant 现有 AWQ → MXFP4**（零内核工作量，骑现成 W4A8 快车道）。
4. fp8 KV kernel 未覆盖：per-token-head scale、投机解码路径、head_dim≠128。
5. 手写 HIP prefill：正确但 VGPR=167 占用=1，2323 µs 打不过 Triton 1680。要继续先压寄存器。

## 环境速查（宿主 82）

- 容器：`hy-mt2-vllm`（podman，`--network host` 必须）；日志 `podman logs hy-mt2-vllm`；
- llama.cpp：`~/llama.cpp/build-amd/bin/llama-server`，env wrapper `~/llama-amd.sh`；
- 开发目录：`~/fp8kv-dev/`（本仓 `vllm-fp8kv/` + `h4mv/` + `profiling/` 的原件；`ref/` 未入仓）；
- 模型：`~/models/hy-mt2-7b-awq2-mxfp4`（4.66GB）+ `~/models/Hy-MT2-7B-Q4_K_M.gguf`；BF16 源已删，重新量化需重新下载；
- GPU 功耗档：`/sys/class/drm/card1/device/power_dpm_force_performance_level`（high，恢复改 auto）。
