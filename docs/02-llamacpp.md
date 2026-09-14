---
layout: default
title: "02 · llama.cpp 基线与调优"
---

# 02 · llama.cpp 基线与调优

llama.cpp 是这场战役的"对照组"，也是单路 decode 的标杆（68.4 t/s @ 4k ctx，至今未被 vLLM 侧追平）。

## 后端选择：Vulkan 为主

- 同一二进制下 `--device Vulkan0` vs `ROCm0`：09-04 实测 tg128 **Vulkan 77.6 vs HIP 74.4**（RADV GFX1201 coopmat）；09-13 三家对比 decode 聚合 **Vulkan 83 > ROCm 74** t/s。RDNA4 上 Vulkan 对 7B decode 稳定快 4-10%。
- `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json`——注意不是 `radeon_icd.x86_64.json`。
- `HIP_VISIBLE_DEVICES=0` 避开 9700X 核显 gfx1036（给它分层必失败）。
- 环境包装见 `llamacpp/llama-amd.sh` / `llamacpp/run-amd.sh`（后者 exec 任意二进制）。

## 值守配置（09-13 起的形态）

```bash
~/llama-amd.sh -m ~/models/Hy-MT2-7B-Q4_K_M.gguf -a Hy-MT2-7B \
  -c 98304 -np 8 --kv-unified --flash-attn on \
  -ctk q8_0 -ctv q8_0 --kv-offload --repack --op-offload \
  -ngl 99 --device Vulkan0 --host 0.0.0.0 --port 8080
```

- **KV 池 = 98,304 tokens**（8 slot 共享，VRAM 10.94/12GiB；96k 是安全上限，102k 只多 8% 但运行时有 OOM 风险）。
- **q8 KV + FA**：实测无质量退化（md5 等价验证），是 llama.cpp 侧 KV 减半的标准解。
- 这版二进制 `--kv-offload`/`--repack`/`--op-offload` 默认已开，显式传参保险。
- 一键脚本 `llamacpp/vk_launch.sh <ctx>`（含就绪探测与 OOM 检测），可挂额外参数的变体 `vk_launch2.sh`。

## 并发特性（09-13 bench1.py 实测，4k ctx）

| 并发 | Vulkan (q8 KV) | 备注 |
|---|---|---|
| 1 路 | **68.4 t/s** | 单路标杆 |
| 4 路 | 125 t/s 聚合 | |
| 8 路 | 136 t/s 聚合 | |

（09-08 旧配置 -c 32000 曾测 4 路 180 / 8 路 210——不同池深与测量口径，说明池配置对并发吞吐影响很大。）

- prefill 单流 205→3086 tok 全程 ~2900-3050 tok/s 平坦；prefix cache 重放只重算 1 token。
- **真正的坑：unified 池被前一波请求填满后，新一波首个请求吃 ~2.4s 池维护停顿**（与 prompt 长度无关）——并发墙钟抖动来自这里，不是 prefill 慢。

## ngram 投机解码（负结论，09-13 实测）

`--spec-type ngram-simple`：

- 同段重复 8× 的翻译：71 → **210 t/s（2.9×）**——输出历史押中草稿；
- 4 段互不相同的真实翻译：72.7 vs 无投机 72.9——**零收益**（md5 输出等价验证通过，投机本身无损）；
- 单调计数任务：零收益（序列永不重复）。

结论：翻译类负载**不开**；列表/代码/改写类重复结构任务可临时开。

## 选型口径

- ≤3 路 decode：本配置（单路 68 t/s 领先 vLLM 37%）
- ≥4 路：换 vLLM（`llamacpp/switch_to_vllm.sh`）
- prefill 重：永远 vLLM（~5.1k vs ~2.8k tok/s）

切换前必须 `pkill llama-server`，否则 VRAM 撞车 init fail。
