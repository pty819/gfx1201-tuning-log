# gfx1201 调优实录

一张 RDNA4 消费卡（AMD RX 9070 GRE 12GB, gfx1201）上，把 7B 翻译模型 Hy-MT2-7B 从 vLLM 单路 29.6 tok/s 推到现行 **单路 80.1 / 8 路 224 tok/s**，并把 prefill 端到端打到 **~6k tok/s**。

时间跨度 2026-09-02 → 2026-09-15。数字只认 {doc}`scoreboard`；每次测量按 {doc}`perf-log` 的固定条目写，不再往 README 里堆过期表。

```{toctree}
:maxdepth: 2
:caption: 目录

scoreboard
perf-log
01-stack-and-quantization
02-llamacpp
03-fp8kv-pathology-and-kernel
04-w4a8-matvec-wall
05-playbook
06-w4gemm-hip-roadmap
07-prefill
```

## 这张卡上现在跑什么

| 项 | 值 |
|---|---|
| GPU | RX 9070 GRE 12GB（gfx1201，无 `HSA_OVERRIDE`） |
| 宿主 | 192.168.1.82 CachyOS，功耗档 high |
| 服务 | `hy-mt2-vllm`，`:8080`，`magiccodingman/vllm-radiance:latest` |
| 模型 | Hy-MT2-7B Quark-AWQ-MXFP4 W4A8（h128 / GQA=4 / 32 层） |
| 快车道 | decode：自写 fp8-KV Triton + h4mv HIP W4；prefill：自写 fp8-KV Triton tile 64/32/2/2 |
| r4d 注意力 | **未启用**。r4d 0.5.0 只编了 `h256/gqa6`，本模型几何对不上就静默回退 |

仓库：<https://github.com/pty819/gfx1201-tuning-log>
