# gfx1201 调优实录（RX 9070 GRE · vLLM / llama.cpp）

Hy-MT2-7B 在一张 gfx1201（12GB）上的推理调优记录。文档站是 Sphinx RTD 主题（左目录、右正文）：

**https://pty819.github.io/gfx1201-tuning-log/**

- 现行数字只认 [成绩单](https://pty819.github.io/gfx1201-tuning-log/scoreboard.html)
- 每次测量按固定字段写进 [实测日志](https://pty819.github.io/gfx1201-tuning-log/perf-log.html)，不再在 README 里堆过期表

本地预览：`pip install -r docs/requirements.txt && make -C docs html`，打开 `docs/_build/html/index.html`。

## 现行成绩（2026-09-15）

| | |
|---|---|
| decode 1/4/8 路（4k ctx） | **80.1 / 199.9 / 223.9** tok/s |
| prefill e2e ~2050 tok | **345 ms / 5940 tok/s** |
| prefill 隔离 kernel @2048 | **1680 µs/layer**（Triton tile 64/32/2/2） |
| KV 池 | 80,960 tokens |

服务：`bash h4mv/serve-h4mv.sh`（`FP8KV_HIPW4=1 FP8KV_PREFILL=1`）。几何对不上 r4d 注意力（只要 h256/gqa6）。

## 目录

```
docs/            Sphinx 源（成绩单 / 实测日志 / 各战役）
vllm-fp8kv/      decode + prefill Triton 内核、override、sitecustomize
h4mv/            HIP W4 GEMM 生产内核 + serve 脚本
vllm-w4matvec/   Triton matvec 负结论
w4-hip-demo/     HIP demo 397–400 GB/s
profiling/       引擎内 profiler 输出
bench/           并发基准、翻译 md5、e2e prefill
llamacpp/        Vulkan 备份
```
