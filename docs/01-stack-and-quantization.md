# 01 · 栈与量化

## 三套推理栈并存

同一台机器上长期并存三套可用栈，各司其职：

1. **llama.cpp 一体树**（`~/llama.cpp/build-amd`，09-12 更新至 790cf51）：HIP(ROCm10) + ZenDNN + Vulkan 三后端一个二进制，`--device` 切换。构建命令：

```bash
source ~/rocm10/bin/rocm-env.sh
cmake -S ~/llama.cpp -B ~/llama.cpp/build-amd \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGPU_TARGETS=gfx1201 \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_NATIVE=ON \
  -DGGML_ZENDNN=ON -DZENDNN_ROOT=$HOME/ZenDNN/build/install \
  -DGGML_VULKAN=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14
```

   - ZenDNN（WW28 tag，gcc-14 编）只加速 CPU 侧 MUL_MAT（FP32/BF16/Q8_0），是 CPU offload / 大模型溢出层用的；7B 全量上卡时几乎不干活。
   - 09-12 拉新的 166 commits 里重点是 `16378d9` Flash Attention tuning **专 gfx1201**、Vulkan FA 反量化路径修复、Q4_K/Q5_K 无分支解包——tg128 从 74.4 提到 75.1。
   - `GGML_CUDA_FORCE_MMQ` 在 gfx1201 上是负优化别开；`FA_ALL_QUANTS` 旧记录说负优，09-12 实测 fa=1 无回退且 exotic KV 量化组合也能吃 FA。
   - 裸跑 llama-server 会 dlopen 失败 `libhipblas.so.3`：ROCm10 是 pip venv，运行库在 site-packages 深处，必须走 `llamacpp/llama-amd.sh` wrapper。

2. **vLLM（vllm-radiance 容器）**：`magiccodingman/vllm-radiance:latest`，Docker 专打 gfx1201。快车道 = FP8 GEMM + Quark-MXFP4 W4A8（`radiance_mxfp4_fp8.so`，RDNA4 原生 FP8 WMMA，手写 325 TF vs 通用 Triton 43 TF）。AWQ/GPTQ/AutoRound 普通格式全走通用慢车道。源码仓私有（gitlab.sayou.io），只能从 image 自省。

3. **（历史）FP8 权重版**：7.89GiB，4 并发聚合 100 tok/s——被 MXFP4 全面替代后按用户要求删除。

## 量化：Quark-AWQ-MXFP4 流水线（自制 checkpoint）

**为什么是它**：12GB 显存装 7B 的最优解是 W4A8——fork 的 W4A8 GEMM 是几何无关快车道，任何模型都能吃。Quark（AMD 官方，amd-quark 0.12.post1 镜像，pip 版缺 quark.shares 弃用）出 `w_mxfp4_a_mxfp4` scheme。

**配方**（校准在 CPU 上跑，~406-500s/层 × 32 层 ≈ 3.6-4.5h）：

```
--quant_scheme w_mxfp4_a_mxfp4 --group_size 32 --quant_algo awq
--num_calib_data 24 --seq_len 1024 --device cpu
```

- **v2 校准语料 = OPUS-100 en-zh validation 1936 对 × 双向**，官方翻译指令模板。v1 用合成语料，出重复词退化（"7B model models"），v2 修复（→"7B model sizes"）——**校准分布必须贴近真实负载**。
- 对 Hunyuan 类 dense 层打了补丁：model_preparation 加 `"HunYuanDense": "hunyuan"` 映射、注册 hunyuan_v1_dense 模板 + llama 同款 AWQConfig、data_preparation 加 `local:` jsonl 分支、去 llm_eval 依赖。

**三大坑**（都踩过）：

1. 校准样本必须**等长**（拼流定长切块，否则 AWV 检查重放 kwargs 时 RoPE 维度崩）；
2. **OOM**：校准块 ≤32、`MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=134217728`、量化期间停 vllm 容器 + drop_caches（页缓存挤压会引来 oom-kill，且 `journalctl -k` 只 grep oom 会漏）；24×1024 ✅ / 32×1024 ✗，OOM 线在中间；
3. 量化中后台下载 15GB 会挤压内存致死。

**Qronos 负结论**（arXiv 2505.11695，Hessian 类校准算法）：算法本体在 CPU 上跑通了（补 numpy Cholesky shim——镜像 torch 没编 LAPACK），但 O(L²) 双遍收集 ~88s/样本 × 128 块结构性不可行。GPU 才是它的主场（论文报 9-20% 校准开销）。放弃。

**架构认知**（实测确认）：quark loader 的 scheme 仅 NVFP4 / OCP-MX / W4A8-MXFP4-FP8 / W8A8-FP8 / W8A8-INT8——**INT4-W4A16 无加载路径**，这是普通 AWQ checkpoint 无法原生加速的根因（内核之外，缺的是 loader）。

## 生效验证

serve 启动日志必看三行：

```
quantization=quark
W4A8 fp8-WMMA GEMM ENABLED
linear layers: 128/128 on our kernel, 0 FORCED ONTO AITER
```

必带 env：`WEIGHT_QUANTIZATION=auto RADIANCE_MXFP4=1 RADIANCE_MXFP4_W4A8=1 RADIANCE_MXFP4_W4A8_MIN_M=0 RADIANCE_MXFP4_DECODE_MAX_M=64`。
容器网络必须 `--network host`（pasta 坑）。
