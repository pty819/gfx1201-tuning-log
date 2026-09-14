#!/bin/bash
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video | cut -d: -f3)
podman rm -f hy-mt2-vllm 2>/dev/null
podman run -d --name hy-mt2-vllm \
  --device /dev/kfd --device /dev/dri \
  --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
  --shm-size 4g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /home/liyifan/models:/models:ro \
  -v /home/liyifan/vllm-cache:/cache \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1 \
  -e WEIGHT_QUANTIZATION=auto \
  -e RADIANCE_MXFP4=1 \
  -e RADIANCE_MXFP4_W4A8=1 \
  -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
  -e RADIANCE_MXFP4_DECODE_MAX_M=64 \
  -v /home/liyifan/fp8kv-dev:/opt/fp8kv -e PYTHONPATH=/opt/fp8kv -e FP8KV_HIPW4=1 --network host \
  docker.io/magiccodingman/vllm-radiance:latest \
  /models/hy-mt2-7b-awq2-mxfp4 \
  --served-model-name Hy-MT2-7B hy-mt2-7b --attention-backend TRITON_ATTN \
  --port 8080 --kv-cache-dtype fp8 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching
