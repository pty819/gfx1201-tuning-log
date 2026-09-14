#!/bin/bash
# $1 = ctx size
pkill -f llama-[s]erver; sleep 2
nohup ~/llama-amd.sh -m ~/models/Hy-MT2-7B-Q4_K_M.gguf -a Hy-MT2-7B \
  -c "$1" -np 8 --kv-unified --flash-attn on \
  -ctk q8_0 -ctv q8_0 --kv-offload --repack --op-offload \
  -ngl 99 --device Vulkan0 --host 0.0.0.0 --port 8080 > ~/bench-server.log 2>&1 &
for i in $(seq 1 40); do
  sleep 3
  if curl -s -o /dev/null http://127.0.0.1:8080/v1/models; then echo "READY ctx=$1"; exit 0; fi
  if grep -qiE "out of memory|failed to allocate|error" ~/bench-server.log; then echo "FAILED ctx=$1"; exit 1; fi
done
echo TIMEOUT; exit 1
