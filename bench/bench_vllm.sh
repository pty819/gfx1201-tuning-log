#!/bin/bash
pkill -f llama-[s]erver; sleep 2
podman start hy-mt2-vllm && echo started
for i in $(seq 1 60); do
  sleep 5
  if podman logs --since 6m hy-mt2-vllm 2>&1 | grep -q "startup complete"; then echo VLLM-READY; break; fi
done
python3 /tmp/bench4.py http://127.0.0.1:8000 hy-mt2-7b 2>&1 | tail -6
