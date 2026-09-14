#!/bin/bash
pkill -f llama-[s]erver; sleep 2
podman rm -f hy-mt2-vllm >/dev/null 2>&1
grep -cE "attention-backend|PYTHONPATH" ~/fp8kv-dev/serve-fp8kv-best.sh
bash ~/fp8kv-dev/serve-fp8kv-best.sh >/dev/null 2>&1 && echo launched
for i in $(seq 1 70); do
  sleep 5
  podman logs --since 7m hy-mt2-vllm 2>&1 | grep -q "startup complete" && { echo READY; break; }
done
podman logs hy-mt2-vllm 2>&1 | grep -E "KV cache size|128/128" | tail -2
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"hy-mt2-7b","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' >/dev/null
sleep 2
podman logs hy-mt2-vllm 2>&1 | grep -cE "ENGAGED|installed"
