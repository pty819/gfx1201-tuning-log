#!/bin/bash
pkill -f llama-[s]erver; sleep 2
podman rm -f hy-mt2-vllm >/dev/null 2>&1
TB="TRITON_ATTN"
sed -e "s|--network host|-v /home/liyifan/fp8kv-dev:/opt/fp8kv -e PYTHONPATH=/opt/fp8kv -e FP8KV_PROF_START=180 -e FP8KV_PROF_STEPS=60 -e FP8KV_PROF_OUT=/opt/fp8kv/prof_engine.txt --network host|" \
    -e "s|--served-model-name hy-mt2-7b|--served-model-name hy-mt2-7b --attention-backend $TB|" \
    ~/serve-hy-mt2-vllm.sh > /tmp/serve-prof.sh
bash /tmp/serve-prof.sh >/dev/null 2>&1 && echo launched
for i in $(seq 1 70); do
  sleep 5
  podman logs --since 7m hy-mt2-vllm 2>&1 | grep -q "startup complete" && { echo READY; break; }
done
rm -f ~/fp8kv-dev/prof_engine.txt
python3 /tmp/bench_tr.py http://127.0.0.1:8000 hy-mt2-7b >/dev/null 2>&1 &  # warmup decode ~1150 steps? 384 tokens*3 rounds
sleep 30
python3 /tmp/bench1.py http://127.0.0.1:8000 hy-mt2-7b 1 >/dev/null 2>&1
sleep 5
ls -la ~/fp8kv-dev/prof_engine.txt 2>/dev/null || { echo NO-FILE; podman logs hy-mt2-vllm 2>&1 | grep profhook | tail -4; }
