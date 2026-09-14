#!/usr/bin/env python3
import sys, time, uuid, json, threading, requests
BASE, MODEL = sys.argv[1].rstrip('/'), sys.argv[2]
CONC, PT, NT, ROUNDS = 4, 4096, 256, 2
PT_USE = PT
S = requests.Session()
FILL = "The quick brown fox jumps over the lazy dog while the server processes tokens at a steady pace and the network forwards every packet without delay. "
INSTR = "Summarize the following passage in about 200 words. Do not refuse.\n\n"

def chat(prompt, max_tokens, stream=True):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": stream}
    r = S.post(BASE + "/v1/chat/completions", json=body, timeout=600, stream=stream)
    r.raise_for_status()
    if not stream:
        u = (r.json().get("usage") or {})
        return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0), 0.0, 0.0
    t0 = time.time(); ttft = None; ntok = 0; tlast = t0
    for line in r.iter_lines():
        if not line: continue
        line = line.decode("utf-8", "ignore")
        if not line.startswith("data: ") or line[6:] == "[DONE]": continue
        try: d = json.loads(line[6:])
        except Exception: continue
        c = d.get("choices") or []
        if c and (c[0].get("delta") or {}).get("content"):
            ntok += 1; tlast = time.time()
            if ttft is None: ttft = time.time()
    return PT_USE, ntok, (ttft - t0) if ttft else 0.0, tlast - t0

base_tok = chat("hello", 1, False)[0]
t_unit = chat("hello " + FILL, 1, False)[0]
unit = max(1, t_unit - base_tok)
k = max(1, round((PT - base_tok - 30) / unit))
ptok = chat(INSTR + FILL * k, 1, False)[0]
PT_USE = ptok
print(f"calib: base={base_tok} unit={unit} k={k} prompt_tokens={ptok}")

def run_round(tag, max_tokens):
    res = [None] * CONC
    def w(i):
        p = f"[{uuid.uuid4().hex[:8]}] " + INSTR + FILL * k
        res[i] = chat(p, max_tokens)
    th = [threading.Thread(target=w, args=(i,)) for i in range(CONC)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    ptoks = sum(r[0] for r in res); otoks = sum(r[1] for r in res)
    ttfts = [r[2] for r in res]; dec = [r[3] - r[2] for r in res]
    pp_agg = ptoks / wall
    tg_agg = (otoks / max(dec)) if max_tokens > 1 else 0
    per_tg = [((r[1] - 1) / (r[3] - r[2])) if (r[3] > r[2] and r[1] > 1) else 0 for r in res]
    per_pp = [(r[0] / r[2]) if r[2] > 0 else 0 for r in res]
    print(f"{tag}: wall={wall:.2f}s pp_agg={pp_agg:.0f} pp_stream={['%.0f'%x for x in per_pp]} "
          f"ttft_avg={sum(ttfts)/CONC:.2f}s out={otoks} tg_agg={tg_agg:.1f} tg_stream={['%.1f'%x for x in per_tg]}")

for r in range(ROUNDS):
    run_round(f"P{r+1}(pp)", 1)
for r in range(ROUNDS):
    run_round(f"T{r+1}(pp+tg)", NT)
