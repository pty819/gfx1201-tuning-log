import json, time, threading, urllib.request

URL = "http://localhost:8000/v1/chat/completions"
_counter = [0]

PARAS = [
    "Machine translation systems have improved dramatically in recent years thanks to large language models trained on massive parallel corpora. However real-world documents often contain domain-specific terminology and idiomatic expressions that still challenge even the strongest models. Careful evaluation across genres from legal contracts to casual social media posts remains essential for measuring true progress in the field.",
    "The graphics card market has shifted toward dedicated AI accelerators that support low precision arithmetic such as FP8 and MXFP4. Consumer boards with twelve gigabytes of memory can now serve seven billion parameter models comfortably when weights and key value caches are both quantized. Quantization aware calibration preserves translation quality while cutting memory bandwidth requirements roughly in half.",
    "Distributed inference across multiple devices introduces subtle scheduling problems. Chunked prefill interleaves long prompt processing with ongoing token generation so that interactive users do not starve behind bulk jobs. The scheduler must balance fairness against throughput and respect the finite key value cache pool shared by all concurrent sequences in the engine.",
    "Open source translation models released this year cover thirty three languages including several minority languages. They follow natural language instructions that specify target language tone and formatting constraints. On standard benchmarks they outperform much larger general purpose models in fast thinking mode while running an order of magnitude faster on commodity hardware.",
]

def build_prompt(worker, reps, nonce):
    # unique opening line => prefix cache can never hit across phases/workers
    marker = f"Document reference {nonce}-{worker}. All rights reserved.\n\n"
    paras = PARAS[worker % 4:] + PARAS[:worker % 4]
    doc = " ".join(p + "\n\n" for p in (paras * 40)[:reps])
    return "Translate the following document into Chinese, without additional explanation.\n\n" + marker + doc

def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)

def nonstream(prompt):
    t0 = time.time()
    u = json.load(post({"model": "hy-mt2-7b", "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1}))["usage"]
    return time.time() - t0, u["prompt_tokens"]

def stream(prompt, max_tokens=512):
    t0 = time.time(); ttft = None; usage = None
    r = post({"model": "hy-mt2-7b", "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "stream": True,
              "stream_options": {"include_usage": True}})
    for line in r:
        line = line.strip()
        if not line.startswith(b"data: "): continue
        p = line[6:]
        if p == b"[DONE]": break
        o = json.loads(p)
        if o.get("usage"): usage = o["usage"]
        ch = o.get("choices") or []
        if ch and ch[0].get("delta", {}).get("content") and ttft is None:
            ttft = time.time() - t0
    total = time.time() - t0
    return ttft, total, (usage or {}).get("prompt_tokens"), (usage or {}).get("completion_tokens")

def nonce():
    _counter[0] += 1
    return f"N{_counter[0]:03d}"

REPS = 78  # ~4.4k tokens

print("== A: single ~7k-token prompt, max_tokens=1 (multi-chunk prefill TTFT) ==")
n = nonce()
t, pt = nonstream(build_prompt(0, 124, n))
print(f"nonce={n} pt={pt}  time={t:.2f}s  -> {pt/t:.0f} tok/s prefill")

print("\n== B: 4x concurrent ~4.4k-token prefill (cold, distinct) ==")
n = nonce()
res = {}
def w(i): res[i] = nonstream(build_prompt(i, REPS, f"{n}-{i}"))
th = [threading.Thread(target=w, args=(i,)) for i in range(4)]
t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
wall = time.time() - t0
for i in range(4):
    t, pt = res[i]; print(f"  w{i}: pt={pt} time={t:.2f}s  ({pt/t:.0f} tok/s effective)")
print(f"  wall={wall:.2f}s  aggregate prefill = {sum(r[1] for r in res.values())/wall:.0f} tok/s")

print("\n== C: 4 concurrent long convos ~4.4k in + 512 out (cold prefixes) ==")
n = nonce()
res = {}
def w2(i): res[i] = stream(build_prompt(i, REPS, f"{n}-{i}"))
th = [threading.Thread(target=w2, args=(i,)) for i in range(4)]
t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
wall = time.time() - t0
agg = 0
for i in range(4):
    ttft, total, pt, ct = res[i]
    d = ct/(total-ttft); agg += d
    print(f"  w{i}: pt={pt} ct={ct} ttft={ttft:.2f}s total={total:.2f}s decode={d:.1f} tok/s")
print(f"  wall={wall:.2f}s  per-stream={agg/4:.1f} tok/s  aggregate decode={agg:.0f} tok/s")
print(f"  phase decode-only tail: last-unfinished stream finished at {wall:.2f}s")
