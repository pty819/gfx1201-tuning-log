#!/usr/bin/env python3
import sys, time, json, requests
BASE, MODEL = sys.argv[1].rstrip('/'), sys.argv[2]
S = requests.Session()
ZH = "这台服务器搭载 AMD RX 9070 GRE 显卡，显存为 12GB，配合 FP8 量化技术可以流畅运行 7B 参数的大语言模型。系统使用 Vulkan 后端进行推理，KV 缓存采用 q8_0 量化，上下文长度可达 96000 个 token，聚合解码速度约为每秒 136 个 token。"
for rnd in range(3):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": "把下面的文本翻译成英语，不要额外解释。\n\n" + ZH * 8}],
            "temperature": 0, "max_tokens": 384, "stream": True}
    t0 = time.time(); ttft = None; ntok = 0; tlast = t0; txt = []
    r = S.post(BASE + "/v1/chat/completions", json=body, timeout=600, stream=True)
    r.raise_for_status()
    for line in r.iter_lines():
        if not line: continue
        line = line.decode("utf-8", "ignore")
        if not line.startswith("data: ") or line[6:] == "[DONE]": continue
        try: d = json.loads(line[6:])
        except Exception: continue
        c = d.get("choices") or []
        if c and (c[0].get("delta") or {}).get("content"):
            ntok += 1; tlast = time.time(); txt.append(c[0]["delta"]["content"])
            if ttft is None: ttft = time.time()
    dec = tlast - ttft
    print(f"tr{rnd+1}: out={ntok} tg={(ntok-1)/dec if dec>0 else 0:.1f} t/s  head={''.join(txt[:12])!r}")
