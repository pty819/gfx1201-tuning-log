#!/usr/bin/env python3
import sys, time, json, hashlib, requests
BASE, MODEL = sys.argv[1].rstrip('/'), sys.argv[2]
S = requests.Session()
ZH = (
 "这台服务器搭载 AMD RX 9070 GRE 显卡，显存为 12GB，配合 FP8 量化技术可以流畅运行 7B 参数的大语言模型。",
 "量化校准数据集来自 OPUS-100 中英双语验证集，共一千九百三十六个句对，覆盖新闻、口语和科技三类文体，双向混合后随机采样。",
 "家里的宽带是千兆光纤，光猫改桥接模式，主路由负责拨号，两台子路由通过无线回程组网，客厅和书房的信号都满格。",
 "RK3588 开发板上跑着 ARM 版系统，Chromium 的视频硬解需要 VA-API 桥接层，社区维护的补丁已经支持 H.264 和 HEVC 的零拷贝路径。"
)
for rnd in range(3):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": "把下面的文本翻译成英语，不要额外解释。\n\n" + "\n".join(ZH)}],
            "temperature": 0, "max_tokens": 512, "stream": True}
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
    full = "".join(txt)
    print(f"tr{rnd+1}: out={ntok} tg={(ntok-1)/dec if dec>0 else 0:.1f} t/s md5={hashlib.md5(full.encode()).hexdigest()[:10]}")
