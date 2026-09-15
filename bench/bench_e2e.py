# -*- coding: utf-8 -*-
"""e2e prefill/decode bench against local vLLM OpenAI API."""
import json, time, uuid, urllib.request

URL = "http://127.0.0.1:8080/v1/completions"
MODEL = "Hy-MT2-7B"
UNIT = "The quick brown fox jumps over the lazy dog. 数九寒天翻译测试句。"


def post(payload, timeout=180):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return time.perf_counter() - t0, body


def make_prompt(n_units, tag):
    return f"[{tag} {uuid.uuid4()}] " + (UNIT * n_units)


def calib(target):
    lo, hi = 1, 400
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        dt, body = post(
            {
                "model": MODEL,
                "prompt": make_prompt(mid, f"cal{target}"),
                "max_tokens": 1,
                "temperature": 0,
            }
        )
        n = body["usage"]["prompt_tokens"]
        print(f"  calib units={mid} -> {n} tok  {dt*1000:.0f}ms", flush=True)
        best = mid
        if n < target:
            lo = mid + 1
        elif n > target + 32:
            hi = mid - 1
        else:
            return mid, n
    dt, body = post(
        {
            "model": MODEL,
            "prompt": make_prompt(best, f"cal{target}f"),
            "max_tokens": 1,
            "temperature": 0,
        }
    )
    return best, body["usage"]["prompt_tokens"]


def bench_prefill(units, rounds=5):
    times, toks = [], []
    for i in range(rounds):
        dt, body = post(
            {
                "model": MODEL,
                "prompt": make_prompt(units, f"pf{i}"),
                "max_tokens": 1,
                "temperature": 0,
            }
        )
        n = body["usage"]["prompt_tokens"]
        times.append(dt)
        toks.append(n)
        print(f"  prefill[{i}] {n} tok  {dt*1000:.1f} ms  {n/dt:.0f} tok/s", flush=True)
    times.sort()
    mid = times[len(times) // 2]
    n = toks[-1]
    return n, mid, n / mid


def bench_decode(units, gen=64, rounds=3):
    out = []
    for i in range(rounds):
        dt, body = post(
            {
                "model": MODEL,
                "prompt": make_prompt(units, f"dc{i}"),
                "max_tokens": gen,
                "temperature": 0,
                "ignore_eos": True,
            }
        )
        u = body["usage"]
        ct = u["completion_tokens"]
        # crude: remaining time after a same-size prefill is unknown; report e2e
        print(
            f"  decode[{i}] prompt={u['prompt_tokens']} gen={ct}  "
            f"wall={dt*1000:.0f}ms  e2e_gen={ct/dt:.1f} tok/s",
            flush=True,
        )
        out.append((u["prompt_tokens"], ct, dt))
    return out


if __name__ == "__main__":
    print("warmup", flush=True)
    post({"model": MODEL, "prompt": make_prompt(4, "wu"), "max_tokens": 1, "temperature": 0})
    print("calibrate ~850", flush=True)
    u850, n850 = calib(850)
    print("calibrate ~2048", flush=True)
    u2048, n2048 = calib(2048)
    print(f"\n=== prefill ~850 (units={u850}, got {n850}) ===", flush=True)
    n, mid, tps = bench_prefill(u850)
    print(f"MEDIAN {n} tok  {mid*1000:.1f} ms  {tps:.0f} tok/s  ({mid*1000/32:.1f} ms/layer if 32L)", flush=True)
    print(f"\n=== prefill ~2048 (units={u2048}, got {n2048}) ===", flush=True)
    n, mid, tps = bench_prefill(u2048)
    print(f"MEDIAN {n} tok  {mid*1000:.1f} ms  {tps:.0f} tok/s  ({mid*1000/32:.1f} ms/layer if 32L)", flush=True)
    print("\n=== decode 64 from short prompt ===", flush=True)
    bench_decode(8, gen=64, rounds=3)
    print("done", flush=True)
