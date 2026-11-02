#!/usr/bin/env python3
"""Multi-turn prefix-cache probe against the running serve.

For each length, turn 1 sends a long document with a needle plus a question, and turn 2 sends the same
document (same prefix) with a second question, as a chat client does on its next turn. Reports TTFT per turn,
the serve's prefix-cache query and hit counters over each turn, whether both answers carry the needle, and
whether the serve is still up afterwards. Usage: prefix_probe.py [--lengths 8192,32768] [--out FILE]
"""
import argparse
import json
import random
import re
import time
import urllib.request

WORDS = ("amber basil cedar delta ember fjord garnet harbor indigo juniper kestrel lagoon meadow nectar orchid "
         "pebble quartz raven saffron tundra umber violet willow xenon yarrow zephyr").split()


def document(tokens: int, needle: str, seed: int) -> str:
    rng = random.Random(seed)
    lines, n = [], 0
    while n < tokens * 4:
        line = f"Line {len(lines) + 1}: " + " ".join(rng.choice(WORDS) for _ in range(12)) + "."
        lines.append(line)
        n += len(line) + 1
    lines.insert(len(lines) // 2, f"Line {len(lines) // 2}b: The vault code is {needle}.")
    return "\n".join(lines)


def metrics(url: str) -> dict:
    text = urllib.request.urlopen(f"{url}/metrics", timeout=30).read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(vllm:prefix_cache_(?:queries|hits)(?:_total)?)\{[^}]*\}\s+([0-9.e+]+)", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def chat(url: str, model: str, content: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 24,
            "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, ttft, text = time.perf_counter(), None, ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            text += piece
    return {"ttft_s": round(ttft, 3) if ttft else None, "total_s": round(time.perf_counter() - t0, 3), "text": text}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--lengths", default="8192,32768")
    ap.add_argument("--out")
    args = ap.parse_args()
    model = json.loads(urllib.request.urlopen(f"{args.url}/v1/models").read())["data"][0]["id"]
    rows = []
    for i, n in enumerate(int(x) for x in args.lengths.split(",")):
        needle = f"{random.Random(n).randint(100000, 999999)}"
        doc = document(n, needle, seed=n * 7 + int(time.time()))
        turns = []
        for q in ("What is the vault code? Reply with the number only.",
                  "Repeat the vault code from the document, digits only."):
            before = metrics(args.url)
            r = chat(args.url, model, f"{doc}\n\n{q}")
            after = metrics(args.url)
            delta = {k.split(":")[1]: after.get(k, 0) - before.get(k, 0) for k in after}
            turns.append({"ttft_s": r["ttft_s"], "total_s": r["total_s"], "needle_ok": needle in r["text"],
                          "answer": r["text"][:60], **delta})
        rows.append({"target_tokens": n, "turn1": turns[0], "turn2": turns[1]})
        print(json.dumps(rows[-1]), flush=True)
    alive = True
    try:
        urllib.request.urlopen(f"{args.url}/v1/models", timeout=30).read()
    except Exception:
        alive = False
    result = {"rows": rows, "serve_alive_after": alive}
    print(f"PREFIX alive={alive} " + " ".join(
        f"{r['target_tokens']}:ttft {r['turn1']['ttft_s']}->{r['turn2']['ttft_s']}s "
        f"hits {r['turn2'].get('prefix_cache_hits_total', r['turn2'].get('prefix_cache_hits'))}" for r in rows))
    if args.out:
        open(args.out, "w").write(json.dumps(result, indent=1) + "\n")
    return 0 if alive else 1


if __name__ == "__main__":
    raise SystemExit(main())
