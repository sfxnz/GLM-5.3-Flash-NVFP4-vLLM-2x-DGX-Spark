#!/usr/bin/env python3
"""prefill_table.py --arm NAME FILE [FILE ...] [--arm NAME FILE ...]: cell E per panel and per arm.

For each bench.json that has cell E (bench_decode.py summary rows E@32k / E@128k) prints server prefill tok/s,
TTFT, decode tok/s, acceptance and step_ms, then the arm means and every later arm's change against the first arm
(prefill tok/s, TTFT) and against each other arm. kit/compare.py handles the decode verdicts; it does not read
prefill tok/s. Panels are the unit here (cell E runs one request per panel). Stdlib only.

Also reports the cell's untimed ~7k-token warmup request as E@7kwarm (bench_decode.py keeps it in cells.E.warmup).
A warmup under 900 tok/s had a JIT compile in it (first E after a cold-cache boot) and is left out of the means.
"""
import json
import statistics
import sys


def rows(path):
    b = json.load(open(path))
    out = {}
    for s in b["summary"]:
        if s["group"] in ("E@32k", "E@128k"):
            g = lambda k: (s.get(k) or {}).get("mean")  # noqa: E731
            out[s["group"]] = dict(prefill=g("server_prefill_tok_s"), ttft=g("ttft_s"), tok_s=g("tok_s"),
                                   acc=g("acceptance_len"), step=g("step_ms"), prompt=g("prompt_tokens"))
    w = b["cells"]["E"]["warmup"]
    if w.get("server_prefill_tok_s"):
        r = w["requests"][0]
        out["E@7kwarm"] = dict(prefill=w["server_prefill_tok_s"], ttft=r["ttft_s"], tok_s=w["tok_s"],
                               acc=w["acceptance_len"], step=w["step_ms"], prompt=r["prompt_tokens"])
    return out


arms, cur = {}, None
for a in sys.argv[1:]:
    if a == "--arm":
        cur = None
        continue
    if cur is None:
        cur = a
        arms[cur] = []
        continue
    arms[cur].append(a)

print(f"{'arm':6} {'panel':58} {'cell':7} {'prefill':>8} {'TTFT s':>8} {'tok/s':>7} {'acc':>6} {'step':>6}")
means = {}
for name, files in arms.items():
    per = {"E@7kwarm": [], "E@32k": [], "E@128k": []}
    for f in files:
        for cell, r in rows(f).items():
            if cell != "E@7kwarm" or r["prefill"] >= 900:
                per[cell].append(r)
            print(f"{name:6} {f[-58:]:58} {cell:7} {r['prefill']:8.1f} {r['ttft']:8.2f} {r['tok_s']:7.2f} "
                  f"{r['acc']:6.3f} {r['step']:6.1f}")
    means[name] = {c: {k: statistics.mean(r[k] for r in v) for k in ("prefill", "ttft", "tok_s", "acc", "step")}
                   for c, v in per.items() if v}
    for c, m in means[name].items():
        n = len(per[c])
        sd = statistics.stdev(r["prefill"] for r in per[c]) if n > 1 else float("nan")
        print(f"{name:6} {'mean of ' + str(n) + ' panel(s)':58} {c:7} {m['prefill']:8.1f} {m['ttft']:8.2f} "
              f"{m['tok_s']:7.2f} {m['acc']:6.3f} {m['step']:6.1f}   prefill sd {sd:.1f}")
names = list(means)
print()
for i, a in enumerate(names):
    for b in names[i + 1:]:
        for c in ("E@7kwarm", "E@32k", "E@128k"):
            if c in means[a] and c in means[b]:
                pa, pb = means[a][c]["prefill"], means[b][c]["prefill"]
                ta, tb = means[a][c]["ttft"], means[b][c]["ttft"]
                print(f"{b} vs {a} {c:7} prefill {pa:7.1f} -> {pb:7.1f} ({(pb / pa - 1) * 100:+.1f}%), "
                      f"us/token {1e6 / pa:6.1f} -> {1e6 / pb:6.1f} ({1e6 / pb - 1e6 / pa:+.1f}); "
                      f"TTFT {ta:7.2f} -> {tb:7.2f} s ({(tb / ta - 1) * 100:+.1f}%)")
