#!/usr/bin/env python3
"""Tier-1 per-task table from the per-item JSONL (last row per item wins, like tier1.summarize)."""
import collections
import json
import statistics
import sys

path = sys.argv[1]
rows = [json.loads(x) for x in open(path) if x.strip()]
meta = next((r["meta"] for r in rows if "meta" in r), {})
last = {}
for r in rows:
    if "task" in r:
        last[(r["task"], str(r["id"]))] = r
order = ["ifeval", "gsm8k", "mmlu_pro", "bfcl", "chartqa", "ocrbench", "mmmu"]
group = {"ifeval": "IFEval", "gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro", "bfcl": "BFCL",
         "chartqa": "Vision", "ocrbench": "Vision", "mmmu": "Vision"}
print(f"meta: {json.dumps(meta)}")
print(f"{'task':<10}{'n':>5}{'correct':>8}{'acc':>8}{'errors':>7}{'length':>7}{'pred None':>10}{'med tok':>8}{'med s':>7}")
tot = collections.Counter()
by_group = collections.defaultdict(lambda: [0, 0])
for t in order:
    rs = [r for (task, _), r in last.items() if task == t]
    if not rs:
        continue
    n, ok = len(rs), sum(r["correct"] for r in rs)
    err = sum("error" in r for r in rs)
    length = sum(r.get("finish_reason") == "length" for r in rs)
    nopred = sum(r.get("pred") is None for r in rs if "error" not in r)
    toks = [r["completion_tokens"] for r in rs if r.get("completion_tokens")]
    secs = [r["s"] for r in rs if r.get("s")]
    print(f"{t:<10}{n:>5}{ok:>8}{ok / n:>8.3f}{err:>7}{length:>7}{nopred:>10}"
          f"{statistics.median(toks) if toks else 0:>8.0f}{statistics.median(secs) if secs else 0:>7.1f}")
    tot["n"] += n
    tot["ok"] += ok
    tot["err"] += err
    by_group[group[t]][0] += n
    by_group[group[t]][1] += ok
print(f"{'pooled':<10}{tot['n']:>5}{tot['ok']:>8}{tot['ok'] / max(1, tot['n']):>8.3f}{tot['err']:>7}")
print("groups: " + ", ".join(f"{g} {ok}/{n} = {ok / n:.3f}" for g, (n, ok) in by_group.items()))
