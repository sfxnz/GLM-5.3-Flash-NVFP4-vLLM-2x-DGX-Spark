#!/usr/bin/env python3
"""Pair two thinking-on subset runs (quality/tier1.py JSONL) item by item: accuracy, completion tokens and
reasoning length. Usage: think_compare.py A.jsonl B.jsonl"""
import json
import statistics
import sys


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("id") is not None and not r.get("error"):
            rows[(r.get("task"), r["id"])] = r
    return rows


a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) & set(b), key=str)
print(f"paired items {len(keys)} (A only {len(set(a) - set(b))}, B only {len(set(b) - set(a))})")
for task in sorted({k[0] for k in keys}):
    ks = [k for k in keys if k[0] == task]
    ca, cb = sum(bool(a[k].get("correct")) for k in ks), sum(bool(b[k].get("correct")) for k in ks)
    only_a = sum(bool(a[k].get("correct")) and not b[k].get("correct") for k in ks)
    only_b = sum(bool(b[k].get("correct")) and not a[k].get("correct") for k in ks)
    ta = [a[k]["completion_tokens"] for k in ks if a[k].get("completion_tokens")]
    tb = [b[k]["completion_tokens"] for k in ks if b[k].get("completion_tokens")]
    ratios = [b[k]["completion_tokens"] / a[k]["completion_tokens"] for k in ks
              if a[k].get("completion_tokens") and b[k].get("completion_tokens")]
    ra = [a[k].get("reasoning_chars", 0) for k in ks]
    rb = [b[k].get("reasoning_chars", 0) for k in ks]
    print(f"{task:9s} n={len(ks)} correct A={ca} B={cb} (A-only {only_a}, B-only {only_b}) | completion tokens mean "
          f"A={statistics.mean(ta):.0f} B={statistics.mean(tb):.0f}, median per-item B/A={statistics.median(ratios):.3f} | "
          f"reasoning chars mean A={statistics.mean(ra):.0f} B={statistics.mean(rb):.0f}")
