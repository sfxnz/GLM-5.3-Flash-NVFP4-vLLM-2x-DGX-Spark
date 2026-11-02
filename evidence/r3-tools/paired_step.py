#!/usr/bin/env python3
"""Paired per-prompt step_ms of cell A between two boots: mean difference B - A with a t interval (df = n - 1),
plus the same for acceptance length. Waves pair by position: bench_decode.py sends cell A's prompts in a fixed order. Usage: paired_step.py A/bench-fast/bench.json B/bench-fast/bench.json [cell]"""
import json
import statistics
import sys

T95 = {7: 2.365, 6: 2.447, 5: 2.571, 3: 3.182, 2: 4.303}


def waves(path, cell):
    d = json.load(open(path))
    return [w for w in d["waves"] if w["group"] == cell and w.get("step_ms") is not None]


def paired(a, b, key):
    d = [y[key] - x[key] for x, y in zip(a, b)]
    m = statistics.mean(d)
    h = T95[len(d) - 1] * statistics.stdev(d) / len(d) ** 0.5
    return m, m - h, m + h, sum(x < 0 for x in d), len(d)


cell = sys.argv[3] if len(sys.argv) > 3 else "A"
a, b = waves(sys.argv[1], cell), waves(sys.argv[2], cell)
assert len(a) == len(b) == 8, "cell A sends the same 8 prompts in the same order on every boot"
for key in ("step_ms", "acceptance_len"):
    m, lo, hi, neg, n = paired(a, b, key)
    print(f"{cell} {key}: B-A {m:+.3f} [{lo:+.3f},{hi:+.3f}] (t 95%), lower on {neg}/{n}")
