#!/usr/bin/env python3
"""Per-cell summary across bench.json files: tok/s, step_ms, acceptance_len and
the per-position acceptance curve (P(draft position i accepted)).

    python3 evidence/e1-tools/per_pos.py LABEL=path/bench.json [LABEL=path/bench.json ...]
"""
import json
import sys

rows = []
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    rep = json.load(open(path))
    for s in rep["summary"]:
        rows.append((s["group"], label, s))
print(f"{'cell':8} {'run':14} {'tok/s':>7} {'step_ms':>8} {'acc_len':>8}  per_pos p1..p7")
for group in dict.fromkeys(g for g, _, _ in rows):
    for g, label, s in rows:
        if g != group:
            continue
        pp = " ".join(f"{x:.3f}" for x in (s.get("per_pos") or []))
        val = lambda k: s[k]["mean"] if s.get(k) else float("nan")  # noqa: E731
        print(f"{g:8} {label:14} {val('tok_s'):7.2f} {val('step_ms'):8.1f} {val('acceptance_len'):8.3f}  {pp}"
              f"{'' if s['valid'] else '  INVALID ' + str(s['invalid_reason'])}")
