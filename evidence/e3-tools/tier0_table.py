#!/usr/bin/env python3
"""Side-by-side Tier-0 numbers from tier0.json files: compare mode (vs a reference) and record mode (the
within-boot A/A in nll.rerun and greedy.aa). Extends evidence/e2-tools/tier0_table.py with the 128k needle,
the record A/A rows and the greedy divergence count.

    python3 evidence/e3-tools/tier0_table.py LABEL=path/tier0.json [...]
"""
import json
import sys

runs = []
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    runs.append((label, json.load(open(path))))

w = 12
print(f"{'':24}" + "".join(f"{lab:>{w}}" for lab, _ in runs))


def row(name, fn, fmt):
    vals = []
    for _, r in runs:
        try:
            v = fn(r)
            vals.append(format(v, fmt) if v is not None else "-")
        except (KeyError, TypeError, IndexError, StopIteration):
            vals.append("-")
    print(f"{name:24}" + "".join(f"{v:>{w}}" for v in vals))


C = lambda r: r["components"]  # noqa: E731
nll = lambda r: C(r)["nll"]["vs_ref"]  # noqa: E731
gold = lambda r: C(r)["greedy"]["golden"]  # noqa: E731
row("dNLL vs ref", lambda r: nll(r)["delta"], "+.6f")
row("top1_agree vs ref", lambda r: nll(r)["top1_agree"], ".6f")
row("KL top20 vs ref", lambda r: nll(r)["kl"], ".3e")
row("greedy hazard vs ref", lambda r: gold(r)["hazard"], ".6f")
row("greedy diverged/20", lambda r: gold(r)["diverged"], "d")
row("A/A |dNLL| (record)", lambda r: C(r)["nll"]["rerun"]["abs_delta"], ".6f")
row("A/A top1 (record)", lambda r: C(r)["nll"]["rerun"]["top1_agree"], ".6f")
row("A/A KL (record)", lambda r: C(r)["nll"]["rerun"]["kl"], ".3e")
row("A/A greedy hazard", lambda r: C(r)["greedy"]["aa"]["hazard"], ".6f")
row("A/A greedy diverged/20", lambda r: C(r)["greedy"]["aa"]["diverged"], "d")
row("count", lambda r: C(r)["count"]["n_numbers"], "d")
row("kwargs", lambda r: C(r)["kwargs"]["passed"], "d")
row("utf8 sq_err", lambda r: C(r)["utf8"]["square_errors"], "d")
row("tools json_valid", lambda r: C(r)["tools"]["json_valid"], ".2f")
row("tools args_ok", lambda r: C(r)["tools"]["args_ok"], ".2f")
row("vision", lambda r: C(r)["vision"]["passed"], "d")
for n in ("8192", "32768", "131072"):
    row(f"needle {n}", lambda r, n=n: C(r)["needle"]["per_length"][n], "d")
names = list(dict.fromkeys(c["name"] for _, r in runs for c in r.get("criteria", [])))
for n in names:
    row(n, lambda r, n=n: next((("PASS" if c["pass"] else "FAIL") for c in r["criteria"] if c["name"] == n), "-"), "s")
for lab, r in runs:
    print(f"{lab}: mode={r.get('mode')} stage={r.get('stage')} {r.get('verdict')}")
