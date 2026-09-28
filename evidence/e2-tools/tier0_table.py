#!/usr/bin/env python3
"""Side-by-side Tier-0 numbers from several tier0.json files (compare mode vs one reference).

    python3 evidence/e2-tools/tier0_table.py LABEL=path/tier0.json [...]

Prints the headline nll/greedy numbers, per-domain dNLL / top-1 / KL, the probe
results and each criterion's PASS/FAIL. Stdlib only.
"""
import json
import sys

runs = []
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    runs.append((label, json.load(open(path))))

w = 12
print(f"{'':22}" + "".join(f"{lab:>{w}}" for lab, _ in runs))


def row(name, fn, fmt):
    vals = []
    for _, r in runs:
        try:
            v = fn(r)
            vals.append(format(v, fmt) if v is not None else "-")
        except (KeyError, TypeError, IndexError):
            vals.append("-")
    print(f"{name:22}" + "".join(f"{v:>{w}}" for v in vals))


nll = lambda r: r["components"]["nll"]["vs_ref"]  # noqa: E731
row("dNLL", lambda r: nll(r)["delta"], "+.6f")
row("top1_agree", lambda r: nll(r)["top1_agree"], ".6f")
row("KL top20", lambda r: nll(r)["kl"], ".3e")
row("greedy hazard", lambda r: r["components"]["greedy"]["golden"]["hazard"], ".6f")
row("greedy diverged/20", lambda r: r["components"]["greedy"]["golden"]["diverged"], "d")
row("greedy first_div med", lambda r: sorted(x if x is not None else 200 for x in r["components"]["greedy"]["golden"]["first_div"])[10], "d")
domains = sorted({d for _, r in runs for d in nll(r).get("by_domain", {})})
for d in domains:
    row(f"{d} dNLL", lambda r, d=d: nll(r)["by_domain"][d]["delta"], "+.5f")
    row(f"{d} top1", lambda r, d=d: nll(r)["by_domain"][d]["top1_agree"], ".4f")
    row(f"{d} KL", lambda r, d=d: nll(r)["by_domain"][d]["kl"], ".3e")
row("count", lambda r: r["components"]["count"]["n_numbers"], "d")
row("kwargs", lambda r: r["components"]["kwargs"]["passed"], "d")
row("utf8 sq_err", lambda r: r["components"]["utf8"]["square_errors"], "d")
row("tools json_valid", lambda r: r["components"]["tools"]["json_valid"], ".2f")
row("tools args_ok", lambda r: r["components"]["tools"]["args_ok"], ".2f")
row("vision", lambda r: r["components"]["vision"]["passed"], "d")
row("needle 8k", lambda r: r["components"]["needle"]["per_length"]["8192"], "d")
row("needle 32k", lambda r: r["components"]["needle"]["per_length"]["32768"], "d")
names = list(dict.fromkeys(c["name"] for _, r in runs for c in r["criteria"]))
for n in names:
    row(n, lambda r, n=n: next((("PASS" if c["pass"] else "FAIL") for c in r["criteria"] if c["name"] == n), "-"), "s")
for lab, r in runs:
    print(f"{lab}: stage={r.get('stage')} {r['verdict']}")
