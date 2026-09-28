#!/usr/bin/env python3
"""One row per stream of decode_prefill_probe.py output: decode-vs-prefill and prefill A/A |delta| (mean, p99, max),
the share of tokens above 0.05 nats, and top-1 flips, all past the skip.

    python3 evidence/e3-tools/probe_table.py DIR/probe-kpool.json
"""
import json
import sys

d = json.load(open(sys.argv[1]))
print(f"n={d['n']} skip={d['skip']}  (d = decode - prefill1, aa = prefill2 - prefill1; |.| in nats)")
print(f"{'ctx':>7} {'c':>2} {'s':>2} {'prompt':>7} {'gen s':>6}  {'d mean':>7} {'aa mean':>7}  {'d p99':>6} {'aa p99':>6}"
      f"  {'d max':>6} {'aa max':>6}  {'d>.05':>6} {'aa>.05':>6}  flips p1/p2")
for r in d["results"]:
    if "error" in r:
        print(f"{r['ctx']:>7} {r['c']:>2} {r['stream']:>2} ERROR {r['error']}")
        continue
    a, b = r["decode_vs_prefill"], r["prefill_aa"]
    print(f"{r['ctx']:>7} {r['c']:>2} {r['stream']:>2} {r['prompt_tokens']:>7} {r['gen_s']:>6.0f}  "
          f"{a['mean_abs']:>7.4f} {b['mean_abs']:>7.4f}  {a['p99_abs']:>6.3f} {b['p99_abs']:>6.3f}  "
          f"{a['max_abs']:>6.2f} {b['max_abs']:>6.2f}  {100 * a['frac_gt_0.05']:>5.1f}% {100 * b['frac_gt_0.05']:>5.1f}%  "
          f"{r['flips_prefill1']}/{r['flips_prefill2']}")
