#!/usr/bin/env python3
"""buckets_summary.py DIR [RANK]: one table of the F3 step_buckets.py JSONs (DIR/<kind>-rank<R>.json) side by side,
ms per step and % of wall per bucket. Stdlib only."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
R = sys.argv[2] if len(sys.argv) > 2 else "0"
kinds = [k for k in ("prose", "code", "structured", "prose2", "prefill") if (D / f"{k}-rank{R}.json").exists()]
d = {k: json.loads((D / f"{k}-rank{R}.json").read_text()) for k in kinds}
print(f"step_buckets.py, rank {R}, ms per step (F3: census + torch-profiler boot; not a perf number)")
print("prose/code/structured: c=1 verify steps (8 rows); prose2: c=2 verify steps (16 rows);"
      " prefill: prefill-chunk steps (3 x 1152 + 1 x 1267 tokens)")
print(f"{'':18}" + "".join(f"{k:>13}" for k in kinds))
print(f"{'steps':18}" + "".join(f"{d[k]['steps']:>13}" for k in kinds))
for key in ("wall_ms", "busy_ms", "idle_ms", "bucket_sum_ms"):
    print(f"{key:18}" + "".join(f"{d[k][key]:>13.2f}" for k in kinds))
for b in [r["bucket"] for r in d[kinds[0]]["buckets"]]:
    vals = [next(r for r in d[k]["buckets"] if r["bucket"] == b) for k in kinds]
    print(f"{b:18}" + "".join(f"{v['ms']:>8.2f} {v['pct_wall']:>3.0f}%" for v in vals))
