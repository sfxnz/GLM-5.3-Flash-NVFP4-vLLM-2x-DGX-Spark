#!/usr/bin/env bash
# rerun_flaky.sh DIR N: rerun only the Tier-0 utf8 and vision components N times on the running serve
# (quality/tier0.py compare --only utf8,vision), to tell a flaky greedy probe from a real regression.
# Writes DIR/tier0-flaky-K.json and DIR/tier0-flaky-K.console.txt; prints one summary line per run.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; N="${2:-3}"
cd "$R" || exit 1
for k in $(seq 1 "$N"); do
  bash evidence/e1-tools/act.sh "$D" "tier0-flaky-$k" python3 quality/tier0.py compare --ref nvidia-v11-k7 \
    --only utf8,vision --out "$D/tier0-flaky-$k.json" > /dev/null
  python3 - "$D/tier0-flaky-$k.json" "$k" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))["components"]
u, v = t["utf8"], t["vision"]
bad = [f"{c['name']}={c.get('answer')}" for c in v.get("checks", []) if not c.get("pass")]
print(f"run {sys.argv[2]}: utf8 pass={u['pass']} rows={u['rows']} sq_err={u['square_errors']} finish={u['finish_reason']}"
      f" | vision {v['passed']}/{v['total']} {bad}")
EOF
done
