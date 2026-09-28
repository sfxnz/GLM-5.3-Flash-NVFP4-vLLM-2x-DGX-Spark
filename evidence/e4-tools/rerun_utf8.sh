#!/usr/bin/env bash
# rerun_utf8.sh DIR FIRST LAST: utf8-only Tier 0 reruns K = FIRST..LAST on the running serve (numbered after
# rerun_flaky.sh's 1..N), DIR/tier0-flaky-K.json + console; one summary line per run.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; F="$2"; L="$3"
cd "$R" || exit 1
for k in $(seq "$F" "$L"); do
  bash evidence/e1-tools/act.sh "$D" "tier0-flaky-$k" python3 quality/tier0.py compare --ref nvidia-v11-k7 \
    --only utf8 --out "$D/tier0-flaky-$k.json" > /dev/null
  python3 - "$D/tier0-flaky-$k.json" "$k" <<'EOF'
import json, sys
u = json.load(open(sys.argv[1]))["components"]["utf8"]
keys = ("pass", "rows", "square_errors", "cube_errors", "chinese_numeral_errors", "finish_reason", "error")
print(f"run {sys.argv[2]}: utf8 " + " ".join(f"{k}={u.get(k)}" for k in keys if k in u))
EOF
done
