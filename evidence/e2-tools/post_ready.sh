#!/usr/bin/env bash
# post_ready.sh DIR STAGE: the E2 post-ready sequence on a ready serve, run from the repo root.
#   serve capture (args/env/image parity), backend lines, MemAvailable snapshot,
#   count-200 probe, ruler v2 fast gate x2, cell T x1, Tier 0 compare --stage STAGE vs nvidia-v11-k7.
# Each step goes through evidence/e1-tools/act.sh (console + activity.tsv). Stops nothing on failure.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; STAGE="$2"
cd "$R" || exit 1
T=evidence/e1-tools
echo "== $(date -u +%FT%TZ) post_ready $D stage=$STAGE"
bash $T/capture_serve.sh "$D" > /dev/null 2>&1
bash $T/backend.sh > "$D/backend.txt" 2>&1
bash evidence/e2-tools/nodes.sh "after ready" > "$D/ready-mem.txt" 2>&1
bash $T/act.sh "$D" probe-count python3 $T/probes.py
bash $T/act.sh "$D" bench-1 python3 bench_decode.py --remote-meminfo --out "$D/bench-1"
bash $T/act.sh "$D" bench-2 python3 bench_decode.py --remote-meminfo --out "$D/bench-2"
bash $T/act.sh "$D" bench-T python3 bench_decode.py --remote-meminfo --cells T --out "$D/bench-T"
# STAGE=none: plain compare (the E1a gate), for configs that leave every target weight as it was.
stage_args=(--stage "$STAGE")
[[ "$STAGE" == none ]] && stage_args=()
bash $T/act.sh "$D" tier0-compare python3 $T/memguard.py --thresh 3.0 --log "$D/memguard-tier0-compare.log" -- \
  python3 quality/tier0.py compare --ref nvidia-v11-k7 "${stage_args[@]}" --out "$D/tier0.json"
python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
echo "== $(date -u +%FT%TZ) post_ready done"
