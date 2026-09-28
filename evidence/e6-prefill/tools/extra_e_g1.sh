#!/usr/bin/env bash
# extra_e_g1.sh DIR: two more cell E panels (bench-E3, bench-E4) on the ready G1 serve, added after the planned two
# because the planned E1@128k panel (1167 tok/s) sat 116 tok/s under E2@128k (1283) while every other G1 E cell read
# 1245-1283. Same command and 3.0 GiB memguard as post_ready_g1.sh. Stops nothing.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$1"
cd "$R" || exit 1
T=evidence/e1-tools
echo "== $(date -u +%FT%TZ) extra_e_g1 $D"
bash evidence/e2-tools/nodes.sh "before extra E" > "$D/extra-e-before-mem.txt" 2>&1
for i in 3 4; do
  bash $T/act.sh "$D" "bench-E$i" python3 $T/memguard.py --thresh 3.0 --log "$D/memguard-bench-E$i.log" -- \
    python3 bench_decode.py --remote-meminfo --cells E --out "$D/bench-E$i"
done
python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
echo "== $(date -u +%FT%TZ) extra_e_g1 done"
