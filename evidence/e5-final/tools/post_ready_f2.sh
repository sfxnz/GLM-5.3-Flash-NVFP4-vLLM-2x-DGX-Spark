#!/usr/bin/env bash
# post_ready_f2.sh DIR: the E5 F2 post-ready sequence (second boot of plain ./run.sh). Stops nothing.
#   serve capture + docker inspect diff vs F1, backend + GLM53_ lines, memory, count-200 + thinking-off smoke,
#   ruler v2 fast gate twice (bench-1, bench-2), then cell E (prefill cross-check), memory per phase.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$1"
cd "$R" || exit 1
T=evidence/e1-tools
E5=evidence/e5-final/tools
echo "== $(date -u +%FT%TZ) post_ready_f2 $D"
bash $T/capture_serve.sh "$D" > /dev/null 2>&1
python3 $E5/inspect_diff.py evidence/e5-final/f1 "$D" > "$D/docker-inspect-vs-f1.txt" 2>&1
bash $T/backend.sh > "$D/backend.txt" 2>&1
bash evidence/e3-tools/glm53_lines.sh > "$D/glm53-lines.txt" 2>&1
bash evidence/e2-tools/nodes.sh "after ready" > "$D/ready-mem.txt" 2>&1
bash $T/act.sh "$D" probe-count python3 $T/probes.py
bash $T/act.sh "$D" bench-1 python3 bench_decode.py --remote-meminfo --out "$D/bench-1"
bash $T/act.sh "$D" bench-2 python3 bench_decode.py --remote-meminfo --out "$D/bench-2"
# Extra, after the two fast gates: cell E again, because F1's prefill (1193 tok/s) sat ~10% under E0/E3a.
bash $T/act.sh "$D" bench-E python3 $T/memguard.py --thresh 3.0 --log "$D/memguard-bench-E.log" -- \
  python3 bench_decode.py --remote-meminfo --cells E --out "$D/bench-E"
bash evidence/e2-tools/nodes.sh "after post_ready" > "$D/end-mem.txt" 2>&1
python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
echo "== $(date -u +%FT%TZ) post_ready_f2 done"
