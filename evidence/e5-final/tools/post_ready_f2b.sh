#!/usr/bin/env bash
# post_ready_f2b.sh DIR: the E5 F2 post-ready sequence on the rerun boot (f2b/; the first F2 boot in f2/ was stopped
# at ready by the pause). post_ready_f2.sh plus a second cell E and one cell T. Stops nothing.
#   serve capture + docker inspect diff vs F1, backend + GLM53_ lines, memory, count-200 + thinking-off smoke,
#   ruler v2 fast gate twice (bench-1, bench-2), cell E twice (bench-E1, bench-E2: prefill re-check, 3.0 GiB
#   memguard), cell T once (bench-T), memory per phase.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$1"
cd "$R" || exit 1
T=evidence/e1-tools
E5=evidence/e5-final/tools
echo "== $(date -u +%FT%TZ) post_ready_f2b $D"
bash $T/capture_serve.sh "$D" > /dev/null 2>&1
python3 $E5/inspect_diff.py evidence/e5-final/f1 "$D" > "$D/docker-inspect-vs-f1.txt" 2>&1
bash $T/backend.sh > "$D/backend.txt" 2>&1
bash evidence/e3-tools/glm53_lines.sh > "$D/glm53-lines.txt" 2>&1
bash evidence/e2-tools/nodes.sh "after ready" > "$D/ready-mem.txt" 2>&1
bash $T/act.sh "$D" probe-count python3 $T/probes.py
bash $T/act.sh "$D" bench-1 python3 bench_decode.py --remote-meminfo --out "$D/bench-1"
bash $T/act.sh "$D" bench-2 python3 bench_decode.py --remote-meminfo --out "$D/bench-2"
for i in 1 2; do
  bash $T/act.sh "$D" "bench-E$i" python3 $T/memguard.py --thresh 3.0 --log "$D/memguard-bench-E$i.log" -- \
    python3 bench_decode.py --remote-meminfo --cells E --out "$D/bench-E$i"
done
bash $T/act.sh "$D" bench-T python3 bench_decode.py --remote-meminfo --cells T --out "$D/bench-T"
bash evidence/e2-tools/nodes.sh "after post_ready" > "$D/end-mem.txt" 2>&1
python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
echo "== $(date -u +%FT%TZ) post_ready_f2b done"
