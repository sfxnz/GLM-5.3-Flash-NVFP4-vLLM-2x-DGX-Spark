#!/usr/bin/env bash
# post_ready_g1.sh DIR: the E6 G1 post-ready sequence on a ready serve booted with PREFILL_DEQUANT_MIN_M=512 ./run.sh.
# Stops nothing. evidence/e5-final/tools/post_ready_f2b.sh (same order: fast gate x2, cell E x2, cell T) plus the
# dequant line check, Tier 0 compare --stage fp8 and the flaky utf8/vision reruns (x3), as F1 ran them.
# Client steps that can load memory (cell E, Tier 0) run under a 3.0 GiB memguard.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$1"
cd "$R" || exit 1
T=evidence/e1-tools
E5=evidence/e5-final/tools
G="python3 $T/memguard.py --thresh 3.0"
echo "== $(date -u +%FT%TZ) post_ready_g1 $D"
bash $T/capture_serve.sh "$D" > /dev/null 2>&1
python3 $E5/inspect_diff.py evidence/e5-final/f1 "$D" > "$D/docker-inspect-vs-f1.txt" 2>&1
bash $T/backend.sh > "$D/backend.txt" 2>&1
bash evidence/e3-tools/glm53_lines.sh > "$D/glm53-lines.txt" 2>&1
bash evidence/e2-tools/nodes.sh "after ready" > "$D/ready-mem.txt" 2>&1
bash evidence/e6-prefill/tools/dq_lines.sh > "$D/dq-lines.txt" 2>&1
bash $T/act.sh "$D" probe-count python3 $T/probes.py
bash $T/act.sh "$D" bench-1 python3 bench_decode.py --remote-meminfo --out "$D/bench-1"
bash $T/act.sh "$D" bench-2 python3 bench_decode.py --remote-meminfo --out "$D/bench-2"
for i in 1 2; do
  bash $T/act.sh "$D" "bench-E$i" $G --log "$D/memguard-bench-E$i.log" -- \
    python3 bench_decode.py --remote-meminfo --cells E --out "$D/bench-E$i"
done
bash $T/act.sh "$D" bench-T python3 bench_decode.py --remote-meminfo --cells T --out "$D/bench-T"
bash $T/act.sh "$D" tier0-compare $G --log "$D/memguard-tier0-compare.log" -- \
  python3 quality/tier0.py compare --ref nvidia-v11-k7 --stage fp8 --out "$D/tier0.json"
bash evidence/e2-tools/rerun_flaky.sh "$D" 3
bash evidence/e6-prefill/tools/dq_lines.sh > "$D/dq-lines-end.txt" 2>&1
bash evidence/e2-tools/nodes.sh "after post_ready" > "$D/end-mem.txt" 2>&1
python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
echo "== $(date -u +%FT%TZ) post_ready_g1 done"
