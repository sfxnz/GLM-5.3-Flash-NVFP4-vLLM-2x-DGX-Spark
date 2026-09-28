#!/usr/bin/env bash
# post_ready_g2.sh DIR [REF_DIR]: the E6 G2 post-ready sequence (the final published boot). Stops nothing.
#   serve capture + docker inspect diff vs REF_DIR (default G1), backend + GLM53_ + dequant lines, memory,
#   count-200 + thinking-off smoke, ruler v2 full panel (bench-1, 3.0 GiB memguard), fast gate (bench-2),
#   memory per phase. Same bench order as evidence/e5-final/tools/post_ready_f1.sh (full panel, then fast gate).
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$1"; REF="${2:-evidence/e6-prefill/g1}"
cd "$R" || exit 1
T=evidence/e1-tools
E5=evidence/e5-final/tools
G="python3 $T/memguard.py --thresh 3.0"
echo "== $(date -u +%FT%TZ) post_ready_g2 $D (ref $REF)"
bash $T/capture_serve.sh "$D" > /dev/null 2>&1
python3 $E5/inspect_diff.py "$REF" "$D" > "$D/docker-inspect-vs-ref.txt" 2>&1
python3 $E5/inspect_diff.py evidence/e5-final/f1 "$D" > "$D/docker-inspect-vs-f1.txt" 2>&1
bash $T/backend.sh > "$D/backend.txt" 2>&1
bash evidence/e3-tools/glm53_lines.sh > "$D/glm53-lines.txt" 2>&1
bash evidence/e2-tools/nodes.sh "after ready" > "$D/ready-mem.txt" 2>&1
bash evidence/e6-prefill/tools/dq_lines.sh > "$D/dq-lines.txt" 2>&1
bash $T/act.sh "$D" probe-count python3 $T/probes.py
bash $T/act.sh "$D" bench-1 $G --log "$D/memguard-bench-1.log" -- \
  python3 bench_decode.py --full --remote-meminfo --out "$D/bench-1"
bash $T/act.sh "$D" bench-2 python3 bench_decode.py --remote-meminfo --out "$D/bench-2"
bash evidence/e6-prefill/tools/dq_lines.sh > "$D/dq-lines-end.txt" 2>&1
bash evidence/e2-tools/nodes.sh "after post_ready" > "$D/end-mem.txt" 2>&1
python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
echo "== $(date -u +%FT%TZ) post_ready_g2 done"
