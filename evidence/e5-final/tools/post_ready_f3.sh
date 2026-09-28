#!/usr/bin/env bash
# post_ready_f3.sh DIR PROF_OUT: the E5 F3 sequence on a ready census + torch-profiler boot. Stops nothing.
#   serve capture, GLM53_ lines (census recording line on both ranks), memory;
#   one profiled capture each (prof_step.py): prose c=1, code c=1, structured c=1, distinct prose c=2 -> PROF_OUT/<kind>;
#   census filler: bench_decode.py --cells A (not a perf number) until both ranks log the end of the census window,
#   then count-200 + thinking-off (the census only reads routing, so output must stay lossless).
# Request order fixes the census uids: prose warm 0, prose 1, code warm 2, code 3, structured warm 4, structured 5,
# prose2 warm 6, prose2 7-8, then the filler.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$1"; P="$2"
cd "$R" || exit 1
T=evidence/e1-tools
E5=evidence/e5-final/tools
C=glm53-flash-nvfp4
echo "== $(date -u +%FT%TZ) post_ready_f3 $D prof_out=$P"
bash $T/capture_serve.sh "$D" > /dev/null 2>&1
bash evidence/e3-tools/glm53_lines.sh > "$D/glm53-lines.txt" 2>&1
bash evidence/e2-tools/nodes.sh "after ready" > "$D/ready-mem.txt" 2>&1
for k in prose code structured prose2; do
  bash $T/act.sh "$D" "prof-$k" python3 $E5/prof_step.py --kind "$k" --out "$P/$k"
  bash evidence/e2-tools/nodes.sh "after prof-$k" > "$D/mem-after-prof-$k.txt" 2>&1
done
n=0
until [[ "$(docker logs $C 2>&1 | grep -c 'GLM53_EXPERT_CENSUS: rank 0 recorded')" -ge 1 && \
         "$(ssh spark2 "docker logs $C 2>&1 | grep -c 'GLM53_EXPERT_CENSUS: rank 1 recorded'")" -ge 1 ]]; do
  n=$((n + 1))
  if (( n > 4 )); then echo "census window not closed after $((n - 1)) filler panels"; break; fi
  bash $T/act.sh "$D" "census-filler-$n" python3 bench_decode.py --remote-meminfo --cells A --out "$D/census-filler-$n"
done
bash $T/act.sh "$D" probe-count python3 $T/probes.py
bash evidence/e3-tools/glm53_lines.sh > "$D/glm53-lines-end.txt" 2>&1
bash evidence/e2-tools/nodes.sh "after post_ready" > "$D/end-mem.txt" 2>&1
python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
echo "== $(date -u +%FT%TZ) post_ready_f3 done"
