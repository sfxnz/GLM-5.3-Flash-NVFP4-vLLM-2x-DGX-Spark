#!/usr/bin/env bash
# dq_lines.sh: the GLM53_WQ_DEQUANT_MIN_M patch line, the INT8 swap line and the container's dequant env on both
# ranks, plus a count of Triton / JIT compile lines (the dequant kernel compiles on the first prefill >= MIN_M rows).
C="${CONTAINER_NAME:-glm53-flash-nvfp4}"
for n in head worker; do
  if [[ $n == head ]]; then run=(bash -c); else run=(ssh spark2); fi
  echo "== $n $(date -u +%FT%TZ)"
  "${run[@]}" "docker exec $C env | grep -E '^GLM53_WQ_DEQUANT' | sort"
  "${run[@]}" "docker logs $C 2>&1 | grep -E 'GLM53_WQ_DEQUANT_MIN_M=|GLM53_INT8_W8A16|GLM53_NVFP4_W4A16' | sed -E 's/^.*\] //' | cut -c1-400"
  echo "dequant lines: $("${run[@]}" "docker logs $C 2>&1 | grep -c 'GLM53_WQ_DEQUANT_MIN_M='")"
  echo "Traceback/ERROR lines: $("${run[@]}" "docker logs $C 2>&1 | grep -cE 'Traceback| ERROR '")"
done
