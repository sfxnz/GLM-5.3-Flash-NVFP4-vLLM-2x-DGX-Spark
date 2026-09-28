#!/usr/bin/env bash
# glm53_lines.sh: every distinct GLM53_ log line on both ranks, with its count, plus the KV pool line.
C="${CONTAINER_NAME:-glm53-flash-nvfp4}"
for n in head worker; do
  if [[ $n == head ]]; then run=(bash -c); else run=(ssh spark2); fi
  echo "== $n $(date -u +%FT%TZ)"
  "${run[@]}" "docker logs $C 2>&1 | grep -E 'GLM53_' | sed -E 's/^.*\] //' | cut -c1-300 | sort | uniq -c"
  "${run[@]}" "docker logs $C 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|Model loading took|Using V2 Model Runner' | cut -c1-240 | awk '!seen[\$0]++'"
done
