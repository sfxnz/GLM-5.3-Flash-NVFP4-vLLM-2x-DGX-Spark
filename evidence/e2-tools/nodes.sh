#!/usr/bin/env bash
# nodes.sh LABEL: docker ps -a, free -h and /proc/meminfo MemAvailable/Swap on both nodes (stdout).
echo "== $(date -u +%FT%TZ) $1"
for n in spark1 spark2; do
  if [[ $n == spark1 ]]; then run=(bash -c); else run=(ssh spark2); fi
  echo "== $n docker ps -a"
  "${run[@]}" "docker ps -a --format '{{.ID}} {{.Image}} {{.Status}} {{.Names}}'"
  echo "== $n free -h"
  "${run[@]}" "free -h; grep -E '^(MemAvailable|SwapTotal|SwapFree):' /proc/meminfo"
done
