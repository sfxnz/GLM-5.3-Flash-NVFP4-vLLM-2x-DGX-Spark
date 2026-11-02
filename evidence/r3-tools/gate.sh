#!/usr/bin/env bash
# gate.sh MAX_GIB -- CMD...: spark1 is shared with other projects' processes (2026-11-02: Playwright and node bursts of
# 8-22 GiB RSS). Wait until the summed RSS of every host process stays under MAX_GIB for 3 samples 20 s apart, log the
# wait, then exec CMD.
MAX="$1"; shift; [[ "$1" == "--" ]] && shift
ok=0
while (( ok < 3 )); do
  rss=$(ps -eo rss --no-headers | awk '{s+=$1} END {printf "%.1f", s/1048576}')
  if awk -v r="$rss" -v m="$MAX" 'BEGIN {exit !(r < m)}'; then ok=$((ok + 1)); else ok=0; fi
  echo "gate $(date -u +%FT%TZ) host_rss=${rss}G max=${MAX}G quiet=$ok/3"
  (( ok < 3 )) && sleep 20
done
exec "$@"
