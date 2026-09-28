#!/usr/bin/env bash
# stop_serve.sh DIR LABEL [WATCHER_PID]: snapshot both nodes, kill the SERVING watcher (if given),
# ./stop.sh, snapshot again. Writes DIR/LABEL-before.txt, DIR/LABEL.log, DIR/LABEL-after.txt.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; L="$2"; W="${3:-}"
cd "$R" || exit 1
bash evidence/e2-tools/nodes.sh "before $L" > "$D/$L-before.txt" 2>&1
{
  if [[ -n "$W" ]]; then
    echo "watcher: $(ps -o pid=,etime=,args= -p "$W" 2>/dev/null || echo "pid $W not running")"
    kill "$W" 2>/dev/null && echo "killed watcher $W"
    sleep 1
    ps -p "$W" >/dev/null 2>&1 && echo "watcher $W still alive" || echo "watcher $W gone"
  fi
  ./stop.sh
  echo "stop.sh rc=$?"
} > "$D/$L.log" 2>&1
sleep 3
bash evidence/e2-tools/nodes.sh "after $L" > "$D/$L-after.txt" 2>&1
cat "$D/$L.log" "$D/$L-after.txt"
