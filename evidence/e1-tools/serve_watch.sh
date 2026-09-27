#!/usr/bin/env bash
# serve_watch.sh DIR: re-attach kit/uma_watch.py to a ready serve (SERVING floor 2 GiB), detached
# with setsid, logging into DIR (outside the repo for a long-lived serve). Prints the watcher PID.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"
mkdir -p "$D"
cd "$R" || exit 1
setsid nohup python3 kit/uma_watch.py --evidence "$D" --phase SERVING > "$D/watcher.out" 2> "$D/watcher.err" < /dev/null &
echo $! > "$D/watcher.pid"
echo "serving watcher pid $(cat "$D/watcher.pid") -> $D"
