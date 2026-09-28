#!/usr/bin/env bash
# launch.sh OUT SCRIPT [ARGS...]: run evidence/e4-tools/SCRIPT detached (setsid nohup) from the repo root,
# console to OUT. Prints the PID.
R="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$1"; S="$2"; shift 2
cd "$R" || exit 1
setsid nohup bash "evidence/e4-tools/$S" "$@" > "$OUT" 2>&1 < /dev/null &
echo "started $S pid $! -> $OUT"
