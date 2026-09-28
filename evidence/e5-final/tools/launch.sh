#!/usr/bin/env bash
# launch.sh OUT SCRIPT [ARGS...]: run SCRIPT (a path relative to the repo root) detached with setsid nohup from the
# repo root, console to OUT. Prints the PID. Same as evidence/e4-tools/launch.sh, for scripts outside e4-tools.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$1"; S="$2"; shift 2
cd "$R" || exit 1
setsid nohup bash "$S" "$@" > "$OUT" 2>&1 < /dev/null &
echo "started $S pid $! -> $OUT"
