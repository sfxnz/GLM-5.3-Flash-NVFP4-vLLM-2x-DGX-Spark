#!/usr/bin/env bash
# act.sh DIR LABEL CMD...: run CMD from the repo root, tee its console to DIR/LABEL.console.txt,
# and append start/end/label/rc to DIR/activity.tsv (phase_report.py reads it).
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; L="$2"; shift 2
cd "$R" || exit 1
[[ -s "$D/activity.tsv" ]] || printf 'start\tend\tlabel\trc\n' > "$D/activity.tsv"
t0="$(date -u +%FT%T.%3NZ)"
"$@" > "$D/$L.console.txt" 2>&1
rc=$?
printf '%s\t%s\t%s\t%s\n' "$t0" "$(date -u +%FT%T.%3NZ)" "$L" "$rc" >> "$D/activity.tsv"
tail -25 "$D/$L.console.txt"
echo "[$L rc=$rc]"
exit $rc
