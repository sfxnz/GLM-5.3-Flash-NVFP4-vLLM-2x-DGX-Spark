#!/usr/bin/env bash
# table.sh DIR...: one line per cell (tok/s, step_ms, acc_len, prefill) from each cycle's bench consoles.
for d in "$@"; do
  echo "== $d $(grep -h 'PASS\|FAIL' "$d/probe-count.console.txt" 2>/dev/null | tail -1)"
  grep -hE '^(A|B|J@c1|J@c2|H|K@c1|K@c2|E@32k|E@128k|T) +[0-9]' "$d"/bench-*.console.txt | awk '{printf "  %-7s tok/s %7s step %6s acc %6s prefill %6s\n",$1,$4,$6,$7,$10}'
  grep -h "MemAvailable" "$d/ready-mem.txt" 2>/dev/null | tr '\n' ' '; echo
done
