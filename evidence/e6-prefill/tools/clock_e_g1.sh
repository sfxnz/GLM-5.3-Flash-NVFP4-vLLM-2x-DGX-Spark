#!/usr/bin/env bash
# clock_e_g1.sh DIR LABEL: one more cell E panel on the ready serve (bench-LABEL) while sampling SM clock, power,
# temperature and clock-event reasons on both nodes every second (nvidia-smi query; no memory fields). Diagnostic
# for the G1 cell E spread: 7k warmups read +9% over F, 32k/128k panels 1150-1283 tok/s. Stops nothing.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$1"; L="$2"
cd "$R" || exit 1
Q="timestamp,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,clocks_event_reasons.active"
nvidia-smi --query-gpu="$Q" --format=csv,noheader -l 1 > "$D/clocks-$L-spark1.csv" 2>&1 &
p1=$!
ssh spark2 "nvidia-smi --query-gpu=$Q --format=csv,noheader -l 1" > "$D/clocks-$L-spark2.csv" 2>&1 &
p2=$!
bash evidence/e1-tools/act.sh "$D" "bench-$L" python3 evidence/e1-tools/memguard.py --thresh 3.0 \
  --log "$D/memguard-bench-$L.log" -- python3 bench_decode.py --remote-meminfo --cells E --out "$D/bench-$L"
kill $p1 $p2 2>/dev/null
ssh spark2 "pkill -f 'nvidia-smi --query-gpu=$Q'" 2>/dev/null
echo "== $(date -u +%FT%TZ) clock_e_g1 done"
