#!/usr/bin/env bash
# post_ready.sh DIR a|b: the E3 post-ready sequence on a ready serve, run from the repo root. Stops nothing on failure.
#   both: serve capture (args/env/image parity), backend lines, GLM53_ lines per rank, MemAvailable snapshot,
#         count-200 + thinking-off probe, ruler v2 fast gate x2, cell T.
#   a:    + cell E (32k/128k), Tier 0 compare --long vs nvidia-v11-k7, Tier 0 record --name e3a-det --only nll,greedy
#         (the within-boot A/A; the other components ran in the compare),
#         decode-vs-prefill kpool probe (1.5k/32k/128k at c=1 and c=2, 2000 tokens).
#   b:    + Tier 0 compare vs nvidia-v11-k7.
# Each step goes through evidence/e1-tools/act.sh (console + activity.tsv); client steps under a 3.0 GiB memguard.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; MODE="$2"
cd "$R" || exit 1
T=evidence/e1-tools
G="python3 $T/memguard.py --thresh 3.0"
echo "== $(date -u +%FT%TZ) post_ready $D mode=$MODE"
bash $T/capture_serve.sh "$D" > /dev/null 2>&1
bash $T/backend.sh > "$D/backend.txt" 2>&1
bash evidence/e3-tools/glm53_lines.sh > "$D/glm53-lines.txt" 2>&1
bash evidence/e2-tools/nodes.sh "after ready" > "$D/ready-mem.txt" 2>&1
bash $T/act.sh "$D" probe-count python3 $T/probes.py
bash $T/act.sh "$D" bench-1 python3 bench_decode.py --remote-meminfo --out "$D/bench-1"
bash $T/act.sh "$D" bench-2 python3 bench_decode.py --remote-meminfo --out "$D/bench-2"
bash $T/act.sh "$D" bench-T python3 bench_decode.py --remote-meminfo --cells T --out "$D/bench-T"
if [[ "$MODE" == a ]]; then
  bash $T/act.sh "$D" bench-E $G --log "$D/memguard-bench-E.log" -- \
    python3 bench_decode.py --remote-meminfo --cells E --out "$D/bench-E"
  bash $T/act.sh "$D" tier0-compare $G --log "$D/memguard-tier0-compare.log" -- \
    python3 quality/tier0.py compare --ref nvidia-v11-k7 --long --out "$D/tier0.json"
  bash $T/act.sh "$D" tier0-record $G --log "$D/memguard-tier0-record.log" -- \
    python3 quality/tier0.py record --name e3a-det --only nll,greedy --out "$D/tier0-record.json"
  bash $T/act.sh "$D" probe-kpool $G --log "$D/memguard-probe-kpool.log" -- \
    python3 evidence/e3-tools/decode_prefill_probe.py --ctx 1500,32768,131072 --c 1,2 --n 2000 --out "$D/probe-kpool.json"
else
  bash $T/act.sh "$D" tier0-compare $G --log "$D/memguard-tier0-compare.log" -- \
    python3 quality/tier0.py compare --ref nvidia-v11-k7 --out "$D/tier0.json"
fi
python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
echo "== $(date -u +%FT%TZ) post_ready done"
