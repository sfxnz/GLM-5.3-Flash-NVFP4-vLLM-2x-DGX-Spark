#!/usr/bin/env bash
# cycle.sh DIR [NAME=VALUE ...]: one A/B boot. boot.sh (uma_watch + ./run.sh with the env), probes,
# fast gate + cell E, prefill chunk shapes from the head log, memory, then stop both nodes.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; shift
cd "$R" || exit 1
T=evidence/e1-tools
mkdir -p "$D"
echo "== $(date -u +%FT%TZ) cycle $D env: $*"
VALIDATE_ONLY=1 env "$@" ./run.sh > "$D/validate-only.txt" 2>&1 || { echo "validate failed"; cat "$D/validate-only.txt"; exit 1; }
bash $T/boot.sh "$D" "$@"
tail -3 "$D/run.log"
if curl -sf http://127.0.0.1:8000/v1/models > /dev/null; then
  bash evidence/e2-tools/nodes.sh "after ready" > "$D/ready-mem.txt" 2>&1
  bash $T/act.sh "$D" probe-count python3 $T/probes.py
  bash $T/act.sh "$D" bench-fast python3 bench_decode.py --remote-meminfo --out "$D/bench-fast"
  bash $T/act.sh "$D" bench-E python3 bench_decode.py --cells E --remote-meminfo --out "$D/bench-E"
  bash evidence/e2-tools/nodes.sh "after bench" > "$D/end-mem.txt" 2>&1
  docker logs glm53-flash-nvfp4 2>&1 | grep -E "GLM53|max_num_batched|chunk|Loading weights took|init engine|KV cache|Maximum concurrency" | head -60 > "$D/head-lines.txt"
  docker logs glm53-flash-nvfp4 > "$D/head.log" 2>&1
  python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
fi
bash evidence/e2-tools/stop_serve.sh "$D" stop "$(cat "$D/watcher.pid" 2>/dev/null)" > /dev/null 2>&1
echo "== $(date -u +%FT%TZ) cycle done"
grep -h "PASS\|FAIL" "$D/probe-count.console.txt" 2>/dev/null | tail -1
