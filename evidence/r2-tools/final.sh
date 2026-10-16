#!/usr/bin/env bash
# final.sh DIR [tier0]: one published-defaults boot. Plain ./run.sh (boot.sh with no env), probes, fast gate,
# cell E, cell T; with "tier0" also Tier 0 compare (--stage fp8 vs nvidia-v11-k7) and the vision smoke. Then stop.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; Q="${2:-}"
cd "$R" || exit 1
T=evidence/e1-tools
mkdir -p "$D"
echo "== $(date -u +%FT%TZ) final $D $Q"
VALIDATE_ONLY=1 ./run.sh > "$D/validate-only.txt" 2>&1 || { echo "validate failed"; exit 1; }
bash $T/boot.sh "$D"
tail -2 "$D/run.log"
if curl -sf http://127.0.0.1:8000/v1/models > /dev/null; then
  bash evidence/e2-tools/nodes.sh "after ready" > "$D/ready-mem.txt" 2>&1
  bash $T/act.sh "$D" probe-count python3 $T/probes.py
  bash $T/act.sh "$D" bench-fast python3 bench_decode.py --remote-meminfo --out "$D/bench-fast"
  bash $T/act.sh "$D" bench-E python3 bench_decode.py --cells E --remote-meminfo --out "$D/bench-E"
  bash $T/act.sh "$D" bench-T python3 bench_decode.py --cells T --remote-meminfo --out "$D/bench-T"
  if [[ "$Q" == tier0 ]]; then
    bash $T/act.sh "$D" tier0-compare python3 quality/tier0.py compare --ref nvidia-v11-k7 --stage fp8 --out "$D/tier0.json"
    bash $T/act.sh "$D" smoke-vision python3 smoke_vision.py
  fi
  bash evidence/e2-tools/nodes.sh "after bench" > "$D/end-mem.txt" 2>&1
  docker logs glm53-flash-nvfp4 > "$D/head.log" 2>&1
  python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
fi
bash evidence/e2-tools/stop_serve.sh "$D" stop "$(cat "$D/watcher.pid" 2>/dev/null)" > /dev/null 2>&1
echo "== $(date -u +%FT%TZ) final done"
