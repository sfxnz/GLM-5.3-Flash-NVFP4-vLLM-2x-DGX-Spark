#!/usr/bin/env bash
# One hillclimb iteration against the shared TP=2 serve. Serial. Owns restart.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
ITER="${1:?usage: run-iter.sh <id>}"
OUT="evidence/iter-$ITER"
mkdir -p "$OUT"
export STATE_DIR=".run-state"
mkdir -p "$STATE_DIR"

doctor() { kit/doctor.sh .; }

echo "==> stopping previous serve"
./stop.sh | tee "$OUT/stop.txt"
printf 'started %s iter=%s host=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ITER" "$(hostname -s)" >"$STATE_DIR/started"

echo "==> starting ./run.sh (SPEC_CONFIG=${SPEC_CONFIG:-default} COMPILATION_CONFIG=${COMPILATION_CONFIG:-default})"
set +e
./run.sh >"$OUT/run.log" 2>&1
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then
  echo "run.sh failed rc=$rc" | tee "$OUT/result.txt"
  tail -80 "$OUT/run.log"
  exit "$rc"
fi

set +e
doctor >"$OUT/doctor.txt"
drc=$?
set -e
if [[ "$drc" -ne 0 ]]; then
  echo "doctor failed" | tee "$OUT/result.txt"
  cat "$OUT/doctor.txt"
  exit "$drc"
fi

python3 kit/probes/count.py . "$OUT" >"$OUT/count.txt" 2>&1
python3 kit/bench_decode.py --recipe . --phase both --out "$OUT" >/dev/null 2>&1
python3 - "$OUT/bench.txt" <<'PY'
import json, pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text()
idx = text.rfind("SUMMARY")
if idx < 0:
    raise SystemExit("no SUMMARY")
blob = text[idx + len("SUMMARY"):].strip()
pathlib.Path(sys.argv[1] + ".summary.json").write_text(blob + "\n")
rows = json.loads(blob)
for r in rows:
    if r["phase"] == "prose" and r["concurrency"] == 1:
        print(f"prose_c1={r['median_decode_tok_s']:.2f} accept={r.get('acceptance_len', '')}")
PY
echo "artifacts=$OUT"
