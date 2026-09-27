#!/usr/bin/env bash
# Profile ~30 steady-state verify steps of a running GLM-5.3 TP=2 serve, on
# both ranks, and save the traces for tools/step_buckets.py.
#
# Why the torch profiler and not nsys: glm53-sm121-v11 (and v13) has no nsys
# (docker history shows no nsight layer), and run.sh cannot start the
# container under a bind-mounted host nsys (no entrypoint or volume hook). The
# image's own --profiler-config works through EXTRA_ARGS on both ranks and
# records every kernel inside the CUDA graphs. tools/README.md has the nsys
# route for when run.sh grows that hook.
#
# 1. Boot a profiling serve (not a perf arm: the profiler adds overhead):
#      IMAGE=glm53-sm121-v13 EXTRA_ARGS="--profiler-config $(tools/nsys_step.sh --print-config)" ./run.sh
#    Traces land in the HF cache mount on each node: $HF_CACHE/glm53-prof.
# 2. tools/nsys_step.sh            (CONCURRENCY=2 for the c=2 shape)
#    Warmup request, POST /start_profile, one streaming request per stream,
#    POST /stop_profile. The workers skip DELAY steps (prefill, first
#    decodes), record STEPS steps, and stop on their own.
# 3. ./stop.sh, then on an idle host, one rank at a time:
#      python3 tools/step_buckets.py OUT_DIR/rank0/<trace>.pt.trace.json.gz
#
# DELAY and STEPS are fixed at boot (--print-config reads them). This script
# never starts or stops the serve.
set -euo pipefail

DELAY="${DELAY:-8}"
STEPS="${STEPS:-30}"
PROF_DIR_IN_CONTAINER="/cache/huggingface/glm53-prof"
config() {
  printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false,"ignore_frontend":true,"delay_iterations":%d,"max_iterations":%d}\n' \
    "$PROF_DIR_IN_CONTAINER" "$DELAY" "$STEPS"
}
if [[ "${1:-}" == "--print-config" ]]; then
  config
  exit 0
fi

PORT="${PORT:-8000}"
API="${API:-http://127.0.0.1:$PORT}"
WORKER_HOST="${WORKER_HOST:-spark2}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PROF_DIR="$HF_CACHE/glm53-prof"
OUT_DIR="${OUT_DIR:-$HOME/projects/data/glm53-prof/$(date +%Y%m%d-%H%M%S)}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
RANK1="${RANK1:-1}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-600}"
FLUSH_S="${FLUSH_S:-15}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PROMPTS=(
  "Write a long, flowing essay about how river deltas form, change over centuries and support the people who farm them. Keep going in plain prose."
  "Write a long, flowing essay about the history of lighthouses, their keepers and the technology that replaced them. Keep going in plain prose."
)

model="$(curl -sf --max-time 10 "$API/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" \
  || { echo "no serve at $API" >&2; exit 1; }
body() {
  python3 - "$model" "$1" "$2" "$MAX_TOKENS" <<'PY'
import json, sys
model, prompt, stream, n = sys.argv[1], sys.argv[2], sys.argv[3] == "1", int(sys.argv[4])
print(json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": n, "min_tokens": n, "ignore_eos": True, "temperature": 0,
                  "stream": stream, "chat_template_kwargs": {"enable_thinking": False}}))
PY
}

echo "== warmup (profiler off) =="
curl -sf --max-time 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "$(body "${PROMPTS[0]}" 0)" >/dev/null

start_epoch="$(date +%s)"
echo "== start_profile (workers skip $DELAY steps, record $STEPS) =="
code="$(curl -s -o "$TMP/start.txt" -w '%{http_code}' --max-time 60 -X POST "$API/start_profile")"
if [[ "$code" != 200 ]]; then
  echo "start_profile HTTP $code: boot with EXTRA_ARGS=\"--profiler-config \$(tools/nsys_step.sh --print-config)\"" >&2
  exit 1
fi

echo "== $CONCURRENCY profiled stream(s), $MAX_TOKENS tokens each =="
pids=()
for ((i = 0; i < CONCURRENCY; i++)); do
  curl -sN --max-time 600 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(body "${PROMPTS[i % ${#PROMPTS[@]}]}" 1)" >"$TMP/stream.$i.sse" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "== stop_profile (blocks until the trace is written) =="
stop_rc=0
curl -s --max-time "$STOP_TIMEOUT_S" -X POST "$API/stop_profile" >/dev/null || stop_rc=$?
sleep "$FLUSH_S"

mkdir -p "$OUT_DIR/rank0"
echo "== rank 0 traces -> $OUT_DIR/rank0 =="
find "$PROF_DIR" -name '*.pt.trace.json*' -newermt "@$start_epoch" -exec cp {} "$OUT_DIR/rank0/" \;
ls -la "$OUT_DIR/rank0"
if [[ "$RANK1" == 1 ]]; then
  echo "== rank 1 traces ($WORKER_HOST) -> $OUT_DIR/rank1 =="
  mkdir -p "$OUT_DIR/rank1"
  mapfile -t remote < <(ssh "$WORKER_HOST" "find $(printf '%q' "$PROF_DIR") -name '*.pt.trace.json*' -newermt @$start_epoch")
  for f in "${remote[@]}"; do scp -q "$WORKER_HOST:$f" "$OUT_DIR/rank1/"; done
  ls -la "$OUT_DIR/rank1"
fi
config >"$OUT_DIR/profiler-config.json"
free -h | sed -n 1,2p
[[ "$stop_rc" == 0 ]] || { echo "WARN: stop_profile curl rc=$stop_rc; traces may be partial" >&2; exit 1; }
echo "Next: ./stop.sh, then python3 tools/step_buckets.py $OUT_DIR/rank0/<trace>.pt.trace.json.gz --json $OUT_DIR/rank0-buckets.json"
