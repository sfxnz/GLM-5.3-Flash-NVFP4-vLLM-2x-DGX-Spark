#!/usr/bin/env bash
# vendored from sfxnz/forge kit @ 6285f70
# Read-only: is this recipe's serve ready on every rank? Usage: kit/doctor.sh <recipe-dir>
# Prints one line: status=ready|missing|loading|mismatch|worker-down container=... worker=up|down|skipped ...
# Exit: 0 ready on all ranks, 1 missing, 2 loading, 3 mismatch (image, served name or
# max_model_len differ from recipe.yaml), 4 head ready but the worker container is not running.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
recipe_load "${1:?usage: kit/doctor.sh <recipe-dir>}"

container_state="absent"
image_running=""
api_ok=0
model_id=""
max_model_len=""
spec_method=""
worker="skipped"

if command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER_NAME"; then
  container_state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo unknown)"
  image_running="$(docker inspect -f '{{.Config.Image}}' "$CONTAINER_NAME" 2>/dev/null || true)"
fi

models_tmp="$(mktemp)"
trap 'rm -f "$models_tmp"' EXIT
if curl -sf --max-time 3 "${API}/models" >"$models_tmp" 2>/dev/null; then
  api_ok=1
  model_id="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["data"][0]["id"])' "$models_tmp")"
  max_model_len="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["data"][0].get("max_model_len",""))' "$models_tmp")"
fi

if [[ "$container_state" == "running" ]]; then
  spec_method="$(
    docker inspect -f '{{json .Config.Cmd}}' "$CONTAINER_NAME" 2>/dev/null | python3 -c '
import json, sys
cmd = json.load(sys.stdin)
spec = "unknown"
for i, arg in enumerate(cmd):
    if arg == "--speculative-config" and i + 1 < len(cmd):
        try:
            spec = json.loads(cmd[i + 1]).get("method") or "unknown"
        except (json.JSONDecodeError, TypeError, AttributeError):
            spec = "unknown"
        break
print(spec)
'
  )"
fi

if [[ "$NNODES" -gt 1 ]] && command -v ssh >/dev/null 2>&1; then
  if worker_container_up; then
    worker=up
  else
    worker=down
  fi
fi

if [[ "$container_state" == "absent" && "$api_ok" -eq 0 ]]; then
  status=missing
  exit_code=1
elif [[ "$container_state" == "running" && "$api_ok" -eq 0 ]]; then
  status=loading
  exit_code=2
elif [[ "$api_ok" -eq 1 ]]; then
  mismatch=0
  if [[ -n "$image_running" && "$image_running" != "$IMAGE" && "$image_running" != "$IMAGE:latest" ]]; then
    mismatch=1
  fi
  if [[ "$model_id" != "$SERVED_NAME" ]]; then
    mismatch=1
  fi
  if [[ -n "$MAX_MODEL_LEN" && "$max_model_len" != "$MAX_MODEL_LEN" ]]; then
    mismatch=1
  fi
  if [[ "$mismatch" -eq 1 ]]; then
    status=mismatch
    exit_code=3
  elif [[ "$NNODES" -gt 1 && "$worker" != up ]]; then
    status="worker-down"
    exit_code=4
  else
    status=ready
    exit_code=0
  fi
else
  status=missing
  exit_code=1
fi

printf 'status=%s worker=%s container=%s container_state=%s image=%s api=%s model=%s max_model_len=%s spec=%s exit=%s\n' \
  "$status" "$worker" "$CONTAINER_NAME" "$container_state" "$image_running" "$API" "$model_id" "$max_model_len" "$spec_method" "$exit_code"
exit "$exit_code"
