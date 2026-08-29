#!/usr/bin/env bash
# Read-only: is this GLM-5.3-Flash serve worth driving?
# Exit: 0 ready, 1 missing, 2 loading, 3 mismatch
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

status=missing
container_state="absent"
image_running=""
api_ok=0
model_id=""
max_model_len=""
spec_method=""
worker="skipped"
owned_by_verify=0

if [[ -f "$STATE_DIR/started" ]]; then
  owned_by_verify=1
fi

if command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  container_state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo unknown)"
  image_running="$(docker inspect -f '{{.Config.Image}}' "$CONTAINER_NAME" 2>/dev/null || true)"
fi

if curl -sf --max-time 3 "${API}/models" >/tmp/verify-glm53-models.$$ 2>/dev/null; then
  api_ok=1
  model_id="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["data"][0]["id"])' /tmp/verify-glm53-models.$$)"
  max_model_len="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["data"][0].get("max_model_len",""))' /tmp/verify-glm53-models.$$)"
fi
rm -f /tmp/verify-glm53-models.$$

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

if command -v ssh >/dev/null 2>&1; then
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" \
      "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'" >/dev/null 2>&1; then
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
  expect_len="${MAX_MODEL_LEN:-327680}"
  if [[ "$max_model_len" != "$expect_len" ]]; then
    mismatch=1
  fi
  if [[ "$mismatch" -eq 1 ]]; then
    status=mismatch
    exit_code=3
  else
    status=ready
    exit_code=0
  fi
else
  status=missing
  exit_code=1
fi

cat <<EOF
status=$status
container=$CONTAINER_NAME
container_state=$container_state
image=$image_running
api=$API
model=$model_id
max_model_len=$max_model_len
spec=$spec_method
worker=$worker
owned_by_verify=$owned_by_verify
exit=$exit_code
EOF
exit "$exit_code"
