# vendored from sfxnz/forge kit @ 6285f70
# shellcheck shell=bash
# Shared helpers for the recipe kit. Source from kit/*.sh and kit/probes/*.sh:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"; recipe_load "$1"
# recipe_load reads recipe.yaml and exports the serve values the kit needs. An
# env var that is already set wins, the same NAME="${NAME:-value}" rule run.sh uses.
# SC2034: the exported variables are read by the scripts that source this file.
# shellcheck disable=SC2034

log() { printf '==> %s\n' "$*"; }

utc_stamp() { date -u +%Y%m%dT%H%M%SZ; }

host_short() { hostname -s | tr '[:upper:]' '[:lower:]'; }

# head on spark1 (or anything that is not spark2), worker on spark2. ROLE overrides.
detect_role() {
  if [[ -n "${ROLE:-}" ]]; then
    printf '%s\n' "$ROLE"
    return
  fi
  case "$(host_short)" in
    spark2*) printf 'worker\n' ;;
    *) printf 'head\n' ;;
  esac
}

# recipe_load <recipe-dir>: sets RECIPE_DIR, PORT, CONTAINER_NAME, SERVED_NAME, IMAGE,
# WORKER_HOST, NNODES, MAX_MODEL_LEN, API, CHAT_TEMPLATE_KWARGS (JSON, from bench.chat_template_kwargs).
recipe_load() {
  RECIPE_DIR="$(cd "$1" && pwd)"
  if [[ ! -f "$RECIPE_DIR/recipe.yaml" ]]; then
    echo "recipe_load: $RECIPE_DIR/recipe.yaml not found" >&2
    return 1
  fi
  local line
  while IFS= read -r line; do
    eval "$line"
  done < <(python3 - "$RECIPE_DIR/recipe.yaml" <<'PY'
import json, os, shlex, sys, yaml
r = yaml.safe_load(open(sys.argv[1]))
env = r["serve"]["env"]
out = {
    "PORT": env.get("PORT", 8000),
    "CONTAINER_NAME": env.get("CONTAINER_NAME", ""),
    "SERVED_NAME": r["model"]["served_name"],
    "IMAGE": env.get("IMAGE", ""),
    "WORKER_HOST": env.get("WORKER_HOST", "spark2"),
    "NNODES": env.get("NNODES", 1),
    "MAX_MODEL_LEN": env.get("MAX_MODEL_LEN", ""),
    "CHAT_TEMPLATE_KWARGS": json.dumps((r.get("bench") or {}).get("chat_template_kwargs") or {}),
}
for k, v in out.items():
    # A caller's env override wins, like run.sh's NAME="${NAME:-value}".
    print(f"{k}={shlex.quote(os.environ.get(k) or str(v))}")
PY
  )
  API="http://127.0.0.1:${PORT}/v1"
}

# api_ready: /v1/models answers and names SERVED_NAME.
api_ready() {
  local body
  body="$(curl -sf --max-time 3 "${API}/models" 2>/dev/null || true)"
  [[ -n "$body" && "$body" == *"$SERVED_NAME"* ]]
}

# wait_ready [tries] [sleep_s]: poll /v1/models (default 480 x 5 s = 40 min).
wait_ready() {
  local tries="${1:-480}" pause="${2:-5}" i
  log "Waiting for ${API}/models"
  for ((i = 1; i <= tries; i++)); do
    if api_ready; then
      log "Ready → ${API}"
      return 0
    fi
    if command -v docker >/dev/null 2>&1 && [[ -n "$CONTAINER_NAME" ]] \
      && ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      echo "Container $CONTAINER_NAME is not running" >&2
      return 1
    fi
    sleep "$pause"
    if (( i % 12 == 0 )); then
      log "still loading… (${i}×${pause}s) — docker logs -f $CONTAINER_NAME"
    fi
  done
  echo "Timed out waiting for ${API}/models" >&2
  return 1
}

# worker_ssh <cmd...>: run a command on WORKER_HOST non-interactively.
worker_ssh() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "$@"
}

# worker_container_up: exit 0 when CONTAINER_NAME is running on WORKER_HOST.
worker_container_up() {
  command -v ssh >/dev/null 2>&1 \
    && worker_ssh "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'" >/dev/null 2>&1
}
