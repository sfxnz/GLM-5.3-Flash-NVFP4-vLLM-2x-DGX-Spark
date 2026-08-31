#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 on 2x DGX Spark (GB10) — vLLM TP=2
set -euo pipefail

MODEL="${MODEL:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
SERVED_NAME="${SERVED_NAME:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
IMAGE="${IMAGE:-glm53-sm121-v11}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-nvfp4}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29521}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
WORKER_HOST="${WORKER_HOST:-spark2}"
IFACE="${IFACE:-enp1s0f1np1}"
HCA="${HCA:-rocep1s0f1}"
TP="${TP:-2}"
NNODES="${NNODES:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-327680}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
UTIL="${UTIL:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
# Empty: vLLM sets 2048 under DFlash2. 4096 at seqs=2 shrank the fp8 pool
# (372877→363476) and slowed structured c=2 (55.5→51.6). Leave unset.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
FORCE_UNSAFE_CTX="${FORCE_UNSAFE_CTX:-0}"
# Empty: engine auto-enables breakable CUDA graphs. 0 slowed structured
# c=1 69.4→67.0 and c=2 59.7→52.1. Leave unset.
VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/chat_template.jinja}"
# GB10 UMA: 4.14 GiB is the safe KV pin on TP=2. Dropping the pin OOMs
# (NV_ERR_NO_MEMORY). Raising it boots but degrades: 5.0 GiB slowed decode
# ~20% at every concurrency (UMA pressure), and 5.14 GiB crashed under
# concurrent load. Tony's 3.0 GiB pin (3221225472) cannot hold 327680
# (vLLM wants 3.62 GiB; estimated max len 239616). 4.0 GiB boots but
# structured c=2 fell 59.5→52.4 (pool 372877→361577).
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-4445787956}"
# DeepGEMM arch-12 fp8 paged-MQA only accepts 64-entry pool pages. 2304 tiles that.
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HF_HOME_IN_CONTAINER="/cache/huggingface"
# caca4e6 adds calibrated MoE input_scale (LibertAI 2026-08-30). aa28e1f is the rollback.
SNAPSHOT_REV="${SNAPSHOT_REV:-caca4e6a4ebbd66f159d3d2fc256683fd6e27177}"
SNAPSHOT="${HF_CACHE}/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/${SNAPSHOT_REV}"
SNAPSHOT_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/${SNAPSHOT_REV}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"
REASONING_PARSER="${REASONING_PARSER:-glm45}"
DRAFT_MODEL="${DRAFT_MODEL:-incoai/GLM-5.3-Flash-DFlash2}"
DRAFT_SNAPSHOT="${HF_CACHE}/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/7d74cdd881ed7e32c31175984a67823127b66cfe"
DRAFT_SNAPSHOT_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/7d74cdd881ed7e32c31175984a67823127b66cfe"
# SPEC picks the drafter: dflash2 (incoai DFlash2 block-diffusion draft, needs
# the glm53-sm121-v11 image) or mtp (GLM's native MTP head).
SPEC="${SPEC:-dflash2}"
if [[ -z "${SPEC_CONFIG:-}" ]]; then
  case "$SPEC" in
    dflash2)
      # Default is the trained block (7 of 8). That occupancy fits two
      # sequences on the 4.14 GiB pin. MAX_NUM_SEQS=3 does not starve the
      # third stream (TTFT ~0.4 s, even per-stream) but structured c=2
      # fell 59.5→50.7. NUM_SPECULATIVE_TOKENS=5 MAX_NUM_SEQS=4 is the
      # four-way rollback (positions 5-6 accept <15% on prose, and each
      # extra slot is a KDA copy that starves the 4th request at 7).
      SPEC_CONFIG='{"method":"dflash","model":"'"$DRAFT_SNAPSHOT_IN_CONTAINER"'","num_speculative_tokens":'"$NUM_SPECULATIVE_TOKENS"'}'
      ;;
    mtp)
      SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":4}'
      ;;
    *)
      echo "Unknown SPEC=$SPEC (want dflash2 or mtp)" >&2
      exit 1
      ;;
  esac
fi
# CUDA graphs (default). Capture sizes are 1/2/4 plus (num_spec+1)×{1..MAX_NUM_SEQS}.
# ENFORCE_EAGER=1 is the rollback.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  if [[ "$SPEC" == dflash2 ]]; then
    step=$((NUM_SPECULATIVE_TOKENS + 1))
    sizes="1,2,4"
    i=1
    while (( i <= MAX_NUM_SEQS )); do
      sizes+=",$((step * i))"
      i=$((i + 1))
    done
    COMPILATION_CONFIG='{"cudagraph_capture_sizes":['"$sizes"']}'
  else
    COMPILATION_CONFIG='{"cudagraph_capture_sizes":[1,2,4,8,16,24]}'
  fi
fi
# fp8 hybrid pool is ~400k tokens on the 4.14 GiB pin. Native 1,048,576 does not
# fit. Packed NVFP4 KV is a different image/backend, not MAX_MODEL_LEN on this pin.
if [[ "$KV_CACHE_DTYPE" == fp8_e4m3 && "$MAX_MODEL_LEN" -gt 327680 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "fp8 KV pin (~400k tokens, 4.14 GiB) cannot hold --max-model-len $MAX_MODEL_LEN. A 1M request needs ~8.2 GiB of this hybrid layout and GB10 UMA OOMs above ~5.1 GiB. Do not advertise a window the pool cannot serve. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
# 327680 needs more than the displayed 3.62 GiB (3886945403 still estimates
# max len 327168). 3.0 GiB estimates 239616. 4.14 GiB is the known-good pin.
if [[ "$MAX_MODEL_LEN" -gt 239616 && "$KV_CACHE_MEMORY" -le 3886945403 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "KV pin $KV_CACHE_MEMORY cannot hold --max-model-len $MAX_MODEL_LEN (need more than 3.62 GiB; 3886945403 estimates max len 327168). Tony's 3.0 GiB pin is a 262144-ctx budget. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf '==> validate-only spec=%s seqs=%s spec_tokens=%s eager=%s compilation=%s snapshot=%s moe=%s\n' \
    "$SPEC" "$MAX_NUM_SEQS" "$NUM_SPECULATIVE_TOKENS" "$ENFORCE_EAGER" "$COMPILATION_CONFIG" \
    "$SNAPSHOT_REV" "$MOE_BACKEND"
  exit 0
fi
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
# Extra vllm serve args, word-split on purpose (e.g. "--load-format dummy").
EXTRA_ARGS="${EXTRA_ARGS:-}"

log() { printf '==> %s\n' "$*"; }

host_short() { hostname -s | tr '[:upper:]' '[:lower:]'; }

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

hf_bin() {
  if command -v hf >/dev/null 2>&1; then
    echo hf
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo huggingface-cli
  else
    return 1
  fi
}

token_env() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN"
    return
  fi
  if [[ -f "$HOME/.cache/huggingface/token" ]]; then
    tr -d '[:space:]' <"$HOME/.cache/huggingface/token"
  fi
}

resolve_model() {
  if [[ -d "$SNAPSHOT" ]]; then
    printf '%s\n' "$SNAPSHOT_IN_CONTAINER"
  else
    printf '%s\n' "$MODEL"
  fi
}

maybe_drop_caches() {
  if sudo -n true >/dev/null 2>&1; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  fi
}

ensure_image() {
  log "Ensuring image $IMAGE"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not found. Build the local image chain through glm53-sm121-v11 first (see README). Do not use stock vllm/vllm-openai on sm_121." >&2
    exit 1
  fi
}

ensure_weights() {
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    return
  fi
  local HF=""
  HF="$(hf_bin || true)"
  if [[ -d "$SNAPSHOT" ]]; then
    log "Using pinned snapshot $SNAPSHOT"
  elif [[ -n "$HF" ]]; then
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    log "Downloading $MODEL (resumes under $HF_CACHE)"
    "$HF" download "$MODEL"
  else
    log "No hf CLI on PATH — vLLM will pull weights on first load"
  fi
  if [[ "$SPEC_CONFIG" != *'"dflash"'* ]]; then
    return
  fi
  if [[ -d "$DRAFT_SNAPSHOT" ]]; then
    log "Using pinned draft snapshot $DRAFT_SNAPSHOT"
  elif [[ -n "$HF" ]]; then
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    log "Downloading $DRAFT_MODEL (resumes under $HF_CACHE)"
    "$HF" download "$DRAFT_MODEL"
  else
    # The dflash config points at the pinned snapshot path inside the
    # container, so vLLM cannot pull it on demand.
    echo "Draft snapshot $DRAFT_SNAPSHOT missing and no hf CLI on PATH." >&2
    exit 1
  fi
}

stop_local() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "Removing existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
}

start_local() {
  local rank="$1"
  mkdir -p "$HF_CACHE"
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found" >&2
    exit 1
  fi
  maybe_drop_caches
  stop_local
  ensure_image
  ensure_weights

  local serve_model
  serve_model="$(resolve_model)"

  local tok
  tok="$(token_env || true)"
  local env_args=(
    -e "HF_HOME=$HF_HOME_IN_CONTAINER"
    -e "TORCH_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_DISABLE_VERSION_CHECK=1"
    -e "VLLM_ENGINE_READY_TIMEOUT_S=3600"
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    -e "NCCL_SOCKET_IFNAME=$IFACE"
    -e "GLOO_SOCKET_IFNAME=$IFACE"
    -e "TP_SOCKET_IFNAME=$IFACE"
    -e "NCCL_IB_HCA=$HCA"
    -e "NCCL_NET=IB"
    -e "NCCL_IB_DISABLE=0"
    -e "NCCL_CROSS_NIC=1"
    -e "NCCL_NVLS_ENABLE=0"
    -e "NCCL_CUMEM_ENABLE=0"
    -e "NCCL_DEBUG=WARN"
  )
  if [[ -n "${VLLM_USE_BREAKABLE_CUDAGRAPH}" ]]; then
    env_args+=(-e "VLLM_USE_BREAKABLE_CUDAGRAPH=$VLLM_USE_BREAKABLE_CUDAGRAPH")
  fi
  local host_ip="$HEAD_IP"
  if [[ "$rank" != "0" ]]; then
    host_ip="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
    host_ip="${host_ip:-10.100.8.2}"
  fi
  env_args+=(-e "VLLM_HOST_IP=$host_ip")
  if [[ -n "$tok" ]]; then
    env_args+=(-e "HF_TOKEN=$tok" -e "HUGGING_FACE_HUB_TOKEN=$tok")
  fi

  local rank_args=()
  if [[ "$rank" == "0" ]]; then
    rank_args+=(--host 0.0.0.0 --port "$PORT")
  else
    rank_args+=(--headless)
  fi

  local eager_args=()
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    eager_args+=(--enforce-eager)
  else
    eager_args+=(--compilation-config "$COMPILATION_CONFIG")
  fi

  local vol_args=(-v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}")
  local template_args=()
  if [[ "$rank" == "0" && -f "$CHAT_TEMPLATE" ]]; then
    vol_args+=(-v "${CHAT_TEMPLATE}:/chat_template.jinja:ro")
    template_args+=(--chat-template /chat_template.jinja)
  fi
  local batched_args=()
  if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    batched_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi

  log "Starting $CONTAINER_NAME rank=$rank model=$serve_model ctx=$MAX_MODEL_LEN kv=$KV_CACHE_MEMORY eager=$ENFORCE_EAGER spec=$SPEC"
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart no \
    --gpus all \
    --network host \
    --ipc host \
    --shm-size 32g \
    --device /dev/infiniband \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1:-1 \
    "${vol_args[@]}" \
    "${env_args[@]}" \
    "$IMAGE" \
    "$serve_model" \
    --tensor-parallel-size "$TP" \
    --nnodes "$NNODES" \
    --node-rank "$rank" \
    --distributed-executor-backend mp \
    --master-addr "$HEAD_IP" \
    --master-port "$MASTER_PORT" \
    "${rank_args[@]}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --kv-cache-memory "$KV_CACHE_MEMORY" \
    --gpu-memory-utilization "$UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    "${batched_args[@]}" \
    "${eager_args[@]}" \
    --block-size "$BLOCK_SIZE" \
    --moe-backend "$MOE_BACKEND" \
    --speculative-config "$SPEC_CONFIG" \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --reasoning-parser "$REASONING_PARSER" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    "${template_args[@]}" \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    $EXTRA_ARGS
}

wait_ready() {
  log "Waiting for http://127.0.0.1:${PORT}/v1/models"
  local i
  for i in $(seq 1 480); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      log "Ready → http://127.0.0.1:${PORT}/v1  (context=$MAX_MODEL_LEN)"
      curl -s "http://127.0.0.1:${PORT}/v1/models" || true
      echo
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      echo "Container exited early. Logs:" >&2
      docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
      exit 1
    fi
    sleep 5
    if (( i % 12 == 0 )); then
      log "still loading… (${i}×5s) — docker logs -f $CONTAINER_NAME"
    fi
  done
  echo "Timed out waiting for API. Recent logs:" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
  exit 1
}

ROLE="$(detect_role)"
log "role=$ROLE host=$(host_short)"

if [[ "$ORCHESTRATE" == "auto" && "$ROLE" == "head" ]]; then
  if command -v ssh >/dev/null 2>&1 && ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
    log "Starting worker on $WORKER_HOST first"
    scp -q "$0" "${WORKER_HOST}:/tmp/glm53-run.sh"
    ssh "$WORKER_HOST" \
      "ROLE=worker ORCHESTRATE=0 IMAGE='$IMAGE' CONTAINER_NAME='$CONTAINER_NAME' PORT='$PORT' MASTER_PORT='$MASTER_PORT' HEAD_IP='$HEAD_IP' IFACE='$IFACE' HCA='$HCA' MAX_MODEL_LEN='$MAX_MODEL_LEN' MAX_NUM_SEQS='$MAX_NUM_SEQS' UTIL='$UTIL' KV_CACHE_MEMORY='$KV_CACHE_MEMORY' KV_CACHE_DTYPE='$KV_CACHE_DTYPE' BLOCK_SIZE='$BLOCK_SIZE' TP='$TP' NNODES='$NNODES' SERVED_NAME='$SERVED_NAME' SKIP_DOWNLOAD='$SKIP_DOWNLOAD' SPEC='$SPEC' SPEC_CONFIG='$SPEC_CONFIG' NUM_SPECULATIVE_TOKENS='$NUM_SPECULATIVE_TOKENS' ENFORCE_EAGER='$ENFORCE_EAGER' COMPILATION_CONFIG='$COMPILATION_CONFIG' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' FORCE_UNSAFE_CTX='$FORCE_UNSAFE_CTX' VLLM_USE_BREAKABLE_CUDAGRAPH='$VLLM_USE_BREAKABLE_CUDAGRAPH' SNAPSHOT_REV='$SNAPSHOT_REV' MOE_BACKEND='$MOE_BACKEND' REASONING_PARSER='$REASONING_PARSER' EXTRA_ARGS='$EXTRA_ARGS' bash /tmp/glm53-run.sh"
    log "Worker container started. Waiting 25s for NCCL listen, then starting head"
    sleep 25
  else
    log "Cannot SSH to $WORKER_HOST — starting local rank only. Run ROLE=worker ./run.sh on the other Spark first."
  fi
  start_local 0
  wait_ready
  log "Stop with: ./stop.sh"
elif [[ "$ROLE" == "worker" ]]; then
  start_local 1
  log "Worker rank 1 is up. Head should start next."
else
  start_local 0
  wait_ready
  log "Stop with: ./stop.sh"
fi
