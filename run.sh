#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 on 2x DGX Spark (GB10) — vLLM TP=2
set -euo pipefail

# BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py
MODEL="${MODEL:-nvidia/GLM-5.3-Flash-NVFP4}"
SERVED_NAME="${SERVED_NAME:-$MODEL}"
IMAGE="${IMAGE:-glm53-sm121-v13}"
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
FORCE_UNSAFE_MOE="${FORCE_UNSAFE_MOE:-0}"
FORCE_UNSAFE_LINEAR="${FORCE_UNSAFE_LINEAR:-0}"
FORCE_UNSAFE_SPEC="${FORCE_UNSAFE_SPEC:-0}"
FORCE_UNSAFE_VISION="${FORCE_UNSAFE_VISION:-0}"
FORCE_UNSAFE_IMAGE="${FORCE_UNSAFE_IMAGE:-0}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
# vLLM default is 4 GiB of processed MM tensors in the head EngineCore (UMA).
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-1}"
# Server-wide output ceiling (--override-generation-config max_new_tokens).
# Without it one thinking-on request could decode for hours. 0 drops it.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-65536}"
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
# Official NVIDIA ModelOpt pin. LibertAI rollback is
# MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 SNAPSHOT_REV=caca4e6a4ebbd66f159d3d2fc256683fd6e27177
SNAPSHOT_REV="${SNAPSHOT_REV:-09b04e5e74bca08ca8549fc736d4cdd8624bfde3}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"
# --moe-backend covers routed experts only. --linear-backend covers the dense
# NVFP4 linears (nvidia pack layers 0-2 MLP); auto picks a JIT FP4 GEMM on sm_121.
LINEAR_BACKEND="${LINEAR_BACKEND:-marlin}"
REASONING_PARSER="${REASONING_PARSER:-glm45}"
DRAFT_MODEL="${DRAFT_MODEL:-incoai/GLM-5.3-Flash-DFlash2}"
# DFlash2 snapshot (full commit sha). bf582e4 (2026-08-31) and dc77ff1 (2026-08-28)
# are weights-only updates with the same config.json. E1b: bf582e4 moved no acceptance.
DRAFT_REV="${DRAFT_REV:-7d74cdd881ed7e32c31175984a67823127b66cfe}"
DRAFT_SNAPSHOT="${HF_CACHE}/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/${DRAFT_REV}"
DRAFT_SNAPSHOT_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/${DRAFT_REV}"
# SPEC picks the drafter: dflash2 (incoai DFlash2 block-diffusion draft, needs
# glm53-sm121-v11 or later) or mtp (GLM's native MTP head; LibertAI pack only,
# with ADAPTIVE_VERIFY=0).
SPEC="${SPEC:-dflash2}"
# JIT_CACHE=1 keeps the FlashInfer / Triton / TileLang / DeepGEMM / vLLM compile
# caches in JIT_CACHE_DIR/<image id>/ on each node, so later boots skip those
# JIT builds. JIT_CACHE=0 gives every boot an empty cache, as before.
JIT_CACHE="${JIT_CACHE:-1}"
JIT_CACHE_DIR="${JIT_CACHE_DIR:-$HOME/projects/data/glm53-jit-cache}"
# v13 switches (docker/README-v13.md). Each one that is on becomes GLM53_* env on
# both ranks, and needs a glm53-sm121-v13 image. Off adds nothing. v11 rollback:
# IMAGE=glm53-sm121-v11 DRAFT_WEIGHTS=bf16 TARGET_WEIGHT_GROUPS_INT8=none KPOOL_TAIL_FIX=0 ADAPTIVE_VERIFY=0.
# DRAFT_WEIGHTS: bf16, or nvfp4 for the DFlash2 drafter's linears in NVFP4 W4A16
# (GLM53_NVFP4_W4A16=draft). E2e: the target is untouched, step A -4 ms, Tier 1
# 856 vs 857 of 1010.
DRAFT_WEIGHTS="${DRAFT_WEIGHTS:-nvfp4}"
# Comma list of target groups in INT8 W8A16 (GLM53_INT8_W8A16): shared, mla,
# kda_o, kda_in, lm_head. none keeps them BF16. E4a over E3b: step A -17 ms,
# prose A +24.9%; Tier 0 at the cross-boot A/A; Tier 1 863 vs 857 of 1010.
TARGET_WEIGHT_GROUPS_INT8="${TARGET_WEIGHT_GROUPS_INT8:-shared,mla,kda_o,kda_in,lm_head}"
# Rows at or above which the swapped kda_in GEMM dequantizes to BF16 and runs
# cuBLAS instead of Marlin (GLM53_WQ_DEQUANT_MIN_M, GLM53_WQ_DEQUANT_GROUPS=kda_in).
# 0 is off. Values below 64 are refused: they would reach the captured decode graphs.
# E5: INT8 Marlin kda_in is 3.6x BF16 at the 1152-row prefill chunk, 93% of the
# -11% prefill. E6: 512 gave +6.4% / +3.5% at 32k / 128k, short of the +5% rule.
PREFILL_DEQUANT_MIN_M="${PREFILL_DEQUANT_MIN_M:-0}"
# 1: indexer tail ring sized for the verify window (GLM53_KPOOL_TAIL_FIX), so
# rejected drafts no longer write committed pool keys. E3a: needles pass to
# 128k, and decode-built pools match prefill within the prefill A/A.
KPOOL_TAIL_FIX="${KPOOL_TAIL_FIX:-1}"
# 1: verify only the leading drafts whose running DFlash2 confidence is at least
# ADAPTIVE_VERIFY_TAU, at fixed shapes (GLM53_ADAPTIVE_VERIFY, _TAU). Lossless.
# E3b at 0.2 vs E3a: prose A +14.9%, H +20.0%, B +5.3%, T +10.1%, J flat.
# E4b at 0.3 vs 0.2: A +5.2%, H +8.0%, B +4.3%, T +2.1%.
ADAPTIVE_VERIFY="${ADAPTIVE_VERIFY:-1}"
ADAPTIVE_VERIFY_TAU="${ADAPTIVE_VERIFY_TAU:-0.3}"
# END generated
hub_slug="models--${MODEL//\//--}"
SNAPSHOT="${SNAPSHOT:-${HF_CACHE}/hub/${hub_slug}/snapshots/${SNAPSHOT_REV}}"
SNAPSHOT_IN_CONTAINER="${SNAPSHOT_IN_CONTAINER:-${HF_HOME_IN_CONTAINER}/hub/${hub_slug}/snapshots/${SNAPSHOT_REV}}"
# Per-request MM cap. MM profiling still runs (one max-size video item, 32242-token encoder budget).
if [[ "$LANGUAGE_MODEL_ONLY" == "0" && -z "${LIMIT_MM_PER_PROMPT:-}" ]]; then
  LIMIT_MM_PER_PROMPT='{"image":4,"video":1}'
fi
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
# fp8 hybrid pool is 372,877 tokens (1.14x at 327,680) on the 4.14 GiB pin at DFlash2-7. Native 1,048,576 does not
# fit. Packed NVFP4 KV is a different image/backend, not MAX_MODEL_LEN on this pin.
if [[ "$KV_CACHE_DTYPE" == fp8_e4m3 && "$MAX_MODEL_LEN" -gt 327680 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "fp8 KV pin (372,877 tokens at DFlash2-7, 4.14 GiB) cannot hold --max-model-len $MAX_MODEL_LEN. A 1M request needs ~8.2 GiB of this hybrid layout and GB10 UMA OOMs above ~5.1 GiB. Do not advertise a window the pool cannot serve. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
# 327680 needs more than the displayed 3.62 GiB (3886945403 still estimates
# max len 327168). 3.0 GiB estimates 239616. 4.14 GiB is the known-good pin.
if [[ "$MAX_MODEL_LEN" -gt 239616 && "$KV_CACHE_MEMORY" -le 3886945403 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "KV pin $KV_CACHE_MEMORY cannot hold --max-model-len $MAX_MODEL_LEN (need more than 3.62 GiB; 3886945403 estimates max len 327168). Tony's 3.0 GiB pin is a 262144-ctx budget. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
# CUTLASS fused-MoE JIT exhausted the ~18 GiB left after 90.67 GiB weights.
if [[ "$MOE_BACKEND" != marlin && "$FORCE_UNSAFE_MOE" != 1 ]]; then
  echo "MOE_BACKEND=$MOE_BACKEND OOM'd spark2 during flashinfer_cutlass JIT after 90.67 GiB weights (global UMA, NV_ERR_NO_MEMORY). Stay on marlin. FORCE_UNSAFE_MOE=1 overrides." >&2
  exit 1
fi
if [[ "$LINEAR_BACKEND" != marlin && "$FORCE_UNSAFE_LINEAR" != 1 ]]; then
  echo "LINEAR_BACKEND=$LINEAR_BACKEND: the nvidia pack's layers 0-2 dense MLP is NVFP4, and non-Marlin NVFP4 GEMMs JIT-compile on sm_121 during the first profile forward. Six 2026-09-16 boots collapsed there. Stay on marlin. FORCE_UNSAFE_LINEAR=1 overrides." >&2
  exit 1
fi
if [[ ! "$MAX_NEW_TOKENS" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS: want a positive integer, or 0 to drop the ceiling." >&2
  exit 1
fi
if [[ "$SPEC" == mtp && "$MODEL" == nvidia/GLM-5.3-Flash-NVFP4 && "$FORCE_UNSAFE_SPEC" != 1 ]]; then
  echo "SPEC=mtp on $MODEL: its layer-45 MTP weights are 13.84 GiB BF16 and not in the quant ignore list, so they cannot load or fit. MTP rollback is the LibertAI pack: MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 SNAPSHOT_REV=caca4e6a4ebbd66f159d3d2fc256683fd6e27177 SPEC=mtp ADAPTIVE_VERIFY=0. FORCE_UNSAFE_SPEC=1 overrides." >&2
  exit 1
fi
if [[ "$LANGUAGE_MODEL_ONLY" != 0 && "$LANGUAGE_MODEL_ONLY" != 1 ]]; then
  echo "LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY: want exactly 0 or 1." >&2
  exit 1
fi
# The draft path is snapshots/<rev>, and the hub names snapshot dirs by full commit sha only.
if [[ ! "$DRAFT_REV" =~ ^[0-9a-f]{40}$ ]]; then
  echo "DRAFT_REV=$DRAFT_REV: want a full 40-hex commit sha (the draft path is snapshots/\$DRAFT_REV)." >&2
  exit 1
fi
if [[ "$JIT_CACHE" != 0 && "$JIT_CACHE" != 1 ]]; then
  echo "JIT_CACHE=$JIT_CACHE: want exactly 0 or 1." >&2
  exit 1
fi
if [[ "$LANGUAGE_MODEL_ONLY" != 0 && "$FORCE_UNSAFE_VISION" != 1 ]]; then
  echo "LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY hides the native GLM-5.3-Flash vision tower. The NVIDIA pack ships vision_config and processor_config.json. Leave LANGUAGE_MODEL_ONLY=0. FORCE_UNSAFE_VISION=1 overrides." >&2
  exit 1
fi
if [[ "$DRAFT_WEIGHTS" != bf16 && "$DRAFT_WEIGHTS" != nvfp4 ]]; then
  echo "DRAFT_WEIGHTS=$DRAFT_WEIGHTS: want bf16 or nvfp4." >&2
  exit 1
fi
# docker/patch_v13_fp8.py GROUPS, less draft: DRAFT_WEIGHTS owns the drafter.
# "none" keeps every target group BF16 (an empty value falls back to the default).
int8_groups=()
if [[ "$TARGET_WEIGHT_GROUPS_INT8" != none ]]; then
  IFS=, read -r -a int8_groups <<<"$TARGET_WEIGHT_GROUPS_INT8"
fi
for group in "${int8_groups[@]}"; do
  case "$group" in
    shared | mla | kda_o | kda_in | lm_head) ;;
    draft)
      echo "TARGET_WEIGHT_GROUPS_INT8=$TARGET_WEIGHT_GROUPS_INT8 lists draft, the drafter's group. DRAFT_WEIGHTS sets the drafter's weights; list target groups only (shared,mla,kda_o,kda_in,lm_head)." >&2
      exit 1
      ;;
    *)
      echo "TARGET_WEIGHT_GROUPS_INT8=$TARGET_WEIGHT_GROUPS_INT8: unknown group '$group'. Want a comma list of shared,mla,kda_o,kda_in,lm_head (docker/patch_v13_fp8.py)." >&2
      exit 1
      ;;
  esac
done
if [[ ! "$PREFILL_DEQUANT_MIN_M" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "PREFILL_DEQUANT_MIN_M=$PREFILL_DEQUANT_MIN_M: want a positive integer (rows), or 0 for off." >&2
  exit 1
fi
# Capture sizes reach 16 (24 on the four-way rollback); keep dequant out of decode graphs.
if (( PREFILL_DEQUANT_MIN_M != 0 && PREFILL_DEQUANT_MIN_M < 64 )); then
  echo "PREFILL_DEQUANT_MIN_M=$PREFILL_DEQUANT_MIN_M is below 64 and would dequantize inside captured decode steps. Use 0 (off) or at least 64." >&2
  exit 1
fi
if [[ "$KPOOL_TAIL_FIX" != 0 && "$KPOOL_TAIL_FIX" != 1 ]]; then
  echo "KPOOL_TAIL_FIX=$KPOOL_TAIL_FIX: want exactly 0 or 1." >&2
  exit 1
fi
if [[ "$ADAPTIVE_VERIFY" != 0 && "$ADAPTIVE_VERIFY" != 1 ]]; then
  echo "ADAPTIVE_VERIFY=$ADAPTIVE_VERIFY: want exactly 0 or 1." >&2
  exit 1
fi
if [[ ! "$ADAPTIVE_VERIFY_TAU" =~ ^0?\.[0-9]*[1-9][0-9]*$ ]]; then
  echo "ADAPTIVE_VERIFY_TAU=$ADAPTIVE_VERIFY_TAU: want a decimal strictly between 0 and 1, e.g. 0.2." >&2
  exit 1
fi
if [[ "$ADAPTIVE_VERIFY" == 1 && "$SPEC" != dflash2 ]]; then
  echo "ADAPTIVE_VERIFY=1 needs SPEC=dflash2: the verify width comes from the DFlash2 drafter's selector scores (docker/README-v13.md). Set ADAPTIVE_VERIFY=0 with SPEC=$SPEC." >&2
  exit 1
fi
# GLM53_* env for both ranks, from the knobs above.
glm53_env=()
if [[ "$DRAFT_WEIGHTS" == nvfp4 ]]; then glm53_env+=(GLM53_NVFP4_W4A16=draft); fi
if (( ${#int8_groups[@]} > 0 )); then glm53_env+=("GLM53_INT8_W8A16=$TARGET_WEIGHT_GROUPS_INT8"); fi
if [[ "$PREFILL_DEQUANT_MIN_M" != 0 ]]; then
  glm53_env+=("GLM53_WQ_DEQUANT_MIN_M=$PREFILL_DEQUANT_MIN_M" GLM53_WQ_DEQUANT_GROUPS=kda_in)
fi
if [[ "$KPOOL_TAIL_FIX" == 1 ]]; then glm53_env+=(GLM53_KPOOL_TAIL_FIX=1); fi
if [[ "$ADAPTIVE_VERIFY" == 1 ]]; then glm53_env+=(GLM53_ADAPTIVE_VERIFY=1 "GLM53_ADAPTIVE_VERIFY_TAU=$ADAPTIVE_VERIFY_TAU"); fi
# Older images do not read GLM53_*, so a switch there would silently do nothing.
if (( ${#glm53_env[@]} > 0 )) && [[ "$IMAGE" != glm53-sm121-v13* && "$FORCE_UNSAFE_IMAGE" != 1 ]]; then
  echo "IMAGE=$IMAGE is not a glm53-sm121-v13 image and would ignore ${glm53_env[*]}. The v11 rollback turns the switches off: IMAGE=glm53-sm121-v11 DRAFT_WEIGHTS=bf16 TARGET_WEIGHT_GROUPS_INT8=none KPOOL_TAIL_FIX=0 ADAPTIVE_VERIFY=0. FORCE_UNSAFE_IMAGE=1 overrides." >&2
  exit 1
fi
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
# Extra vllm serve args, word-split on purpose (e.g. "--load-format dummy").
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Extra container env on both ranks: space-separated NAME=VALUE pairs, e.g.
# EXTRA_ENV='MAX_JOBS=2 NCCL_DEBUG=INFO'. Engine/runtime names only.
EXTRA_ENV="${EXTRA_ENV:-}"
extra_env_args=()
jit_verbose="" jit_debug=""
read -r -a extra_env_pairs <<<"$EXTRA_ENV"
for pair in "${extra_env_pairs[@]}"; do
  case "$pair" in
    FLASHINFER_JIT_VERBOSE=*) jit_verbose="${pair#*=}" ;;
    FLASHINFER_JIT_DEBUG=*) jit_debug="${pair#*=}" ;;
  esac
  name="${pair%%=*}"
  [[ "$pair" == *=* ]] || name="(an entry without =)"
  if [[ "$pair" != *=* || "$name" =~ TOKEN|KEY|SECRET ]] ||
    ! [[ "$name" =~ ^(NCCL|VLLM|PYTORCH|TORCH|CUDA|OMP|FLASHINFER|TRITON|TILELANG|GLM53)_[A-Z0-9_]+$ || "$name" == MAX_JOBS ]]; then
    echo "EXTRA_ENV refuses '$name': want NAME=VALUE with NAME matching ^(NCCL|VLLM|PYTORCH|TORCH|CUDA|OMP|FLASHINFER|TRITON|TILELANG|GLM53)_[A-Z0-9_]+\$ or MAX_JOBS, and no TOKEN, KEY or SECRET in the name." >&2
    exit 1
  fi
  case "$name" in
    GLM53_NVFP4_W4A16) knob=DRAFT_WEIGHTS ;;
    GLM53_INT8_W8A16) knob=TARGET_WEIGHT_GROUPS_INT8 ;;
    GLM53_WQ_DEQUANT_MIN_M | GLM53_WQ_DEQUANT_GROUPS) knob=PREFILL_DEQUANT_MIN_M ;;
    GLM53_KPOOL_TAIL_FIX) knob=KPOOL_TAIL_FIX ;;
    GLM53_ADAPTIVE_VERIFY) knob=ADAPTIVE_VERIFY ;;
    GLM53_ADAPTIVE_VERIFY_TAU) knob=ADAPTIVE_VERIFY_TAU ;;
    *) knob="" ;;
  esac
  if [[ -n "$knob" ]]; then
    echo "EXTRA_ENV sets $name, which run.sh sets from $knob on both ranks. Set $knob instead, so one setting owns $name." >&2
    exit 1
  fi
  extra_env_args+=(-e "$pair")
done
# This image's FlashInfer reads FLASHINFER_JIT_VERBOSE=1 as FLASHINFER_JIT_DEBUG=1 when DEBUG
# is unset (flashinfer/jit/core.py:525-528): every JIT kernel builds -O0 --device-debug.
if [[ "$jit_verbose" == 1 && "$jit_debug" != 0 ]]; then
  echo "EXTRA_ENV FLASHINFER_JIT_VERBOSE=1 without FLASHINFER_JIT_DEBUG=0 builds every FlashInfer JIT kernel -O0 --device-debug (flashinfer/jit/core.py:525-528). The serving kernels would be debug builds, and on 2026-09-27 the topk ptxas grew past 3.28 GiB and PROFILE fell under the 8 GiB floor. Add FLASHINFER_JIT_DEBUG=0 to keep verbose ninja output with -O3 builds." >&2
  exit 1
fi

log() { printf '==> %s\n' "$*"; }

# One mount at /jit-cache. Each engine's own cache variable points into it
# (checked against the v11 source; the image runs as root with HOME=/root):
#   FLASHINFER_WORKSPACE_BASE  flashinfer/jit/env.py: <base>/.cache/flashinfer, base defaults to ~
#   VLLM_CACHE_ROOT            vllm/envs.py: torch_compile_cache, flashinfer_autotune_cache
#   DG_JIT_CACHE_DIR           vllm/utils/deep_gemm.py defaults it to $VLLM_CACHE_ROOT/deep_gemm
#   TRITON_CACHE_DIR           Triton default ~/.triton/cache (Triton kernels, autotune results)
#   TILELANG_CACHE_DIR         TileLang default ~/.tilelang/cache (mHC kernels)
jit_args=()
set_jit_args() {
  jit_args=()
  [[ "$JIT_CACHE" == 1 ]] || return 0
  jit_args=(
    -v "$1:/jit-cache"
    -e FLASHINFER_WORKSPACE_BASE=/jit-cache/flashinfer
    -e VLLM_CACHE_ROOT=/jit-cache/vllm
    -e DG_JIT_CACHE_DIR=/jit-cache/vllm/deep_gemm
    -e TRITON_CACHE_DIR=/jit-cache/triton
    -e TILELANG_CACHE_DIR=/jit-cache/tilelang
  )
}

# Keyed by this node's image ID (first 12 hex digits), so a new image never
# reuses kernels built by an old one.
jit_cache_host_dir() {
  local id
  id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
  id="${id#sha256:}"
  printf '%s/%s\n' "$JIT_CACHE_DIR" "${id:0:12}"
}

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
    echo "Image $IMAGE not found. Build the local image chain through glm53-sm121-v13 first (see README). Do not use stock vllm/vllm-openai on sm_121." >&2
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
    log "Downloading $MODEL @ $SNAPSHOT_REV (resumes under $HF_CACHE)"
    "$HF" download "$MODEL" --revision "$SNAPSHOT_REV"
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
    log "Downloading $DRAFT_MODEL @ $DRAFT_REV (resumes under $HF_CACHE)"
    "$HF" download "$DRAFT_MODEL" --revision "$DRAFT_REV"
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
  stop_local
  maybe_drop_caches
  ensure_image
  ensure_weights

  local serve_model
  serve_model="$(resolve_model)"

  if [[ "$JIT_CACHE" == 1 ]]; then
    local jit_dir
    jit_dir="$(jit_cache_host_dir)"
    mkdir -p "$jit_dir"
    log "JIT cache $jit_dir -> /jit-cache"
    set_jit_args "$jit_dir"
  fi

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
  local pair
  for pair in "${glm53_env[@]}"; do
    env_args+=(-e "$pair")
  done
  env_args+=("${extra_env_args[@]}")
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
  local mm_args=()
  if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
    mm_args+=(--language-model-only)
  else
    mm_args+=(--mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB")
    if [[ -n "${LIMIT_MM_PER_PROMPT:-}" ]]; then
      mm_args+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
    fi
  fi
  local gen_args=()
  if [[ "$MAX_NEW_TOKENS" != 0 ]]; then
    gen_args+=(--override-generation-config "{\"max_new_tokens\": $MAX_NEW_TOKENS}")
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
    "${jit_args[@]}" \
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
    --linear-backend "$LINEAR_BACKEND" \
    --speculative-config "$SPEC_CONFIG" \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --reasoning-parser "$REASONING_PARSER" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    "${template_args[@]}" \
    "${mm_args[@]}" \
    "${gen_args[@]}" \
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

# One list drives the worker launch: every variable rank 1 needs to build the
# same serve as rank 0 (all generated defaults plus the derived values).
FORWARD_ENVS=(
  MODEL SERVED_NAME IMAGE CONTAINER_NAME PORT MASTER_PORT HEAD_IP WORKER_HOST IFACE HCA TP NNODES
  MAX_MODEL_LEN MAX_NUM_SEQS UTIL KV_CACHE_DTYPE NUM_SPECULATIVE_TOKENS MAX_NUM_BATCHED_TOKENS
  FORCE_UNSAFE_CTX FORCE_UNSAFE_MOE FORCE_UNSAFE_LINEAR FORCE_UNSAFE_SPEC FORCE_UNSAFE_VISION FORCE_UNSAFE_IMAGE
  LANGUAGE_MODEL_ONLY MM_PROCESSOR_CACHE_GB MAX_NEW_TOKENS VLLM_USE_BREAKABLE_CUDAGRAPH CHAT_TEMPLATE
  KV_CACHE_MEMORY BLOCK_SIZE HF_CACHE SNAPSHOT_REV MOE_BACKEND LINEAR_BACKEND REASONING_PARSER
  DRAFT_MODEL DRAFT_REV SPEC JIT_CACHE JIT_CACHE_DIR
  DRAFT_WEIGHTS TARGET_WEIGHT_GROUPS_INT8 PREFILL_DEQUANT_MIN_M KPOOL_TAIL_FIX ADAPTIVE_VERIFY ADAPTIVE_VERIFY_TAU
  SNAPSHOT SNAPSHOT_IN_CONTAINER LIMIT_MM_PER_PROMPT HF_HUB_DISABLE_XET SPEC_CONFIG ENFORCE_EAGER
  COMPILATION_CONFIG SKIP_DOWNLOAD EXTRA_ARGS EXTRA_ENV
)

# The ssh command that starts rank 1. printf %q keeps quotes and JSON intact
# (the worker's login shell must be bash).
worker_command() {
  local words=(env ROLE=worker ORCHESTRATE=0) v
  for v in "${FORWARD_ENVS[@]}"; do
    words+=("$v=${!v-}")
  done
  printf '%q ' "${words[@]}"
  printf 'bash /tmp/glm53-run.sh\n'
}

worker_ssh_ok() {
  command -v ssh >/dev/null 2>&1 && ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1
}

# TP ranks must run identical bits. Separate builds give different image IDs,
# so warn (do not refuse) and print the sync command.
check_image_parity() {
  local local_id="" remote_id=""
  if command -v docker >/dev/null 2>&1; then
    local_id="$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
  fi
  remote_id="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" \
    "docker image inspect -f '{{.Id}}' $(printf '%q' "$IMAGE")" 2>/dev/null || true)"
  if [[ -n "$local_id" && "$local_id" == "$remote_id" ]]; then
    log "Image parity OK: $IMAGE is $local_id on $(host_short) and $WORKER_HOST"
  else
    echo "WARN image $IMAGE differs: $(host_short)=${local_id:-missing} $WORKER_HOST=${remote_id:-missing}. Sync from the head with: docker save $IMAGE | ssh $WORKER_HOST docker load" >&2
  fi
}

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf '==> validate-only spec=%s seqs=%s spec_tokens=%s eager=%s compilation=%s snapshot=%s draft_rev=%s moe=%s linear=%s served=%s mm_cache_gb=%s max_new_tokens=%s\n' \
    "$SPEC" "$MAX_NUM_SEQS" "$NUM_SPECULATIVE_TOKENS" "$ENFORCE_EAGER" "$COMPILATION_CONFIG" \
    "$SNAPSHOT_REV" "$DRAFT_REV" "$MOE_BACKEND" "$LINEAR_BACKEND" "$SERVED_NAME" "$MM_PROCESSOR_CACHE_GB" "$MAX_NEW_TOKENS"
  # The real key is each node's own image ID; validate-only does not call docker.
  set_jit_args "$JIT_CACHE_DIR/<image-id>"
  printf '==> jit_cache=%s args: %s\n' "$JIT_CACHE" "${jit_args[*]}"
  # Every GLM53_* the containers get: the knobs' first, then EXTRA_ENV's.
  shown=("${glm53_env[@]}")
  for pair in "${extra_env_pairs[@]}"; do
    if [[ "$pair" == GLM53_* ]]; then shown+=("$pair"); fi
  done
  printf '==> glm53_env: %s\n' "${shown[*]}"
  printf '==> worker command: %s' "$(worker_command)"
  echo
  if [[ "$ORCHESTRATE" == auto && "$(detect_role)" == head ]] && worker_ssh_ok; then
    check_image_parity
  fi
  exit 0
fi

ROLE="$(detect_role)"
log "role=$ROLE host=$(host_short)"

if [[ "$ORCHESTRATE" == "auto" && "$ROLE" == "head" ]]; then
  if worker_ssh_ok; then
    check_image_parity
    log "Starting worker on $WORKER_HOST first"
    scp -q "$0" "${WORKER_HOST}:/tmp/glm53-run.sh"
    ssh "$WORKER_HOST" "$(worker_command)"
    log "Worker container started. Waiting 25s for NCCL listen, then starting head"
    sleep 25
  else
    echo "Cannot SSH to $WORKER_HOST. ORCHESTRATE=auto refuses a lone TP=2 head rank. Set ROLE=worker on the other Spark first, or fix SSH." >&2
    exit 1
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
