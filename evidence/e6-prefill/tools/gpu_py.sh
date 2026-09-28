#!/usr/bin/env bash
# gpu_py.sh LOG SCRIPT [ARGS...]: run a python script from the repo root in glm53-sm121-v13 with --gpus all (no serve up),
# with the mounts of evidence/e4-tools/microbench.sh (repo at /work, HF cache read-only at /hf). Console to LOG.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG="$1"; shift
cd "$R" || exit 1
{
  echo "start $(date -u +%FT%TZ) git $(git rev-parse HEAD) image glm53-sm121-v13 $(docker image inspect -f '{{.Id}}' glm53-sm121-v13) host $(hostname): $*"
  echo "running containers: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
  docker run --rm --gpus all --entrypoint python3 -v "$R":/work -w /work \
    -v "$HOME/.cache/huggingface":/hf:ro -e HF_HUB_CACHE=/hf/hub glm53-sm121-v13 "$@"
  echo "rc=$? end $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
tail -30 "$LOG"
