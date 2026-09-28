#!/usr/bin/env bash
# cpu_py.sh LOG SCRIPT [ARGS...]: run a python script from the repo root in glm53-sm121-v11 (no GPU), with the
# mounts and env of evidence/e4-tools/cpu_tests.sh (repo at /r, HF cache read-only). Console to LOG.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG="$1"; shift
cd "$R" || exit 1
{
  echo "start $(date -u +%FT%TZ) git $(git rev-parse HEAD) image glm53-sm121-v11 $(docker image inspect -f '{{.Id}}' glm53-sm121-v11): $*"
  docker run --rm --entrypoint python3 -v "$R":/r -w /r -v "$HOME/.cache/huggingface":/root/.cache/huggingface:ro \
    -e GLM53_V11_SRC=/usr/local/lib/python3.12/dist-packages glm53-sm121-v11 "$@"
  echo "rc=$? end $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
tail -30 "$LOG"
