#!/usr/bin/env bash
# cpu_tests.sh LOG: the six v13 CPU test files in glm53-sm121-v11 (torch + triton), repo mounted at /r.
R="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$1"
cd "$R" || exit 1
t0=$(date -u +%s)
{
  echo "start $(date -u +%FT%TZ) git $(git rev-parse HEAD)"
  docker run --rm --entrypoint python3 -v "$R":/r -w /r -v "$HOME/.cache/huggingface":/root/.cache/huggingface:ro \
    -e GLM53_V11_SRC=/usr/local/lib/python3.12/dist-packages glm53-sm121-v11 \
    -m unittest -v docker/test_v13_misc.py docker/test_v13_fp8.py docker/test_v13_census.py \
      docker/test_v13_determinism.py docker/test_v13_verify.py docker/test_v13_kpool_tail.py
  echo "rc=$? end $(date -u +%FT%TZ) elapsed_s=$(( $(date -u +%s) - t0 ))"
} > "$LOG" 2>&1
tail -5 "$LOG"
