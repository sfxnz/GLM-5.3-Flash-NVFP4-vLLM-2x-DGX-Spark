#!/usr/bin/env bash
# cpu_tests.sh LOG: every docker/test_v13_*.py in glm53-sm121-v11 (torch + triton), repo at /r, HF cache at
# /root/.cache/huggingface. evidence/e3-tools/cpu_tests.sh still lists the removed test_v13_determinism.py.
R="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$1"
cd "$R" || exit 1
t0=$(date -u +%s)
tests=(docker/test_v13_*.py)
{
  echo "start $(date -u +%FT%TZ) git $(git rev-parse HEAD) tests: ${tests[*]}"
  docker run --rm --entrypoint python3 -v "$R":/r -w /r -v "$HOME/.cache/huggingface":/root/.cache/huggingface:ro \
    -e GLM53_V11_SRC=/usr/local/lib/python3.12/dist-packages glm53-sm121-v11 \
    -m unittest -v "${tests[@]}"
  echo "rc=$? end $(date -u +%FT%TZ) elapsed_s=$(( $(date -u +%s) - t0 ))"
} > "$LOG" 2>&1
tail -5 "$LOG"
