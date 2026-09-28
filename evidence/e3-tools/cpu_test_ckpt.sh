#!/usr/bin/env bash
# cpu_test_ckpt.sh LOG: the one checkpoint-dependent misc test. test_v13_misc.py hard-codes the host path
# /home/sfxnz/.cache/huggingface/hub/..., so the HF cache is mounted at that same path here.
R="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$1"
cd "$R" || exit 1
{
  echo "start $(date -u +%FT%TZ)"
  docker run --rm --entrypoint python3 -v "$R":/r -w /r -v "$HOME/.cache/huggingface":"$HOME/.cache/huggingface":ro \
    -e GLM53_V11_SRC=/usr/local/lib/python3.12/dist-packages glm53-sm121-v11 \
    -m unittest -v docker.test_v13_misc.ApplyTests.test_skip_prefix_matches_exactly_the_nvidia_mtp_layer
  echo "rc=$? end $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
cat "$LOG"
