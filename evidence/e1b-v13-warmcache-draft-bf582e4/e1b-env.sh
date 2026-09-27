#!/usr/bin/env bash
# E1b serve settings: E1a (IMAGE=glm53-sm121-v13, EXTRA_ENV=MAX_JOBS=2, all GLM53_* unset,
# JIT_CACHE=1 now warm, WARM_SHARDS=1) plus the newer DFlash2 drafter via SPEC_CONFIG.
#   bash e1b-env.sh validate   -> VALIDATE_ONLY=1 ./run.sh
#   bash e1b-env.sh boot       -> evidence/e1-tools/boot.sh on this dir
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$(cd "$(dirname "$0")" && pwd)"
SPEC='{"method":"dflash","model":"/cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a","num_speculative_tokens":7}'
cd "$R" || exit 1
case "$1" in
  validate) IMAGE=glm53-sm121-v13 EXTRA_ENV=MAX_JOBS=2 SPEC_CONFIG="$SPEC" VALIDATE_ONLY=1 ./run.sh ;;
  boot) bash evidence/e1-tools/boot.sh "$D" IMAGE=glm53-sm121-v13 EXTRA_ENV=MAX_JOBS=2 SPEC_CONFIG="$SPEC" ;;
  *) echo "usage: $0 validate|boot" >&2; exit 2 ;;
esac
