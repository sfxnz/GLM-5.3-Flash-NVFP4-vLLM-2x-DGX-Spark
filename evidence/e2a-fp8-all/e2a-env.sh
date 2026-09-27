#!/usr/bin/env bash
# E2a serve settings: glm53-sm121-v13, FP8 W8A16 Marlin on every BF16 non-MoE group
# (docker/patch_v13_fp8.py), warm JIT cache, no shard warmer, default drafter 7d74cdd.
#   bash e2a-env.sh validate   -> VALIDATE_ONLY=1 ./run.sh
#   bash e2a-env.sh boot       -> evidence/e1-tools/boot.sh on this dir
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$(cd "$(dirname "$0")" && pwd)"
ENVX='MAX_JOBS=2 GLM53_FP8_W8A16=draft,shared,mla,kda_o,kda_in,lm_head'
cd "$R" || exit 1
case "$1" in
  validate) IMAGE=glm53-sm121-v13 WARM_SHARDS=0 EXTRA_ENV="$ENVX" VALIDATE_ONLY=1 ./run.sh 2>&1 | tee "$D/validate-only.txt" ;;
  boot) setsid nohup bash evidence/e1-tools/boot.sh "$D" IMAGE=glm53-sm121-v13 WARM_SHARDS=0 EXTRA_ENV="$ENVX" \
          > "$D/boot.out" 2>&1 < /dev/null &
        echo "boot.sh started pid $!" ;;
  *) echo "usage: $0 validate|boot" >&2; exit 2 ;;
esac
