#!/usr/bin/env bash
# cfg.sh DIR FP8_GROUPS NVFP4_GROUPS validate|boot
# E2 serve: glm53-sm121-v13, WARM_SHARDS=0, default drafter, EXTRA_ENV='MAX_JOBS=2' plus
# GLM53_FP8_W8A16=FP8_GROUPS and/or GLM53_NVFP4_W4A16=NVFP4_GROUPS (either may be "-" for unset).
#   validate -> VALIDATE_ONLY=1 ./run.sh, tee DIR/validate-only.txt
#   boot     -> evidence/e1-tools/boot.sh DIR ... detached (setsid), console in DIR/boot.out
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; FP8="$2"; FP4="$3"; MODE="$4"
mkdir -p "$D"
D="$(cd "$D" && pwd)"
ENVX='MAX_JOBS=2'
[[ "$FP8" != - ]] && ENVX+=" GLM53_FP8_W8A16=$FP8"
[[ "$FP4" != - ]] && ENVX+=" GLM53_NVFP4_W4A16=$FP4"
cd "$R" || exit 1
echo "EXTRA_ENV=$ENVX"
case "$MODE" in
  validate) IMAGE=glm53-sm121-v13 WARM_SHARDS=0 EXTRA_ENV="$ENVX" VALIDATE_ONLY=1 ./run.sh 2>&1 | tee "$D/validate-only.txt" ;;
  boot) setsid nohup bash evidence/e1-tools/boot.sh "$D" IMAGE=glm53-sm121-v13 WARM_SHARDS=0 EXTRA_ENV="$ENVX" \
          > "$D/boot.out" 2>&1 < /dev/null &
        echo "boot.sh started pid $!" ;;
  *) echo "usage: $0 DIR FP8_GROUPS NVFP4_GROUPS validate|boot" >&2; exit 2 ;;
esac
