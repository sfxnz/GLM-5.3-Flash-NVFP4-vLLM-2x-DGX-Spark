#!/usr/bin/env bash
# cfg.sh DIR validate|boot [EXTRA GLM53_* pairs...]
# E3 serve: glm53-sm121-v13 (6-patch build), EXTRA_ENV='MAX_JOBS=2 GLM53_NVFP4_W4A16=draft GLM53_KPOOL_TAIL_FIX=1
# GLM53_DETERMINISTIC_MLA_INDEX=1' plus any extra pairs (E3b: GLM53_ADAPTIVE_VERIFY=1 GLM53_ADAPTIVE_VERIFY_TAU=<t>).
#   validate -> VALIDATE_ONLY=1 ./run.sh, tee DIR/validate-only.txt
#   boot     -> evidence/e1-tools/boot.sh DIR ... detached (setsid; kit/uma_watch.py), console in DIR/boot.out
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; MODE="$2"; shift 2
mkdir -p "$D"
D="$(cd "$D" && pwd)"
ENVX="MAX_JOBS=2 GLM53_NVFP4_W4A16=draft GLM53_KPOOL_TAIL_FIX=1 GLM53_DETERMINISTIC_MLA_INDEX=1"
for kv in "$@"; do ENVX+=" $kv"; done
cd "$R" || exit 1
echo "EXTRA_ENV=$ENVX"
case "$MODE" in
  validate) IMAGE=glm53-sm121-v13 EXTRA_ENV="$ENVX" VALIDATE_ONLY=1 ./run.sh 2>&1 | tee "$D/validate-only.txt" ;;
  boot) setsid nohup bash evidence/e1-tools/boot.sh "$D" IMAGE=glm53-sm121-v13 EXTRA_ENV="$ENVX" \
          > "$D/boot.out" 2>&1 < /dev/null &
        echo "boot.sh started pid $!" ;;
  *) echo "usage: $0 DIR validate|boot [NAME=VALUE ...]" >&2; exit 2 ;;
esac
