#!/usr/bin/env bash
# cfg.sh DIR validate|boot EXTRA_ENV: E4 serve on glm53-sm121-v13 (5-patch build) with the given EXTRA_ENV string.
#   validate -> VALIDATE_ONLY=1 ./run.sh, tee DIR/validate-only.txt
#   boot     -> evidence/e1-tools/boot.sh DIR ... detached (setsid; kit/uma_watch.py), console in DIR/boot.out
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; MODE="$2"; ENVX="$3"
mkdir -p "$D"
D="$(cd "$D" && pwd)"
cd "$R" || exit 1
echo "EXTRA_ENV=$ENVX"
case "$MODE" in
  validate) IMAGE=glm53-sm121-v13 EXTRA_ENV="$ENVX" VALIDATE_ONLY=1 ./run.sh 2>&1 | tee "$D/validate-only.txt" ;;
  boot) setsid nohup bash evidence/e1-tools/boot.sh "$D" IMAGE=glm53-sm121-v13 EXTRA_ENV="$ENVX" \
          > "$D/boot.out" 2>&1 < /dev/null &
        echo "boot.sh started pid $!" ;;
  *) echo "usage: $0 DIR validate|boot EXTRA_ENV" >&2; exit 2 ;;
esac
