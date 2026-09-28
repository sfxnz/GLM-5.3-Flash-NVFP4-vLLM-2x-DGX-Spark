#!/usr/bin/env bash
# f3_cfg.sh DIR RUN validate|boot: the E5 F3 census + torch-profiler serve, i.e. the recipe defaults (plain ./run.sh)
# plus EXTRA_ENV="GLM53_EXPERT_CENSUS=/cache/huggingface/glm53-census/RUN" and
# EXTRA_ARGS="--profiler-config $(tools/nsys_step.sh --print-config)" (tools/README.md).
#   validate -> VALIDATE_ONLY=1 ./run.sh, DIR/validate-only.txt
#   boot     -> evidence/e1-tools/boot.sh DIR ... detached (setsid; kit/uma_watch.py), console in DIR/boot.out
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$1"; RUN="$2"; MODE="$3"
mkdir -p "$D"
D="$(cd "$D" && pwd)"
cd "$R" || exit 1
ENVX="GLM53_EXPERT_CENSUS=/cache/huggingface/glm53-census/$RUN"
ARGX="--profiler-config $(tools/nsys_step.sh --print-config)"
echo "EXTRA_ENV=$ENVX"
echo "EXTRA_ARGS=$ARGX"
case "$MODE" in
  validate) EXTRA_ENV="$ENVX" EXTRA_ARGS="$ARGX" VALIDATE_ONLY=1 ./run.sh > "$D/validate-only.txt" 2>&1
            echo "validate rc=$?" ;;
  boot) setsid nohup bash evidence/e1-tools/boot.sh "$D" EXTRA_ENV="$ENVX" EXTRA_ARGS="$ARGX" \
          > "$D/boot.out" 2>&1 < /dev/null &
        echo "boot.sh started pid $!" ;;
  *) echo "usage: $0 DIR RUN validate|boot" >&2; exit 2 ;;
esac
