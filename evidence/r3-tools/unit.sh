#!/usr/bin/env bash
# unit.sh DIR STEPS [NAME=VALUE ...]: one r3 boot. VALIDATE_ONLY, boot.sh (uma_watch + ./run.sh with the env), then
# the comma-separated STEPS in order, then memory snapshots, the head log and a stop of both nodes.
# Steps: probes  count5  videomp4  fast  E  T  vision  prefix  utf8ab  t0cmp=<ref>  t0rec=<name>  think=<name>  tier1=<name>  hold (skip the stop).
# The body is one function so bash parses it whole before running: editing this file mid-boot cannot shift a
# running copy.
main() {
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; STEPS="$2"; shift 2
cd "$R" || exit 1
T=evidence/e1-tools
EVALS="$HOME/projects/data/glm53-evals"
mkdir -p "$D"
echo "== $(date -u +%FT%TZ) unit $D steps=$STEPS env: $*"
VALIDATE_ONLY=1 env "$@" ./run.sh > "$D/validate-only.txt" 2>&1 || { echo "validate failed"; tail -5 "$D/validate-only.txt"; exit 1; }
bash $T/boot.sh "$D" "$@"
tail -2 "$D/run.log"
hold=0
if curl -sf http://127.0.0.1:8000/v1/models > /dev/null; then
  bash evidence/e2-tools/nodes.sh "after ready" > "$D/ready-mem.txt" 2>&1
  { docker exec glm53-flash-nvfp4 env; ssh 10.100.8.2 docker exec glm53-flash-nvfp4 env | sed "s/^/worker: /"; } 2>&1 | grep GLM53_ > "$D/glm53-env.txt"
  IFS=, read -ra steps <<< "$STEPS"
  for s in "${steps[@]}"; do
    arg="${s#*=}"
    { echo "== $(date -u +%FT%TZ) before $s: $(uptime)"; ps -eo pcpu,comm --sort=-pcpu | sed -n 2,6p; } >> "$D/load.log"
    case "$s" in
      probes)  bash $T/act.sh "$D" probe-count python3 $T/probes.py ;;
      fast)    bash $T/act.sh "$D" bench-fast python3 bench_decode.py --remote-meminfo --out "$D/bench-fast" ;;
      E|T)     bash $T/act.sh "$D" "bench-$s" python3 bench_decode.py --cells "$s" --remote-meminfo --out "$D/bench-$s" ;;
      vision)  bash $T/act.sh "$D" smoke-vision python3 smoke_vision.py ;;
      count5)  bash $T/act.sh "$D" count5 python3 evidence/r3-tools/count_repeat.py 5 "$D/count5.json" ;;
      videomp4) bash $T/act.sh "$D" video-probe python3 evidence/r3-tools/video_probe.py "$HOME/projects/data/glm53-r3/mm/clip20.mp4" "$D/video.json" ;;
      utf8ab)  bash $T/act.sh "$D" utf8-ab python3 evidence/r3-tools/utf8_ab.py "$D/utf8-ab-texts.json" ;;
      prefix)  bash $T/act.sh "$D" prefix-probe python3 evidence/r3-tools/prefix_probe.py --out "$D/prefix.json" ;;
      t0cmp=*) bash $T/act.sh "$D" tier0-compare python3 quality/tier0.py compare --ref "$arg" --stage fp8 --out "$D/tier0.json" ;;
      t0rec=*) bash $T/act.sh "$D" tier0-record python3 quality/tier0.py record --name "$arg"
               (cd "$EVALS/ref" && sha256sum "$arg.json") > "$D/tier0-ref-$arg.sha256" 2>/dev/null ;;
      think=*) mkdir -p "$EVALS/think/$arg"
               bash $T/act.sh "$D" think python3 quality/tier1.py run --tasks gsm8k,mmlu_pro --limit 50 \
                 --chat-kwargs '{"enable_thinking": true}' --think-budget 8192 \
                 --out "$EVALS/think/$arg/think.jsonl" --label "$arg"
               cp "$EVALS/think/$arg/think.summary.json" "$D/" 2>/dev/null ;;
      tier1=*) bash evidence/e2-tools/tier1.sh "$D" "$arg" ;;
      hold)    hold=1 ;;
      *)       echo "unknown step $s" ;;
    esac
  done
  bash evidence/e2-tools/nodes.sh "after steps" > "$D/end-mem.txt" 2>&1
  docker logs glm53-flash-nvfp4 > "$D/head.log" 2>&1
  python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
fi
if [[ $hold == 0 ]]; then
  bash evidence/e2-tools/stop_serve.sh "$D" stop "$(cat "$D/watcher.pid" 2>/dev/null)" > /dev/null 2>&1
fi
echo "== $(date -u +%FT%TZ) unit done (hold=$hold)"
grep -h "PASS\|FAIL" "$D/probe-count.console.txt" 2>/dev/null | tail -1
}
main "$@"; exit $?
