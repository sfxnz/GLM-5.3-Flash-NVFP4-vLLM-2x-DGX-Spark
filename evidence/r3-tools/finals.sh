#!/usr/bin/env bash
main() {
R="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$R" || exit 1
STEPS=fast,E,T,count5
PR16_ENV=(IMAGE=glm53-sm121-v14 SNAPSHOT_REV=09b04e5e74bca08ca8549fc736d4cdd8624bfde3
       CHAT_TEMPLATE="$HOME/projects/data/glm53-r3/chat_template.r2.jinja"
       KDA_RECOVER=0 MLA_PLAN_NOSYNC=0 PREFIX_CACHE_DFLASH=0 VIDEO_FRAMES_FIX=0 KDA_TRIM=0)
for i in 1 2 3; do
  bash evidence/r3-tools/gate.sh 11 -- bash evidence/r3-tools/unit.sh "evidence/r3/final-a$i" "$STEPS" "${PR16_ENV[@]}"
  B_STEPS="$STEPS"; [[ $i == 3 ]] && B_STEPS="$STEPS,tier1=r3-final-1d00738"
  bash evidence/r3-tools/gate.sh 11 -- bash evidence/r3-tools/unit.sh "evidence/r3/final-b$i" "$B_STEPS"
done
echo "== $(date -u +%FT%TZ) finals done"
}
main "$@"; exit $?
