#!/usr/bin/env bash
# skill_probes.sh DIR: the legacy verify-glm53-flash skill probes on the running serve, each through
# evidence/e1-tools/act.sh (console + activity.tsv): smoke.sh (README curl), thinking_off_probe.py,
# tool_call_probe.py, needle_probe.py --prompt-tokens 8192 (unique salt). smoke.sh writes its request/response into
# .cursor/skills/verify-glm53-flash/artifacts/serve-smoke/<stamp>; that directory is moved to DIR/skill-smoke.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$1"
cd "$R" || exit 1
T=evidence/e1-tools
S=.cursor/skills/verify-glm53-flash/scripts
bash $T/act.sh "$D" skill-smoke bash $S/smoke.sh
ev="$(sed -n 's/^evidence=//p' "$D/skill-smoke.console.txt")"
if [[ -n "$ev" && -d "$ev" ]]; then
  rm -rf "$D/skill-smoke"
  mv "$ev" "$D/skill-smoke" && echo "moved $ev -> $D/skill-smoke"
fi
bash $T/act.sh "$D" skill-thinking-off python3 $S/thinking_off_probe.py
bash $T/act.sh "$D" skill-tool-call python3 $S/tool_call_probe.py
bash $T/act.sh "$D" skill-needle-8k python3 $S/needle_probe.py --prompt-tokens 8192
