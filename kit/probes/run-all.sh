#!/usr/bin/env bash
# vendored from sfxnz/forge kit @ 6f40808
# Run every probe enabled in recipe.yaml `probes:`. Usage: kit/probes/run-all.sh <recipe-dir> <evidence-dir>
# Each probe writes <evidence-dir>/<probe>.txt; this writes <evidence-dir>/probes.json and exits
# non-zero if any enabled probe failed. Probes: smoke count thinking_off tool_call hermes_two_turn needle vision.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
source "$here/../lib.sh"
recipe_load "${1:?usage: kit/probes/run-all.sh <recipe-dir> <evidence-dir>}"
OUT="${2:?usage: kit/probes/run-all.sh <recipe-dir> <evidence-dir>}"
mkdir -p "$OUT"

mapfile -t enabled < <(python3 - "$RECIPE_DIR/recipe.yaml" <<'PY'
import sys, yaml
for item in yaml.safe_load(open(sys.argv[1])).get("probes") or []:
    print(item if isinstance(item, str) else next(iter(item)))
PY
)
if (( ${#enabled[@]} == 0 )); then
  echo "no probes enabled in $RECIPE_DIR/recipe.yaml" >&2
  exit 1
fi

results=()
failed=0
for name in "${enabled[@]}"; do
  if [[ -x "$here/$name.sh" ]]; then
    cmd=("$here/$name.sh" "$RECIPE_DIR" "$OUT")
  elif [[ -f "$here/$name.py" ]]; then
    cmd=(python3 "$here/$name.py" "$RECIPE_DIR" "$OUT")
  else
    echo "unknown probe: $name" >&2
    results+=("{\"probe\":\"$name\",\"status\":\"unknown\",\"exit\":127}")
    failed=1
    continue
  fi
  log "probe $name"
  start=$(date +%s)
  "${cmd[@]}"
  rc=$?
  status=PASS
  if (( rc != 0 )); then status=FAIL; failed=1; fi
  results+=("{\"probe\":\"$name\",\"status\":\"$status\",\"exit\":$rc,\"seconds\":$(( $(date +%s) - start ))}")
  log "probe $name $status"
done

{
  printf '{"recipe":"%s","utc":"%s","failed":%s,"results":[' "$RECIPE_DIR" "$(utc_stamp)" "$failed"
  for i in "${!results[@]}"; do
    (( i > 0 )) && printf ','
    printf '%s' "${results[$i]}"
  done
  printf ']}\n'
} | python3 -m json.tool >"$OUT/probes.json"
echo "evidence=$OUT/probes.json"
exit "$failed"
