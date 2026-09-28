#!/usr/bin/env bash
# buckets_all.sh PROF_OUT OUT_DIR [RLIMIT_GIB] [KIND...]: tools/step_buckets.py on every rank trace of the F3 profile
# captures (PROF_OUT/<kind>/rank{0,1}/*.pt.trace.json.gz) -> OUT_DIR/<kind>-rank<R>.{txt,json}. Kind "prefill" is
# parsed with --only '_context_1\(' (prefill-chunk steps) instead of the default pure verify steps.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
P="$1"; O="$2"; L="${3:-1}"; shift 3 2>/dev/null || shift $#
kinds=("$@")
[[ ${#kinds[@]} -gt 0 ]] || kinds=(prose code structured prose2 prefill)
mkdir -p "$O"
for k in "${kinds[@]}"; do
  for r in 0 1; do
    t=$(ls "$P/$k/rank$r/"*.pt.trace.json.gz 2>/dev/null | head -1)
    [[ -n "$t" ]] || { echo "== $k rank$r: no trace"; continue; }
    only=()
    [[ "$k" == prefill ]] && only=(--only '_context_1\(')
    nice python3 "$R/tools/step_buckets.py" "$t" --rlimit-gib "$L" "${only[@]}" --json "$O/$k-rank$r.json" \
      > "$O/$k-rank$r.txt" 2>&1
    echo "== $k rank$r rc=$?"
    head -n 17 -- "$O/$k-rank$r.txt"
  done
done
