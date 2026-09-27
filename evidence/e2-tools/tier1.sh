#!/usr/bin/env bash
# tier1.sh DIR NAME: full Tier 1 (quality/tier1.py run, greedy, serve-default kwargs, c=2) on the running serve under
# a 3.0 GiB client memguard, results in ~/projects/data/glm53-evals/tier1/NAME/tier1.jsonl, then
# quality/compare_tier1.py vs nvidia-v11-k7. Console, summary and compare land in DIR. Run it detached.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; N="$2"
OUT="$HOME/projects/data/glm53-evals/tier1/$N"
REF="$HOME/projects/data/glm53-evals/tier1/nvidia-v11-k7/tier1.jsonl"
cd "$R" || exit 1
mkdir -p "$OUT"
bash evidence/e1-tools/act.sh "$D" tier1 python3 evidence/e1-tools/memguard.py --thresh 3.0 --log "$D/memguard-tier1.log" -- \
  python3 quality/tier1.py run --out "$OUT/tier1.jsonl" --label "$N"
bash evidence/e1-tools/act.sh "$D" tier1-compare python3 quality/compare_tier1.py "$REF" "$OUT/tier1.jsonl" \
  --json "$D/compare-tier1-vs-nvidia-v11-k7.json"
cp "$OUT/tier1.summary.json" "$D/tier1.summary.json" 2>/dev/null
sha256sum "$OUT/tier1.jsonl" "$REF" > "$D/tier1-sha256.txt"
echo "== $(date -u +%FT%TZ) tier1.sh done"
