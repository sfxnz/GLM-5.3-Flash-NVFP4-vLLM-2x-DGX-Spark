#!/usr/bin/env bash
# analyze.sh DIR: kit/compare.py vs E2e and E0 (compare-vs-e2e.*, compare-vs-e0.*, T vs E2e), per-position
# acceptance, Tier 0 table vs E0/E1a/E2e, memory per phase and the GLM53_ lines for one E3 evidence dir.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"
cd "$R" || exit 1
E0=evidence/e0-nvidia-v11
E2E=evidence/e2e-nvfp4-draft-only
python3 kit/compare.py --a "$E2E/bench-1/bench.json" "$E2E/bench-2/bench.json" \
  --b "$D/bench-1/bench.json" "$D/bench-2/bench.json" --json "$D/compare-vs-e2e.json" > "$D/compare-vs-e2e.txt"
python3 kit/compare.py --a "$E0/bench-1/bench.json" "$E0/bench-2/bench.json" \
  --b "$D/bench-1/bench.json" "$D/bench-2/bench.json" --json "$D/compare-vs-e0.json" > "$D/compare-vs-e0.txt"
[[ -f "$D/bench-T/bench.json" ]] && python3 kit/compare.py --a "$E2E/bench-T/bench.json" --b "$D/bench-T/bench.json" \
  > "$D/compare-T-vs-e2e.txt"
echo "== compare vs E2e"; cat "$D/compare-vs-e2e.txt"
echo "== compare vs E0"; cat "$D/compare-vs-e0.txt"
[[ -f "$D/compare-T-vs-e2e.txt" ]] && { echo "== T vs E2e"; cat "$D/compare-T-vs-e2e.txt"; }
echo "== per_pos"
args=(e2e-b1="$E2E/bench-1/bench.json" e2e-b2="$E2E/bench-2/bench.json" e2e-T="$E2E/bench-T/bench.json"
      b1="$D/bench-1/bench.json" b2="$D/bench-2/bench.json")
[[ -f "$D/bench-T/bench.json" ]] && args+=(T="$D/bench-T/bench.json")
[[ -f "$D/bench-E/bench.json" ]] && args+=(e0-b1="$E0/bench-1/bench.json" E="$D/bench-E/bench.json")
python3 evidence/e1-tools/per_pos.py "${args[@]}" | grep -E '^(cell|A |B |J@c1|J@c2|H |T |E@)'
echo "== Tier 0"
t=(E0="$E0/tier0.json" E1a=evidence/e1a-v13-off/tier0.json E2e="$E2E/tier0.json" this="$D/tier0.json")
python3 evidence/e3-tools/tier0_table.py "${t[@]}"
echo "== memory"; head -9 "$D/memory-by-phase.txt"
sed -n '/activity windows/,/compilers:/p' "$D/memory-by-phase.txt"
echo "== boot"; cat "$D/run.start"; grep -E 'parity' "$D/docker-args-check.txt"
cat "$D/glm53-lines.txt"
