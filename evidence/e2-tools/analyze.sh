#!/usr/bin/env bash
# analyze.sh DIR: kit/compare.py vs E0 and E1a (compare-vs-e0.*, compare-vs-e1a.*), then print per-position
# acceptance, memory per phase, boot timings and parity for one E2 evidence dir.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"
cd "$R" || exit 1
E0=evidence/e0-nvidia-v11
E1A=evidence/e1a-v13-off
python3 kit/compare.py --a "$E0/bench-1/bench.json" "$E0/bench-2/bench.json" \
  --b "$D/bench-1/bench.json" "$D/bench-2/bench.json" --json "$D/compare-vs-e0.json" > "$D/compare-vs-e0.txt"
python3 kit/compare.py --a "$E1A/bench-1/bench.json" "$E1A/bench-2/bench.json" "$E1A/bench-3-A/bench.json" \
  --b "$D/bench-1/bench.json" "$D/bench-2/bench.json" --json "$D/compare-vs-e1a.json" > "$D/compare-vs-e1a.txt"
echo "== compare vs E0"; cat "$D/compare-vs-e0.txt"
echo "== per_pos"
python3 evidence/e1-tools/per_pos.py b1="$D/bench-1/bench.json" b2="$D/bench-2/bench.json" T="$D/bench-T/bench.json" \
  | grep -E '^(cell|A |B |J@c1|J@c2|H |T )'
echo "== J@c1 output hashes"
python3 - "$D" <<'EOF'
import json, sys
for b in ("bench-1", "bench-2"):
    d = json.load(open(f"{sys.argv[1]}/{b}/bench.json"))
    print(b, [(r["sha256"][:10], round(w["acceptance_len"], 2)) for w in d["waves"] if w["group"] == "J@c1" for r in w["requests"]])
EOF
echo "== memory"; head -9 "$D/memory-by-phase.txt"
sed -n '/activity windows/,/compilers:/p' "$D/memory-by-phase.txt"
echo "== boot"; cat "$D/run.start"
grep -h -E 'Loading weights took|Model loading took|init engine|GLM53_' "$D/head.stream.log" "$D/worker.stream.log" | cut -c1-330
grep -E 'parity' "$D/docker-args-check.txt"
