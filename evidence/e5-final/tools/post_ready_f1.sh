#!/usr/bin/env bash
# post_ready_f1.sh DIR: the E5 F1 post-ready sequence on a ready serve booted with plain ./run.sh. Stops nothing.
#   serve capture (args/env/mounts/image + parity), docker inspect diff vs E4b, backend + GLM53_ lines, memory,
#   count-200 + thinking-off smoke, QUAL-2 kwarg matrix, vision suite, legacy skill probes,
#   ruler v2 full panel (bench-1), fast-gate repeat (bench-2), Tier 0 compare --stage fp8 vs nvidia-v11-k7,
#   flaky utf8/vision reruns (x3), utf8 texts (x3), lossless probe, memory per phase.
# Client steps that can load memory (bench-1 E/I cells, Tier 0) run under a 3.0 GiB memguard.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$1"
cd "$R" || exit 1
T=evidence/e1-tools
E5=evidence/e5-final/tools
G="python3 $T/memguard.py --thresh 3.0"
echo "== $(date -u +%FT%TZ) post_ready_f1 $D"
bash $T/capture_serve.sh "$D" > /dev/null 2>&1
python3 $E5/inspect_diff.py evidence/e4b-int8-tau0.3 "$D" > "$D/docker-inspect-vs-e4b.txt" 2>&1
bash $T/backend.sh > "$D/backend.txt" 2>&1
bash evidence/e3-tools/glm53_lines.sh > "$D/glm53-lines.txt" 2>&1
bash evidence/e2-tools/nodes.sh "after ready" > "$D/ready-mem.txt" 2>&1
bash $T/act.sh "$D" probe-count python3 $T/probes.py
bash $T/act.sh "$D" probe-qual2 python3 evidence/e0-nvidia-v11/scripts/qual2_probe.py "$D/probe-qual2.json"
bash $T/act.sh "$D" probe-vision python3 smoke_vision.py
bash $E5/skill_probes.sh "$D"
bash $T/act.sh "$D" bench-1 $G --log "$D/memguard-bench-1.log" -- \
  python3 bench_decode.py --full --remote-meminfo --out "$D/bench-1"
bash $T/act.sh "$D" bench-2 python3 bench_decode.py --remote-meminfo --out "$D/bench-2"
bash $T/act.sh "$D" tier0-compare $G --log "$D/memguard-tier0-compare.log" -- \
  python3 quality/tier0.py compare --ref nvidia-v11-k7 --stage fp8 --out "$D/tier0.json"
bash evidence/e2-tools/rerun_flaky.sh "$D" 3
bash $T/act.sh "$D" utf8-texts python3 evidence/e3-tools/utf8_texts.py --n 3 --out "$D/utf8-texts.json"
python3 evidence/e4-tools/utf8_scan.py "$D/utf8-texts.json" > "$D/utf8-scan.txt" 2>&1
bash $T/act.sh "$D" probe-lossless python3 evidence/e0-nvidia-v11/scripts/lossless_probe.py 600
cp "$D/probe-lossless.console.txt" "$D/probe-lossless.txt"
bash evidence/e2-tools/nodes.sh "after post_ready" > "$D/end-mem.txt" 2>&1
python3 $T/phase_report.py "$D" > "$D/memory-by-phase.txt" 2>&1
echo "== $(date -u +%FT%TZ) post_ready_f1 done"
