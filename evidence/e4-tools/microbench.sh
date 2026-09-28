#!/usr/bin/env bash
# microbench.sh DIR [ARGS...]: tools/bench_fp8_marlin.py in glm53-sm121-v13 on the head (no serve up), with the
# docstring's command (--entrypoint python3, repo at /work, HF cache at /hf). Console in DIR/bench.log.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"; shift
cd "$R" || exit 1
mkdir -p "$D"
{
  echo "start $(date -u +%FT%TZ) git=$(git rev-parse HEAD) image=$(docker image inspect -f '{{.Id}}' glm53-sm121-v13) host=$(hostname)"
  echo "running containers:"; docker ps --format '{{.Names}} {{.Image}}'
  echo "MemAvailable: $(grep MemAvailable /proc/meminfo)"
  t0=$(date +%s)
  docker run --rm --gpus all --entrypoint python3 -v "$R":/work -w /work \
    -v "$HOME/.cache/huggingface":/hf:ro -e HF_HUB_CACHE=/hf/hub \
    glm53-sm121-v13 tools/bench_fp8_marlin.py --json "$D/bench.json" "$@"
  echo "rc=$? elapsed_s=$(( $(date +%s) - t0 )) end $(date -u +%FT%TZ)"
} > "$D/bench.log" 2>&1
tail -70 "$D/bench.log"
