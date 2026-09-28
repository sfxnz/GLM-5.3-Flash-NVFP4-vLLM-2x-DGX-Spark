#!/usr/bin/env bash
# pad_bench.sh DIR [ARGS...]: evidence/e5-final/tools/bench_pad_n.py in glm53-sm121-v13 on the head (no serve up), the
# same docker run as evidence/e4-tools/microbench.sh. Console in DIR/bench.log.
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$1"; shift
cd "$R" || exit 1
mkdir -p "$D"
{
  echo "start $(date -u +%FT%TZ) git=$(git rev-parse HEAD) image=$(docker image inspect -f '{{.Id}}' glm53-sm121-v13) host=$(hostname)"
  echo "running containers:"; docker ps --format '{{.Names}} {{.Image}}'
  echo "MemAvailable: $(grep MemAvailable /proc/meminfo)"
  t0=$(date +%s)
  docker run --rm --gpus all --entrypoint python3 -v "$R":/work -w /work \
    glm53-sm121-v13 evidence/e5-final/tools/bench_pad_n.py --json "$D/bench.json" "$@"
  echo "rc=$? elapsed_s=$(( $(date +%s) - t0 )) end $(date -u +%FT%TZ)"
} > "$D/bench.log" 2>&1
tail -20 "$D/bench.log"
