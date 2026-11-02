#!/usr/bin/env bash
# run_bench.sh LABEL [PIP_SPEC] [BENCH_ARGS...]: one moe_bench.py run on this node (spark1) in
# glm53-sm121-v14, Humming cache cold in the container's /tmp. PIP_SPEC (e.g.
# humming-kernels==0.1.19, or "" for the image's own) is pip-installed --no-deps into the
# throwaway container first. Appends free -m and compiler RSS every 2 s to mem.log.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
LABEL="$1"; PIP="${2:-}"; shift; shift || true
if docker ps -a --format '{{.Names}}' | grep -qx glm53-flash-nvfp4; then
  echo "container glm53-flash-nvfp4 exists: not running" >&2; exit 3
fi
mem="$D/mem.log"
{ echo "== $LABEL start $(date -u +%FT%TZ) pip=${PIP:-image}"; free -h; uptime
  docker ps --format '{{.Names}} {{.Image}}'; } >> "$mem"
(
  while :; do
    echo "$(date -u +%T) $(free -m | awk '/^Mem:/{print "used_mib=" $3, "avail_mib=" $7}')" \
      "$(ps -eo rss=,comm= | awk '$2 ~ /^(nvrtc_compile|nvcc|ptxas|cicc|cc1plus|cudafe|ninja|g\+\+|c\+\+)/ {printf "%s:%dMiB ", $2, $1 / 1024}')"
    sleep 2
  done >> "$mem"
) &
mon=$!
inner="python3 evidence/r3/moe-microbench/moe_bench.py --json evidence/r3/moe-microbench/run-$LABEL.json $*"
[[ -n "$PIP" ]] && inner="pip install --no-deps -q $PIP 2>&1 | grep -v 'notice' ; $inner"
t0=$(date +%s)
docker run --rm --gpus all --ipc=host --name "moe-microbench-$LABEL" -v "$R":/w -w /w \
  -e HUMMING_CACHE_DIR=/tmp/humming-cache --entrypoint bash glm53-sm121-v14 -c "$inner" \
  > "$D/run-$LABEL.log" 2>&1
rc=$?
kill "$mon"
{ echo "== $LABEL end rc=$rc elapsed_s=$(( $(date +%s) - t0 )) $(date -u +%FT%TZ)"; free -h; } >> "$mem"
docker run --rm --entrypoint chown -v "$D":/d glm53-sm121-v14 -R "$(id -u):$(id -g)" /d
tail -40 "$D/run-$LABEL.log"
exit "$rc"
