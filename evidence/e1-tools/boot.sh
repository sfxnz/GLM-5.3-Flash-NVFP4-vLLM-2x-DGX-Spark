#!/usr/bin/env bash
# boot.sh DIR [NAME=VALUE ...]: preflight, start kit/uma_watch.py (setsid) on DIR,
# then ./run.sh with the given env. Everything is logged into DIR.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"
shift
mkdir -p "$D"
cd "$R" || exit 1
{
  echo "== $(date -u +%FT%TZ) preflight; env for run.sh: $*"
  echo "git $(git rev-parse HEAD) $(git rev-parse --abbrev-ref HEAD)"
  git status --short
  echo "GLM53_* in this shell: $(env | grep -c '^GLM53_')"
  sudo -n true 2>/dev/null && echo "sudo -n: yes" || echo "sudo -n: no (drop_caches no-ops)"
  for n in spark1 spark2; do
    if [[ $n == spark1 ]]; then run=(bash -c); else run=(ssh spark2); fi
    echo "== $n docker ps -a"; "${run[@]}" "docker ps -a --format '{{.ID}} {{.Image}} {{.Status}} {{.Names}}'"
    echo "== $n free -h"; "${run[@]}" "free -h"
    echo "== $n image glm53-sm121-v13"; "${run[@]}" "docker image inspect -f '{{.Id}}' glm53-sm121-v13"
    echo "== $n JIT cache"; "${run[@]}" "ls -la ~/projects/data/glm53-jit-cache/ 2>&1; du -sh ~/projects/data/glm53-jit-cache/* 2>/dev/null"
    echo "== $n GPU compute apps (process list, not memory)"; "${run[@]}" "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader"
  done
} > "$D/preflight.txt" 2>&1
{ date -u +%FT%TZ; git rev-parse HEAD; echo "$*"; } > "$D/run.start"
setsid nohup python3 kit/uma_watch.py --evidence "$D" > "$D/watcher.out" 2> "$D/watcher.err" < /dev/null &
echo $! > "$D/watcher.pid"
env "$@" ./run.sh > "$D/run.log" 2>&1
echo "run.sh rc=$? $(date -u +%FT%TZ)" >> "$D/run.log"
