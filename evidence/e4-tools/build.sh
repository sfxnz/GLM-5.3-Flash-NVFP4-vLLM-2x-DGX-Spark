#!/usr/bin/env bash
# build.sh DIR: rebuild glm53-sm121-v13 on the head from HEAD (--no-cache --progress=plain, 5 patch layers:
# misc, fp8, census, verify, kpool_tail), ship it with docker save | ssh spark2 docker load, check image IDs.
# Same as evidence/e3-tools/build.sh, with glm53-sm121-v13-e3 (the E3 image) in the parity list.
R="$(cd "$(dirname "$0")/../.." && pwd)"
D="$1"
cd "$R" || exit 1
mkdir -p "$D"
{ git rev-parse HEAD; git status --short; sha256sum docker/Dockerfile.sm121-v13 docker/patch_v13_*.py; } > "$D/git-head.txt"
t0=$(date -u +%FT%TZ)
docker build --no-cache --progress=plain -f docker/Dockerfile.sm121-v13 -t glm53-sm121-v13 docker > "$D/build-spark1.log" 2>&1
rc=$?
echo "rc=$rc start $t0 end $(date -u +%FT%TZ)" >> "$D/build-spark1.log"
[[ $rc == 0 ]] || { echo "build failed rc=$rc"; exit $rc; }
docker image inspect -f '{{.Id}} {{.Created}}' glm53-sm121-v13 > "$D/image-spark1.txt"
t1=$(date -u +%s)
{ echo "start $(date -u +%FT%TZ)"; docker save glm53-sm121-v13 | ssh spark2 docker load; echo "rc=${PIPESTATUS[*]} end $(date -u +%FT%TZ) elapsed_s=$(( $(date -u +%s) - t1 ))"; } > "$D/ship-spark2.log" 2>&1
{
  for n in spark1 spark2; do
    if [[ $n == spark1 ]]; then run=(bash -c); else run=(ssh spark2); fi
    for t in glm53-sm121-v13 glm53-sm121-v13-e3 glm53-sm121-v11; do
      echo "$n $t $("${run[@]}" "docker image inspect -f '{{.Id}}' $t")"
    done
  done
} > "$D/image-parity.txt" 2>&1
cat "$D/ship-spark2.log" "$D/image-parity.txt"
echo "== build.sh done $(date -u +%FT%TZ)"
