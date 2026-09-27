#!/usr/bin/env bash
# jit_final.sh DIR IMAGE_KEY: JIT cache sizes on both nodes + serving-time JIT warnings from DIR/head.stream.log
D="$1"; K="$2"
echo "== $(date -u +%FT%TZ) JIT cache $K"
echo "spark1:"; du -sh ~/projects/data/glm53-jit-cache/"$K"/*; echo "files: $(find ~/projects/data/glm53-jit-cache/"$K" -type f | wc -l)"
echo "spark2:"; ssh spark2 "du -sh ~/projects/data/glm53-jit-cache/$K/*; echo files: \$(find ~/projects/data/glm53-jit-cache/$K -type f | wc -l)"
for r in head worker; do
  echo "$r: JIT-during-inference warnings: $(grep -c 'JIT compilation during inference' "$D/$r.stream.log")"
  grep 'JIT compilation during inference' "$D/$r.stream.log" | sed -E 's/.*\[jit_monitor.py:[0-9]+\] (.*) during inference: ([^.]*)\..*/\1: \2/' | sort | uniq -c
  echo "$r: TileLang compiles: $(grep -c 'TileLang begins to compile' "$D/$r.stream.log")"
done
