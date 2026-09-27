#!/usr/bin/env bash
# backend.sh: key boot lines from both ranks' docker logs, GLM53_ count, JIT cache sizes.
C="${CONTAINER_NAME:-glm53-flash-nvfp4}"
PAT='for NVFP4 GEMM|MoE backend|Using .*Marlin|scale_2|Loading weights took|Model loading took|GPU KV cache size|Maximum concurrency|init engine|torch.compile takes|Compil(ing|ation) .* took|compile range|Graph capturing finished|capturing CUDA graph|TileLang begins|JIT compil|Encoder cache|KV cache memory|Available KV|speculative|DFlash|drafter|Application startup complete|GLM53_'
for n in head worker; do
  if [[ $n == head ]]; then run=(bash -c); host=spark1; else run=(ssh spark2); host=spark2; fi
  echo "== $n ($host) $(date -u +%FT%TZ)"
  echo "GLM53_ lines: $("${run[@]}" "docker logs $C 2>&1 | grep -c GLM53_")"
  echo "fp4_gemm / JIT-debug lines: $("${run[@]}" "docker logs $C 2>&1 | grep -cE 'fp4_gemm|device-debug'")"
  "${run[@]}" "docker logs $C 2>&1 | grep -E '$PAT' | cut -c1-260 | awk '!seen[\$0]++' | head -80"
  echo "== $n JIT cache sizes"
  "${run[@]}" "cd ~/projects/data/glm53-jit-cache && du -sh */ 2>/dev/null; for d in */; do [ \$d = logs/ ] || du -sh \$d* 2>/dev/null; done; find . -path ./logs -prune -o -type f -print 2>/dev/null | wc -l"
done
