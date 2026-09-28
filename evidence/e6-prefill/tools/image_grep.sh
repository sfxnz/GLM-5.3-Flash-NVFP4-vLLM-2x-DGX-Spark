#!/usr/bin/env bash
# image_grep.sh: files in the vLLM tree of glm53-sm121-v13 and glm53-sm121-v13-e5 that mention the dequant
# variables, so the rebuilt image is shown to carry the new code and the E5 image to lack it.
V=/usr/local/lib/python3.12/dist-packages/vllm
for img in glm53-sm121-v13 glm53-sm121-v13-e5; do
  echo "== $img $(docker image inspect -f '{{.Id}}' $img)"
  for var in GLM53_WQ_DEQUANT_MIN_M GLM53_WQ_DEQUANT_GROUPS; do
    echo "-- $var"
    docker run --rm --entrypoint grep "$img" -rlc --include='*.py' "$var" "$V"
  done
done
