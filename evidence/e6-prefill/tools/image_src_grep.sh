#!/usr/bin/env bash
# image_src_grep.sh PATTERN [SUBDIR]: grep -rn the vLLM tree of glm53-sm121-v13 (read-only, no GPU) for PATTERN.
V=/usr/local/lib/python3.12/dist-packages/vllm
docker run --rm --entrypoint grep glm53-sm121-v13 -rnE --include='*.py' -- "$1" "$V/${2:-}"
