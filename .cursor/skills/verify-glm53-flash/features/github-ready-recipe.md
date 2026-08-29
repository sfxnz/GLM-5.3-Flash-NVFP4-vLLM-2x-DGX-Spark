# GitHub-ready recipe

A stranger cloning this repo can follow `README.md` to build the local image chain, start the 2× Spark serve, smoke the API, bench decode, and stop, with licenses and defaults matching `run.sh`.

## Sub-features

- `recipe-files` ships `README.md`, `LICENSE`, `run.sh`, `stop.sh`, `bench_decode.py`, and the v8–v11 Dockerfiles plus the patches they `COPY`.
- `recipe-exec` keeps `run.sh` and `stop.sh` executable.
- `recipe-defaults` documents the same `IMAGE`, port, context, KV pin, spec, and served name that `run.sh` defaults to.
- `recipe-images` documents the four-step `docker build` chain starting from `vllm/vllm-openai:glm53-flash-arm64-cu130` and ending at `glm53-sm121-v11`.
- `recipe-license` states MIT for the scripts and CC BY-NC-ND for the DFlash2 draft.

## How to get to it (user POV)

- Clone the GitHub repo and open `README.md`.
- Run the documented `docker build -f docker/Dockerfile.sm121-v8` … `v11` commands.
- Run `./run.sh`, the smoke `curl`, `python3 bench_decode.py`, and `./stop.sh` as written in `README.md`.

## Driving it with verify-glm53

Preconditions:

- Working directory is the repo root.
- The serve does not need to be up.

- **Lint the clone.** Run `python3 .cursor/skills/verify-glm53-flash/scripts/recipe-lint.py`. Exit code `0` and stdout contain `result=pass`.
- **Proof.** Copy that stdout into `artifacts/github-ready-recipe/<stamp>/recipe-lint.txt`. The transcript lists no `FAIL` lines.

## Gotchas

- Local Docker images being absent is not a recipe failure. The README tells the user to build them; lint does not require `docker image inspect`.
- `patch_v7.py` and `patch_v8_fp8.py` are leftover sources. The published chain starts at `Dockerfile.sm121-v8` (inline patches). Do not require those extra files in the README.
- A README that mentions the right numbers in prose but not the `docker build -f docker/Dockerfile.sm121-v*` commands is incomplete.
- `run.sh` refuses the stock `vllm/vllm-openai` tag. A default `IMAGE` of that stock tag is a recipe regression.
