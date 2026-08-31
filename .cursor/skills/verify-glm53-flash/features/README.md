# GLM-5.3-Flash NVFP4 verification map

This directory is the maintained source for verifying the user-facing behavior of the GLM-5.3-Flash NVFP4 vLLM recipe. Read the index before driving, then use the matching feature file.

## Baseline preconditions

- Work from the git repo root (the directory that contains `run.sh` and `README.md`).
- Run `.cursor/skills/verify-glm53-flash/scripts/doctor.sh` before any live drive.
- Drive a live API only when doctor prints `status=ready` for `LibertAIDAI/GLM-5.3-Flash-NVFP4` at `http://127.0.0.1:8000/v1` with `max_model_len=327680`.
- Never start a second serve. Host network, port 8000, container name `glm53-flash-nvfp4`, and all GPUs are shared with the lab.
- Never run `./run.sh` or `./stop.sh` against an instance this run did not start (`owned_by_verify=0`).
- Recipe-lint does not need the serve. Live smoke and benches do.

## Driving conventions

- Start every recipe from doctor `status=ready` unless the feature is clone-shape only.
- Treat every command as literal. Keep model ids, flags, and JSON keys unchanged.
- HTTP drives use curl against `/v1/models` and `/v1/chat/completions`.
- Bench drives use `python3 bench_decode.py` from the repo root.
- Clone-shape drives use `python3 .cursor/skills/verify-glm53-flash/scripts/recipe-lint.py`.
- Restore nothing on the serve after a completion; do not remove proof artifacts during cleanup.

## Proof and skip reporting

- Capture the user action and the resulting state, not only the final token.
- HTTP proof includes request JSON, response JSON, and HTTP status.
- Bench proof includes the command line, the per-run lines, and the `SUMMARY` JSON.
- Recipe proof includes the linter transcript with `result=pass`.
- Record the feature ID and entry point used with every artifact.
- Report an unreachable path with the attempted command and the unmet doctor/`status=` (or missing image, SSH, weights).
- Do not report a skipped entry point as verified through a different path.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with verify-glm53` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [GitHub-ready recipe](./github-ready-recipe.md) covers the clone-and-follow-README surface: files, licenses, image chain, documented defaults.
- [Serve smoke](./serve-smoke.md) covers the README curl against `/v1/chat/completions`.
- [Decode bench](./decode-bench.md) covers `python3 bench_decode.py` prose and structured phases.
- [Quality probes](./quality-probes.md) covers thinking-off leak, parsed tool calls, greedy count, and unique-salt prefill.
- [Serve start and stop](./serve-start-stop.md) covers `./run.sh` and `./stop.sh` on both Sparks.
