---
name: verify-glm53-flash
description: Drive the GLM-5.3-Flash NVFP4 vLLM recipe (GitHub clone → image chain → 2× DGX Spark OpenAI API → decode bench). Use when proving recipe readiness, serve smoke, bench_decode.py, or run.sh/stop.sh. Shared host-network GPU serve — never start a second instance, never stop one this run did not start.
---

# Verify GLM-5.3-Flash NVFP4 recipe

This repo is a GitHub-ready local-model recipe plus a live OpenAI-compatible serve on two DGX Sparks. There is no web UI. The user surfaces are `README.md` + `run.sh`/`stop.sh` + `docker/Dockerfile.sm121-v8`…`v11`, then `http://127.0.0.1:8000/v1`, then `python3 bench_decode.py`.

Repo root: parent of `.cursor/`. Helpers live in `.cursor/skills/verify-glm53-flash/scripts/`. Feature recipes: `features/`. Defaults (`IMAGE`, `PORT`, `CONTAINER_NAME`, KV pin, spec, …) live in `run.sh`; do not restate them here.

Two instances cannot coexist: host network, port 8000, container name `glm53-flash-nvfp4`, `--gpus all` on both ranks. If anything is already bound to that name or port, attach or refuse. Never `./run.sh` over a running container (it `docker rm -f`s the name). Never `./stop.sh` unless this run created `.cursor/skills/verify-glm53-flash/.run-state/started`.

## Launch

1. `cd` to the repo root. `chmod +x run.sh stop.sh`.
2. Run doctor (below). Branch on `status=`:
   - `ready` — attach. Do **not** write `.run-state/started`.
   - `loading` — wait and re-doctor. Do **not** start another container. Ready line from `run.sh` is `Ready → http://127.0.0.1:8000/v1`. First boot is 15–20 minutes; `wait_ready` polls `/v1/models` for up to 40 minutes.
   - `mismatch` — stop. Do not drive, do not `./run.sh`, do not `./stop.sh`.
   - `missing` — only then launch:

```bash
STATE=".cursor/skills/verify-glm53-flash/.run-state"
mkdir -p "$STATE"
./run.sh
# wait until doctor exits 0
printf 'started %s host=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" >"$STATE/started"
```

`./run.sh` on the head (`spark1`) SSHes to `spark2`, starts rank 1, waits 25s, starts rank 0, then blocks in `wait_ready`. Needs Docker, the local `glm53-sm121-v11` image (stock `vllm/vllm-openai` is refused), and the pinned HF snapshots under `~/.cache/huggingface` (or `hf` on PATH). Worker-only: `ROLE=worker ./run.sh`.

## Doctor

```bash
.cursor/skills/verify-glm53-flash/scripts/doctor.sh
```

Read-only. Exit `0` ready, `1` missing, `2` loading (container up, API not answering), `3` mismatch (wrong image, served name, or `max_model_len`). Prints `status=`, `container_state=`, `image=`, `api=`, `model=`, `max_model_len=`, `spec=`, `worker=`, `owned_by_verify=`. Re-run whenever anything looks off, after a failed drive, and before the first drive of a session.

`worker=down` with `status=ready` still means the head API is serving; note it, do not treat it as a pass for TP=2 health.

## Drive

Harness is the scripts plus curl and `python3 bench_decode.py`. Read `features/README.md`, then the matching feature file. Stable handles:

- Completions: `POST http://127.0.0.1:8000/v1/chat/completions` with `"model": "LibertAIDAI/GLM-5.3-Flash-NVFP4"`.
- Models: `GET http://127.0.0.1:8000/v1/models`.
- Metrics: `GET http://127.0.0.1:8000/metrics` (`vllm:spec_decode_*_total`).
- Smoke: `.cursor/skills/verify-glm53-flash/scripts/smoke.sh` (README payload: “Say hello in one sentence.”, `max_tokens` 64, thinking off).
- Recipe clone-shape: `python3 .cursor/skills/verify-glm53-flash/scripts/recipe-lint.py`.
- Bench: `python3 bench_decode.py` from the repo root (see `features/decode-bench.md`).

Do not call vLLM-internal setters, dummy loaders, or `--load-format dummy` as proof of the published recipe.

## Evidence

Write under `.cursor/skills/verify-glm53-flash/artifacts/<feature>/<UTC-stamp>/`. Smoke does this itself (`evidence=` line). For other drives, copy the command, stdout, stderr, exit code, and a doctor dump into that directory.

Proof standards:

- Real user path: README curl, `python3 bench_decode.py`, or `./run.sh` — not a test-only endpoint.
- Capture the action and the resulting state (request JSON + response JSON, or bench `SUMMARY` JSON).
- Side effects: completions must have non-empty `choices[0].message.content`; benches must print `SUMMARY`; recipe-lint must print `result=pass`.
- A dry-run or skipped path is only verified by the unmet precondition you observed (doctor `status=`, missing image, SSH fail), never by the skip’s name.
- Record the feature ID and the flags actually used.

## Cleanup

```bash
.cursor/skills/verify-glm53-flash/scripts/cleanup.sh
```

Stops via `./stop.sh` only if `.run-state/started` exists. Leaves `artifacts/` in place. After cleanup, confirm the evidence directory still exists. If this run only attached, cleanup is a no-op on containers — the lab serve stays up.

## Helpers

All under `.cursor/skills/verify-glm53-flash/scripts/`. `lib.sh` is sourced by the bash helpers (same env defaults as `run.sh`).

| Script | When |
|---|---|
| `doctor.sh` | First, and whenever the serve looks off |
| `recipe-lint.py` | GitHub-ready recipe feature |
| `smoke.sh` | README chat-completions smoke |
| `count_probe.py` | Greedy 1→200 consecutive-integer gate |
| `needle_probe.py` | Long-prompt needle retrieval (`--prompt-tokens`) |
| `cleanup.sh` | End of a run, and after every failed iteration |
