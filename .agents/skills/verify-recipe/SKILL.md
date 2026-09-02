---
name: verify-recipe
description: Verify the GLM-5.3-Flash NVFP4 vLLM recipe (this repo) with the vendored forge kit. Lease → doctor → probes → bench → evidence under evidence/<unit>-<UTC>/. Use when proving a recipe boots, answers, and holds its published decode numbers. Shared GPU serve — never start a second instance, never stop one this run did not start.
---

# Verify GLM-5.3-Flash NVFP4

Repo root is the directory with `recipe.yaml`, `run.sh`, `stop.sh`, `kit/`. Every value (port, container, served name, worker host, thinking-off kwargs, probe list, refusals) comes from `recipe.yaml`; do not restate them. Read `AGENTS.md` in the recipe repo first.

## 0. No GPU: clone shape

```sh
python3 kit/recipe_lint.py .          # must print result=pass
VALIDATE_ONLY=1 ./run.sh              # must print ==> validate-only ...
python3 kit/render.py --check --strict
```

This is what CI runs. Fix the recipe until all three pass before touching a GPU.

## 1. Lease

```sh
FORGE=~/projects/ai-lab/forge
$FORGE/gpu/lease.sh acquire pair --ttl 3h --unit <unit>     # lanes: spark1, spark2, pair (TP=2 recipes need pair)
```

No lease, no `./run.sh`. Release at the end.

## 2. Doctor

```sh
kit/doctor.sh .        # one line: status=ready|missing|loading|mismatch|worker-down worker=up|down ...
```

Exit `0` ready on all ranks; `1` missing; `2` loading (container up, API silent); `3` mismatch (image, served name or `max_model_len` differ from `recipe.yaml`); `4` head ready, worker container down. Branch:

- `ready` — attach. Do not `./run.sh`. Do not `./stop.sh` at the end (not yours).
- `loading` — wait, re-run doctor. First boot is 15–20 min. Never start another container.
- `mismatch` — stop. Report. Do not drive, start or stop.
- `missing` — `./run.sh` (blocks until `Ready →`), then doctor again until exit 0. You own this serve; `./stop.sh` at the end.

## 3. Evidence dir

```sh
UNIT=<unit>; UTC=$(date -u +%Y%m%dT%H%M%SZ); EV=evidence/$UNIT-$UTC; mkdir -p "$EV"
kit/doctor.sh . | tee "$EV/doctor.txt"
```

Everything below writes into `$EV`. `evidence/` is never gitignored.

## 4. Probes

```sh
kit/probes/run-all.sh . "$EV"       # runs recipe.yaml probes:, writes <probe>.txt each + probes.json, exit 1 on any FAIL
```

One probe: `kit/probes/<name>.py . "$EV"` (`smoke.sh` for smoke). Probes: `smoke` (README hello), `count` (1→200 consecutive run), `thinking_off` (no `<think>` leak), `tool_call` (parsed `get_weather`), `hermes_two_turn` (tool result → answer), `needle` (unique-salt retrieval at the lengths in `recipe.yaml`, `-c2` entries also prove occupancy), `vision` (image_url). Each `<probe>.txt` ends in `verdict=PASS` or `verdict=FAIL reason=...`.

## 5. Bench (the frozen ruler)

```sh
python3 kit/bench_decode.py --recipe . --phase both --out "$EV"    # bench.txt + bench.json
python3 kit/bench_compare.py evidence/rebench-20260902T204243Z/bench.json "$EV/bench.json"   # ±5% on decode and aggregate
```

Streamed greedy, thinking off, 200 tokens, 3-run median, c=1 and c=2. Do not pass `--max-tokens`, `--runs` or `--concurrency` for published numbers. README publishes the prose rows; `--phase both` keeps the ruler comparable.

## 6. Publish a number

Only from a `bench.json` in `evidence/`. Edit `recipe.yaml measured.decode.rows` (`decode`, `aggregate`, `ttft_p50`, `evidence: evidence/<unit>-<UTC>/bench.txt`), run `python3 kit/render.py`, commit on `agent/**`, open a PR. No file, no number.

## 7. Cleanup

```sh
./stop.sh 2>&1 | tee "$EV/stop.txt"      # only if this run started the serve
$FORGE/gpu/lease.sh release pair
```

## Proof standards

- Real user path only: `./run.sh`, the kit probes, `kit/bench_decode.py`. No `--load-format dummy`, no engine internals.
- Capture action and result: `<probe>.txt` has request, response, HTTP status, verdict. `bench.txt` ends with `SUMMARY`.
- A skipped step is verified only by the precondition you observed (doctor `status=`, missing image), never by the skip's name.
- Record the unit id and every flag actually used in `$EV/commands.sh`.
