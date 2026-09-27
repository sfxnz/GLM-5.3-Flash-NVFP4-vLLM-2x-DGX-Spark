# Decode bench

`bench_decode.py` is ruler v2. It streams completions against the live API in cells. The published decode score is cell A only: prose, 8 distinct prompts × 512 forced tokens, greedy, thinking off. Each wave is factorized from `/metrics` deltas into `acceptance_len`, `step_ms` and `tok_s`. The script prints one line per wave, a table, and a `SUMMARY` JSON array (one object per cell group).

## Sub-features

- `bench-published` is cell A (prose 8×512, c=1).
- `bench-code` is cell B (code 8×512, c=1).
- `bench-structured` is cell J (“Count from 1 to 200”, max 200, c=1 and c=2). Acceptance ceiling, not a published score.
- `bench-concurrency` is cell H (distinct prompts, c=2) and cell I (c=4, only with `--cells ...,I` or `--full`).
- `bench-long` is cell E (32k / 128k salted prompts: TTFT, prefill tok/s). `bench-vision` is cell F (generated PNG). `bench-sampled` is cell G (T=1.0, top_p 0.95, fixed seed).
- `bench-legacy` is cell K (the pre-v2 ~98-token prose prompt, c=1 and c=2) for continuity with old receipts.
- `bench-acceptance` reads `vllm:spec_decode_*` (including per-position) per wave. Missing counters leave `step_ms`/`acceptance_len` empty; they do not fail the bench.
- `bench-hygiene` samples `/proc/meminfo` before and after each cell and marks the cell INVALID when swap use grows more than 64 MiB. `--remote-meminfo` adds spark2 over ssh.

## How to get to it (user POV)

- With the serve ready, from the repo root: `python3 bench_decode.py` (cells A,B,J,H,K).
- Everything: `python3 bench_decode.py --full --out evidence/<stamp>`.
- Pick cells: `python3 bench_decode.py --cells J`.
- Compare boots: `python3 kit/compare.py --a base1/bench.json base2/bench.json --b arm1/bench.json arm2/bench.json`.

## Driving it with verify-glm53

Preconditions:

- Doctor prints `status=ready`.
- Working directory is the repo root.
- The default command occupies the serve for roughly 10 minutes. A harness proof may use `--cells J --tokens 64` when the goal is only “the script streams and prints `SUMMARY`”, and must record the flags. Claims about the README tok/s table require cell A. Do not score decode from other cells.

- **Published bench.** Run `python3 bench_decode.py --out <dir>`. Exit code `0` (1 means a failed wave or an INVALID cell). The `SUMMARY` object with `group` `A` has `tok_s.mean`, `step_ms.mean` and `acceptance_len.mean` greater than 0, `sanity_ok` true and `short_requests` 0.
- **Harness-sized bench.** Run `python3 bench_decode.py --cells J`. Exit code `0`. `SUMMARY` has groups `J@c1` and `J@c2`.
- **Proof.** Keep `<dir>/bench.txt` and `<dir>/bench.json` plus a doctor dump. `bench.json` carries `ruler_version`, the served model id and the git sha.

## Gotchas

- Each cell sends one discarded warm-up first, so JIT does not land in measured waves.
- `--model` defaults to the first id from `/v1/models`, so a rollback serve is labelled with its real name.
- Cell E sizes prompts from a calibration warm-up; the reported `prompt_tokens` are the real lengths.
- `SPEC=mtp` vs DFlash2 changes acceptance. Doctor `spec=` must match the claim you are proving.
