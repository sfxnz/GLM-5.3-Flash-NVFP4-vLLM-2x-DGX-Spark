# Decode bench

`bench_decode.py` streams greedy completions against the live API in two regimes — prose (low draft acceptance) and structured count-to-200 (high acceptance) — and prints per-stream tok/s, TTFT, and a `SUMMARY` JSON that includes spec-decode acceptance when `/metrics` exposes it.

## Sub-features

- `bench-prose` runs the sparse-attention paragraph prompt.
- `bench-structured` runs “Count from 1 to 200” with digits-only output.
- `bench-concurrency` repeats each phase at c=1,2 (the published table; default `max-num-seqs` is 2).
- `bench-acceptance` records `acceptance_len` / `draft_acceptance_rate` from `vllm:spec_decode_*` when those counters exist.

## How to get to it (user POV)

- With the serve ready, from the repo root: `python3 bench_decode.py`.
- One phase: `python3 bench_decode.py --phase structured`.
- Custom URL/model: `python3 bench_decode.py --url http://127.0.0.1:8000/v1/chat/completions --model LibertAIDAI/GLM-5.3-Flash-NVFP4`.

## Driving it with verify-glm53

Preconditions:

- Doctor prints `status=ready`.
- Working directory is the repo root.
- The published user command is `python3 bench_decode.py` (both phases, `--runs 3`, `--concurrency 1 2`, `--max-tokens 200`). That occupies the shared serve for several minutes. A harness proof may shrink `--runs`, `--concurrency`, and `--max-tokens` only when the goal is “the script streams and prints `SUMMARY`”, and must record those flags. Claims about the README tok/s table require the full command.

- **Published bench.** Run `python3 bench_decode.py`. Exit code `0`. Stdout contains a `SUMMARY` JSON array with `phase` `prose` and `structured` and `concurrency` 1 and 2. Each object has `median_decode_tok_s` and `median_ttft_s` greater than 0.
- **Harness-sized bench.** Run `python3 bench_decode.py --phase structured --runs 1 --concurrency 1 --max-tokens 64`. Exit code `0`. Stdout contains `SUMMARY` with one object, `phase=structured`, `concurrency=1`, and `median_completion_tokens` greater than 0.
- **Acceptance.** When `/metrics` includes `vllm:spec_decode_num_drafts_total`, the `SUMMARY` object also has `acceptance_len`. Structured should sit well above prose. Missing counters are a skip of `bench-acceptance`, not a bench failure.
- **Proof.** Save the full stdout as `artifacts/decode-bench/<stamp>/bench.txt` plus a doctor dump. Keep the exact argv in that file’s first line (`url=... phases=...`).

## Gotchas

- First wave after a restart pays Triton JIT; TTFT is not comparable to the README table until a warm wave.
- `--phase structured` output is a counting sequence. Do not fail the bench because the model omitted a number; the metric is tok/s and acceptance, not exact 1–200.
- `temperature` is hardcoded 0. Do not add a temperature flag as a “fix”.
- Running the full published command on the shared 18-hour lab instance will contend with other users of port 8000. Prefer the harness-sized command unless the task is to reproduce the README table.
- `SPEC=mtp` vs DFlash2 changes acceptance. Doctor `spec=` must match the claim you are proving.
