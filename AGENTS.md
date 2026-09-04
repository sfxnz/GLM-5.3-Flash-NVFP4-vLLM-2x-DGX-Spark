# AGENTS.md — GLM-5.3-Flash-NVFP4 · 2× DGX Spark

Serve `LibertAIDAI/GLM-5.3-Flash-NVFP4` at TP=2. Local image chain through `glm53-sm121-v11`. Checkpoint `caca4e6`. Default drafter is DFlash2-7 (`SPEC=dflash2`). `SPEC=mtp` rolls back to MTP-4.

Humans read [README.md](README.md). LibertAI's GB10 recipe is a different stack (MTP-3, eager, 64K).

## Working rules

- `recipe.yaml` is the source of truth for pins and generated blocks. Edit it, then `python3 kit/render.py`. Do not hand-edit `# BEGIN generated` or `<!-- BEGIN generated` blocks.
- Change one knob at a time against `python3 bench_decode.py`. Revert if it does not beat noise or it regresses another cell. Record the revert in `evidence/` (`trail.tsv`, `decision.tsv`).
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`. Defaults in `run.sh` are `enp1s0f1np1` / `rocep1s0f1`.
- Keep `chat_template.jinja`. The stock HF template always opens `<think>`, so `enable_thinking: false` used to leak chain-of-thought into `content`.
- Do not turn on InstantTensor. That loader killed TP=2 ranks here.
- Do not set `VLLM_GLM53_MOE_INPUT_SCALE=1.0`. That constant underflows per 16-element block.
- `run.sh` already calls `maybe_drop_caches`. It no-ops without passwordless sudo.

Default occupancy is DFlash2-7 at two sequences. Four-way admission needs the rollback `NUM_SPECULATIVE_TOKENS=5 MAX_NUM_SEQS=4`. Leave `--async-scheduling` off. Leave `VLLM_USE_BREAKABLE_CUDAGRAPH` on auto.

## Refuse-guards (`run.sh`)

- `--max-model-len` above 327680 on `fp8_e4m3` unless `FORCE_UNSAFE_CTX=1`. Native context is 1,048,576. A 1M request needs ~8.2 GiB of this hybrid layout and GB10 UMA OOMs above ~5.1 GiB.
- Any `MOE_BACKEND` other than `marlin` unless `FORCE_UNSAFE_MOE=1`. `flashinfer_cutlass` OOM'd spark2 during JIT after 90.67 GiB weights.
- KV pin at or below `3886945403` (3.62 GiB) cannot hold 327680. Tony's 3.0 GiB pin is a 262144-ctx budget.

`--kv-cache-memory 4445787956` (4.14 GiB) stays the pin. Dropping it OOMs. Raising it boots but backfires under UMA pressure.

## Verify

```bash
python3 kit/render.py --check
python3 bench_decode.py                    # both phases, c=1 and 2, after serve is up
```

Thinking-off smoke must not start `content` with chain-of-thought. Greedy count stays lossless (200 consecutive integers with thinking off).

## Never touch

- Live HF tokens
- Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` as the serve image. `run.sh` refuses it. Build v8 → v9 → v10 → v11.
- Hand-edited generated README / `run.sh` blocks
- Advertising a 1M window on this fp8 pin
