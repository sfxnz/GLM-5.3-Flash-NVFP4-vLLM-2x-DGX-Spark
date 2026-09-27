# AGENTS.md — GLM-5.3-Flash-NVFP4 · 2× DGX Spark

Serve `nvidia/GLM-5.3-Flash-NVFP4` at TP=2. Local image chain through `glm53-sm121-v11`. Checkpoint `09b04e5`. Default drafter is DFlash2-7 (`SPEC=dflash2`). MTP-4 rollback is the LibertAI pack plus `SPEC=mtp` (`MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 SNAPSHOT_REV=caca4e6a4ebbd66f159d3d2fc256683fd6e27177 SPEC=mtp`); `run.sh` refuses `SPEC=mtp` on the nvidia pack. Vision is on (`LANGUAGE_MODEL_ONLY=0`).

Humans read [README.md](README.md). NVIDIA's card is a GB200 TP=4 / EP / 32-seq recipe. Do not copy those flags onto 2× Spark. LibertAI's GB10 recipe is a different stack (MTP-3, eager, 64K). Rollback: `MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 SNAPSHOT_REV=caca4e6a4ebbd66f159d3d2fc256683fd6e27177`.

## Working rules

- `recipe.yaml` is the source of truth for pins and generated blocks. Edit it, then `python3 kit/render.py`. Do not hand-edit `# BEGIN generated` or `<!-- BEGIN generated` blocks.
- Change one knob at a time against `python3 bench_decode.py`. Revert if it does not beat noise or it regresses another cell. Record the revert in `evidence/` (`trail.tsv`, `decision.tsv`).
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`. Defaults in `run.sh` are `enp1s0f1np1` / `rocep1s0f1`.
- Keep `chat_template.jinja`. The stock HF template and the official NVIDIA Hub template always open `<think>`, so `enable_thinking: false` used to leak chain-of-thought into `content`.
- Leave vision on. `LANGUAGE_MODEL_ONLY=1` is refused unless `FORCE_UNSAFE_VISION=1`. Cap is `--limit-mm-per-prompt '{"image":4,"video":1}'`. Do not skip MM profiling into a max-size dummy.
- Published decode score is prose only. Do not score decode from structured, code, or other cells.
- Do not turn on InstantTensor. That loader killed TP=2 ranks here.
- Do not set `VLLM_GLM53_MOE_INPUT_SCALE=1.0`. That constant underflows per 16-element block.
- `run.sh` already calls `maybe_drop_caches`. It no-ops without passwordless sudo.

Default occupancy is DFlash2-7 at two sequences. Four-way admission needs the rollback `NUM_SPECULATIVE_TOKENS=5 MAX_NUM_SEQS=4`. Leave `--async-scheduling` off. Leave `VLLM_USE_BREAKABLE_CUDAGRAPH` on auto.

## Refuse-guards (`run.sh`)

- `--max-model-len` above 327680 on `fp8_e4m3` unless `FORCE_UNSAFE_CTX=1`. Native context is 1,048,576. A 1M request needs ~8.2 GiB of this hybrid layout and GB10 UMA OOMs above ~5.1 GiB. 1M on 2× Spark needs a packed `nvfp4_ds_mla` lane (different image/backend), not this pin.
- Any `MOE_BACKEND` other than `marlin` unless `FORCE_UNSAFE_MOE=1`. `flashinfer_cutlass` OOM'd spark2 during JIT after 90.67 GiB weights. Stay on Marlin until an SM121 W4A4 MoE path exists that does not JIT-OOM.
- Any `LINEAR_BACKEND` other than `marlin` unless `FORCE_UNSAFE_LINEAR=1`. `--moe-backend` covers routed experts only; the nvidia pack's layer 0-2 dense MLP is NVFP4, and auto selection picks a FlashInfer FP4 GEMM that JIT-compiles on sm_121 during the first profile forward.
- `SPEC=mtp` with `MODEL=nvidia/GLM-5.3-Flash-NVFP4` unless `FORCE_UNSAFE_SPEC=1`. Its layer-45 MTP weights are 13.84 GiB BF16 and not in the quant ignore list, so they cannot load or fit. LibertAI's MTP experts are NVFP4.
- `LANGUAGE_MODEL_ONLY` other than `0` unless `FORCE_UNSAFE_VISION=1`, and anything other than exactly `0` or `1` always.
- KV pin at or below `3886945403` (3.62 GiB) cannot hold 327680. Tony's 3.0 GiB pin is a 262144-ctx budget.

`--kv-cache-memory 4445787956` (4.14 GiB) stays the pin. Dropping it OOMs. Raising it boots but backfires under UMA pressure.

## Verify

```bash
python3 kit/render.py --check
python3 bench_decode.py                    # published score: prose, c=1 and 2, after serve is up
python3 smoke_vision.py                    # must not return HTTP 400 "is not a multimodal model"
```

Thinking-off smoke must not start `content` with chain-of-thought. Greedy count stays lossless (200 consecutive integers with thinking off). Published decode cells on this image are the LibertAIDAI pin; the nvidia pack is unmeasured until an exclusive TP=2 slot.

## Never touch

- Live HF tokens
- Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` as the serve image. `run.sh` refuses it. Build v8 → v9 → v10 → v11.
- Hand-edited generated README / `run.sh` blocks
- Advertising a 1M window on this fp8 pin
- Copying NVIDIA's TP=4 / `--enable-expert-parallel` / `--max-num-seqs 32` / `--max-num-batched-tokens 8192` onto 2× Spark
