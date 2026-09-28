# AGENTS.md — GLM-5.3-Flash-NVFP4 · 2× DGX Spark

Serve `nvidia/GLM-5.3-Flash-NVFP4` at TP=2. Local image chain through `glm53-sm121-v13` (v11 plus opt-in `GLM53_*` patches, `docker/README-v13.md`). Checkpoint `09b04e5`. Default drafter is DFlash2-7 (`SPEC=dflash2`) with its linears in NVFP4 W4A16 (`DRAFT_WEIGHTS=nvfp4`), the kpool tail-ring fix (`KPOOL_TAIL_FIX=1`) and adaptive verify at tau 0.2 (`ADAPTIVE_VERIFY=1`); the target's non-MoE linears stay BF16 (`TARGET_WEIGHT_GROUPS_INT8` empty). v11 rollback: `IMAGE=glm53-sm121-v11 DRAFT_WEIGHTS=bf16 KPOOL_TAIL_FIX=0 ADAPTIVE_VERIFY=0`. MTP-4 rollback is the LibertAI pack plus `SPEC=mtp` (`MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 SNAPSHOT_REV=caca4e6a4ebbd66f159d3d2fc256683fd6e27177 SPEC=mtp ADAPTIVE_VERIFY=0`); `run.sh` refuses `SPEC=mtp` on the nvidia pack. Vision is on (`LANGUAGE_MODEL_ONLY=0`).

Humans read [README.md](README.md). NVIDIA's card is a GB200 TP=4 / EP / 32-seq recipe. Do not copy those flags onto 2× Spark. LibertAI's GB10 recipe is a different stack (MTP-3, eager, 64K). Rollback: `MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 SNAPSHOT_REV=caca4e6a4ebbd66f159d3d2fc256683fd6e27177`.

## Working rules

- `recipe.yaml` is the source of truth for pins and generated blocks. Edit it, then `python3 kit/render.py`. Do not hand-edit `# BEGIN generated` or `<!-- BEGIN generated` blocks.
- Change one knob at a time against `python3 bench_decode.py`. Revert if it does not beat noise or it regresses another cell. Record the revert in `evidence/` (`trail.tsv`, `decision.tsv`).
- The v13 switches are recipe knobs: `DRAFT_WEIGHTS`, `TARGET_WEIGHT_GROUPS_INT8`, `KPOOL_TAIL_FIX`, `ADAPTIVE_VERIFY`, `ADAPTIVE_VERIFY_TAU`. Change their defaults in `recipe.yaml`, or set one per boot as env. `run.sh` turns them into `GLM53_*` env on both ranks; do not set those variables through `EXTRA_ENV`. Other `GLM53_*` switches still go through `EXTRA_ENV`. `VALIDATE_ONLY=1 ./run.sh` prints every `GLM53_*` the containers get (`==> glm53_env:`).
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`. Defaults in `run.sh` are `enp1s0f1np1` / `rocep1s0f1`.
- Keep `chat_template.jinja`. The stock HF template and the official NVIDIA Hub template always open `<think>`, so `enable_thinking: false` used to leak chain-of-thought into `content`. `thinking` is an alias of `enable_thinking` (the `glm45` parser's rule). The only allowed difference from the Hub copy in `tests/data/` is the generation prompt; `tests/test_chat_template.py` enforces it.
- Leave vision on. `LANGUAGE_MODEL_ONLY=1` is refused unless `FORCE_UNSAFE_VISION=1`. Cap is `--limit-mm-per-prompt '{"image":4,"video":1}'`. Do not skip MM profiling. Vision was already on in every LibertAI measurement; the cap is new.
- Published decode score is prose only. Do not score decode from structured, code, or other cells.
- Do not turn on InstantTensor. That loader killed TP=2 ranks here.
- Do not set `VLLM_GLM53_MOE_INPUT_SCALE=1.0`. That constant underflows per 16-element block.
- `run.sh` already calls `maybe_drop_caches`. It no-ops without passwordless sudo.

`DRAFT_REV` pins the DFlash2 snapshot (default `7d74cdd`, from `recipe.yaml`; a full 40-hex sha). `bf582e4` and `dc77ff1` are weights-only updates with the same `config.json`. Do not move the default until the A/B against the pin lands.

Default occupancy is DFlash2-7 at two sequences. Four-way admission needs the rollback `NUM_SPECULATIVE_TOKENS=5 MAX_NUM_SEQS=4`. Async scheduling is already auto-on for DFlash; do not pass `--no-async-scheduling`. Leave `VLLM_USE_BREAKABLE_CUDAGRAPH` on auto.

## Refuse-guards (`run.sh`)

- `--max-model-len` above 327680 on `fp8_e4m3` unless `FORCE_UNSAFE_CTX=1`. Native context is 1,048,576. A 1M request needs ~8.2 GiB of this hybrid layout and GB10 UMA OOMs above ~5.1 GiB. 1M on 2× Spark needs a packed `nvfp4_ds_mla` lane (different image/backend), not this pin.
- Any `MOE_BACKEND` other than `marlin` unless `FORCE_UNSAFE_MOE=1`. `flashinfer_cutlass` OOM'd spark2 during JIT after 90.67 GiB weights. Stay on Marlin until an SM121 W4A4 MoE path exists that does not JIT-OOM.
- Any `LINEAR_BACKEND` other than `marlin` unless `FORCE_UNSAFE_LINEAR=1`. `--moe-backend` covers routed experts only; the nvidia pack's layer 0-2 dense MLP is NVFP4, and auto selection picks a FlashInfer FP4 GEMM that JIT-compiles on sm_121 during the first profile forward.
- `SPEC=mtp` with `MODEL=nvidia/GLM-5.3-Flash-NVFP4` unless `FORCE_UNSAFE_SPEC=1`. Its layer-45 MTP weights are 13.84 GiB BF16 and not in the quant ignore list, so they cannot load or fit. LibertAI's MTP experts are NVFP4.
- `LANGUAGE_MODEL_ONLY` other than `0` unless `FORCE_UNSAFE_VISION=1`, and anything other than exactly `0` or `1` always.
- KV pin at or below `3886945403` (3.62 GiB) cannot hold 327680. Tony's 3.0 GiB pin is a 262144-ctx budget.
- `EXTRA_ENV` with `FLASHINFER_JIT_VERBOSE=1` unless it also sets `FLASHINFER_JIT_DEBUG=0`. This image's FlashInfer reads verbose as debug when debug is unset (`flashinfer/jit/core.py:525-528`), so every JIT kernel builds `-O0 --device-debug`; on 2026-09-27 that pushed spark1 under the PROFILE floor.
- Any v13 knob that is on while `IMAGE` is not a `glm53-sm121-v13*` tag, unless `FORCE_UNSAFE_IMAGE=1`. Older images ignore `GLM53_*`.
- `ADAPTIVE_VERIFY=1` with any `SPEC` but `dflash2`. The verify width comes from DFlash2's selector scores.
- `TARGET_WEIGHT_GROUPS_INT8` with `draft` (`DRAFT_WEIGHTS` owns the drafter) or a name outside `patch_v13_fp8.py`'s groups; `DRAFT_WEIGHTS` other than `bf16` / `nvfp4`; `KPOOL_TAIL_FIX` or `ADAPTIVE_VERIFY` other than `0` / `1`; `ADAPTIVE_VERIFY_TAU` that is not a decimal strictly between 0 and 1.
- `EXTRA_ENV` setting a knob's variable: `GLM53_NVFP4_W4A16`, `GLM53_INT8_W8A16`, `GLM53_KPOOL_TAIL_FIX`, `GLM53_ADAPTIVE_VERIFY`, `GLM53_ADAPTIVE_VERIFY_TAU`.

`--kv-cache-memory 4445787956` (4.14 GiB) stays the pin. Dropping it OOMs. Raising it boots but backfires under UMA pressure.

## Measured (E1, 2026-09-27)

- Build once on the head and ship it: `docker save glm53-sm121-v13 | ssh spark2 docker load` took 253 s and gave both nodes the same image ID. Separate builds do not; E0 ran two different v11 builds (`evidence/e1-v13-build/`). `run.sh` warns when the two image IDs differ.
- `JIT_CACHE=1` pays from the second boot on an image: ready 21.2 → 16.3 min, init engine 333 → 46 s, no compiler process at boot or while serving, and the first c=2 wave's TTFT falls from 8.7 s to 0.6 s. The cache is ~131 MB per node (`evidence/e1b-v13-warmcache-draft-bf582e4/`).
- Non-MoE linears (KDA, MLA, shared experts, lm_head read twice, drafter), per rank per verify step at M=8: BF16 41.1 ms, Marlin FP8 W8A16 20.8 ms, Marlin NVFP4 W4A16 12.4 ms. These are kernel times; a serve saves at most ~20 ms (FP8) or ~29 ms (NVFP4) of a ~115 ms step (`evidence/e1-microbench/`).

## Verify

```bash
python3 kit/render.py --check
python3 -m unittest discover -s tests      # CPU: VALIDATE_ONLY guards, v13 knobs, worker forwarding, JIT cache, template kwargs + Hub parity
GLM53_V11_SRC=/path/to/v11src python3 docker/test_v13_misc.py   # v13 patches; the fp8 test reads the same var
python3 bench_decode.py                    # ruler v2 fast gate after serve is up; cell A (prose, c=1) is the published score
python3 smoke_vision.py                    # must not return HTTP 400 "is not a multimodal model"
```

After a boot with `JIT_CACHE=1`, `~/projects/data/glm53-jit-cache/<image id>/` is non-empty on both nodes, and a second boot on the same image runs no FlashInfer / DeepGEMM nvcc or ptxas. Output must stay byte-identical with the cache on and off.

Thinking-off smoke must not start `content` with chain-of-thought. Greedy count stays lossless (200 consecutive integers with thinking off). The README decode table is still the 2026-09-02 rebench: the LibertAIDAI pin on v11, old ~98-token prose prompt (cell K), no v13 switches. The nvidia pack's ruler-v2 receipts are in `evidence/` (E0 on v11, E1-E3 on v13). Do not change the `recipe.yaml` `measured` rows until a final measurement of the current defaults replaces the table.

## Never touch

- Live HF tokens
- Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` as the serve image. `run.sh` refuses it. Build v8 → v9 → v10 → v11 → v13.
- Hand-edited generated README / `run.sh` blocks
- Advertising a 1M window on this fp8 pin
- Copying NVIDIA's TP=4 / `--enable-expert-parallel` / `--max-num-seqs 32` / `--max-num-batched-tokens 8192` onto 2× Spark
