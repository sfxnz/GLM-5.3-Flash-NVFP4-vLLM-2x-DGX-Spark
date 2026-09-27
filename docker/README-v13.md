# v13 image layer: opt-in runtime patches

`Dockerfile.sm121-v13` builds on `glm53-sm121-v11` and adds two Python-only patch layers:

- `patch_v13_misc.py` (this file documents it)
- `patch_v13_fp8.py` (from the fp8 lane)

Each behaviour is gated by a `GLM53_*` environment variable, and all of them are off by default. With no `GLM53_*` set, v13 serves exactly like v11. Nothing in the v11 hot path changes until a switch is set.

```bash
docker build -f docker/Dockerfile.sm121-v13 -t glm53-sm121-v13 docker   # needs docker/patch_v13_fp8.py
GLM53_V11SRC=/path/to/v11/dist-packages/vllm python3 docker/test_v13_misc.py   # CPU tests
```

The patch script takes the vllm root as `argv[1]`. Each edit is an exact-substring replace. If the replacement text is already present, the edit is skipped, so reruns are no-ops. If any anchor is missing or appears more than once, the script refuses before writing anything. `test_v13_misc.py` checks the following on a copy of the v11 tree:

- Applying twice is idempotent.
- Every anchor appears exactly once.
- The refusal on drift fires.
- `py_compile` passes on every touched file.
- The pure-logic pieces behave as intended.
- The KDA kernel is bit-exact. This runs under the Triton CPU interpreter and compares trim on and trim off against v11.

**Forwarding (recipe lane).** `run.sh` only passes the variables it lists. Every `GLM53_*` switch needs `-e GLM53_…` in `env_args` on both ranks, and it must also appear in the worker's ssh command line. Set the switches identically on head and worker: TP ranks must build identical graphs and workspaces.

## Switches

| Env var | Changes | Expected effect | Risk |
|---|---|---|---|
| `GLM53_ROUTER_FP32=1` | `Glm5NextMoE` sets `gate.allow_cublas_router_gemm`, so the router GEMM runs `torch.mm(x, W.T, out_dtype=float32)` (GateLinear tier 5) instead of BF16 `F.linear` + `.to(fp32)` (tier 6) | Router logits stay fp32, as `moe_router_dtype` asks, which restores parity with the SM90/SM100 path. About 1e-3 score shift at top-8 boundaries (MOE-3). Removes 42 cast kernels per step (about +0.1 ms). | cuBLAS `out_dtype` on sm_121. An unsupported case errors at warmup or capture, which makes the failure visible at boot. |
| `GLM53_INDEXER_WS_FACTOR=<int ≥1>` | `get_max_prefill_buffer_size` returns `max_model_len * factor` instead of `* 40`. The chunk planner and the indexer op both size from it. | At 327680 context, factor 1 cuts the indexer gather workspace from 1.61 GiB to 41 MiB per rank, freeing about 1.57 GiB of UMA per rank (XP-4). | Long prefills split into more indexer chunks, so check the 318k needle and prefill speed. An invalid value raises `ValueError` at boot. |
| `GLM53_MHC_WARMUP=1` | New `warmup/glm5next_mhc_warmup.py`, called from `kernel_warmup`. It runs `hc_pre`, `hc_fused_post_pre` and `hc_post` of one Glm5Next layer once per distinct TileLang specialisation up to `max_num_batched_tokens`. At 2048 tokens on 48 SMs that is 14 sizes and 12 `n_splits` classes. | No more ~5-7 s first-shape TTFT stalls while serving (NMK-4 / XP-9). Boot takes about 60-70 s longer with a cold TileLang cache. | 64 MiB of transient bf16 tensors at 2048 tokens. The compiles move into boot. |
| `GLM53_KDA_TRIM=1` | `fused_recurrent_kda` stops calling `.contiguous()` on q/k/v/beta when each token's `[H, D]` block is dense. The kernel takes the token strides instead (`STRIDED_QKVB`). The GDN call site passes `STRIDED_QKVB=False`. | 4 fewer copy kernels per KDA layer per verify step, about 136 kernel launches in total. The estimate is about 0.3-0.7 ms per step (NMK-7), not measured. Output is bit-exact: the kernel reads the same elements. | Pointer math. Covered by the interpreter test, including spec-decode state slots and COMPUTE_GATE. Varlen, B=1 path only. Anything else falls back to `.contiguous()`. |
| `GLM53_SKIP_MTP_WEIGHTS=1` | When the speculative method is not `mtp`, `Glm5NextModel` registers `model.language_model.layers.45.` (and the other name forms) in `ep_weight_filter.SKIP_NAME_PREFIXES`. The default safetensors iterator then skips those tensors before `get_tensor`. | Each rank reads 13.84 GiB less (889 tensors in the nvidia pack, PR11-9), so the load gets roughly 30-50 s shorter. Resident memory does not change. | None with DFlash2, because the draft's tensor names are `layers.0-4`. `SPEC=mtp` ignores the switch. Only the default loader applies the filter. `--load-format fastsafetensors`, `instanttensor` and multithread loading do not. |
| `GLM53_DFLASH_PREFIX_CACHE_FIX=1` | A port of tonyd2wild's `patch_prefix_cache_draft_group.py` into `kv_cache_coordinator.py`, with two changes. (1) When no group is flagged EAGLE, only the DFlash draft sliding-window group gets the EAGLE last-block drop, where v11 applies it to every group. (2) The draft group never shrinks the hit that the target MLA and KDA groups agreed on. A shorter draft hit is dropped, so the draft gets fresh pages. | Prefix-cache hits come back with DFlash2 (EXT-5). Tony measured `0 hits / 35,280 queries → floor(len/2304)*2304` cached, and a 5178-token repeat going from 4.3 s to 0.7 s. | A dropped draft hit leaves the draft window without KV for the cached span, so acceptance is lower on those requests. Output is still lossless because the target verifies every draft token. One deviation from Tony: a shorter draft hit also clears draft blocks recorded in an earlier fixed-point pass. Without that, a stale longer block list could survive. |

## GPU validation (Sparks, one switch at a time)

Follow AGENTS.md: exclusive GPUs, one knob per boot, record everything in `evidence/iter-<name>/` with `trail.tsv` and `decision.tsv` rows. Read memory with `free -h`, never with `nvidia-smi`. Use `L` as shorthand for both ranks' engine logs (`docker logs glm53-flash-nvfp4` on spark1 and on spark2).

**P3.0 gate: v13 with no `GLM53_*` set equals v11.** Boot `IMAGE=glm53-sm121-v13` with default settings.

- `L | grep -c GLM53_` must print 0 on both ranks.
- Greedy count-200 must be lossless.
- Thinking-off smoke must not start `content` with chain-of-thought.
- Greedy prose output must be byte-identical to the v11 capture.
- `python3 bench_decode.py` must be within noise of v11.

**ROUTER_FP32**

- `L | grep "GLM53_ROUTER_FP32: MoE router GEMM uses cuBLAS"` must print one line per rank.
- Boot must reach ready. A cuBLAS `out_dtype` failure shows up during capture.
- Count-200 lossless, thinking-off smoke, Tier 0.
- Expect small greedy divergences against the flag-off run. That is the intended fidelity change, so compare on the quality eval, not byte equality.
- Bench prose c=1 and c=2 must be within noise.

**INDEXER_WS_FACTOR=1**

- `L | grep "GLM53_INDEXER_WS_FACTOR=1: indexer prefill buffer 327680 entries"`.
- After ready, run `free -h` on both nodes and log `MemAvailable` at ready, after smoke and after a 32k needle.
- Expect about +1.5 GiB per node against a same-day factor-40 control.
- The 318k needle must still pass, and prefill tok/s on a 64k prompt must stay flat.

**MHC_WARMUP**

- Before `Application startup complete`, both ranks must log `GLM53_MHC_WARMUP: compiling ... token sizes [1, 8, 17, ...]` and `GLM53_MHC_WARMUP: finished <n> sizes in <s> s` (n = 14 at `max_num_batched_tokens` 2048).
- After ready, send single prompts of about 30, 100, 300, 700, 1500 and 2000 tokens, then one c=2 pair.
- `L | grep -c "JIT compilation during inference: mhc"` must be 0 on both ranks. `jit_monitor` logs each kernel name only once, so also check that `L | grep -c "TileLang begins to compile"` does not grow after ready.
- Every request's TTFT must stay under 1 s.
- Run 1 and run 3 of `bench_decode.py` should no longer differ by the ~6 s first-wave TTFT spike.

**KDA_TRIM**

- Check that the switch is set: `docker exec glm53-flash-nvfp4 env | grep GLM53_KDA_TRIM` on both nodes.
- Greedy outputs (prose, structured and count-200, temperature 0) must be byte-identical to the same image with the switch off.
- nsys on 30 verify steps at c=1 should show 136 fewer copy kernels (`elementwise`/`copy_`) per step.
- Bench prose c=1 and c=2: keep the switch only if it beats noise, per AGENTS.md.

**SKIP_MTP_WEIGHTS** (with `SPEC=dflash2`)

- `L | grep "GLM53_SKIP_MTP_WEIGHTS: not reading ('model.language_model.layers.45.'"`.
- `Loading weights took` must drop against the nvidia-pack control, with an expected read of 176.5 GiB instead of 190.4 GiB.
- `Model loading took X GiB` must stay the same to within 0.01.
- Count-200 must stay lossless.
- With `SPEC=mtp`, that log line must be absent and MTP acceptance must be normal.

**DFLASH_PREFIX_CACHE_FIX** (with `SPEC=dflash2`)

- Add `EXTRA_ARGS="--enable-prompt-tokens-details"`.
- On the head, `L | grep "GLM53_DFLASH_PREFIX_CACHE_FIX: EAGLE block drop on draft KV group(s)"`.
- Send one identical 9216-token prompt three times at temperature 0 with `max_tokens` 32.
- After each send, record `curl -s localhost:8000/metrics | grep -E 'prefix_cache_(queries|hits)_total'` and `usage.prompt_tokens_details.cached_tokens`.
- Measure the switch-off baseline first. EXT-5 predicts 0 hits, and that prediction is unverified locally.
- With the switch on, sends 2 and 3 must report cached tokens of at least 4608, and at most 6912. The count is a multiple of 2304 (the block size).
- TTFT must drop accordingly.
- All three completions must be byte-identical to each other and to the switch-off run.
- Count-200 must stay lossless.
- Compare the spec-decode acceptance counters on sends 2 and 3 against send 1. A large drop means dropped draft hits (risk above).

## Not in this layer

- **f_b/g_b GEMM merge (NMK-7 part b).** Not done because it cannot be made bit-exact by construction. f_a and g_a are adjacent 128-wide slices, so one GEMM would need either of two things:
  - A block-diagonal `[4096, 256]` weight. Its f and g outputs come out as strided halves, and the recurrent kernel and `o_norm` would then copy them back.
  - A `bmm` over stacked weights. That swaps cuBLAS `gemm` for `gemmStridedBatched`, and Inductor may add a layout copy.

  Neither is guaranteed to reproduce the separate GEMMs' bits, and the gain is at most 34 launches (≈0.1-0.17 ms per step).
- **Scheduler half of upstream vLLM #54163.** This is the `last_cache_position` back-off for DFlash in mamba-align mode. Tony's coordinator-only patch already measured `floor(len/2304)*2304` hits on the same design. The back-off only costs prompts shorter than two blocks. Revisit if the DFLASH_PREFIX_CACHE_FIX check shows sub-2-block misses.
- **Persistent TileLang, Triton and DeepGEMM caches.** These are `run.sh` volume mounts, which belong to the recipe lane. With `/root/.tilelang` and `/root/.cache` persisted, MHC_WARMUP compiles hit the cache after the first boot.
