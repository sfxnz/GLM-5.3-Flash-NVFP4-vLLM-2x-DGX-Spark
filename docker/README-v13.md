# v13 image layer: opt-in runtime patches

`Dockerfile.sm121-v13` builds on `glm53-sm121-v11` and adds Python-only patch layers, among them:

- `patch_v13_misc.py` (this file documents it)
- `patch_v13_fp8.py` (from the fp8 lane)
- `patch_v13_determinism.py` (`GLM53_DETERMINISTIC_MLA_INDEX`, and the lm_head check behind `LOGITS_FP32`; this file documents it)

Each behaviour is gated by a `GLM53_*` environment variable, and all of them are off by default. With no `GLM53_*` set, v13 serves exactly like v11. Nothing in the v11 hot path changes until a switch is set.

```bash
docker build -f docker/Dockerfile.sm121-v13 -t glm53-sm121-v13 docker   # needs docker/patch_v13_fp8.py
GLM53_V11_SRC=/path/to/v11src python3 docker/test_v13_misc.py                 # CPU tests
GLM53_V11_SRC=/path/to/v11src python3 -m unittest docker/test_v13_fp8.py -v   # CPU tests
GLM53_V11_SRC=/path/to/v11src python3 docker/test_v13_determinism.py          # CPU tests; kernel test needs torch + triton
```

Both test files read `GLM53_V11_SRC`: the directory that holds the `vllm/` package tree copied out of `glm53-sm121-v11` (for example a `dist-packages` copy). Without it the source-dependent tests skip. `test_v13_misc.py` still accepts the old `GLM53_V11SRC`, which points at `vllm/` itself, when `GLM53_V11_SRC` is unset.

The patch script takes the vllm root as `argv[1]`. Each edit is an exact-substring replace. If the replacement text is already present, the edit is skipped, so reruns are no-ops. If any anchor is missing or appears more than once, the script refuses before writing anything. `test_v13_misc.py` checks the following on a copy of the v11 tree:

- Applying twice is idempotent.
- Every anchor appears exactly once.
- The refusal on drift fires.
- `py_compile` passes on every touched file.
- The pure-logic pieces behave as intended.
- The KDA kernel is bit-exact. This runs under the Triton CPU interpreter and compares trim on and trim off against v11.

`test_v13_determinism.py` runs the same apply checks on its three files. It also runs the patched index kernel under the Triton CPU interpreter: with the switch off the kernel's output matches v11 bit for bit, and with it on the output is a stable column-order compaction from one program per row. The lm_head test shows that v11's `_apply_head` raises on an `UnquantizedLinearMethod` head, and that the patched one returns fp32.

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
| `GLM53_DETERMINISTIC_MLA_INDEX=1` (`patch_v13_determinism.py`; set it with `EXTRA_ENV`, which `run.sh` passes to both ranks) | The sparse-MLA index conversion compacts each row's valid KV slots into `[0, valid_count)`. With 17 tiles per 2176-wide row it reserves slots through `tl.atomic_add`, so the order follows tile scheduling (`sparse_utils.py:113`, `:146-151`). With the switch on, a compacted row gets one Triton program padded to 4096 lanes (16 warps, masked loads), which keeps the input column order. `sparse_attn_indexer_kpool` also sorts `pool_topk` per row (`torch.sort`, `[rows, 512]` int32) before `expand_pools_and_append_tail`, so the list is ascending whatever order the top-k kernels emit. Off, the kernel's `NUM_COLS` defaults to 0, its branch compiles out, and both host helpers return their input. | Run-to-run fixed kv-index order, so the FA2 MLA kernel accumulates its online softmax and bf16 P in the same order every run. Within-boot greedy divergence (e0: 14/20 prompts) and Tier-0 top-1 disagreement (e0: 1.55%) should drop sharply. Estimated cost at most ~0.2 ms per ~116 ms verify step: 11 MLA layers, each with one program per row instead of 17 tiles plus one small sort. Not measured. | Selection ties at the top-k threshold are still decided by the top-k kernel. Cross-boot sources are untouched (KDA autotune, JIT flags). Tier 0's nll documents prefill as one ≤2048-row chunk without top-k, where the tiles should already run in order, so the nll gain may be smaller than the greedy gain (quality/README.md, Determinism). The MLA kernel is not changed. |
| `LOGITS_FP32=1` (recipe knob, `run.sh`) | `run.sh` passes `--hf-overrides '{"text_config":{"head_dtype":"float32"}}'`. `patch_v13_determinism.py` lets `LogitsProcessor._apply_head` accept `UnquantizedLinearMethod`, which ModelOpt gives the excluded lm_head (`modelopt.py:185-187`). v11 accepts only `UnquantizedEmbeddingMethod` there and raises `ValueError` at the first logits (`logits_processor.py:144`). | fp32 logits from `torch.mm(out_dtype=float32)` on the bf16 lm_head, for the target and the DFlash2 drafter, which shares that lm_head. The drafter's top-k values were already fp32. | The logits buffers double (634 MB per 1024-row `prompt_logprobs` chunk). cuBLAS `out_dtype` on sm_121 carries the same risk as `ROUTER_FP32`. A quantized lm_head (the `GLM53_FP8` lm_head group) still raises. `run.sh` refuses it on `glm53-sm121-v11`. |

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

**DETERMINISTIC_MLA_INDEX** (A/A with and without the switch, both on `IMAGE=glm53-sm121-v13`, everything else identical)

- Boot the off arm with no `GLM53_*`, and the on arm with `EXTRA_ENV='GLM53_DETERMINISTIC_MLA_INDEX=1'`.
- On the on arm, `L | grep "GLM53_DETERMINISTIC_MLA_INDEX: kpool pools sorted per row"` must print at least one line per rank. On the off arm it must print nothing.
- On each boot, run `python3 quality/tier0.py record --name v13-detidx-<off|on>`, then `compare --ref v13-detidx-<off|on>`. `record` captures nll and greedy twice on the same boot, so its `nll.rerun` and `greedy.aa` are the within-boot A/A.
- Compare the two arms' A/A:
  - `nll.rerun.top1_agree`: e0 had a 1.55% disagreement.
  - |dNLL| and KL.
  - `greedy.aa`: the number of prompts that diverge (e0: 14/20) and the hazard.
- Both should drop sharply on the on arm. If greedy converges but nll top-1 does not, the nll noise comes from outside the compaction. Tier 0's nll documents prefill as one ≤2048-row chunk, where the tiles should already be in order. See quality/README.md, Determinism.
- Count-200 must stay lossless, and the thinking-off smoke must pass.
- Run the ruler v2 fast gate on each boot: `python3 bench_decode.py --cells A,B --out evidence/<run>/bench-<arm>-<boot>`. Then run `python3 kit/compare.py --a <off bench.json files> --b <on bench.json files>`.
  - `step_ms` must show no regression beyond noise. The expected cost is about 0.2% or less.
  - `acceptance_len` may move, because the greedy drafts now follow a different, fixed order.
- For a verdict, run 2 boots per arm, ABAB.

**LOGITS_FP32** (`LOGITS_FP32=1 IMAGE=glm53-sm121-v13`)

- The boot must reach ready. On v11 it raises `ValueError: A head_dtype different from the model dtype is only supported for an unquantized lm_head` at the first logits, so `run.sh` refuses that image.
- On both ranks, `docker inspect glm53-flash-nvfp4 --format '{{.Args}}'` must show `--hf-overrides {"text_config":{"head_dtype":"float32"}}`.
- Tier-0 capture: with bf16 logits, top-20 lists often hold two exactly equal logprobs (an exact bf16 tie). With fp32 logits those ties should all but vanish. This is the cheap check that fp32 reached the target head.
- Run the Tier-0 A/A as above, and watch `free -h` during `record` (larger `prompt_logprobs` transient).
- Count-200 lossless.
- Fast gate `step_ms` within noise.

## Not in this layer

- **f_b/g_b GEMM merge (NMK-7 part b).** Not done because it cannot be made bit-exact by construction. f_a and g_a are adjacent 128-wide slices, so one GEMM would need either of two things:
  - A block-diagonal `[4096, 256]` weight. Its f and g outputs come out as strided halves, and the recurrent kernel and `o_norm` would then copy them back.
  - A `bmm` over stacked weights. That swaps cuBLAS `gemm` for `gemmStridedBatched`, and Inductor may add a layout copy.

  Neither is guaranteed to reproduce the separate GEMMs' bits, and the gain is at most 34 launches (≈0.1-0.17 ms per step).
- **Scheduler half of upstream vLLM #54163.** This is the `last_cache_position` back-off for DFlash in mamba-align mode. Tony's coordinator-only patch already measured `floor(len/2304)*2304` hits on the same design. The back-off only costs prompts shorter than two blocks. Revisit if the DFLASH_PREFIX_CACHE_FIX check shows sub-2-block misses.
- **Persistent TileLang, Triton and DeepGEMM caches.** These are `run.sh` volume mounts, which belong to the recipe lane. With `/root/.tilelang` and `/root/.cache` persisted, MHC_WARMUP compiles hit the cache after the first boot.

## GLM53_ADAPTIVE_VERIFY: fixed-shape adaptive verification (`patch_v13_verify.py`)

DFlash2-7 verifies 8 rows per request per step, and every row pulls its own top-8 routed experts. On prose, positions 4-7 are almost never accepted (E0 cell A: .640 .345 .169 .065 .029 .012 .003). Code uses the whole block (cell B: .817 .657 .515 .406 .322 .238 .182). vLLM's adaptive verification (#52228) changes the per-request query length, which the DSA indexer, the KDA backend and the FlashInfer draft attention reject on sm_121. This switch keeps every shape and stops paying expert reads for the rows the drafter does not believe in.

| Env var | Default | Meaning |
|---|---|---|
| `GLM53_ADAPTIVE_VERIFY=1` | off | Master switch. Off means no hook is bound, nothing is allocated and the graphs are the v11 graphs. |
| `GLM53_ADAPTIVE_VERIFY_TAU=<f>` | `0.1` | Verify width m = the number of leading drafts whose running product of DFlash2 top probabilities is ≥ tau. `0` turns the confidence rule off. |
| `GLM53_ADAPTIVE_VERIFY_MAX=<n>` | k | Fixed per-request cap in 1..k. With `TAU=0`, every step verifies min(n, drafts) positions. |

- m is always in [1, MAX], so at least one draft is verified.
- The DFlash2 probability of draft step i is the maximum of the softmax over the selector walk's realized scores (`_selector_scores`). These are the same top-K scores the probabilistic draft path uses as q.
- `TAU` needs DFlash2. `TAU=0` with `MAX<k` works with any V2 drafter.
- Only the V2 runner creates the state, which means `SPEC=dflash2`. `SPEC=mtp` logs nothing.
- At boot it refuses PP, DP, PCP, DBO, sequence-parallel MoE, monolithic MoE kernels, a rejection sampler that is not the stock V2 class, and vLLM's own adaptive verification.

```bash
GLM53_V11_SRC=/path/to/v11src python3 docker/test_v13_verify.py   # CPU; torch + triton add the kernel tests
```

### Mechanism

One predicate on logit rows drives everything: `local_pos > verify_len[req_state]`. Local position 0 is the anchor (the last sampled token) and local position i is draft d_i. The state lives in `v1/worker/gpu/spec_decode/glm53_adaptive_verify.py`.

1. `prepare()` runs in the runner after `prepare_attn`, outside every graph and before `model_state.prepare_attn` builds the attention metadata. It zeroes `masked[:num_tokens_after_padding]`, scatters the predicate onto the input rows `logits_indices`, and stores each row's anchor row as `logits_indices - local_pos`.
2. `compute_kpool_tail_slot_mapping` (`indexer.py`) runs in the kpool tail metadata build, which is eager. After v11 fills the tail slots, it sets the slot of every masked row to -1. The indexer's K-pool update kernel then skips that row's stash into the `pos % index_kpool` tail ring (`kpool_compress.py:596`). This costs one eager kernel per step. The lossless section says why the ring needs it.
3. `remap()` runs in `BaseRouter._select_experts`, after `_compute_routing` and before the census capture and EPLB. That code is inside the opaque `moe_forward` op, so the remap is captured in the FULL and PIECEWISE graphs. A masked row takes its anchor's top-8 ids with weight 0. This costs 4 fixed-shape kernels per MoE layer: the out-of-place `masked_fill` is a clone plus `masked_fill_`, then a gather and a `where`.
4. `mask_drafts()` runs in `RejectionSampler._verify`, after `apply_sampling_params`. A masked row's draft id becomes -1 in the tensor handed to `rejection_sample`. The sampling-parameter kernels still see the real ids.
5. `record()` runs in the runner after the draft. It sets `verify_len[idx_mapping] = m` for the drafts just proposed. The next step's 1-4 read it, in stream order.

The buffers are persistent: `masked` and `anchor` have `max_num_batched_tokens` rows, and `verify_len` has `max_num_seqs` entries. The indexer hook holds a reference to `masked`. Nothing calls `.item()` or `.cpu()`, and no shape depends on data. `anchor[t] <= t` always holds, so the gather stays inside the batch at every graph size. Both TP ranks compute m from the draft scores that already give them identical draft tokens, so both ranks mask the same rows.

### Why it is lossless (index arithmetic of this tree)

Take one verify request with n drafts (n = k = 7 unless the scheduler truncated) and width m < n.

- **Layout.** The runner writes the query as input rows q..q+n = [x_0 = anchor, x_1 = d_1, ..., x_n = d_n] (`input_batch.py:415,433,443`). The logit rows are the last n+1 rows: `logits_indices[c+j] = q+j` and `expanded_local_pos[c+j] = j`.
- **Which logits test which draft.** The sampler's draft vector is `input_ids[logits_indices]` (`rejection_sampler.py:259`), so its row c+j holds x_j. Iteration i of `_rejection_kernel` tests `draft_sampled[c+i+1] = d_{i+1}` against logit row c+i, which is the target distribution after x_0..x_i. So the logits of input i predict the token after input i, and the draft at local position j is tested by row j-1.
- **Forced rejection.** Masking sets d_j = -1 for every j > m. At i = m the kernel meets d_{m+1} = -1 (`rejection_sampler_utils.py:558`):
  - Greedy: `accepted &= is_valid_draft` (:588) is false, so the kernel stores the argmax of row m at `sampled[m]` (:593) and stops.
  - Sampled (standard, block or synthetic): `verifying &= is_valid_draft` (:565) stops before any test. `_resample_kernel` sees `rejected_draft_token < 0` and draws from row m's raw target logits (:767-770), with Gumbel noise keyed by (seed, pos of row m).

  Under k' = m, the same row m is the bonus row. Greedy takes its argmax (the leftmost maximum on both paths), and sampled draws from its raw target logits with the same key. A rejection before m is identical in both cases, because test i < m sees the same row and the same uniform keyed by (seed, pos of row i). Block verification's look-ahead at i = m-1 also sees no next draft in both cases: the placeholder in one, the end of the drafts in the other. So the emitted tokens and `num_sampled` equal k' = m speculation, and the sampler reads rows 0..m only.
- **Rows 0..m do not see the masked inputs.** The masked inputs are rows m+1..n. Every Glm5Next block is either per-token or causal:
  - Per-token: embedding, norms, mHC, shared and dense MLPs, lm_head, and routed MoE (routing is per token).
  - Causal: MLA and the DSA indexer read positions ≤ their own, and KDA and its conv1d advance token by token.

  The remap only rewrites the masked rows' own ids and weights. In exact arithmetic, rows 0..m equal the rows of a k' = m forward. The sampling parameters of row j (penalties, bad words, thinking budget, logit bias, grammar) read x_1..x_j only, and they get the real ids.
- **State after the step.** `num_sampled = a+1 ≤ m+1`, where a is the number of accepted drafts. Masked rows sit at positions after x_a. The state they write falls into three classes.
  - **Position-addressed, rewritten before it is read.** The next step for the request starts at pos(x_a)+1, at or before the first masked position (a ≤ m). With k drafts its rows cover every masked position. With fewer, its queries stop before the positions it does not cover.
    - MLA KV. The step writes all its rows' KV before attention reads it, and a query reads only positions at or before its own.
    - Indexer pool cache. A masked row at a pool boundary (pos % 4 = 3) compresses that pool from garbage. A query scores only complete pools: the builder floor-divides its token seq_len by `compress_ratio` = index_kpool (`indexer.py:1277-1284`). The indexer runs the K-pool update over all of a step's rows, in position order, before it scores (`sparse_attn_indexer_kpool.py:725` before `:788`). So the next step recompresses that pool before any query reads it.
    - KDA and conv states. The next step starts from spec-state column `num_accepted - 1 = a` (`fused_recurrent.py:114-121`, with `num_accepted = max(num_sampled, 1)` from `mamba_hybrid.py:342`). The conv window offset is also `num_accepted - 1`. Neither starts from a masked row's column.
    - DFlash2 draft KV. The context K/V precompute also covers masked rows (`dflash/speculator.py:416`), at positions after `last_valid_pos = pos(x_a)` (:522, :531). The next draft query block covers pos(x_a)+1 .. pos(x_a)+k+1 (:554) and writes those slots before its attention reads them. The context it reads ends at `last_valid_pos` (:593).
  - **Position-aliased: the kpool tail ring. Lossless only with stash suppression.** The indexer keeps each request's open pool in a ring of `index_kpool` = 4 slots addressed by `pos % 4` (`indexer.py:565`; 09b04e5 config). Every verify row stashes its indexer K and gate there, gated only on its tail slot being ≥ 0 (`kpool_compress.py:596, 687-697`). A pool completion reads the other three slots (:609-653).
    - A row at position c+4 lands on the slot of position c. If c is committed and its pool is still open when the step ends, nothing rewrites that slot before the pool completes in a later step.
    - The first version of this patch let masked rows stash. On v11's kernel, with history 0..8, an anchor at 9, m = 2 and a = 0, masked rows 12..16 overwrote the slots of committed positions 8 and 9, so pool [8..11] came out different from k' = 2's.
    - The patch now gives every masked row tail slot -1 (step 2 of the mechanism), and the kernel skips its stash. A masked row at a pool boundary still compresses its pool, reading ring block 0 instead of its own (`kpool_compress.py:587`). That pool entry is in the first class.
    - So the ring holds exactly the stashes of rows 0..m, as under k' = m.
  - **Shared with baseline k=7: rejected rows in the ring.** Rows a+1..m are rejected, but they stash the same way under masked k=7 and under k' = m. Baseline k=7 stashes all of rows a+1..7.
    - When the verify window crosses into the next pool while a committed pool is still open, a rejected row overwrites a committed slot.
    - This is a pre-existing v11 defect of every draft length k ≥ 2, DFlash2 and MTP alike. `KpoolTailSpec` and the kernel docstring assume completed pools never roll back, and speculative decoding breaks that assumption.
    - It matters above index_topk = 2048 tokens, where the indexer really ranks pools.
    - This lane does not fix it. Adaptive verify equals k' = m exactly and overwrites fewer committed slots than baseline k=7.
    - The root fix belongs in its own lane: a ring of at least kpool - 1 + k + 1 slots addressed `pos % R`, or a re-stash after acceptance.
- **Conclusion.** The step's output distribution equals k' = m speculation from the same drafts, and so does every piece of state a later step reads. For sampling that is lossless by Leviathan et al., and for greedy by construction, up to the ring defect that k' = m and baseline k=7 share.

`test_v13_verify.py` checks this on v11's own kernels under the Triton interpreter.

- **Within a step.** The test runs the real layout kernels, then `prepare()` and `mask_drafts()`, then `rejection_sample`. The k=7 batch has garbage logits on its masked rows, and the reference is the scheduler-truncated k' = m batch.
  - The emitted tokens are equal in greedy, sampled, block-verification and probabilistic-draft modes. The trials cover both "all m accepted" and "rejected early".
  - The check is sensitive. Each of these module mutations makes it fail: masking row m, dropping the -1, masking one row late, masking row m on the input side only, and an off-by-one anchor.
- **Across steps.** The test runs `_kpool_decode_update_batched_kernel` over a history plus four verify steps at c=2, with index_kpool = 4, k = 7 and random m and a.
  - The tail slots come from `prepare()` and `compute_kpool_tail_slot_mapping` exactly as the patch writes it.
  - The whole tail ring equals truncated k' = m's, and so does every committed pool entry the steps touch: 0 of 83 differ. Letting masked rows stash, as the first version did, changes 35 of those 83.
  - The tail-slot tests fail if the mask is a no-op, shifted by one row, applied before v11 fills the slots, or never bound.

**Floating point.** Rows 0..m are not bit-equal to an unmasked k=7 step. Marlin splits its work over SMs by total block count, so fewer distinct experts changes the fp32 reduction grouping. The serve is already not run-to-run deterministic: in the E0 A/A, top-1 disagreed on 1.55% of positions and greedy diverged on 14 of 20 prompts. The GPU check therefore compares against that band.

### Expected gain

The step model:

- Routed-expert bytes grow with the distinct experts per layer, D(n) = 288(1-(1-8/288)^n), at 7.08 MB per expert per layer per rank over 42 layers.
- Masked rows add no experts, so a c=1 step costs D(m+1) instead of D(8).
- Two slopes bound the saving per distinct expert per layer: 0.75 ms (the SD-1 fit to the k=5/k=7 receipts) and 1.25 ms (bytes at 237 GB/s).
- The E0 baselines are A 115.3 ms at 19.79 tok/s, B 120.0 ms at 34.90 tok/s and H 161.1 ms at 13.69 tok/s. Acceptance is 1 + the sum of the verified positions.

| Cell | Drafts verified (m) | Acceptance | Step ms (0.75 / 1.25 slope) | tok/s (0.75 / 1.25 slope) |
|---|---|---|---|---|
| A prose c=1 | 7 (today) | 2.263 | 115.3 | 19.8 |
| A prose c=1 | 3 | 2.154 (−4.8%) | 94.7 / 80.9 | 22.9 (+16%) / 26.8 (+36%) |
| A prose c=1 | 2 | 1.985 (−12%) | 89.2 / 71.7 | 22.4 (+13%) / 27.9 (+41%) |
| H prose c=2 | 3 per request | −4.8% (A's curve) | 126.3 / 102.9 | 16.6 (+21%) / 20.4 (+49%) |
| B code c=1 | 7 | 4.137 | 120.0 | 34.9 |
| B code c=1 | 3 (fixed cap) | 2.989 (−28%) | 99.4 / 85.6 | 30.4 (−13%) / 35.4 (+1%) |

- **Fixed cap vs confidence rule.** A fixed cap trades code for prose. The confidence rule is meant to give both. If the drafter's running product tracked the E0 curves, tau = 0.1 would verify 3 prose drafts (4 live rows instead of 8) and all 7 code drafts. The unit test checks exactly this arithmetic. The SD-1 oracle line of `tools/census_report.py` bounds any cut rule.
- **Upper slope.** The 1.25 ms slope multiplies the independent-routing D(n). If real routing is correlated, the real D(8) - D(4) drop is smaller than 27.4. The E1 microbench's dense linears (41.1 ms per step at M=8, BF16, drafter included) plus D(8) expert bytes (58.1 × 42 × 7.08 MB at 237 GB/s ≈ 73 ms) already reach ~114 ms of the 115.3 ms step, before attention, indexer and KDA. Expect the lower end of the range until the census prefix curve and the microbench (step 6 below) settle it.
- **Overhead.** Per target forward, prefill included: 168 small in-graph kernels, 4 per MoE layer. Per step: about 18 eager kernels (prepare 6, tail-slot mask 1, mask_drafts 3, record 8). Roughly 0.5 ms (estimate), and code steps with m = k pay it for nothing. A precomputed source index and keep mask per step would cut the 168 to 84, and a fused Triton remap to 42, if nsys shows it matters.
- **Lower bound 0.** Masked rows still run attention, KDA, the shared expert and lm_head at full shape. If the k-dependence of step time is per-token work rather than expert reads (the review's alternative fit, step ≈ 79.5 + 4.75·n ms), the gain is 0.

### Risks

- **Calibration.** The selector softmax covers the top-K candidates only, so it overstates confidence. A tau that is too high also cuts positions that would have been accepted, and code and structured output lose acceptance. Sweep tau, and fall back to `MAX` alone. The walk is greedy, so its probabilities ignore the request temperature. On sampled traffic the rule over-verifies, which costs speed and never correctness.
- **Rank agreement.** Both ranks must mask the same rows. m comes from rank-identical draft scores, and `EXTRA_ENV` sets the switch on both ranks. A width mismatch is worse than a draft-token mismatch: the ranks would emit different `num_sampled` and diverge for good. Only the census check `rank0 vs rank1 routing identical` would show it. `TAU=0` with `MAX` alone cannot mismatch.
- **EPLB.** It is not refused. With redundant experts, `_apply_eplb_mapping` picks a replica by token index, so a masked row can land on a different replica than its anchor and add reads. This costs speed only, and the recipe does not use EP.
- **c=2 Marlin block spill.** At 8-16 rows Marlin uses 8-row blocks (`marlin_moe.py:333`). An anchor expert shared by both requests can pass 8 rows and be read twice. The expected count is about 8·8/288 ≈ 0.22 experts per layer, at most ~65 MB per step per rank.
- **Metrics.** Masked drafts count as drafted and rejected, so the acceptance rate drops by construction, and per-position acceptance past m is 0. Judge by acceptance_len and step_ms.
- **Census.** It records the ids after the remap: masked rows copy the anchor. That is the intended measurement.
- **No GPU run yet.** Everything above comes from the v11 source and the CPU tests. The plan below does not cover async scheduling, which the recipe keeps off.

### GPU validation (exclusive TP=2 slot, one switch per boot)

Follow AGENTS.md and keep receipts in `evidence/iter-adaptive-verify/`, with `trail.tsv` and `decision.tsv` rows. `L` means both ranks' engine logs.

1. **Build and gate.** Build v13 on the head and `docker save glm53-sm121-v13 | ssh spark2 docker load`. Boot with nothing set: `L | grep -c GLM53_ADAPTIVE_VERIFY` must print 0 on both ranks, and the P3.0 checks above must pass.
2. **Mechanism, fixed cap.** The cap separates the mechanism from calibration.
   ```bash
   IMAGE=glm53-sm121-v13 EXTRA_ENV="GLM53_ADAPTIVE_VERIFY=1 GLM53_ADAPTIVE_VERIFY_TAU=0 GLM53_ADAPTIVE_VERIFY_MAX=3" ./run.sh
   ```
   - Both ranks log `GLM53_ADAPTIVE_VERIFY: tau=0 max=3 (k=7); masked verify rows reuse their anchor's experts in 42 MoE layers`.
   - The boot must reach ready. Graph capture includes the remap.
3. **Overhead only (A/A).** `TAU=0` with `MAX=k` is refused, so boot with `EXTRA_ENV="GLM53_ADAPTIVE_VERIFY=1 GLM53_ADAPTIVE_VERIFY_TAU=1e-9"`.
   - The pinned drafter has `selector_top_k` = 16, so every selector probability is ≥ 1/16 and the running product is ≥ 16^-7 ≈ 3.7e-9. m = k on every step: every hook runs and nothing is masked.
   - Run ABAB against off on cells A and B. This separates the mechanism cost (~0.5 ms estimated) from the truncation gain.
   - Optionally take one nsys trace to confirm the remap kernels sit inside the graph segments.
4. **Lossless.** Three checks and one limit:
   - **Greedy identity, allowing for known nondeterminism.** `python3 quality/tier0.py compare --ref nvidia-v11-k7` must PASS. That covers count-200 exact, the thinking-off kwargs cells, and greedy.hazard within 2× the reference's A/A.
   - **Acceptance-weighted distribution.** Run `python3 bench_decode.py --cells A,B,G`.
     - Acceptance at the first three draft positions must stay near E0 (A .640 .345 .169, B .817 .657 .515). Step boundaries move with the cap, so a few percent of drift is expected; a collapse is not.
     - The last four draft positions must be exactly 0.
     - If masked rows leaked into live logits, acceptance at the first three positions would collapse, and the text would degrade.
   - **Census.** Run the same setting with `GLM53_EXPERT_CENSUS=/cache/huggingface/glm53-census/$RUN` added to `EXTRA_ENV`, then `tools/census_report.py` (see tools/README.md).
     - Census positions 4-7 (the draft rows past m = 3) show overlap 1.000, new 0.00 and accept 0.000.
     - `distinct/layer` at c=1 matches the off-census prefix curve at n=4, not at n=8.
     - `rank0 vs rank1 routing identical: True`.
   - **What these cannot see.** No GPU check here tests cross-step state against a clean reference. Count-200 and greedy.hazard stay under index_topk = 2048 tokens, where every pool is selected anyway. Ruler v2 against E0 shares the ring defect of the lossless section. The CPU kernel test is the gate for "identical to k' = m" across steps.
5. **Step time and acceptance.** One boot per setting, ABAB against off: `MAX=3` with `TAU=0`, then `TAU` ∈ {0.3, 0.2, 0.1, 0.05}.
   - Per boot, run the ruler v2 fast gate plus the c=2 cell (`python3 bench_decode.py --cells A,B,H`) and Tier 0 (`python3 quality/tier0.py compare --ref nvidia-v11-k7`).
   - Record acceptance_len, step_ms, tok/s and per-position acceptance for A, B and H.
   - Keep a setting only if A and H beat noise and B does not regress beyond noise.
   - step_ms should fall toward the table. Acceptance should fall by at most the share of the truncated positions.
6. **Optional microbench.** On one Spark, time the image's `fused_marlin_moe` on one nvidia-pack layer at M=8 rows. Draw `topk_ids` to touch D ∈ {8, 16, 24, 32, 48, 58} distinct experts, with 200 graph replays each. The slope in ms per distinct expert, times 42 layers, settles 0.75 against 1.25 ms and so the table's range.
