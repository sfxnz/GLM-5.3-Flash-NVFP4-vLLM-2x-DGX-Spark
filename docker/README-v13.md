# v13 image layer: opt-in runtime patches

`Dockerfile.sm121-v13` builds on `glm53-sm121-v11` and adds two Python-only patch layers:

- `patch_v13_misc.py` (this file documents it)
- `patch_v13_fp8.py` (from the fp8 lane)

Each behaviour is gated by a `GLM53_*` environment variable, and all of them are off by default. With no `GLM53_*` set, v13 serves exactly like v11. Nothing in the v11 hot path changes until a switch is set.

```bash
docker build -f docker/Dockerfile.sm121-v13 -t glm53-sm121-v13 docker   # needs docker/patch_v13_fp8.py
GLM53_V11_SRC=/path/to/v11src python3 docker/test_v13_misc.py                 # CPU tests
GLM53_V11_SRC=/path/to/v11src python3 -m unittest docker/test_v13_fp8.py -v   # CPU tests
```

Both test files read `GLM53_V11_SRC`: the directory that holds the `vllm/` package tree copied out of `glm53-sm121-v11` (for example a `dist-packages` copy). Without it the source-dependent tests skip. `test_v13_misc.py` still accepts the old `GLM53_V11SRC`, which points at `vllm/` itself, when `GLM53_V11_SRC` is unset.

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
    - The root fix belongs in its own lane: a ring of at least kpool - 1 + k + 1 slots addressed `pos % R`, or a re-stash after acceptance. `GLM53_KPOOL_TAIL_FIX` (below) is that fix.
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

## GLM53_KPOOL_TAIL_FIX: a tail ring sized for the verify window (`patch_v13_kpool_tail.py`)

Under speculative decoding, v11's DSA indexer builds committed pool keys from rejected drafts. The switch below fixes that. After every step, every committed pool and every tail-ring slot a later step can read equals what non-speculative decoding writes. The published serve (DFlash2-7) and the MTP-4 rollback are both affected once a context passes index_topk = 2048 tokens.

| Env var | Default | Meaning |
|---|---|---|
| `GLM53_KPOOL_TAIL_FIX=1` | off | Give each request's tail ring enough slots for the verify window, and write the prefill seed through the tail view's real strides. Off keeps every v11 address. |

```bash
GLM53_V11_SRC=/path/to/v11src python3 docker/test_v13_kpool_tail.py   # CPU; torch + triton add the kernel simulation (~2 min)
```

### The bug: rejected rows overwrite committed slots

- **The ring.** Each of the 11 indexer layers keeps a request's open pool in a tail block of index_kpool = 4 slots. A slot holds one token's raw indexer K and gate, addressed `pos % 4` (`indexer.py:565`; the `KpoolTailSpec` block size is kpool). The K-pool update kernel stashes every row. When the pool's last token arrives, the kernel compresses the pool from the three stashed slots plus the current row and writes the fp8 pool key (`kpool_compress.py:596-697`).
- **A verify step** runs 1 + k rows at positions P..P+k (k = 7 for DFlash2, 4 for MTP). It stashes all of them in position order, gated only on the tail slot, so row r overwrites the slot of position r − 4.
  - After rejection sampling, P..P+a is committed. The open pool at P' = P + a + 1 has up to three committed positions, ⌊P'/4⌋·4 .. P'−1.
  - Rows four positions later overwrote their slots. Those rows are rejected drafts, or the row at the bonus position.
  - Nothing rewrites those slots before the pool completes in a later step. That step compresses the rejected drafts' K and gate into a committed pool key.
- **Example.** P = 10, so pool [8..11] is open with 8 and 9 committed. The step runs rows 10..17 and accepts nothing (a = 0, P' = 11). Rows 12, 13 and 14 overwrote the slots of positions 8, 9 and 10. The next step's row 11 completes pool [8..11] from those three rejected rows.
- **Scope.**
  - Any k ≥ 2 is affected. k = 1 is safe: for it, the bound below asks for R ≥ kpool, which is v11's ring.
  - Only pools built during decode are affected. Prefill compresses pools straight from the batch (`_kpool_compress_insert`) and never reads the ring.
  - Below index_topk = 2048 tokens every pool is selected, so nothing changes. Above it the indexer ranks pools by these keys, so a corrupted pool can drop out of the 512 selected pools, or take another pool's place.
  - The E0 long-context cells (E at 32k and 128k) ran with the defect.
- **With adaptive verify.** Masked rows stash nothing, so adaptive verify overwrites fewer committed slots than baseline k = 7. Its live rejected rows still overwrite them (the "Shared with baseline k=7" bullet above).

### A second v11 defect in the same ring: the prefill seed writes the wrong bytes

- **What the seed does.** `kpool_seed_tail_cache` copies the last kpool tokens of each prefill chunk into the ring, so that decode can complete the prompt's boundary pool. Its kernel puts tail block b at element `(b·2·kpool + pos % kpool)·128` (`kpool_compress.py:481`), which is the address in a contiguous `[num_blocks, 2, kpool, 128]` tensor.
- **The real layout is not contiguous.**
  - The tail co-owns the indexer tensor: its page is padded to the indexer page (`kv_cache_utils.py:1461`).
  - The runner carves the tail view with the padded page as its block stride (`attn_utils.py:295-316`). That stride is idx_page = 2304/4 · 132 = 76,032 bytes.
  - The decode kernel addresses the ring through that stride (`TAIL_BLOCK_ELEMS = stride(0)`, `kpool_compress.py:783`). The seed kernel does not.
- **Where the seed lands.** The seed for tail block b goes to byte b·2048 of each of the 11 indexer tensors.
  - For b ≤ 36 that is the null block's page.
  - For larger b it lands in block ⌊b·2048/76032⌋'s page, over 2 KB of the compressed pool keys (or their scales) of whichever request owns that block.
- **What the boundary pool reads.** The ring keeps what the KV block zeroer left there, which is zeros. So when the prompt length is not a multiple of 4, the boundary pool compresses zeros for its prompt tokens.
- **Speculation.** This defect does not need speculative decoding.
- **Why the fix touches it.** A larger ring changes the seed's block layout. v11's addressing on a 16-slot block would scatter 8 KB per request instead of 2 KB, so the fix has to address the seed correctly.

### The fix

- **Ring size.** `KpoolTailSpec.block_size`, the ring, becomes R = the next power of two ≥ k + kpool − 1. That is 16 for DFlash2-7 and 8 for MTP-4 (and for k = 5). The size comes from `glm53_tail_ring_slots` in `kpool_compress.py`, which `Glm5NextTailCache.get_kv_cache_spec` calls. The tail metadata builder already computes slots as `own_block · block_size + pos % block_size`, so it is unchanged.
- **Kernels.**
  - The K-pool update kernel stashes at `pos % RING`, reads a completion's three slots at `pos % RING`, and takes the tail block as `tail_slot // RING`. RING is the tail view's slot count.
  - The seed kernel uses the same arithmetic and addresses the block through the view's two strides.
  - With the switch off, RING = kpool and the seed gets v11's contiguous strides, so every address is v11's. The CPU test compares the whole indexer tensor byte for byte.
- **Why R ≥ k + kpool − 1.**
  - Row r overwrites the slot of position r − R.
  - A later step reads only the committed positions of the open pool. The lowest of those is ⌊P'/4⌋·4 ≥ P − (kpool − 2), reached when a = 0 and P % 4 = 2. The last row is P + k. So no readable slot is overwritten when P − (kpool − 2) + R > P + k.
  - A completion inside the step reads the three positions just before it. For any R ≥ kpool those are still the latest writes to their slots.
  - Rejected rows stash at positions ≥ P'. The next step rewrites those positions in order before any completion reads them.
  - The CPU test shows the bound is exact: R = k + kpool − 1 is lossless, and one slot less is not.
- **Why a power of two.** The scheduler block size is the LCM of every KV group's block size, and that includes the tail group (`kv_cache_utils.py:639`). A power of two ≤ 128 divides every block size the kpool indexer accepts (a multiple of index_kpool · 32), so the LCM stays at 2304. `get_kv_cache_spec` asserts this at boot.
- **Memory.** Nothing is allocated. v11 already pads the tail page to 76,032 bytes. The 16-slot ring uses 8 KB of it (v11 uses 2 KB). KV accounting charges the tail one whole block per request either way, so the KV pool size does not change.
- **Edited files.** The patch edits `models/glm5next/nvidia/ops/kpool_compress.py` and `models/glm5next/nvidia/attention.py`, and nothing else. No other v13 patch edits them.

### Why this design

- **(c) Snapshot and restore the committed pre-step slots: not enough.** Committed rows of the current step are overwritten too. With P = 8 and a = 1, rows 8 and 9 are committed and rows 12 and 13 overwrite them. To restore them after the step you need somewhere to keep them, which is the extra storage (a) adds, plus a restore kernel per layer after acceptance.
- **(b) Stash the accepted rows after rejection sampling: more machinery.** The rows' K and gate only exist during the forward, one set per layer. So (b) needs a buffer per layer and a hook after acceptance in both runners (V2 for DFlash2, V1 for MTP). The in-step completions would also have to read the batch instead of the ring.
- **(a) An enlarged ring: neither.** The ring stays addressed by position, as in v11. Nothing is keyed by request or by step, and nothing runs after acceptance.
  - It holds for any acceptance pattern, by the bound above.
  - It works the same in both runners.
  - It covers prefill chunks: the seed writes the last kpool tokens, and a chunk of ≤ 1 + k tokens runs through the decode kernel.
  - It covers the MTP layer's own indexer. Its first draft pass writes the verify rows' positions, which is the same pattern. The later single-token passes, when they run the indexer at all (`skip_topk` skips it), write at most k − 1 positions past the bonus token, which stays inside the same bound.
  - No kernel is added, no shape changes and the host never syncs. The indexer op runs on the eager path of the breakable graph.
- **Adaptive verify.** The two switches compose. Masked rows still get tail slot −1 and stash nothing. With both on, k' = m and baseline k = 7 both leave exact state. The CPU test applies every v13 patch in Dockerfile order and simulates the combination.

### Proof: v11's own kernels under the Triton CPU interpreter

`test_v13_kpool_tail.py` builds the simulation from v11's source:

- the K-pool update, pool-compress and seed kernels, and the prefill insert helper,
- the tail slot mapping, plain and with adaptive verify's hook,
- a shared indexer tensor with the tail view carved exactly as `_reshape_attention_kv_cache` carves it. The padded page holds two requests' pool pages and tail blocks in one buffer.

Each trial runs a prompt, then five verify steps at c = 1 or c = 2 with random acceptance. The prompt takes the prefill path, or the decode path when it has ≤ 1 + k tokens. The reference is v11 non-speculative decoding on a contiguous tail, where v11's seed is right. Stale ring contents are random, so reading a slot nobody wrote shows up as a differing pool.

| Scenario (10 trials × 5 steps) | fix: pools differ | fix: readable ring slots differ | v11, contiguous tail (overwrite only) | v11 non-speculative, real layout (seed only) |
|---|---|---|---|---|
| DFlash2 k=7, c=1 | 0/58 | 0/16 | 18/58 | 4/58 |
| DFlash2 k=7, c=2 | 0/105 | 0/30 | 30/105 | 9/105 |
| MTP k=4, c=2 | 0/100 | 0/34 | 25/100 | 15/100 |
| k=7, c=2 + adaptive verify | 0/91 | 0/34 | 22/91 | 10/91 |

- The fix writes no byte outside the requests' own blocks. v11's seed writes 14-35 KB per scenario into blocks no request owns.
- With the switch off, the patched source leaves the whole shared tensor byte-identical to v11's.
- The test fails under each of these mutations of the fix (k = 7, c = 2, 100 committed pools):
  - the stash still at `pos % kpool`: 57 pools differ;
  - completions reading `pos % kpool`: 48 pools differ;
  - the decode block id taken by kpool: 8 pools and 28 ring slots differ, and 159 KB land in foreign blocks;
  - the seed with v11's contiguous strides: 8 pools differ;
  - the seed block id taken by kpool: 8 pools differ, and 71 KB land in foreign blocks;
  - a ring one slot below the bound: 8 pools differ.
- It also checks:
  - every anchor occurs once in v11, a rerun is a no-op, and drift refuses;
  - `py_compile` and the Dockerfile chain (misc, fp8, census, verify, this patch) apply and rerun clean;
  - the ring-size helper;
  - `get_kv_cache_spec` off gives v11's arguments; on gives 16, 8 or 4 slots, the log line and the block-size assert.

### Cost

- Memory, allocations and launches: none added. The update kernel does the same loads and stores, taking `% 16` of a constexpr where v11 takes `% 4`. The seed kernel does the same stores, at the right address.
- Step time: no change expected. The fast gate below checks it.

### Risks

- **Output changes above 2048 tokens.** Pool keys now match non-speculative decoding, so long-context greedy text can differ from v11, which read corrupted keys. Judge the change by the quality gates, not by byte equality.
- **Block size.** R must divide `--block-size`, and the spec asserts it. 2304 passes, and so does any multiple of 128 for every k up to 125.
- **Both ranks.** The switch changes the tail KV spec, so set it on both ranks, which `EXTRA_ENV` does. Otherwise the ranks disagree on the tail spec.
- **PD connectors** transfer the tail block by its unpadded page, now 8 KB instead of 2 KB. The recipe does not use PD.
- **No GPU run yet.** Everything above comes from the v11 source and the CPU test.

### GPU validation (exclusive TP=2 slot, one switch per boot)

Follow AGENTS.md. Keep receipts in `evidence/iter-kpool-tail/` with `trail.tsv` and `decision.tsv` rows. `L` means both ranks' engine logs.

1. **Build and gate.** Build v13 on the head and copy it with `docker save glm53-sm121-v13 | ssh spark2 docker load`. Boot with nothing set. `L | grep -c GLM53_KPOOL_TAIL_FIX` must print 0 on both ranks, and the P3.0 checks must pass.
2. **Boot with the switch.** `IMAGE=glm53-sm121-v13 EXTRA_ENV="GLM53_KPOOL_TAIL_FIX=1" ./run.sh`
   - Both ranks log `GLM53_KPOOL_TAIL_FIX: kpool tail ring 16 slots (index_kpool=4, k=7)`.
   - The KV pool size (372,877 tokens on the E0 pin) and `free -h` MemAvailable at ready match the off boot.
3. **Short context, where no pool is ranked.** This must equal off within the A/A band. Check count-200 lossless and the thinking-off smoke, then run `python3 quality/tier0.py compare --ref nvidia-v11-k7`, which must PASS.
4. **Long context.**
   - **Needles.** Run `python3 quality/tier0.py compare --ref nvidia-v11-k7 --long`. It covers 8k, 32k and 128k, and each length must find 2 of 3 depths.
   - **Decode against prefill, on decode-built pools.** This probe targets the fixed state directly.
     - Use a unique-salt filler of 32k and of 128k tokens. Generate 2,000 greedy tokens with thinking off, `logprobs: true` and `return_token_ids: true`. The V2 rejection sampler returns the target's raw logprobs.
     - Send the prompt ids plus the generated ids to `/v1/completions` with `prompt_logprobs: 0` and `max_tokens: 1`. The prefill path never reads the ring, so its pools are right in both builds.
     - Per generated token, Δ = decode logprob − prefill logprob. Report the mean and p99 of |Δ| after the first 64 tokens.
     - Run the same probe at a 1,500-token context, where no pool is ranked, for the numerical floor. Run it with the switch off on the same day.
     - Expected: with the switch on, |Δ| at 32k and 128k is at the floor. With it off, it may sit above.
   - **Long-context speed.** Run `python3 bench_decode.py --cells E` and compare acceptance and step_ms with E0 (32k: 2.327 at 111.5 ms; 128k: 2.822 at 112.7 ms).
5. **Speed.** Run the ruler v2 fast gate, `python3 bench_decode.py --cells A,B`, ABAB against off. step_ms must stay within noise (E0: A 115.3 ms, B 120.0 ms).
6. **Keep or revert.** Keep the switch only if 3 and 5 pass and 4 is no worse than off. If it is kept, record a new Tier 0 reference with it on (`python3 quality/tier0.py record --name nvidia-v13-kpooltail`), so later lanes compare against the fixed state.
7. **With adaptive verify.** Boot both switches on the setting its sweep chose and repeat 3 to 5.
8. **Optional, MTP rollback.** Boot `MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 SNAPSHOT_REV=caca4e6a4ebbd66f159d3d2fc256683fd6e27177 SPEC=mtp` with the switch. It must log `8 slots (index_kpool=4, k=4)`, and count-200 must stay lossless.
