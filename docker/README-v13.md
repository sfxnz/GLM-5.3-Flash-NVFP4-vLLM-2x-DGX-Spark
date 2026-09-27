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

1. `prepare()` runs in the runner after `prepare_attn`, outside every graph. It zeroes `masked[:num_tokens_after_padding]`, scatters the predicate onto the input rows `logits_indices`, and stores each row's anchor row as `logits_indices - local_pos`.
2. `remap()` runs in `BaseRouter._select_experts`, after `_compute_routing` and before the census capture and EPLB. That code is inside the opaque `moe_forward` op, so the remap is captured in the FULL and PIECEWISE graphs. A masked row takes its anchor's top-8 ids with weight 0. This costs 3 fixed-shape kernels per MoE layer.
3. `mask_drafts()` runs in `RejectionSampler._verify`, after `apply_sampling_params`. A masked row's draft id becomes -1 in the tensor handed to `rejection_sample`. The sampling-parameter kernels still see the real ids.
4. `record()` runs in the runner after the draft. It sets `verify_len[idx_mapping] = m` for the drafts just proposed. The next step's 1-3 read it, in stream order.

The buffers are persistent: `masked` and `anchor` have `max_num_batched_tokens` rows, and `verify_len` has `max_num_seqs` entries. Nothing calls `.item()` or `.cpu()`, and no shape depends on data. `anchor[t] <= t` always holds, so the gather stays inside the batch at every graph size. Both TP ranks compute m from the draft scores that already give them identical draft tokens, so both ranks mask the same rows.

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
- **State after the step.** `num_sampled = a+1 ≤ m+1`, where a is the number of accepted drafts.
  - The MLA and indexer KV of masked rows sits at positions after x_a. The next query starts at pos(x_a)+1 and spans k+1 positions, so it rewrites those slots before any read. Every ordinary rejected draft already takes this path.
  - KDA and conv states: the next step starts from spec-state slot `num_accepted - 1 = a` (`fused_recurrent.py:114-121`, with `num_accepted = max(num_sampled, 1)` from `mamba_hybrid.py:342`). It never starts from a masked row's slot.
  - DFlash2: the context K/V precompute also covers masked rows (`dflash/speculator.py:416`), at positions after `last_valid_pos = pos(x_a)` (:522, :531). The next draft query block covers pos(x_a)+1 .. pos(x_a)+k+1 (:554) and writes those slots before its attention reads them. The context it reads ends at `last_valid_pos` (:593).
- **Conclusion.** The step's output distribution and the next step's state equal k' = m speculation from the same drafts. That is lossless: Leviathan et al. for sampling, and by construction for greedy.

`test_v13_verify.py` checks this on v11's own kernels under the Triton interpreter:

- It runs the real layout kernels, then `prepare()` and `mask_drafts()`, then `rejection_sample`. The k=7 batch has garbage logits on its masked rows, and the reference is the scheduler-truncated k' = m batch.
- The emitted tokens are equal in greedy, sampled, block-verification and probabilistic-draft modes. The trials cover both "all m accepted" and "rejected early".
- The check is sensitive. Each of these module mutations makes it fail: masking row m, dropping the -1, masking one row late, masking row m on the input side only, and an off-by-one anchor.

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
- **Overhead.** 126 small in-graph kernels and about 17 eager ones per step, roughly 0.5 ms (estimate). A fused Triton remap would cut the 126 to 42, if nsys shows it matters.
- **Lower bound 0.** Masked rows still run attention, KDA, the shared expert and lm_head at full shape. If the k-dependence of step time is per-token work rather than expert reads (the review's alternative fit, step ≈ 79.5 + 4.75·n ms), the gain is 0.

### Risks

- **Calibration.** The selector softmax covers the top-K candidates only, so it overstates confidence. A tau that is too high also cuts positions that would have been accepted, and code and structured output lose acceptance. Sweep tau, and fall back to `MAX` alone. The walk is greedy, so its probabilities ignore the request temperature. On sampled traffic the rule over-verifies, which costs speed and never correctness.
- **Rank agreement.** Both ranks must mask the same rows, or they would accept different tokens. m comes from rank-identical draft scores, and `EXTRA_ENV` sets the switch on both ranks. The census check `rank0 vs rank1 routing identical` covers it.
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
3. **Lossless.** Three checks:
   - **Greedy identity, allowing for known nondeterminism.** `python3 quality/tier0.py compare --ref nvidia-v11-k7` must PASS. That covers count-200 exact, the thinking-off kwargs cells, and greedy.hazard within 2× the reference's A/A.
   - **Acceptance-weighted distribution.** Run `python3 bench_decode.py --cells A,B,G`.
     - Acceptance at the first three draft positions must stay near E0 (A .640 .345 .169, B .817 .657 .515). Step boundaries move with the cap, so a few percent of drift is expected; a collapse is not.
     - The last four draft positions must be exactly 0.
     - If masked rows leaked into live logits, acceptance at the first three positions would collapse, and the text would degrade.
   - **Census.** Run the same setting with `GLM53_EXPERT_CENSUS=/cache/huggingface/glm53-census/$RUN` added to `EXTRA_ENV`, then `tools/census_report.py` (see tools/README.md).
     - Census positions 4-7 (the draft rows past m = 3) show overlap 1.000, new 0.00 and accept 0.000.
     - `distinct/layer` at c=1 matches the off-census prefix curve at n=4, not at n=8.
     - `rank0 vs rank1 routing identical: True`.
4. **Step time and acceptance.** One boot per setting, ABAB against off: `MAX=3` with `TAU=0`, then `TAU` ∈ {0.3, 0.2, 0.1, 0.05}.
   - Record cells A, B and H: acceptance_len, step_ms, tok/s and per-position acceptance.
   - Keep a setting only if A and H beat noise and B does not regress beyond noise.
   - step_ms should fall toward the table. Acceptance should fall by at most the share of the truncated positions.
5. **Optional microbench.** On one Spark, time the image's `fused_marlin_moe` on one nvidia-pack layer at M=8 rows. Draw `topk_ids` to touch D ∈ {8, 16, 24, 32, 48, 58} distinct experts, with 200 graph replays each. The slope in ms per distinct expert, times 42 layers, settles 0.75 against 1.25 ms and so the table's range.
