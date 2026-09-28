# Plan: faster, higher-quality decode for GLM-5.3-Flash-NVFP4 on 2× DGX Spark (official nvidia pack, vision on)

This work goes on branch `opt/nvidia-perf-quality`, which starts from PR #11 head 29954c8. It is a plan only: no code has been written and nothing has run.

**Labels used throughout:**
- **[C]** confirmed by the review's verifiers.
- **[P]** plausible, not confirmed.
- **[M]** measured.
- **[Mod]** modeled at 205-232 GB/s.
- **[R]** a read-only check I made while writing this plan, not independently verified.

All step times are per rank at TP=2 and come from the LibertAI caca4e6 pack. The nvidia 09b04e5 pack has no decode measurement at all.

## 1. Executive summary

### (a) The nvidia pack has never booted

**What happened.** On 2026-09-16 there were six exclusive TP=2 boots. Each one loaded 90.36 GiB per rank and got through the TileLang JIT. Then free memory (MemAvailable) fell at 2.4-3.8 GiB/s on both ranks during the first profiling forward pass, before the KV cache was allocated. The 8 GiB watcher killed every run. The reported 0.45-10.4 GiB values are the watcher's kill-time readings, not the true lows [R].

**What the variants tell us.** The variants were stacked rather than tested one at a time: skip-mm included video:0, and the eager and batch1024 runs were skip-mm plus one more change. So they only show that no single one of those changes prevents the collapse. Eager mode is irrelevant here: profiling runs before CUDA-graph capture, and the compile mode resolves to NONE [R].

**Leading cause.** The code path is confirmed [C]; that it causes the crash is still a hypothesis [P].
- On the nvidia pack, the dense MLP in layers 0-2 is NVFP4. On LibertAI it is BF16. `--moe-backend marlin` does not cover these layers.
- On sm_121, automatic kernel selection picks FlashInferCutlass (`kernels/linear/__init__.py:500-512`) [R].
- The first call to that kernel JIT-compiles the SM120 FP4 GEMM: 17 translation units, 22 parallel ninja jobs, no pre-built JIT cache and no cache mount.
- This is the same class of failure that caused spark2's global OOM on 08-31.
- The collapse starts 2-7 s after `mhc_post` compiles, which is exactly when layer 0's MLP runs [R].

**A watcher artifact is the least likely explanation.** LibertAI's 5.0-5.7 GiB low comes after KV allocation. Its memory during the profiling phase was never sampled, so there is no healthy baseline that the nvidia runs fell short of.

**Fix:** pass `--linear-backend marlin` through `EXTRA_ARGS`, which reaches both ranks. Marlin is ahead-of-time compiled W4A16, needs no JIT, and costs about nothing in decode. The line `Using … for NVFP4 GEMM` is logged at model construction, 12-14 minutes before the danger phase, so a bad boot can be aborted early. Caveat: dense Marlin NVFP4 has never run in this lab [R].

### (b) Where decode time goes

At k=7, c=1 the step is **114.5-124.6 ms** [M]; at k=5 it is 105-108 ms [M].

| Bucket | Bytes per step | ms | Basis |
|---|---|---|---|
| BF16 non-MoE weights (KDA 4.7, MLA+indexer 1.55, shared experts 1.06, lm_head 0.63, rest 0.37 GB) | 8.33 GB | 36-41 | bytes [C] |
| DFlash2 drafter + second lm_head pass | 1.85-1.9 GB | 8-9 | [C] |
| KDA speculative-state writes | 0.63 GB | 2-4 | [P] |
| Non-byte overhead (~100 cross-node all-reduces, ~1,600 graph nodes, host work) | – | 10-25 | [Mod], unmeasured |
| Routed MoE (the remainder) | 7.4-14.6 GB | 36-63 (30-55%) | [Mod]; implies 25-49 distinct experts per layer vs 58.1 if routing were independent |

Only two things here are measured: the total step time, and the marginal cost of one extra verify token (3.2-7.0 ms) [M]. There is no nsys trace yet. For structured output, step time varies 3.0% between boots but only 0.2-2.4% within a boot [R].

### (c) Ranked levers

The baseline step time S0 is about 117 ms.

| # | Lever | Prose c=1 | Prose c=2 | Structured | Prefill / TTFT | Status |
|---|---|---|---|---|---|---|
| R0 | nvidia pack + Marlin linear kernel | ≈+1% | ≈+1% | ≈+1% | load +~35 s | code path [C], cause [P] |
| R1 | Drafter 7d74cdd → bf582e4 / dc77ff1 | 0 to +30% | same | ≈0 | 0 | stale pin [C] |
| R2 | k=5 (then 4/3) | +3-14% [M, mixed evidence] | +5-30% [M] | **−18% [M]** | 0 | [P] |
| R3 | FP8 W8A16 weights: drafter, shared experts, MLA (saves 1.75 GB) | +4-7% [Mod] | similar | similar | small | [P] |
| R3d | Also KDA o_proj/in_proj (saves 2.36 GB) | +6-9% more | | | | [P], high quality risk |
| R3n | Alternative: NVFP4-W4A16 attention | +14-22% [Mod]; Tony's TP4 run measured ×1.14-1.30 | | | | [R], needs a new checkpoint |
| R4 | Overhead trims chosen from nsys | +3-9% | | same | | [P] |
| R5 | MoE-aware tail masking | 0-20% | | 0 | | [P] |
| R6 | MoE kernel swap | 0-5% | | | W4A4 helps prefill only | [P]; step times equal (107 vs 105 ms) |
| – | Warmup + compile caches | 0 | | | removes ~6 s stall; boot −60-150 s | [C]/[R] |

### (d) What "substantial" realistically means

- **Baseline.** The real baseline is **≈19.2 tok/s** (acceptance 2.315 at 120.8 ms/step) [C], not the published 21.2. About 60-75% of the drop from 28.3 to 21.2 comes from the benchmark switching to thinking-off, not from the engine [C].
- **Measure first.** Today's benchmark (the "ruler") can only detect changes of about 31% [C]. Ruler v2 detects about 5% per change at 3 boots per arm [R], so changes worth 2-3% have to be tested in bundles.
- **Realistic gain on prose c=1** [Mod]:
  - The stack is k=5 (×1.03-1.14), FP8 drafter/shared/MLA (×1.04-1.07) and overhead trims (×1.03-1.09).
  - Together that is **+10-33%, midpoint about +20%**.
  - Adding KDA FP8 gives +17-45%. A better drafter would multiply all of this by 1.0-1.3.
- **Structured output:** +7-17%, or +13-27% with KDA FP8.
- **Target:** +20% on prose c=1, measured on ruler v2, with no Tier-0 or Tier-1 quality regression. Reaching +35% or more needs KDA quantization or a drafter win, and neither has been demonstrated.
- **Quality: the pack switch is the largest candidate quality lever, but its effect may be zero.**
  - On LibertAI, 69% of experts have different gate and up scales. vLLM keeps only the gate scale (modelopt.py:1530-1538; this is logged in the published runs). So up_proj is mis-scaled on those experts: median 1.067×, p99 1.455× [R; mechanism C].
  - The nvidia pack has no mismatches, and its dense-layer scales are tied [R].
  - However, on Marlin the nvidia pack's calibrated activation scales go unused, and 453M more dense parameters are quantized. Until a paired Tier-0 NLL comparison exists, treat #11 as a change of provenance, not a proven quality gain [C].

## 2. Serving the nvidia pack with vision on

**Status:** PR #11 pins 09b04e5, sets `LANGUAGE_MODEL_ONLY=0` and caps inputs at `{"image":4,"video":1}`. It has never booted. Vision was already implicitly on in every LibertAI run; #11 only adds the cap [C].

### Corrections (#11-A)

Each correction goes into recipe.yaml, then through `kit/render.py`, one concern per commit.

1. **Add `LINEAR_BACKEND=marlin`**, emitted as `--linear-backend`. Refuse any other value unless `FORCE_UNSAFE_LINEAR=1`. It affects only the layer 0-2 MLP [C], and it also guards future MXFP8 stages [R].
2. **SPEC=mtp is broken on the nvidia pack** [C]: layer 45 is 13.84 GiB of BF16. Refuse it unless the layer-45 dtype read from the safetensors header is U8. Fix AGENTS.md:3, README:64/143 and recipe.yaml:81 to say "MTP rollback = LibertAI + SPEC=mtp".
3. **run.sh fixes** [C]:
   - Set `SERVED_NAME=${SERVED_NAME:-$MODEL}`.
   - Use one FORWARD_ENVS list: SNAPSHOT*, LIMIT_MM_PER_PROMPT, HF_CACHE, HF_HUB_DISABLE_XET, LINEAR_BACKEND.
   - Add `EXTRA_ENV` with an allowlist: `^(NCCL|VLLM|PYTORCH|TORCH|CUDA|OMP|FLASHINFER|TRITON|TILELANG)_[A-Z0-9_]+$` or `MAX_JOBS`. Deny `*TOKEN*|*KEY*|*SECRET*`, and quote values with `printf %q`.
   - Call stop_local before drop-caches, and remove `sleep 25`.
   - Add an optional compile-cache mount keyed by image ID under `~/projects/data/`, off by default.
4. **Add `--mm-processor-cache-gb 1`** [P]. Require `LANGUAGE_MODEL_ONLY` to be exactly 0 or 1.
5. **Docs:**
   - Correct "~400k / 1.22×" to 372,877 / 1.14×.
   - Say vision was already on.
   - Drop the unevidenced "max-size dummy OOMs" claim, but keep the rule.
   - Note that the table is greedy while the served default is T=1.0 / top_p 0.95.
   - Note that async scheduling is automatically on for DFlash [C].
   - Explain moe-backend vs linear-backend.
6. **Vision smoke test.** `smoke_vision.py` sends a 1×1 JPEG and passes on any reply [C]. Replace it with the image+video suite in §6.
7. **Image parity** [C]. The image IDs differ between nodes. Build once, then `docker save` / `docker load` to the other node, record the digest and refuse on a mismatch. VALIDATE_ONLY should check both nodes.

### Diagnostic boot

This runs in slot 1 on the v11 image, after one LibertAI calibration boot under the same watcher.

**Watcher** (runs locally on both nodes):
- **Sampling and kill.** Read `/proc/meminfo` every 0.5 s. To kill, run `docker kill -s KILL` on both ranks in parallel over a pre-opened ssh ControlMaster, snapshot the logs, then remove the containers. The old serial path took 11-20 s and lost the logs [R].
- **Memory floors by phase:**

  | Phase | Kill below |
  |---|---|
  | Weight loading | 10 GiB |
  | Profiling (P1) | 10 GiB, capped at LibertAI's P1 low − 1 GiB |
  | KV allocation → ready | 3 GiB |
  | Serving | 2 GiB, plus a swap-rate rule |

  The P1 floor is F = rate × (sample interval + kill latency) + reserve = 5 GiB/s × 1.5 s + 2.5 GiB = 10 GiB.
- **P1 slope rule:** kill if at least 0.75 GiB is lost in each of two consecutive samples while below 14 GiB. The normal encoder dip bottoms out at 14.53 GiB.
- **Kill latency:** prove it is 1 s or less in a dry run first.
- **Rules dropped:** "below 1.5 GiB for 5 s" and PSI avg10. Both would fire only after memory is already exhausted.
- **Compiler census** (every 1 s, container processes by working directory and command line):
  - Kill on any `cached_ops/fp4_gemm*|cutlass` ninja build.
  - Log RSS and `memory.current` to separate compiler RAM from unified-memory growth.
  - b12x compiles in-process, so only the memory rules can catch it.
- **Kernel log:** follow `journalctl -k -f` for NVRM, Xid and OOM lines.

**Launch:**
```
MODEL=nvidia/… SNAPSHOT_REV=09b04e5… \
EXTRA_ARGS='--linear-backend marlin' \
EXTRA_ENV='MAX_JOBS=2 FLASHINFER_JIT_VERBOSE=1'
```
Keep both environment variables on every Phase-1 boot of both packs. Without VERBOSE, FlashInfer's nvcc builds log nothing [R].

**Gates:**
1. About 1 minute in, both ranks log `Using MarlinNvFp4LinearKernel for NVFP4 GEMM`. Abort if `Filesystem type` appears first.
2. No `fp4_gemm*` build appears.
3. The P1 memory low is within about 1 GiB of LibertAI's.
4. `/v1/models` answers, `Encoder cache … 32242` is logged, and there are no scale-mismatch lines.
5. Count-200 (200 consecutive integers) is lossless.

**Proving the cause (decision D12, revised).** Don't run a live boot in auto-kernel mode: TileLang would trip the compiler census first, and b12x would run unguarded. Instead, on one node with no model loaded, build `get_gemm_sm120_module_cutlass_fp4()`, once with MAX_JOBS unset and once with MAX_JOBS=2. If the peak memory drop is above about 18 GiB (what is left after weights load), that proves the cause without risking the pair. The built cache can then be mounted.

### Fallback ladder

One rung per boot:
1. **Wrong kernel selected:** fix the plumbing, or set `VLLM_DISABLED_KERNELS=FlashInferCutlassNvFp4LinearKernel,FlashInferB12xNvFp4LinearKernel,CutlassNvFp4LinearKernel`.
2. **Another module compiles:** pre-build it in isolation and mount it.
3. **No compile, but profiling still collapses (so not JIT).** Try these one at a time:
   - The NCCL buffer set `NCCL_BUFFSIZE=1048576 NCCL_LL128_BUFFSIZE=262144 NCCL_PROTO=^LL128 NCCL_MAX_NCHANNELS=8`. On DSv4.1 this cut pinned memory from 4.7 to 0.14 GiB.
   - Indexer workspace factor 1 (saves 1.57 GiB [R]).
   - A 256 MB logits cap.
   - DEBUG memory summaries.
   - `video:0`, as a diagnostic only.
4. **Marlin dense kernel errors or fails count-200:** use `--linear-backend emulation`.
5. **Otherwise:** keep LibertAI as the default and document its mis-scale.

Dropped from the ladder:
- SKIP_MM_PROFILING: it already failed at 6.46/9.77 GiB, and it breaks an AGENTS.md rule.
- Page-cache fadvise: page cache already counts as available memory.
- The "watcher artifact" rung.

### Acceptance before nvidia becomes the published default (#11-B)

- **Stability:** all gates pass on 4 boots. A 30-minute soak with images shows no Xid errors. MemAvailable once ready is at least LibertAI's minus 0.5 GiB.
- **Correctness:**
  - count-200 is lossless;
  - no chain-of-thought appears in content with thinking off;
  - the 12-cell kwarg matrix passes;
  - the image+video suite passes;
  - there are 0 U+FFFD characters;
  - tool-call JSON parses at least 98% of the time.
- **Speed:** boots interleaved L N N L L N N L (LibertAI / nvidia, 4 per arm, same flags). The upper 95% bound on step_ms must be at most +5%, and the paired acceptance lower bound at least −3%. A +2% margin would need about 16 boots per arm [R].
- **Quality:** Tier 0 and the pooled Tier 1 (§6) pass.
- **Docs:** the README decode numbers are re-measured on the nvidia pack.

## 3. Open PR dispositions

| PR | Verdict | Required fixes | Order |
|---|---|---|---|
| #11 | Split. **A** = §2 items 1-7 plus the #12 lint. **B** = the MODEL default flip. | A: CI passes, including the VALIDATE_ONLY refuse cases. B: §2 acceptance, with results in `evidence/iter-nvidia-pin/`. | 1 (A now, B after slot 1) |
| #8 kit | Rebase onto #11-A | Fold in c0c4af3 and e898a80. Fix the checks that hard-code LibertAI and the `{env:{}}` case. Enable the vision probe. Add per-run acceptance and ms/step. | 2 |
| #12 UMA guards | Rework into the §2 watcher | Its 16 GiB floor kills healthy boots [C]. Needs: phase floors, the slope rule, a ~107 GiB start floor, abort after 3 SSH failures, GPU-use detection, a trap if the head exits, replay tests. | 3 |
| #9 hillclimb | Evidence only | Drop a76474e (the k=5 change). Relabel H1 "inconclusive; structured −17.6% / −31%". Mark H2 as LibertAI-only. | 4 |
| b12x worktree | Evidence + refuse-guard PR | Fix the caca4e6 wording. Park the kernel work (K5). | 5 |
| #3 ABLIT | **Close** | Reopen only as an opt-in overlay: pinned sha256, access guard, Tier 0 and Tier 1. | – |

## 4. Findings

### Performance bottlenecks, ranked

1. **The nvidia pack cannot boot.** Code path confirmed, cause plausible [C/P].
2. **Routed MoE is 30-55% of the step** [Mod]. Independent routing would need about 27 GB/step, so expert choices must be correlated. No expert census exists yet.
3. **BF16 non-MoE weights are 8.33 GB per step** [C].
   - KDA and MLA layers strip their quantization config [C].
   - Offline FP8 `.o_proj.` weights that use `weight_scale` are never loaded (`model.py:1228-1240`) [R]. So community FP8 packs are unsafe on this image, not just useless.
4. **Verifying at k=7 costs 6-15 ms per step** [M]. Per-position acceptance is about [.68, .40, .17, .10, .05, .01, 0] [P].
5. **The drafter pin is stale.** The community speed gap comes from acceptance, not step time [C].
6. **Overhead is unmeasured.** Identical-prompt c=2 adds 16-35 ms per step with no extra expert reads [M].
7. **Minor:** drafter plus second lm_head pass 8-9 ms [C]; KDA state 2-4 ms [P]; `.contiguous()` copies 0.4-0.9 ms [C].

### Quality defects and risks, ranked

1. **LibertAI up_proj mis-scale** in the current published default [R/C].
2. **`thinking:true` loses the answer** (QUAL-2 [C]). A request with `{thinking:true}` returns empty `content`. The parser treats either kwarg as "on", but the template reads only `enable_thinking`.
3. **Thinking-off renders `Effort: Max` with an empty think block** [P].
4. **The vision smoke test cannot fail**, and there is no video test [C].
5. **No max-tokens cap.** The worst case is about 4.7 hours for one request [C].
6. **Router logits are rounded to BF16**, which can change the top-8 expert choice [C].
7. **FP8 KV cache rests on a single needle test** [P].
8. **Chain-of-thought leaks at 8-16k context** (1 in 5 runs) [P]. The needle test's random salt fails on refusals [C].
9. **Smaller issues:**
   - invalid UTF-8 tokens (#54150) [P];
   - ABLIT has no evaluations [C];
   - unknown `effort` values silently become Max [C].

### Correctness and ops issues

All confirmed [C] unless marked.

- **Config:** the §2 items 2, 3 and 7 problems.
- **Watcher:** #12's floors are wrong. The current watcher sits above the healthy low and samples only every 2-4 s.
- **Lint:** it reverts k edits because it hard-codes the capture sizes.
- **Benchmark gaps:**
  - c=2 uses identical prompts;
  - there is no c=4 cell;
  - TTFT counts content deltas only;
  - there is a ~6 s stall on the first new shape;
  - prefix-cache hits may be zero with DFlash [P].
- **spark1:**
  - Its head rank loads in 629-785 s against 190-243 s on spark2.
  - It runs with swappiness 60 and 3.5 GiB of swap in use [R].
- **Ledger errors:** 98 tokens, not 200; structured +21%, not +34%; mislabelled h4 rows.

## 5. Phased execution plan

### Phase 0: GPU-free work, now

| # | Work | Needs Spark resources? |
|---|---|---|
| 0.1 | #11-A commits; lint capture-ladder fix; context guard as a function of k | No |
| 0.2 | Watcher and #12 rework, with replay tests. The stock drop 13.82 → 10.33 → 0.59 GiB (00:46:51-:58) must trigger a kill; a healthy LibertAI curve (5.0 GiB + 6 GiB swap) must not. | No |
| 0.3 | Benchmark v2 with SSE-replay tests; `kit/spec_model.py` | No |
| 0.4 | Tier 0 (port from DSv4.1, BOS `[gMASK]<sop>`), pooled Tier-1 scorer, image+video suite, fixed-salt needle, probes, kwarg-matrix CPU test | No |
| 0.5 | Fix QUAL-2 with an alias in `chat_template.jinja`, plus a CI check against the Hub template that allows only the documented differences | No |
| 0.6 | Add `ruler_version` to the ledger; mark old verdicts with \|Δ\| under 30% as inconclusive | No |
| 0.7 | v12 image layer (Python-only), **every patch off by default**: router fp32, indexer factor, mHC warmup, prefix-cache fix, layer-45 filter | Build needs owner OK |
| 0.8 | Kernel code, not built: K1, K4 hook, census hook, microbenchmarks, nsys wrapper | No |
| 0.9 | Inspect the v11 kernel library for sm_120 Marlin FP8/FP4 kernels; check whether nsys/ncu are in the image | `docker create`, owner OK |
| 0.10 | Download drafters bf582e4 and dc77ff1 after a license check | Disk, owner OK |

**Phase 0 gate:**
- `render --check`, the lint and the tests pass.
- The VALIDATE_ONLY refusals work.
- Head and worker receive the same arguments and environment.

### Every slot: before and after

**Before:**
- Save a host snapshot to `evidence/<run>/host-before-<node>.txt`.
- Confirm DeepSeek is stopped, MemAvailable is at least 115 GiB, and `sudo -n` works.
- Set `vm.swappiness=0` at runtime, keeping swap on. Treat this as a fixed precondition (H0), not something to A/B test.
- Cycle swap, quiet spark1, and check that the images match.

**After:**
- No containers left, and MemAvailable back to at least 115 GiB.
- Swappiness (60) and read_ahead restored, verified, and logged in trail.tsv.
- The DeepSeek owner confirms it is healthy.

### Phase 1: decide the pack

About 15-16 hours, including 5.5 hours of unattended Tier 1. Can be split across two slots.

| Step | Hours |
|---|---|
| Pre-slot checks + kill-latency dry run | 0.5 |
| Isolated FP4 compile measurement | 0.4 |
| Microbenchmarks with CUDA events: dense Marlin NVFP4 on layer 0-2 shapes vs emulation; Marlin FP8 vs cuBLAS BF16; Marlin MoE at M=8/16; STREAM bandwidth. ncu needs SYS_ADMIN because `RmProfilingAdminOnly: 1` [R] | 1.0 |
| LibertAI: 4 interleaved repeat boots, full panel, forced-vs-natural length check, Tier 0 ×2, per-phase memory curve | 3.1 |
| nvidia diagnostic boot (+0.4 h per ladder rung, 2 rungs budgeted) | 0.5-1.3 |
| 3 more nvidia boots; Tier 0; Tier 1 on both packs | 7.6 |
| nsys boot: CUDA trace of 30 verify steps at c=1, structured and distinct-prompt c=2. Buckets must sum to within 5% of step time | 1.0 |
| Expert census boot: routed experts, distinct experts at n=4/6/8 | 0.75 |

**Exit:** a pack is chosen, and **all later phases run on it**. Acceptance doubles as a quantization-fidelity signal, so results do not transfer between packs.

### Phase 2 onward: one change at a time

Each change records `evidence/iter-<name>/` (benchmark, memory sampler, both engine logs, Tier 0) plus rows in trail.tsv and decision.tsv. A change that should not affect output costs about 3.0 hours (2.1 with a shared control arm). A change that affects acceptance costs about 2.2 hours.

| # | Change | Mechanism | Expected | Risk | Gate | Rollback |
|---|---|---|---|---|---|---|
| P2.0 | Compile-cache mount | No recompiling at boot | boot −60-150 s | Stale cache | Byte-identical output | Unmount |
| P2.1 | Drafter bf582e4, then dc77ff1 | Higher acceptance | 0-30% | License | Paired acceptance CI above 0; count-200 | 7d74cdd |
| P2.2 | k = 5, 4, 3 | Fewer verify tokens | +3-14% prose | Structured −18% | Step and paired acceptance; full graph capture logged; D4 weights | k=7 |
| P2.3 | effort low + clear_thinking | No "Max" with empty thinking | quality ± | +5-14 s per reply | Corruption probe, Tier-1-lite | Revert |
| P3.0 | v12 image, all patches off | – | 0 | Build drift | Byte-identical to v11 | v11 |
| P3.1 | Drafter FP8 (`VLLM_TEST_FORCE_FP8_MARLIN=1`) | −0.58 GB/step | ~2% | Acceptance | Step saving within ±30% of microbenchmark; acceptance | Off |
| P3.2 | Shared experts, then MLA, FP8 | −1.17 GB/step | ~5% as a bundle | Long context | Tier 0 per stage + needles to 318k | BF16 |
| P3.3 | KDA o_proj, then in_proj, FP8 (D6) | −2.36 GB/step | +6-9% | High | Tier 0 + needle at 128K or more + Tier 1 | BF16 |
| P3.4 | Router fp32 | Fidelity | ~0 | Low | Tier 0 | Env off |
| P3.5 | NCCL buffer set; indexer factor 1 | Memory headroom | +1.6 GiB | 318k needle | Memory once ready | Unset |
| P3.6 | Overhead trims chosen from nsys | Fewer graph nodes and all-reduces | +3-9% | Numerics | Keep rule; bit-exact | Revert |
| P3.7 | Tail masking (SD-1), only if the census shows ≥2 ms per tail token | Rejected tail tokens read no new experts | 0-20% | Hook bugs | Byte-identical | Off |
| P3.8 | MoE backend swap, only if Marlin reaches under 80% of STREAM bandwidth | Kernel efficiency | 0-5% | Stability (#3383); duplicate weights | Cosine > 0.9999; soak | Marlin |
| P3.9 | lm_head FP8 (last) | −0.63 GB/step | 2-3% | Target logits | Tier 0; acceptance | BF16 |
| P4 | read_ahead; KV 5/6 GiB; nccl-tests (needs authorization) | Variance | 0-3% | Host-wide | vmstat / PSI | Restore |

Two TTFT-only checks at the start of Phase 3:
- **Prefix cache:** a 9,216-token prompt sent 3 times should show about 4,608 cached tokens.
- **Warmup:** after post-ready warmup, the first c=2 TTFT should be 0.5 s or less.

### Kernel workstreams

| Workstream | Scope | Effort | Depends on |
|---|---|---|---|
| K1 FP8 W8A16 | A per-channel method (the existing PTPC method refuses Marlin) and a per-layer wrapper (`kda.py:172`, `model.py:331`, `:216-220`). Indexer, router, mHC, embed, kv_b, vision and `fused_qkv_a` stay BF16. | M-L | 0.9 plus a microbenchmark showing at least 0.85× cuBLAS bandwidth. If FP8 Marlin is missing from the library, so is MXFP8, and this becomes an image rebuild (XL). |
| K1n NVFP4 attention | Remove the forced-BF16 setting + build an offline checkpoint | L | D13; Tier 1 |
| K2 lm_head MXFP8 | DSv4.1 script + loader | M | K1 |
| K3 KDA speculative state | Commit-by-replay (+1-4% [P]); `.contiguous` trims | L / S | nsys |
| K4 SD-1 | Router hook in fused MoE | M | Census |
| K5 b12x W4A16 | Backport #52018 or a clamp patch | M-L | Microbenchmark |
| K6 / K7 | Router fp32; indexer factor 40 → 1 | S | Tier 0 / needle |

### GPU-slot budget

| Phase | Slot-hours |
|---|---|
| Phase 1 | ≈15-16 |
| Phase 2 (drafters 3.7, k 6.3-9, thinking 1.5, cache 1) | ≈12-15 |
| Phase 3 (conditional rows P3.7-P3.9 cost 0 if skipped) | ≈27-41 |
| Phase 4 | ≈4-6 |
| **Total** | **≈58-78** |

Each boot takes about 20 minutes. Over 80+ boots, the boot-time levers (quieting spark1, compile caches) are worth 1.5-10 slot-hours.

## 6. Measurement protocol and quality gates

### Ruler v2

**What gets published:** acceptance length, step_ms, tok/s = acceptance length / step_ms, and `ruler_version`.

**How steps are counted:** from per-request deltas of `num_drafts`, which stays valid with async scheduling on [R].
- step_ms = decode time / Δdrafts
- acceptance length = (tokens − 1) / Δdrafts
- Sanity check: inter-token count − Δdrafts must be 0 or 1.

**Cells:**

| Cell | Content |
|---|---|
| **A (published)** | 8 distinct prose prompts, exactly 512 tokens, greedy, thinking off, effort pinned |
| B | Code, 8 × 512 tokens |
| J | Structured count (internal only) |
| H / I | Distinct-prompt c=2 / c=4 |
| E | 32k and 128k TTFT |
| F | Image |
| G | Sampled |
| K | Legacy ~98-token prose, for continuity |

**Hygiene:**
- one discarded warm-up per cell;
- per-run `/metrics`, including per-position acceptance;
- per-stream sha256 and finish_reason;
- any swap activity invalidates the run.

The fast gate (A+B) takes about 7 minutes; the full panel about 25.

**Forced-length check (slot 1).** Run 16 prompts at natural length and forced to 512 tokens. Every prompt's natural length must be at least 576. The acceptance difference must be within the repeat-run spread, and step time within 1%.

**Keep rule** (3 boots per arm, z-test):
- **Keep** if the lower bound is above 0 and the gain is at least +3%.
- **Revert** if the upper bound is below +1%.
- **Otherwise** the result is inconclusive, never "reverted".

The smallest detectable change is about 5% at the observed 2.2% boot-to-boot variation. Changes that affect acceptance use a paired per-prompt bootstrap, which resolves about 2%.

### Quality tiers

**Tier 0** (every change, about 15 minutes):
- **Teacher-forced NLL and top-1 agreement** over a frozen 300k-token corpus. NLL may move by at most max(3σ, 0.005 nats/token). Top-1 agreement must stay within 0.5 points of a rerun.
- **Weight-quantization stages** must also reach top-1 agreement of at least 99% for FP8 or 98% for NVFP4, and top-20 KL of at most 1e-3 or 3e-3 respectively.
- **Behavioural checks:**
  - greedy first-divergence;
  - count-200;
  - kwarg matrix;
  - numeric-table and UTF-8 probes;
  - tool JSON at least 98%;
  - vision;
  - needle 2 of 3 or better.

**Tier 1** (pack, template, defaults, KV dtype, KDA quantization, ABLIT; about 2.75 hours per config):
- **Tasks:** IFEval 150, GSM8K 200, MMLU-Pro 280, BFCL 120, needle-in-haystack up to 300k, ChartQA 100, OCRBench 100, MMMU 60.
- **Pooled test:** the paired difference over all 1,010 items must be −2.0 points or better (standard error 0.93).
- **Per-group test:** the groups are IFEval, GSM8K, MMLU-Pro, BFCL and Vision (260 items). A group fails only if McNemar p < 0.01 **and** its drop is 5 points or more.
- **False-failure rate:** about 6.5%. A simple per-task −5-point rule would fail two identical packs 93-100% of the time.

**Tier-1-lite** (60 minutes) is used for the thinking-mode A/B.

## 7. Decisions needed

| # | Question | Recommended default |
|---|---|---|
| D1 | Publish ruler v2 cell A as the decode score? | **Yes.** Keep legacy row K and record the change in trail.tsv / decision.tsv. |
| D2 | Make nvidia the default before it has its own measurements? | **No.** Merge #11-A now and #11-B after the §2 acceptance criteria. |
| D3 | Default thinking mode | **Off**, plus the alias fix. Move to `effort:low` only if the corruption probe fires or Tier-1-lite improves. |
| D4 | k policy and workload weights (e.g. prose .4, code .3, thinking .2, structured .1) | **k=7** until P2.2. Structured output is a reported trade-off, not a veto. |
| D5 | Adopt newer DFlash2 revisions? | **A/B them** after a license check. |
| D6 | FP8 attention scope | Drafter, shared experts and MLA under Tier 0; **KDA opt-in only**, with Tier 1. |
| D7 | PR #3 | **Close.** |
| D8 | Stay on v11 or rebase onto v0.30.0? | **Stay on v11** plus the default-off v12 layer. v0.30.0 lacks DFlash2 aux capture and the SM121 sparse-MLA path. |
| D9 | Host policy: swappiness 0, passwordless sudo, quiet spark1, restore afterwards | **Authorize before Phase 1.** |
| D10 | Max-new-tokens ceiling | **65,536**, via `--override-generation-config`. |
| D11 | Video cap | **Keep {image:4, video:1}** and add a video test. |
| D12 | Prove the cause with an isolated compile test instead of a live auto-kernel boot? | **Yes.** |
| D13 | Pursue NVFP4-W4A16 attention (K1n)? | **Yes, after K1's kernel check**, as the second lever. |
| D14 | Grant SYS_ADMIN for ncu? | **Microbenchmark container only**, never the serve. |

## 8. Appendix

### Refuted or discarded

- **Speculative decoding:** adaptive verification on sm_121 (XL); block verification; "the community gap is step time".
- **Memory:** "vision adds 15.7 GiB"; page-cache fadvise; mm-cache shm; "8 GiB MM cache"; headroom as the sole boot cause; the 5.0 GiB KV rule; the 64 MB logits cap.
- **Performance models:** the 51+8.5n step model; the 85-95% bandwidth-bound framing; the +70% stacked figure.
- **Kernels and communication:** W4A4 MoE for decode; CPU pinning; PP/EP; custom all-reduce; RoCEnante; the SM12x varlen indexer (XL).
- **Dual rail:** its TTFT gain was about 3× overstated, and it would move the pinned HCA.
- **Measurement:** "swap costs 4-13%"; the ±2% ruler; "CI lacks render --check"; the "leave async off" A/B.
- **Weights and loading:** "LibertAI MTP is BF16"; the nvidia layer-2 1.40× mis-scale (the scales are tied); DFlash2-G; multithread, `sharded_state` and prefetch loading [R].

### Reinstated

NVFP4 attention. The claimed link to output corruption has no support: LibertAI keeps `self_attn` in BF16 and is still the pack implicated in #54150.

### Review comments I only partly accepted

- **Layer-45 skip:** it saves 0-53 s, not about 50 s, because those tensors may never be read from disk at all. Measure with `/proc/<pid>/io`.
- **QUAL-2 alias location:** it stays in the template, because the parser lives in the frozen image. The template-parity CI allows it as a documented difference.
- **Tier-1 false-failure rate:** the original rule was worse than the review estimated, 93-100% rather than 20-35%.

### Open questions

1. What is LibertAI's memory low during profiling?
2. Does DeepSeek run under Docker?
3. Is passwordless sudo available?
4. How large is the effect of the LibertAI mis-scale?
5. Were the nvidia `hc_*` weights rounded to BF16?
6. How much memory do the NCCL communicators use?
7. Why does spark1 load slowly?
8. How many distinct experts do rejected draft rows use?
9. What bandwidth does Marlin reach at M=8/16?
10. Does `prompt_logprobs` work with DFlash2?
11. Where does greedy nondeterminism come from?
12. Does the drafter always run block 8 regardless of k?
13. Does expert capture work with DFlash2?
14. Is transformers at 5.16.1 or later?
15. Does the TileLang JIT run as a subprocess?
16. What are spark2's VM settings?

### Evidence

- Full evidence: [`findings/`](findings/) (`INDEX.md` plus 11 dimension files, each finding with the verifier's verdict and corrections).
- Critic notes and gap-fill answers: [`critic.json`](critic.json), [`gaps.md`](gaps.md).
- Review inputs: vLLM/FlashInfer Python source copied read-only from `glm53-sm121-v11` (session scratchpad, not kept); nvidia boot proofs in `~/projects/ai-lab/local-ai-lab/internal/glm53-nvidia-*-proof.md`.