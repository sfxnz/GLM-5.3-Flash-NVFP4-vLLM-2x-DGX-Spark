# Dimension: sibling-recipes

## Reviewer summary

The richest sibling is DeepSeek-V4.1-Flash-EXL3 (34 recorded rounds on the same 2x GB10 / RoCE / NCCL 2.30.7 / mHC + sparse-MLA stack). It supplies measured levers, a noise-aware ABAB method, a quality gate, and profiling tools that carry over to GLM almost unchanged. Findings 1 and 2 matter most and are already backed by GLM's own data. First, PR #9's NUM_SPECULATIVE_TOKENS=5 arm measured +14.2% prose c=1 (19.23→21.97, acceptance 2.31, about 105 ms/step against about 114.6 ms/step at k=7) and +15.8% at c=2. It was reverted only because render/lint failed, not because of performance. DSv4.1's k-sweep with matched captures found the same effect (k=3 best, +5.5% from matched captures). Second, GLM's prose ruler is too noisy for its past verdicts: a natural stop at about 100 tokens, a 22% spread within 3 runs, and 21.2 vs 19.2 across boots of the same config (PR #8 parity). The DSv4.1 method fixes this: a longer prose cell, ms/step at matched acceptance, ABAB boots and a noise gate. On UMA, DSv4.1's NCCL AR-tail set (+2.8–3.5 GiB MemAvailable per rank) and its indexer-workspace factor (1.61 GiB→41 MiB at GLM's 327680) go straight at the main blocker: every L.A.I.L attempt to boot PR #11's nvidia pack died after weight load, at 0.45–10.4 GiB MemAvailable. The fadvise page-cache drop (it needs no sudo) goes after the same blocker. On kernels, the local evidence contradicts the b12x worktree's conclusion that no SM12x silu+clamp MoE kernel exists. Anemll's b12x patch (in the Mia lab repo) plumbs swiglu_limit into the standalone b12x TP MoE (W4A16). DeepSeek-V4-Flash, which uses the same 4096/2048 expert shape, silu and swiglu_limit=10, serves on b12x on 2x Spark. Separately, flashinfer_cutlass most likely died because runtime JIT ran with no jit-cache (v8 uninstalls it) and MAX_JOBS was unset (ninja defaults to about 22 parallel nvcc jobs). Qwen3.8-NVFP4 serves with FLASHINFER_CUTLASS on the same hardware, and v0.30.0 ships a matched flashinfer-jit-cache. GLM also reads its BF16 lm_head twice per step, because DFlash2 shares the target head. DSv4.1's lm_head MXFP8 (+5.9%, quality-neutral) therefore transfers with double weight, about −2.7 ms/step. A header-byte model of GLM puts about 28 GB/rank/step under independent routing, and 114.6 ms/step implies an implausible 98% of 250 GB/s. So routing must be strongly correlated, as DSv4.1's census found (duplicate fraction 0.30 vs 0.023 random), and GLM should run that census before sizing the MoE-kernel payoff. Smaller carry-overs: the post-ready warmup (GLM compiles the same lazy TileLang mHC kernels), PM QoS and NCCL eager-twin (about 1–3%), a single FORWARD_ENVS list plus an engagement audit, and post-ready boot floors in place of PR #12's 16 GiB wait-abort, which a healthy boot's construct floor already crosses. Everything here was read-only. No GPU, container or bench was touched.

## XP-1: Re-open the spec-k sweep: PR #9's k=5 win was reverted by a lint failure, and DSv4.1 found k=3 plus matched captures optimal

- kind=perf component=speculative decoding (DFlash2 num_speculative_tokens) + cudagraph capture sizes impact=5 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Prose decode is limited by verify-batch weight streaming, so k=7 is past the optimum for low-acceptance text. GLM's own k=5 arm beat k=7 on both c=1 and c=2 but never shipped, because the hillclimb driver's render/lint step failed. DSv4.1 swept k on the same hardware class and found k=3 best, with matched capture sizes worth another +5.5%.

**Mechanism:** Each extra verify row adds routed experts that must be streamed. Independent routing gives E[distinct] = 288·(1−(1−8/288)^m) per layer: 58.1 at m=8, 44.8 at m=6, 30.7 at m=4. At about 7.5 MB per expert per rank (header: 172.97 GiB of routed experts / 43 layers / 288 / TP 2) over 42 MoE layers, that is 18.3 / 14.1 / 9.7 GB per rank per step. Acceptance drops only slowly with k on prose (2.43 at k=7, 2.31 at k=5), so the bytes saved outweigh the tokens lost. The measured −9.5 ms for m 8→6 is smaller than the −21 ms the independence model predicts, which is consistent with correlated routing (see XP-10).

**Evidence:**
- PR #9 verdict.json (branch agent/hillclimb-20260903, evidence/iter-H1-20260903-20260903T010419Z/verdict.json): reason 'render or lint failed after keep', before_c1 19.23 → after_c1 21.97, before_c2 15.13 → after_c2 17.52
- PR #9 bench.txt: k=5 prose c=1 runs 21.97/23.62/19.86, acceptance_len 2.308; c=2 acceptance 2.41
- Published k=7: prose c=1 21.2 @ acceptance 2.43 → 21.2/2.43 = 8.72 steps/s = 114.6 ms/step; k=5: 21.97/2.308 = 9.52 steps/s = 105.1 ms/step (−9.5 ms, −8.3%)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-fusion-host/flags.md:25 (L.A.I.L k5 26.32 / k4 25.87 / k3 28.76 / k2 26.89; E6 k=10 collapsed to 12.1)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-fusion-host/flags.md:26 and :325-341 (captures [1,5,6,10,12]→[1,3,4,6,8] = +5.5%, 27.27→28.76)
- Qwen3.8 NVFP4 k=3→2: prose inside noise, structured c=8 −6.97 (Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark/evidence/decision.tsv row c5-mtp-k2)
- GLM run.sh:88-97 already derives captures 1,2,4 + (k+1)×{1..seqs}; at k=5 that gives [1,2,4,6,12], so they stay matched automatically

**Proposed action:** Run NUM_SPECULATIVE_TOKENS ∈ {3,4,5} against 7 using the XP-2 protocol. Captures follow automatically from the run.sh formula. Use SPEC=dflash2 and keep MAX_NUM_SEQS=2 for comparability. Also record the c=4 occupancy row: AGENTS.md already says k=5 is what admits four sequences. Fix the kit's render/lint path so a keep can land.

**Est. impact:** Measured on one boot: k=5 = +14.2% prose c=1 (21.97 vs 19.23) and +15.8% c=2, −8.3% ms/step vs the published k=7. For k=3/4, estimate a step of about 90–95 ms (a further −10–15 ms from about 20 fewer distinct experts per layer at correlated routing). At an estimated acceptance of about 2.1–2.2 that is about 22–24 tok/s, roughly flat to +5% over k=5. The expected optimum is k=4–5.

**Validation:** ABAB (XP-2): A = k7, B = k5, then A = k7 vs B = k4 and k3, 2 boots each. Primary metric: prose_long c=1 ms/step, with per-boot acceptance reported. Gate: the B-vs-A improvement must exceed the larger per-arm boot spread, and every B boot must beat the A median. Non-inferiority on structured c=1/c=2 and the count probe (200 consecutive).

**Risks:** Structured (high-acceptance) cells lose at lower k: 67.6 tok/s at acceptance 7.84 needs k≥7 (a structured-heavy workload may prefer k=7). DFlash2 is trained on block 8, so draft-slot acceptance per position may not be monotone. This is a deliberate change to the recipe decision 'default DFlash2-7', made because PR #9 evidence shows a prose win that was never evaluated on performance.

**Verifier reasoning:** Checked PR #9 via gh api. The iter-H1 verdict.json does say 'render or lint failed after keep', 19.23->21.97 c=1 and 15.13->17.52 c=2. The reviewer missed the later commits on the same PR. Commit 'perf(recipe): H1-20260903 NUM_SPECULATIVE_TOKENS=5 (+2.74 tok/s c=1, hand-applied)' fixed lint (c0c4af3) and set recipe.yaml serve.env NUM_SPECULATIVE_TOKENS: 5 (recipe.yaml:49 on agent/hillclimb-20260903). decision.tsv row H1-20260903-apply says 'kept'. A full DFlash2-5 rebench followed (evidence/rebench-dflash5-20260903T045815Z/summary.json): prose c=1 21.76 (acc 2.354), prose c=2 19.62/stream (acc 2.57), structured c=1 55.73 (acc 6.0, capped at k+1), structured c=2 41.56/stream. That rebench failed its gate only on a needle-8192 refusal. PR #9 is still OPEN and unmerged, so main stays at k=7. So 'never shipped because lint failed / never evaluated on performance' is wrong. The +14.2% compares against a low k=7 parity boot (19.23). Other k=7 boots: 21.18 (published rebench-20260902T204243Z) and 18.98/20.85 (b12x-A-marlin result.txt). ms/step does fall on two cells: prose 21.76/2.354 = 108 ms vs 114.6 ms, and structured 55.73/6.0 = 107.7 ms vs 67.64/7.84 = 116 ms. That is about 3.5 ms per draft slot, not the 90-95 ms extrapolated for k=3/4. The capture formula at run.sh:88-97 gives [1,2,4,6,12] at k=5, which I confirmed. DSv4.1's k=3 optimum is for DSpark (block 5, a different drafter), so it transfers weakly.

**Verifier corrected claim:** k=5 was already hand-applied and rebenched on PR #9 (still open). On one boot it measured prose c=1 21.76 (+2.6% vs published 21.2; about +7% vs the mean of the three k=7 boots, 20.4) and prose c=2 19.6/stream (+18%). Structured regressed: c=1 55.7 vs 67.6 (-17.6%) and c=2 41.6 vs 60.4/stream (-31%). Step time fell about 7 ms (114.6 -> about 108 ms), so each draft slot costs about 3.5 ms. The gate failure was a needle refusal, not performance.

**Verifier corrected impact:** Prose c=1 +3-7% (inside the current single-boot noise). Structured c=1 -18% measured. At about 3.5 ms/slot, k=3/4 would reach about 101-105 ms/step. With acceptance around 2.1-2.2 that is roughly flat to k=5 on prose and worse on structured. It is a workload trade-off, not a +14% win.

## XP-2: Adopt DSv4.1's bench-honesty and ABAB protocol: GLM's prose ruler has a 22% within-boot spread and ±10% between boots

- kind=methodology component=bench_decode.py / keep-revert discipline (evidence/decision.tsv) impact=4 confidence=5 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** Most past GLM keep/revert calls (+/−5–15%) sit inside ruler noise. The prose prompt stops naturally at about 100 tokens (about 40 verify steps), and each verdict comes from one boot and 3 runs. DSv4.1 found the same pathology and replaced it with a longer prose cell, ms per verify step at matched acceptance, interleaved ABAB boots and a noise-derived gate.

**Mechanism:** Short cells amplify TTFT/first-step jitter, JIT and swap faults, and step-quantization error: at 2.4 tokens per step, ±1 step is ±2.5%. Single-boot A/B cannot separate a lever from boot-to-boot variance: allocator layout, swap state, and NCCL rank skew (DSv4.1 measured rank-skew waits of 1.34–3.92 ms/step).

**Evidence:**
- PR #8 PARITY-RESULT.md (branch agent/kit, evidence/parity-20260902T225109Z/): same config, prose c=1 21.2 ref vs 18.9 / 19.2; the reference's own three runs were 23.40 / 18.75 / 21.18 (22% spread)
- GLM bench_decode.py:18-22 (prompt asks for about eighty words), :30-41 (max_tokens 200, no ignore_eos); README.md:20 'Prose now stops near ... (~105 tokens)'; PR #9 median_completion_tokens 100
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-comm/ARMS.md:47-91 (ABAB, ms/step primary metric, noise = larger boot spread, honest cells: natural_finish_reason=length, quality gate before KEEP)
- ARMS.md:50-53: identical DSv4.1 config measured 27.40 / 26.13 / 28.01 across boots; lm_head went from REVERT at +2.5% to KEEP at +5.9% at higher n
- DSv4.1 README.md:171 and recipe.yaml:117 (ignore_eos prose had 59% post-EOS text; L.A.I.L prose = 512 tokens, t=0.2, n=10 per boot)

**Proposed action:** Add a prose_long phase: a 512-token natural-length prompt at t=0 plus a t=0.2 variant, runs ≥9, with the finish_reason recorded. Compute ms/step = decode_s / (decode_tokens / acceptance_len) per run from /metrics deltas. Gate keeps on ABAB boots with DSv4.1's noise rule. Keep the 'prose-only published score' rule, but publish the long prose cell. Port tools/four_numbers.sh and tools/measure_lail_prose.py (the harness must also check that ranks match: serve_env_ranks_match).

**Est. impact:** No direct tok/s. Lowers the detectable effect from about 10% to about 3%, which is what XP-4/7/9/11-sized levers (2–5% each) need. It would have prevented the k=5 loss (XP-1, +14%).

**Validation:** Offline: re-analyze existing bench.json files for spread. Online: one A/A pair of boots of the current config under the new ruler to set the noise band before any arm.

**Risks:** Costs about 4 boots per decision (about 15–20 min each on GLM). A 512-token prose cell has a different acceptance mix from the published 100-token one, so a new baseline is needed. The AGENTS.md rule 'Do not score decode from structured' is kept.

**Verifier reasoning:** rebench-20260902T204243Z/bench.txt lines 2-4 show prose c=1 runs 23.40/18.75/21.18. (23.40-18.75)/21.18 = 22%, and these are greedy t=0 on an identical prompt, so acceptance is identical and the whole spread is timing noise. bench_decode.py:18-22 asks for about 80 words with temperature 0 and max_tokens 200, and median_completion_tokens is 98/100. Across boots of the same k=7 config: 21.18, 19.23 (parity) and 18.98/20.85 (b12x-A-marlin result.txt waves). The PR #9 hillclimb itself used an 8% noise band that the k=5 arm passed only because the baseline boot was low. The ms/step metric works: the rebench numbers give a consistent about 115 ms/step on both prose and structured.

**Verifier corrected impact:** No direct tok/s. It is the prerequisite for any lever under about 10%. Note that the PR #9 k=5 'loss' was not a loss (see XP-1), so the claim that the protocol 'would have prevented the k=5 loss' is moot.

## XP-3: A silu + swiglu_limit SM12x fused-MoE path exists: the standalone b12x TP-MoE (W4A16) used by Anemll, eugr and DSV4-Flash stacks

- kind=perf component=routed MoE kernel (MOE_BACKEND=marlin today) impact=5 confidence=2 effort=L needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The b12x worktree reverted because flashinfer 0.6.18's FLASHINFER_B12X wrapper does not clamp silu. The standalone lukealonso/b12x package's b12x.integration.tp_moe does take swiglu_limit with activation 'silu' and a W4A16 quant_mode. It is wired into vLLM by Anemll's official-main-b12x-nvfp4-python.patch, and the eugr/spark-vllm-b12x image ships it. DeepSeek-V4-Flash, which shares GLM's expert shape (hidden 4096, moe_intermediate 2048, silu, swiglu_limit 10.0), runs its MoE on this b12x path on 2x Spark.

**Mechanism:** Marlin on sm_121 is a generic W4A16 dequant GEMM path. It logs 'GPU does not have native support for FP4'. b12x is a CuTe-DSL SM12x kernel family tuned for small-M decode, with fused routing→W1→SwiGLU(clamp)→W2 and tunable blocks per SM and tiles (VLLM_B12X_W4A16_FORCE_*). W4A16 keeps activations in BF16, so it has the same numerics class as Marlin and no W4A4 quality risk.

**Evidence:**
- /home/sfxnz/lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/patches/official-main-b12x-nvfp4-python.patch:261-340 (b12x_mxfp4_moe.py: plan_tp_moe_scratch(... activation, swiglu_limit, source_format, quant_mode)), :380-430 (b12x_moe_fp4 binding with swiglu_limit), :433 (SILU→'silu'), :948/1007/1040 (quant_mode='w4a16'), :951 (swiglu_limit=gemm1_clamp_limit)
- DeepSeek-V4-Flash-0731 and Vision-Exp config.json: hidden_size 4096, moe_intermediate_size 2048, hidden_act silu, swiglu_limit 10.0 (HF cache snapshots 7872f01 / 86f746b)
- docker history eugr/spark-vllm-b12x:latest: B12X_REPO=lukealonso/b12x B12X_REF=master, CUTLASS_DSL_VERSION=4.7.0, TORCH_CUDA_ARCH_LIST=12.1a (image built 2026-08-23)
- /home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4-Flash-Vision-Exp-vLLM-2x-DGX-Spark/run.sh:27,222,226 (MOE_BACKEND b12x, VLLM_USE_B12X_MOE=1, B12X_MOE_FORCE_A8=1); README prose c=1 26.2
- /home/sfxnz/lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/docs/GLM-NEW-REPORT.md:14-38 (--moe-backend flashinfer_b12x, VLLM_USE_B12X_MOE=1) and results/RESULTS-2026-08-14.md (c=1 62–83 tok/s, ignore_eos, 128-token cells)
- GLM b12x worktree evidence/hypotheses.md H-b12x: 'Need a silu+clamp SM12x fused-MoE kernel before this is a tok/s hill'
- v11 vllm model_executor/layers/fused_moe/oracle/nvfp4.py:191-199 (FLASHINFER_B12X absent from NVFP4_BACKENDS_WITH_CLAMP)

**Proposed action:** Build a v12+ image layer: pip install b12x at a pinned commit plus nvidia-cutlass-dsl 4.7.0. Port a B12xNvfp4Experts class modeled on Anemll's b12x_mxfp4_moe.py with source_format for NVFP4 (block 16 plus a global scale), quant_mode w4a16, activation silu and swiglu_limit=10. Register it in oracle/nvfp4.py and add it to NVFP4_BACKENDS_WITH_CLAMP. First run a single-GPU microbench on one GB10 against Marlin at the serve shapes (E=288, top-8, K=4096, N=1024 per rank, M ∈ {1,2,6,8,12,16}) with bitwise or tolerance correctness.

**Est. impact:** Unmeasured for GLM. Routed MoE is an estimated 12–18 GB of the roughly 22–28 GB streamed per rank per step (XP-10), about 55–65% of the 114.6 ms step. If Marlin runs at about 55% of 250 GB/s and b12x reaches DSv4.1's MoE kernel efficiency (78% at 194 GB/s, results/2026-09-25-kernels/profile/timeline-r3.txt), MoE time drops by about 30%, so the step drops by 17–20% (+20–25% tok/s). If Marlin is already at 70% or more, the gain is under 5%.

**Validation:** Step 1, single-GPU microbench on spark2 when free: ms per MoE layer, Marlin vs b12x at M=8/16 with L2 cold (DSv4.1's 2x-L2 flush methodology, k3-comm iterations.txt #6). Step 2: serve ABAB with MOE_BACKEND=b12x_nvfp4 vs marlin, then count probe, quality_eval (XP-12) and XP-2 gates.

**Risks:** CuTe-DSL JIT after 90 GiB of weights can OOM like cudafe++ did (mitigate with XP-5's persistent cache pre-warm). The Qwen guard cites 'Xid 31 reports on sm_121' for b12x, but no local evidence file backs that claim (grep found only the guard string). b12x master is unpinned in eugr, so pin a commit. The NVFP4 source_format and 16-element scales are assumed from the b12x_moe_fp4 signature and must be verified in the package source. The rule 'any MOE_BACKEND other than marlin' needs an explicit exception after measurement.

**Verifier reasoning:** A silu+clamp SM12x fused MoE does exist, but it is already inside GLM's own v11 image, so no standalone lukealonso/b12x or Anemll patch is needed. v11 flashinfer fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py:4280-4294 (_clamp_swiglu_inputs) clamps gate<=limit and up to [-limit,limit] whenever swiglu_limit is set, for silu (:221-222 normalizes the limit for 'silu'). b12x_moe.py:270-285: B12xMoEWrapper takes quant_mode ('w4a16'), source_format='modelopt' (NVFP4) and swiglu_limit. Only the W4A4 micro/dynamic path ignores the limit for silu (moe_activation.py:112-118 clamps only swigluoai). The actual blocker is vLLM's wrapper: flashinfer_b12x_moe.py:242-251 builds B12xMoEWrapper without swiglu_limit or quant_mode, and oracle/nvfp4.py:191-199 leaves FLASHINFER_B12X out of the clamp set. The Anemll patch the reviewer cites is MXFP4-only: patch :703-713 asserts weight_quant_dtype == 'mxfp4' with source_format 'fp4_e8m0_k32', so it does not port to NVFP4 as described. The DSV4-Flash shape match (4096/2048/silu/10.0) is confirmed in the HF cache configs. The perf estimate (+20-25%) rests on an unmeasured Marlin efficiency. The 'native FP4' warning cited as mechanism is unconditional in prepare_nvfp4_moe_layer_for_marlin (marlin_utils_fp4.py:352-357), so it is not evidence about Marlin efficiency.

**Verifier corrected claim:** GLM's v11 FlashInfer 0.6.18 already ships a b12x W4A16 MoE kernel that applies swiglu_limit on silu and reads ModelOpt NVFP4 weights. The b12x worktree's conclusion ('kernel clamps only swigluoai; do not retry on this image') holds only for the W4A4 path. Enabling it needs a small vLLM patch: pass swiglu_limit, quant_mode='w4a16' and source_format='modelopt' into B12xMoEWrapper, check the weight prep for w4a16, and add FLASHINFER_B12X to NVFP4_BACKENDS_WITH_CLAMP when w4a16 is used. No new package, CuTe-DSL bump or Anemll port is needed.

**Verifier corrected impact:** Unmeasured. It could be 0-20% of step time if routed-MoE streaming is the bottleneck and Marlin is well below b12x bandwidth efficiency. Effort drops from L to about M (a v12 layer patch plus a single-GPU microbench).

## XP-4: Port DSv4.1's UMA headroom bundle (NCCL AR-tail set, indexer-workspace factor, fadvise page-cache drop, adaptive empty_cache): about 5 GiB per rank aimed at the nvidia-pack boot cliff

- kind=ops component=run.sh env / vLLM indexer workspace / NCCL buffers impact=4 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The nvidia pack cannot currently reach /v1/models under an 8 GiB watcher. Six L.A.I.L variants all died after the 90.36 GiB load, at 0.45–10.4 GiB MemAvailable. DSv4.1 measured env-only levers that return about 5 GiB per rank on the same NCCL 2.30.7 / GB10 stack with flat decode. GLM sets none of them, and its maybe_drop_caches no-ops without passwordless sudo.

**Mechanism:** On GB10, cudaMalloc, NCCL buffers (NCCL_CUMEM_ENABLE=0, so cudaMalloc'd), the profile-run workspaces and anonymous host pages share one 121 GiB pool. The post-load cliff comes from profile + KV + graph + workspace temps on about 20 GiB of leftover. Shrinking fixed NCCL buffers and the 1.61 GiB indexer gather buffer (it only has to hold one chunk's summed prefix) raises that trough directly. The fadvise drop returns clean shard pages that the driver does not reclaim.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark/results/2026-09-20-nccl/VERDICT.md: NCCL_BUFFSIZE=1048576, NCCL_LL128_BUFFSIZE=262144, NCCL_PROTO=^LL128, NCCL_MAX_NCHANNELS=8 → pinned buffers 4.7→0.14 GiB; MemAvailable after 32k prefill +3.49 / +2.79 GiB; prose flat (34.69 vs 34.67); L.A.I.L +3.1–4.8%; 0 NCCL WARN
- results/2026-09-20-memhygiene/VERDICT.md: indexer factor 1 + page-cache drop + logits cap 256 MB + empty-cache floor → +5.82 / +6.33 GiB; fadvise drop 'MemFree 13.07→33.39 GiB' on TP0
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-comm/docker/patch/indexer_workspace.py (max_model_len*40 → *factor), drop_page_cache.py (POSIX_FADV_DONTNEED per shard, no sudo), prefill_empty_cache.py
- GLM uses the same buffer: v11 models/glm5next/nvidia/attention.py:300-313 get_max_prefill_buffer_size → v1/attention/backends/mla/indexer.py:636-646 returns max_model_len*40 = 327680×40×132 B = 1.61 GiB per rank (factor 1: 41 MiB)
- v11 sparse_attn_indexer_kpool.py:313-315 (profile sentinel = max(decode logits, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB))
- GLM run.sh:177-181 (maybe_drop_caches needs sudo -n), :250-264 (no NCCL buffer env)
- /home/sfxnz/projects/ai-lab/local-ai-lab/internal/glm53-nvidia-{spark,novideo,skipmm,eager,langonly,batch1024}-proof.md: post-load troughs 0.59/0.45, 5.01/2.81, 6.46/9.77, 2.08/3.57, 10.43/7.35, 6.39/4.37 GiB
- Host now: vm.swappiness=60, 16G swapfile with 3.5G used, cgroup2 (so docker --memory-swappiness is ignored)

**Proposed action:** One ABAB arm per item, in this order. (a) The NCCL AR-tail set: 4 env vars, forwarded to the worker. (b) GLM_INDEXER_PREFILL_FACTOR=1 through an idempotent source patch of indexer.py:646, or a v12 layer. (c) VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256. (d) The in-container fadvise DONTNEED of model and draft shards, wrapped around Worker.load_model and compile_or_warm_up_model. (e) Optionally the adaptive prefill empty_cache. Record MemAvailable at ready, after smoke and after a 32k needle on both nodes.

**Est. impact:** Memory: about +3.5 GiB (NCCL, DSv4.1-measured) plus 1.57 GiB (indexer, computed) ≈ +5 GiB per rank. That lifts every recorded nvidia-pack trough (0.45–10.4 GiB) by the same amount, clearing 8 GiB in 4 of 6 variants on arithmetic alone. Decode: DSv4.1 measured prose flat and L.A.I.L +3–5% (inside noise). For GLM, less swap should remove the first-wave penalty (structured c=2 about 52 vs 60 while 1.3 GiB was swapped, README.md:22).

**Validation:** Boot the nvidia pack with (a) and (b) only, watcher at 8 GiB or better still post-ready floors (XP-13), and log MemAvailable per 1 s through construct, profile, TileLang and capture. Then run XP-2 ABAB decode for non-inferiority, plus a 2×20k-needle occupancy probe for the indexer factor (it must still hit).

**Risks:** Indexer factor 1 caps the gathered prefix of a single chunk at max_model_len entries. The chunk planner splits beyond that (DSv4.1 found pp flat), but GLM's kpool path must be checked with a 318k needle. NCCL_PROTO=^LL128 is safe for decode-size ARs (DSv4.1 isolated AR: LL 43 µs vs LL128 81.8 µs). Page-cache drop adds I/O on reboot only. Lowering host vm.swappiness is host-wide and needs user consent, so it is not proposed as a default.

**Verifier reasoning:** Indexer math is confirmed: indexer.py:636-646 returns max_model_len*40, attention.py:300-313 feeds it to SparseAttnIndexerKpool, and _gather_workspace_shapes (sparse_attn_indexer_kpool.py:201-216) gives (T,128) fp8 + (T,4) = 132 B. 327680*40*132 = 1.611 GiB, and factor 1 gives 41 MiB. The WorkspaceManager (v1/worker/workspace.py:119-179) grows one buffer to the largest request. The SM90 MLA backend keeps its own workspace (flashinfer_mla_sparse_sm90.py:75), so the indexer is plausibly the dominant requester and a saving of about 1.57 GiB is credible. VLLM_SPARSE_INDEXER_MAX_LOGITS_MB defaults to 512 (envs.py:59), so setting 256 saves at most 256 MiB, and only in the profiling sentinel. The NCCL set is absent from run.sh:250-264 (confirmed), but the 4.7->0.14 GiB figure is DSv4.1's own and GLM's is unmeasured. The central framing is wrong. The proof docs show every nvidia-pack 'death' was an 8 GiB watcher kill with no dmesg OOM (glm53-nvidia-spark-proof.md:92,137). The 0.45-10.4 numbers are the first sample below the threshold, not troughs: the spark proof fell 13.8->0.5 GiB in 7 s during encoder profile + TileLang, so a +5 GiB lift does not clear 8 GiB 'on arithmetic'. The nvidia per-rank working set, 90.36 GiB, is smaller than LibertAI's 90.67. A LibertAI boot on the same image also ran the max-size video encoder profile and TileLang (rebench engine-rank1.log.tail:47-52) and survived without a watcher. The fadvise drop is near-null for MemAvailable here: page cache already counts as available, and after stop buff/cache was only 0.6-0.8 GiB (glm53-nvidia-spark-proof.md:93). DSv4.1's cited gain was MemFree.

**Verifier corrected claim:** The NCCL AR-tail set and the indexer factor are reasonable headroom levers: about 1.57 GiB computed for the indexer, NCCL unmeasured on GLM, and at most 0.25 GiB from the logits cap. The fadvise drop should be dropped for GLM. Headroom is not what blocks the nvidia pack: its boots were stopped by an 8 GiB watcher at a transient dip, which the published LibertAI config would also cross.

**Verifier corrected impact:** About +1.6 GiB (computed) plus an unknown NCCL gain per rank. Reduced swap pressure could be a small decode win. It does not by itself turn 'cannot boot' into 'boots'.

## XP-5: The flashinfer_cutlass OOM is a runtime-JIT artifact: GLM's image uninstalled flashinfer-jit-cache and runs ninja at default parallelism. Pre-build into a persistent cache and cap MAX_JOBS

- kind=ops component=image build (v8 base layer) / run.sh JIT environment impact=3 confidence=3 effort=M needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The v8 base layer runs 'pip uninstall -y flashinfer-jit-cache' next to flashinfer-python 0.6.18.dev20260819, so every FlashInfer CUDA module compiles at runtime. FlashInfer's ninja uses MAX_JOBS only when set, so the default is nproc+2 (about 22 jobs on 20 cores). The CUTLASS fused-MoE JIT therefore launched many cudafe++/cicc processes with about 18 GiB free and OOM'd spark2. The same FLASHINFER_CUTLASS NVFP4 MoE runs on 2x Spark in the Qwen3.8 recipe (43.2 tok/s prose c=1), where 63.6 GiB/rank of weights leaves room to JIT. v0.30.0 ships a matched flashinfer-jit-cache==0.6.18.post1.

**Mechanism:** JIT peak memory scales with ninja parallelism times per-translation-unit cudafe++/cicc memory (CUTLASS MoE TUs take several GB each). It lands after weights are resident because MoE kernel selection happens at process_weights_after_loading and warmup. A pre-populated cache means zero compiles at serve time. A small MAX_JOBS bounds any residual compile.

**Evidence:**
- docker history glm53-sm121-v11: 'pip install flashinfer-python==0.6.18.dev20260819 flashinfer-cubin==0.6.18.dev20260819 ... && pip uninstall -q -y flashinfer-jit-cache && ... nvidia-cutlass-dsl==4.6.2'
- docker history vllm/vllm-openai:v0.30.0-aarch64: FLASHINFER_VERSION=0.6.18.post1, 'uv pip install flashinfer-jit-cache==${FLASHINFER_VERSION}', NCCL_VERSION=2.30.7, TORCH_CUDA_ARCH_LIST '8.0 ... 12.0' (no 12.1a)
- v11 flashinfer/jit/cpp_ext.py:346-350 (_get_num_workers returns None unless MAX_JOBS is set); :94-97 FLASHINFER_NVCC_THREADS default 1
- v11 flashinfer/fused_moe/core.py:567-569 get_cutlass_fused_moe_module('120') → gen_cutlass_fused_moe_sm120_module().build_and_load(); jit/env.py:59-62,157,162 cache at $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<ver>/<arch>/cached_ops
- /home/sfxnz/projects/ai-lab/recipes/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark/evidence/boot/backend-needles.txt:1 ('Using FLASHINFER_CUTLASS NvFp4 MoE backend'); evidence/opt-c5-mtp-k2/run.log:108 (Model loading took 63.61 GiB), :143 (autotune cache .../0.6.18/121a/...)
- GLM run.sh:315-322 (only the HF cache is mounted; no ~/.cache/flashinfer, triton, tilelang or vllm compile cache); README.md:22 'First wave after restart pays Triton JIT per batch shape'

**Proposed action:** (1) Add -v $HOME/.cache/glm53-jit:/root/.cache (or FLASHINFER_WORKSPACE_BASE plus TRITON_CACHE_DIR plus the TileLang cache) to both ranks. (2) Add a prewarm step: a model-less container runs get_cutlass_fused_moe_module('120') (and later any b12x/CuTe kernels) with MAX_JOBS=4 and FLASHINFER_CUDA_ARCH_LIST=12.1a while about 115 GiB is free. (3) Export MAX_JOBS=2 in the serve env. (4) Longer term, rebuild the chain on a matched flashinfer-python + flashinfer-jit-cache pair (0.6.18.post1), after checking the jit-cache archs cover 12.1a or 12.0f. Only then retest FORCE_UNSAFE_MOE=1 MOE_BACKEND=flashinfer_cutlass, the W4A4 path that uses the calibrated input_scale tensors.

**Est. impact:** It unblocks a backend that could not be tested. Its decode speed on GLM is unknown (Qwen3.8 is a different model). It also removes per-boot Triton/TileLang JIT, so first-request TTFT and bench variance fall (XP-9), and boot time drops by the JIT minutes.

**Validation:** Prewarm needs no model and possibly no GPU (nvcc only). Check that the .so lands in cached_ops/. Then boot with FORCE_UNSAFE_MOE=1 MOE_BACKEND=flashinfer_cutlass under a MemAvailable log and grep for 'cudafe' or 'Compiling' after load. There must be none.

**Risks:** FlashInfer keys cached ops by version, arch and source hash, so the cache is invalidated by any flashinfer bump. A shared host cache across images must be per-image. W4A4 changes numerics against Marlin's W4A16 and needs XP-12 gating. It is also Tony's/LibertAI calibration: AGENTS.md says do not set VLLM_GLM53_MOE_INPUT_SCALE=1.0.

**Verifier reasoning:** docker history glm53-sm121-v8 shows 'pip install flashinfer-python==0.6.18.dev20260819 flashinfer-cubin==... && pip uninstall -q -y flashinfer-jit-cache'. The base jit-cache was 0.6.17, so it was a version mismatch. v11 flashinfer jit/cpp_ext.py:346-350 returns None unless MAX_JOBS is set, so ninja uses its default (nproc=20 -> 22 jobs). v0.30.0-aarch64 history has FLASHINFER_VERSION=0.6.18.post1 and flashinfer-jit-cache==${FLASHINFER_VERSION}. evidence/oom-20260831/diagnosis.txt lists global_oom, invokers 'dockerd, cudafe++' and 'remaining ~18GiB exhausted during cudafe++/graph capture'. run.sh mounts only the HF cache (plus the chat template on head). Causation by parallelism specifically is inferred, not measured.

**Verifier corrected impact:** Unblocks testing flashinfer_cutlass. Its decode value on GLM is unknown. Note that the in-image b12x W4A16 route (XP-3 corrected) and --moe-backend cutlass (XP-6) are cheaper W4x MoE candidates to try first.

## XP-6: Build vLLM _C for 12.1a as eugr does, enabling the AOT VLLM_CUTLASS NVFP4 MoE: clamp-capable, family-120, no runtime JIT

- kind=perf component=image build (vLLM wheel arch list) impact=3 confidence=2 effort=L needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** vLLM's own CutlassExpertsFp4 supports device family 120, is in NVFP4_BACKENDS_WITH_CLAMP, and consumes the NVFP4 input scales (W4A4). It is compiled into _C ahead of time, so there is no cudafe++ at serve time. It needs FP4 kernels built for sm_121a, but GLM's image inherits TORCH_CUDA_ARCH_LIST '8.0 ... 12.0'. eugr/spark-vllm-b12x builds vLLM and FlashInfer wheels with TORCH_CUDA_ARCH_LIST=12.1a.

**Mechanism:** Arch-specific (the 'a' suffix) FP4 block-scaled MMA cubins built for sm_120a do not run on sm_121. A 12.1a build puts real SASS for GB10 into _C, which selects a mature CUTLASS grouped GEMM for NVFP4 MoE and dense FP4, with nothing to JIT.

**Evidence:**
- v11 vllm model_executor/layers/fused_moe/experts/cutlass_moe.py:690-721 (CutlassExpertsFp4: family 100/110/120, kNvfp4Static x kNvfp4Dynamic, process_weights_after_loading folds input_scale)
- v11 oracle/nvfp4.py:191-199 (VLLM_CUTLASS is in NVFP4_BACKENDS_WITH_CLAMP)
- docker image inspect eugr/spark-vllm-b12x:latest: TORCH_CUDA_ARCH_LIST=12.1a, FLASHINFER_CUDA_ARCH_LIST=12.1a, MAX_JOBS=16; history: 'uv pip install /workspace/flashinfer-wheels/*.whl /workspace/vllm-wheels/*.whl'
- Task facts / v0.30.0 inspect: official images carry TORCH_CUDA_ARCH_LIST '8.0 8.7 8.9 9.0 10.0 11.0 12.0'

**Proposed action:** Add a build-only lane: a Dockerfile stage that compiles the fork's vLLM (the glm5next code is fork-only, so it must be the image's own vLLM source) with TORCH_CUDA_ARCH_LIST=12.1a and MAX_JOBS≈8, on a node with no serve up. Microbench CutlassExpertsFp4 against Marlin at the XP-3 shapes before any serve boot.

**Est. impact:** Unknown until microbenched. It is an alternative route to a native-FP4 MoE if b12x (XP-3) stalls. W4A4 halves activation bytes, which is negligible at decode, so the gain has to come from kernel efficiency. The expected range is the same as XP-3 (0 to 20% step).

**Validation:** Build without GPU (compile only). Validate with a single-GPU kernel microbench plus an NVFP4 numerics check against Marlin (max abs error), then the serve ABAB and quality_eval.

**Risks:** A multi-hour build on a Spark (CPU/RAM contention with the busy workload, so schedule it in a free window). The fork source must be available. W4A4 quality has to be checked against the W4A16 Marlin baseline. It may be slower than Marlin at M≤16, since grouped GEMMs are tuned for large M.

**Verifier reasoning:** CutlassExpertsFp4 accepts family 120 (cutlass_moe.py:708-714) and VLLM_CUTLASS is in the clamp set (oracle/nvfp4.py:191-199). The mapping 'cutlass' -> VLLM_CUTLASS already exists (oracle/nvfp4.py:147-148), and the b12x-B failure message itself lists 'cutlass' as clamp-capable. The premise that the image's _C lacks sm_121 FP4 SASS is not verified. The 'native support for FP4' warning is printed unconditionally by the Marlin prep path, and no log shows cutlass_fp4_supported() failing or a 'Using ... for NVFP4 GEMM' line. vLLM builds on CUDA 13 may compile FP4 for the 12.0f family, which would run on sm_121. The cheapest first step is therefore a single-GPU test of --moe-backend cutlass on the existing v11 image, not a multi-hour 12.1a rebuild.

**Verifier corrected claim:** VLLM_CUTLASS NVFP4 MoE is already selectable with --moe-backend cutlass on v11. Whether its sm_12x kernels run on sm_121 in this build is unknown. Test that before any rebuild. A 12.1a rebuild is needed only if the kernel fails with 'no kernel image' or is not supported.

**Verifier corrected impact:** Unknown. It is a W4A4 route with the same 0-20% range as XP-3. Effort is S to try on the existing image and L only if a rebuild is needed.

## XP-7: lm_head to MXFP8/FP8: DSv4.1 won +5.9% at no quality cost, and GLM reads its BF16 head twice per step because DFlash2 shares it

- kind=perf component=lm_head (target verify + DFlash2 draft candidate top-k) impact=3 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** GLM's lm_head is BF16 [154880,4096] (1.18 GiB) and is streamed twice per step: once for target verify logits and once for DFlash2's get_top_k_tokens, which calls _apply_head on the shared target head. DSv4.1 re-encoded only lm_head to MXFP8 and measured a KEEP with neutral quality.

**Mechanism:** Weight-streaming bound: per rank, 2 × 634 MB = 1.27 GB per step BF16 against 2 × 330 MB MXFP8 (1 B per element plus 1/32 ue8m0 scale).

**Evidence:**
- Safetensors header (nvidia 09b04e5): lm_head.weight BF16 [154880,4096] = 1.18 GiB (xpoll_bytes.py tally)
- v11 vllm/v1/worker/gpu/spec_decode/dflash/utils.py:65-72 (dflash_model.lm_head = target_lm_head); DFlash2 snapshot 7d74cdd model.safetensors header has no lm_head tensor (1.17B params)
- v11 model_executor/models/qwen3_dflash2.py:283-287 and model_executor/layers/logits_processor.py:241-264 (get_top_k_tokens runs _apply_head on the full vocab shard)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-fusion-host/flags.md:54 (R33 KEEP, L.A.I.L 31.37→33.23 = +5.9%; s12 quality A/B ΔNLL −0.00045, GSM8K 94/100 for both heads, MMLU 197/228 for both; stock head −6.2%)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-dense-gemv/results/2026-09-25-kernels/profile/timeline-r3.txt §3 (DSv4.1 lm_head mxfp8 341 MB at 227.8 GB/s = 91% of peak, 1.5 ms per read, 2 reads per step)
- v11 flashinfer gemm/gemm_mm_mxfp8_cute_dsl.py:33-133 (b12x MXFP8 dense GEMM backend for SM12x is present in GLM's image)
- DSv4.1 tools/quantize_lmhead_mxfp8.py (builds the one-shard derivative in about 10 min)

**Proposed action:** Produce a GLM derivative snapshot with lm_head.weight in MXFP8 (block 32, UE8M0) or FP8 per-channel, and load it through a small patch that routes ParallelLMHead to FlashInfer mm_mxfp8 (backend b12x) for both the target and draft calls. Self-disarm on a stock pack, as DSv4.1's lever does. Leave embed_tokens BF16: it is a gather, not a stream.

**Est. impact:** BF16 2 × 0.634 GB at about 228 GB/s = 5.6 ms. MXFP8 2 × 0.327 GB = 2.9 ms. Saving about 2.7 ms of 114.6 ms/step = −2.4% step, +2.4% tok/s (prose 21.2→about 21.7). At k=5 (105 ms), +2.6%.

**Validation:** Offline numerics: top-1 agreement and logit max-abs error on sampled hidden states (CPU/torch in image, no GPU needed for a check on a small slice). Serve: ABAB on ms/step at matched acceptance. Draft acceptance must not drop, since the candidate top-16 is computed through the same head. Then quality_eval --full against the BF16 baseline (NLL within +0.01, GSM8K and tool rates within Wilson bounds).

**Risks:** Draft candidate ranking sensitivity: DFlash2's selector_top_k=16 relies on head logits, so an acceptance drop could cancel the gain (the DSv4.1 draft had no such selector). It needs a checkpoint derivative (a new pin), and AGENTS.md pins 09b04e5, so it would be published as a separate revision like DSv4.1's 2.0bpw-mcg-lmhead-mxfp8.

**Verifier reasoning:** Header check: lm_head BF16 is 1.182 GiB (my own tally over the 33 shards). dflash/utils.py:65-72 shares the target lm_head. dflash2/speculator.py:199 calls compute_candidates once per step, which runs get_top_k_tokens -> _apply_head over the full vocab shard (qwen3_dflash2.py:283-287, logits_processor.py:241-264). So there are two head reads per step, target verify plus draft candidates. Arithmetic: 2*634 MB vs 2*327 MB, about 0.61 GB saved per rank per step, 2.7 ms at 228 GB/s. That is +2.4%, which I confirmed, and it sits inside current noise. DSv4.1's quality result applies to DSv4.1 (vocab 129k, different model). GLM's selector_top_k over head logits is an untested sensitivity, which the reviewer notes.

**Verifier corrected impact:** About -2.7 ms/step (+2.4%) if the BF16 head currently streams at about 228 GB/s. More if the BF16 GEMM at M=8-16 is less efficient. Not detectable without the XP-2 protocol.

## XP-8: BF16 attention projections are the largest non-expert stream (about 6 GB/rank/step); DSv4.1's exact MXFP8 dense path is the template

- kind=perf component=KDA + MLA projection weights (excluded from ModelOpt quant, BF16) impact=4 confidence=2 effort=L needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** The nvidia pack keeps self_attn in BF16: 11.17 GiB across 45 layers, about 5.6 GiB per rank streamed every verify step. That is roughly a quarter of the step. DSv4.1's dense projections run as MXFP8 on b12x small-M tiles at 76–84% of peak, and an exact FP8 re-encode of one projection (wo_a) was worth +7% prose.

**Mechanism:** Decode GEMMs at M ≤ 16 are pure weight streaming. Halving bytes (BF16→FP8 weight-only) halves time if the FP8 kernel reaches similar bandwidth efficiency (b12x MXFP8 did so on the same SoC).

**Evidence:**
- xpoll_bytes.py header tally: attn_other 11.17 GiB, shared_expert 2.02 GiB, lm_head 1.18 GiB, routed_experts 172.97 GiB (nvidia 09b04e5)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-dense-gemv/results/2026-09-25-kernels/profile/timeline-r3.txt §3 (wq_b 190.7 GB/s = 76.3%, wo_b 205.3 = 82.1%, wo_a 210.8 = 84.3%, on b12x MXFP8 at M=4)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-fusion-host/flags.md:193 (E11 fix_o_proj_woa_fp8 exact requant: prose 31.4→33.6, 43/43 layers engaged)
- hf_quant_config/task facts: self_attn, shared experts, router, lm_head excluded from NVFP4 (BF16)

**Proposed action:** This is a quality-gated experiment, not a default. Build a derivative with KDA q/k/v/o and MLA q_b/kv_b/o projections in MXFP8 (block 32) using round-to-nearest (weight-only, no calibration). Run it through FlashInfer mm_mxfp8(backend=b12x). Measure per-layer ms in a single-GPU microbench first, then serve ABAB and quality_eval --full (NLL, GSM8K, tool calls, needle 32k/128k). Keep the head-gate weights_proj in fp32 as the model code requires (attention.py:321-329).

**Est. impact:** About 6.0 GB/rank at about 205 GB/s = 29 ms. MXFP8 about 3.1 GB = 15 ms. Saving about 14 ms of 114.6 = −12% step, +14% tok/s if bandwidth efficiency holds. Shared experts (2.02 GiB, about 1.08 GB/rank) would add −2.5 ms more.

**Validation:** Microbench per shape (M ∈ {8,16}), BF16 cutlass vs b12x MXFP8. The serve ABAB must show ms/step down beyond noise at matched acceptance. The quality gate must pass. For long context, a 318k needle must still hit, since KDA state accuracy compounds over length.

**Risks:** NVIDIA deliberately excluded attention from quantization, and KDA recurrent-state error accumulates over long contexts, so quality risk is real. The head-gate note in attention.py shows the sensitivity. This contradicts the checkpoint's own quant recipe, which is a revisit justified only if quality_eval passes. It needs a derivative pack plus a loader patch.

**Verifier reasoning:** Header tally: attention tensors 11.32 GiB BF16, dominated by KDA o/q/k/v (3.5/2.125/2.125/2.125 GiB), so about 6.08 GB per rank. The arithmetic of about 29 ms at 205 GB/s holds. The DSv4.1 precedent is misapplied. E11 'fix_o_proj_woa_fp8 (exact requant)' (k3-fusion-host flags.md:190-195) fixed an MXFP8 emulation dequant on an already-FP8 checkpoint weight ('scales retained -> exact roundtrip'). It is not a lossy BF16->FP8 re-encode, so its +7% says nothing about quality or speed for GLM's BF16 attention. NVIDIA deliberately excluded self_attn, and attention.py:321-329 documents bf16-vs-fp32 head-gate sensitivity, so the quality risk is real.

**Verifier corrected claim:** A lossy BF16->MXFP8 re-encode of KDA/MLA projections could save about 14 ms/step if b12x MXFP8 reaches about 205 GB/s. There is no precedent for this on either stack: DSv4.1's E11 was an exact requant of native FP8 weights.

**Verifier corrected impact:** Up to -12% step time if bandwidth holds and quality passes. The quality risk is higher than stated.

## XP-9: Port DSv4.1's post-ready warmup: GLM compiles the same lazy TileLang mHC kernels and pays Triton JIT on the first wave

- kind=ops component=run.sh post-ready (WARMUP) impact=2 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** GLM's boot logs show mhc_pre_big_fuse_with_norm_tilelang and mhc_post_tilelang compiling at boot, and README.md says the first wave pays Triton JIT per batch shape. In DSv4.1 the fused mHC prenorm compiles once per n_splits bucket (16 at ≤576 tokens, 4 at 577–1536, 1 above), about 6 s per rank, on the first user request of a new size. DSv4.1's warmup (nonce greedy, t=0.7, 300/1k/3k prefills, a small image) left 0 TileLang compiles after ready.

**Mechanism:** JIT and swap fault-in land on the first requests after boot, inflating TTFT and contaminating the first bench waves. That is part of the 22% prose spread in XP-2. Nonce prompts avoid prefix-cache hits.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-comm/tools/warmup.py:1-22
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-dense-gemv/results/2026-09-25-kernels/profile/timeline-r3.txt §2 ('TileLang compiles after ready: 0 on both ranks'; 3 Triton kernels remain)
- /home/sfxnz/projects/ai-lab/local-ai-lab/internal/glm53-nvidia-spark-proof.md timeline 00:46:45–51Z (GLM TileLang mHC compiles)
- GLM README.md:22 (first wave pays Triton JIT; first structured c=2 wave about 52 while 1.3 GiB swapped)

**Proposed action:** Copy tools/warmup.py and adapt it: CHAT kwargs {enable_thinking:false}, GLM tool/vision request shapes, and a 2-concurrent short request that exercises the c=2 capture. Call it from run.sh after wait_ready (WARMUP=1 default) on the head only.

**Est. impact:** First-user TTFT −6–12 s per new batch-size bucket (DSv4.1 measured about 6 s per compile per rank). Steady-state decode is unchanged. Bench variance should narrow, but by an unquantified amount.

**Validation:** After ready plus warmup, run smoke_vision and a 4k prompt, and grep both ranks' logs for 'TileLang begins to compile' and Triton compile lines: the target is 0. Compare the run-1 vs run-3 spread in bench_decode.

**Risks:** About 30–60 s longer to 'ready'. Warmup requests must not fill the prefix cache with user-visible content (nonce-seeded).

**Verifier reasoning:** There is stronger GLM-native evidence than the reviewer cited. On the LibertAI boot, rebench-20260902T204243Z/engine-rank1.log.tail shows jit_monitor warnings after ready: Triton _compute_local_logits_stats_kernel, _rejection_kernel and _resample_kernel (21:03:54), _prepare_dflash_inputs_kernel (21:06:13), and 'TileLang JIT compilation during inference: mhc_pre_big_fuse_with_norm_tilelang' (21:06:21). TileLang mHC kernels compile repeatedly per shape bucket during warmup (20:56-20:59). README.md:22 notes that the first wave pays JIT.

**Verifier corrected impact:** Removes several seconds of first-request stall per new shape and a source of first-run bench noise. Steady-state decode is unchanged.

## XP-10: Measure routed-expert overlap with DSv4.1's census before sizing MoE levers; GLM's step time implies strongly correlated routing

- kind=methodology component=MoE routing analysis (--enable-return-routed-experts + tools/moe_census.py) impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The weight-bytes hypothesis for the 115 ms step only closes if consecutive verify tokens share many experts. Independent routing at m=8 predicts about 28 GB/rank/step, which would need 98% of the 250 GB/s reference bandwidth. DSv4.1 measured 0.299 duplicate fraction at m=4 against 0.023 for random routing, so real routing is about 13× more correlated than independent. GLM's v11 already has the capture plumbing.

**Mechanism:** At a realistic 78% efficiency (195 GB/s), 114.6 ms corresponds to about 22.3 GB/rank. That implies routed experts of about 12.6 GB, about 40 distinct per layer at m=8 (duplicate fraction about 0.37). The marginal cost of each extra draft slot, and so the k-optimum (XP-1) and the ceiling of any MoE-kernel gain (XP-3/6), depend on this number.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-comm/results/2026-09-24-review/campaign/s10-diag-census-skew/moe_census.json (dup_mean_over_layers 0.2987, random_baseline_dup 0.0232, m=4, top_k 6; L.A.I.L 0.280, code 0.308)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-comm/tools/moe_census.py:1-50 (sliding windows of m accepted tokens from vLLM's own capture)
- v11 vllm model_executor/layers/fused_moe/routed_experts_capturer.py, engine/arg_utils.py (enable_return_routed_experts present)
- GLM byte model (xpoll_bytes.py + arithmetic): experts 18.3 GB (independent, m=8) + attention about 6.0 + shared 1.08 + lm_head ×2 1.27 + draft about 1.2 + dense layers 0–2 0.13 ≈ 28 GB/rank → 112 ms at 250 GB/s vs 114.6 measured

**Proposed action:** One diagnostic boot with EXTRA_ARGS=--enable-return-routed-experts (it does not count as a perf arm). Run tools/moe_census.py adapted to N_EXPERTS=288, TOP_K=8, m ∈ {4,6,8}, over the prose and code prompts. Pair it with one torch-profiler window (--profiler-config, DSv4.1 tools/profile_window.sh + decode_timeline.py) to get the real per-kernel split: Marlin MoE, BF16 attention GEMMs, KDA kernels, NCCL, eager gaps.

**Est. impact:** No direct tok/s. It turns XP-1/3/6/7/8 from estimates into a ranked, measured budget. DSv4.1's equivalent (timeline-r3) is what ranked its last 5 levers.

**Validation:** Census outputs duplicate fraction per layer. Check that the implied per-step expert bytes (distinct × 7.5 MB × 42) plus the non-expert bytes reproduce the measured ms/step within 15% at the profiled bandwidth.

**Risks:** The capture buffer costs host RAM (DSv4.1: 3.82 GB CPU buffer), so run it only on a diagnostic boot with MAX_NUM_BATCHED_TOKENS small. Capture only covers accepted tokens, so sliding windows approximate verify rows. The profiler adds overhead, so its absolute numbers are not published.

**Verifier reasoning:** The census file exists: s10-diag-census-skew/moe_census.json has dup_mean_over_layers 0.2987, random_baseline_dup 0.0232, m=4, top_k 6. The GLM header per expert is 14,155,800 B (gate/up [2048,2048] U8 + [2048,256] E4M3 scales each, down [4096,1024] + [4096,128]), so 7.08 MB per rank, not 7.5. The 172.97 GiB routed total includes MTP layer 45 (43 layers with experts). Independent m=8 gives 58.2 distinct experts -> 17.3 GB, plus about 9.8 GB non-expert = about 27 GB. At 273 GB/s peak that is 99 ms, 86% of 114.6 ms, before NCCL (about 90 ARs), KDA and launch gaps. The conclusion that routing must be correlated, or the model is otherwise wrong, holds. The diagnostic is the right step.

**Verifier corrected impact:** No direct tok/s. Use 7.08 MB per expert per rank in the budget.

## XP-11: Comm and host-latency levers from DSv4.1 k3/comm: PM QoS (about −1.1 ms/step) and a NCCL eager-twin with graph mixing off (modeled about −3 ms/step); skip custom AR and dual-rail

- kind=perf component=NCCL / CUDA-graph launch / host cpuidle impact=2 confidence=2 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** GB10 deep cpuidle states (LPI-1/2/3 exit latency 42/231/433 µs, confirmed on this host) delay CUDA-graph host nodes after host idle. DSv4.1 books 1.13 ms/step to that wait, and a 20 µs PM QoS request cut the wake-up wait from 481–586 µs to 3.4–14.7 µs. NCCL graph mixing adds about 3 ms/step of neighbour slowdown across about 92 collectives per step. GLM has the same collective structure: 2 ARs per layer around mHC, ×45 layers, plus draft ARs.

**Mechanism:** Graph replay issues NCCL proxy host nodes, and a CPU in LPI-3 takes about 0.4–0.6 ms to wake. With mixing on, each captured collective adds serialEvent waits and event-record nodes, and it slows the adjacent bandwidth-bound kernel by 13.5% (DSv4.1 microbench).

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-comm/flags.md:133-135 (NCCL_GRAPH_MIXING_SUPPORT=0 modeled −3.08 ms/step c=1, unsafe as a plain env; DSV41_NCCL_EAGER_TWIN implemented; DSV41_PM_QOS_US=20 measurements)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-comm/results/2026-09-25-kernels/comm/iterations.txt #0 (88 AR + 4 AG per step, graph startup 1.13 ms/step), #3 (custom AR / symm-mem / FlashInfer AR not applicable across 2 nodes on GB10)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-comm/results/2026-09-25-kernels/comm/per-size/decode_collectives.txt (AR m=8 KEEP 63.2 µs gapped vs 29.1 µs mixing off; RoCE floor 20.7 µs)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-fusion-host/flags.md:139 (dual-rail rejected: small-message latency +9.5–30%), :141 (NCCL knob sweep: no knob beats KEEP)
- Host: /sys/devices/system/cpu/cpu0/cpuidle state latencies 0/42/231/433 µs (LPI-0..3)

**Proposed action:** (1) Port docker/patch/pm_qos.py: hold /dev/cpu_dma_latency=20 µs while serving, pass --device /dev/cpu_dma_latency, and gate it behind GLM_PM_QOS_US. (2) After XP-2 exists, port nccl_eager_twin.py (285 lines, PyNccl router) and run DSv4.1's tools/nccl_twin_check.sh on both GPUs, then serve ABAB. Do not pursue custom all-reduce, symm-mem, dual-rail or NCCL_ALGO/PROTO sweeps: all were measured null or negative on this fabric.

**Est. impact:** PM QoS: about −1.1 ms of 114.6 = +1.0% tok/s. Eager twin: up to −3 ms = +2.7% (modeled in DSv4.1, not yet measured in any serve). Together +3–4%, within XP-2's detectable range but not single-boot.

**Validation:** PM QoS: DSv4.1 host_node_latency.py on one GPU (graph host-node wait after 20 ms idle), then serve ABAB. Eager twin: nccl_twin_check.sh two-node bitwise check, then ABAB ms/step at c=1 and c=2.

**Risks:** PM QoS is host-wide: it raises idle power and heat on the Spark shared with the busy workload, so it needs user consent. The eager twin is a second communicator, which costs a little more NCCL buffer memory (offset by XP-4) and adds patch-anchor fragility to vLLM internals. DSv4.1 has not yet serve-validated either lever.

**Verifier reasoning:** Host cpuidle latencies 0/42/231/433 us LPI-0..3 are confirmed on spark1. k3-comm flags.md:133-135 confirms that NCCL_GRAPH_MIXING_SUPPORT=0 is modeled at -3.08 ms/step and is 'unsafe as a plain env var'. The eager twin and PM QoS are 'implemented, default off; serve ABAB pending'. None of this has been serve-validated even on DSv4.1, and GLM's collective count and graph mode (breakable, FULL_AND_PIECEWISE) differ.

**Verifier corrected impact:** +1-4% modeled and unvalidated on any serve.

## XP-12: Reuse DSv4.1's quality_eval.py (baseline-gated NLL, decode-vs-prefill, tools, needle, self-consistency, vision) plus tool-eval-bench for the nvidia-vs-LibertAI and requant decisions

- kind=quality component=quality gate (tests/) impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 4)

**Claim:** GLM's quality checks are pass/fail smokes: thinking-off, count, tool smoke and needle. They cannot tell whether the nvidia W4A4 pack under Marlin W4A16, an lm_head/attention re-encode, or a new MoE kernel shifts output quality. DSv4.1 ships a stdlib HTTP-only gate with statistical thresholds that ports by changing one kwargs constant. GLM's v11 V2 runner supports prompt_logprobs.

**Mechanism:** Teacher-forced NLL on fixed passages plus a decode-vs-prefill logprob probe detect small numeric drift from kernel or quant changes that greedy smokes miss. A/A flip hazard calibrates run-to-run nondeterminism.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-comm/tests/quality_eval.py:1-40, :64-65 (CHAT_KWARGS {thinking:false, reasoning_effort:low}), :673-690 (--quick / --full / --baseline / --result offline re-gate); tests/quality/{nll_passages.jsonl, tools30.json, gsm8k_100.jsonl, mmlu_228.jsonl}
- DSv4.1 README.md:103-132 (gates: NLL ≤ baseline + max(0.01, 3× repeat noise); decode-probe median |Δlogprob| ≤ +0.05; Wilson bounds on rates; flip hazard ≤ 2× A/A)
- v11 vllm/v1/worker/gpu/model_runner.py:412,1852-1869 (PromptLogprobsWorker in the V2 runner)
- /home/sfxnz/projects/ai-lab/tool-eval-bench/README.md (69 tool-call scenarios in 15 categories plus GSM8K/MMLU/IFEval plugins, infrastructure failures excluded from scoring)
- DSv4.1 flags.md:122 (MHC split-K 40 was a 7 µs/call kernel win rejected by the self-consistency flip-hazard gate), which shows the gate catches non-bit-exact kernel changes

**Proposed action:** Copy quality_eval.py and the tests/quality data. Set CHAT_KWARGS={'enable_thinking': False} and THINK_KWARGS={'enable_thinking': True}, and point the tools30 items at glm47 parser expectations. Record a baseline on the LibertAI caca4e6 pin and on nvidia 09b04e5 (both Marlin). Run tool-eval-bench --short plus IFEval as a secondary agentic score. Require --quick to pass before any keep (DSv4.1 ARMS.md step 6).

**Est. impact:** No direct tok/s. It gives the first quantitative answer to whether the nvidia pack improves quality over LibertAI (the stated goal of PR #11), and it lets XP-3/5/6/7/8 ship with evidence rather than smokes.

**Validation:** Run twice on the same boot to establish A/A noise, then across the two packs. It passes when the gates behave as in DSv4.1 (the A/A flip hazard stays within its own limit).

**Risks:** About 6–9 min per --quick on a slow serve, maybe longer on GLM at about 21 tok/s. prompt_logprobs with spec-decode and vision on must be smoke-tested once. Needle 128k in --full needs the XP-4 memory headroom.

**Verifier reasoning:** The tests/quality data files exist, and PromptLogprobsWorker is in v11 model_runner.py:127,412. 'Ports by changing one kwargs constant' is overstated. quality_eval.py:62 hardcodes the DeepSeek BOS_TEXT '<｜begin▁of▁sentence｜>' used for teacher-forced NLL prompts, and the tools30 expectations are for the deepseek_v41 parser. GLM needs its own prompt prefix ([gMASK]<sop>), glm47 tool expectations and fresh baselines. The value of the gate is sound.

**Verifier corrected claim:** DSv4.1's quality_eval is a good template, but porting needs GLM-specific prompt/BOS construction, tool-parser expectations and per-pack baselines, not just CHAT_KWARGS.

## XP-13: Ops hardening from DSv4.1: one FORWARD_ENVS list (GLM's worker ssh line drops LIMIT_MM_PER_PROMPT), an engagement audit, and post-ready floors replacing PR #12's 16 GiB wait-abort

- kind=ops component=run.sh orchestration / PR #12 guards impact=3 confidence=4 effort=S needs_gpu=False
- **verdict: plausible** (corrected confidence 4)

**Claim:** GLM hand-maintains a long env string for the worker (run.sh:386) that omits LIMIT_MM_PER_PROMPT. The novideo proof had to start ranks manually to change it, and any head-only override silently desyncs the ranks. PR #12's 16 GiB MemAvailable abort while waiting cannot pass: a healthy LibertAI boot and every nvidia boot sit at 16.1–18.8 GiB during construct and fill. DSv4.1 uses a single FORWARD_ENVS array enforced by a unit test, a strict log audit of LOG_ENGAGED/LOG_DISARMED markers, and boot floors measured after ready (≥12 GiB) and after smoke (≥8 GiB).

**Mechanism:** Orchestration drift and over-tight watchers produce false negatives: the nvidia pack is still 'unmeasured' after 7 attempts. Levers can also silently disarm on one rank.

**Evidence:**
- GLM run.sh:386 (worker env string: no LIMIT_MM_PER_PROMPT, CHAT_TEMPLATE or NCCL tuning)
- /home/sfxnz/projects/ai-lab/local-ai-lab/internal/glm53-nvidia-novideo-proof.md ('Stock run.sh auto-SSH does not forward LIMIT_MM_PER_PROMPT')
- /home/sfxnz/projects/ai-lab/local-ai-lab/internal/glm53-5shard-uma.md (healthy LibertAI boot 18–20 GiB after construct; PR11 16 GiB watcher fired at 15.78 during the normal sawtooth)
- gh pr view 12 (UMA_ABORT_GIB=16 while wait_ready)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-comm/AGENTS.md:23 and run.sh:74,144,405,646,682 (FORWARD_ENVS feeding docker -e and the worker ssh line; tests/test_recipe_ops.py enforces it); ARMS.md:21-24 (floors), tools/engagement_audit.py, tools/disarm_scan.sh

**Proposed action:** Replace the worker env string with a FORWARD_ENVS array plus a test asserting that every env read by run.sh or a patch is forwarded. In PR #12, drop the load-time abort to about 4 GiB (a genuine OOM guard) or make it phase-aware (no abort during construct/fill), and add DSv4.1's post-ready floors. Add AUDIT=warn|strict log scanning for backend lines (MARLIN/B12X), the capture sizes and the spec method on both ranks.

**Est. impact:** No tok/s. It unblocks the first nvidia-pack decode measurement (PR #11's stated goal) and prevents rank-asymmetric configs from polluting A/B results.

**Validation:** tests/run_sh_harness.py-style dry run: stub docker/ssh and diff the head and worker docker-run argv and env. Unit test for FORWARD_ENVS coverage. render --check.

**Risks:** A looser load-time watcher raises the chance of a real UMA exhaustion during boot. Pair it with XP-4's headroom and dmesg/NVRM monitoring. It also touches PR #12, which is stacked on #11.

**Verifier reasoning:** Confirmed that the run.sh:386 worker env string omits LIMIT_MM_PER_PROMPT. The worker falls back to the run.sh:62-63 default, so only a head-side override desyncs. CHAT_TEMPLATE is head-only by design (run.sh:297-300), so omitting it is correct. PR #12 does set UMA_ABORT_GIB=16 while waiting. The proposal to add DSv4.1's post-ready floors (>=12 GiB at ready, >=8 after smoke) would abort GLM's published healthy config. evidence/rebench-20260902T204243Z/free-after-bench.txt shows 'available 5' GiB with 6 GiB swap used during a healthy LibertAI serve, and PR #12's own text says a healthy serve 'can sit near 8 GiB'. Any floors must come from GLM's own measurements.

**Verifier corrected claim:** The FORWARD_ENVS refactor is right (LIMIT_MM_PER_PROMPT and any head-only override desync the ranks). The load-time 16 GiB abort is too tight. DSv4.1's 12/8 GiB post-ready floors must not be copied: GLM's healthy steady state is about 5-8 GiB MemAvailable. Calibrate floors from GLM data, or use rank death / dmesg NVRM as the signal.

## XP-14: Re-test VLLM_USE_BREAKABLE_CUDAGRAPH=0 on prose ms/step: cross-stack evidence conflicts, and GLM measured only structured c=2 on a single boot

- kind=perf component=CUDA graph mode impact=2 confidence=2 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** Mia's DSV4-Flash 2x Spark lane measured regular (non-breakable) graphs at +28.6% c=1 decode and +13.1% c=2 aggregate. GLM rejected =0 on one structured c=2 number (59.7→52.1) with no prose c=1 cell. DSv4.1 needs breakable=1 for DSpark, so the effect is stack-specific and unresolved for DFlash2.

**Mechanism:** Breakable graphs split the step at attention or spec boundaries into several graph launches with eager glue in between. Each boundary adds launch and host latency. The cost is larger on a slow-waking GB10 host (XP-11).

**Evidence:**
- /home/sfxnz/lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/docs/GLM-NEW-REPORT.md:131-145 (breakable 74.6 vs regular 95.9 tok/s c=1; c=2 aggregate 134.2 vs 151.8)
- GLM run.sh:30-32 and README.md:20 ('VLLM_USE_BREAKABLE_CUDAGRAPH=0 slowed structured c=2 59.7 → 52.1')
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-fusion-host/flags.md:43 (DSv4.1: breakable required for graphs+DSpark)
- XP-2: GLM c=2 structured varies 60.4→55.5 between runs of the same config (PR #8 parity)

**Proposed action:** One ABAB arm, =0 vs auto, on prose_long c=1 ms/step with structured c=2 non-inferiority. Also confirm from logs that FULL graphs capture the DFlash2 draft (DSv4.1 found its census blind to FULL-graph replays until patched).

**Est. impact:** Between −12% (GLM's structured c=2 reading) and +28% (Mia's DSV4 lane) on c=1. Low prior for a large win on GLM because its step is weight-bound (about 115 ms vs about 10–15 ms in Mia's lane), so launch overhead is a smaller share. A 0–3% effect is likely.

**Validation:** XP-2 ABAB. Check the engine logs for the captured graph modes and count graph launches per step with the profiler.

**Risks:** =0 may fail capture with DFlash2 KV groups (v11 patch) or corrupt spec state. Keep the count probe (200 consecutive) as a hard gate.

**Verifier reasoning:** GLM-NEW-REPORT.md:131-145 (a report on DeepSeek-V4-Flash-0731, despite its name) shows breakable 74.6 vs regular 95.9 c=1. GLM's README.md:20 records only structured c=2 59.7->52.1. The DSV4 lane runs async-scheduling, 6 seqs and MTP-5 on a different model. The low prior stated for GLM is reasonable.

## XP-15: Head rank loads 3.8× slower than the worker (733 s vs 191 s): boot time is the throughput limit on ABAB experiments

- kind=ops component=weight loading / boot time impact=2 confidence=3 effort=S needs_gpu=False
- **verdict: plausible** (corrected confidence 3)

**Claim:** Every nvidia-pack attempt shows head 'Loading weights took' 716–733 s against 191–194 s on the worker. With XP-2 requiring about 4 boots per decision, this adds about 36 min per decision. DSv4.1's stream-feed lever cut load 6–21%, and Ornith uses --load-format fastsafetensors on GB10.

**Mechanism:** The spark1 snapshot sits on a slower or colder filesystem path, or fragmented blobs. The same 33 shards read about 3.8× slower on the head than on spark2.

**Evidence:**
- /home/sfxnz/projects/ai-lab/local-ai-lab/internal/glm53-nvidia-spark-proof.md ('Head I/O 733 s (cold EXT4, many small tensors). Worker 191 s')
- glm53-nvidia-langonly-proof.md (716.07 s head vs 194.17 s worker)
- /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-fusion-host/flags.md:56 (DSV41_STREAM_FEED: TP0 334→313 s, TP1 191→151 s)
- /home/sfxnz/projects/ai-lab/recipes/Ornith-1.5-35B-A3B-NVFP4-DGX-Spark/run.sh (--load-format fastsafetensors)
- GLM AGENTS.md: 'Do not turn on InstantTensor. That loader killed TP=2 ranks here.'

**Proposed action:** Diagnose first (read-only): compare the spark1 and spark2 filesystem, device and fragmentation of the blob dir (filefrag, fio read bench when idle). Options: relocate the head's HF cache to the NVMe used by spark2's layout, pre-read shards in parallel before start (but see the XP-4 page-cache caveat), or trial fastsafetensors only on a diagnostic boot. Keep InstantTensor off.

**Est. impact:** Head load from 12 min toward about 3–4 min, saving about 8–9 min per boot and about 35 min per ABAB decision. No decode change.

**Validation:** A fio/dd sequential read of one shard on each node with the serve down (host-only, no GPU), then 'Loading weights took' on the next boot.

**Risks:** Pre-reading fills the page cache, which on UMA competes with construct: drop it via XP-4's fadvise after load. fastsafetensors behaviour under TP=2 on GB10 has not been tested for GLM (the InstantTensor precedent).

**Verifier reasoning:** Head vs worker load times are confirmed across the proofs (697-733 s vs 186-195 s). The LibertAI boot showed the same asymmetry (629 s vs 197 s, glm53-5shard-uma.md). DSv4.1 on the same hosts shows TP0 334 s vs TP1 191 s (k3-fusion-host flags.md:56), so the cause is host-level on spark1, not GLM-specific. Read-only check on spark1: the HF cache sits on the root nvme0n1p2 ext4, 3.7T with 3.1T used (89% full), which is consistent with fragmentation or cold-read slowness.

**Verifier corrected claim:** spark1 (head) loads 3.6-3.8x slower than spark2 on every stack, including DSv4.1. spark1's root ext4 NVMe is 89% full, the likely cause (fragmentation). Diagnose with filefrag on the blobs and a sequential read with the serve down.

**Verifier corrected impact:** About 8-9 min saved per boot if spark1 reaches worker-like I/O.

## Open questions
- What is Marlin's achieved bandwidth on GLM's MoE at M=8/16 on sm_121? Without a torch-profiler window (XP-10), XP-3/XP-6 gains could be anywhere from 0% to 20% of the step.
- Does the standalone b12x package (lukealonso/b12x) accept NVFP4 source weights (16-element FP8 block scales plus a global scale) with activation='silu', swiglu_limit and quant_mode='w4a16', or only MXFP4? Anemll's patch wires it only through the MXFP4 oracle.
- The 'Xid 31 reports on sm_121' for b12x in the Qwen run.sh guard have no local evidence file. Where did that claim come from, and does it apply to the b12x W4A16 MoE or only to other b12x kernels?
- Does flashinfer-jit-cache 0.6.18.post1 (the v0.30.0 image) contain sm_121a or 12.0f builds of the SM120 CUTLASS fused-MoE module? If yes, a matched python + jit-cache pair removes the runtime JIT entirely.
- With DFlash2, does the draft forward always process block_size=8 positions regardless of num_speculative_tokens? That decides whether k<7 also shrinks draft compute or only verify experts.
- Does lowering the lm_head precision (XP-7) degrade DFlash2 draft acceptance through its candidate_selector top-16, where DSv4.1 had no such selector?
- Are the nvidia-pack post-load troughs (0.45–10.4 GiB) real OOM risks or transient? The healthy LibertAI boot was never logged at 1 s resolution through profile and capture. A LibertAI-pin boot with the same logger would separate a pack-specific regression from the watcher threshold.
- Would a smaller max-model-len for a decode-focused lane change step time? Qwen3.8 saw +3.59 prose c=1 going from 1M to 262144 (a single boot, possibly noise), and the GLM indexer workspace and logits buffers scale with max_model_len.
- Could SGLang's GLM-5.3 path (lmsysorg/sglang:glm-5.3-flash-arm64, local) load the official nvidia pack? The only local attempt failed on an older LibertAI snapshot's qkv shape (/home/sfxnz/lab/glm53-ray/sgl-rank0.log).

## Verifier: missed issues
- PR #9 k=5 was hand-applied and rebenched after the lint failure: commit 'perf(recipe): H1-20260903 NUM_SPECULATIVE_TOKENS=5 (hand-applied)', recipe.yaml:49 on agent/hillclimb-20260903, and evidence/rebench-dflash5-20260903T045815Z/summary.json. It measured structured c=1 55.7 vs 67.6 (-17.6%) and structured c=2 41.6 vs 60.4/stream (-31%). The rebench gate failed only on a needle-8192 refusal. PR #9 is still open. Any k proposal must weigh this measured structured regression.
- FlashInfer 0.6.18 in the v11 image already contains a clamp-capable b12x W4A16 NVFP4 MoE: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py:4280-4294 clamps gate and up for silu when swiglu_limit is set, and b12x_moe.py:270-285 accepts quant_mode='w4a16', source_format='modelopt' and swiglu_limit. vLLM's wrapper (model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:242-251) passes neither, so the b12x worktree's 'do not retry on this image' is wrong. A small vLLM patch is enough; no new package is needed.
- The nvidia-pack 'boot failures' were all 8 GiB watcher kills with no kernel OOM (glm53-nvidia-spark-proof.md:92,137). The recorded values are the first sample below 8 GiB, not troughs. The nvidia per-rank working set, 90.36 GiB, is smaller than LibertAI's 90.67. The published LibertAI boot on v11 ran the same max-size video encoder profile and TileLang compiles (rebench-20260902T204243Z/engine-rank1.log.tail:47-52) and then served at only 5 GiB MemAvailable with 6 GiB swap used (free-after-bench.txt). An 8 GiB watcher would kill the published config too. The right next step is an unwatched (dmesg/NVRM-monitored) boot, not headroom levers.
- The 'Your GPU does not have native support for FP4' warning is emitted unconditionally by prepare_nvfp4_moe_layer_for_marlin (marlin_utils_fp4.py:352-357) whenever Marlin is used. It is not evidence that _C lacks sm_121 FP4 kernels. --moe-backend cutlass (VLLM_CUTLASS, oracle/nvfp4.py:147-148) can be tried on the existing image before any rebuild.
- The fadvise page-cache drop gives almost nothing for GLM's MemAvailable. Page cache already counts as available, and post-stop buff/cache was only 0.6-0.8 GiB (glm53-nvidia-spark-proof.md:93). DSv4.1's reported win was MemFree.
- Per-expert per-rank bytes are 7.08 MB (header: 14,155,800 B per expert at layer 10), not 7.5 MB. The 172.97 GiB routed total spans 43 layers including MTP layer 45, so byte budgets based on 172.97/43 overstate per-layer expert bytes by about 6%.
- The spark1 head's slow load recurs on DSv4.1 too (TP0 334 s vs TP1 191 s, k3-fusion-host flags.md:56), and spark1's root ext4 NVMe is 89% full (df: 3.1T/3.7T). This points to a host-level I/O cause, not the recipe.
