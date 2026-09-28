# Dimension: moe-kernels

## Reviewer summary

Scope: which MoE and FP4 GEMM kernels run on sm_121 in glm53-sm121-v11 for the nvidia pack, and what a better kernel would buy. Paths below use v11src = /tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-GLM-5-3-Flash-NVFP4-vLLM-2x-DGX-Spark/29f0f0df-0b0e-4224-a895-0ea5d735b7de/scratchpad/v11src, and repo = the opt-nvidia worktree. Nothing here was run on a GPU. Every speed or quality number is an estimate unless it is tagged as measured.

What runs today, traced in code: the router GEMM falls to F.linear in BF16 and is then cast to fp32, because the faster router tiers are enabled only for SM90 and SM100. Routing itself is one fused topk_sigmoid kernel that applies the bias, renormalizes and folds in the routed scale of 2.5. The routed experts go through five Marlin launches per layer: align, W4A16 GEMM1, silu_and_mul_with_clamp, GEMM2, and moe_sum. Weights are FP4 dequantized to BF16 with BF16 MMA, and block_size_m is 8 at decode. The shared expert is a BF16 MLP on an aux CUDA stream that overlaps the routed path. There is one TP all-reduce per MoE layer. The dense layers 0-2 are auto-selected to the FlashInfer CUTLASS W4A4 GEMM, which is JIT-compiled at boot.

From the safetensors headers, each rank streams 7.08 MB per active expert per layer. BF16 weights outside the MoE add about 8 GB per rank per step. Under an independence model, an 8-token DFlash2-7 verify step touches up to 58 distinct experts per layer, which is 17.3 GB per rank. Checked against the measured ~115-120 ms step and the measured k=5 delta (-14.5 ms), expert streaming is roughly 40-60% of decode time.

Conclusions:
- Native W4A4 moves the same weight bytes as Marlin. Measured on GB10 elsewhere, W4A16 decodes about 6% faster than W4A4 at batch 1. So W4A4 is a prefill lever, not a decode lever, and W4A16 is numerically the higher-fidelity path. Marlin ignoring input_scale costs no quality.
- The best decode kernel candidate is already in the image: FlashInfer's SM12x W4A16 fused b12x path. Its kernel has the silu clamp, but the dispatch code drops swiglu_limit, and vLLM's wrapper hard-codes W4A4.
- The vLLM b12x W4A4 wrapper has two numerics bugs besides the missing clamp. It bakes weight_scale_2 (about 4e-5) into the E4M3 block scales, which puts about 96% of experts' scales in the subnormal range. It also forces the FC2 input global scale to 1.0.
- The flashinfer_cutlass OOM came from a runtime JIT: ninja defaults to about 22 parallel nvcc jobs on 20 cores, there is no persistent cache, and the jit-cache wheel was uninstalled in v8. Two fixes: an ahead-of-time build in the image, or vLLM's own ahead-of-time CUTLASS path (--moe-backend cutlass). The latter is plausibly already built as 12.0f cubins under CUDA 13.0.1, which run on sm_121; that is not yet confirmed.
- Quality fix, cheap: router logits are BF16-rounded on sm_121 even though the config asks for fp32.
- Largest measured decode lever that MoE bytes explain: verify width. k=5 measured +14% prose c=1 (PR #9).

Ranked plan: (1) profile and microbench to pin Marlin's GB/s and the real distinct-expert count; (2) fp32 router on SM12x; (3) retest verify width k=5 on the nvidia pack; (4) the b12x W4A16 fused path; (5) fix the JIT or use the cutlass backend so prefill W4A4 can be tried with a larger chunk; (6) dense layers to Marlin W4A16 for quality; (7) FP8 shared experts.

## MOE-1: Verify-batch expert streaming dominates the decode step; DFlash2-7 pays ~15 ms/step of MoE bytes for drafts prose rarely accepts

- kind=perf component=MoE weight streaming x speculative verify width (Marlin W4A16) impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Each verify step streams every distinct routed expert touched by the (k+1)*c verify tokens. At k=7, c=1 that is up to 58 experts per layer, about 17.3 GB per rank per step, against about 8 GB for all BF16 weights outside the MoE. Prose acceptance is 2.3-2.4 tokens per step, so most of the k=7 expert traffic serves drafts that get rejected. Cutting to k=5 was measured at +14% prose c=1 on this stack.

**Mechanism:** Decode is weight-bandwidth-bound on 273 GB/s LPDDR5X. MoE bytes per step scale with the number of distinct experts across the verify block, not with accepted tokens. Rejected draft positions still pull their experts' weights. Cutting k from 7 to 5 removes about 12.6 experts per layer, about 3.75 GB per rank per step.

**Evidence:**
- Safetensors headers (nvidia 09b04e5): experts.N.gate/up_proj.weight U8 [2048,2048], down_proj U8 [4096,1024], F8_E4M3 scales [*,256]/[4096,128] -> per-rank (TP=2) per-expert bytes = 4,194,304+524,288+2,097,152+262,144 = 7,077,888 B
- Header totals: routed_experts 159.47 GiB, attn_kda 8.73, attn_mla 2.56, shared_experts 1.97, lm_head 1.18, router 0.09, dense_mlp 0.24 GiB -> non-MoE per-rank ~8.0 GB/step
- E[distinct] = 288*(1-(1-8/288)^n): n=1 -> 8, n=6 -> 45.5, n=8 -> 58.1, n=16 -> 104.5 (independence upper bound)
- n=8: 58.1*42*7.078 MB = 17.3 GB/rank -> 63 ms @273 GB/s, 77 ms @225 GB/s (225 GB/s ~ achieved W4A16 GEMV on GB10 per ai-muninn.com 8B W4A16 40.85 tok/s)
- Measured step times: rebench 21.2/2.43 = 114.6 ms; b12x-A marlin wave1 18.98/2.28 = 120 ms (opt-b12x/evidence/iter-b12x-A-marlin/bench.txt.summary.json); PR#9 H1 k=5: 21.97/2.308 = 105.1 ms vs same-night k=7 19.23 (~119.6 ms) (git show origin/agent/hillclimb-20260903:evidence/iter-H1-20260903-20260903T010419Z/bench.json)
- Independence model predicts k7->k5 MoE delta (58.1-45.5)*42*7.078 MB = 3.75 GB = 16.7 ms @225 + ~4 ms per-token non-MoE (structured c1->c2 identical prompts: 122.5 -> 139.9 ms for +8 tokens, ~2.2 ms/token); measured 14.5 ms => real expert overlap is somewhat higher than independent but MoE is ~40-60% of the step
- repo recipe.yaml:49 NUM_SPECULATIVE_TOKENS: 7; PR#9 decision.tsv H1-20260903-apply 'kept' only on unmerged branch

**Proposed action:** Retest NUM_SPECULATIVE_TOKENS=5 on its own on the nvidia pack at seqs=2, using the published prose ruler. This revisits the H3 decision to adopt DFlash2-7: that decision was made on structured gains (61.86 tok/s), but AGENTS.md says the published score is prose-only. Log structured separately, because it loses: its acceptance is capped at 6, so about 57 tok/s versus 67.6 now. Optionally test k=6 as well.

**Est. impact:** Prose c=1: +10-14%, measured +14% on the LibertAI pack (19.23 -> 21.97). Prose c=2 measured +16% (15.13 -> 17.52). Structured c=1 estimated -15% (6/0.105 s ~ 57 tok/s versus 67.6).

**Validation:** Boot the nvidia pack with NUM_SPECULATIVE_TOKENS=5 as the only change and run python3 bench_decode.py twice. Compare prose c=1 and c=2 against a k=7 run from the same boot day. Record acceptance_len and compute step_ms = 1000*acceptance_len/tok_s.

**Risks:** Structured and code workloads regress. The spec-decode dimension may prefer a different k or an adaptive k. The measurements are from the LibertAI pack, and the nvidia pack's acceptance may differ.

**Verifier reasoning:** Header bytes check out: I parsed all 33 shard headers. experts.N.{gate,up}_proj.weight is U8 [2048,2048], down_proj is U8 [4096,1024], and the F8 scales are [2048,256]/[4096,128]. That gives 14,155,776 B per expert, or 7,077,888 B per rank. Routed experts in layers 3-44 total 3.797 GiB/layer x 42 = 159.5 GiB. The non-MoE total of ~8 GB/rank also holds (attn 11.29 GiB + shared 1.97 + lm_head 1.18 + dense 0.24 + router), but it leaves out the DFlash2 drafter: model.safetensors is 2,342,169,800 B, about 1.2-2.3 GB/rank/step, plus the draft logits. E[distinct] at n=8 is 58.1, but at n=6 it is 44.8, not 45.5. The step times in the cited evidence reproduce: parity k=7 19.23/2.292 = 119.2 ms, H1 k=5 21.97/2.308 = 105.1 ms, b12x-A marlin 18.98/2.281 = 120.3 ms. However, the reviewer missed an independent k=5 rebench on the same branch, evidence/rebench-dflash5-20260903T045815Z/summary.json (LibertAI caca4e6, v11). It shows prose c=1 21.76 (acc 2.354, 108.2 ms), prose c=2 19.62/stream, structured c=1 55.73 (acc 6.0), and structured c=2 41.56/stream. Against the published k=7 rebench (21.2 at acc 2.43, 114.6 ms), k=5 prose c=1 is only +2.6% and the step saving is ~6.4 ms, not 14.5 ms. The +14% figure depends on a low same-night k=7 baseline (19.23). The independence model predicts ~17.6 ms MoE saving plus ~4 ms per-token cost at c=1, but the measured saving is 6-14 ms. So real expert overlap between adjacent positions is much higher than independent, and the 40-60% MoE share is not established. At c=2 prose the fit is better: k=7 step 156.8 ms (parity) versus k=5 131 ms, against a predicted ~29 ms from n=16 to n=12. Structured c=2 shows no step saving (140 ms at k=7 versus 144 ms at k=5). Direction is right: k=5 does not lose prose and gains clearly at c=2. The claimed magnitude at c=1 is overstated.

**Verifier corrected claim:** Moving DFlash2 from k=7 to k=5 cuts the verify block by 2 tokens per sequence. On the LibertAI pack this measured prose c=1 +2.6% to +14%, depending on which k=7 baseline is used (published 21.2 or same-night 19.23; k=5 was 21.76 and 21.97 across two runs). Prose c=2 per stream measured +16% to +30% (15.13 or 16.6 at k=7, 17.52 or 19.62 at k=5). Structured measured -18% at c=1 (55.7 vs 67.6) and -31% per stream at c=2 (41.6 vs 60.4). The share of step time spent streaming MoE bytes is not established. The c=1 data implies far fewer distinct experts per verify window than the independence model predicts.

**Verifier corrected impact:** Prose c=1 +3% to +14%, measured on the LibertAI pack. Prose c=2 +16% to +30%. Structured c=1 -18% and c=2 -31% (measured, not estimated). The effect on the nvidia pack is unmeasured.

## MOE-2: MoE share of step time is modeled, not measured: profile and microbench Marlin at the exact shapes, and measure the real distinct-expert count

- kind=methodology component=Marlin MoE (ops.moe_wna16_marlin_gemm) on sm_121 impact=3 confidence=5 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 5)

**Claim:** Every kernel decision in this plan depends on two unmeasured numbers. The first is how much of peak bandwidth Marlin achieves at E=288, N=2048/K=4096 (w13) and N=4096/K=1024 (w2), with M of 8-16 tokens, topk 8 and block_size_m 8. The second is the real distinct-expert count across an 8-token verify window. The bracket today is 40-60% of the step, and Marlin's efficiency is unknown.

**Mechanism:** If Marlin already reaches about 90% of attainable bandwidth (roughly 225 GB/s on GB10), a new kernel can only save launch overhead. If it reaches about 70%, a fused kernel saves 15-20% of MoE time. The correlation between routing of consecutive tokens determines how much verify width costs in bytes.

**Evidence:**
- v11src/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:333-346 block_size_m heuristic -> 8 for M*topk/E = 8*8/288 = 0.22 < 0.9*8
- marlin_moe.py:135-162, 163-185, 202-229, 391-397: 5 launches per layer (align, gemm1, clamp-act, gemm2, moe_sum); use_fp32_reduce=True, use_atomic_add=False hard-coded (159-160, 226-227)
- v11src/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py + config/model.py:242 enable_return_routed_experts exists to capture per-token expert ids
- No nsys/torch-profiler trace exists in repo evidence/ (ls evidence/)

**Proposed action:** When the GPUs are free: (a) run nsys over 20 decode steps with NVTX around moe_forward, and sum the kernel time for Marlin, activation, align and sum; (b) run a standalone microbench of fused_marlin_moe with random topk ids at M=1, 8 and 16 on the real layer shapes, and report bytes divided by time; (c) boot once with --enable-return-routed-experts, capture ids over the prose bench, and compute the distinct experts per layer across each verify window.

**Est. impact:** No direct gain. It decides whether items MOE-4 through MOE-6 are worth 0% or up to about 10% on decode.

**Validation:** The microbench's achieved GB/s, compared with a memcpy-style streaming baseline on GB10, gives Marlin's efficiency. The nsys MoE kernel time divided by step time gives the MoE share.

**Risks:** Profiler overhead. Must be done in an exclusive GPU window, because AGENTS.md forbids running this alongside the other serve.

**Verifier reasoning:** The code facts are verified. marlin_moe.py:333-335: with M=8, topk 8 and E=288, M*topk/E/8 = 0.028 < 0.9, so block_size_m = 8. There are 5 launches: align at :340, GEMM1 at :135-162 (use_atomic_add=False, use_fp32_reduce=True at :159-160), activation at :170-185, GEMM2 at :202-229, and moe_sum/torch.sum at :391-397. enable_return_routed_experts exists at config/model.py:242 and is wired in gpu_model_runner.py:7854+. The evidence/ directory holds no profiler trace. One caveat: the capturer is indexed by KV slot mapping (gpu_model_runner.py:2407-2415, 3836-3841), so it likely reports routing only for committed tokens, and slots of rejected draft positions get overwritten. Measuring distinct experts across the full verify window, rejected drafts included, may need a custom hook on topk_ids. The methodology is sound and is needed, because MOE-1's c=1 data already contradicts the independence model.

## MOE-3: Router logits are BF16-rounded on sm_121 even though the config asks for fp32 routing

- kind=quality component=GateLinear (router GEMM) tier selection impact=2 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** GateLinear enables the fp32-output tiers (ll_bf16, cuBLAS torch.mm with out_dtype=float32) only on SM90 and the SM100 family. On sm_121 it falls to tier 6: a BF16 F.linear followed by .to(float32). The 288 router logits are therefore rounded to 8 mantissa bits before sigmoid, bias correction and top-8. That breaks moe_router_dtype=float32 and diverges from the GB200 reference path. It also adds a cast launch per MoE layer (42 per step).

**Mechanism:** BF16 rounding of logits around ±5 is about ±0.01, so the sigmoid score moves by up to about 0.0025. With 288 experts the gap between the 8th and 9th scores is often of that size, so boundary expert choices flip for some tokens. The flipped expert carries near-equal weight, so the error is small per token but systematic.

**Evidence:**
- nvidia config.json text_config.moe_router_dtype = "float32", scoring_func sigmoid, topk_method noaux_tc, n_routed_experts 288
- v11src/vllm/model_executor/models/deepseek_v2.py:123-133 _get_moe_router_dtype -> torch.float32
- v11src/vllm/model_executor/layers/fused_moe/router/gate_linear.py:59-63 can_use_specialized_kernels = is_device_capability((9,0)) or is_device_capability_family(100)
- gate_linear.py:122-130 allow_cublas_router_gemm requires allow_specialized_router_gemm; :222-224 Tier 5 torch.mm(out_dtype=float32); :226-231 Tier 6 BF16 F.linear then output.to(out_dtype)
- v11src/vllm/platforms/interface.py:481-493 family check is capability//10 -> 12 != 10 for sm_121
- v11src/vllm/models/glm5next/nvidia/model.py:182-188 gate built with out_dtype=router_dtype

**Proposed action:** Add a one-line patch in a v13 layer: allow the Tier-5 cuBLAS bf16xbf16->fp32 path when is_device_capability_family(120). For example, set allow_cublas_router_gemm when the weight is bf16 and out_dtype is fp32 on any CUDA device. torch.mm with out_dtype does not depend on PDL or clusters. Leave the ll_bf16 tier gated, because PDL is disabled on SM12x by v8.

**Est. impact:** Decode about +0.1 ms per step, from removing 42 cast kernels (under 0.2%). Quality: restores fp32 routing fidelity. The size of the quality change is unquantified; the expected result is a small reduction in top-8 boundary flips relative to the reference.

**Validation:** GPU-free first: in a CPU/torch unit test, compare BF16-rounded against fp32 logits on captured hidden states and count top-8 set differences. On the Sparks: count-200 lossless, thinking-off smoke, and a small quality eval (for example greedy agreement against a reference) before and after. The decode bench should be unchanged within noise.

**Risks:** cuBLAS out_dtype support on sm_121 needs confirming. If unsupported it errors at warmup, which is caught at boot.

**Verifier reasoning:** gate_linear.py:59-63 sets can_use_specialized_kernels only for (9,0) or family 100. interface.py:491 computes the family as to_int()//10, so 121//10 = 12, which is not 10. allow_cublas_router_gemm at :123-130 therefore requires allow_specialized_router_gemm, and it is False on sm_121. forward() falls to Tier 6 (:226-231): a BF16 F.linear, then output.to(fp32). The checkpoint's layers.N.mlp.gate.weight is BF16 [288,4096] (header), and config.json text_config.moe_router_dtype is "float32". Glm5NextMoE takes _get_moe_router_dtype from deepseek_v2 (model.py:57, :182-188). So logits are rounded to BF16 on sm_121, whereas SM90/SM100 get fp32 output from the same BF16 inputs. The rounding estimate is overstated. BF16 half-ulp is 0.0156 in [4,8), where sigmoid'(5) is 0.0066, so the score moves by ~1e-4. The largest score shift is near |x|<2: half-ulp ≤0.0039 x 0.25 gives ~1e-3, not 0.0025. The patch is a sound, low-risk fix.

**Verifier corrected claim:** On sm_121, GateLinear falls to Tier 6. The router logits pass through BF16 (relative rounding 2^-9) before sigmoid, bias and top-8, even though moe_router_dtype=float32. The resulting sigmoid-score perturbation is at most ~1e-3 near the decision region, and ~1e-4 for |logit| of about 5.

**Verifier corrected impact:** Decode about +0.1 ms per step. Quality change is small and unquantified: a few top-8 boundary flips per token set.

## MOE-4: The image already contains a fused SM12x W4A16 NVFP4 MoE with a silu+clamp kernel; FlashInfer dispatch and the vLLM wrapper need a small patch to use it

- kind=perf component=FlashInfer b12x_fused_moe quant_mode='w4a16' (CuTe DSL, blackwell_sm12x/moe_w4a16_*) impact=3 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** FlashInfer 0.6.18.dev20260819 in v11 ships a W4A16 fused MoE for SM120/121. It takes modelopt NVFP4 weights, uses BF16 activations with BF16 mma m16n8k16 (the same numerics class as Marlin), and does routing, FC1, activation, FC2 and the top-k sum in one pipeline. Its kernel already implements the GLM clamp (gate <= limit, up in [-limit, limit]). Two gaps: _launch_sm120_w4a16_moe never forwards swiglu_limit to run_w4a16_moe, and vLLM's FlashInferB12xExperts always builds the W4A4 wrapper.

**Mechanism:** Fusing the five Marlin launches plus the moe_sum into one or two launches, with a tile schedule built for decode, removes launch gaps and intermediate round-trips. More importantly, it may raise achieved bandwidth on the 42 expert-streaming phases per step. Numerics stay W4A16, so quality is expected to match Marlin, not the lower W4A4 level.

**Evidence:**
- v11src/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py:4281-4295 _clamp_swiglu_inputs (gate>limit->limit; up clamped to [-limit,limit]) == vllm SiluAndMulWithClamp semantics (v11src/vllm/model_executor/layers/activation.py:205-206, 236-239)
- moe_w4a16_kernel.py:6678-6713 run_w4a16_moe(..., swiglu_limit=None, ...) accepts and normalizes the limit for 'silu'
- v11src/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py:3232-3251 quant_mode=='w4a16' calls _launch_sm120_w4a16_moe without swiglu_limit; :2810-2826 run_w4a16_moe call lacks swiglu_limit
- v11src/flashinfer/fused_moe/cute_dsl/b12x_moe.py:143-152 quant_mode 'w4a16' + source_format 'modelopt' documented
- moe_w4a16_prepare.py:696-730 modelopt prep keeps normal K/16 block grid + raw weight global scale (no bake-in)
- v11src/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:242-251 wrapper built without quant_mode/swiglu_limit; oracle/nvfp4.py:191-199 FLASHINFER_B12X not in NVFP4_BACKENDS_WITH_CLAMP
- moe_dispatch.py:2644-2681 _get_w4a16_packed_weights caches a prepared copy keyed by data_ptr (would duplicate weights if the functional path is used)

**Proposed action:** Patch in a v13 layer: (1) FlashInfer moe_dispatch.py: thread swiglu_limit through _launch_sm120_w4a16_moe into run_w4a16_moe. (2) vLLM FlashInferB12xExperts: add a W4A16 mode with quant_mode='w4a16', source_format='modelopt' and swiglu_limit=moe_config.swiglu_limit. Skip the scale bake-in and the a2_gscale fill for this mode. Call prepare_w4a16_packed_weights once in process_weights_after_loading, pass the result as _prepared_weights, and free the original w13/w2 tensors. (3) Add B12X-w4a16 to NVFP4_BACKENDS_WITH_CLAMP. (4) Before serving, run a unit test on one layer's real weights (CPU-loaded shard, one GPU) comparing against Marlin: max-abs and cosine of the MoE output. Keep Marlin as the default until A/B.

**Est. impact:** If Marlin is at about 70-80% of attainable bandwidth and the fused kernel reaches about 90%: MoE time -10-20%, which is about -6 to -12 ms per step, or +5-10% prose c=1 (21.2 -> 22.3-23.3 tok/s). If Marlin is already about 90%: 0-2% from launch fusion (about 126-168 fewer graph nodes, 0.2-0.5 ms). Prefill: similar BF16 MMA, so neutral.

**Validation:** Run the per-shape microbench against Marlin first (MOE-2). Then do a single-knob serve A/B: MOE_BACKEND=flashinfer_b12x with W4A16 mode, FORCE_UNSAFE_MOE=1, then bench_decode.py, count-200, smoke_vision.py, and dmesg Xid grep before and after. Check free -h for weight duplication (the MoE should stay at about 79.7 GiB per rank).

**Risks:** (a) Double memory: if the FlashInfer functional cache is used, a second prepared copy of 79.7 GiB per rank means an immediate UMA OOM. (b) CuTe DSL compile time and host memory at first call after weights are loaded (JIT per shape). Warm up before graph capture, and consider cutlass-dsl cache persistence. (c) Unproven on sm_121 in this lab. (d) EP is unsupported; irrelevant at TP=2.

**Verifier reasoning:** Several claims are verified. moe_dispatch.py:3232-3252 calls _launch_sm120_w4a16_moe without swiglu_limit, and :2810-2827 calls run_w4a16_moe without it. run_w4a16_moe accepts the limit (moe_w4a16_kernel.py:6699). _clamp_swiglu_inputs (:4281-4295) clamps gate at +limit and up at ±limit, matching SiluAndMulWithClamp (activation.py:238-239). The kernel uses bf16 mma m16n8k16 (:23, :915). The vLLM wrapper builds B12xMoEWrapper without quant_mode or swiglu_limit (flashinfer_b12x_moe.py:242-251), and FLASHINFER_B12X is absent from the clamp set (oracle/nvfp4.py:191-199). Three gaps in the plan: (a) B12xMoEWrapper.run() has no _prepared_weights parameter. Its w4a16 branch passes _weight_views=None and falls into _get_w4a16_packed_weights, a data_ptr-keyed cache that makes a second copy (b12x_moe.py:664-694, moe_dispatch.py:2644-2703). The patch therefore has to bypass the wrapper and call launch_sm120_moe directly, or patch the wrapper. FlashInfer provides prepare_w4a16_modelopt_nvfp4_weights(reuse_input_storage=True), which repacks in place (moe_w4a16_prepare.py:696-720); the reviewer did not mention it. (b) The prepared layout reuses Marlin-style _permute_nvfp4_scales with E4M3 k16 scales, so it is essentially a Marlin-derived W4A16 kernel with fusion. The bandwidth upside is unproven. (c) Upstream vLLM PR #52018 (merged 2026-08-21) adds a direct B12X MoE backend. It maps (nvfp4, None) to w4a16/modelopt_nvfp4, plumbs swiglu_limit, and has reuse_packed_weight_storage. Backporting it may be cleaner than patching the vendored FlashInfer copy. The impact is conditional on MOE-2.

**Verifier corrected claim:** FlashInfer 0.6.18 in v11 contains a W4A16 SM12x fused MoE whose kernel supports the silu clamp. Dispatch drops swiglu_limit, and the vLLM wrapper never selects quant_mode='w4a16'. The FlashInfer wrapper also cannot accept pre-prepared weights, so a patch must call launch_sm120_moe directly with prepare_w4a16_modelopt_nvfp4_weights(reuse_input_storage=True) to avoid a second 79.7 GiB copy. An alternative is to backport vLLM PR #52018's direct B12X backend, which already plumbs W4A16 and swiglu_limit.

**Verifier corrected impact:** Unknown until MOE-2 is measured: between 0 and about +10% prose c=1. The kernel is Marlin-derived, so large bandwidth gains are not assured.

## MOE-5: Native W4A4 is not a decode win: bytes are identical, W4A16 is measured faster at batch 1 on GB10, and W4A4 is the lower-fidelity path

- kind=methodology component=MoE backend choice (Marlin W4A16 vs CUTLASS/b12x W4A4) impact=3 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** At decode the expert weight bytes are the same (FP4 plus E4M3 scales) under every backend. W4A4 adds activation quant kernels, and the vLLM CUTLASS path adds shuffle kernels (about 10 launches per layer against Marlin's 5), with SM120 FP4 MMA tiles padded from 1-2 rows up to 128. External GB10 data shows W4A16 1.06x faster than W4A4 at batch 1. Separately, the H-cutlass rationale that Marlin ignores the calibrated input_scale and so loses quality is backwards. W4A16 keeps activations in BF16, so it is numerically at least as good as NVIDIA's W4A4 reference, and input_scale is only needed to quantize activations.

**Mechanism:** Decode is bound by weight bytes. The FP4 tensor cores only help compute-bound shapes (prefill). W4A4 adds FP4 quantization of activations, with about 8-10% relative RMS error per element for E2M1 with block-16 scales, which W4A16 does not have.

**Evidence:**
- v11src/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py:587-676 run_cutlass_moe_fp4: get_cutlass_moe_mm_data, shuffle_rows, scaled_fp4_experts_quant, cutlass_fp4_moe_mm, apply_moe_activation (clamp path unfused, 643-660), scaled_fp4_experts_quant, cutlass_fp4_moe_mm, shuffle_rows, mul+sum
- ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute: GB10 8B batch-1 decode NVFP4 W4A4 38.59 tok/s vs NVFP4A16 W4A16 40.85 tok/s
- opt-b12x/evidence/hypotheses.md H-cutlass: 'Marlin never reads them [input scales]'; repo evidence PR#9 decision.tsv h-snap note 'marlin ignores input_scale'
- v11src/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:61-122 Marlin NVFP4 scale conversion is exact for normal E4M3 scales (S0E5M3, only E4M3 subnormals zeroed at :116-117; nvidia pack scales are normalized to max 448 so sf=1)
- .quant_summary.txt: expert FC2 input amax p50 81.75, max 100.0 (= clamp bound 10*silu(10)); W4A4 must quantize these activations to E2M1

**Proposed action:** Keep Marlin, or the W4A16 fused path from MOE-4, as the decode MoE backend. Pursue W4A4 only for prefill (MOE-6/7). Correct the recipe notes: Marlin ignoring input_scale is not a quality loss. Do not frame flashinfer_cutlass or b12x W4A4 as a quality upgrade.

**Est. impact:** Avoids an estimated 0 to -10% decode regression and a small quality regression from switching the default to W4A4. Saves engineering time.

**Validation:** If W4A4 is ever booted: bench_decode.py prose c=1 against Marlin on the same day, plus a greedy-agreement quality check against the Marlin output on a fixed prompt set.

**Risks:** The external data point is a dense 8B model, not a MoE at 1-2 rows per expert. A decode-specialized W4A4 kernel such as the b12x micro backend could still edge out Marlin through fusion.

**Verifier reasoning:** I fetched the external data point from ai-muninn: Qwen3-8B dense on GB10, batch-1 decode W4A4 38.59 versus NVFP4A16 40.85 tok/s (1.059x). run_cutlass_moe_fp4 has the extra quant and shuffle kernels as described. marlin_utils_fp4.py:61-122 converts scales to S0E5M3 exactly and zeroes only scale*2^7 < 2 (subnormal E4M3). The nvidia pack's block scales are normalized with weight_scale_2 = amax/2688, so only effectively-zero blocks are affected. W4A16 keeps BF16 activations, so treating input_scale as needed only for activation quantization is correct. The hypotheses.md H-cutlass text ('Marlin never reads them') and the decision.tsv h-snap note frame this as a loss, which is backwards. The caveat stands: the external number is a dense 8B model, not an MoE with 1-2 rows per expert.

## MOE-6: vLLM's b12x W4A4 wrapper has numerics bugs beyond the missing clamp: the scale bake-in underflows E4M3 and FC2 scale 1.0 underflows small blocks

- kind=correctness component=vllm FlashInferB12xExperts.process_weights_after_loading + FlashInfer moe_activation.gated_activation_f32 impact=3 confidence=4 effort=M needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** Fixing only the swiglu_limit gate would still give a numerically broken model on this checkpoint. (1) The wrapper multiplies the E4M3 block scales (up to 448) by weight_scale_2 (amax/2688, about 4e-5) and stores the result back as E4M3. The median expert's largest block scale becomes 0.0195. About 96% of experts have every block scale below 2^-5, mostly in E4M3 subnormals with a 2^-9 step, which gives roughly 10-25% error per block scale. (2) It forces fc2_input_scale to 1.0, so FC2-input blocks with amax below about 0.094 are subnormal and blocks below about 0.006 flush to zero. (3) The SM12x W4A4 kernels apply swiglu_limit only for swigluoai_uninterleave, not silu.

**Mechanism:** E4M3's smallest normal is 2^-6 and its subnormal step is 2^-9. Folding a roughly 4e-5 global scale into per-block FP8 scales moves them out of their normalized range. With a global scale of 1.0 for FP4 activation quantization, small-magnitude SwiGLU outputs get subnormal or zero block scales. FC2 input amax reaches 100, the clamp bound, so the clamp matters.

**Evidence:**
- v11src/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:106-114 w13/w2_weight_scale = (scale.float()*scale_2).to(fp8 dtype); scale_2.fill_(1.0)
- flashinfer_b12x_moe.py:123-125 a2_gscale.fill_(1.0)
- v11src/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_activation.py:111-128 clamp applied only if activation=='swigluoai_uninterleave'; moe_dispatch.py:1370-1377 non-swigluoai forces swiglu_limit=None
- v11src/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:191-199, 259-269 explicit-backend ValueError (opt-b12x/evidence/iter-b12x-B-flashinfer/diagnosis.txt)
- nvidia .quant_summary.txt (756 expert weight quantizers): gate_up amax min 0.102/p50 0.117/max 1.25; down amax min 0.0938/p50 0.109/max 2.0 -> weight_scale_2 = amax/2688 = 3.5e-5..7.4e-4; baked max block scale p50 0.0195/0.0182; fraction of experts with baked max < 2^-5: 96.7% (gate_up), 95.2% (down)
- v11src/flashinfer/fused_moe/cute_dsl/b12x_moe.py:121-126 input_global_scale lets w1_alpha carry exact fp32 weight scale (no bake needed)
- repo AGENTS.md: 'Do not set VLLM_GLM53_MOE_INPUT_SCALE=1.0. That constant underflows per 16-element block' (same failure class as (2))

**Proposed action:** If b12x W4A4 is ever tried (for prefill, or a decode A/B): (1) do not bake. Pass w1_alpha/w2_alpha = weight_scale_2 in fp32 and input_global_scale = the calibrated per-layer 1/input_scale, checking the kernel's input_scales_are_reciprocal convention. (2) Pass the calibrated FC2 global scale, about 2688/81.75 ≈ 33 at the median, instead of 1.0. (3) Add a silu clamp branch in gated_activation_f32 (g = min(g, lim); u = clamp(u, -lim, lim)) and plumb swiglu_limit for silu through the micro/static/dynamic compile keys. (4) Only then add FLASHINFER_B12X to the clamp set.

**Est. impact:** Correctness gate. Without these fixes a b12x W4A4 run would show large quality loss (block-scale error of roughly 10-25%) that decode tok/s would not reveal.

**Validation:** GPU-free: emulate the bake in numpy on the header-derived amax range, round-tripping through E4M3, to show the scale error distribution. On GPU: per-layer MoE output cosine against Marlin (target above 0.999 for W4A4), then count-200 and a greedy-agreement eval.

**Risks:** Kernel convention mismatches (reciprocal scales). The upstream eugr/b12x reportedly supports 'SiLU with clamp 10' and W4A8. Backporting from it may be simpler than patching the FlashInfer vendored copy.

**Verifier reasoning:** flashinfer_b12x_moe.py:106-114 multiplies the E4M3 block scales by w13/w2_weight_scale_2 and casts back to fp8. Here scale_2 is the raw ModelOpt value passed through untouched by prepare_nvfp4_moe_layer_for_fi_or_cutlass (flashinfer_fp4_moe.py:321-437); it is amax/2688. :123-125 fills a2_gscale with 1.0. Because input_global_scale is not passed, B12xMoEWrapper uses w1_alpha (now 1.0) as the FC1 input global scale too (b12x_moe.py docstring and run()). moe_activation.py:113-121 clamps only swigluoai_uninterleave, and moe_dispatch.py:1370-1377 sets swiglu_limit to None for other activations. I recomputed the .quant_summary.txt stats: 756 expert weight quantizers, gate_up amax min/p50/max 0.102/0.117/1.25 and down 0.0938/0.109/2.0. The fraction with amax/6 < 2^-5 is 96.7% for gate_up and 95.2% for down; none fall below 2^-6. FC2 input amax is 20.1/81.75/100.0, where 100 is 10*silu(10). The 2^-6*6 = 0.094 and 2^-10*6 = 0.0059 thresholds are correct. Caveat: the summary lists only experts 0-17 per layer (756 = 42x18), so the percentages come from a 6% sample.

## MOE-7: flashinfer_cutlass OOM came from a runtime JIT, not the kernel: pre-build it in the image, or use vLLM's already-compiled CUTLASS FP4 MoE with no JIT

- kind=ops component=FlashInfer JIT (fused_moe_120), image build, run.sh volumes impact=3 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The spark2 global OOM happened while fused_moe_120 was being JIT-compiled after 90.67 GiB of weights were loaded. That module is about 25 heavy CUTLASS/TRT-LLM translation units. FlashInfer runs ninja without -j unless MAX_JOBS is set, so ninja defaults to nproc+2 = 22 parallel nvcc/cudafe++ processes on 20 cores, with about 18 GiB free. v8 uninstalled flashinfer-jit-cache, and run.sh mounts no FlashInfer cache, so every boot would JIT again. The fix is to build ahead of time, not to avoid the backend. vLLM's own CUTLASS FP4 MoE (--moe-backend cutlass) is compiled into _C; with CUDA 13.0.1 its SM120 kernels are plausibly built as 12.0f family cubins, which run on sm_121. It is in the clamp-capable set and needs no JIT.

**Mechanism:** CUTLASS template translation units need several GB of host RAM each in cicc and cudafe++. With 22 in parallel on unified memory already holding the weights, the kernel OOM killer fires. The GPU never runs out; the compiler does.

**Evidence:**
- repo evidence/oom-20260831/diagnosis.txt: kernel_oom_host=spark2, invokers=cudafe++, spark2_avail_ram_at_draft_load=18.21GiB, weights_gib=90.67, image v12 (adds cuda-nvrtc-dev per docker history glm53-sm121-v12)
- v11src/flashinfer/jit/fused_moe.py:58-73 gen_cutlass_fused_moe_sm120_module; :162-213 source list (~25 .cu/.cpp incl. all moe_gemm_kernels_*.cu + generated instantiations)
- v11src/flashinfer/jit/cpp_ext.py:346-365 ninja -j only if MAX_JOBS set (ninja default = nproc+2; nproc=20 on host)
- v11src/flashinfer/jit/env.py:58-63,148-163 cache at $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<ver>/<arch>/cached_ops (host shows ~/.cache/flashinfer/0.6.18/121a)
- repo docker/Dockerfile.sm121-v8: 'pip uninstall -q -y flashinfer-jit-cache'; repo run.sh:293 only -v HF_CACHE (+template), run.sh:250-252 FLASHINFER_CUDA_ARCH_LIST=12.1a
- vllm CMakeLists (github main): CUDA>=13.0 -> FP4_SM120_ARCHS '12.0f', MARLIN_MOE_ARCHS '8.0+PTX;12.0f'; image env CUDA_VERSION=13.0.1, TORCH_CUDA_ARCH_LIST includes 12.0 (docker image inspect glm53-sm121-v11)
- v11src/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py:707-713 CutlassExpertsFp4 supports family 120; oracle/nvfp4.py:191-199 VLLM_CUTLASS in clamp set
- Qwen3.8 NVFP4 recipe booted FLASHINFER_CUTLASS NvFp4 MoE on the same sm_121 hosts (Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark/evidence/boot/backend-needles.txt:1)
- repo run.sh:114-118 refuses any MOE_BACKEND != marlin without FORCE_UNSAFE_MOE=1

**Proposed action:** Option A, cheapest: A/B --moe-backend cutlass with FORCE_UNSAFE_MOE=1; no JIT is expected. First confirm the _C.abi3.so contains sm_120f FP4 grouped-GEMM kernels, using cuobjdump in a no-GPU container during an idle window. Option B: add a v13 Dockerfile layer that installs cuda-nvrtc-dev, then runs FLASHINFER_CUDA_ARCH_LIST=12.1a MAX_JOBS=4 python3 -c 'from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module as g; g().build()'. That bakes cached_ops/fused_moe_120 into /root/.cache/flashinfer/<ver>/121a. Also set -e MAX_JOBS=2 in run.sh as a safety net, and optionally mount a host FlashInfer cache dir so any remaining JIT persists.

**Est. impact:** No decode change by itself. It unblocks the W4A4 prefill experiments in MOE-8 without risking another spark2 OOM, and removes minutes of per-boot JIT for any FlashInfer module used.

**Validation:** Build v13 on a machine or at a time without GPU load; the build needs no GPU. Check that the .so exists in the image. Boot with MOE_BACKEND=flashinfer_cutlass (or cutlass) and FORCE_UNSAFE_MOE=1 while watching free -h. The engine log must show no ninja or nvcc activity, OOMKilled must be false, then run count-200.

**Risks:** Hash or path mismatch (FlashInfer version string, arch suffix '121a', HOME) causes a silent re-JIT; keep MAX_JOBS=2 as a guard. The claim that vLLM _C has 12.0f cubins is inferred from upstream CMake and must be checked with cuobjdump. flashinfer_cutlass autotuning at warmup may add memory.

**Verifier reasoning:** Verified. oom-20260831/diagnosis.txt records spark2 global OOM with cudafe++ among the invokers and 18.21 GiB available at draft load. It also says 'cudafe++/graph capture', so the exact JIT module and the parallelism are inferred, not logged. cpp_ext.py:346-365 adds -j only when MAX_JOBS is set, and ninja's default is nproc+2. Dockerfile.sm121-v8:67 uninstalls flashinfer-jit-cache (flashinfer-cubin stays installed). run.sh:293-296 mounts only the HF cache and the template. The CutlassExpertsFp4 family-120 support and the VLLM_CUTLASS entry in the clamp set are correct. Corroboration: the Qwen3.8 boot log shows FLASHINFER_CUTLASS selected at 15:25:39 and the next MoE log at 15:32:18. That ~6.6-minute gap is consistent with a JIT that succeeded under more headroom. Weakening factor: v11's Env shows VLLM_BUILD_PIPELINE=local and VLLM_IMAGE_TAG=local/vllm-openai:dev, so _C is a local fork build. The upstream-CMake inference that sm_120f FP4 MoE cubins exist is weaker than stated and must be checked with cuobjdump.

## MOE-8: Prefill: native FP4 MoE helps, but at chunk 2048 its gain is capped by re-streaming all 288 experts every chunk; pair it with a larger chunk

- kind=perf component=MoE prefill (Marlin BF16-MMA vs CUTLASS FP4 MMA) x max_num_batched_tokens impact=3 confidence=2 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** The measured prefill rate is about the same across context lengths: 1425 tok/s at 10k tokens and 1298 tok/s at 318k. So prefill cost is dominated by per-token GEMM work, not attention. At the current 2048-token chunk, each chunk touches all 288 experts per layer, which is 85.6 GB per rank, about 0.38 s at 225 GB/s, against a 1.44 s chunk. Marlin's BF16 math is about 17.3 TFLOP per rank per chunk. The MoE phase is therefore close to bandwidth-bound already, and FP4 math saves at most about 0.1 s per chunk. At 8192-token chunks the weight-streaming floor per token drops 4x and the FP4 math advantage becomes visible.

**Mechanism:** Chunk time is roughly max(weight stream, math) per layer. Marlin runs about 40-60% of 99.8 TFLOPS BF16, while FP4 MMA runs at up to about 511 TFLOPS. A larger chunk amortizes the fixed 85.6 GB expert stream.

**Evidence:**
- repo README.md:22: 10271-token prompt prefill 1425 tok/s (TTFT 7.2 s); 318,123-token prompt in 4m05s (=1298 tok/s)
- repo evidence/rebench-20260902T204243Z/engine-rank1.log.tail:12 'max_num_scheduled_tokens is set to 2048'; recipe.yaml:50 MAX_NUM_BATCHED_TOKENS ""
- Routed FLOPs/token/rank = 8 experts*12.58M params*2*42 layers = 8.46 GFLOP -> 17.3 TFLOP per 2048 chunk; 69 TFLOP per 8192
- All-expert stream per chunk = 42*288*7.078 MB = 85.6 GB/rank (P(expert untouched) ~ e^-57)
- GB10 measured ~99.8 TFLOPS BF16, ~207.7 FP8, ~511 FP4 dense (ai-muninn.com; forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4...)
- v11src/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:361-363 NVFP4 MoE + FP8/INT8 activation raises RuntimeError (no Marlin W4A8 shortcut)

**Proposed action:** Order, one knob at a time: (1) raise MAX_NUM_BATCHED_TOKENS to 4096, then 8192, on Marlin and measure the TTFT of the 10k and 318k needles plus UMA headroom. This is a separate knob from the kernel and may belong to another dimension. (2) Once MOE-7 has removed the JIT risk, A/B --moe-backend cutlass (vLLM, no JIT) or flashinfer_cutlass at the best chunk size for prefill only. Keep it only if decode stays within noise (MOE-5) and quality gates pass.

**Est. impact:** Estimates: at chunk 2048, native FP4 MoE saves at most about 0.1 s of each 1.44 s chunk, about +5-8% prefill. At chunk 8192, a larger chunk alone gives +5-10% on Marlin; adding FP4 MoE gives about +15-20% more, about +20-30% combined (1425 -> ~1800-1900 tok/s; 318k TTFT about 245 s -> ~185-200 s).

**Validation:** Use the unique-salt 8k and ~300k needle prompts and record TTFT and prefill tok/s, 3 runs each, per knob. Watch free -h and swap. Decode bench must stay within noise.

**Risks:** Larger chunks raise activation memory on UMA that is already about 115/121 GiB (Marlin cache13 at 8192 tokens is about 0.54 GB, plus attention and KDA workspaces). W4A4 lowers activation fidelity. FlashInfer autotune memory. The Marlin efficiency assumption (40-60%) is unmeasured.

**Verifier reasoning:** The arithmetic checks out. Per-rank expert params are 12.58M, giving 8.45 GFLOP/token/rank and 17.3 TFLOP per 2048-token chunk. The all-expert stream is 42*288*7.078 MB = 85.6 GB. The README gives 1425 tok/s at 10271 tokens and 318123 tokens in 245 s (1298 tok/s). engine-rank1.log.tail:12 shows max_num_scheduled_tokens=2048. However, the proposal to raise MAX_NUM_BATCHED_TOKENS to 4096 or 8192 does not address two prior reverts. The README records that 4096 at two sequences was measured and reverted: structured c=2 55.5 -> 51.6, KV pool 372877 -> 363476. decision.tsv h2 records that 4096 was reverted for a 10x c=4 TTFT. AGENTS.md requires naming the revisited decision and explaining why the evidence differs. The Marlin 40-60% BF16 efficiency is an assumption, so the +20-30% combined figure is speculative.

**Verifier corrected claim:** The prefill arithmetic is sound. Larger chunks were already tried: 4096 was reverted twice, for decode and admission regressions and a smaller KV pool. Any chunk-size retest must be framed as revisiting those decisions, for example as a prefill-only lane. The FP4 MoE prefill gain is an upper-bound estimate.

**Verifier corrected impact:** Prefill between +5% and +30% (estimate). Prior measurements show decode and KV-pool regressions at 4096.

## MOE-9: Dense layers 0-2 auto-select the FlashInfer CUTLASS W4A4 GEMM on sm_121; --linear-backend marlin would give W4A16 fidelity at no decode cost

- kind=quality component=ModelOptNvFp4LinearMethod kernel selection (init_nvfp4_linear_kernel) impact=2 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The NVFP4 linear candidate order puts FlashInferCutlassNvFp4LinearKernel before Marlin. Its gate is cutlass_fp4_supported() and has_device_capability(100); sm_121 passes because 121 >= 100. So the three dense MLPs (the only NVFP4 linear layers: attention, shared experts and vision are BF16) run W4A4, with calibrated static activation scales and a FlashInfer JIT at boot. Forcing Marlin W4A16 removes activation quantization error in the first three layers, where errors propagate through the whole stack.

**Mechanism:** W4A16 dequantizes the weights and keeps activations in BF16, so the per-element E2M1 activation error disappears. Bytes are identical, so decode time is the same.

**Evidence:**
- v11src/vllm/model_executor/kernels/linear/__init__.py:500-512 _POSSIBLE_NVFP4_KERNELS order (FlashInferCuteDsl, FlashInferCutlass, FlashInferB12x, Cutlass, Marlin, ...)
- v11src/vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:106-119 gate has_device_capability(100); :35-42 CuteDsl requires family 100 (excluded)
- v11src/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py:56-61 cutlass_fp4_supported -> torch.ops._C.cutlass_scaled_mm_supports_fp4(capability)
- v11src/vllm/config/kernel.py:142-154 linear_backend accepts 'marlin'
- nvidia hf_quant_config.json exclude list: all self_attn, shared_experts, mlp.gate, lm_head, embed, visual excluded; layers 0-2 mlp quantized; .quant_summary.txt layers.0.mlp.down_proj input amax 0.309, weight amax 0.117
- Header: dense_mlp 0.237 GiB total -> 0.127 GB/rank/step (~0.5 ms at 225 GB/s)

**Proposed action:** First confirm from the boot log which kernel is chosen (grep 'for NVFP4 GEMM' / 'Selected'; this line is not captured in evidence). If it is FlashInferCutlass, A/B EXTRA_ARGS='--linear-backend marlin' as a single knob and add it to recipe.yaml if quality improves or stays neutral. Check that it applies only to NVFP4 linears.

**Est. impact:** Decode about 0% (0.127 GB per rank per step either way). Prefill about -1-2% (3 of 45 layers, about 0.45 of 19 GFLOP per token per rank). Quality: small, unquantified improvement toward the BF16 reference. Removes one boot-time JIT (fp4 gemm sm120).

**Validation:** Check the boot log line for the selected kernel. Run count-200, the thinking-off smoke, and greedy agreement against a reference on a fixed prompt set, before and after. bench_decode.py should be within noise.

**Risks:** Whether --linear-backend covers other quantized linears (there are none in this pack except NVFP4 dense). The Marlin NVFP4 linear has had SM121 illegal-memory-access reports upstream (vLLM issue #54666 cites #49926/#50934/#52225).

**Verifier reasoning:** _POSSIBLE_NVFP4_KERNELS (kernels/linear/__init__.py:500-512) orders FlashInferCuteDsl, FlashInferCutlass, FlashInferB12x, Cutlass, then Marlin. CuteDsl requires family 100. FlashInferCutlass requires cutlass_fp4_supported() and has_device_capability(100), and 121 >= 100 passes. Whether cutlass_scaled_mm_supports_fp4(121) returns True depends on the local _C build, which is unverified. The dense layers 0-2 are NVFP4 in the header (U8 weights plus input_scale). --linear-backend marlin is accepted and falls back per layer type with a warning (:336-358). No boot log in evidence captures the selected NVFP4 linear kernel. The quality benefit covers 3 of 45 layers and is unquantified.

## MOE-10: Shared expert: BF16, already overlapped on an aux stream; the only lever is bytes (FP8 weight-only would save ~0.53 GB/step)

- kind=perf component=Glm5NextMoE.shared_experts + SharedExperts aux stream impact=2 confidence=3 effort=M needs_gpu=True
- **verdict: confirmed** (corrected confidence 3)

**Claim:** The shared expert (1x2048 intermediate, BF16, excluded from quantization) is a separate MergedColumn/RowParallel MLP with reduce_results=False. Its output is added to the routed output before a single TP all-reduce per MoE layer. It runs MULTI_STREAM_OVERLAPPED on an aux CUDA stream when tokens <= 256, so it already overlaps with the gate, router and Marlin. Both paths are bandwidth-bound, so the overlap hides launch gaps but not bytes: 25.2 MB per layer per rank, 1.06 GB per step, about 4.7 ms at 225 GB/s.

**Mechanism:** Weight bytes, not overlap, set its cost. FP8 per-channel weight-only (Marlin fp8_w8a16 or CUTLASS FP8) halves them.

**Evidence:**
- v11src/vllm/models/glm5next/nvidia/model.py:210-225 shared_experts Glm5NextMLP(reduce_results=False, swiglu_limit); :227-250 FusedMoEFactory(shared_experts=...)
- v11src/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:99-119 MULTI_STREAM_OVERLAPPED when tokens <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD; v11src/vllm/envs.py:285 default 256
- v11src/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:726-771 shared+routed summed then one _maybe_reduce_final_output all-reduce
- Header: layers.3.mlp.shared_experts.{gate,up,down}_proj.weight BF16 16,777,216 B each -> 50.3 MB/layer, 25.2 MB/rank; 42 layers -> 1.06 GB/rank/step

**Proposed action:** Low priority, do after the higher-ranked items. Add an online FP8 per-output-channel weight-only quant patch for shared_experts; this needs a patch because ModelOpt excluded them. A/B it for quality. Do not fold the shared expert into the routed Marlin call as a 289th expert: Marlin does not take BF16, and it would force NVFP4 on a module NVIDIA kept in BF16.

**Est. impact:** -0.53 GB per rank per step, about -2.3 ms per step, about +2% decode. The same arithmetic on the BF16 attention projections (6.06 GB per rank per step, about 27 ms) would be about +10% but belongs to the attention dimension and carries higher quality risk.

**Validation:** bench_decode.py prose c=1/2 and count-200, plus a greedy-agreement quality check, each as a single knob.

**Risks:** NVIDIA deliberately excluded shared experts (quality). Adds patch surface on the fork-only model code.

**Verifier reasoning:** The header shows shared_experts gate/up/down as BF16 at 16,777,216 B each. That is 50.3 MB per layer, 25.2 MB per rank, and 1.057 GB per rank over 42 layers, about 4.7 ms at 225 GB/s. shared_experts.py:99-119 selects MULTI_STREAM_OVERLAPPED when tokens ≤ VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD, which defaults to 256 (envs.py:285). model.py:216-225 sets reduce_results=False. The FP8 halving (-0.53 GB, ~2.3 ms, ~2%) is correct arithmetic. The quality risk from overriding NVIDIA's exclusion is correctly flagged.

## MOE-11: Stability evidence: the Xid 31 concern about b12x is unsubstantiated in lab logs, b12x MoE runs in production on these Sparks, and upstream reports Marlin NVFP4 crashes on SM121

- kind=ops component=Backend stability on sm_121 (Marlin vs b12x) impact=2 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** The only lab reference to b12x Xid 31 is a refuse string in the Qwen3.8 run.sh and AGENTS.md; no log shows an Xid. The DeepSeek-V4 recipe on the same two Sparks runs b12x MoE (MXFP4, B12X_MOE_FORCE_A8=1) at TP=2 as its default. Upstream vLLM issue #54666 reports repeated Marlin NVFP4 illegal-memory-access crashes on SM121, and B12X clean in a 26-minute stress run. FlashInfer issue #3383 reports b12x illegal-address errors on sm_121a only with EP>1, which does not apply at TP=2.

**Mechanism:** Risk is set by the kernel family and the parallelism mode. There is no local evidence that b12x is less stable than Marlin at TP=2 on GB10.

**Evidence:**
- Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark/run.sh:83-84 'b12x ... has Xid 31 reports on sm_121'; Qwen3.8-...-pr8/AGENTS.md:18
- DeepSeek-V4-Flash-Vision-Exp-vLLM-2x-DGX-Spark/run.sh:27 MOE_BACKEND b12x, :222 VLLM_USE_B12X_MOE=1, :226 B12X_MOE_FORCE_A8=1; README.md:133
- github.com/vllm-project/vllm/issues/54666 (Marlin NVFP4 SM121 EngineDeadError/illegal memory access; B12X no traceback over 18.5M prompt tokens)
- github.com/flashinfer-ai/flashinfer/issues/3383 (b12x illegal address, sm_121a, EP>1)
- v11src/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:176-179 B12X excluded from auto 'until the upstream CUTLASS SM121 MMA op guard is resolved'

**Proposed action:** Treat Xid risk as a validation gate, not a veto. For any MoE backend A/B: run dmesg | grep -i xid before and after on both nodes, run count-200, and do a 30-minute mixed soak (prose, structured and vision) with c=2. Keep the FORCE_UNSAFE_MOE guard until the soak passes. Watch for Marlin illegal-address errors on the current default as well.

**Est. impact:** Risk management. It prevents a Marlin crash and lets the W4A16 b12x path (MOE-4) be evaluated on evidence rather than rumor.

**Validation:** Zero NVRM or Xid lines across the soak on both nodes. Engine alive, with health 200 throughout.

**Risks:** The DeepSeek recipe uses the eugr fork image and MXFP4, not the FlashInfer-vendored NVFP4 path in v11, so its stability only partially transfers.

**Verifier reasoning:** The local parts hold. Xid mentions in the lab are refuse strings (Qwen3.8 run.sh:83-84) and unrelated NVRM/Xid-zero checks, and I found no b12x Xid log. DeepSeek-V4 run.sh:27/222/226 does use b12x. But FlashInfer #3383 is mischaracterized: it also reports that non-EP (plain TP=8, all experts per rank) b12x fails on sm_121a with cudaErrorInvalidValue inside launch_sm120_moe, 'regardless of EP'. The 'only EP>1' reading is wrong. vLLM #54666's clean B12X run uses the direct b12x backend from vLLM PR #52018 (NvFp4MoeBackend.B12X), which v11 does not have; v11's enum has only FLASHINFER_B12X (oracle/nvfp4.py:39-48). DeepSeek likewise uses the eugr image. Neither result is evidence for the FlashInfer-vendored path in v11. The Marlin SM121 crash reports are real, and treating Xid as a gate rather than a veto is reasonable.

**Verifier corrected claim:** There is no local log of b12x Xid 31. Upstream evidence cuts both ways. vLLM #54666 reports Marlin NVFP4 illegal-memory-access crashes on SM121 and a clean run with the separate direct-b12x backend (vLLM #52018, absent from v11). FlashInfer #3383 reports the FlashInfer-vendored b12x failing on sm_121a both with EP (illegal address) and without EP (cudaErrorInvalidValue). The stability of v11's FLASHINFER_B12X at TP=2 is unknown, and its soak gate should be strict.

**Verifier corrected impact:** Risk management. Stability evidence for the FlashInfer b12x path in MOE-4 is weaker than presented.

## Open questions
- Which NVFP4 linear kernel do dense layers 0-2 actually select at boot on v11? The code order says FlashInferCutlassNvFp4LinearKernel (W4A4), but no 'for NVFP4 GEMM' log line is captured in evidence/.
- Does vLLM _C in glm53-flash-arm64-cu130 contain SM120-family (12.0f) cutlass_fp4_moe_mm and cutlass_scaled_fp4_mm cubins that load on sm_121? This is inferred from upstream CMake plus CUDA 13.0.1 and needs cuobjdump --list-elf on _C.abi3.so in an idle window.
- Marlin MoE achieved bandwidth on GB10 at E=288, N=2048/K=4096 and N=4096/K=1024, M=8-16: is it about 70% or about 90% of the ~225 GB/s attainable? This decides whether MOE-4 is worth 0% or about 10%.
- Real distinct experts per layer across an 8-token DFlash2 verify window. The independence bound is 58; the measured k5-vs-k7 delta suggests somewhat fewer. Capture with --enable-return-routed-experts.
- The nvidia checkpoint's MTP layer (layers.45) carries 13.84 GiB of BF16 routed experts, and hf_quant_config does not exclude layer 45. Are these weights skipped when SPEC=dflash2, and which unquantized MoE backend runs them under SPEC=mtp with --moe-backend marlin (the Qwen3.8 recipe hit 'marlin not supported for unquantized MoE')?
- Is humming-kernels[cu13], installed in the base image via requirements-cuda, importable in v11, and what is its JIT memory footprint? It is another clamp-capable W4A16 MoE backend worth a microbench against Marlin.
- CuTe DSL compile time and peak host RAM for the b12x W4A16 kernels at first call after 90+ GiB of weights. Can the cutlass-dsl compile cache be persisted or pre-warmed at image build like the FlashInfer AOT fix?
- Does the b12x W4A4 kernel expect reciprocal or direct global scales (input_scales_are_reciprocal)? vLLM's comment that calibrated a2_gscale would saturate FP4 contradicts the checkpoint's FC2 input amax of about 82-100, which implies a global scale of about 27-33.

## Verifier: missed issues
- An independent k=5 rebench exists and was not cited: git show origin/agent/hillclimb-20260903:evidence/rebench-dflash5-20260903T045815Z/summary.json (v11, LibertAI caca4e6, DFlash2-5). It shows prose c=1 21.76 (acc 2.354), prose c=2 19.62/stream (acc 2.57), structured c=1 55.73 (acc 6.0), structured c=2 41.56/stream. Structured is measured, not just estimated, at -18% c=1 and -31% c=2 against the published k=7. Prose c=1 gains only +2.6% against the published 21.2.
- vLLM v11 already ships a HUMMING NvFp4 MoE backend. It is in NVFP4_BACKENDS_WITH_CLAMP (oracle/nvfp4.py:191-199) and selectable with --moe-backend humming (:149-157), with a W4A16-class kernel and VLLM_HUMMING_MOE_GEMM_TYPE/VLLM_HUMMING_INPUT_QUANT_CONFIG knobs (envs.py:185-188). It is a zero-patch, clamp-correct Marlin alternative that the reviewer never considered. Whether the humming package is installed in v11 is unverified, but the boot log lists it among the potential backends.
- Upstream vLLM PR #52018 (merged 2026-08-21) adds a direct B12X MoE backend (vllm/model_executor/layers/fused_moe/b12x.py). It has modes (nvfp4,None)->w4a16 modelopt, (nvfp4,mxfp8)->w4a8, plumbs swiglu_limit, and uses reuse_packed_weight_storage. It is a cleaner path than patching FlashInfer's vendored copy (MOE-4/6), and it is the backend that vLLM #54666's stability data actually covers. A matching local image (eugr/spark-vllm-b12x:latest) exists.
- The FlashInfer B12xMoEWrapper.run() cannot accept pre-prepared W4A16 weights. It always goes through _get_w4a16_packed_weights, a data_ptr-keyed cache, and duplicates the expert weights (b12x_moe.py:664-694; moe_dispatch.py:2641-2703). A safe patch must call prepare_w4a16_modelopt_nvfp4_weights(reuse_input_storage=True) and launch_sm120_moe directly. Without this the MOE-4 plan OOMs UMA immediately.
- The MTP layer (layers.45) routed experts are BF16 in the nvidia pack: 13.5 GiB versus 3.797 GiB for each NVFP4 MoE layer (safetensors headers). The AGENTS.md rollback SPEC=mtp on the nvidia pack would therefore add about 6.75 GiB/rank if TP-split, on UMA already at ~115/121 GiB with a 4.14 GiB KV pin. That rollback is likely to OOM and has not been tested on this pack.
- The non-MoE per-step byte budget leaves out the DFlash2 drafter: model.safetensors is 2,342,169,800 B, about 1.2-2.3 GB/rank per step depending on TP sharding, plus the draft-position logits. Non-MoE traffic is closer to 9-10 GB/rank/step than 8. At 225 GB/s the model then leaves little room for NCCL, attention compute and launch gaps. That further suggests MoE distinct-expert bytes are overestimated by the independence model.
- The .quant_summary.txt holds only experts 0-17 per layer (756 = 42 x 18 weight quantizers). All per-expert amax statistics used in MOE-6 and elsewhere come from a 6% sample, not all 288 experts.
