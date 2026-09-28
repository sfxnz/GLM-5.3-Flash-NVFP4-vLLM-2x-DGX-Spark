# Dimension: external-research

## Reviewer summary

External research as of 2026-09-25. No commits have landed on nvidia/GLM-5.3-Flash-NVFP4 after 09b04e5 (2026-09-11). The card only covers GB200 TP4/EP/32-seq and says nothing about Spark. vLLM v0.30.0 (the local image, commit ced6857, built 2026-09-21) supports glm5_next natively and ships a b12x fused-MoE backend for SM12x that passes swiglu_limit to the kernel for SiLU. That is the clamp the b12x worktree was blocked on. It also ships adaptive verification for DFlash2. It still lacks three things this recipe needs: GLM DFlash2 aux capture (PRs #55682 and #56983 are open), an SM121 NoPE sparse-MLA decode path (the SM90 backend is still gated to major==9), and the DFlash prefix-cache fix (#54163 is open). Staying on the v11 chain and backporting is cheaper than rebasing.

Community numbers on the same hardware put this recipe's prose c=1 of 21.2 tok/s (LibertAI pin, DFlash2-7) near the bottom. On the same nvidia pin 09b04e5, 0rand/pilcothink (b12x MoE, DFlash2 k=5, async on) reports prose/filler 33.5 and c1 decode 32.5-33.1. Tony reports prose 28.4 at k=7. MiaAI EXL3 gets prose 27.1 at the same acceptance of about 2.4, which is an 88 ms step against our about 115 ms. The sparkring b12x MTP3 lane runs 13.1 target steps/s.

A calibrated cost model fits our own measurements, both the k=5 hillclimb gain (+14%) and the structured cell (67.6). In it, step ≈ 51 ms plus 8.5 ms per verify token. Prose acceptance collapses after draft slot 3, so the biggest cheap lever is verify width: adaptive verification (#52228), or k=3. The model predicts +30-40% prose with structured unchanged under adaptive trimming. The next lever is the 7.8 GB/rank of BF16 attention, shared-expert and lm_head weights read on every step. Community FP8, MXFP8 and NVFP4 attention packs exist, with measured x1.14-1.30 at TP4. The vendor itself keeps KDA attention in BF16, though, so this is a quality-gated experiment.

Four quality and ops risks for PR #11. First, the local DFlash2 drafter is the stale initial release; two retrained checkpoints followed on 08-28 and 08-31. Second, the DFlash2 draft-KV-group design very likely zeroes prefix-cache hits (two independent reports, and the same code path is in v11). Third, SPEC=mtp cannot work as the rollback on the nvidia pack, whose MTP layer is 13.84 GiB of BF16. Fourth, vision on is reported to add about 15.7 GiB on the front-end, while the node is already at about 115/121 GiB. Numbers from other stacks use different prompts and generation lengths, so treat them as directional until re-benched with bench_decode.py.

## EXT-1: Trim verify width on prose: backport adaptive verification (vLLM #52228) to DFlash2, or use k=3 for prose-heavy serving

- kind=perf component=spec-decode / vllm/v1/worker/gpu/spec_decode (v11 DFlash2 backport) impact=5 confidence=4 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** Step time grows about linearly with verify tokens (≈8.5 ms/token at TP2). DFlash2 prose acceptance is near zero after draft slot 3. So DFlash2-7 spends about 34 ms per step verifying slots that are almost never accepted. Upstream #52228 (merged 2026-09-14) trims the verified drafts each step with an online acceptance estimator and now covers DFlash2. v11 allows adaptive verification only with DSpark.

**Mechanism:** On 273 GB/s LPDDR5X the verify pass is weight-read-bound. Distinct routed experts per layer for v verify tokens are about 288*(1-(1-8/288)^v): 58.1 at v=8 and 30.6 at v=4, at 14.16 MB each (159.469 GiB / 42 layers / 288 from the safetensors headers). That is 17.3 GB/rank at v=8 against 9.1 GB/rank at v=4, a difference of about 30 ms at 273 GB/s. The draft slots that pay for this rarely land on prose.

**Evidence:**
- v11src vllm/config/speculative.py:1177-1178: `if self.method != "dspark" and self.enable_adaptive_verification: raise ValueError("Adaptive verification only supported with DSpark")`
- https://github.com/vllm-project/vllm/pull/52228 merged 2026-09-14T22:58:49Z, 9 files +900/-28, touches vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py and adds acceptance_estimator.py; reported +14.7% for DFlash2 (Muse-Glimmer 1xGB200), +19.0% for MiMo+DFlash, +44.5% for Kimi-K2.5+DFlash, no GSM8K loss
- Measured locally (PR #9 evidence/iter-H1-20260903): NUM_SPECULATIVE_TOKENS=5 took prose c=1 from 19.23 to 21.97 (+14%), c=2 from 15.13 to 17.52
- Conditional per-position prose acceptance for DFlash2 K7 on 2x Spark: 0.75/0.58/0.41/0.28/0.16/0.09/0.06, mean 2.39 (https://github.com/Kristianaaron/GLM-5.3-Flash-EXL3-2x-DGX-Sparks); structured 0.98/0.98/0.94/0.94/0.91/0.83/0.83
- Tony's TP4 cost model step_ms = 34.6 + 4.12 x verify_tokens (https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark docs/EXPERIMENTAL-NVFP4-ATTENTION.md); Tony README: k=5 beats k=7 by +29.9% at C4 and +18.3% at C6

**Proposed action:** (a) Backport #52228: acceptance_estimator.py, the dflash2/speculator.py hooks, and the speculative.py gate change. Then set enable_adaptive_verification true in the dflash2 speculative-config. (b) Interim, one knob: benchmark NUM_SPECULATIVE_TOKENS=3 (and 4) against 7 on both the prose and structured cells, and record it in decision.tsv as a workload-dependent default, not a universal keep.

**Est. impact:** Calibrated model: step = 51 + 8.5*(k+1) ms. Tokens/step = 1 + sum of cumulative conditional acceptance. Prose: k=7 gives 2.42/119 ms = 20.4 tok/s (measured 19.2-21.2); k=5 gives 23.7 (+16% predicted, +14% measured); k=3 gives 2.363/85 = 27.8 (+36%); k=2 gives 28.6. Structured: k=7 gives 7.84/119 = 65.9 (measured 67.6); k=3 gives 3.84/85 = 45 (-32%). Adaptive trimming should land near the per-workload optimum: about +30-40% prose c=1 (21 to about 28 tok/s) with structured unchanged. Estimate, not measured.

**Validation:** Before touching anything else, run bench_decode.py for prose and structured at c=1 and c=2, 3-run median, at k=7, 5, 4 and 3 on the current image. Fit step_ms = a + b*(k+1) from those runs. Then run the adaptive backport at k=7 and require prose to be at least the k=3 cell and structured to stay within 8% noise of k=7. Also confirm greedy count 200/200 is still lossless.

**Risks:** Adaptive verification with c>1 needs varlen verification. The SM12x DSA indexer is not varlen (v11 indexer.py:649-653 requires capability family 100 plus deep_gemm), so mixed widths fall back to the flattened or uniform path, and CUDA-graph capture sizes must still cover the trimmed widths. The estimator adds a small sampling cost. A fixed k=3 regresses structured by about 32%.

**Verifier reasoning:** Checked: v11src vllm/config/speculative.py:1177-1178 does gate adaptive verification to dspark. gh api pulls/52228 shows it merged 2026-09-14T22:58:49Z, 9 files, +900/-28, and it touches spec_decode/dflash2/speculator.py plus acceptance_estimator.py. The local PR #9 receipts (gh pr diff 9, iter-H1 bench.json) show k=5 prose c=1 at 21.97 with acceptance 2.31 against a same-night k=7 baseline of 19.23 with acceptance 2.29. The reviewer's arithmetic from the MiaAI per-position rates reproduces (tokens/step 2.421 at k=7, 2.420 at k=5, 2.363 at k=3). Part (a) has a blocker the reviewer missed, and it is not just a c>1 risk. v11 already ships the adaptive-verification framework (v1/worker/gpu/spec_decode/adaptive_verification.py). maybe_create_adaptive_verification_manager (lines 453-468) raises ValueError if any attention backend has supports_device_cpu_query_lens_mismatch()==False, or if min cudagraph support is not ALWAYS. GLM's kpool/DSA indexer backend is KpoolTailBackend, a subclass of DeepseekV32IndexerBackend (v11 indexer.py:130-139, 180). That method returns _supports_varlen_paged_mqa_logits(), which is capability family 100 plus DeepGEMM only (indexer.py:649-653). So on sm_121, enabling it fails at init even at c=1. Upstream v0.30.0 is the same: indexer.py:190-196 and 734-747 add only a Hopper (family 90) flattened path. Part (b) is contested by evidence the reviewer left out. Tony's README (lines 265-279) reports k=5 'losing on every single-stream prompt' and, on a second pair, 'flat on prose'. His per-position acceptance is nearly flat (0.93/.../0.94), which contradicts the 'collapses after slot 3' premise. Across sessions, our own k=5 rebench (PR #9 rebench-dflash5: prose c=1 21.76) is only +2.6% over the published k=7 21.2, inside the ±11% 3-run spread PR #9 itself records. Refitting the step model to our two same-night points (119 ms at k=7, 105 ms at k=5) gives b≈7 ms/token and a≈63 ms, not 8.5 and 51. k=5 also cost structured c=1 67.6→55.73 (-18%, rebench-dflash5).

**Verifier corrected claim:** Adaptive verification cannot be enabled for GLM-5.3 on GB10 in v11 or in v0.30.0. The DSA/kpool indexer backend reports no device/CPU query-length mismatch support off sm_100/sm_90, so AdaptiveVerificationManager raises at init. A backport of #52228 alone is useless without an SM12x varlen or flattened indexer path (for example, porting v0.30.0's Hopper _supports_flattened_device_query_lens approach). A fixed lower k is a valid single-knob experiment. The evidence on the c=1 prose gain is mixed: +14% same-night locally, +2.6% cross-session, and flat or negative in Tony's single-stream sweeps. The structured cost is measured at -18% for k=5.

**Verifier corrected impact:** Fixed k=3 or k=4: prose c=1 somewhere between 0 and +25% (refit model with a=63, b=7 and our lower acceptance of about 2.24 at k=3 gives about 24.6 tok/s against 19.23 same-night, but Tony's data suggests flat). Structured drops 18-32%. Adaptive trimming is not attainable on this hardware until the indexer is made varlen.

## EXT-2: Community lanes on the same GB10 pair decode prose 27-33 tok/s against our 21.2; the gap is step time (b12x MoE, smaller verify width), not acceptance

- kind=perf component=MoE backend / image impact=5 confidence=3 effort=M needs_gpu=True
- **verdict: refuted** (corrected confidence 3)

**Claim:** Several reproducible 2x Spark lanes beat this recipe's per-step time by 20-35% at equal acceptance. The most directly comparable one runs the same nvidia pin 09b04e5 with b12x MoE/linear and DFlash2 k=5.

**Mechanism:** Same hardware and similar acceptance but a shorter step means per-step cost is the issue. Candidate causes, in order of evidence: verify width (k=5 or MTP3 against k=7), the b12x fused MoE (one fused dispatch+GEMM+SwiGLU+reduce, FP4 tensor cores) against Marlin W4A16 dequant, and async scheduling. The bench methods differ (tg1024 against 200 tokens, different prose prompts), so absolute gaps are directional.

**Evidence:**
- 0rand, docs/FINDINGS.md, 2026-09-12/13: image pilcothink/vllm_spark_glm53:0.28 (supplies b12x MoE/linear), weights nvidia/GLM-5.3-Flash-NVFP4@09b04e5e, DFlash2 k=5, graphs <=16, async ON, GMU 0.88. Warmed spec-bench: structured 49.7 (alpha 88%), code 43.0 (71%), filler 33.5 (52%); c1 decode 32.5/33.1/33.1 at depth 0/2K/8K; hardmode 94/100 (https://github.com/0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks)
- MiaAI EXL3 4bpw with DFlash2 K7 on 2x Spark: prose c=1 27.1 tok/s at acceptance 0.341 / 2.39, which is 88 ms/step; ours is 21.2/2.43, or 115 ms/step (https://github.com/Kristianaaron/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
- sparkring R33 TP2 (NVFP4-Spark, B12X attention/MoE/linear, MTP3, RoCEnante, 2026-09-11): 13.1 target steps/s at c=1 (76 ms), C1 33.1/31.9/30.9 tok/s at 8K/16K/32K (performance/records/glm53-flash/r33-image020-tp2-sparkcache-20260911.md, marked research-only)
- Tony README: nvidia pack, DFlash2 k=7, marlin: prose 28.4, code 51.9, count 42.5, math 34.9 tok/s
- SGLang DFlash2 on 2x Spark (flashinfer_cutlass MoE, bf16 KV, 2026-08-29): prose 23.6, code 28.6 (https://huggingface.co/randomllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark); SGLang is not faster
- Local published: prose c=1 21.2 (acc 2.43), structured 67.6 (acc 7.84), LibertAI caca4e6, v11, marlin

**Proposed action:** Do not copy another stack wholesale. Take the pieces one knob at a time on this recipe's ruler: (1) k (see EXT-1); (2) MOE_BACKEND=b12x using the upstream B12xExperts path (see EXT-3), not flashinfer_b12x; (3) optionally, with evidence, revisit --async-scheduling, which 0rand runs with DFlash2 on the nvidia pack. As a reference point, boot the 0rand/pilcothink 0.28 profile on an exclusive slot and run bench_decode.py against it once. That separates the method gap from the stack gap.

**Est. impact:** Target envelope from the community data: prose c=1 27-33 tok/s (+27% to +56% over 21.2), a 76-88 ms step against 115 ms. Structured falls if k<7 (0rand 49.7 at k=5 against our 67.6 at k=7). Estimates only, because the benches are not identical.

**Validation:** Run python3 bench_decode.py (prose, c=1 and c=2, 3-run median) on (a) the current v11/marlin/k7, (b) v11/marlin/k5, (c) the pilcothink 0.28 profile with the same nvidia pin and chat_template.jinja. Record steps/s = tok_s / acceptance_len for each, to separate engine speed from acceptance.

**Risks:** The pilcothink image is third-party (supply chain, unknown patches). The b12x JIT can wedge on UMA OOM (eugr #404). 0rand's GMU gate behaviour differs between builds. Numbers are not portable across benches.

**Verifier reasoning:** The reviewer's key mechanism ('the gap is step time, not acceptance') fails on the most directly comparable lane. 0rand FINDINGS.md:12-28 runs the nvidia pin 09b04e5 with DFlash2 k=5 and reports filler 33.5 t/s at α=52%, code 43.0 at α=71%, and structured 49.7 at α=88%. Our bench uses vLLM's draft-acceptance convention (acceptance_len = 1 + k*rate; parity k=7: 1+7*0.18 ≈ 2.29). Under that convention 0rand's tokens/step are 3.6, 4.55 and 5.4, so steps/s are 9.3, 9.45 and 9.2, a step of about 106-108 ms. Our own same-image k=5 run (PR #9 iter-H1) was 21.97/2.31 = 9.5 steps/s, about 105 ms. The b12x MoE, async scheduling and the pilcothink image therefore show no step-time advantage over v11/Marlin at equal k. The throughput gap comes from acceptance: their filler prompt drafts at 52% against our prose at 18-26%. 0rand's own FINDINGS:163-164 warns that throughput is not comparable across stacks and that 'per-workload spec rates are the comparable numbers'. The MiaAI figure '0.341 / 2.39' is internally inconsistent: 1+7*0.341 = 3.39, not 2.39. The claimed 88 ms step therefore depends on which definition is right; at 3.39 it would be about 125 ms. Tony's prose 28.4 at k=7 comes with no acceptance number, so no step time can be derived from it.

**Verifier corrected claim:** On the same nvidia pin at k=5, the 0rand b12x/async lane runs about the same ~107 ms per verify step as our v11/Marlin k=5 run (~105 ms). Its higher tok/s comes from higher draft acceptance on its prompts, not faster steps. Other community numbers cannot be converted into step time without acceptance data. The lever these comparisons point to is acceptance (drafter revision and prompt mix; see EXT-7), not the MoE backend.

**Verifier corrected impact:** No step-time improvement demonstrated from copying the b12x/async stack. The 27-33 tok/s 'target envelope' reflects different prompts and acceptance, not engine speed.

## EXT-3: The b12x MoE clamp blocker is solved upstream: v0.30.0 B12xExperts (--moe-backend b12x) passes swiglu_limit for SiLU

- kind=perf component=fused MoE backend (Marlin to b12x) impact=4 confidence=3 effort=M needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The opt-b12x experiment died because FlashInfer's flashinfer_b12x path clamps only swigluoai_uninterleave. vLLM v0.29/0.30 has a second, native b12x MoE backend (fused_moe/b12x.py, PR #52018) built on the b12x pip package. For SiLU it forwards the checkpoint's swiglu_limit (10.0) into plan_execution. It supports NVFP4 in nvfp4 (W4A4, uses the checkpoint's input scales), w4a8_nvfp4 and w4a16 modes.

**Mechanism:** Marlin dequantizes FP4 to BF16 in registers and runs a BF16 MMA, which is compute and latency heavy on GB10's 48 SMs for small-M MoE. b12x fuses routing dispatch, both FP4 block-scaled GEMMs on native SM12x FP4 tensor cores, SwiGLU with the clamp, and the weighted reduce into planned kernels.

**Evidence:**
- v0.30.0 (ced6857) vllm/model_executor/layers/fused_moe/b12x.py:258-273: `limit = self.quant_config.gemm1_clamp_limit ... if limit is None: limit = self.moe_config.swiglu_limit; if activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE: return limit, None, None`; :86-96 passes swiglu_limit to fused_moe.plan_execution; :37-46 mode table (nvfp4,nvfp4)->nvfp4 and (nvfp4,None)->w4a16; :486-492 SILU supported; :495-503 no EP
- v0.30.0 flashinfer_b12x_moe.py:46-49 maps SILU to 'silu' with no clamp parameter (the path the b12x worktree tried)
- v11src has only fused_moe/experts/flashinfer_b12x_moe.py; there is no fused_moe/b12x.py or vllm/utils/b12x.py
- v0.30.0 docs/features/quantization/b12x.md: `uv pip install "vllm[b12x]"`, `--moe-backend b12x`, VLLM_B12X_MOE_FP4_FORCE_A16=1 forces BF16 activations; setup.py:1531 b12x==1.3.0
- PyPI b12x 1.3.0 (2026-08-15) requires nvidia-cutlass-dsl==4.6.2 (matches Dockerfile.sm121-v8 pin; v0.30.0 image pins 4.7.1); 'Kernels compile JIT on first use'
- nvidia checkpoint: 36,297 input_scale tensors (model.safetensors.index.json), so W4A4 mode has calibrated activation scales
- eugr/spark-vllm-docker recipes/glm-5.3-flash.yaml runs --moe-backend b12x --linear-backend b12x --attention-backend B12X for GLM-5.3 on 2x Spark

**Proposed action:** Build a v12 layer that backports fused_moe/b12x.py, utils/b12x.py and warmup/b12x_warmup.py from v0.30.0, pip-installs b12x==1.3.0 (keeping cutlass-dsl 4.6.2) and pre-warms all capture shapes at boot. A/B it one knob at a time: MOE_BACKEND=b12x with VLLM_B12X_MOE_FP4_FORCE_A16=1 (same W4A16 numerics as Marlin), then W4A4 (nvfp4) mode. Relax the run.sh MOE_BACKEND refuse-guard for 'b12x' only after an OOM-free boot and a greedy-count pass.

**Est. impact:** Unknown until measured. The MoE is about 60-70% of the step (routed reads about 63 ms of 115 ms per the bandwidth model). A 10-25% MoE kernel-efficiency gain would give +6-17% decode. The 0rand b12x lane reaches 32-33 tok/s c1 (confounded with k=5 and async).

**Validation:** Boot with FORCE_UNSAFE_MOE=1 MOE_BACKEND=b12x on an exclusive slot. Watch free -h through JIT and warmup, and abort if MemAvailable drops below 3 GiB. Pass greedy count 200/200 and the thinking-off smoke. Then run bench_decode.py prose and structured c=1/c=2 against Marlin. Check that swiglu_limit is logged or asserted as 10.0 in the plan.

**Risks:** CuTeDSL JIT at inference can wedge the engine under UMA pressure (eugr/spark-vllm-docker #404, NV_ERR_NO_MEMORY, 2x GB10 GLM-5.3). No EP support. W4A4 mode changes numerics against Marlin's W4A16, so run the quality probes. The cutlass-dsl pin conflicts with a v0.30.0 base. AGENTS.md refuse-guard exception needed.

**Verifier reasoning:** I confirmed the vLLM-side code claim against the real tag: gh api contents fused_moe/b12x.py?ref=v0.30.0 returns blob sha d31ba1489d20…, which matches git hash-object of the scratch copy. In that file, _B12X_MOE_MODES maps (nvfp4,nvfp4) to nvfp4 and (nvfp4,None) to w4a16. _swiglu_params returns (limit, None, None) for SILU, with limit taken from gemm1_clamp_limit or moe_config.swiglu_limit, and _b12x_moe_execution_plan passes swiglu_limit into fused_moe.plan_execution. _supports_parallel_config rejects EP. v11src has only fused_moe/experts/flashinfer_b12x_moe.py (find). Dockerfile.sm121-v8:68-70 pins nvidia-cutlass-dsl==4.6.2. Not verified: that the b12x 1.3.0 pip kernel actually applies the clamp in silu mode, since that code lives outside vLLM. For reference, v11 Marlin does clamp (fused_moe/activation.py:161-190 swiglu_limit_func). The impact claim is not supported: the only same-pin b12x lane (0rand) shows no step-time gain over our Marlin at k=5 (see EXT-2).

**Verifier corrected impact:** Unknown, likely small at c=1. The one same-pin b12x datapoint (0rand, k=5) implies a ~107 ms step against our Marlin ~105 ms. W4A4 mode also changes numerics. Treat it as a quality and robustness experiment with a UMA-JIT risk, not a primary speed lever.

## EXT-4: About 7.8 GB per rank of BF16 attention, shared-expert and lm_head weights is read every step; FP8 or MXFP8 packs exist but need a quality gate

- kind=perf component=checkpoint quantization (non-expert weights) impact=4 confidence=3 effort=L needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** From the safetensors headers, the nvidia pack keeps 11.31 GiB of attention, 1.97 GiB of shared experts and 1.18 GiB of lm_head in BF16. At TP2 that is 7.76 GB/rank read on every decode step regardless of verify width, which is 28-32 ms of the roughly 115 ms step. Community packs quantize these. Z.ai's own FP8 release deliberately keeps KDA attention in BF16 and quantizes shared experts to FP8.

**Mechanism:** Decode on GB10 is LPDDR5X-bandwidth-bound. Halving the bytes of per-step-constant weights shortens every step: 7.76 GB/rank to about 3.9 GB saves about 14 ms at 273 GB/s (about 16 ms at the measured 241 GB/s).

**Evidence:**
- Header scan (read-only, first 8+N bytes) of nvidia 09b04e5: routed_experts 159.469 GiB (U8 141.75 + F8 17.72), attn BF16 11.308, mtp_layer45 BF16 13.844, shared_experts BF16 1.969, lm_head BF16 1.182, dense_mlp_L0-2 NVFP4 0.237, visual BF16 1.050, total 190.38 GiB
- LibertAI caca4e6 header scan: attn BF16 11.314, dense MLP BF16 0.844, MTP NVFP4 4.141
- local-inference-lab/GLM-5.3-Flash-NVFP4-Spark (2026-09-17): MXFP8 attention and shared experts plus NVFP4 routed experts, about 165.5 GB (hf_quant_config); used by the eugr and sparkring lanes
- Tony docs/EXPERIMENTAL-NVFP4-ATTENTION.md: NVFP4 attention and MLP projections measured x1.14-1.30 at TP4 (2026-09-20), predicted x1.21 at TP2, frees 5.21 GiB/rank; needs patches at kda.py:172 (quant_config=None) and model.py:331
- tacos4me/GLM-5.3-Flash-NVFP4-FP8ATTN-512K: teacher-forced top-1 95.57% against the NVFP4 parent (n=271)
- Void-Z/GLM-5.3-Flash-NVFP4-FP8-Hybrid (nvidia experts plus FP8 attention/shared): GSM8K 96.06 against 95.00, IFBench loose 80.00 to 77.62
- zai-org/GLM-5.3-Flash config.json quantization_config (fp8, block 128x128): modules_to_not_convert includes every self_attn KDA projection (q/k/v/o_proj, f_a/f_b/g_a/g_b, b_proj) but not shared_experts

**Proposed action:** Tier 1, sanctioned by the vendor's own FP8 release: graft Z.ai's official FP8 (block 128) shared_experts onto the nvidia pack, saving 0.49 GiB/rank. Tier 2, experiment only, behind a quality gate: FP8 block-128 KDA and MLA projections (not the indexer, router, f_b/g_b gates, embed or lm_head), which needs the kda.py and model.py quant_config patches. Do not use NVFP4 attention; Tony found ModelOpt attention quantization correlated with token corruption. Gate on KLD/top-1 against the BF16 reference, IFBench, a tool-call JSON validity check, and a non-Latin UTF-8 probe.

**Est. impact:** Tier 1: about 0.53 GB/rank, about 2 ms/step, +1.7%. Tier 2 (FP8 attention and shared): about 3.6 GB/rank, 13-15 ms/step, about +12-14% decode at any k, and frees about 3.3 GiB/rank of UMA, which helps the vision headroom (EXT-6). Combined with k=3 the model predicts 2.363/(51-15+34) = about 33.8 tok/s prose. Estimates.

**Validation:** First run offline CPU KLD/top-1 on a small prompt set is not feasible at this size, so use a GPU slot. Measure a teacher-forced top-1 agreement delta against the unmodified nvidia pack on 300+ positions, IFBench subset, tool-call probes and a UTF-8 probe. Then run bench_decode.py prose/structured. Keep only if top-1 is at least 97% and IFBench is within noise.

**Risks:** Vendor precision choice is BF16 for KDA attention, and there are documented quality regressions (IFBench -2.4 points). Needs model-code patches because glm5next hardcodes BF16 for these projections. Needs an FP8 linear kernel that works on sm_121 (the v11 _C is built without 12.1a). The abliteration interaction is unknown.

**Verifier reasoning:** A header re-scan (my own hdr_ext.py, headers only) of nvidia 09b04e5 gives attn 11.289 GiB (BF16 11.275), shared 1.969, lm_head 1.182, routed 159.469 (U8 141.75 + F8 17.719), mtp45 13.844 BF16, visual 1.05, total 190.38. That matches within about 0.02 GiB. kda.py:168-174 strips quant_config and model.py:331 sets quant_config=None for MLA, as the reviewer said. Two misses. (1) The fork already has _try_load_fp8_attn_proj (model.py:1203-1218), which dequantizes FP8 q_a/kv_a/o_proj to BF16 at load. Dropping in any FP8-attention community pack (Void-Z, tacos) would give zero bandwidth saving; the model needs real FP8 linear layers, not just the patches listed. (2) Several attention linears are ReplicatedLinear (attention.py:250 wq_b, :464 kv_a_proj_with_mqa), so per-rank BF16 bytes exceed the halved 7.76 GB and the saving is somewhat larger per rank than computed. Tier 1 (+1.7%, about 2 ms) sits well inside the ±8-11% bench noise recorded in PR #9, so the current ruler cannot measure it. Tier 2 numbers are a bandwidth estimate only.

**Verifier corrected claim:** About 7.8+ GB/rank of BF16 attention, shared-expert and lm_head weights is read every step (more than half of 15.5 GB because some projections are replicated). This fork upcasts FP8 attention projections to BF16 on load, so an FP8 pack only helps after model-code changes that keep FP8 weights and use an sm_121-capable FP8 GEMM.

**Verifier corrected impact:** Tier 1 about 2 ms/step, not measurable with the current 3-run ruler. Tier 2 is 10-15 ms/step by the bandwidth model only after code changes; unmeasured, with quality risk.

## EXT-5: DFlash2 on the v11 draft-KV-group design very likely gets zero prefix-cache hits

- kind=correctness component=vllm/v1/core/kv_cache_coordinator.py + v11 draft KV group patch impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** v11 adds the DFlash2 drafter as its own sliding-window KV group. No group carries is_eagle_group (only DeepseekV4 is annotated), so the coordinator falls back to flagging every group as EAGLE. Two independent 2x-Spark DFlash2 lanes measured zero hits. The README's prefix-cache evidence was measured under MTP-4, not DFlash2.

**Mechanism:** The draft group's shorter sliding-window hit length and the EAGLE last-block drop are applied to all groups. With --block-size 2304 the eagle drop removes one 2304-token block, and the draft group's reduced hit then zeroes the target's hit.

**Evidence:**
- v11src vllm/v1/core/kv_cache_coordinator.py:107-109: `if use_eagle and not self.eagle_group_ids: self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))`; config/speculative.py:1517 use_eagle() includes 'dflash'
- v11src kv_cache_utils.py:2046-2066: only _annotate_eagle_groups_deepseek_v4 sets is_eagle_group
- docker/patch_v11_dflash_kv_groups.py:1-15, 86-92 (separate SlidingWindowSpec draft group)
- Tony PR #18 (merged 2026-09-16): fixes 'Zero-hit prefix cache on the TP2 DFlash2 recipe', flags only the draft group and stops it reducing the target hit length; afterwards hit rate 0.986, warm TTFT p95 2.98 s. Issue #21: '0 hits / 35,280 queries' before. PR #22: 13K-token repeat TTFT 21.1 s to 6.3 s
- gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile issue #1 (2026-09-20): '51,016 total queries but 0 cache hits' with DFlash2 on GLM-5.3 (eagle block drop at 4608-token blocks); upstream vLLM PR #54163 still OPEN
- v0.30.0 kv_cache_coordinator.py:110-112 has the same fallback (not fixed upstream)
- README.md:22 prefix-cache hit (4608 cached tokens) measured on MTP-4 eager

**Proposed action:** Verify first: repeat a block-aligned 5-13K-token prompt 3 times and read vllm:prefix_cache_hits. If zero, port Tony's patch_prefix_cache_draft_group.py (two exact-string edits to kv_cache_coordinator.py plus exact-type SlidingWindowSpec matching) as an image layer. Also track PR #54163 for the eagle-block-drop part.

**Est. impact:** Decode unchanged. Multi-turn and agent TTFT on repeated prefixes: Tony measured 21.1 s to 6.3 s (-70%) on a 13K repeat and a 6.1x warm re-prefill. At the recipe's measured 1425 tok/s prefill, a 10K shared prefix saves about 7 s per turn.

**Validation:** With SPEC=dflash2, send an identical 9216-token (4 blocks) prompt 3 times, temp 0. Expect hits of 0 before the patch and at least 6912 after. Greedy output must stay byte-identical, and count 200/200 must hold.

**Risks:** Coordinator changes can corrupt KV reuse if the draft group's windows are mis-handled. Validate with needle and greedy parity. Tony's patch was verified on his v11, not ours (same design, unverified identity).

**Verifier reasoning:** Confirmed in code: v11 kv_cache_coordinator.py:104-109 falls back to flagging all groups as EAGLE when none is flagged, and use_eagle() (speculative.py:1517) includes 'dflash'. patch_v11_dflash_kv_groups.py adds a SlidingWindowSpec draft group at block size attn_block//4 = 576. The fixed-point loop (lines 719-852) applies drop_eagle_block per group and clears eagle_verified when the length shrinks. A ratchet to zero is plausible but not derivable statically, and there is no local DFlash2 prefix-hit measurement. The README:22 evidence (4608 cached) is MTP-4 eager, as the reviewer says. Tony's README (line 281) confirms a draft-group prefix fix exists (#13, not only #18). gh shows #54163 still OPEN. The validation threshold is wrong. Under MTP-4, a 10,271-token prompt hit only 4608 tokens (two blocks). A 9216-token prompt has max_cache_hit_length 9215, which is three full blocks minus the eagle drop, so a healthy post-fix result would be about 4608, not ≥6912.

**Verifier corrected claim:** The v11 DFlash2 draft group plus the all-groups EAGLE fallback may zero prefix hits. This is unmeasured locally, and the README evidence is MTP-only. Verify first; if confirmed, port the draft-group-only EAGLE flag.

**Verifier corrected impact:** Decode unchanged. TTFT savings on repeated long prefixes only if zero hits is confirmed. The post-fix expectation for a 9216-token repeat is about 4608 cached tokens (the MTP-4 baseline pattern), not ≥6912.

## EXT-6: Vision-on default (PR #11) has reported front-end memory costs of about 15.7 GiB on 2x Spark; the node is already at about 115/121 GiB

- kind=ops component=run.sh multimodal args / UMA budget impact=4 confidence=3 effort=S needs_gpu=True
- **verdict: refuted** (corrected confidence 4)

**Claim:** Several independent 2x Spark GLM-5.3 reports say enabling the multimodal front-end costs about 15.7 GiB. Mitigations used were a lower KV pin, skip-mm-profiling (forbidden here by AGENTS.md), fewer images and video:0. PR #11 enables image:4,video:1 with the default 4 GiB mm processor cache per process and no measured memory delta.

**Mechanism:** The multimodal processor cache, video frame decoding (ffmpeg or torchcodec), and MM encoder profiling allocate host RAM on rank0. On UMA that is the same pool as weights and KV, so it can push rank0 into swap (a decode slowdown) or OOM.

**Evidence:**
- NVIDIA forum 381541 (Ama5u, 2026-09-01): 'The multimodal processor adds ~15.7 GiB to the API front-end'; KV pin 10 to 9 GiB with vision; decode unchanged
- DevelopersIO (2026-09-01/02): '--language-model-only is for skipping the multimodal front-end load, and without it you consume an extra 15.7 GiB'
- Tony README: '--limit-mm-per-prompt {"image":2,"video":0}' boot default; forum 381429: vision (image:4) cuts max ctx 262,144 to 244,224; chat_template_mm.jinja needed or image requests 500
- run.sh:62-63 default LIMIT_MM_PER_PROMPT='{"image":4,"video":1}'; run.sh:303-307
- v11src vllm/config/multimodal.py:152 mm_processor_cache_gb default 4; :179 mm_encoder_tp_mode 'weights'
- Task facts: about 115/121 GiB used while serving, worker had about 1.3 GiB in swap; KV 5.0 GiB pin slowed decode about 20%, 5.14 GiB crashed
- 0rand FINDINGS §3: VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 returned 2.11 GiB (their GMU/pool setup)

**Proposed action:** Before merging PR #11 as default: measure free -h and per-process RSS on the head with LANGUAGE_MODEL_ONLY=1 against 0 at idle and after smoke_vision.py. Default to --mm-processor-cache-gb 1 (or 0), set video:0 unless a video use case exists, and keep image<=4. Consider mm_encoder_tp_mode 'data' only if the encoder weights are not the issue. Add a UMA refuse-guard (PR #12) that accounts for the measured vision delta.

**Est. impact:** Risk avoidance: prevents rank0 swap, which costs about 20% decode per the KV 5.0 evidence, and OOM. --mm-processor-cache-gb 1 caps up to 3 GiB/process that would otherwise be allowed (about 6 GiB across the API server and engine core). Quantity unverified locally.

**Validation:** Exclusive slot: boot vision on and off, record free -h and swap on both nodes at ready, after 4-image smoke, after bench_decode.py. Require prose c=1 with vision on within noise of vision off and no swap growth.

**Risks:** The 15.7 GiB figure may be image- or version-specific and includes profiling buffers (it may be lower with the kv pin). Reducing the processor cache increases re-processing of repeated images.

**Verifier reasoning:** The premise that PR #11 newly enables vision on a node already at about 115/121 GiB is wrong. The multimodal front-end and encoder were already active in the published measurements. evidence/rebench-20260902T204243Z/engine.log.tail:57 (and engine-rank1.log.tail:48) logs 'Encoder cache will be initialized with a budget of 32242 tokens, and profiled with 1 video items of the maximum feature size'. The LibertAI caca4e6 pack is Glm5NextForConditionalGeneration with vision_config, processor_config.json and 347 model.visual.* tensors. Before PR #11, run.sh passed no --language-model-only or --limit-mm-per-prompt (git grep on 63a433a: no hits), and the class is registered with MULTIMODAL_REGISTRY (v11 glm5next model.py:1025-1033). The 115/121 GiB, swap and decode figures therefore already include the MM front-end at default limits. PR #11 (git show 29954c8) only makes vision explicit and adds a cap of image:4, video:1, which can only lower the profiling budget. mm_processor_cache_gb default 4 (multimodal.py:152) is real but was already present. The 15.7 GiB figure comes from other stacks and was not reproduced here.

**Verifier corrected claim:** Vision was already on (implicitly, with default MM limits) in every published LibertAI measurement. PR #11 adds a cap and does not add a new memory cost. Optional hardening: video:0 to shrink the encoder profiling budget (currently profiled with one max-size video item), and a smaller mm_processor_cache_gb, measured with free -h. A related real issue: the DFlash2 drafter receives no multimodal embeddings (engine.log.tail:51), so acceptance on image prompts will be lower.

**Verifier corrected impact:** Low. Possible small UMA headroom from video:0 or a smaller processor cache; unmeasured.

## EXT-7: Local DFlash2 drafter is the stale initial release; two retrained checkpoints followed

- kind=quality component=draft model pin (recipe.yaml:13) impact=3 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** recipe.yaml pins incoai/GLM-5.3-Flash-DFlash2 @7d74cdd (the 2026-08-27 'Release'). The repo has since pushed 'Checkpoint update' dc77ff1 (2026-08-28) and bf582e4 (2026-08-31), each with different model.safetensors weights and an unchanged config. The model card has withdrawn its old benchmark table and says the numbers are being re-measured.

**Mechanism:** Acceptance length multiplies tok/s directly, since steps/s is roughly constant at a given k. A retrained drafter with better prose alignment raises prose throughput without touching the target's quality, because verification stays lossless.

**Evidence:**
- recipe.yaml:13 `draft: incoai/GLM-5.3-Flash-DFlash2 @ 7d74cdd881ed7e32c31175984a67823127b66cfe`; the local HF cache has only this snapshot (refs/main dated Aug 28 08:59)
- https://huggingface.co/api/models/incoai/GLM-5.3-Flash-DFlash2/commits/main: bf582e4 2026-08-31T00:44:54Z 'Checkpoint update'; dc77ff1 2026-08-28T21:37:58Z 'Checkpoint update'; 7d74cdd 2026-08-27T23:02:24Z 'Release'
- LFS sha256: 7d74cdd 8931dc52…, dc77ff1 b33c0347…, bf582e4 b038e1d9…; config.json oid 083085a… identical in all three; size 2,342,169,800 B
- Old README (7d74cdd): acceptance vs MTP GSM8K 5.78, MATH-500 5.86, HumanEval 5.32, MBPP 4.85, MT-Bench 4.03; current README: 'Benchmark numbers for this checkpoint are being re-measured'
- canada-quant DFlash2-F/G cards (2026-09-25): incoai reference mean acceptance 3.632 at K=7 (B300, c16); DFlash2-G 3.676 (+0.044) but 1.84B params / 6.2 GB
- local-inference-lab/GLM-5.3-Flash-DFlash2 (MXFP8 of dc77ff1, lm_head excluded, 2026-09-16)

**Proposed action:** A/B one knob: re-download at revision bf582e4 and bench prose and structured at k=7 (and the k chosen in EXT-1). Keep the pin that wins on the prose c=1 acceptance_len. Then optionally try the MXFP8 drafter (halves drafter bytes). Do not adopt canada-quant G: +1% acceptance for about 2.6x the drafter bytes is net negative on a bandwidth-bound GB10.

**Est. impact:** Unknown (the changelog does not state what changed). Every +0.1 prose acceptance_len is about +4% prose tok/s at 2.43. The MXFP8 drafter saves about 0.59 GB/rank per step, about 2 ms (+2%), if the draft is TP-sharded.

**Validation:** bench_decode.py prose/structured c=1, c=2, 3-run median at each drafter revision. Compare the acceptance_len fields and require greedy lossless count 200/200.

**Risks:** Gated repo (CC BY-NC-ND): licence terms are unchanged. The update might target SGLang capture semantics; our v10 capture follows SGLang #36708 (v11src glm5next model.py:743-767), so the risk is low but unverified.

**Verifier reasoning:** The HF API commits list shows bf582e4 (2026-08-31 'Checkpoint update'), dc77ff1 (2026-08-28 'Checkpoint update') and 7d74cdd (2026-08-27 'Release'). The local snapshot model.safetensors symlinks to blob 8931dc522be0…, the initial release, with config blob 083085a…. run.sh:52 hard-codes the 7d74cdd snapshot path in the generated block, recipe.yaml:13. The reviewer understated the impact. Tony's README:292-299 says the drafter 'shipped different bytes under the same tag (sha256 b33c0347 vs 8931dc52)', and 'two pairs on the same recipe measured 0.73 acceptance and one measured 0.35, with that hash mismatch sitting right there'. Our prose draft-acceptance rate is 0.18-0.26 (parity 2.29 at k=7), well below Tony's prose ~0.33 (README:191) and 0rand's filler 0.52. Acceptance is the one lever where EXT-2's cross-stack gap actually shows up.

**Verifier corrected impact:** Potentially the largest cheap lever. At a roughly constant ~9 steps/s, prose tok/s scales with acceptance_len: going from 2.3 to 3.0 would be about +30%. Unknown until an A/B at bf582e4 (and dc77ff1). recipe.yaml's draft pin and the generated DRAFT_SNAPSHOT path both need changing (head and worker).

## EXT-8: SPEC=mtp rollback cannot work on the nvidia pack: its MTP layer is 13.84 GiB BF16 and outside the quant ignore list

- kind=correctness component=run.sh SPEC switch / AGENTS.md rollback claim impact=3 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** AGENTS.md says `SPEC=mtp` rolls back to MTP-4. On nvidia 09b04e5 the MTP layer 45 is entirely BF16 (13.84 GiB, including 288 BF16 experts), while hf_quant_config's exclude list stops at layer 44. The MTP MoE is therefore declared NVFP4 but stored in BF16, and no sm121 MoE backend serves an NVFP4 target plus an unquantized draft MoE. Even if it loaded, it adds about 6.9 GiB/rank against about 2.1 GiB for LibertAI.

**Mechanism:** vLLM builds the MTP layer's FusedMoE with the checkpoint's NVFP4 quant method (layer 45 is not excluded) and then tries to load BF16 expert tensors. Marlin rejects unquantized MoE.

**Evidence:**
- Header scan: nvidia mtp_layer45 BF16 13.844 GiB; LibertAI caca4e6 mtp_layer45 4.141 GiB (U8 3.375 + F8 0.422 + BF16 0.344)
- nvidia hf_quant_config.json exclude_modules covers layers 0-44 only (self_attn, mlp.gate, shared_experts) plus lm_head, embed_tokens, model.visual*
- Tony README (2026-09-24): 'MTP does not work on the nvidia build… marlin refuses unquantized MoE, triton refuses NVFP4, flashinfer_trtllm is sm100-only, flashinfer_cutlass cannot JIT'
- stihahi/GLM-5.3-Flash-NVFP4-mtp-nvfp4-delta (2026-09-22): re-quantizes 864 MTP expert tensors to NVFP4; per-rank weights 95.93 to 91.08 GiB; MTP-2 on 2x Spark 27.4 tok/s c=1, acceptance 61-69%
- 0rand .env.sample: 'The official NVFP4 checkpoint ships NO MTP heads' (usable MTP), so a separate draft model is mandatory

**Proposed action:** In run.sh, refuse SPEC=mtp when MODEL=nvidia/GLM-5.3-Flash-NVFP4, unless an explicit MTP delta (e.g. stihahi's NVFP4 MTP overlay) is applied, or require the LibertAI rollback pin together with SPEC=mtp. Fix the AGENTS.md and README rollback text accordingly. Edit recipe.yaml and render; do not hand-edit generated blocks.

**Est. impact:** Prevents a failed or OOM rollback boot. If the stihahi delta is adopted, it enables an MTP lane measured at 27.4 tok/s c=1 on 2x Spark (bench unknown).

**Validation:** CPU-only: header scan (done) plus a run.sh unit test that SPEC=mtp with the nvidia MODEL exits non-zero with a clear message. GPU confirmation is optional: boot SPEC=mtp on an exclusive slot and expect a load failure.

**Risks:** The stihahi delta is an unofficial derivative (9.5% relative dequant error on MTP experts), which affects only draft acceptance, not target quality.

**Verifier reasoning:** The header scan shows mtp45 at 13.844 GiB, all BF16, with 864 BF16 expert tensors (e.g. layers.45.mlp.experts.139.down_proj.weight BF16 [4096,2048]). hf_quant_config.json exclude_modules (132 entries) stop at layer 44 plus lm_head, embed_tokens and model.visual*; layer 45 is not excluded, and config.json quantization_config.ignore matches. v11 glm5next mtp.py:44 builds the MTP decoder layer with vllm_config.quant_config, so the MoE is constructed as NVFP4 and fed BF16 tensors. Tony's README:29 independently reports that MTP does not work on the nvidia build. Supporting local evidence: PR #9 H2-20260903 booted SPEC=mtp on the LibertAI pin and measured prose c=1 19.22 (probes pass). That confirms the fix of requiring the LibertAI pin with SPEC=mtp. The rollback text 'SPEC=mtp rolls back to MTP-4' (AGENTS.md, README) is wrong for the default MODEL. CPU-only verification is sufficient for the refuse-guard.

## EXT-9: vLLM v0.30.0 image supports glm5_next natively but is not a drop-in; rebasing still needs about 4 patches plus b12x

- kind=ops component=image chain (v8 to v11) against upstream v0.30.0 impact=3 confidence=4 effort=L needs_gpu=True
- **verdict: plausible** (corrected confidence 4)

**Claim:** vllm/vllm-openai:v0.30.0-aarch64 (ced6857, 2026-09-21) contains vllm/models/glm5next/*, the dflash2 speculator, b12x MoE/linear/attention for SM120/121, FlashKDA prefill, EPLB, and generalized adaptive verification. It still lacks four things: GLM-5.3 DFlash2 aux-hidden capture (PR #55682 open, #55423 closed, #56983 open), an SM121 NoPE sparse-MLA decode path (FLASHINFER_MLA_SPARSE_SM90 gated to capability.major==9; the SM120 sparse impl requires fp8_ds_mla), the DFlash prefix-cache fix, and the b12x pip package (optional extra). It still builds _C without 12.1.

**Mechanism:** Moving to v0.30.0 would bring upstream fixes (FI 0.6.18 final SM12x MoE sync, jit-cache that removes most runtime JIT and cudafe++ storms, native adaptive verification, b12x backends) while replacing the fork-only glm5next snapshot.

**Evidence:**
- docker image inspect vllm/vllm-openai:v0.30.0-aarch64: VLLM_BUILD_COMMIT=ced6857afa0e…, created 2026-09-21T23:15:58Z, TORCH_CUDA_ARCH_LIST='8.0 8.7 8.9 9.0 10.0 11.0 12.0', FLASHINFER_VERSION=0.6.18.post1 + flashinfer-jit-cache, NCCL_VERSION=2.30.7
- GitHub tree @ced6857: vllm/models/glm5next/nvidia/{model,attention,kda,mtp,multimodal}.py, vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py, fused_moe/b12x.py, v1/attention/backends/b12x.py (supports (12,0),(12,1))
- v0.30.0 glm5next/nvidia/model.py: no aux_hidden/SupportsEagle3 (grep), whereas v11src model.py:580,713-767 has them (v10 patch)
- v0.30.0 flashinfer_mla_sparse_sm90.py:106-107 `return capability.major == 9`; flashinfer_mla_sparse_sm120.py:63 requires kv_cache_dtype fp8_ds_mla
- vLLM PR #53906 '[Model] add GLM-5.3-Flash support' merged 2026-09-03 (after the v0.29.0 cut); v0.30.0 notes: EPLB #55119, FlashKDA #55737, NoPE dense/masked-MHA sparse prefill #55738, adaptive verification #52228
- requirements/cuda.txt @ced6857: flashinfer-python==0.6.18.post1, nvidia-cutlass-dsl[cu13]==4.7.1, torch==2.13.0; setup.py:1531 b12x extra b12x==1.3.0
- FlashInfer v0.6.18 (2026-08-29): 'sync SM12x NVFP4 fused-MoE kernels to b12x HEAD', 'SM12x W4A16 fused MoE family', 'm=1 stream-GEMV decode tactics for mm_bf16_fp4'; cu130 aarch64 jit-cache ships SM120-family cubins (sm_121a JIT at runtime); v0.7.0 released 2026-09-22
- r0b0tlab/glm53-flash-nvfp4-sm121 serves the nvidia pack on vLLM 0.28.1rc1.dev580 + FI 0.6.18 with exactly 4 patches (SM90 NoPE on SM121, persistent_topk SM gate, DFlash2 aux capture, drafter KV group)

**Proposed action:** Plan a v0.30.0-based lane (e.g. glm53-sm121-v20) as a separate track, not a PR #11 change. Port the v7 topk init, the v8 SM90 NoPE gate and PDL gate (re-check whether PDL is still a problem on 0.30), the v10 aux capture (or cherry-pick #56983) and the v11 draft group plus the prefix fix. Install b12x==1.3.0 with a cutlass-dsl pin check. Compare on the bench ruler against v11 before switching the default.

**Est. impact:** Indirect. It enables EXT-1 and EXT-3 without a hand backport and cuts first-boot JIT time (Tony: boot 41 to 16-19 min after cache persistence). Decode delta unknown until measured.

**Validation:** Build without GPU, then on an exclusive slot run render check, smoke_vision.py, greedy count 200, thinking-off smoke and bench_decode.py prose c=1/2 against v11. Keep only if it is not worse than v11 on every cell.

**Risks:** The cutlass-dsl 4.6.2 (b12x) against 4.7.1 (v0.30.0 FA4/quack) conflict. MRV2 default changes. The 'all' Mamba cache mode is deprecated (#55041) and may affect KDA state handling. Rebase churn on fork-only glm5next code.

**Verifier reasoning:** I verified against fetched v0.30.0 files (b12x.py matches the tag's blob sha). fi_mla_sparse_sm90.py:107 has 'return capability.major == 9'. fi_mla_sparse_sm120.py:63-66 requires fp8_ds_mla. v030 glm5next model.py has no aux_hidden or SupportsEagle3 (grep empty). v030 kv_cache_coordinator.py:112 keeps the all-groups fallback. #54163 is open (gh). One stated benefit is wrong: 'native adaptive verification' does not help on sm_121, because the v0.30.0 indexer (indexer.py:190-196, 734-747) supports query-length mismatch only on family 100 (varlen) or 90 (flattened). The rebase would not unlock EXT-1(a). b12x MoE brings no demonstrated step-time gain (EXT-2).

**Verifier corrected claim:** v0.30.0 supports glm5_next and ships b12x, but on sm_121 it still needs the NoPE SM90-on-SM121 gate, DFlash2 aux capture, the draft KV group and a prefix fix. Its adaptive verification cannot run with the GLM DSA indexer on SM12x. The rebase mainly buys jit-cache and boot-time improvements, plus a supported b12x path.

**Verifier corrected impact:** Indirect and mostly operational (boot/JIT time, upstream maintenance). No decode gain expected from adaptive verification on this hardware.

## EXT-10: Checkpoint quality facts: nvidia pack is W4A4-calibrated and run as W4A16 by Marlin; ModelOpt corruption reports need a local UTF-8 probe

- kind=quality component=checkpoint choice / quality probes impact=3 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The nvidia pack stores W4A4 NVFP4 (36,297 input_scale tensors). Attention, shared experts, router, lm_head, embed and visual are excluded, the dense MLP in layers 0-2 is NVFP4, and the MTP layer is BF16. Marlin ignores the input scales, so serving is W4A16, which is at least as precise as NVIDIA's measured W4A4 (GPQA 0.9211, MMMU-Pro 0.763, IFBench 0.6054 against BF16 0.9217/0.7688/0.613). vLLM issue #54150 (open) reports invalid UTF-8 token IDs from ModelOpt NVFP4 checkpoints (LibertAI, dealignai) on SM120 but not from compressed-tensors; Tony reports 0/0/0 on the nvidia pack. The published decode cells are on LibertAI.

**Mechanism:** Corrupted token IDs inside tool-call blocks desync the glm47 parser and can cause repetition lock. PTQ W4A4 activation error is avoided by W4A16 execution. QAD can recover PTQ loss.

**Evidence:**
- nvidia hf_quant_config.json: quant_algo NVFP4, group_size 16, kv_cache_quant_algo FP8, ModelOpt 0.47.0.dev393; model.safetensors.index.json: 36,297 input_scale, 36,297 weight_scale_2, 0 k/v scale tensors
- LibertAI caca4e6 config.json quantization_config: modelopt 0.45.0, weights 4-bit group 16, input_activations None (weight-only); ignore list includes all self_attn projections, mlp.gate and shared_experts. Header scan: attention 11.31 GiB BF16. This contradicts Tony's statement that LibertAI 'quantizes attention'
- https://github.com/vllm-project/vllm/issues/54150 (opened 2026-08-28, open): U+FFFD / invalid UTF-8 token IDs on Korean prompts; root cause unresolved (weights or ModelOpt loader path)
- Tony README 2026-09-24: nvidia pack 0 / 0 / 0 corruption (issue #23); LibertAI 'intermittent corrupted token IDs' breaking tool-call parsing
- DevelopersIO: Japanese character corruption with a ModelOpt checkpoint eliminated by switching to RedHat
- nvidia model card accuracy table (GB200)
- local-inference-lab/GLM-5.3-Flash-NVFP4 is a QAD checkpoint (distilled against a BF16 teacher, ~200M tokens, BF16 attention, MXFP8 MTP experts, about 199 GB) with no published accuracy numbers

**Proposed action:** Add two probes to the verify set (CPU-side scoring, GPU serve): (1) UTF-8 validity over about 50 Korean, Japanese and Chinese prompts (count U+FFFD and invalid byte tokens); (2) tool-call JSON parse rate over about 50 glm47 tool prompts. Run on both the nvidia and LibertAI pins, and only then re-publish decode cells on nvidia. Keep QAD as a later quality A/B after accuracy numbers exist.

**Est. impact:** Quality risk reduction; decides whether the LibertAI rollback pin is safe for agent/tool traffic. No decode change.

**Validation:** Probe counts: nvidia expected 0 invalid-UTF-8 tokens and at least 98% tool JSON parse. If LibertAI shows corruption, keep it as the rollback only with a warning.

**Risks:** Corruption is intermittent (4-9 per 3 runs in Tony's data), so a small sample may miss it. Use at least 3 runs at temp 0 and temp 1.

**Verifier reasoning:** nvidia index counts 36,297 input_scale, 36,297 weight_scale_2 and 0 k/v_scale tensors; hf_quant_config has quant_algo NVFP4, group 16, kv FP8; config.json input_activations is 4-bit. gh confirms vLLM #54150 is open (2026-08-28, ModelOpt NVFP4 invalid UTF-8 on SM120). trail.tsv (h-snap row) records 'marlin ignores input_scale', so serving is W4A16. One minor correction: the LibertAI caca4e6 config is weight-only (input_activations None), but the snapshot does ship model-input-scales.safetensors, and its index has 37,152 input_scale entries. It is not scale-free; Marlin simply ignores them.

## EXT-11: Long-context and concurrency caveats on SM121: indexer not varlen, persistent_topk SMEM limit, indexer logits churn

- kind=ops component=DSA indexer (sparse_attn_indexer_kpool / mla/indexer.py) impact=3 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** On SM12x the DSA indexer decode path has no varlen support, so each draft token of each sequence pays a full-context indexer row. Two concurrent long-context requests collapse decode (Tony #14: about 2-4 tok/s aggregate at 25-114K). Other GB10 reports: persistent_topk needs 128 KB SMEM against about 101 KB available (crash after about 300K tokens), and 230K prefill logits churn exhausts UMA (vLLM #55569) unless VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 or expandable_segments is set.

**Mechanism:** Flattened indexer decode scales with context × draft tokens × sequences. The persistent top-k kernel's shared-memory footprint exceeds GB10 SM limits at long sequence lengths. Variable-size fp32 logits tensors fragment the UMA allocator.

**Evidence:**
- v11src v1/attention/backends/mla/indexer.py:649-653 `_supports_varlen_paged_mqa_logits` requires is_device_capability_family(100) and has_deep_gemm(); :664-667 UNIFORM_BATCH cudagraph otherwise
- v11src model_executor/layers/sparse_attn_indexer_kpool.py:810-822 uses torch.ops._C.persistent_topk when select_k in (512,1024,2048); topk 2048 / kpool 4 = 512
- tonyd2wild issue #14 (2026-08-31): 'the moment 2 requests are in decode, aggregate collapses to ~4 tok/s'; workarounds --max-num-seqs 1 and k=3; widening varlen to family 120 caused illegal memory access with 2 seqs
- note.com tsuru_mitsu (2026-08-29): persistent_topk 'requires 128KB or more of shared memory but only about 101KB is available on GB10' after about 300K tokens; Zeuss5/cuda-exl3 #2 and r0b0tlab patch 2 gate persistent_topk to at least 78 SMs
- https://github.com/vllm-project/vllm/issues/55569: 230K prefill drops MemAvailable 9.6 to <2.5 GB, NV_ERR_NO_MEMORY; fixed by VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 (default 512) or PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- README.md:22: a 318,123-token prompt passed on this recipe (single stream)

**Proposed action:** (1) Set VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 in run.sh (cheap UMA hardening; one knob against prefill throughput). (2) Add a 2-concurrent 100K-context decode probe to the verify set; if it collapses, document max-num-seqs 1 for long-context agent traffic, or make k context-adaptive (EXT-1). (3) Check that the v8/v7 chain gates persistent_topk on GB10, or add the r0b0tlab SM-count gate.

**Est. impact:** Risk reduction for long-context and multi-user workloads (from about 2-4 tok/s collapse to serialized full speed). No change to the published short-prompt cells.

**Validation:** Exclusive slot: 2 concurrent 100K prompts with thinking off, 200 tokens, reading per-stream decode and free -h. Also a 300K single-stream decode of 500+ tokens, checking the logs for persistent_topk errors.

**Risks:** Lowering the logits budget slows long prefill slightly. The persistent_topk gate may slow short-context top-k.

**Verifier reasoning:** Confirmed: v11 indexer.py:649-653 varlen is family-100 only, and sparse_attn_indexer_kpool.py:810-822 calls persistent_topk for select_k in (512,1024,2048); 2048/4 = 512 uses it. No SM-count gate for persistent_topk exists in patch_v7.py or patch_v8_fp8.py. Overstated in two ways. (1) The #55569 fix alternative is already in place: run.sh:254 sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, and the #55569 title (gh) says 'a 64 MiB logits budget or expandable_segments fixes it'. VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 is therefore redundant hardening. (2) README:22 records a 318,123-token single-stream prompt that prefilled and answered correctly on this recipe, which contradicts a hard persistent_topk failure after about 300K for this build (decode length there was short, though). The 2-concurrent long-context collapse (Tony #14) is unverified locally and remains a reasonable probe.

**Verifier corrected claim:** The long-context c=2 decode collapse from the non-varlen SM12x indexer is plausible and should be probed. The logits-churn OOM is already mitigated by run.sh's expandable_segments setting. The persistent_topk SMEM failure has not been reproduced here, since the 318K prompt passed.

**Verifier corrected impact:** Risk reduction for multi-user long context only. The logits-budget knob adds little given expandable_segments.

## EXT-12: RoCEnante one-shot all-reduce and async scheduling: small c=1 gains with stability risks; low priority

- kind=perf component=TP collectives / scheduler impact=2 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** b12x RoCEnante RDMA one-shot all-reduce gave +5-18% aggregate but only flat to +8% single-stream on 2x Spark. It has known wedge and timeout bugs. Async scheduling is used successfully with DFlash2 on the nvidia pack by 0rand, contrary to the AGENTS.md 'leave off'. Its expected c=1 gain is small because the CPU share of a 115 ms step is small.

**Mechanism:** Per-step fixed costs: about 90 small all-reduces over RoCE through NCCL's proxy, plus host scheduling between steps.

**Evidence:**
- Tony PR #22 'Speed night 2026-09-18': RoCEnante aggregate +5-18% C1-C6, single-stream flat to +8%, prefill +26-36% (clamped-fleet caveats)
- NVIDIA forum 381534 post #6: about 37 us per all-reduce × about 90 per step = 3.3 ms/step
- eugr recipes/glm-5.3-flash.yaml: VLLM_ENABLE_ROCE_ALLREDUCE=1, VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB (fork build; the v0.30.0 tree has no RoCE all-reduce)
- sparkring issues #268 and #278 (peer-wait timeout poisons runtime after 1.5 days / 18.5M collectives), local-inference-lab/b12x #313 (graph-replayed collective can wedge one rank)
- 0rand FINDINGS §1: async ON + DFlash2 k=5 + graphs <=16, hardmode 94/100, c1 32.5-33.1

**Proposed action:** Defer until EXT-1, EXT-3 and EXT-5 land. Then test --async-scheduling as a single knob (revisiting the AGENTS.md decision only with bench evidence and greedy parity). Do not adopt RoCEnante for a single-user lane.

**Est. impact:** All-reduce: at most about 3 ms of 115 ms (about 3%) at c=1. Async: a few ms/step, estimated 2-5%. Unmeasured locally.

**Validation:** bench_decode.py prose c=1/c=2 with and without --async-scheduling, plus greedy count 200/200 and a structured JSON validity check.

**Risks:** Async with the DFlash2 backport may be refused or produce races. RoCEnante stability issues on long runs.

**Verifier reasoning:** This is low-priority and mostly external evidence I could not verify locally. The arithmetic of 37 µs × 90 ≈ 3.3 ms per step (about 3% of 105-119 ms) is fine. One supporting datapoint weakens the async case: the 0rand lane (async ON, b12x) implies about 107 ms per step at k=5, the same as our non-async Marlin k=5 (~105 ms, PR #9 iter-H1). So async showed no visible c=1 step-time gain there, even after allowing for the confounding.

**Verifier corrected impact:** ≤3% from all-reduce. Async shows no measurable c=1 gain in the only comparable datapoint.

## Open questions
- What changed in incoai DFlash2 dc77ff1 and bf582e4 compared with 7d74cdd? The model card withdrew its numbers and there is no changelog. Only a local A/B can tell.
- Does the v11 image actually show zero prefix-cache hits under SPEC=dflash2? The code path matches the reported bug, but the README hit evidence is MTP-4 only. Needs one probe run.
- Where exactly does the reported 15.7 GiB multimodal front-end cost come from (processor cache, video decode, MM profiling)? Is it still present with the 4.14 GiB KV pin on v11? Unverified locally.
- Does v0.30.0's B12xExperts SiLU clamp (swiglu_limit=10) match the reference silu-and-mul-with-clamp numerically on the nvidia pack? Is b12x 1.3.0's planned API stable on sm_121a without inference-time JIT?
- Is our bench prose prompt comparable to the 0rand 'filler' and MiaAI prose prompts? Community tok/s differences partly reflect prompt entropy and generation length (tg1024 against 200).
- Does the pilcothink 0.28 image run --moe-backend b12x or marlin for the nvidia pack? The 0rand README says the image 'supplies the b12x MoE/linear backends', but the exact serve flag was not visible.
- Transformers version in glm53-sm121-v11 against the nvidia card's 'transformers>=5.16.1' requirement for the processor. Not inspectable without running the container.
- Is PDL still unsafe on SM12x with FlashInfer 0.6.18.post1 or v0.7.0 (v8 gates it off)? No upstream statement found.
- Whether persistent_topk with select_k=512 (kpool 4) stays under GB10's about 101 KB SMEM at 327,680 context. The recipe passed a 318K needle, but others crashed after about 300K on different paths.

## Verifier: missed issues
- The adaptive-verification blocker on GB10 is structural. KpoolTailBackend subclasses DeepseekV32IndexerBackend (v11 indexer.py:130-139, 180), whose supports_device_cpu_query_lens_mismatch() is False off family 100. maybe_create_adaptive_verification_manager raises ValueError (v11 adaptive_verification.py:453-468). v0.30.0 is the same (indexer.py:190-196, 734-747: only sm_100 varlen or the sm_90 flattened path). The real kernel-level work item is an SM12x varlen or flattened indexer path. That would unlock adaptive verification and also fix the c=2 long-context indexer cost (EXT-11).
- Cross-stack step time is equal, so the gap is acceptance. 0rand (same nvidia pin, DFlash2 k=5, b12x, async) runs at about 9.3 steps/s (33.5/(1+5×0.52), 43.0/(1+5×0.71), 49.7/(1+5×0.88)), against our k=5 at 9.5 steps/s (21.97/2.31, PR #9 iter-H1). The lever with evidence is draft acceptance: drafter revision and prompt mix. Tony README:292-299 links drafter hash 8931dc52 (our local 7d74cdd blob) against b33c0347 (dc77ff1) to 0.35 vs 0.73 acceptance on the same recipe.
- Contradicting k evidence was not cited. Tony README:265-279 says lower k 'loses on every single-stream prompt' and later measured 'flat on prose' at single stream; his per-position acceptance is near-flat (0.93/0.89/0.84/0.81/0.79/0.59/0.94). Our own cross-session k=5 rebench (PR #9 rebench-dflash5: prose c=1 21.76, structured c=1 55.73) is +2.6% prose and -18% structured against the published k=7 (21.2/67.6). That run also FAILED its probe gate (needle-8192 hit=0).
- The decision record conflicts. PR #9 decision.tsv adds 'H1-20260903-apply ... NUM_SPECULATIVE_TOKENS=5 kept' (hand-applied to recipe.yaml serve.env), while PR #11 (this branch) keeps NUM_SPECULATIVE_TOKENS=7 (run.sh:22). Merging both needs an explicit reconcile row in decision.tsv, per AGENTS.md.
- Vision was already on during all published measurements. rebench-20260902T204243Z/engine.log.tail:57 profiles the encoder cache with one max-size video item. Pre-PR #11 run.sh passed no MM flags, and the LibertAI pack has 347 visual tensors. PR #11's UMA risk therefore does not come from turning vision on; a cap was added. Separately, engine.log.tail:51 shows the DFlash2 drafter does not receive target multimodal embeddings ('using text-only draft inputs'), so acceptance and tok/s on image prompts will be lower than the prose cell. This is worth a smoke_vision timing note.
- The fork's loader dequantizes FP8 attention projections to BF16 (glm5next model.py:1203-1218, _try_load_fp8_attn_proj). Swapping in any community FP8-attention checkpoint on this image gives no bandwidth saving without model changes.
- The drafter pin is hard-coded in the generated block (run.sh:52 DRAFT_SNAPSHOT path with 7d74cdd, from recipe.yaml:13). Any drafter A/B must go through recipe.yaml and render. The drafter path is also not forwarded explicitly to the worker except via SPEC_CONFIG, so both nodes need the new snapshot cached.
