# Dimension: regression-bench

## Reviewer summary

This covers why the published prose decode number fell, and whether the bench can detect real improvements. Published prose c=1 fell from 28.30 to 21.2 tok/s. The loss splits exactly into a lower acceptance length (2.875 to 2.426, x0.844) and a slower verify step (101.6 to 114.5 ms, x0.887): 28.30 x 0.844 x 0.887 = 21.19. Most of it is a change in what the bench measures, not a slower serve. The thinking-off template fixes (f59cb7c, then a27e654, which seeds <think></think>) changed the measured text from chain-of-thought plus answer, cut off at 200 tokens, to an answer-only reply of about 98 tokens. That alone took DFlash2-5 acceptance from 2.875 to about 2.33 (-19%) while step time stayed the same (101.6 vs 100.8 ms at H4). The real throughput loss is the DFlash2-7 default. It adds about 13-15 ms per step (+12-14%) with no prose acceptance gain, because draft positions 6 and 7 accept 1.8% and 0% of the time on prose. In exchange, structured decode gains +21% (55.7 to 67.6), not the +34% the README claims. The snapshot change aa28e1f to caca4e6 contributes exactly zero: 125 of 127 files are the same blobs, caca4e6 only adds a 4.8 MB input-scale file, and Marlin never reads it. Going from max-num-seqs 4 to 2 changes nothing at c=1. bench_decode.py timing code has not changed since 08-28. The PR #9 baseline of 19.23 vs 21.2 is session noise at an identical config, not a regression. Across 6 sessions of the same DFlash2-7 config, prose c=1 ranges from 16.9 to 21.2. That spread comes from nondeterministic greedy prose (acceptance 2.17-2.51, 2.28 vs 2.51 inside one boot) and from UMA swap (step 114.5-128.4 ms; 6-7.3 GiB of swap in use during published runs). As a ruler for "substantial improvement", the current bench is not valid: per-run CV is 11%, and the smallest effect a 3-run median can reliably detect is about 31%. The 8% keep/revert gate goes wrong about 48% of the time when nothing changed, and catches a real +10% gain only 57% of the time. The bench also sends the same prompt to both c=2 streams. Those streams then route to the same experts, so the c=2 aggregate is probably overstated by 17-25% (estimate). It has no warm-up, runs greedy only (the server default from generation_config is T=1.0, top_p 0.95), and never measures long-context decode, prefill, vision, thinking-on or tool calls. I propose a factorized ruler: tok/s = acceptance_len / step time. Acceptance comes from per-position spec counters on a paired set of distinct prompts at a fixed length (min_tokens = max_tokens). Step time comes from server-side vllm:inter_token_latency, which the v11 source records once per verify step per request. Boots run ABAB with a swap gate. That should bring the smallest detectable effect to about 3-6% for kernel changes and about 2% for acceptance changes, in roughly 6 minutes of fast-gate time per boot. I also give an nsys protocol for a CUDA-graph verify step, built on the image's --profiler-config and /start_profile hooks and the host's nsys (not verified inside the image), plus a routed-expert capture boot. Together these test the weight-bytes hypothesis. Checkpoint headers put the DFlash2-7 c=1 byte budget at about 26 GB per rank per step, or 96 ms at 273 GB/s. The measured marginal cost of 5.6-7.3 ms per draft slot matches the predicted ~7.2 ms of extra MoE expert bytes.

## FOR-1: ~70% of the 28.30 -> 21.2 prose drop is a ruler change from the thinking-off template, not a throughput regression

- kind=methodology component=chat_template.jinja + bench_decode.py prose phase impact=4 confidence=5 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The 2026-08-29 baseline (28.30) measured chain-of-thought text leaked into content, truncated at 200 tokens (stock template always opened <think>). f59cb7c (honor enable_thinking) and a27e654 (seed '<think></think>' when thinking is off) changed the text being decoded to an answer-only ~98-token paragraph with lower draft acceptability. At constant drafter (DFlash2-5), acceptance went 2.875 -> 2.638 (H4, same day, aa28e1f) -> ~2.33 (PR9 DF5 runs), i.e. x0.81, while step time did not change (101.6 ms baseline vs 100.8 ms H4). In log terms the template accounts for -0.21 of the total -0.29 (72%).

**Mechanism:** tok/s = acceptance_len / step_time. The template changes which tokens get generated. CoT such as 'The user wants a short paragraph...' is formulaic and easy for the drafter to predict, so acceptance is higher. The answer-only output ends at EOS after ~98 tokens. Step time is set by the verify shape (k+1 tokens) and weights, not by the text, so it stayed put.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/baseline-bench.txt:24-30 (28.30 tok/s, acc 2.875, completion 200) => step = 2.875/28.30 = 101.6 ms
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/baseline-count.txt: consecutive=155 nums=159 (CoT preamble in content before H4)
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/iter-h4/bench.txt:24-30 (DF5, honor-thinking template: 26.17 tok/s, acc 2.638, completion 196 => step 100.8 ms)
- git log -p chat_template.jinja: f59cb7c adds template; a27e654 adds {%- else -%}{{- '<think></think>' -}} (now chat_template.jinja:257-260)
- gh pr diff 9 (saved /tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-GLM-5-3-Flash-NVFP4-vLLM-2x-DGX-Spark/29f0f0df-0b0e-4224-a895-0ea5d735b7de/scratchpad/agents/forensics/pr9.diff:62-68 DF5 seqs2 acc 2.308, completion 100; :2546-2554 rebench-dflash5 acc 2.354, completion 97)
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/rebench-20260902T204243Z/bench.txt:18-24 (21.18, acc 2.426, completion 98 => step 114.5 ms)
- Arithmetic: 28.30 x (2.426/2.875) x (101.6/114.5) = 21.19 (matches the published 21.2)
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/decision.tsv:6 labels the H4 drop 'cold', but the step time equals baseline (100.8 vs 101.6 ms), so the whole H4 delta is acceptance/text

**Proposed action:** Record in evidence/decision.tsv that the 08-29 28.30 and every pre-a27e654 prose number sit on a different ruler (CoT text) and cannot be compared with post-a27e654 numbers. Re-label h4 from 'cold' to 'ruler change: acceptance 2.875->2.638, step unchanged'. From now on, publish acceptance_len and step_ms next to every tok/s cell so ruler changes are visible.

**Est. impact:** Changes no serve behaviour. It re-attributes about 5.4 of the 7.1 tok/s lost (x0.81 of x0.748) to the ruler. The real regression to chase is x0.89 (FOR-2), not x0.75.

**Validation:** Already visible in existing receipts (step_ms column). Optional GPU check: one boot of DF5 seqs4 with the old stock template vs the current template, same session. Acceptance should differ by about 19% and step_ms by less than 2%.

**Risks:** The DF5 acceptance on answer-only text (2.33) comes from PR9 sessions on caca4e6, but those weights are bit-identical to aa28e1f under Marlin (FOR-3), so this is not confounded by the snapshot.

**Verifier reasoning:** I re-derived every number. Baseline: evidence/baseline-bench.txt:24-29 gives 28.30 tok/s, acc 2.875 and 200 tokens, so the step is 2.875/28.30 = 101.6 ms. Rebench: rebench-20260902T204243Z/bench.txt:18-23 gives 21.18, acc 2.426 and 98 tokens, so 114.5 ms. 28.30 x (2.426/2.875) x (101.6/114.5) = 21.19. For DF5 on the seeded template: pr9.diff H1 has acc 2.308 over 100 tokens and rebench-dflash5 has 2.354 over 97, mean 2.331, and 2.331/2.875 = 0.81. ln 0.81 = -0.210 against a total ln(21.2/28.3) = -0.289, which is 72%. git diff f59cb7c a27e654 -- chat_template.jinja is exactly the added else branch with '<think></think>'. Two caveats. (a) H4 (iter-h4/bench.txt, 196 tokens) ran the intermediate f59cb7c template, not the answer-only one. I checked wall-clock consistency: run 3 gives 195/(7.77-0.32) = 26.2, matching the reported decode rate, so its decode window really covers the whole generation. That makes 'step unchanged' valid only for CoT vs the intermediate template. (b) On the answer-only template, DF5 steps are 105.1 and 108.2 ms, 3.5-6.5% above 101.6 ms. Part of the step-time rise therefore is not DF7 and remains unexplained: session, seqs 4->2, or UMA.

**Verifier corrected claim:** The thinking-off template change, mostly the '<think></think>' seed in a27e654, explains roughly 60-75% of the log-drop from 28.30 to 21.2. It works through acceptance (2.875 -> ~2.33 at DF5). The remaining ~0.12 in log comes from a slower step. Most of that is DF7, but about 4-6 ms/step also appears at DF5 on the new template and is unattributed.

**Verifier corrected impact:** Re-attributes about 5 of the 7.1 tok/s to the ruler. No serve change.

## FOR-2: DFlash2-7 default is a real ~12-14% prose regression; its structured gain is +21%, not the +34% README claims

- kind=perf component=run.sh SPEC_CONFIG / recipe.yaml NUM_SPECULATIVE_TOKENS=7 impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 4)

**Claim:** Going from 6 to 8 verify tokens adds 13-15 ms per step (DF5 ~101-108 ms to DF7 ~114.5-128 ms). On answer-only prose, DF7 acceptance (mean 2.315 over 6 sessions) is no higher than DF5 (2.331 over 2 sessions): per-position acceptance for slots 5/6/7 is 0.044/0.018/0.000. Net prose c=1 is about -12 to -14% for DF7, which is also the default in the AGENTS.md and README text. Structured c=1 at the same template is 55.7 (DF5) vs 67.6 (DF7) = +21%. README:20 '50.7 -> 68.1' compares a CoT-template DF5 number with a seeded-template DF7 number.

**Mechanism:** Each extra verify token pulls in about 6.5 more distinct experts per MoE layer (uniform-routing estimate 44.8 -> 58.1 experts at 6 -> 8 tokens), about 2 GB/rank/token. That is ~7 ms at 273 GB/s, matching the measured +6-7 ms per slot. On prose the extra slots almost never verify, so this cost buys nothing. On counting they verify about 98% of the time.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/iter-h3/bench.txt:18-24 vs baseline-bench.txt:24-30: same day, same template/snapshot; DF7 step 116.2 ms vs DF5 101.6 ms (+14.4%); prose tok/s 27.32 vs 28.30
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/rebench-20260902T204243Z/engine.log.tail:256,260,266,270 prose-window per-position rates, e.g. '0.616, 0.448, 0.264, 0.056, 0.024, 0.016, 0.000'; weighted over 271 drafts: [0.668,0.450,0.236,0.085,0.044,0.018,0.000]
- DF5 seqs2 caca4e6: pr9.diff:62-68 (21.97, acc 2.308, step 105.1 ms) and :2546-2554 (21.76, acc 2.354, step 108.2 ms)
- DF7 seqs2 caca4e6 prose step ms: 114.5 (rebench), 119.2 (parity pr9.diff:3862-3868), 122.2/128.4 (iter-h-snap), 120.1/120.4 (/home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-b12x/evidence/iter-b12x-A-marlin/bench.txt, bench-wave2.txt)
- Structured same template: DF5 55.60 (iter-h4/bench.txt:57-63), 55.73 (pr9.diff:2570-2576) vs DF7 67.64 (rebench bench.txt:40-46) => +21%; README.md:20 claims 50.7->68.1 (+34%)
- README.md:75 'Positions 5-6 accept under 15% on prose' (measured 4.4% / 1.8%)
- pr9.diff:3916-3917: DF5 keep was hand-applied (a76474e) on the PR #9 branch only, then its rebench 'FAILED gate' on a needle refusal (see BENCH-5); never merged

**Proposed action:** Do not re-litigate by ad-hoc benches. Run the PROTO-1 panel on k in {3,4,5,7} and choose k by a workload-weighted objective (for example 0.4 prose/chat, 0.3 code, 0.2 thinking-on, 0.1 structured/tool), with the weights stated in recipe.yaml. The PROTO-3 model predicts prose: k=3 ~25-26, k=5 ~23.6-23.9, k=7 ~21.3 tok/s; structured: k=5 ~57, k=7 ~67.5. This is an explicit exception to the AGENTS.md line 'Default drafter is DFlash2-7', justified by the per-position counters and step-time data above.

**Est. impact:** Prose c=1 +12-14% if DF5 (21.2 -> ~23.8-24.2, estimated from 1.007/0.883). Structured c=1 -18% (67.6 -> 55.7). Prose c=2 per stream 16.6 -> 17.5-19.6 (measured DF5 seqs2, pr9.diff:72-79, 2556-2565). k=3 is modelled at up to +20% prose but is unmeasured.

**Validation:** ABAB boots DF5/DF7 (2 boots each), PROTO-1 fast gate. Accept if the step_ms difference is 10-14% and the paired prose acceptance difference is within ±2%.

**Risks:** Truncating the DFlash2 block changes the grouped-conv block_size (1+k), so early-position acceptance may drop. The observed DF5 value (2.33) is below the per-position prediction (2.48). The old eager-era 'DF4 acc 2.8, 20.6 tok/s' note (evidence/how-explanation.md:133) contradicts the step model and must be re-measured, not trusted.

**Verifier reasoning:** Mechanism and structured numbers check out. Structured is 67.64/55.73 = 1.214 (rebench bench.txt:40 vs pr9.diff rebench-dflash5), and on the stock template H3 vs baseline is 61.86/50.74 = +22%. README.md:20 '50.7 -> 68.1' mixes templates, since the baseline structured output carried a CoT preamble (baseline-count consecutive=155). Mean DF7 prose acceptance over 6 sessions is 2.315 (2.426, 2.292, 2.209, 2.172, 2.281, 2.509) vs DF5 at 2.331, and the per-position rates I re-weighted from engine.log.tail:256-270 come out [0.668, 0.450, 0.236, 0.085, 0.044, 0.018, 0]. Same-day step deltas are +14.6 ms (h3 vs baseline) and +14.0 ms (parity 119.1 vs H1 105.1), but rebench (114.5) vs rebench-dflash5 (108.2) is only +6.3 ms. The impact estimate '21.2 -> ~23.8-24.2' comes from the model and overshoots what was actually measured. Measured DF5 at seqs=2 is 21.97 and 21.76, which is only +3.6%/+2.6% over the published 21.2. It is +13-14% over the same-night DF7 parity (19.23) and over the 6-session DF7 mean (19.2).

**Verifier corrected claim:** DF7 buys no prose acceptance on answer-only text and costs about 6-15 ms per step (+5-14%) depending on the boot pair. Measured DF5 at seqs=2 prose c=1 is 21.8-22.0, which is +3-4% over the published DF7 cell and about +14% over the DF7 mean of 6 sessions (19.2). Structured c=1 is +21% for DF7 (55.7 -> 67.6), not +34%.

**Verifier corrected impact:** Switching to DF5 changes prose c=1 by about +3% to +14% (measured range) and structured c=1 by -18%. The modelled 23.8-24.2 is not supported: the model overpredicts measured DF5 by about 8%.

## FOR-3: Snapshot aa28e1f -> caca4e6 contributes exactly zero under Marlin (bit-identical served weights)

- kind=correctness component=LibertAI checkpoint pins / MOE_BACKEND=marlin impact=3 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** All 120 weight shards of caca4e6 are the same HF blobs as aa28e1f. caca4e6 only adds model-input-scales.safetensors (4,840,704 B), README and a bigger index, and config.json is content-identical. Marlin NVFP4 MoE only applies input_global_scale when the activation dtype is int8, so under Marlin the served numerics do not change. The h-snap row's 'caca4e6 is a quality pin' and any tok/s difference between the two pins is session noise. The documented rollback pin is effectively the same model as the old pin.

**Mechanism:** The update only added activation-scale tensors, which a W4A16 Marlin path ignores. Snapshot and template changes landed in the same commit (a27e654), which made the snapshot look causal.

**Evidence:**
- CPU check of the HF cache: realpath comparison of /home/sfxnz/.cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/{aa28e1f...,caca4e6...}: 125 files identical blobs, 2 differ (config.json same content, model.safetensors.index.json 12,237,653 -> 16,505,521 B), caca4e6-only: README.md, model-input-scales.safetensors
- /tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-GLM-5-3-Flash-NVFP4-vLLM-2x-DGX-Spark/29f0f0df-0b0e-4224-a895-0ea5d735b7de/scratchpad/v11src/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:128-133,191-196 (input_global_scale used only under input_dtype == int8)
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/decision.tsv:8 (h-snap 'quality pin')
- git show a27e654 message: 'marlin, which never reads an activation scale'

**Proposed action:** Remove 'snapshot' from the list of explanations for the drop. Record in decision.tsv that h-snap was a no-op for Marlin. For the nvidia pack (PR #11), document that Marlin serves it as W4A16 (weights NVFP4, BF16 activations), so its hf_quant_config W4A4 calibration is also unused and its quality and acceptance should be measured on the Marlin path actually served.

**Est. impact:** 0 tok/s by construction. It stops a false lead and makes it clear the rollback pin gives no quality delta under Marlin.

**Validation:** CPU-only blob comparison (done). Optional: greedy count and prose hash on both pins in one boot slot; they should produce identical text modulo nondeterminism.

**Risks:** If a future MOE_BACKEND (cutlass/b12x/W4A4) reads input_scale, the two pins stop being equivalent.

**Verifier reasoning:** CPU blob check (scratchpad agents/snapcmp_regbench.py): 125 of 127 files are the same blob. config.json has equal sha256. Only the index differs (12,237,653 -> 16,505,521 B). The files only in caca4e6 are README.md and model-input-scales.safetensors (4,840,704 B). The header of that file holds 37,152 tensors, all routed-expert {gate,up,down}_proj.input_scale (43 layers incl. MTP x 288 x 3). No existing index key changed file. aa28e1f has 0 input_scale keys. The LibertAI config.json quantization_config has input_activations: null, and dense MLP and shared experts are in its ignore list. Under Marlin, oracle/nvfp4.py:494-501 builds nvfp4_w4a16_moe_quant_config with no a1/a2 gscale, and marlin_moe.py:128-133 applies input_global_scale only for int8. The served numerics are identical. One correction to the proposed action: for the nvidia pack the dense MLP (layers 0-2) is NVFP4 W4A4 through ModelOptNvFp4LinearMethod. It reads input_scale (modelopt.py:1156-1212), so its calibration IS used there. Only the MoE part is served as W4A16.

**Verifier corrected claim:** The two LibertAI pins are numerically identical under Marlin. For the nvidia pack, only the routed-expert MoE ignores input_scale. Dense-MLP NVFP4 linears use it.

## FOR-4: 21.2 vs 19.23 vs 16.9-18.1 at an identical config is session noise (nondeterministic prose text + UMA swap), not a regression

- kind=methodology component=bench_decode.py / kit bench / host UMA impact=4 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The published 21.2 (09-02), the PR #9 baseline 19.23 (09-02 22:51Z, kit bench with the same timing code), h-snap 18.08/16.92 (08-31) and b12x-A 18.98/20.85 (09-08) all ran the same DF7, seqs=2, Marlin, LibertAI config. Their differences split into acceptance (2.17-2.51, CV 5.6%) and step time (114.5-128.4 ms, CV 3.7%). The bench timing semantics have not changed since 08-28: the only bench_decode.py diffs since then are default model/concurrency/phase, and kit/bench_decode.py has the same stream_one/wave/acceptance code.

**Mechanism:** Two independent noise sources multiply. (1) Greedy prose is not batch/run-invariant, so the text and its acceptance differ run to run, and only ~40 verify steps per run means high sampling variance (SE of acceptance ~5.3% per 120-draft block, computed from the per-position distribution: sd of accepted-per-draft = 1.46). (2) UMA/swap state per boot shifts step time by 2-6%.

**Evidence:**
- b12x A arm, same boot: wave1 prose acc 2.281 vs wave2 2.509 (+10%) at the same step ~120 ms: /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-b12x/evidence/iter-b12x-A-marlin/bench.txt and bench-wave2.txt; result.txt
- Structured DF7 acceptance is 7.8441558 to 7 digits in 4 sessions (rebench bench.txt:45, iter-h-snap bench.txt:45 and bench-wave2.txt:45, pr9.diff:3889), so greedy counting is deterministic and greedy prose is not
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/rebench-20260902T204243Z/free-after-bench.txt: used 116, available 5, swap 6 GiB used; b12x iter-b12x-A-marlin/postboot.txt: swap 7.3 GiB; README.md:22 Worker_TP0 ~1.3 GiB swapped, first structured wave ~52 vs 59.9 (-13%)
- iter-h-snap: wave1 -> wave2 (same boot, 'swap' per decision.tsv:8): prose step 122.2 -> 128.4 ms, structured 116.8 -> 121.7 ms
- git diff f59cb7c~1 29954c8 -- bench_decode.py: only --model/--concurrency/--phase defaults changed; kit version at git show e898a80:kit/bench_decode.py has identical measurement code
- /proc/sys/vm/swappiness = 60 on spark1 (read now)
- pr9.diff:37 kit PARITY: 'the 3-run prose ruler spread ±11% across parity runs'

**Proposed action:** Treat 19.23 vs 21.2 as noise and remove any 'regression' narrative about it. Adopt PROTO-1/2 (factorized metrics, paired prompts, swap gate) before accepting any further keep or revert.

**Est. impact:** Stops chasing a phantom ~9% regression. The noise floor itself (±12% per run) is what hides real 5-15% gains (see BENCH-1).

**Validation:** Existing receipts already show it. Under PROTO-1 the same config across 2 boots should give step_ms within 2% and paired acceptance within 1.5%.

**Risks:** Some of the step-time spread may come from other tenants on the Sparks (exclusive-GPU rule), which is unrecorded in some sessions.

**Verifier reasoning:** Recomputed. DF7 prose acceptance is 2.426, 2.292, 2.209, 2.172, 2.281, 2.509: mean 2.315, sd 0.129, CV 5.6%. Steps are 114.5, 119.1, 122.2, 128.4, 120.2, 120.3 ms: CV 3.8%. The b12x A arm (iter-b12x-A-marlin/bench.txt and bench-wave2.txt) has acceptance 2.281 vs 2.509 in the same boot at the same step (120.2 vs 120.3 ms). Structured acceptance is exactly 7.8441558 in rebench, h-snap waves 1 and 2, and parity. git diff f59cb7c~1 29954c8 -- bench_decode.py changes only defaults, and git show e898a80:kit/bench_decode.py has the same prompt, content-only first-token timing and (completion-1)/window formula. Parity 19.23 decomposes to acc 2.292 (-5.5%) and step 119.1 ms (+4%), both inside the spread.

## FOR-5: max-num-seqs 4 -> 2 costs ~0 at c=1 but silently dropped the c=4 capacity cell from the ruler

- kind=methodology component=recipe.yaml MAX_NUM_SEQS / bench defaults impact=2 confidence=4 effort=S needs_gpu=False
- **verdict: plausible** (corrected confidence 3)

**Claim:** At c=1 the verify graph is the same size whatever the seq cap (6 or 8 tokens), and the DF5 step at seqs=2 (105-108 ms) matches seqs=4 (100.8-102.3 ms) within session noise. The c=4 cell (65.7 tok/s aggregate at 08-29) was removed from the bench defaults (concurrency [1,2]), so the loss of four-way admission never shows up as a number.

**Mechanism:** The seq cap changes the capture ladder and how KV/KDA copies are budgeted, not the c=1 kernel shapes.

**Evidence:**
- baseline-bench.txt:44-52 prose c=4 agg 65.70, per-stream 16.99
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/bench_decode.py:169 default concurrency [1, 2]
- step ms: DF5 seqs4 101.6/100.8/102.3 (baseline, H4, H2) vs DF5 seqs2 105.1/108.2 (pr9.diff:62-68, 2546-2554)
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/recipe.yaml:46,49

**Proposed action:** Keep a c=4 distinct-prompt cell (or TTFT/queue_time at c=4) in the panel as a capacity metric, reported separately from per-stream decode.

**Est. impact:** 0% at c=1. It restores visibility of a capacity loss: aggregate throughput at 4 users is now bounded by queueing (2 slots).

**Validation:** The panel includes c=4 with vllm:request_queue_time_seconds.

**Risks:** None beyond bench time (~2 min).

**Verifier reasoning:** The c=4 cell is gone: bench_decode.py:169 defaults to [1,2], baseline-bench.txt:44-52 shows 65.70 agg, and rebench ran only [1,2]. DF5 steps at seqs=2 (105.1, 108.2) are 3-7% above seqs=4 (101.6, 100.8, 102.3 from h2 = 3.020/29.52). They also differ in template, snapshot and day, so 'identical' is not shown. It is only not contradicted, and H3-20260903 MAX_NUM_SEQS=1 (17.40) is not faster either. Low stakes.

**Verifier corrected claim:** At c=1 the seq cap has no demonstrated effect: the DF5 seqs 2 vs 4 gap of 3-7% is confounded. The c=4 capacity cell was silently dropped from the ruler.

## FOR-6: Published claims and ledger rows contain errors that steered decisions

- kind=methodology component=README.md / recipe.yaml / evidence ledger impact=3 confidence=5 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** (a) recipe.yaml:89 and README:11 say '200 completion tokens', but the published prose cells have median 98. (b) README:20 credits DF7 with 50.7 -> 68.1 structured, which mixes templates (true same-template gain +21%). (c) README:153 rejects the packed nvfp4_ds_mla 1M lane because it measured '~22 tok/s prose versus 28 here', but 'here' is now 21.2, so on the numbers as published the 1M lane is at parity, though it was measured on a different ruler. (d) decision.tsv h4 'cold' and h-snap 'quality pin' are mis-attributions (FOR-1, FOR-3). (e) evidence/how-explanation.md still describes DF5/seqs4/aa28e1f and a 400,497-token pool, while README says 372,877.

**Mechanism:** Numbers taken on different rulers were compared as if on one. Stale prose kept the older, higher baseline.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/recipe.yaml:89
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/README.md:11,20,22,153
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/rebench-20260902T204243Z/bench.txt:21 (median_completion_tokens 98)
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/how-explanation.md:5,9,64,96
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/decision.tsv:6,8

**Proposed action:** Fix these via recipe.yaml and render, not hand edits: set the conditions to 'natural stop ~98 tokens', correct the structured comparison, and mark the 1M-lane comparison 'not comparable, re-measure on PROTO-1'. Add a 'ruler_version' field to every measured row.

**Est. impact:** No tok/s change. It removes wrong priors from the next planning round, notably re-opening the 1M nvfp4_ds_mla lane question.

**Validation:** python3 kit/render.py --check after the recipe.yaml edits. A reviewer diff against this list.

**Risks:** None.

**Verifier reasoning:** recipe.yaml:89 and README.md:11 say '200 completion tokens', while rebench bench.txt:21 gives median 98 and README.md:20 itself says '~105 tokens', so the README contradicts itself. README.md:20 has 50.7 -> 68.1. README.md:153 has '~22 tok/s prose versus 28 here'. decision.tsv rows h4 and h-snap are as quoted. how-explanation.md:7,9,64,96 describe DF5, seqs=4 and the 400,497 pool. The reviewer missed that README.md:153 also still says '~400k-token fp8 hybrid pool (1.22x)', against README.md:22's 372,877 (1.14x). Separately, the 1M-lane '~22 tok/s' is someone else's public number (how-explanation.md:105, drowzeys) on a different bench, so it was never comparable.

**Verifier corrected claim:** Same as the reviewer's list, plus README.md:153's stale ~400k/1.22x pool figure. The 1M-lane comparison was never on this ruler in the first place.

## BENCH-1: Current ruler (median of 3, ~98-token prose) can only detect about a 31% change; the 8% gate is close to a coin flip

- kind=methodology component=bench_decode.py / hillclimb keep-revert rule impact=5 confidence=5 effort=M needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** Per-run CV of prose c=1 is 11-12% at ~98 tokens, versus ~4% on the 200-token CoT ruler. With median-of-3, the per-cell SE is about 8% and a two-arm difference has SD about 11.3%. Under no true change, P(|delta| > 8%) = 48%. For a true +10% gain, P(detect) = 57%. The smallest effect detectable (80% power, alpha 0.05) is about 31%. The user's goal of a 'substantial' improvement in the 5-20% range cannot be resolved with this ruler.

**Mechanism:** Few verify steps per sample (~40 per run), natural-EOS stopping so length varies, nondeterministic greedy prose text, and boot-level UMA drift, all folded into one tok/s number.

**Evidence:**
- rebench bench.txt:2-4 runs 23.40/18.75/21.18 (CV 11%); pooled 9 DF7 caca4e6 runs 16.13-23.40, mean 19.0, sd 2.31 (CV 12%)
- baseline-bench.txt:2-4 CoT/200-token runs 28.30/28.68/26.32 (CV ~4.5%)
- SE of median ≈ 1.2533·σ/√3 = 7.96%; SD diff = 11.26%; P(|Z|>0.71)=0.477; P(Z>-0.18)=0.571; MDE = 2.8×11.26% = 31.5% (computed)
- pr9.diff PR body: 'Noise: 8%' gate; H1-20260903 +14% reverted by lint then hand-applied, never merged
- Acceptance sampling SE: 120 drafts per block -> 5.3%; 3,300 drafts -> 1.0%; 7,000 -> 0.7% (sd of accepted-per-draft 1.46 from per-position rates)

**Proposed action:** Replace the keep/revert rule: decide on factorized metrics (PROTO-2) with a stated smallest detectable effect. Report 95% CIs (bootstrap over prompts × boots). Until then, label any |delta| < 30% on the old ruler 'inconclusive' rather than 'reverted'.

**Est. impact:** Removes about 48% false decisions. Fixing the ruler is what unlocks measuring 5-15% kernel wins.

**Validation:** Statistics computed from existing receipts. Re-derive CVs from the first PROTO-1 boots.

**Risks:** The normal approximation with n=3 is rough, and true medians have heavier tails, which strengthens the conclusion.

**Verifier reasoning:** Rebench CV: 23.40/18.75/21.18 gives mean 21.11 and sd 2.33, CV 11.0%. Baseline CoT: 28.30/28.68/26.32 gives CV 4.6%. Arithmetic: 1.2533/sqrt(3) x 11% = 7.96%, diff 11.26%, MDE 2.8 x 11.26 = 31.5%, and P(Z > -0.178) = 0.57. One number is overstated. The hillclimb gate is one-sided: keep only if the delta is above +8%, and 'not past 8% noise' means revert (decision.tsv H2/H3-20260903). Under no change, the harmful error (false keep) is P(Z > 0.71) = 24%, not 48%. A no-op being reverted is not a harmful mistake. Because baselines come from another boot, between-boot variance pushes that to about 25-30%. Also, H1-20260903 (+14%) has z of about 1.25 on this ruler, which is not significant. It is only robust through factorization (step 105.1 vs 119.1 ms at equal acceptance, 2.31 vs 2.29).

**Verifier corrected claim:** The median-of-3 prose ruler (CV ~11%) has an 80%-power MDE of about 31%. The one-sided 8% keep gate falsely keeps a no-op about 24-30% of the time and catches a true +10% only about 57% of the time.

**Verifier corrected impact:** Removes about a quarter of false keeps and makes 5-15% effects measurable.

## BENCH-2: Identical prompts on both c=2 streams share MoE experts and likely overstate the c=2 aggregate by 17-25%

- kind=methodology component=bench_decode.py wave() impact=4 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** wave() submits the same prompt to every stream. At greedy, the structured c=2 streams run in lockstep with identical per-stream rates (60.71/60.72, 60.41/60.41), so the 16-token verify batch routes to the same experts as 8 tokens. Real multi-user traffic has distinct prompts. The uniform-routing estimate puts distinct experts at 104.5 per layer (16 tokens) vs 58.1 (8 tokens): +13.8 GB/rank/step, or +50-63 ms at 273-220 GB/s.

**Mechanism:** MoE decode on UMA is weight-bandwidth bound. The cost of a second stream is the extra distinct experts it touches, and with identical prompts that is roughly zero.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/bench_decode.py:97 (same prompt for every future)
- rebench bench.txt:11-13 structured c=2 per-stream identical to 0.01 tok/s
- step ms: structured c=1 116.0, c=2 (identical) 131.1; prose c=2 150 (16.64/2.496, partially diverged text)
- Header-derived expert size: 14.156 MB per expert (gate/up [2048,2048] U8 + [2048,256] F8 scales; down [4096,1024] + [4096,128]) from /home/sfxnz/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/09b04e5e74bca08ca8549fc736d4cdd8624bfde3 safetensors headers; 7.08 MB per TP rank
- E[distinct] = 288·(1-(1-8/288)^T): T=8 -> 58.1, T=16 -> 104.5 (computed)

**Proposed action:** Use distinct prompts per stream (a fixed, seeded list of prompts). Keep the identical-prompt cell only as a labelled 'shared-expert best case' diagnostic.

**Est. impact:** Predicted realistic prose c=2 per stream ~12.5-13.4 tok/s vs published 16.6 (-19 to -25%), aggregate ~25-27 vs 33.2. This is an estimate, and it changes the seqs and k trade-off at c=2.

**Validation:** In one boot, run the c=2 identical vs c=2 distinct prompt panel. Compare step_ms from ITL. PROTO-5 routed-expert capture gives the measured distinct-expert count.

**Risks:** Real routing has locality, so fewer distinct experts than uniform and a smaller overstatement. Attention/KDA costs also scale with streams.

**Verifier reasoning:** bench_decode.py:97 sends the same prompt to every stream, and structured c=2 per-stream rates are identical (60.71/60.72, 60.41/60.41), so structured c=2 runs in lockstep. The E[distinct] arithmetic checks: 44.8/58.1/104.5 at T=6/8/16, and 13.8 GB/273 = 50.5 ms. But the published cell is prose c=2, not structured. Prose per-stream rates differ within each wave (17.28/16.66, 17.84/16.62, 16.59/14.85), so those texts already diverge and partly route to distinct experts. Its measured step (2.496/16.64 = 150 ms) is already 19 ms above lockstep structured c=2 (131 ms). The uniform model predicts about 180 ms for fully distinct streams (114.5 + 15 non-MoE + 50). How early the texts diverge is unknown, so the 17-25% overstatement is an upper bound from uniform routing, not an estimate for the published prose cell.

**Verifier corrected claim:** Identical prompts make structured c=2 a lockstep best case. The published prose c=2 cell is already partly diverged, so its overstatement versus distinct prompts is probably well below 17-25% (upper bound about 17%, plausibly 0-10%). This is unmeasured.

**Verifier corrected impact:** Prose c=2 per stream is probably overstated by 0-17% (unmeasured). Structured c=2 is overstated more.

## BENCH-3: bench_decode.py timing is sound for thinking-off, but it breaks for thinking-on/tool calls and lacks warm-up, pairing and sampling

- kind=correctness component=bench_decode.py stream_one()/main() impact=4 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** (1) decode_tok_s = (completion_tokens - 1)/(t_done - t_first_content) correctly drops TTFT and the prefill token. The derived step time stayed at 101.6 vs 100.8 ms across the template change, which validates it for thinking-off. (2) 'first' is only set on delta.content (lines 69-71). With enable_thinking=true, reasoning streams in reasoning_content first, so the window starts late while completion_tokens includes all reasoning tokens. Decode tok/s would be inflated, by up to (total/content) times. (3) Spec counters are diffed per 3-run block (lines 187, 208), while tok/s is per-run, so the two cannot be paired and the warm-up run is always included. (4) There is no warm-up wave. A 6.2-6.9 s TTFT stall hits one c=2 wave in most sessions. (5) Only temperature=0 is tested, although the server default (generation_config.json) is T=1.0, top_p=0.95, and acceptance under rejection sampling is unmeasured. (6) No per-run completion_tokens or output hash is logged, so determinism and losslessness cannot be audited from receipts.

**Mechanism:** The measurement was designed for one short thinking-off greedy prompt. Extending it to other regimes without changing where the window starts, or how counters are scoped, silently corrupts results.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/bench_decode.py:35-39 (temperature 0, enable_thinking False), :69-71 (content-only first token), :77-86 (formula), :187-208 (per-block counters)
- c=2 TTFT stalls: rebench bench.txt:5 (6.286), iter-h3/bench.txt:6 (6.594), iter-h4/bench.txt:5 (6.947), iter-h1/bench.txt:6 (6.562), iter-h-snap/bench.txt:5 (6.630), pr9.diff:92 (6.472), :2600 (6.422)
- /home/sfxnz/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/09b04e5e74bca08ca8549fc736d4cdd8624bfde3/generation_config.json (do_sample true, temperature 1.0, top_p 0.95)
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/chat_template.jinja:3 injects '<|system|>Reasoning Effort: Max' unless reasoning_effort is low/high, so length under thinking-on depends on an unpinned kwarg

**Proposed action:** Minimal patch (for the implementation stage): set first on any delta (content, reasoning_content or tool_calls). Diff /metrics per run. Add a discarded warm-up request per (phase, c). Log completion_tokens and sha256(text) per stream. Add --temperature/--top-p/--seed. Pin reasoning_effort in chat_template_kwargs.

**Est. impact:** Prevents a potential 2-10x inflation on any thinking-on cell. Warm-up removes the 6 s outliers. Per-run pairing enables the PROTO-2 factorization.

**Validation:** Unit test on a recorded SSE transcript (CPU only). On GPU, the step_ms from client timing must match vllm:inter_token_latency within 3%.

**Risks:** Changing the ruler again: keep the old prose cell as a continuity row, labelled.

**Verifier reasoning:** Checked bench_decode.py. Lines 35-39 are temperature 0 and enable_thinking False. Lines 69-71 set 'first' only on delta.content. Lines 77-86 are the (completion-1)/(t1-first) formula. Counters are diffed per block (187, 208). No warm-up, no per-run hash. The nvidia generation_config.json is do_sample true, T 1.0, top_p 0.95. chat_template.jinja:2-3 injects 'Reasoning Effort: Max' unless low/high is given, and the stock LibertAI template does the same. The TTFT stalls (6.29-6.95 s) are real but do not touch decode tok/s: decode excludes TTFT, the per-stream rates in those waves are normal, and median TTFT over 6 streams is unaffected. So the warm-up benefit is small for the published cells. The thinking-on inflation risk is latent, because the bench hard-codes thinking off.

**Verifier corrected impact:** The warm-up fix is mainly for TTFT hygiene, not decode. The thinking-on inflation (up to total/content x) is latent until someone adds a thinking-on cell.

## BENCH-4: Coverage blind spots: prefill, long-context decode, vision, thinking-on and tool calls are unmeasured; prefill may have regressed about 38-52%

- kind=methodology component=bench_decode.py + needle_probe impact=4 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The ruler decodes after a ~40-token prompt only. The recipe's value is a 327,680 window with vision on, yet decode at 32k/128k context (DSA indexer top-k 2048 + KV reads), prefill tok/s, image TTFT and thinking-on length are never tracked. The only prefill data points are incidental: README (MTP-4 eager era) 1425 tok/s at 10,271 tokens; rebench DF7 884 tok/s at 16,400; PR9 DF5 690 tok/s at 17,081. The needle probe named '8192' actually sends 16.4-17.1k tokens.

**Mechanism:** Prefill can regress independently of decode (drafter prefill plus aux-hidden capture, 2048-token chunking, UMA). Long-context decode adds per-step attention and indexer bytes that the short prompt never exercises.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/README.md:22 (1425 tok/s, 10271 tokens)
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/rebench-20260902T204243Z/needle-8192.txt:1 (prompt_tokens=16400, ttft 18.55 s, 884.1 tok/s)
- pr9.diff:3414,3426 (request prompt_tokens 8192; actual 17081, ttft 24.75 s, 690.1 tok/s)
- evidence/how-explanation.md:135 (max_num_scheduled_tokens forced to 2048 by the spec settings, so prefill is chunked at 2048)
- PR #11 turns vision on by default with --limit-mm-per-prompt {image:4, video:1} (run.sh:61-64); its UMA and text-decode effect is unmeasured

**Proposed action:** Add PROTO-1 cells: prefill+decode at 8k/32k/128k with deterministic seeded salts, one image request, thinking-on and a tool-call turn. Report prefill tok/s = prompt_tokens / vllm:request_prefill_time_seconds. Fix the needle probe to size by tokens (tokenizer count) and label real lengths.

**Est. impact:** Unknown. Prefill numbers suggest a -38 to -52% prefill change since the MTP-4 era (different lengths and eras, so an estimate). A long-context decode penalty is unquantified.

**Validation:** First exclusive boot: 32k and 128k cells, 2 runs each. Compare prefill tok/s DF7 vs SPEC=mtp vs DF5 in the same boot series.

**Risks:** A 128k prefill takes ~2-2.5 min per run at 900-1300 tok/s. Keep it out of the fast gate.

**Verifier reasoning:** The coverage gap is real: the bench has only a ~40-token prompt, and none of 32k/128k decode, prefill, image, thinking-on or tool calls is tracked. The needle mislabel is confirmed: rebench needle-8192.txt gives prompt_tokens=16400 and 884.1 tok/s, and pr9.diff gives 17081 and 690.1. The '-38 to -52% prefill regression' is not supported. It compares a 10,271-token prompt (README.md:22, MTP-4 eager) with 16-17k prompts, and within the post-change pair DF5 (690) is slower than DF7 (884). That is the opposite of a drafter-cost story, so the spread is dominated by noise, UMA or session effects.

**Verifier corrected claim:** Prefill, long-context decode, vision, thinking-on and tool calls are unmeasured. The existing prefill points are confounded (different lengths, eras, and a DF5 < DF7 inversion), so no prefill regression can be inferred.

**Verifier corrected impact:** Unknown. The regression magnitude is not estimable from existing receipts.

## BENCH-5: A random-salt needle gate blocked a measured prose win by refusing on 'prompt injection' grounds

- kind=methodology component=kit/probes/needle.py gate in hillclimb/rebench impact=3 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The DF5 rebench (09-03) failed its gate because needle-8192 got a refusal ('classic prompt injection / data exfiltration pattern'). The probe uses a fresh random salt per run and a secret-code-style instruction, so the gate result is not reproducible and depends on model safety behaviour, not on the perf change. The same probe passed on the DF7 rebench. This blocked the only measured +14% prose keep from landing.

**Mechanism:** A nondeterministic prompt plus refusal-prone wording makes a binary gate with unknown false-fail rate, coupled to unrelated performance knobs.

**Evidence:**
- pr9.diff:3416-3433 (salt S1788412615137064974, hit=0, verdict=FAIL reason=needle_miss, refusal text)
- pr9.diff:3917 (trail: 'FAILED gate: probes=fail (needle-8192 hit=0, model refused the prompt as injection)')
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/rebench-20260902T204243Z/needle-8192.txt:1 (hit=1 with a different salt)

**Proposed action:** Use a fixed seeded salt list. Reword the needle as a benign retrieval task (e.g. 'what is the project codename mentioned in the document?'). Run k≥3 needles and gate on ≥2/3. Classify refusals separately from misses.

**Est. impact:** Removes a false-fail path that currently rejects perf keeps. The DF5 keep would have landed.

**Validation:** Run the reworded probe 10 times on the current serve and expect 10/10 hits with 0 refusals.

**Risks:** A benign wording may test long-context retrieval less adversarially. Keep the injection-style probe as a separate safety check, not a perf gate.

**Verifier reasoning:** pr9.diff:3416-3433 shows salt S1788412615137064974, hit=0 and the refusal text ('classic prompt injection / data exfiltration pattern'). The trail at pr9.diff:3917 has 'FAILED gate ... model refused the prompt as injection'. Rebench needle-8192.txt passed with a different salt (S1788383589691374158), so salts vary per run. One nuance: the DF5 '+14%' was itself not significant on the old ruler (z of about 1.25). The case for DF5 rests on the step-time and acceptance factorization, so 'blocked a measured win' is somewhat overstated. The gate flaw is real regardless.

## PROTO-1: Cheap ruler v2: fixed-length multi-distribution decode panel with server-side step metrics

- kind=methodology component=bench_decode.py (extend) or in-image `vllm bench serve` impact=5 confidence=4 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 4)

**Claim:** A panel of about 25 minutes per boot (fast gate about 6 minutes) can measure decode, acceptance, step time, prefill and TTFT across realistic distributions, with a smallest detectable effect of 3-6%. The v11 image already exposes every needed metric: per-iteration ITL, per-position acceptance, prefill/decode/queue time histograms. min_tokens is honoured by the V2 runner under spec decode.

**Mechanism:** Holding length and prompts fixed removes length and text variance. Server-side ITL isolates GPU step time from HTTP/detokenize jitter. Distinct prompts restore realistic expert traffic.

**Evidence:**
- /tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-GLM-5-3-Flash-NVFP4-vLLM-2x-DGX-Spark/29f0f0df-0b0e-4224-a895-0ea5d735b7de/scratchpad/v11src/vllm/v1/metrics/stats.py:419-426 (ITL appended once per engine iteration per request = one verify step)
- v11src/vllm/v1/metrics/loggers.py:829-960 (inter_token_latency, request_time_per_output_token, request_prefill_time, request_decode_time, queue_time histograms; _sum/_count give exact means)
- v11src/vllm/v1/spec_decode/metrics.py:229-255 (num_drafts, num_draft_tokens, num_accepted_tokens, num_accepted_tokens_per_pos)
- v11src/vllm/v1/worker/gpu/sample/logit_bias.py:90-94,124-173 (min_tokens via expanded_idx_mapping, so it works with spec)
- v11src/vllm/benchmarks/serve.py:189-240,958-1100 (per-position acceptance), :1693 --num-warmups, :1744 --ignore-eos; v11src/vllm/benchmarks/datasets/datasets.py:1632,1692-1703 (spec_bench dataset, --spec-bench-output-len)

**Proposed action:** Panel, all with a warm-up request per shape, chat_template_kwargs pinned (enable_thinking, reasoning_effort) and min_tokens=max_tokens=N unless noted. A) prose-long: 8 distinct writing prompts, N=512, greedy. B) code: 8 prompts, N=512. C) thinking-on chat: 8 prompts, N=1024, window starts at the first reasoning delta. D) tool call: 4 Hermes-style turns, natural stop, report e2e plus ITL. E) long-context: 32k and 128k seeded-salt documents, N=256, report TTFT, prefill tok/s and ITL. F) vision: 1 image (smoke_vision asset), N=256. G) sampled: A with T=1.0, top_p=0.95, seed fixed. H) c=2 distinct pairs from A, plus c=2 identical as a labelled diagnostic. I) c=4 distinct for capacity (queue time). J) structured count, as an acceptance ceiling. K) legacy prose (the current prompt) as a continuity row. Per run, record deltas of spec counters including per_pos, inter_token_latency _sum/_count (step_ms), request_time_per_output_token, prefill_time, TTFT, sha256(text), completion_tokens. Record env per boot: docker inspect Cmd, image id, git sha, free -h both nodes, VmSwap of every VLLM::Worker pid, vmstat 1 si/so across the run. Fast gate = A+B at c=1 (8×512 at ~21 tok/s ≈ 195 s, plus 8×512 at ~30 tok/s ≈ 140 s).

**Est. impact:** Smallest detectable effect goes from ~31% to ~3-6% (PROTO-2). This covers the regimes the user cares about (vision on, long context, thinking).

**Validation:** First exclusive slot: run the panel twice on the unchanged default in 2 boots. Check that client tok/s ≈ acc_len/step_ms within 3%, and that boot-to-boot step_ms CV is ≤ 2% with the swap gate passing.

**Risks:** min_tokens bans EOS and may push text slightly out of distribution near the end (small, and use prompts whose natural length exceeds N). Panel time competes with the shared GPU schedule, so keep the fast gate for iteration and the full panel for publish.

**Verifier reasoning:** The metric sources exist in v11. stats.py:419-423 appends ITL once per non-prefill engine output per request, which is one verify step when async scheduling is off. spec_decode/metrics.py registers per-position counters (non-diffusion). logit_bias.py:90-94 applies min_tokens and :122-143 goes through expanded_idx_mapping per token. serve.py:1693 has --num-warmups and :1744 has --ignore-eos, and datasets.py:1692-1703 has spec_bench. Two things are unvalidated: the 3-6% MDE (see PROTO-2), and whether EOS-banned 512-token continuations of prompts that naturally stop at ~100 tokens produce representative acceptance. Prompts must be long-form. The fast-gate time math checks: 8x512/21 = 195 s and 8x512/30 = 137 s.

**Verifier corrected impact:** Mostly coverage plus variance reduction. The MDE claim depends on PROTO-2 and PROTO-6.

## PROTO-2: Noise model and decision rule: factorize tok/s = acceptance x steps/s, paired prompts, ABAB boots, swap gate

- kind=methodology component=hillclimb decision procedure impact=5 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Measured noise by component: acceptance per 3-run block CV 5.6% (≈ sampling SE 5.3% at 120 drafts). Step time boot-to-boot CV 2.2% (structured) to 3.7% (prose). Within-boot step CV is 0.2-2.7%. Per-run tok/s CV is 11% (98 tokens) vs 4% (200 tokens). Detecting +5% at 80% power needs SE(diff) ≤ 1.8%: about 76 runs per arm at CV 11%, 10 at CV 4%, 2.5 at CV 2%.

**Mechanism:** The two factors have different noise sources (text vs UMA/boot) and different change types act on only one of them. Lossless kernel work changes step_ms; drafter, k, quant or template changes change acceptance.

**Evidence:**
- Within-boot structured runs: rebench-dflash5 55.95/55.73/55.70 (0.2%), iter-h4 55.47/55.79/55.60 (0.3%), baseline 51.16/50.74/48.59 (2.7%)
- Boot-to-boot DF7 structured step ms: 116.0, 116.8, 117.9, 119.6, 121.7, 122.5 (CV 2.2%); prose 114.5-128.4 (CV 3.7%)
- Required n per arm = 2·(1.96+0.84)²·CV²/0.05²: CV 0.11 -> 75.9, 0.04 -> 10.0, 0.02 -> 2.5 (computed)
- Acceptance SE at 3,300 drafts (fast gate A: 8×512 tokens/2.4) = 1.0%; paired design removes between-prompt variance

**Proposed action:** Rule: for a lossless kernel or config change, the primary metric is step_ms (ITL) on cells A+B. Run 2 boots per arm in ABAB order and keep only runs where worker VmSwap is unchanged and si=so=0. Keep if the mean improves by more than 2×SE (expected SE about 1.5-2%). For acceptance-affecting changes, the primary metric is paired acc_len over the same prompts, with a bootstrap CI. Keep if the CI excludes 0 and composite tok/s improves. Always report both factors plus tok/s. Publish only full-panel numbers.

**Est. impact:** Smallest detectable effect ≈ 3-6% (step) and ≈ 2% (acceptance) at ~4 boots per A/B (each boot ~18 min plus a 6 min fast gate, so ~1.6 h per decision).

**Validation:** The first 2 default boots under PROTO-1 set the actual CVs. Recompute the required boots per arm from them.

**Risks:** Boot count is the dominant cost. If the swap gate fails often, UMA control (PROTO-6) is a prerequisite.

**Verifier reasoning:** The noise numbers reproduce. Within-boot structured CV is 0.2-0.3% (rebench-dflash5 55.95/55.73/55.70, iter-h4). DF7 structured step boot-to-boot (7.844/tok_s) is 116.0, 116.8, 117.9, 121.7 plus the b12x values: CV about 2.2%. n = 2(2.8)^2 CV^2/0.05^2 gives 75.9, 10.0 and 2.5. The MDE is optimistic, though. With 2 boots per arm and today's boot-level step CV of 2.2-3.7%, SE(diff) = CV, so the 80%-power MDE is 2.8 x (2.2-3.7%) = 6-10%, not 3-6%. Reaching 3-6% needs the swap gate to bring boot CV to about 1-1.5%, which is unshown. The rule itself (step_ms primary for lossless changes, paired acceptance for lossy ones) is sound. Existing data already shows it would have decided DF5 vs DF7 correctly.

**Verifier corrected claim:** The factorized decision rule is sound. At current boot-level CV it resolves about 6-10% step-time effects with 2 boots per arm. 3-6% needs PROTO-6 to cut boot variance first.

**Verifier corrected impact:** MDE about 6-10% now, about 3-6% if the swap gate halves boot CV.

## PROTO-3: Step-time byte-budget model supports the weight-bytes hypothesis and gives a k-selection objective

- kind=perf component=verify step (target TP=2 + DFlash2 drafter) impact=4 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Checkpoint headers give per-rank per-step reads at DF7 c=1 of about 26 GB: routed MoE 17.3 GB (uniform routing upper bound, 58.1 distinct experts × 42 layers × 7.08 MB), BF16 attention 6.0 GB, shared experts 1.08 GB, lm_head 0.63 GB, dense MLP 0.13 GB, and drafter ~1 GB. That is ≈ 96 ms at 273 GB/s, versus 114.5-128 ms measured. Measured marginal cost per verify token is 5.6-7.3 ms, close to the predicted +2.0 GB = 7.2 ms at peak (uniform routing). Step time is independent of acceptance (prose 114.5 vs structured 116.0 ms in one session). Combined with per-position acceptance, predicted prose tok/s by k is k=3 ~25-26, k=5 ~23.6-23.9, k=7 ~21.3, and structured k=5 ~57, k=7 ~67.5.

**Mechanism:** At c=1 with an 8-token verify on ~273 GB/s LPDDR5X, GEMVs over NVFP4 expert weights and BF16 attention weights dominate. The unexplained ~20-30 ms per step is candidate non-bandwidth overhead (NCCL latency for about 2 all-reduces per layer over RoCE, launch and scheduling gaps, the serial drafter, latency-bound KDA/MLA kernels) or below-peak achieved bandwidth. PROTO-4/5 separate the two.

**Evidence:**
- Header sums (CPU, headers only) of /home/sfxnz/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/09b04e5e74bca08ca8549fc736d4cdd8624bfde3/*.safetensors: routed experts 172.97 GiB, attention 11.17 GiB, shared experts 2.02 GiB, lm_head 1.18 GiB, embed 1.18 GiB, vision 1.05 GiB, dense MLP 0.24 GiB; expert 14.156 MB (layer 10)
- MoE bytes/rank: T=6 13.31 GB (48.8 ms at 273), T=8 17.27 GB (63.3 ms), T=16 31.06 GB (113.8 ms) (computed)
- Measured step: DF5 101.6/100.8/105.1/108.2 ms; DF6 109.8; DF7 114.5-128.4; MTP-4 113.2 (pr9.diff H2, acc 2.175)
- Draft /home/sfxnz/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/7d74cdd881ed7e32c31175984a67823127b66cfe: 2.2 GB, 5 layers, hidden 4096, block_size 8
- Fitted t(k) ≈ 63-69 ms + 6.0-6.8 ms·(k+1); per-position prose p = [0.668,0.450,0.236,0.085,0.044,0.018,0.000]

**Proposed action:** Use the model as the prior for planning. The largest byte buckets per step are MoE (≈65%) and BF16 attention (≈23%), so quantizing attention/shared-expert weights or cutting distinct experts per step have the biggest ceilings. Choose k by minimizing the workload-weighted time per token using measured per-position acceptance per distribution. Refit t0 and s from PROTO-1 step_ms at k in {3,5,7}.

**Est. impact:** Planning tool, no direct gain. Example ceiling: halving BF16 attention bytes (6.0 -> 3.0 GB) saves ≈11-14 ms/step ≈ 10-12% decode if bandwidth-bound (estimate).

**Validation:** PROTO-4 per-category kernel ms × bytes gives achieved GB/s per category. PROTO-5 gives measured distinct experts. The model is confirmed if Marlin MoE achieves ≥ 70% of 273 GB/s and step ≈ bytes/BW + fixed overhead within 10%.

**Risks:** Uniform routing overstates distinct experts. The attention byte count includes the MTP layer's attention, which is not read under DFlash. KV/indexer reads grow with context and are absent at ~100 tokens.

**Verifier reasoning:** Header check (scratchpad agents/hdr_regbench.py): the layer-10 expert is 14,155,800 B (gate/up U8 [2048,2048] + F8 [2048,256]; down [4096,1024] + [4096,128]). Routed experts are 159.47 GiB over 42 layers plus 13.5 GiB for the MTP layer. Attention is 11.32 GiB (0.23 GiB MTP), shared 1.97 GiB, lm_head 1.18 GiB, vision 1.05 GiB. The MoE bytes/rank arithmetic is 58.1 x 42 x 7.08 MB = 17.27 GB, which is right. The measured marginal cost per slot is 7.0-8.3 ms on same-day pairs (DF5 -> DF6 h1: 3.071/27.96 = 109.9 vs 101.6; DF5 -> DF7: 7.3 and 7.0/slot) but only 3.2 ms/slot on cross-boot pairs, broadly consistent with 7.2 ms predicted. Two caveats. (1) The 'unexplained 20-30 ms' assumes 273 GB/s peak. At a more realistic achieved bandwidth of about 210-230 GB/s, 26 GB alone is 113-124 ms, which could explain the whole step. So 'overhead vs bandwidth' is undetermined without PROTO-4. (2) The model's DF5 prose prediction (23.6-23.9) overshoots the measured 21.76-21.97 by about 8%. Also, the measured steps are on the LibertAI pack, where dense MLP is BF16, not the nvidia pack.

**Verifier corrected claim:** The byte model is consistent with the k-dependence of step time. It cannot distinguish 'fixed overhead' from 'below-peak achieved bandwidth' for the remaining ~20-30 ms, and it overpredicts measured DF5 prose by about 8%.

## PROTO-4: nsys protocol for a kernel-level breakdown of one CUDA-graph verify step on both ranks

- kind=methodology component=run.sh (profiling wrapper, proposal only) + vLLM --profiler-config impact=5 confidence=4 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 4)

**Claim:** The v11 image can bracket an nsys capture from the API. --profiler-config '{"profiler":"cuda"}' registers /start_profile and /stop_profile, which call torch.cuda.profiler.start/stop on every worker, and nsys --capture-range=cudaProfilerApi records just that window. The host has nsys/ncu under /usr/local/cuda-13.0/bin and /opt/nvidia/nsight-systems. Whether nsys exists inside the image is unverified (I did not inspect it), so bind-mount the host copy. --cuda-graph-trace=node exposes the kernels inside FULL graphs.

**Mechanism:** Only a kernel timeline shows whether the ~115 ms step is bandwidth-bound MoE/attention, NCCL latency, drafter serial time, or CPU/scheduling bubbles. torch-profiler traces show only graph launches under FULL graphs.

**Evidence:**
- /tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-GLM-5-3-Flash-NVFP4-vLLM-2x-DGX-Spark/29f0f0df-0b0e-4224-a895-0ea5d735b7de/scratchpad/v11src/vllm/config/profiler.py:16-145 (profiler 'cuda', delay_iterations, max_iterations)
- v11src/vllm/entrypoints/serve/profile/api_router.py:21-45 (routes attached only when a profiler is set)
- v11src/vllm/profiler/wrapper.py:472-490 (CudaProfilerWrapper -> torch.cuda.profiler.start/stop; NVTX ranges)
- host: /usr/local/cuda-13.0/bin/nsys, /usr/local/cuda-13.0/bin/ncu, /opt/nvidia/nsight-systems present
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/run.sh:311-350 (docker run args; the model runs in mp worker subprocesses)

**Proposed action:** First exclusive slot. (1) Add a PROFILE=1 knob that bind-mounts the host nsys read-only and wraps 'vllm serve' as: nsys profile -t cuda,nvtx,osrt -s none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --trace-fork-before-exec=true -o /data/prof/glm-rank<R> vllm serve ... --profiler-config '{"profiler":"cuda"}', on both ranks. (2) Warm up with 3 panel-A requests. (3) POST /start_profile, send 1 c=1 request with N=64 fixed tokens (≈27 verify steps), POST /stop_profile. (4) Repeat with --cuda-graph-trace=graph for undistorted step durations, and once at c=2 distinct. (5) Export: nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum and nsys export --type sqlite. Delimit steps by cudaGraphLaunch or NVTX on the model-runner thread. Bucket kernels by regex (marlin|moe; nccl|AllReduce; mla|fmha|flashinfer; indexer|paged_mqa|deep_gemm; chunk|recurrent|kda|fla|conv; gemm|gemv|cutlass|nvjet; topk|router; sinkhorn|hc; norm; memcpy|copy; sample|reject; drafter NVTX). Compute GPU-idle per step and rank0-vs-rank1 NCCL skew. Join with the PROTO-3 byte buckets to get achieved GB/s. (6) Do not run ncu inside the TP=2 serve: kernel replay would stall the partner rank's NCCL. Run ncu --set full -k regex:marlin -c 5 on a TP=1 single-layer Marlin MoE microbench (8 tokens × top-8 of 288, NVFP4) instead.

**Est. impact:** Turns the 'kernel-level optimisation' ask into a ranked list with ms per step. Expected to locate the ~20-30 ms of non-bandwidth time (estimate).

**Validation:** The sum of bucketed kernel time plus idle equals the graph-mode step within 5%, and the graph-mode step equals ITL step_ms within 5%.

**Risks:** Node-level graph tracing adds overhead, so use it only for proportions. The nsys binary must match container glibc/arch (both aarch64). Trace size is fine for a ~30-step window. The profiler is dev-only (api_router warns), so never leave it in the default run.sh.

**Verifier reasoning:** Verified: config/profiler.py:16,42 has ProfilerKind 'cuda'. profiler/wrapper.py:472-490 has CudaProfilerWrapper calling torch.cuda.profiler.start/stop plus NVTX. Host tools exist: /usr/local/cuda-13.0/bin/nsys and ncu, and /opt/nvidia/nsight-systems and nsight-compute. Not verified: that /start_profile reaches the headless rank-1 worker on spark2 through collective RPC, and that nsys inside a container with a bind-mounted host install works under the ssh-launched rank-1 container. The advice against ncu inside TP=2 is correct.

## PROTO-5: Routed-expert capture boot to measure distinct experts per verify step (tests the bytes hypothesis and c=2 sharing)

- kind=methodology component=vLLM --enable-return-routed-experts on Glm5NextMoE impact=4 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The image has a routed-experts capturer and an --enable-return-routed-experts flag. Glm5NextMoE builds its experts through FusedMoEFactory, whose router exposes capture_fn, so per-token top-8 expert ids can probably be returned. That turns E[distinct experts per (k+1)-token window] and c=2 overlap into measured facts.

**Mechanism:** If real routing shows locality (distinct ≪ 58 at T=8), then MoE bytes are lower than modelled and the step is more overhead-bound. If distinct ≈ uniform, speculation slots and c=2 users are bandwidth-expensive, and expert-deduplicating kernels or smaller k are the lever.

**Evidence:**
- /tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-GLM-5-3-Flash-NVFP4-vLLM-2x-DGX-Spark/29f0f0df-0b0e-4224-a895-0ea5d735b7de/scratchpad/v11src/vllm/engine/arg_utils.py:427,868-869
- v11src/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py:34-60
- v11src/vllm/model_executor/layers/fused_moe/router/base_router.py:183-187 (capture_fn)
- v11src/vllm/models/glm5next/nvidia/model.py:153,227 (Glm5NextMoE uses FusedMoEFactory)

**Proposed action:** One diagnostic boot (not the default): add --enable-return-routed-experts, run panel A/B at c=1 and c=2 distinct, and compute on CPU the distinct experts per layer per step (k+1 accepted-or-not verify tokens) plus overlap across streams. Compare with 288·(1-(1-8/288)^T).

**Est. impact:** Removes the biggest uncertainty (±30%) in the step-time model. It decides between 'fewer bytes' (k, expert dedup) and 'less overhead' (NCCL, launch, drafter) as the next kernel target.

**Validation:** The flag boots and returns ids for the fork model. A sanity check is exactly 8 ids per token per MoE layer (42 layers).

**Risks:** The fork model may not wire capture (verify first). The return path adds D2H traffic, so do not use this boot's timings. It adds UMA memory proportional to max_num_batched_tokens × 42 × 8 × 4 B (small).

**Verifier reasoning:** arg_utils.py:427,868-869 exists. The capturer is initialised from gpu_worker.py:668-669 via model_runner.init_routed_experts_capturer, and both v1/worker/gpu_model_runner.py and v1/worker/gpu/model_runner.py reference routed_experts. glm5next model.py:227 uses FusedMoEFactory. The scheduler refuses it with context parallelism (scheduler.py:329-331), which is not used here. Whether capture works with DFlash2 verify batches and the fork's mHC MoE path is unverified, as the reviewer says.

## PROTO-6: Make UMA/swap state a recorded, gated measurement precondition

- kind=ops component=run.sh maybe_drop_caches / host vm settings / bench harness impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Every published or comparative session ran with 6-7.3 GiB of swap in use and about 5 GiB available (vm.swappiness=60). Swapped worker pages cost 4-13% per step in same-boot comparisons. maybe_drop_caches no-ops without passwordless sudo, so boots start with different page-cache states. None of this is recorded per bench run.

**Mechanism:** On GB10 UMA, CPU swap-out of worker or API pages and page-cache pressure add page faults and lower effective bandwidth during decode. The state differs per boot, which adds boot-to-boot variance.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/rebench-20260902T204243Z/free-after-bench.txt (swap 6 used, available 5)
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-b12x/evidence/iter-b12x-A-marlin/postboot.txt (115Gi used, swap 7.3Gi, available 5.7Gi)
- README.md:22 (Worker_TP0 ~1.3 GiB swapped; first wave 52 vs 59.9 = -13%)
- iter-h-snap wave1->wave2 step +4-5% (decision.tsv:8 '(swap)')
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/run.sh:177-180,238 (drop_caches via sudo -n)
- /proc/sys/vm/swappiness = 60 (read now)

**Proposed action:** Harness: before and after each cell, record VmSwap from /proc/<pid>/status for every VLLM::Worker and APIServer on both nodes, plus free -h, and run vmstat 1 during the cell. Mark a run invalid if si or so > 0 during the cell or worker VmSwap grew. Host (needs the owner's decision and sudo): consider vm.swappiness 1-10 on both Sparks and a passwordless drop_caches rule for the recipe user. This is out of scope for this read-only stage and must not be applied while the other workload runs.

**Est. impact:** Removes an estimated 2-4% of boot-to-boot step variance and 4-13% outliers. It is a prerequisite for the 3-6% smallest detectable effect in PROTO-2.

**Validation:** Two boots with the gate enforced should show step_ms CV ≤ 1.5% (vs 2.2-3.7% now).

**Risks:** Lower swappiness raises OOM risk at ~115/121 GiB used, which ties into the KV-pin and vision-on UMA margins. Host changes affect the co-resident workload, so coordinate first.

**Verifier reasoning:** Confirmed /proc/sys/vm/swappiness = 60. rebench free-after-bench.txt shows used 116, available 5, swap 6. b12x postboot shows swap 7.3 GiB. run.sh:177-181 gates maybe_drop_caches on sudo -n. But the b12x A arm is a counterexample to 'swap in use costs 4-13%': with 7.3 GiB swapped, wave1 and wave2 step were 120.2 vs 120.3 ms. The h-snap +4-5% drift is labelled '(swap)' in decision.tsv:8 without any VmSwap or vmstat measurement, so the causal attribution is asserted, not measured. Gating on active si/so (not on the VmSwap level) is the right form.

**Verifier corrected claim:** UMA/swap state should be recorded and gated. Existing receipts show swap residency alone does not predict slowdown (b12x: 7.3 GiB swapped, step stable). Only active paging during a cell is a plausible cause.

**Verifier corrected impact:** Unknown, estimated at 0-5% of boot variance. It is a prerequisite for sub-5% MDE only if paging activity is actually present.

## Open questions
- What workload mix should define 'substantial improvement' (prose/chat thinking-off vs thinking-on vs code vs structured/tool vs long-context)? The best DFlash2 k depends on it: prose favours k≈3-5, counting favours k=7.
- Is greedy prose nondeterminism (acceptance 2.28 vs 2.51 in one boot) caused by batch-variant kernels such as Marlin split-K, NCCL reduction order or FA2 tiling, or by the drafter? Two identical requests in one boot compared by sha256 would tell.
- Does the nvidia pack (PR #11, served as W4A16 by Marlin) change prose/structured acceptance or step time relative to the LibertAI weights? Every published cell is still LibertAI.
- Does --enable-return-routed-experts actually work with the fork Glm5Next model, and what is the real distinct-expert count per 8-token verify window?
- Does vision-on (the encoder cache plus MM-profiling footprint) cost text decode through UMA pressure? This needs an on/off A/B at the same KV pin.
- Is the ~38-52% lower prefill tok/s at 16-17k tokens (884/690 vs 1425 at 10k in the MTP-4 era) real? If so, is it the drafter prefill, the 2048-token chunking forced by the spec settings, or UMA?
- Is num_speculative_tokens adjustable per request or adaptively in this vLLM build? That would allow high k for structured and low k for prose without choosing one default.
- Is nsys already present inside glm53-sm121-v11, and does the host's aarch64 nsys run inside the container's glibc when bind-mounted?

## Verifier: missed issues
- PR #11's first measurement will change several things at once, against AGENTS.md's one-knob rule: model LibertAI -> nvidia, vision on, and bench_decode.py:166's default --model changed to nvidia/GLM-5.3-Flash-NVFP4. It also adds a new dense-MLP kernel path. The nvidia pack's layers 0-2 dense MLP are NVFP4 W4A4, where the LibertAI config.json lists *.mlp.gate_up_proj/down_proj as ignored, i.e. BF16. On sm_121, init_nvfp4_linear_kernel auto-selects the first supported kernel from v11src/vllm/model_executor/kernels/linear/__init__.py:500-511. CuteDsl needs sm_10x (nvfp4/flashinfer.py:38), but FlashInferCutlassNvFp4LinearKernel only needs has_device_capability(100) (flashinfer.py:106-119), so it is picked. With run.sh:251 FLASHINFER_CUDA_ARCH_LIST=12.1a, that implies a runtime FlashInfer CUTLASS JIT, the same toolchain that global-OOM'd spark2 for MoE (README.md:149). Its speed is also unmeasured. Recommend a bench-only A/B that separates model swap from vision (FORCE_UNSAFE_VISION=1 LANGUAGE_MODEL_ONLY=1 for one arm), plus a log check for 'Using ... for NVFP4 GEMM' on first boot.
- The DF5 vs DF7 decision is already resolvable from existing receipts with the factorized view, which no one applied. The same-night pair gives DF5 step 105.1 ms at acc 2.308 (pr9.diff iter-H1 bench.json) vs DF7 parity 119.1 ms at acc 2.292. On tok/s alone the +14% has z of about 1.25 (not significant), but step time differs by about 12% at equal acceptance, well outside the 2-4% boot CV.
- Structured c=1 already serves as a nearly noise-free step-time probe for lossless kernel changes. Acceptance is bit-deterministic (7.8441558 in 4 sessions), within-boot tok/s CV is 0.2-0.3% (rebench-dflash5 55.95/55.73/55.70, iter-h4 55.47/55.79/55.60), and the step equals prose at the same verify shape (116.0 vs 114.5 ms in one session). Using it as a step_ms ruler, not as a published decode score, would give about 1% resolution today, before any new harness. This is a justified exception to AGENTS.md's 'do not score decode from structured'.
- The CoT-era eager ladder in how-explanation.md:133 ('DF4 acc 2.8, c=1 -> 20.6', i.e. about 136 ms/step at a shorter verify) contradicts the linear step model. The reviewer mentions this only as a risk. It suggests truncating the trained 8-block drafter may have non-bandwidth costs or large acceptance loss (observed DF5 2.33 vs per-position prediction 2.48, -6%). Any k<5 proposal needs direct measurement, not modelled 25-26 tok/s.
- README.md:22 and b12x result.txt use 'wave2' as the published ruler after a first wave. That selection rule is not justified by data: in h-snap wave2 was slower and in b12x wave2 was faster. It adds another source of inconsistency between published cells.
