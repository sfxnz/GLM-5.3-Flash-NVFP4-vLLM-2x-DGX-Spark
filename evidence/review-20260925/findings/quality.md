# Dimension: quality

## Reviewer summary

Output-quality review of the nvidia/GLM-5.3-Flash-NVFP4 @09b04e5 recipe (PR #11 head 29954c8) with vision on. The biggest quality risk is the server-wide default `enable_thinking:false`. The local chat_template.jinja differs from the official template in exactly one place, the generation prompt: it emits `<think></think>`. The official template is byte-identical across the nvidia and all three LibertAI snapshots (md5 91576a54…) and always opens `<think>`. The GLM-5.3 card documents only reasoning_effort low/high/max. So thinking-off is an off-distribution prompt shape. Another 2x Spark recipe reports intermittent token corruption with it (MiaAI-Lab #257), fixed by reasoning_effort=low. vLLM #54744 covers the related parser/template mismatch. Every NVIDIA accuracy number is thinking-on, Max effort, temp 1.0. Two parser/template mismatches were verified in the v11 source. First, the parser accepts `thinking` but the template ignores it, so a `thinking:true` request loses its whole answer into `reasoning`. Second, the reasoning_effort values medium, minimal and xhigh silently map to Max. The template renders history, tools and images exactly like the official template; I checked this by rendering it with jinja2 on the CPU. Prior reasoning is dropped unless the client echoes `reasoning`, and the card recommends clear_thinking=true for chat, which the recipe does not set. On numerics, the SwiGLU clamp (swiglu_limit 10) is applied in the Marlin MoE path, and it matters: the calibrated down_proj input amax sits exactly at 100 (=10x10) in 9 layers. Marlin runs the pack as W4A16 and ignores all 36,297 calibrated input_scales. So on Marlin the nvidia pack is about LibertAI's weight-only experts plus three more quantized layers: dense MLP 0-2, about 453M params. Those dense layers also add a new NVFP4 linear kernel path, auto-selected and not pinned, that LibertAI never exercised on this image. That makes the quality case for PR #11 unproven. The fp8 KV cache uses a per-tensor scale of 1.0 with no calibration; NVIDIA's AA-LCR shows no loss on GB200, but the patched sm_121 sparse-MLA fp8 path is unvalidated beyond one needle near the end of the prompt. KDA recurrent state is fp32 and the indexer fp8 path uses Hadamard with per-token scales, so both are low risk. The recipe has no accuracy evaluation. All probes run greedy with explicit enable_thinking:false. None of them exercises the server-default path, temp 1.0 sampling (the default from generation_config), thinking-on parsing, streaming tool calls, long-context depth, or vision correctness; the vision smoke test sends a 1x1 image. I propose a two-tier gate. Tier 0 runs in about 15 GPU-minutes: teacher-forced NLL and top-1 agreement via prompt_logprobs, a greedy byte-exact set, probes of the default path, and a corruption probe. Tier 1 runs in about 2.5-3 GPU-hours per config: IFEval, GSM8K, MMLU-Pro, BFCL-style tools, RULER NIAH and ChartQA/OCR/MMMU. It is paired across configs with McNemar/bootstrap CIs and catches regressions of 5pp or more; Tier 0 catches sub-point numeric drift.

## QUAL-1: Server-wide enable_thinking:false is an unsupported, off-distribution prompt shape for GLM-5.3-Flash

- kind=quality component=run.sh default-chat-template-kwargs + chat_template.jinja generation prompt impact=5 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The recipe forces every request that does not opt in into a `<|assistant|><think></think>` generation prompt. GLM-5.3's official template has no enable_thinking branch: it always opens `<think>` and exposes only reasoning_effort (low/high/max) and clear_thinking. Thinking-off is therefore outside what the model was tuned for. All published NVIDIA/Z.ai quality numbers are thinking-on, Max effort, temp 1.0. The best-documented failure of this exact shape is intermittent token corruption inside numbers at the default temperature 1.0.

**Mechanism:** The model conditions on an empty think span it was not trained to produce at the generation position. Z.ai removed the GLM-4.x enable_thinking path in GLM-5.3 and replaced it with effort levels. At temperature 1.0 and top_p 0.95 (auto-applied from generation_config), low-confidence off-distribution states sample stray tokens. Greedy probes hide this.

**Evidence:**
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/run.sh:345 --default-chat-template-kwargs '{"enable_thinking": false}'
- /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/chat_template.jinja:255-262 (only diff vs Hub: enable_thinking -> '<think></think>')
- Hub template .../models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/09b04e5.../chat_template.jinja:255-257 always '<think>'; md5 91576a54cc089fb7086d95d81f8fda30 identical for nvidia and all 3 LibertAI snapshots
- nvidia README.md:205 benchmarks at temperature=1.0, top_p=0.95, max_new_tokens=327,680 (thinking on)
- https://huggingface.co/zai-org/GLM-5.3-Flash : reasoning_effort low/high/max (default max); no non-thinking mode documented
- https://recipes.vllm.ai/zai-org/GLM-5.3-Flash : 'Thinking is unconditionally enabled—generation always opens with a <think> block'
- https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/257 : with default enable_thinking:false, foreign tokens were injected inside numbers in numeric markdown tables (e.g. '1 147,应该如何 97'); 'Thinking-off appears to be an unsupported prompt shape'; fixed by reasoning_effort low
- https://github.com/vllm-project/vllm/issues/54744 (open): GLM-5.3 template never reads enable_thinking/thinking
- evidence/decision.tsv h4: thinking-off was adopted as a leak fix, validated only by greedy format probes (count 200/200, leaked_think=0)

**Proposed action:** Keep the local template, since it stays consistent for clients that explicitly send enable_thinking:false. Change the server default to in-distribution thinking: --default-chat-template-kwargs '{"reasoning_effort":"low","clear_thinking":true}' and drop enable_thinking from the defaults. The template defaults it to true, and the parser defaults to thinking when both kwargs are None (glm47_moe.py:185-191), so the two stay consistent. Document per-request `thinking_token_budget` as a cap. It works because ReasoningConfig is auto-built from --reasoning-parser (engine/arg_utils.py:2703-2708; v1/engine/input_processor.py:111-118). Keep enable_thinking:false as an opt-in fast lane. The published decode ruler stays explicit thinking-off, because bench_decode.py passes the kwarg.

**Est. impact:** Quality impact is estimated, not measured on Sparks. It removes an unsupported mode on 100% of default requests and a reported corruption mode. The latency cost is the added reasoning tokens: at the measured 21.2 tok/s c=1 prose, 100-300 'low' reasoning tokens add about 5-14 s per reply (100/21.2=4.7 s; 300/21.2=14.2 s). The '#257: no latency cost' claim is unverified here.

**Validation:** Tier-1 paired run, one knob only: A = current default (thinking-off) vs B = reasoning_effort low, on IFEval-150, GSM8K-200, MMLU-Pro-280 and a 50-sample temp-1.0 numeric-table corruption probe (regex for CJK or letters inside digit groups). Record mean completion and reasoning tokens as well as wall time.

**Risks:** Reasoning tokens raise latency and cost context. Clients that only read `content` are unaffected, but clients that show reasoning will change. The published prose ruler must keep sending enable_thinking:false explicitly or it stops being comparable.

**Verifier reasoning:** Checked the structural facts; they hold. `diff chat_template.jinja <hub>` shows the only difference is lines 256-261 (the enable_thinking branch). The Hub template is md5 91576a54cc089fb7086d95d81f8fda30 for nvidia 09b04e5 and all three LibertAI snapshots (aa28e1f, 11d7321, caca4e6). run.sh:345 sets `--default-chat-template-kwargs '{"enable_thinking": false}'`. The zai-org/GLM-5.3-Flash card lists reasoning_effort low/high/max (default max). WebFetch found no occurrence of 'non-thinking', 'thinking mode' or 'enable_thinking' on the card. recipes.vllm.ai says 'Thinking is always on — the generation prompt opens a <think> block unconditionally'. vLLM #54744 is open, with linked PRs #54825 and #56994.

Three parts are overstated. (1) The template itself renders prior assistant turns as `<|assistant|><think></think>content` (chat_template.jinja:149-153, whenever clear_thinking or a missing reasoning applies). So the empty think span is a shape the model sees in history. 'Off-distribution at the generation position' is plausible but not proven. (2) MiaAI #257 is a different stack: EXL3 quant, FP8, 850k context. It also reports corruption with thinking off at temperature 1.0, 0.6 AND 0. The reviewer's mechanism ('at temperature 1.0 ... Greedy probes hide this') is contradicted by that: greedy also corrupted there. The recipe's greedy probes miss it because none of them generates long numeric tables, not because they are greedy. (3) The nvidia README:205 gives only temp 1.0 / top_p 0.95 / max_new_tokens 327,680. It does not say 'thinking on, Max effort'; that is inferred from the template default.

The proposed action is code-consistent. With default kwargs {reasoning_effort:low, clear_thinking:true}, build_chat_params sets reasoning_effort=None when the request omits it, and merge_kwargs drops None (renderers/params.py:40), so the default survives. The parser sees neither thinking nor enable_thinking and defaults to thinking (glm47_moe.py:187-191). One side effect is missed. bench_decode.py:39 sends only enable_thinking:false, so under the new default its prompt changes from `<|system|>Reasoning Effort: Max` to `...Low` (chat_template.jinja:2-3). The published ruler therefore stops being byte-comparable unless the bench also pins reasoning_effort.

**Verifier corrected claim:** Thinking-off (`<think></think>` at the generation position) is not a documented mode for GLM-5.3-Flash. The card offers only low/high/max effort, and the official template always opens `<think>`. So the server-wide default is an unsupported prompt shape, and on every default request it also carries a contradictory `Reasoning Effort: Max` system prefix. A corruption mode tied to thinking-off was reported on a different 2x Spark stack (EXL3 #257), where it occurred even at temperature 0. It has not been observed or tested on this NVFP4/vLLM stack.

**Verifier corrected impact:** Unmeasured on this stack. It could be anywhere from nothing to intermittent corruption on 100% of default requests. The latency figure (+4.7 to 14.2 s per reply for 100-300 low-effort reasoning tokens at 21.2 tok/s) is arithmetic on assumed token counts. The proposed default also changes the bench prompt (Effort: Low instead of Max), so the published ruler must pin reasoning_effort explicitly to stay comparable.

## QUAL-2: Parser/template kwarg mismatch: `thinking:true` sends the whole answer into `reasoning`; /v1/messages `thinking` is ignored

- kind=correctness component=vllm/parser/glm47_moe.py (glm45/glm47 alias) vs chat_template.jinja impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The Glm47Moe parser, which both glm45 and glm47 resolve to, enables reasoning extraction if `thinking` OR `enable_thinking` is truthy. The template reads only `enable_thinking`. With the server default enable_thinking:false, a client sending chat_template_kwargs {"thinking":true} gets a merged {thinking:true, enable_thinking:false}. The prompt gets `<think></think>`, but the parser starts in REASONING state. No `</think>` is ever emitted, so the entire answer is returned as `reasoning` and `content` is empty. adjust_initial_state_from_prompt is a no-op, so the parser never looks at the prompt. The Anthropic endpoint has no `thinking` request field, so `thinking:{type:enabled}` is silently dropped; only output_config.effort maps. Separately, the tool-parser instance is built without chat_template_kwargs, so it always carries THINK terminals.

**Mechanism:** Two sources of truth for thinking: the parser's state machine and the template's generation prompt. Any disagreement makes the parser misclassify output.

**Evidence:**
- v11src/vllm/parser/glm47_moe.py:184-191 (thinking_enabled = bool(thinking) or bool(enable_thinking)); :125 initial_state=REASONING if thinking
- v11src/vllm/parser/engine/parser_engine.py:185-193 adjust_initial_state_from_prompt returns without doing anything
- v11src/vllm/entrypoints/openai/chat_completion/serving.py:180-190,243-250 parser gets merged default+request kwargs
- v11src/vllm/renderers/params.py:28-40 merge_kwargs (request overrides default; None dropped)
- v11src/vllm/parser/abstract_parser.py:124-128 tool_parser_cls(tokenizer, tools) gets no chat_template_kwargs
- v11src/vllm/entrypoints/anthropic/protocol.py:120-165 no `thinking` field; serving.py:513-514 only output_config.effort -> reasoning_effort
- chat_template.jinja:257 reads only enable_thinking
- LibertAI caca4e6 README: claims vLLM glm45 'silently discards the whole reply' with prompt-side <think>. This is contradicted by v11 code (initial REASONING state) but has never been tested here with thinking on.

**Proposed action:** Make the local template use the parser's rule: set `_think = enable_thinking|default(true)`, then if `thinking is defined` set `_think = thinking or enable_thinking|default(false)`, so both treat thinking as an alias. Also add a probe for each kwarg shape ({}, {enable_thinking:false}, {enable_thinking:true}, {thinking:true}, reasoning_effort=low/none, /v1/messages with output_config.effort) that asserts non-empty content, reasoning present iff thinking, and no '<think>'/'</think>' in content, in both streaming and non-streaming mode.

**Est. impact:** Affects every client that uses the `thinking` kwarg (DeepSeek/SGLang-style clients) or Anthropic-style thinking. Those clients get 100% of answers routed to `reasoning` or get thinking off unexpectedly. No effect on clients using enable_thinking or reasoning_effort.

**Validation:** Run the kwarg-matrix probe (12 cells: 6 kwarg shapes x stream on/off) against a live serve. Most of it is pure template and parser logic, so it can be pre-checked on the CPU by rendering the template and feeding synthetic outputs to Glm47MoeParser.extract_reasoning.

**Risks:** Adds more divergence from the Hub template; keep the change tiny and documented. A future upstream parser fix (vLLM PR #54825/#56994) may change semantics again.

**Verifier reasoning:** Traced in the v11 source. glm47_moe.py:184-191: thinking_enabled = True if both kwargs are None, else bool(thinking) or bool(enable_thinking). :125 sets initial_state=REASONING if thinking. serving.py:243-251 builds the parser with _effective_chat_template_kwargs = build_chat_params(...).with_defaults(server defaults). params.py:115-118 merges the request over the defaults. So {thinking:true} plus the server default {enable_thinking:false} gives both keys. The template (chat_template.jinja:257) reads only enable_thinking and emits `<think></think>`, while the parser starts in REASONING. parser_engine.py:185-187 adjust_initial_state_from_prompt is a bare `return`. extract_reasoning (:493-518) calls _reset() into the config's REASONING state. get_streaming_fallback_content returns None (:620-625), and streaming_parser_engine.finish (:246-275) has no 'reasoning never ended, so reclassify as content' fallback. The whole answer therefore lands in `reasoning` unless the model emits `</think>` on its own. The Anthropic protocol has no request `thinking` field; only output_config.effort maps (anthropic/protocol.py:116, serving.py:513-514). abstract_parser.py:128 builds tool_parser_cls(tokenizer, tools) without kwargs, which is correct as stated. However, tool_call_probe passes with thinking off, and REASONING has a TOOL_START transition (glm47_moe.py:146-149), so that sub-point has no demonstrated consequence.

**Verifier corrected claim:** Confirmed as stated for chat completions. A request with chat_template_kwargs {thinking:true} against this server's default enable_thinking:false gets a `<think></think>` prompt but a parser in REASONING state with no fallback, so content comes back empty and the answer lands in `reasoning`. The tool-parser-without-kwargs sub-point has no demonstrated effect.

**Verifier corrected impact:** This affects only clients that send `thinking` rather than `enable_thinking`/`reasoning_effort` (a subset). It is likely also to block structured-output grammar activation for those requests, since reasoning never 'ends'. That is not verified.

## QUAL-3: PR #11 adds an unpinned NVFP4 dense-linear kernel path (layers 0-2) that the LibertAI pin never exercised

- kind=quality component=ModelOpt NVFP4 linear (dense MLP layers 0-2) kernel auto-selection on sm_121 impact=4 confidence=3 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The nvidia pack quantizes the three dense MLP layers (0-2, intermediate 12288) to NVFP4 W4A4 with calibrated input_scale. LibertAI kept them BF16. vLLM auto-selects the NVFP4 linear kernel from a priority list. On sm_121, FlashInferCutlassNvFp4LinearKernel passes has_device_capability(100) and would be tried first if cutlass_fp4_supported(); otherwise FlashInferB12x, Cutlass or Marlin. Each kernel has different numerics (W4A4 with input_scale vs W4A16), and some need runtime JIT on 12.1a. The recipe pins --moe-backend marlin but not --linear-backend, so these three layers run on a kernel nobody has recorded or validated here. Every token passes through them.

**Mechanism:** A bug in an early-layer GEMM, or a scale mis-application, propagates through all 45 layers and 4 mHC streams. W4A4 activation quantization adds error that NVIDIA validated only on GB200 kernels. JIT on UMA can also OOM.

**Evidence:**
- hf_quant_config.json / config.json ignore list: layers.0-2 exclude only self_attn*, not mlp (config.json:326-358)
- safetensors headers (scratchpad/agents/quality_hdr.py): model.language_model.layers.0.mlp.gate_proj.weight U8 [12288,2048], weight_scale F8_E4M3 [12288,256], input_scale F32 (shards 13-29)
- index grep: 3x layers.N.mlp.{gate,up,down}_proj.input_scale
- v11src/vllm/model_executor/kernels/linear/__init__.py:500-511 priority list; :980-1040 auto-select, a16 forced only for weight-only
- v11src/vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:105-119 (cutlass_fp4_supported and has_device_capability(100))
- LibertAI caca4e6 README 'What's quantized': dense/MTP MLP BF16
- evidence/decision.tsv h-cutlass-oom: flashinfer_cutlass JIT (cudafe++) OOM'd spark2

**Proposed action:** Before any Spark run of PR #11, decide and pin: `--linear-backend marlin` gives W4A16 parity with the MoE path and no JIT, and is the conservative choice. Otherwise, grep the boot log for 'Using ... for NVFP4 GEMM' and record it in evidence/. Gate the choice with Tier-0 NLL/agreement against the LibertAI pin.

**Est. impact:** Dense MLP 0-2 is 3 x 3 x 12288 x 4096 = 453M params, about 2.5% of the 18B active path, but it is on every token. A broken kernel means a global quality collapse; correct W4A4 means about 0 to -0.5pp by NVIDIA's table. Pinning Marlin removes a JIT-OOM risk class.

**Validation:** One knob: A = PR #11 as-is (record the chosen kernel) vs B = --linear-backend marlin. Compare Tier-0 teacher-forced NLL and top-1 agreement on a 300k-token corpus, plus greedy byte-exact on 30 prompts.

**Risks:** Marlin linear repack costs memory. It is small for three layers (about 0.23 GB of FP4 weights per rank), but check UMA headroom (115/121 GiB used).

**Verifier reasoning:** The nvidia index has layers.0-2.mlp.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2,input_scale}. hf_quant_config.json exclude_modules has only layers.0/1/2.self_attn* for those layers. .quant_summary.txt:520-522 shows layers.0.mlp.down_proj input/weight quantizers enabled (MaxCalibrator). LibertAI caca4e6 index has only BF16 layers.0-2.mlp.*.weight, and its README says dense MLP is BF16. model.py:122-130 builds Glm5NextMLP with quant_config, while attention is hardcoded quant_config=None (:331). The linear priority list (kernels/linear/__init__.py:500-511) is: CuteDsl (is_device_capability_family(100), false on sm_121), then FlashInferCutlass (cutlass_fp4_supported() and has_device_capability(100), a >= check that sm_121 passes if _C reports FP4 support), then B12x (capability >= 120 and has_flashinfer_b12x_gemm), then Cutlass, then Marlin. Which one is picked cannot be decided on the CPU. No boot of the nvidia pack exists in any evidence: the b12x A/B engine logs load LibertAI caca4e6 (iter-b12x-B-flashinfer/engine-spark1.log). run.sh has no --linear-backend, and arg_utils.py:1596 shows the flag exists. Arithmetic: 3 layers x 3 projections x 12288 x 4096 = 452,984,832 params. That is ~0.23 GB of FP4 in total, so ~0.11-0.12 GB per rank at TP=2, not 0.23 GB per rank. The '2.5% of 18B active' figure was not verified.

**Verifier corrected claim:** PR #11 introduces NVFP4 W4A4 dense-MLP GEMMs on layers 0-2 (453M params, with calibrated input_scale) that the LibertAI pin never had. The linear kernel is auto-selected, is not pinned, and has never been booted or recorded on sm_121 with this image.

**Verifier corrected impact:** Magnitude unknown. A correct W4A4 path should cost less than 0.5pp; a bad kernel or JIT would fail globally or OOM. The Marlin linear repack is about 0.12 GB per rank, not 0.23 GB.

## QUAL-4: On Marlin the nvidia pack gives no quality advantage over LibertAI; the calibrated W4A4 scales are unused (SwiGLU clamp verified correct)

- kind=quality component=NVFP4 MoE Marlin path (W4A16) + checkpoint choice impact=4 confidence=4 effort=M needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** Marlin dequantizes NVFP4 weights and runs BF16 activations. input_global_scale is read only for int8 activations, so all 36,297 calibrated input_scales in the nvidia pack are ignored. Both packs use plain max-calibrated per-16 NVFP4 weights of the same base: the ModelOpt summary shows MaxCalibrator weight quantizers. So on this stack the nvidia pack is effectively LibertAI's weight-only experts plus three more quantized dense layers, and its only quality claim (NVIDIA's W4A4 table) does not transfer directly. The SwiGLU clamp is applied correctly on Marlin: the MARLIN backend is in NVFP4_BACKENDS_WITH_CLAMP and apply_moe_activation takes clamp_limit from moe_config.swiglu_limit. It is load-bearing: calibrated down_proj input amax hits exactly 100 = 10 x 10 in 9 layers. One minor numerics caveat: Marlin flushes E4M3 block scales below 2^-6 to zero when the max scale is about 448.

**Mechanism:** W4A16 removes activation-quantization error, so routed-expert quality should be at or above NVIDIA's W4A4 table. The dense-layer quantization is the only extra error compared with LibertAI. The scale flush affects only blocks with amax below tensor_amax x 2^-6/448, about 3.5e-5 relative, which is negligible.

**Evidence:**
- v11src/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:125-133,190-197 (input_global_scale only used when input_dtype==int8)
- v11src/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:190-198 MARLIN in NVFP4_BACKENDS_WITH_CLAMP; activation.py:150-165 clamp_limit from moe_config.swiglu_limit; marlin_moe.py:163-175 apply_moe_activation
- model.py:210,249 swiglu_limit passed to FusedMoE; :140-144 SiluAndMulWithClamp for shared/dense MLP
- nvidia .quant_summary.txt: down_proj_input_quantizer amax=1.00e+02 in layers 3,14,18,26,29,37,40,41,43,44; weight quantizers 'MaxCalibrator'
- index: 12096 = 42x288 expert input_scale triples (36,288) + 9 dense
- v11src/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:38-58,111-117 (sf=1 when max>=448; scales*2^7<2 -> 0)
- evidence/decision.tsv h-snap 'marlin ignores input_scale'
- nvidia README.md:171-201 NVFP4 vs BF16 within +/-1.5pp (W4A4, GB200)

**Proposed action:** Treat PR #11 as a provenance change, not a quality upgrade, until a paired eval shows nvidia >= LibertAI. Run Tier 0 + Tier 1 on both pins with identical flags and the thinking mode chosen in QUAL-1. Keep the LibertAI rollback documented. Do not claim NVIDIA's accuracy table for this recipe, because the table is W4A4 on GB200, thinking at Max effort.

**Est. impact:** Expected difference on Marlin: 0 to -0.5pp for nvidia vs LibertAI (estimate, from the extra 453M quantized dense params). Not resolvable by Tier 1 (MDE about 5pp); Tier-0 NLL can resolve it.

**Validation:** Paired Tier-0 NLL/top-1 agreement (both packs vs each other) and Tier-1 McNemar per benchmark; one knob (MODEL/SNAPSHOT_REV) only.

**Risks:** If a future SM121 W4A4 MoE path (flashinfer_b12x/cutlass) lands, the nvidia pack's scales become relevant and this comparison must be redone.

**Verifier reasoning:** marlin_moe.py:128-133 and :191-197: input_global_scale is multiplied in only when input_dtype == int8. The fp8 branch and the BF16 default ignore it. nvidia index: 36,297 input_scale keys = 42 layers x 288 experts x 3 (36,288) + 9 dense, matching per-layer counts (layers 3-44: 864 each; layers 0-2: 3 each; layer 45: 0). oracle/nvfp4.py:191-196 includes MARLIN in NVFP4_BACKENDS_WITH_CLAMP. marlin_utils_fp4.py:116-117 flushes scale*2^7<2, i.e. scale<2^-6, to zero; relative to a max of 448 that is 2^-6/448 = 3.49e-5, as stated. Correction: in .quant_summary.txt, down_proj_input_quantizer amax=1.00e+02 appears in 10 layers (3,14,18,26,29,37,40,41,43,44), not 9; the reviewer's own evidence list names 10. The LibertAI README confirms 'Weight-only NVFP4 (NVFP4-A16)'. Its scales derive from the weights with no calibration data (ModelOpt 0.45), and its input_scales were borrowed from RedHat's calibration only for loader compatibility. The 'both max-based' premise therefore holds, but bitwise weight identity was not checked, because the no-tensor-read constraint rules it out.

**Verifier corrected claim:** As stated, except the amax=100 (=10x10 clamp) signature appears in 10 layers, not 9. On Marlin (W4A16) the nvidia pack's calibrated activation scales are unused. Its only difference from LibertAI's weight-only experts on this stack is the separately quantized expert weights plus NVFP4 dense layers 0-2, so NVIDIA's W4A4/GB200 accuracy table does not transfer.

**Verifier corrected impact:** The -0.5pp to 0 estimate is unmeasured. Treat PR #11 as a provenance change until a paired Tier-0 NLL comparison exists.

## QUAL-5: No accuracy evaluation exists; the probes miss the default serving path, sampling, thinking-on, tools streaming, long-context depth and vision content

- kind=methodology component=bench_decode.py, smoke_vision.py, .cursor/skills/verify-glm53-flash/scripts/* impact=5 confidence=5 effort=M needs_gpu=True
- **verdict: confirmed** (corrected confidence 5)

**Claim:** Every probe hardcodes temperature 0 and chat_template_kwargs {enable_thinking:false}. They check format only: non-empty content, no '<think>', a parsed tool call, needle substring, count of 200. No probe sends a request without kwargs (the real server default), uses the default temp 1.0/top_p 0.95, enables thinking, streams a tool call, or scores an answer. The needle sits 'near the end' (a recency test). The vision smoke sends a 1x1 JPEG and does not assert the answer.

**Mechanism:** Format-only greedy probes cannot detect distributional drift from numerics changes (FP8 attention, W4A4 kernels, KV dtype), template or mode changes, or pack swaps.

**Evidence:**
- grep: thinking_off_probe.py:26-28, tool_call_probe.py:41-43, hermes_probe.py:62-64,99-101, needle_probe.py:49-53, count_probe.py:28-30 all temperature 0 + enable_thinking False
- smoke_vision.py:16-21 ('Color is not asserted'), :79-81 gate is non-empty content only
- needle_probe.py:2 'Prefill a unique needle near the end of a long prompt'
- README.md:22 single 318k needle answered once
- quality-probes.md 'Thinking-on can leave message.content empty' (unexplained; untested)

**Proposed action:** Add a two-tier quality gate, run through the OpenAI API with item IDs, seeds and outputs committed to evidence/quality-<ts>/. TIER 0, run on every perf knob, about 15 GPU-min: (a) teacher-forced NLL: /v1/completions with echo, prompt_logprobs=1, max_tokens=1 over a frozen 300k-token corpus (150 docs x 2k tokens: wiki, code, math, zh, plus 30 chat-rendered transcripts that include reasoning). Prefill at 1425 tok/s gives 300k/1425 = 3.5 min. Report mean NLL and top-1 agreement against a frozen baseline. Set the noise floor from two baseline reruns. Gate: |dNLL| <= max(3 sigma_rerun, 0.005 nats/token) and top-1 agreement >= baseline-rerun agreement minus 0.5pp. (b) Greedy byte-exact: 30 prompts x 256 tokens plus count-200. At c=2 that is 7.7k tokens / 33 tok/s = 4 min. Report the first-divergence index. (c) Default-path matrix from QUAL-2 plus a temp-1.0 numeric-table corruption probe (20 x 400 tokens = 8k tokens, about 4 min). TIER 1, for pack swaps, template or default changes, attention/KV dtype and ABLIT, about 2.5-3 h per config at about 33 tok/s aggregate for c=2: IFEval 150 (about 350 tokens each, 52k tokens, about 26 min); GSM8K 200 at effort low (about 500 tokens, 100k tokens, about 45 min); MMLU-Pro 280 (20 per category x 14, effort low, about 400 tokens, 112k tokens, about 55 min); BFCL-v3-style 120 (simple 50, multiple 30, parallel 20, irrelevance 20), run both non-streaming and streaming (about 2 x 15k tokens, 16 min), plus 10 multi-turn tool loops with and without reasoning echo; RULER NIAH multi-key at depths 10/50/90% x {32k,128k} x 3 (about 19 min) plus 2 x 300k (about 8 min); vision: ChartQA 100 + OCRBench/DocVQA 100 short answers (about 10 min) and MMMU-val 60 at effort low (about 30 min). Use lm-eval `local-chat-completions --apply_chat_template` for gsm8k/ifeval/mmlu_pro, evalscope for BFCL/ChartQA/MMMU, and small scripts for NLL and probes. Gate runs are greedy (for pairing); absolute reporting uses temp 1.0/top_p 0.95 with a fixed per-item seed. Strip <|begin_of_box|>/<|end_of_box|> before scoring.

**Est. impact:** Statistics: 95% CI half-width = 1.96*sqrt(p(1-p)/n). n=200, p=0.9 gives +/-4.2pp; n=280, p=0.7 gives +/-5.4pp. Paired McNemar MDE (80% power) is about 2.8*sqrt(d/n); with discordance d=0.08 and n=200 that is 5.6pp. So Tier 1 catches regressions of 5pp or more, and Tier-0 NLL over about 300k tokens catches sub-point numeric drift that Tier 1 cannot.

**Validation:** First establish noise: run Tier 0 twice on the unchanged baseline (both packs) and Tier 1 once per pack. Then every perf change must pass Tier 0; default/template/pack/KV/attention-dtype changes must pass Tier 1.

**Risks:** GPU time on shared Sparks. prompt_logprobs with DFlash2 and hybrid KDA prefix caching must be confirmed working on this build: sampling_params.py:533 sets skip_reading_prefix_cache. Full GPQA/AA-LCR reproduction of the card numbers (Max effort, 327k-token generations) is out of budget: GPQA-D at about 8k tokens x 198 / 33 tok/s is about 13 h.

**Verifier reasoning:** grep shows temperature 0 + chat_template_kwargs {enable_thinking:False} in bench_decode.py:36-39, smoke_vision.py:47-48, count_probe.py:29-30, thinking_off_probe.py:27-28, tool_call_probe.py:42-43, hermes_probe.py:63-64,100-101 and needle_probe.py:50-53. needle_probe.py:2 says 'near the end'. smoke_vision.py:16 says 'Color is not asserted' and :79-81 gates on non-empty content only. quality-probes.md:45 says 'Thinking-on can leave message.content empty'. Arithmetic re-derived: 300k/1425 = 210 s; 30x256 = 7,680 tokens / 33 = 233 s; 1.96*sqrt(0.09/200) = 4.16pp; 1.96*sqrt(0.21/280) = 5.37pp; 2.8*sqrt(0.08/200) = 5.6pp. prompt_logprobs is implemented with internal chunking in the v2 GPU runner (v1/worker/gpu/sample/prompt_logprob.py:82,199), and sampling_params.py:533 sets skip_reading_prefix_cache as noted. One nuance: the reviewer's Tier-0 corruption probe is at temp 1.0, but #257 also corrupted at temp 0, so a greedy long numeric-table probe is a cheaper first check.

## QUAL-6: Default sampling is temp 1.0 / top_p 0.95 with no max_tokens cap, but everything was measured greedy

- kind=quality component=generation_config auto-application impact=3 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** vLLM's default --generation-config auto applies the pack's generation_config (temperature 1.0, top_p 0.95) to every request that omits them. This matches the model card's evaluation settings, but all recipe probes and benches run at temperature 0. The quality and acceptance of the actual default user experience (thinking-off plus temp 1.0) is unmeasured, and that is the exact regime of the #257 corruption. There is also no default max_new_tokens, so with thinking on at Max effort a single request can generate up to (327680 - prompt) tokens: about 4.3 h at 21 tok/s.

**Mechanism:** Stochastic sampling exposes low-probability tokens that greedy never shows. Uncapped reasoning traces with Max effort run away in latency.

**Evidence:**
- nvidia generation_config.json: do_sample true, temperature 1.0, top_p 0.95; LibertAI caca4e6 same temp/top_p
- v11src/vllm/config/model.py:314 generation_config='auto'; :1676-1716 get_diff_sampling_param applies temperature/top_p/max_new_tokens
- nvidia README.md:205 eval at temperature=1.0, top_p=0.95
- https://huggingface.co/zai-org/GLM-5.3-Flash 'temperature=1.0 and top_p=0.95 for evaluation'
- run.sh:18 MAX_MODEL_LEN 327680; no --override-generation-config

**Proposed action:** Keep temp 1.0/top_p 0.95 (the card's settings). Add --override-generation-config '{"max_new_tokens": 32768}' as a default ceiling (a request can still set max_tokens). Report Tier-1 absolute scores at temp 1.0 with fixed seeds alongside the greedy gate. Add a temp-1.0 prose cell to the bench as a secondary acceptance and quality observation, not the published ruler.

**Est. impact:** Prevents multi-hour runaway generations (327680/21.2 tok/s = 4.3 h worst case). Quality impact of temp 1.0 on thinking-off is unquantified until measured.

**Validation:** Tier-0 corruption probe and Tier-1 IFEval/GSM8K at temp 1.0 (seeded) vs greedy, thinking-off vs effort low.

**Risks:** A 32k default cap could truncate Max-effort reasoning on very hard tasks; document the override.

**Verifier reasoning:** nvidia generation_config.json has do_sample true, temperature 1.0, top_p 0.95 and no max_new_tokens. config/model.py:1676-1716 applies these under generation_config='auto'. run.sh has no --generation-config or --override-generation-config. Minor correction: vLLM's neutral default temperature is already 1.0, so auto mode effectively adds only top_p 0.95. Worst case: 327,680/21.2 = 15,457 s = 4.3 h, correct. The card itself cites max generation lengths of 163,840 and 65,536 for evaluation, so a 32k default cap sits below the card's own eval budgets. It is fine as a server ceiling if documented.

**Verifier corrected claim:** The default requests sample at temperature 1.0 (vLLM's default anyway) with top_p 0.95 from generation_config, and have no max_tokens ceiling. Every recipe measurement is greedy.

## QUAL-7: reasoning_effort mapping: 'medium', 'minimal' and 'xhigh' silently become Max; any non-'none' effort forces thinking on

- kind=quality component=chat_template.jinja line 2 + vLLM build_chat_params impact=2 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** The template accepts only 'low' and 'high'; everything else, including the OpenAI-common 'medium' and 'minimal', becomes 'Reasoning Effort: Max'. vLLM sets enable_thinking = (reasoning_effort != 'none') whenever reasoning_effort is present and enable_thinking is not in the request kwargs. A client sending reasoning_effort:'minimal' to save tokens therefore gets thinking on at Max effort: the longest traces, at about 21 tok/s. 'none' gives '<think></think>' but still prepends 'Reasoning Effort: Max'.

**Mechanism:** Unrecognized effort values fall through to the template default (max).

**Evidence:**
- chat_template.jinja:2-3 effective_reasoning_effort in ['low','high'] else 'max'
- v11src/vllm/entrypoints/openai/chat_completion/protocol.py:245-257 accepts none/minimal/low/medium/high/xhigh/max; :582-583 enable_thinking = reasoning_effort != 'none'
- CPU jinja render (scratchpad/agents/quality_tmpl.py): effort medium -> '<|system|>Reasoning Effort: Max ... <think>'

**Proposed action:** Document the mapping in README (low | high | max; medium/minimal/xhigh -> Max). Optionally add a two-line template alias (minimal->low, medium->high) as an explicit, documented deviation. Validate that High and Low are in-distribution values (they are the card's levels).

**Est. impact:** Latency/cost only for mis-mapped clients: Max vs Low reasoning length is likely several times longer (unmeasured). Quality is not harmed.

**Validation:** CPU render test of each effort value; live check of reasoning_tokens per effort on 20 GSM8K items.

**Risks:** An alias changes what 'medium' means relative to the Hub template; keep it opt-in or documented.

**Verifier reasoning:** chat_template.jinja:2-3 maps anything outside ['low','high'] to 'max', and the `is not none` check always emits the system line. protocol.py:245-257 accepts none/minimal/low/medium/high/xhigh/max. :582-583 sets enable_thinking = (reasoning_effort != 'none') when the request kwargs lack enable_thinking, and that overrides the server default via merge order. The Anthropic effort field (low/medium/high/xhigh/max) routes the same way. recipes.vllm.ai independently states that effort resolves to max unless it is 'low' or 'high'.

## QUAL-8: Multi-turn reasoning history: clear_thinking defaults false, and reasoning is dropped unless the client echoes `reasoning`

- kind=quality component=chat_template.jinja history rendering impact=3 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The template keeps prior reasoning_content (clear_thinking defaults to false), yet the model card says to set clear_thinking=true for chat. vLLM returns the trace as `reasoning` and accepts `reasoning`/`reasoning_content` on input. Most OpenAI-SDK clients do not echo it, so past turns render as '<think></think>'. In thinking-on agent tool loops this silently drops the in-loop reasoning the model was trained to see. In chat with echoing clients, keeping all past reasoning bloats context against the card's advice. Otherwise the template behaves correctly: rendering tests confirmed there is no cross-iteration reasoning leak, that tool_calls with dict arguments render as <arg_key>/<arg_value> exactly like the Hub template, and that tool responses are grouped under <|observation|>.

**Mechanism:** Training used preserved or cleared thinking in specific shapes; serving cannot rebuild reasoning the client dropped.

**Evidence:**
- chat_template.jinja:4 clear_thinking default false; :143-153 history reasoning
- https://huggingface.co/zai-org/GLM-5.3-Flash 'For chat scenarios, explicitly set clear_thinking=true'
- v11src/vllm/entrypoints/openai/chat_completion/protocol.py:539-541 reasoning_content->reasoning; chat_utils.py:1866-1872 passes both to template
- CPU render: [q1, a1(reasoning R1), q2, a2(no reasoning), q3] -> '<think>R1</think>a1 ... <think></think>a2' (no leak); clear_thinking=true -> '<think></think>a1'
- chat_utils.py:1911-1945 JSON-string tool arguments converted to dict before templating

**Proposed action:** Set clear_thinking:true in the server default kwargs (QUAL-1 bundle). Past-turn reasoning is dropped, while reasoning after the last user message, i.e. inside the current tool loop, is kept (loop.index0 > last_user_index). Document that agent clients must echo `reasoning` on assistant messages within a tool loop. Add a Tier-1 multi-turn tool-loop item run with and without echo.

**Est. impact:** Less context bloat in chat (each echoed trace would otherwise add hundreds to thousands of tokens per turn). Agent quality impact of missing echo is unquantified; measure with 10 multi-turn loops.

**Validation:** CPU render diff (done), then Tier-1 multi-turn tool loops: echo vs no-echo, clear_thinking true vs false.

**Risks:** Clients that deliberately preserve thinking across user turns would need to pass clear_thinking:false.

**Verifier reasoning:** chat_template.jinja:4 defaults clear_thinking to false. :143-153 reads only m.reasoning_content, but chat_utils.py:1868-1872 sets both reasoning and reasoning_content when the client sends either (protocol.py:539-541 renames reasoning_content to reasoning on input). The card's verbatim text is 'In the chat template for GLM-5.3-Flash, `clear_thinking` defaults to `false` if not passed. For chat scenarios, explicitly pass `clear_thinking=true`.' The template logic `(not clear_thinking or loop.index0 > ns.last_user_index)` keeps in-loop reasoning as described. Caveat: under the current default (thinking off) there is no reasoning to drop, so the impact only materialises after QUAL-1 is adopted.

**Verifier corrected impact:** Relevant only once thinking is on by default (QUAL-1). Under today's thinking-off default there is no reasoning to preserve or drop.

## QUAL-9: Vision: preprocessing differs from the card's eval setting, video has an uncapped token budget, and correctness is unverified (1x1 smoke)

- kind=quality component=Glm5Next vLLM-native processor + --limit-mm-per-prompt impact=3 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The processor uses min_image_tokens 16 and max_image_tokens 8000 (8000 x 28 x 28 = 6.27 MP). Z.ai evaluates with images upscaled so the shorter side is at least 1.5K px, so small images served at native resolution will score below the card's MMMU-Pro 0.763 / BF16 0.7688. The video budget is max_image_tokens 240000, so one allowed video can take up to 240k of the 327,680 window and a large ViT activation footprint on already-tight UMA (115/121 GiB used). The processor is a fork-only vLLM port of the training pipeline (Glm4v prompt-update machinery, BICUBIC resize, fps-interval sampling). Its correctness has only been checked by a 1x1-pixel smoke test that does not assert the answer. The vision tower is BF16 with the SwiGLU clamp applied, consistent with sglang.

**Mechanism:** The resolution/token budget drives visual detail. A processor port mismatch (resize, normalization, frame timestamps) degrades answers silently. A large video prefill risks OOM.

**Evidence:**
- processor_config.json: image min/max_image_tokens 16/8000, video max_image_tokens 240000, fps 2, patch 14, merge 2
- v11src/vllm/models/glm5next/nvidia/multimodal.py:640-683 (own Glm5NextProcessor, pixel budget), :743-758 (prompt updates owned by vLLM), :118-121,338-340 vision SwiGLU clamp
- v11src/vllm/transformers_utils/processors/glm5next.py:83-130 frame sampler, :365 BICUBIC
- run.sh:62-64 limit image 4, video 1; no --mm-processor-kwargs
- smoke_vision.py:16-21,79-81
- https://huggingface.co/zai-org/GLM-5.3-Flash 'resize the input images such that their shorter side is at least 1.5K pixels'
- nvidia README.md:161-201 MMMU Pro 0.763 NVFP4
- tokenizer.json ids 154852/154853 <|begin_of_box|>/<|end_of_box|> (non-special, appear in content)

**Proposed action:** (1) Replace the 1x1 smoke with deterministic synthetic checks, rendered client-side with PIL: a 6-digit OCR string, a bar chart with known values, 2-image compare; assert exact answers. (2) Add Tier-1 ChartQA/OCRBench/MMMU subsets; test native vs client-upscaled (short side 1.5K) images. (3) Cap video via --mm-processor-kwargs (e.g. max_pixels / fps) or set video:0 until a video gate exists. (4) Strip box tokens in eval scoring.

**Est. impact:** The native-vs-1.5K resolution gap is unquantified (estimate: several pp on chart/OCR tasks with small source images). A video cap removes a 240k-token single-request UMA risk.

**Validation:** Tier-1 vision subsets: A = current vs B = client upscale, and the synthetic OCR/chart suite exact-match; one video probe at a capped budget with a free -h watch.

**Risks:** Upscaling increases image tokens, TTFT and KV use (4 images x about 3.8k tokens at 1.5K x 2K).

**Verifier reasoning:** processor_config.json confirms image min/max_image_tokens 16/8000 and video max_image_tokens 240000, fps 2. _pixel_budget (transformers_utils/processors/glm5next.py:340-341) = tokens x temporal(2) x 28^2. For a still image t_bar=2, so h*w <= 8000*784 = 6.27 MP, as stated. smart_resize rounds only up to the factor with a 16-token floor, so small images are not upscaled to the 1.5K short side the card recommends (the card text is confirmed verbatim). run.sh:61-64 sets the image:4/video:1 limit and passes no mm-processor-kwargs. _get_video_max_pixels honours a max_pixels override (multimodal.py:679-683), so the cap proposal is implementable. That the resolution gap costs 'several pp' and that a 240k-token video is a practical UMA/ViT risk are both unmeasured estimates. Note that 240k < 327,680, so vLLM would accept such a request.

**Verifier corrected impact:** Unquantified. The video-budget OOM risk is plausible but untested, and the native-vs-1.5K accuracy gap is an estimate.

## QUAL-10: fp8_e4m3 KV with an uncalibrated per-tensor scale of 1.0 on the patched sm_121 sparse-MLA path: long-context quality validated by only one recency needle

- kind=quality component=MLA KV cache (kv_fp8_cast) + FLASHINFER_MLA_SPARSE FA2 sm_121 patch impact=3 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The pack ships kv_cache_quant_algo FP8 with no k_scale/v_scale tensors. The ModelOpt k/v_bmm quantizers are amax=448 const, i.e. scale 1.0, so latent KV is cast to E4M3: 3 mantissa bits, up to 6.25% per-element relative error, subnormal below 2^-6. NVIDIA validated this on GB200 kernels (AA-LCR 0.7106 NVFP4+fp8 vs 0.71 BF16). This recipe runs a different, locally patched sparse-MLA FA2 path on sm_121, validated only by single needles placed near the end (8k/20k/64k/318k), which a recency bias passes trivially. The other long-context numerics look fine: KDA recurrent state is fp32 and the indexer uses Hadamard + fp8 with per-token scales.

**Mechanism:** Precision loss in cached latents accumulates across 11 sparse-MLA layers and top-2048 selection. A kernel bug in the fp8 dequant path would show up mainly at depth and length.

**Evidence:**
- hf_quant_config.json kv_cache_quant_algo FP8; config.json kv_cache_scheme dynamic false, 8-bit
- model.safetensors.index.json: no k_scale/v_scale/kv_scale keys (only input_scale)
- nvidia .quant_summary.txt: layers.N.self_attn.k_bmm_quantizer/v_bmm_quantizer amax=4.48e+02(const)
- nvidia README.md:161-201 AA-LCR BF16 0.71 vs NVFP4 0.7106 with --kv-cache-dtype fp8 (README.md:123)
- v11src/vllm/model_executor/layers/mamba/mamba_utils.py:131-137 kda_state_dtype -> recurrent float32
- v11src/vllm/models/glm5next/nvidia/attention.py:371-384 fwht128_quant_fp8 per-token q scales
- README.md:22,151 needles near end; needle_probe.py:2

**Proposed action:** Add RULER-style multi-key NIAH plus a variable-tracking task at depths 10/50/90% x 32k/128k (3 each) and 2 x 300k to Tier 1. For a dtype A/B, run a 64k lane (MAX_MODEL_LEN reduced) with --kv-cache-dtype auto (bf16) vs fp8_e4m3, one knob, and compare NIAH plus Tier-0 NLL on 32k documents.

**Est. impact:** Expected to be small (the NVIDIA GB200 result shows about 0 delta), but unverified on this kernel path; the gate turns a hidden risk into a measured delta.

**Validation:** bf16-vs-fp8 KV at 64k, one knob; NIAH depth sweep on the production pin.

**Risks:** A bf16 KV lane halves pool capacity and may not fit 327k; it is only a reference lane.

**Verifier reasoning:** Facts verified. hf_quant_config kv_cache_quant_algo FP8. .quant_summary.txt:636-637 shows k_bmm/v_bmm_quantizer 'amax=4.48e+02(const)', which is scale 1.0; 45 of 69 k_bmm entries carry it. The index has no k_scale/v_scale keys. E4M3 half-ulp relative error is 2^-4 = 6.25% and the min normal is 2^-6, both correct. README.md:22 and needle_probe.py:2 support 'needle near the end' only. The risk is unmeasured: NVIDIA's AA-LCR (0.7106 vs 0.71) used vLLM fp8 KV on GB200 kernels, not the patched sm_121 sparse-MLA path.

## QUAL-11: SPEC=mtp rollback is likely broken on the nvidia pack (MTP layer is BF16 but not in the quant exclude list)

- kind=correctness component=MTP layer 45 loading with ModelOpt NVFP4 config impact=2 confidence=3 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** AGENTS.md advertises SPEC=mtp as the rollback. In the nvidia pack, layer 45 (the MTP layer) is entirely BF16, including its 288 routed experts, but hf_quant_config exclude_modules has no layers.45 entries. mtp.py builds its MoE with the global quant_config, and ModelOpt's is_layer_excluded only consults exclude_modules. vLLM would therefore allocate NVFP4 (U8) expert params for layer 45 and hit BF16 [2048,4096] tensors. The likely result is a load failure. Even if it loaded, it would add 288 x 3 x 2048 x 4096 x 2 B = 14.5 GB (7.25 GB per rank) on a box with about 6 GiB headroom. This concerns the availability of the quality-safe native drafter, not decode quality itself: speculation is lossless.

**Mechanism:** Quant-config coverage mismatch between checkpoint and model graph for the MTP layer.

**Evidence:**
- safetensors headers: layers.45.mlp.experts.0.{gate,up,down}_proj.weight BF16 [2048,4096]/[4096,2048]; no layers.45 input_scale in index (12096 = 42x288)
- config.json ignore list ends at layer 44 (no layers.45)
- v11src/vllm/models/glm5next/nvidia/mtp.py:44,70 quant_config=vllm_config.quant_config
- v11src/vllm/model_executor/layers/quantization/modelopt.py:139-168 is_layer_excluded
- AGENTS.md: '`SPEC=mtp` rolls back to MTP-4'

**Proposed action:** Mark SPEC=mtp as 'unverified on nvidia pack' in README/AGENTS. If needed, fix with a model-side exclusion of the MTP layer from quant_config when its weights are BF16, verified at the next exclusive slot. Until then, the rollback is the LibertAI pin (its MTP is also BF16, but it loaded in the 2026-09-03 SPEC=mtp run).

**Est. impact:** Prevents a failed rollback during an incident. Memory math: 14.5 GB BF16 MTP experts vs the UMA headroom of about 6 GiB means it cannot fit even if it loads.

**Validation:** VALIDATE_ONLY cannot catch it. Needs an exclusive slot: SPEC=mtp boot on the nvidia pin and grep the load error.

**Risks:** The fork's mtp.py may special-case the BF16 MTP in code I did not trace; confirm before editing.

**Verifier reasoning:** I read the safetensors headers only (scratchpad/agents/qv2_l45.py). nvidia 09b04e5 layer 45 has experts.0.{gate,up}_proj.weight BF16 [2048,4096] and down_proj BF16 [4096,2048], with no weight_scale or input_scale; the per-layer input_scale count for layer 45 is 0. hf_quant_config exclude_modules has no layers.45 entries of any kind: no experts, shared_experts or mlp.gate. mtp.py:44 uses vllm_config.quant_config. The loader (mtp.py:385-405 into routed_experts.py:_load_w13/_load_w2) ends in expert_data.copy_(loaded_weight). Copying a BF16 [1024,4096] per-rank shard into an NVFP4 U8 [1024,2048] param does not broadcast, so load should raise. Attention is safe because quant_config=None is hardcoded (model.py:331). The reviewer's rollback claim is WRONG. LibertAI caca4e6 layer 45 experts are NVFP4: experts.0.gate_proj.weight U8 [2048,2048], weight_scale F8_E4M3 [2048,256], weight_scale_2 F32, and input_scale present. Only eh_proj, shared_experts and o_proj are BF16. Its exclude list also uses prefix-agnostic wildcards ('*.mlp.shared_experts.*', '*.mlp.gate'). That is exactly why SPEC=mtp loaded on LibertAI, and it explains why the nvidia pack differs. Memory math re-derived: 288 x 3 x 2048 x 4096 x 2 B = 14.5 GB total, 7.25 GB per rank.

**Verifier corrected claim:** On the nvidia pin, the documented SPEC=mtp rollback (AGENTS.md; run.sh:76-77) should fail at weight load. The MTP layer's routed experts (and its shared_experts and gate) are BF16 in the checkpoint but not excluded from NVFP4 quantization, so FusedMoE allocates U8 params and copy_ hits a shape mismatch. LibertAI's MTP routed experts are NVFP4 (U8 plus scales), not BF16, which is why SPEC=mtp worked there. The MTP rollback therefore requires MODEL/SNAPSHOT_REV to go back to LibertAI as well.

**Verifier corrected impact:** The rollback path is broken on PR #11's default pin, confirmed by code trace and headers but not by a boot. Even with a model-side exclusion, BF16 MTP experts need 7.25 GB per rank, which does not fit the roughly 6 GiB UMA headroom.

## QUAL-12: DFlash2 losslessness and thinking-budget under sampling are asserted but only checked greedily

- kind=quality component=v1/worker/gpu/spec_decode rejection sampler + thinking_budget in spec mode impact=2 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Standard rejection sampling with a greedy draft (draft probs one-hot) is distribution-exact in theory, and greedy outputs matched MTP byte for byte. Nothing has tested the sampled path (temp 1.0, top_p 0.95 applied in apply_sampling_params before rejection), nor thinking_token_budget forcing '</think>' with 7 speculative slots.

**Mechanism:** Top-p and temperature interacting with the verification kernel, or a spec-mode budget off-by-one, would bias outputs without any error.

**Evidence:**
- v11src/vllm/config/speculative.py:81-82,220 rejection_sample_method 'standard', draft_sample_method greedy|probabilistic
- v11src/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:135-170 apply_sampling_params then rejection_sample
- v11src/vllm/v1/sample/thinking_budget_state.py (in_spec_mode handling)
- README.md:59 'our greedy outputs matched MTP's byte for byte'

**Proposed action:** Add a SPEC=none reference lane (a run.sh knob that omits --speculative-config) for eval only. Compare first-token and 32-token n-gram distributions over 300 seeded samples of 5 prompts with spec vs no spec (chi-square). Test thinking_token_budget=256 with effort low on 20 GSM8K items: assert that '</think>' is forced and that content follows.

**Est. impact:** Low expected risk; closes the only unverified correctness assumption in the spec path.

**Validation:** Distributional test as above (p>0.01 for no difference).

**Risks:** The no-spec lane is slower, which costs GPU time only.

**Verifier reasoning:** config/speculative.py:81-82 defines RejectionSampleMethod standard/synthetic/block and DraftSampleMethod greedy/probabilistic. thinking_budget_state.py:50-70 has in_spec_mode handling with a mask capacity of max_reqs*(k+1). README.md:59 says greedy outputs matched MTP byte for byte. The sampled path and the spec-mode budget forcing have indeed never been tested here. The risk is theoretical and low; no concrete bug was found.

## QUAL-13: PR #3 ABLIT o_proj transplant changes model behavior across 30 layers and must be quality-gated

- kind=pr-review component=PR #3 (glm53-sm121-v12 ABLIT=1) impact=2 confidence=3 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** ABLIT=1 replaces o_proj weights of layers 15-44 with weights from an 'uncensored' finetune donor. That changes refusal behavior by design, and potentially general capability and tool-use accuracy, on every token passing through those layers. PR #3 was validated only by SHA and shape checks and a single chat smoke test. Its interaction with the nvidia pack is also untested: the donor is based on the dealignai UNCENSORED-NVFP4 lineage.

**Mechanism:** Weight surgery on attention output projections shifts the residual stream distribution.

**Evidence:**
- gh pr view 3: 'fetch only layers 15–45 from the donor', validation = compile checks + 'OpenAI chat smoke test completed successfully'

**Proposed action:** Require the Tier 0 + Tier 1 gate (ABLIT=0 vs 1, same pack) before merge, and report the deltas in the PR. Keep ABLIT=0 as the default.

**Est. impact:** Unknown; the gate quantifies it (MDE about 5pp per benchmark, sub-point via NLL).

**Validation:** Paired Tier-1 run, one knob (ABLIT).

**Risks:** External contributor PR; license and attribution noted in the PR.

**Verifier reasoning:** `gh pr view 3` gives the title 'feat: add optional verified ABLIT o_proj transplant' and says it fetches 'only layers 15–45 from the donor' (dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4@e90ef41). Validation consisted of shell and compile checks, SHA/shape/TP-slice self-checks, and 'an OpenAI chat smoke test completed successfully'. There was no accuracy evaluation. ABLIT=0 is the default.

## Open questions
- Which NVFP4 linear kernel does vLLM auto-select for dense MLP layers 0-2 on sm_121 in glm53-sm121-v11? It depends on cutlass_fp4_supported() given TORCH_CUDA_ARCH_LIST built without 12.1a, and on has_flashinfer_b12x_gemm(). Only a boot log can tell ('Using ... for NVFP4 GEMM').
- Does vLLM `prompt_logprobs` work on this build with DFlash2 speculative decoding plus hybrid KDA state (needed for the Tier-0 NLL gate)? Code suggests yes (sampling_params.py:533 skips prefix cache) but it is untested.
- How long is 'low' reasoning effort in practice on this model (reasoning tokens per simple and per hard prompt)? This decides the latency cost of moving the default off thinking-off (QUAL-1). The MiaAI claim of 'no latency cost' is unverified.
- Is the LibertAI README claim that vLLM glm45 'silently discards the whole reply' with a prompt-side <think> stale? The v11 parser starts in REASONING (glm47_moe.py:125), but thinking-on has never been exercised live on this recipe, and quality-probes.md notes 'Thinking-on can leave message.content empty'.
- Does the fork's mtp.py special-case the BF16 MTP layer in the nvidia pack (it would make SPEC=mtp load)? I found no such code path, but full load_weights tracing was out of scope.
- Are the nvidia and LibertAI expert weights numerically near-identical (same max-calibrated NVFP4 of the same BF16 base)? Answering needs reading tensor data (weight_scale_2 / packed weights), which this read-only, no-tensor review did not do.
- Should the video modality stay enabled at all (video:1 with max_image_tokens 240000) before a video quality gate and a UMA-safe mm_processor_kwargs cap exist?
- Are reference thinking-off scores for GLM-5.3-Flash published anywhere? None were found on the Z.ai card, so the thinking-off vs effort-low comparison must be measured locally.

## Verifier: missed issues
- No evidence exists anywhere that the nvidia 09b04e5 pack has ever booted on glm53-sm121-v11. The Sep-8 iter-b12x-A-marlin and B runs, the only post-PR-#11 boots, load /cache/.../LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/caca4e6 (opt-b12x/evidence/iter-b12x-B-flashinfer/engine-spark1.log). So every quality or perf property of PR #11, including load success, the dense NVFP4 kernel choice and vision, has zero receipts. PR #11 should not merge as the default until one exclusive-slot boot and Tier-0 run exist.
- The exclude-list styles differ between packs, and this matters for load, not only MTP. LibertAI uses prefix-agnostic wildcards ('*.mlp.shared_experts.*', '*.mlp.gate', '*.self_attn.*'). nvidia uses fully qualified 'model.language_model.layers.N.mlp.shared_experts*' patterns, which depend on Glm4v's hf_to_vllm_mapper ('model.language_model.' -> 'language_model.model.') being applied via ModelOptQuantConfigBase.apply_vllm_mapper (modelopt.py:218-237). That path works for the main model but has never been exercised with this checkpoint. It is the mechanism behind the QUAL-11 MTP failure, because the MTP model has no layer-45 entries in any form.
- The current default path is internally contradictory. Every request without kwargs renders `[gMASK]<sop><|system|>Reasoning Effort: Max ... <|assistant|><think></think>` (chat_template.jinja:2-3 plus 260): a Max-effort instruction with an empty think span. The Tier-0 default-path probe should test this exact shape. Separately, the QUAL-1 fix changes the prefix of the published bench prompt to 'Low' even though bench_decode.py still sends enable_thinking:false, so the ruler must pin reasoning_effort to stay comparable.
- MiaAI #257 reports the thinking-off corruption at temperatures 1.0, 0.6 AND 0 (WebFetch of the issue), not only under sampling. The corruption is therefore detectable with greedy probes that are cheap and paired. The recipe's probes miss it because none of them produces long numeric markdown tables (count/PING/tool/needle only), not because they are greedy.
- The reviewer's QUAL-11 rollback advice was factually wrong. LibertAI's MTP routed experts are NVFP4 (U8 [2048,2048] plus F8_E4M3 scales plus input_scale in the headers), not BF16. On PR #11, the AGENTS.md line '`SPEC=mtp` rolls back to MTP-4' is false unless MODEL/SNAPSHOT_REV also roll back to LibertAI caca4e6. AGENTS.md and README should say so explicitly.
