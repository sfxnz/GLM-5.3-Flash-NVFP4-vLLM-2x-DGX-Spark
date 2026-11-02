# r3 bug hunt: our recipe, patches and evidence

Goal. Find defects in our own recipe, image patches and evidence that cost quality, speed, memory headroom or reliability on the 2x Spark GLM-5.3-Flash serve, and root-cause each one. This was a read-only pass on 2026-11-02. No serve was booted. vLLM source was read from `glm53-sm121-v14` (9599a9fc5fe5) in CPU-only containers. Some checks ran real vLLM code on the CPU against the pinned checkpoint.

Labels. **Verified** means I read the code path and the log or output that proves it. **Measured** means a number from evidence or from a CPU run today. **Inference** means the mechanism fits the evidence but was not traced end to end.

## Ranked by user impact

| # | Defect | Who notices | Label | Fix size |
|---|---|---|---|---|
| 1 | Thinking-off default corrupts numeric and structured output | every default request that writes tables, lists of numbers or code | symptom measured, cause inference | S (default kwarg) after a 4-arm A/B |
| 2 | Short native videos (32 frames or fewer) get 3x more placeholders than embeddings | any client sending a short clip; likely kills the engine | mismatch verified on CPU, crash inference | S (patch layer) |
| 3 | Prefix cache with DFlash2 probably never hits; the built fix is off | multi-turn chat and agents, every turn | code verified, zero hits inference | S (one EXTRA_ENV boot) |
| 4 | Same-boot nondeterminism is far above rounding noise (1.55% argmax flips, 13-26 nat outliers) | reproducibility, every gate, flaky probes | measured, source untraced | M (localize first) |
| 5 | Tier 0 KL gate is right; the relaxed limit hides a real INT8 shift | maintainers | verified | S (re-record reference) |
| 6 | mHC kernels still JIT-compile during serving; the built warmup is off | first request of a new shape class, both nodes | verified | S (one EXTRA_ENV boot) |
| 7 | Head loads weights 3.1x slower than the worker (683 s vs 219 s) | every boot, every A/B | measured, cause inference | S levers, ops |
| 8 | spark1 headroom 3-6 GiB below spark2 | OOM risk on the head | measured, split inference | S levers, ops |
| 9 | Pinned `09b04e5` experts are the miscalibrated export | everyone (quality) | scale diff verified | planned as X1 |
| 10 | utf8 and tools scorers misjudge; utf8 text is not saved | maintainers | verified | S |
| 11 | Video probe sits on a near-tie; not a preprocessing bug | maintainers | verified | S (probe) |
| 12 | JIT cache seeding loss in r2 B6 was harmless | none | verified | none needed |

Six opt-in switches are built into the v13/v14 image, unit-tested on the CPU, and were never booted on the Sparks: `GLM53_DFLASH_PREFIX_CACHE_FIX`, `GLM53_MHC_WARMUP`, `GLM53_INDEXER_WS_FACTOR`, `GLM53_SKIP_MTP_WEIGHTS`, `GLM53_ROUTER_FP32` and `GLM53_KDA_TRIM`. No `trail.tsv`, `decision.tsv` or `notes.txt` row in any GLM worktree mentions them (grep over every `evidence/`). Items 3, 6, 7, 8 and 4 each map to one of them. They are the cheapest P4 units.

---

## 1. Thinking-off default corrupts numeric and structured output

**Symptom.** Tier 0 `utf8` fails on almost every boot. The failing sub-checks are `square_errors > 0`, `finish_reason` `length`, and sometimes `complete` false. It is never U+FFFD. Every saved run has `fffd: 0`, so streaming detokenization and the glm45 reasoning parser are not involved.

**Evidence.**
- `evidence/r2/r1/tier0.json`: `square_errors 8, cube_errors 6, chinese_numeral_errors 5, finish_reason length, fffd 0`.
- Saved texts: `evidence/e3b-av-tau0.1/utf8-texts.json`, `evidence/e3b-av-tau0.2/utf8-texts.json`, `evidence/e4b-int8-tau0.3/utf8-texts.json`, `evidence/e5-final/f1/utf8-texts.json`. `evidence/e4-tools/utf8_scan.py` over all four files flags all 12 runs. Every run has leaked words or broken row sequences, including the three the scorer passed.
- The leaked cells are words, not digits. Examples are `| 13 |  numerals | 2197 |`, `| 15 | 225 | 3375 | fifteen |`, `5 characters`, `34 conversation`, `50 glitches`, `40 user`, `GPT`, `English`. Some are copied from the prompt (`n³`, `二十三`, `numerals`, `Output only the table, no other text.`). The model often writes `| 2 | 2 | 8 |` (n instead of n²) for n = 2..9, then rewrites the table with "Wait, I need to correct several errors" until it hits 3000 tokens. In the 15-row runs (E3b tau 0.1, E4b reruns, F1, G1) it emits EOS right after `| 15 | 225 | 3375 | fifteen |`.
- E0 (v11, BF16 non-MoE linears) shows the same pattern (`evidence/e0-nvidia-v11/probe-utf8.txt`: `language`, `English`, `40 user`, `GPT`, `78 response`). So INT8 targets, adaptive verify and v14 did not introduce it.
- E0's teacher-forced check found the wrong tokens are the target's own top-1 (`evidence/e0-nvidia-v11/tier0-notes.txt`; E4b `probe-lossless.txt`: 1 of 221 tokens off top-1, a near-tie). The drafter is not the cause.
- The rendered default prompt is contradictory (rendered today from `chat_template.jinja` on the CPU):
  `[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>...<|assistant|><think></think>`.
  The system line says Max effort and the assistant prefix closes thinking. The official template always opens `<think>` and documents only low, high and max effort (`evidence/review-20260925/findings/quality.md`, QUAL-1 and quality-MISSED-3).
- An independent 2x Spark lane on a different quantization (EXL3) reports the same thing: foreign tokens injected inside numbers in markdown tables with thinking off, at temperature 1.0, 0.6 and 0, fixed by `reasoning_effort: low` (MiaAI-Lab issue #257, cited in `quality.md:23` and `:399`).

**Root cause (inference).** The server default `--default-chat-template-kwargs '{"enable_thinking": false}'` (`run.sh` start_local) puts every default request in an off-distribution prompt shape. The model then injects meta-words and prompt fragments into dense numeric output. A second candidate is the miscalibrated `09b04e5` experts (item 9). The A/B below separates the two.

**User impact.** Every request that does not set thinking kwargs gets this prompt. Tables, numeric lists, code with constants and tool arguments are the exposed outputs. The probe corrupts in 12 of 12 saved runs. Short answers (count-200, "Paris", needles) pass, which is why the other gates stay green.

**Fix.** Change the server default from `enable_thinking: false` to `reasoning_effort: low` (`<think>` opened, "Reasoning Effort: Low"). E0 measured that low effort closes `<think>` at once on a simple question (`evidence/e0-nvidia-v11/probe-effort-low.txt`: `'</think>Paris'`), so the cost on simple prompts is near zero. If arm `off_low` below is clean, a template-only fix is enough: drop the `Reasoning Effort: Max` line when thinking is off. Size S, one line in `recipe.yaml` / `run.sh`. It changes the published prose ruler, so it needs its own boots (FOR-1 showed the template moves the ruler).

**Live test (port 8000, about 20-40 min).** Decision written first. If `off_max` is clean in 2 of 6 or fewer and `on_low` in 5 of 6 or more, the prompt shape is the cause. If `off_low` is also clean in 5 of 6 or more, the contradictory Max line is the trigger. If every arm corrupts, run the same script on the X1 (`1d00738`) boot.

```bash
cd ~/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-r3
python3 - <<'EOF'
import re, sys
sys.path.insert(0, "quality")
import tier0
from common import Client
c = Client("http://127.0.0.1:8000")
NUM, CJK = re.compile(r"^[\d,]+$"), re.compile(r"^[一-鿿]+$")
def scan(text):
    rows = [l for l in text.splitlines() if l.strip().startswith("|") and "---" not in l][1:]
    prev = breaks = odd = 0
    for l in rows:
        cells = [x.strip() for x in l.strip().strip("|").split("|")]
        n = int(cells[0]) if cells[0].isdecimal() else None
        breaks += n != prev + 1
        prev = n if n is not None else prev
        odd += sum(bool(x) and not NUM.match(x) and not CJK.match(x) for x in cells[1:])
    return len(rows), breaks, odd
ARMS = {"off_max": {"chat_template_kwargs": {"enable_thinking": False}},
        "off_low": {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}},
        "on_low": {"reasoning_effort": "low"},
        "on_max": {"chat_template_kwargs": {"enable_thinking": True}}}
for arm, kw in ARMS.items():
    clean = 0
    for i in range(6):
        out = c.chat(tier0.UTF8_PROMPT, stream=True, temperature=0, max_tokens=12000, **kw)
        rows, breaks, odd = scan(out["content"])
        ok = rows == 40 and breaks == 0 and odd == 0 and out["finish_reason"] == "stop"
        clean += ok
        print(arm, i, out["finish_reason"], rows, breaks, odd, len(out["reasoning"]), "clean" if ok else "CORRUPT", flush=True)
    print("ARM", arm, f"{clean}/6 clean", flush=True)
EOF
```

---

## 2. Short native videos get 3x more placeholders than embeddings

**Symptom.** Not seen yet, because every probe sends videos as JPEG/PNG frame lists. Found by reading the video path while checking the 3186 answer (item 11).

**Evidence (verified, CPU run today).** Two different frame samplers build one request.
- The video processor samples with `glm_sample_frame_indices` (fps_interval 2, `vllm/transformers_utils/processors/glm5next.py:83-153`).
- The prompt placeholder is built from `Glm4vProcessingInfo._get_video_second_idx_glm46v` (dynamic 3 fps x temporal 2, `vllm/model_executor/models/glm4_1v.py:1264-1340`), called from `_construct_video_placeholder` (`:1413-1458`). `Glm5NextMultiModalProcessor` overrides neither (`models/glm5next/nvidia/multimodal.py`).
- The two agree only when `do_sample_frames` is false. vLLM's opencv loader sets it true when it loaded every frame, i.e. clips with 32 frames or fewer (`vllm/multimodal/video.py:210`, default `num_frames` 32). Frame lists (`data:video/jpeg`) set it false (`multimodal/media/video.py:165`), which is why `quality/vision.py` never hit this.
- I ran vLLM's own `VideoMediaIO` loader and the model's multimodal processor (`MULTIMODAL_REGISTRY.create_processor`, real `09b04e5` config and tokenizer, no GPU) on ffmpeg test clips:

| clip | frames loaded | do_sample_frames | grid_thw | encoder embeddings | placeholder embeddings |
|---|---|---|---|---|---|
| 2 s at 10 fps | 20 | true | [2,16,16] | 128 | 384 |
| 1 s at 30 fps | 30 | true | [1,16,16] | 64 | 192 |
| 10 s at 30 fps | 32 of 300 | false | [16,16,16] | 1024 | 1024 |

**Root cause (verified up to the processor output).** The prompt advertises 3x as many video frames as the encoder produces. The engine-side failure is inference. `_gather_mm_embeddings` slices the 128-row encoder output with indices up to 384, and `_merge_multimodal_embeddings` (`model_executor/models/utils.py:655-672`) then raises `ValueError("Attempted to assign ... multimodal tokens to ... placeholders")` inside the worker's forward. In vLLM v1 that is an EngineCore fatal error, so the whole TP=2 serve goes down until a ~15 min reboot.

**User impact.** Any client that uploads a short clip (a GIF-like clip, a 1-3 s phone video) can take the serve down. Low frequency, outage severity.

**Fix.** A v15 patch layer: in `Glm5NextProcessingInfo`, compute video timestamps from `glm_sample_frame_indices` with the processor's `fps_interval` and `max_frame_count_dynamic` (the sampler the processor already uses and calls training-reference parity), then `[int(i / fps) for i in idx][::2]`. About 15 lines plus a CPU test that asserts placeholder embeddings equal `t*h*w/4` for the three clips above. Size S.

**Test.** CPU first (safe, no serve): the probe that produced the table is below. After the fix every row must show equal counts. Live, after the fix: send the 20-frame clip and expect HTTP 200 with `/health` still 200. Before the fix, run it only as the last step of a session, since the expected result is a dead engine.

```bash
mkdir -p ~/projects/data/glm53-r3/mm && cd ~/projects/data/glm53-r3/mm
ffmpeg -loglevel error -y -f lavfi -i testsrc=size=224x224:rate=10 -t 2 -pix_fmt yuv420p clip20.mp4
cat > probe_mp4.py <<'EOF'
import sys
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.media.video import VideoMediaIO
from vllm.multimodal.media.image import ImageMediaIO
snap, clip = sys.argv[1], sys.argv[2]
mc = ModelConfig(model=snap, trust_remote_code=True, limit_mm_per_prompt={"image": 4, "video": 1}, mm_processor_cache_gb=0)
proc = MULTIMODAL_REGISTRY.create_processor(mc)
frames, meta = VideoMediaIO(ImageMediaIO()).load_bytes(open(clip, "rb").read()).media
items = proc.info.parse_mm_data({"video": [(frames, meta)]})
out = proc.apply(ProcessorInputs(prompt="<|begin_of_video|><|video|><|end_of_video|>describe", mm_data_items=items), TimingContext())
t, h, w = (int(x) for x in out["mm_kwargs"]["video"][0]["video_grid_thw"].data)
print("encoder", t * h * w // 4, "placeholder", out["mm_placeholders"]["video"][0].get_num_embeds())
EOF
docker run --rm --network none -e HF_HUB_OFFLINE=1 -e HF_HOME=/cache/huggingface \
  -v $HOME/.cache/huggingface:/cache/huggingface:ro -v $PWD:/probe:ro --entrypoint python3 glm53-sm121-v14 -I \
  /probe/probe_mp4.py /cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/09b04e5e74bca08ca8549fc736d4cdd8624bfde3 /probe/clip20.mp4
# live, after the fix only
python3 - <<'EOF'
import base64, json, urllib.request
u = "http://127.0.0.1:8000"
m = json.load(urllib.request.urlopen(u + "/v1/models"))["data"][0]["id"]
v = "data:video/mp4;base64," + base64.b64encode(open("clip20.mp4", "rb").read()).decode()
body = {"model": m, "max_tokens": 64, "temperature": 0, "messages": [{"role": "user", "content": [
    {"type": "video_url", "video_url": {"url": v}}, {"type": "text", "text": "Describe the video in one sentence."}]}]}
req = urllib.request.Request(u + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
print(urllib.request.urlopen(req, timeout=600).read()[:300])
EOF
curl -s -o /dev/null -w 'health %{http_code}\n' http://127.0.0.1:8000/health
```

---

## 3. Prefix cache with DFlash2 probably never hits; the built fix is off

**Symptom.** R1's 101 engine logger lines all read `Prefix cache hit rate: 0.0%` (`evidence/r2/r1/head.log`). That alone proves nothing. No R1 workload repeated a prefix of at least one 2304-token block. `bench_decode.py` cell H's description ("A's prompts: prefix-cache hit") is also wrong, since A's prompts are shorter than one block.

**Evidence.** `vllm/v1/core/kv_cache_coordinator.py:120-145` in v14: when no KV group is flagged EAGLE, every group gets the EAGLE last-block drop unless `GLM53_DFLASH_PREFIX_CACHE_FIX=1`, which limits it to the DFlash draft sliding-window group. `run.sh` never sets it. The review (EXT-5) and two other 2x Spark DFlash2 lanes measured zero hits with this design (Tony: 0 hits in 35,280 queries; a 13k-token repeat went from 21.1 s to 6.3 s TTFT after the fix).

**Root cause.** Code path verified. Zero hits on our serve is inference.

**User impact.** Multi-turn chat and agent loops re-prefill the full conversation every turn. At the measured ~1590 tok/s prefill, a 32k-token context costs ~20 s TTFT per turn instead of a few seconds.

**Fix.** One boot with `EXTRA_ENV='GLM53_DFLASH_PREFIX_CACHE_FIX=1'`. If hits appear and count-200, needles and greedy parity hold, promote it to a recipe knob. Size S. Expect about 4608 cached tokens on a ~10-12k repeat (the review verifier's corrected figure), not the full prompt.

**Live test.** Run on the current default boot and again on the fix boot. Before: `hit 0` on runs 1 and 2. After: `hit` at least 4608 on runs 1 and 2, and TTFT falls by about the hit fraction.

```bash
cd ~/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-r3
python3 - <<'EOF'
import re, sys, time
sys.path.insert(0, "quality")
import tier0
from common import Client
c = Client("http://127.0.0.1:8000")
doc = "\n\n".join(tier0.filler(4242, 140)) + "\n\nSummarize the notes above in one line."
print("prompt tokens", c.tokenize_count(doc))
def counters():
    m = c.get("/metrics")
    get = lambda k: sum(float(x) for x in re.findall(rf"^{k}\{{[^}}]*\}} ([0-9.e+]+)$", m, re.M))
    return get("vllm:prefix_cache_queries_total"), get("vllm:prefix_cache_hits_total")
for i in range(3):
    q0, h0 = counters(); t = time.time()
    c.chat(doc, max_tokens=1, temperature=0, chat_template_kwargs={"enable_thinking": False})
    q1, h1 = counters()
    print(f"run {i}: {time.time() - t:.1f} s, queried {q1 - q0:.0f}, hit {h1 - h0:.0f} tokens")
EOF
```

---

## 4. Same-boot nondeterminism is far above rounding noise

**Symptom.** Two identical teacher-forced captures in one boot, at c=1 with the same shapes, disagree on the argmax at 1.55% of positions (E0 `tier0.py record` A/A: top-1 98.447%, KL 5.137e-3). In greedy decoding, 14 of 20 prompt pairs diverge within 200 tokens in the same boot (hazard 0.0096). E3a's probe found 23% of positions moving more than 0.05 nats between two prefills of the same 1.5k ids, with outliers of 14-26 nats (`evidence/e3a-kpool-det/notes.txt`). `evidence/e3b-av-tau0.1/notes.txt` shows swings of 13 nats inside a single 2040-row chunk at c=1, recurring at the same positions. A same-shape rerun at c=1 would normally be close to bit-identical (expectation, not measured on this stack).

**Root cause (unknown, inference only).** Ruled out by evidence: the sparse-MLA index compaction race (`GLM53_DETERMINISTIC_MLA_INDEX` changed nothing, E3a), the kpool ring and top-k (swings occur below 2048 tokens), Marlin atomic-add reduce (off by default, `marlin_utils.py:633-653`), the mHC split-K GEMM (partials go to separate offsets, `sm120_tf32_hc_prenorm_gemm.cuh:252-268`). Ruled out today: Marlin's NVFP4 scale conversion zeroes e4m3 block scales below 2^-6 (`marlin_utils_fp4.py:115-116`, and the MoE scale factor is 1 whenever any expert uses the full range, `:36-56`), but layers 10 and 44 of `09b04e5` have zero subnormal or zero block scales, so nothing is flushed. Open candidates: router logits rounded to BF16 on sm_121 (MOE-3) creating top-8 near-ties that tiny noise flips; a run-to-run nondeterministic kernel not yet identified; stale KV or state memory read across requests.

**User impact.** Greedy output is not reproducible run to run. Probes flip (items 1 and 11). Every quality gate has a 5e-3 KL noise floor, so a sub-floor regression is invisible.

**Fix.** Localize before fixing. Three cheap experiments, each on the current boot or one extra boot:
1. Order test. Rerun `evidence/e3-tools/prefill_repeat.py` twice: once as is, once with an unrelated 8k-token request between repeats. If the spread grows with the polluter, stale memory is in play.
2. `EXTRA_ENV='GLM53_ROUTER_FP32=1'` boot (built in v13, never booted), then `tier0.py record --only nll` A/A. If top-1 A/A agreement rises above ~99.5%, routing near-ties are the amplifier, and ROUTER_FP32 becomes a knob.
3. The same A/A on the X1 boot, to see whether the re-exported experts change the floor.

**Live test.** `python3 quality/tier0.py record --name r3-aa-<boot> --only nll`, then read `nll.rerun` (`top1_agree`, `kl`) from `~/projects/data/glm53-evals/ref/r3-aa-<boot>.json`. Baseline is 0.98447 / 5.137e-3.

---

## 5. Tier 0 KL gate is right; the relaxed limit hides a real INT8 shift

**Symptom.** The tool fails `nll.kl_stage` on R1, F1 and G1. The notes then re-judge against a "cross-boot A/A" limit of 6.864e-3 and pass R1 at 6.815e-3 (`evidence/r2/notes.txt`).

**Evidence (verified arithmetic).**
- Tool limit = reference same-boot A/A KL 5.137e-3 + fp8 stage margin 1e-3 = 6.137e-3 (`quality/tier0.py` STAGES, `~/projects/data/glm53-evals/ref/nvidia-v11-k7.json`).
- Relaxed limit = E1a's KL 5.864e-3 + 1e-3 = 6.864e-3, and the relaxed top-1 limit 0.97826 = E1a's 0.983264 - 0.005. E1a is not a pure A/A. It ran v13 against the v11 reference, and v13 compiles a different `fused_recurrent_kda` binary even with every switch off (`docker/README-v13.md`).
- BF16-target v13 boots sit at the same-boot floor: E2e 6.05e-3, E3a 5.27e-3, E3b 5.15e-3 and 5.17e-3. INT8-target boots: E4a 6.76e-3, E4b 6.83e-3, F1 6.63e-3, G1 6.66e-3, R1 6.82e-3. INT8 W8A16 adds about +1.3 to +1.6e-3 of KL.

**Root cause.** The gate correctly sees the INT8 drift as larger than the fp8 stage budget. The relaxed limit absorbed it. That was a defensible call (Tier 1 863 vs 857 found no loss), but R1 now passes the relaxed limit by 0.05e-3, so any further change fails both limits.

**User impact.** Maintainers only. A gate that always fails gets ignored.

**Fix.** Record a new Tier 0 reference on today's defaults during P2's B0 boot (`tier0.py record --name b0-v14-int8`), and judge P4 units against it with the tool's own same-boot rule. Write the INT8 +1.5e-3 KL into `quality/README.md` as an accepted cost backed by Tier 1. X1 needs its own reference anyway. Size S.

**Live test.** `python3 quality/tier0.py compare --ref b0-v14-int8 --stage fp8` on the next boot of the same config must pass `nll.kl_stage`.

---

## 6. mHC kernels still JIT-compile during serving; the built warmup is off

**Symptom.** R1 ran nvcc, cicc and ptxas on both nodes in the SERVING phase from 20:36:28 to 20:40:51, inside Tier 0 (`evidence/r2/r1/compilers.tsv`, `memory-by-phase.txt`). The kernels are DeepGEMM `sm120_tf32_hc_prenorm_gemm` variants plus TileLang mHC kernels, about 6-10 s apart.

**Evidence (verified).** The variants differ only in the split count template argument. Older caches hold 12 classes (1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 24, 48). The current image's cache, `~/projects/data/glm53-jit-cache/9599a9fc5fe5/vllm/deep_gemm/cache`, holds 10 and still lacks 4 and 5, so a future prompt length can still trigger a compile. `GLM53_MHC_WARMUP` (`vllm/model_executor/warmup/glm5next_mhc_warmup.py`) compiles every specialisation up to `max_num_batched_tokens` at boot, and run.sh never sets it.

**User impact.** The first request whose token count lands in a new split class stalls for one compile on each node (~5-10 s, NMK-4). With `JIT_CACHE=1` this is paid once per image per node, so every image rebuild re-exposes it.

**Fix.** One boot with `EXTRA_ENV='GLM53_MHC_WARMUP=1'`, then a knob. Size S. Expect +60-70 s on a cold cache and nothing on a warm one.

**Live test.** During `python3 quality/tier0.py compare --ref <ref> --only nll,needle` run `docker exec glm53-flash-nvfp4 pgrep -af 'nvcc|cicc|ptxas'` every few seconds on spark1 and spark2. With the warmup on it must print nothing. Then `ls ~/projects/data/glm53-jit-cache/<id>/vllm/deep_gemm/cache | grep -c hc_prenorm` should be 12.

---

## 7. Head loads weights 3.1x slower than the worker

**Symptom.** "Loading weights took" on the head vs the worker: R1 680.7 / 216.7 s, R2 683.3 / 220.6 s, R3 685.9 / 219.6 s, B0 709.3 / 224.9 s. Ready takes 15-16 min, and about 7.7 min of that is the head waiting on itself.

**Evidence (measured).** The head reads shards 2-3 in 4-5 s each, then about 21-23 s per shard from shard 4 on (R1 `head.log` progress lines). On spark1 today, direct 4 MiB reads of a shard run at 8.6 GB/s and direct 128 KiB reads at 594 MB/s, so the disk is not the limit. The safetensors loader reads through mmap page faults (`read_ahead_kb` 128). The post-load phase is also 2.2x slower on the head (125 s vs 56 s). spark1 is a shared desktop with agent tooling (load average 7.5 today). The root filesystem is 90% full.

**Root cause (inference).** Host contention on spark1 (CPU and IO from non-vLLM processes) on a latency-bound mmap read path. I could not compare spark2's `read_ahead_kb` (no ssh by brief).

**User impact.** Every boot and every one-knob A/B pays ~8 extra minutes.

**Fix levers.** `EXTRA_ENV='GLM53_SKIP_MTP_WEIGHTS=1'` skips the 13.84 GiB BF16 layer-45 tensors per rank (built, never booted). Each rank reads the whole 190.4 GiB checkpoint, so the head runs at ~0.28 GiB/s and the skip saves about 50 s there and ~16 s on the worker. Quiescing spark1's tooling during boots is the larger lever. Raising `read_ahead_kb` needs sudo. `--safetensors-load-strategy=prefetch` exists but prefetches whole files into page cache, which is risky with ~20 GiB free on UMA. Size S each.

**Live test.** `docker logs glm53-flash-nvfp4 2>&1 | grep -E "Loading weights took|Model loading took"` on both nodes for the SKIP_MTP boot vs the previous boot. Record `uptime` and `cat /proc/pressure/io` on spark1 at boot start.

---

## 8. spark1 headroom is 3-6 GiB below spark2

**Symptom.** MemAvailable after ready: R1 9.4 / 12.8 GiB, R2 6.4 / 13.3 GiB, R3 8.4 / 13.0 GiB (spark1 / spark2).

**Evidence (measured).** R1 preflight: spark1 used 6.6 GiB and 4.6 GiB swap before the boot, spark2 4.2 GiB and 2.3 GiB. LOAD started at 115.06 vs 117.41 GiB available (a 2.35 GiB gap). SERVING ended at 8.85 vs 12.13 GiB (a 3.28 GiB gap). So about 2.4 GiB of the gap is spark1's own host load before vLLM starts, and about 1 GiB is head-only vLLM processes. The rest of the boot-to-boot swing (R2's 6.4 GiB) tracks whatever else runs on spark1.

**Root cause (inference for the split).** The head-only vLLM processes are the API server (tokenizer, chat template, MM processor) and the EngineCore. `--mm-processor-cache-gb 1` is an LRU cache that vLLM documents as duplicated per API process and engine core (`vllm/config/multimodal.py:152-160`). It is near empty after the probes and can grow to 1-2 GiB with real vision traffic. The worker runs none of these.

**User impact.** spark1 is the OOM-limiting node. E3a's spark1 fell to 3.53 GiB at ready.

**Fix levers.** `EXTRA_ENV='GLM53_INDEXER_WS_FACTOR=1'` frees ~1.57 GiB per rank of indexer gather workspace at 327680 context (built, never booted; check cell E and the long needle). Lower `MM_PROCESSOR_CACHE_GB` to 0.5 if vision traffic is light. Quiesce spark1 tooling during serving. Size S each.

**Live test.** On each node: `free -h`, then `docker top glm53-flash-nvfp4 -eo pid,rss,args | sort -k2 -n | tail -6`. On the head, compare the API server and EngineCore RSS before and after `python3 quality/vision.py` plus 50 distinct images. With INDEXER_WS_FACTOR, MemAvailable after ready should rise ~1.5 GiB on both nodes.

---

## 9. Pinned `09b04e5` experts are the miscalibrated export

**Evidence (verified today).** NVIDIA's `1d00738` commit message says about a third of routed experts had an uncalibrated `weight_scale_2` (research note, `../GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-research-fp4local/evidence/research-20261030/fp4-local-and-spark-moe.md`). A `1d00738` download is in progress on spark1. Its first 16 shards are present. I compared every expert of layers 3, 10, 24 and 44 found in them against `09b04e5`:
- `weight_scale_2` changed on 10-32 of 54-72 experts per projection. The old value is always larger, by 1.07x to 5.0x (layer 44 down_proj worst).
- The FP4 payload differs on every compared expert, including experts whose scales did not change. `1d00738` is a full re-export, not a scale patch.
- Old layer-44 experts use only 88-160 of the e4m3 block-scale range of 448. No block scale is subnormal, so Marlin's flush-to-zero does not bite (item 4).
- vLLM keeps only the gate's global scale for the fused w13 (`modelopt.py:1530-1538`). On `09b04e5` gate and up carry the same scale (the w1/w3 warning is gone on the nvidia pack, `evidence/iter-nvidia-linear-marlin/SUMMARY.md`). Check the same on `1d00738` at its first boot: `docker logs glm53-flash-nvfp4 2>&1 | grep -c "must match w3_weight_scale_2"` must print 0.

**Fix.** X1 as planned. Add item 1's 4-arm utf8 script and item 4's A/A to the X1 boot. They tell whether X1 also fixes the table corruption and the noise floor.

---

## 10. utf8 and tools scorers misjudge; utf8 text is not saved

**Evidence (verified).**
- `judge_utf8` (`quality/tier0.py:262-278`) keeps the last occurrence of each row number, so a table that corrupts and then rewrites itself can pass. Three of the 12 saved runs passed with leaked cells (e3b tau0.1 run 0 copied the prompt sentence into a cell; F1 run 1 had 13 odd cells and 11 row breaks).
- `run_utf8` returns only counts. Every diagnosis so far needed a separate `utf8_texts.py` rerun.
- `tools args_ok 0.96` is the same two items on E0, F1 and R1, one streamed and one blocking: t31 and t32. The model returns `"start": "Central Park, New York"` and `"start": "Louvre, Paris"` where the data expects `Central Park` and `Louvre`. The JSON, the function name and the glm47 parse are all correct. This is scorer strictness, not a parser, template or model defect.

**Fix.** Return `text`, `odd_cells` and `n_breaks` from `run_utf8` and fail on either count (the `utf8_scan.py` rules). Accept a string argument when the expected value is a case-folded prefix of the returned value, or list both forms in `quality/data/tools50.json` for t31 and t32. Size S. Update the test in `tests/` that covers the scorer.

**Test.** CPU: `judge_utf8` on the four saved `utf8-texts.json` files must fail all 12 runs. Live: `python3 quality/tier0.py compare --ref <ref> --only tools` gives `args_ok` 1.00.

---

## 11. Video probe sits on a near-tie; not a preprocessing bug

**Symptom.** `smoke_vision.py` answers 3186 for frames 3,3,8,8,1,1,6,6.

**Evidence (verified).** For the frame-list path, vLLM loads 8 frames with fps 1 and `do_sample_frames` false. The processor keeps all 8, groups them into the temporal pairs (3,3), (8,8), (1,1), (6,6), and the prompt gets timestamps 0, 2, 4 and 6 seconds (`glm4_1v.py:1264-1340`, CPU run of both samplers). Order is preserved. The answer flips between identical requests on the same boot (compare run, then the three flaky reruns): E2e FAIL/PASS/FAIL/PASS, E3b tau 0.1 PASS/FAIL/PASS/PASS, E4a PASS/FAIL/FAIL/PASS, F1 FAIL/FAIL/PASS/PASS. R1 passed in Tier 0 and then failed in `smoke_vision.py` 20 s later. Over the 34 Tier 0 JSONs plus R1's smoke run there are 22 PASS and 13 FAIL.

**Root cause.** A near-tie in the model's reading of the middle scenes, flipped by item 4's noise. E0's frame-size sweep (`evidence/e0-nvidia-v11/probe-video-cache.txt`) agrees.

**Fix.** Probe only. Use four frames per digit (two temporal groups per scene) or larger glyphs, and score the majority of three calls. Size S.

---

## 12. JIT cache seeding loss in r2 B6 was harmless

**Evidence (verified).** The three files that did not copy are `vllm/modelinfos/*.json` (model-inspection cache, regenerated in seconds) and `flashinfer_autotune_cache/.../autotune_configs.json`. R1's head log shows that file holds no tuning at all: `Loaded 0 configs`, autotune ran 4 s, `Saved 0 configs` (`evidence/r2/r1/head.log:117-122`). They are root-owned mode 0600 because vLLM and FlashInfer write them with `tempfile.mkstemp` plus `os.replace` (`flashinfer/autotuner/autotuner.py:2619-2639`), so a non-root `cp` skipped them. B6's single 128k sample of 1299 tok/s is not explained by them, and B6 was reverted on decode, not prefill. `run.sh` keys the cache by image ID (`jit_cache_host_dir`), which is the safe default. FlashInfer's `cached_ops` are keyed by version directory, not by source hash, so seeding across images is only safe when the new layer leaves FlashInfer untouched.

**Residual.** The cache tree is root-owned, so cleanup needs sudo. Serve-time compiles are item 6. No run.sh change needed.

## Correction after measurement (2026-11-02, r3 B0)

The claim that DFlash2 prefix-cache hits never happen was wrong. On B0 (v14, no fix) a repeated 8k and 32k document hit 4608 and 32256 tokens, and turn-2 TTFT fell from 8.29 to 3.79 s and from 25.40 to 4.66 s (`evidence/r3/b0/prefix.json`). The 0.0% hit rate in r2 logs came from the bench's salted prompts. `PREFIX_CACHE_DFLASH` adds the last block the EAGLE drop removed: 32k turn 2 4.66 to 1.61 s (`evidence/r3/pv/prefix.json`).
