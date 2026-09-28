# Quality gates

Two tiers from the 2026-09-25 review plan (PLAN.md section 6). Both talk to
the running serve over its OpenAI API and use only the Python standard
library. Neither ever starts, stops or touches the serve. Concurrency never
goes above 2 (`MAX_NUM_SEQS=2`). Do not run them next to a bench.

| Tier | When | Runtime | Command |
|---|---|---|---|
| 0 | every knob change | ~15 min (`record` ~25 min) | `python3 quality/tier0.py compare --ref <name>` |
| 1 | pack, template, defaults, KV dtype, attention/KDA quant, ABLIT | ~2.75 h per config | `python3 quality/tier1.py run --out <file>` |
| vision only | any boot | < 1 min | `python3 smoke_vision.py` |

Large outputs live in `~/projects/data/glm53-evals/` (`GLM53_EVALS_DIR`
overrides): `ref/` (Tier-0 references, ~40 MB each), `runs/` (Tier-0 results),
`datasets/` (Tier-1 cache). Copy the small result files (`tier0.json`, Tier-1
JSONL and summaries, `compare_tier1.py --json`) into `evidence/<run>/`.

## Tier 0

```bash
# once per baseline (serve up, unchanged config):
python3 quality/tier0.py record --name libertai-caca4e6
# after each change:
python3 quality/tier0.py compare --ref libertai-caca4e6
# weight-quantization stages widen the nll gates around the reference's A/A:
python3 quality/tier0.py compare --ref bf16-attn --stage fp8     # or nvfp4
# the PLAN's fixed floors instead (they cannot pass on this serve, see below):
python3 quality/tier0.py compare --ref bf16-attn --stage fp8 --stage-absolute
```

`record` captures the logit reference and greedy outputs twice, so the
reference carries its own rerun noise (sigma), then runs the probes. `compare`
captures once and gates against the reference. `--only nll,tools` runs a
subset; `--skip-gate greedy.hazard` reports a criterion without gating it
(for example a pack change, where greedy text is expected to move and Tier 1
is the judge). `--long` adds the 128k needle; `--needle-lengths` sets custom
lengths. `--no-video` drops the video probe.

| Criterion | Pass rule |
|---|---|
| `nll.delta` | \|mean NLL - ref\| <= max(3 sigma, floor) nats/token, sigma = the reference's rerun \|delta\|; floor 0.005 (`--stage nvfp4`: 0.01) |
| `nll.top1_rerun` | top-1 agreement >= the reference's rerun agreement - 0.5 points (no `--stage`, or `--stage-absolute`) |
| `nll.top1_stage` | top-1 agreement >= the reference's rerun agreement - 0.5 points (`fp8`) or - 1.5 points (`nvfp4`); replaces `nll.top1_rerun` |
| `nll.kl_stage` | mean top-20 KL <= the reference's rerun KL + 1e-3 (`fp8`) or + 3e-3 (`nvfp4`) |
| `greedy.hazard` | per-token divergence hazard vs the reference's first run <= 2 x max(ref A/A hazard, 0.005) |
| `count` | thinking off, the integers are exactly 1..200 |
| `kwargs.core` | the 10 cells other than `thinking: true` pass (below) |
| `kwargs.thinking_alias` | both `thinking: true` cells pass (the template's `thinking` alias, below) |
| `utf8` | zero U+FFFD, rows 1..40 present, every n^2 right, finish `stop` |
| `tools.json_valid` | >= 98% of 50 tool calls parse as a JSON object (a missing call counts as a failure) |
| `vision` | every image check passes; the video check may SKIP |
| `needle` | at each length (8k, 32k, optionally 128k), 2 of 3 depths found |

`tier0.json` lists every criterion with PASS/FAIL, value and limit, plus a
one-line verdict, for example
`PASS tier0 compare vs libertai-caca4e6: 10/10 criteria`. Exit code 1 on any FAIL.

What each component sends:

- **nll**: 96 frozen documents (186,526 tokens; prose, code, math,
  multilingual, chat-shaped; see `data/SOURCES.md`) as `/v1/completions`
  with `prompt = "[gMASK]<sop>" + doc`, `prompt_logprobs=20`, `max_tokens=1`,
  at c=1. The first two prompt ids must be `[gMASK]` `<sop>` (154822, 154824).
  KL is KL(ref || cand) over the reference's top-20 plus one "rest" bucket; a
  reference top-20 token missing from the candidate's returned list gets
  min(the candidate's 20th prob, its unreturned mass / (missing + 1)).
  `prompt_logprobs=20` needs `--max-logprobs` >= 20 (vLLM default 20).
- **greedy**: 20 fixed prompts x 200 tokens, thinking off, c=1,
  `return_token_ids`.
- **count**: "Count from 1 to 200 ..." with thinking off.
- **kwargs**: "What is the capital of France?" through six shapes, each
  streamed and not: no kwargs (serve default, `--default-thinking off`),
  `enable_thinking: false`, `enable_thinking: true`, `thinking: true`,
  `reasoning_effort: low`, `reasoning_effort: none`. A cell passes when
  content is non-empty and says Paris, has no `<think>` tags, finishes with
  `stop`, reasoning is non-empty exactly when thinking is expected, and with
  thinking off the content does not open with chain-of-thought. `effort_low`
  may leave reasoning empty (the template opens `<think>`, and the model closes
  it at once on this question); then its content is checked for chain-of-thought.
  The two `thinking: true` cells guard QUAL-2: a template that ignores the
  `thinking` alias while the parser treats it as on puts the whole answer in
  `reasoning`. `chat_template.jinja` honors the alias, and both cells pass
  (e0 compare, 2026-09-27: 2/2).
- **utf8**: a streamed 40-row markdown table (n, n², n³, Chinese numeral).
  n³ and numeral errors are reported, not gated.
- **tools**: `data/tools50.json`, `tool_choice: auto`, odd items streamed.
  Name and exact-argument rates are reported, not gated.
- **vision**: `quality/vision.py`, below.
- **needle**: fixed seeds (no per-run salt), so a refusal is reproducible;
  depths 0.1/0.5/0.9.

## Vision suite

`python3 quality/vision.py` (or `python3 smoke_vision.py`, the same thing).
Images are drawn in code and written as PNG with zlib + struct; no PIL.

| Check | Asserts |
|---|---|
| `tokens` | prompt_tokens(448x448 image + text) - prompt_tokens(text) = 256 + 2 |
| `solid.*` | red, green, blue, yellow named in one word |
| `quadrants` | red / green / blue / yellow quadrants all named (catches patch-order bugs) |
| `count.3`, `count.5` | number of black circles |
| `ocr.*` | `7F3A91` and `K9P2X` from a 5x7 bitmap font, exact |
| `order.*` | (red, blue) -> first; (blue, red) -> second (multi-image ordering) |
| `video` | 8 PNG frames as `data:video/jpeg;base64,f1,f2,...` (vLLM's frame-list form), digits 3 8 1 6 in order; SKIP (reason recorded) on any HTTP error, so this optional probe never blocks the smoke |

HTTP 400 `is not a multimodal model` fails the suite with that message.

## Tier 1

```bash
python3 quality/tier1.py fetch                        # once; CPU + network only
python3 quality/tier1.py run --out ~/projects/data/glm53-evals/runs/A.jsonl --label libertai
# change one thing, reboot, then
python3 quality/tier1.py run --out ~/projects/data/glm53-evals/runs/B.jsonl --label nvidia
python3 quality/compare_tier1.py ~/projects/data/glm53-evals/runs/A.jsonl ~/projects/data/glm53-evals/runs/B.jsonl --json cmp.json
```

1,010 items, ids pinned in `data/tier1_ids.json` (seed 20260925):

| Group | Items | Prompt | Score |
|---|---:|---|---|
| IFEval | 150 | the prompt as is | strict prompt-level: every instruction passes (`ifeval.py`) |
| GSM8K | 200 | + "step by step ... Answer: <number>" | numeric equality |
| MMLU-Pro | 280 (20 x 14 categories) | CoT, "The answer is (X)" | letter |
| BFCL | 120 (60 simple, 60 multiple) | BFCL functions as OpenAI tools, `tool_choice: auto` | one call, right name, every arg in the allowed values |
| Vision | 100 ChartQA + 100 OCRBench (10 per type) + 60 MMMU (val, multiple choice, <= 4 images; 27 subjects, see `data/SOURCES.md`) | image(s) + question | ChartQA relaxed (5%), OCRBench substring, MMMU letter |

Runs are greedy (`--temperature 0`) so two configs pair item by item. The
serve's default kwargs apply unless you pass `--chat-kwargs '{"enable_thinking": true}'`
or `--reasoning-effort low`; with thinking on, add `--think-budget 8192` so
reasoning does not truncate answers. `--temperature 1 --seed N` gives seeded
absolute scores. A run is resumable: rerun the same command and it skips
finished items and retries errored ones. `--limit 3` is a quick smoke.

`compare_tier1.py` (A = baseline, B = candidate):

- **pooled**: mean paired difference over all items, in points, with its SE.
  PASS needs the point estimate >= -2.0 (SE about 0.93 at 1,010 items).
- **groups** IFEval, GSM8K, MMLU-Pro, BFCL, Vision: a group FAILS only if the
  exact McNemar p < 0.01 **and** its accuracy drops >= 5 points.
- `INVALID` (exit 2, no verdict) when more than 2% of pairs are lost to errors,
  when either run has items the other lacks (a crashed or `--limit` run), when
  the runs' `meta.ids_sha256` differ, or when the pairs miss any id pinned in
  `quality/data/tier1_ids.json`. `--allow-subset` drops only the last check, for
  two runs made with the same `--tasks` subset.

## Validating on the Sparks

1. Serve up, exclusive GPUs, nothing else running. `python3 smoke_vision.py`
   must print `VISION {"pass": true ...}`.
2. `python3 quality/tier0.py record --name <pack>-<sha>` then, on the same
   boot, `python3 quality/tier0.py compare --ref <pack>-<sha>`. The A/A compare
   must PASS with `nll.delta` near 0 and `greedy.hazard` within its limit;
   if it does not, the noise floor is wrong, not the serve.
3. Negative control: `compare --stage nvfp4` against a reference from the
   other pack (LibertAI vs nvidia) should show a non-zero KL and delta. A
   `LANGUAGE_MODEL_ONLY=1 FORCE_UNSAFE_VISION=1` boot must make `vision` FAIL
   with `is not a multimodal model`.
4. Tier 1 once per pack; `compare_tier1.py` prints the pooled and per-group table.

If `nll` errors on the first live run, check `prompt_logprobs` with DFlash2
(PLAN open question 10) before anything else: the request and response shape
are in `capture_doc` in `tier0.py`. `prompt_logprobs=20` on a ~2k-token
document materialises full-vocab logprobs per rank (a transient of roughly
1-3 GB on UMA next to a 4.14 GiB KV pin), so watch `free -h` on both nodes
during the first `record`.

The reference stores its A/A in `nll.rerun` (`abs_delta` = sigma, `top1_agree`,
`kl`): the two captures `record` makes on one boot. The serve is not run-to-run
deterministic. On the e0 serve (nvidia 09b04e5, v11) the A/A was |dNLL| 6.3e-4,
top-1 98.45% and KL 5.1e-3, and greedy text diverged in 14/20 prompts (hazard
0.0096; `evidence/e0-nvidia-v11/tier0-notes.txt`). The PLAN's fixed stage floors
(fp8 top-1 >= 99% and KL <= 1e-3; nvfp4 98% / 3e-3) sit inside that noise and
can never pass, so `--stage` judges top-1 and KL as margins around the
reference's A/A. `--stage-absolute` keeps the fixed floors for a serve made
deterministic. 3 sigma is 0.0019 on e0, so the 0.005 floor
still sets the `nll.delta` limit. For pack or kernel A/Bs, also run the step-2
A/A compare after a reboot: a cross-boot `nll.delta` above 0.005 means the floor
is too tight for that comparison.

## Determinism

Source read of the v11 tree (paths under `vllm/`). E3a measured the fix for
the confirmed source below; see Result (E3a).

**Confirmed source: the sparse-MLA index compaction races.**

- `forward_mqa` converts each row's top-k token ids to KV slots with
  `triton_convert_req_index_to_global_index(..., return_valid_counts=True)`
  (`v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py:464-471`).
- The row is 2048 + 3 (kpool tail) = 2051 wide, rounded up to 2176
  (`models/glm5next/nvidia/model.py:611-624`). `_remap_tiling` gives one
  program per row only for a power-of-two width
  (`v1/attention/backends/mla/sparse_utils.py:146-151`), so each row gets 17
  tiles.
- The tiles reserve output slots with `tl.atomic_add` (`sparse_utils.py:113`).
  The comment says the order within the prefix is unspecified (`:32-33`). The
  FA2 MLA kernel is deterministic for a given index order, but the order
  changes its online-softmax rescaling and the bf16 rounding of P.
- The grid is `(num_tokens, 17)`. The race is real for decode and verify steps
  (8-16 rows) and for small prefill chunks. A short prompt is a small chunk:
  in e0's greedy A/A, one prompt diverged at token 0. In a chunk of about
  2000 rows, the tiles of one row are dispatched about 2000 blocks apart. That
  is more blocks than the GPU runs at once, so they land in order in practice.

**Probable source (C++ not in the tree).** The indexer's top-k kernels,
`_C.top_k_per_row_prefill` and `_C.persistent_topk`
(`model_executor/layers/sparse_attn_indexer_kpool.py:559-568, 815-822`), may
emit the selected pools in a varying order, and the pools are expanded in
that order (`:570-590`, `:848-874`). A prefill whose longest context is at
most 2048 tokens skips top-k and selects every token in position order
(`:449-480`). Decode always runs top-k.

**Not sources.**

- Marlin MoE: `use_atomic_add=False` and fp32 reduce
  (`fused_moe/experts/marlin_moe.py:159-160, 226-227`), fixed-order sum (`:395`).
- Dense Marlin: atomics need `VLLM_MARLIN_USE_ATOMIC_ADD` and n < 2048
  (`marlin_utils.py:633-653`). Here n is at least 4096.
- NCCL at TP=2: each element sums exactly two operands, and a+b == b+a.
- mHC: TileLang reduces serially (`kernels/mhc/tilelang_kernels.py:105-110`).
- KDA: no atomics on the paths used here.
- Prefix cache: requests with `prompt_logprobs` skip it (`sampling_params.py:529-533`).

**Boot to boot only.** Triton autotune picks for the KDA chunk kernels
(prefill), and FlashInfer JIT debug or release flags.

**`VLLM_BATCH_INVARIANT=1`** refuses this model: the sparse MLA backends, the
KDA backend and the Marlin MoE have no batch-invariant path.

**Logit precision.** `head_dtype` defaults to the model dtype, bf16
(`config/model.py:1908-1936, 2311-2332`). Near logits of 16-32 the bf16
spacing is 0.125, so near-ties are common and ULP-level noise flips top-1.

### Result (E3a)

`GLM53_DETERMINISTIC_MLA_INDEX=1`, a v13 patch now removed, gave each
compacted row one Triton program, so the slot prefix followed the input column
order, and sorted the kpool pools per row before expansion. E3a served it on
both ranks: the Triton cache held only the deterministic kernel variant (16
warps, no atomic). The within-boot A/A (`tier0.py record`, nll and greedy,
`evidence/e3a-kpool-det/notes.txt`) did not move:

| Within-boot A/A | E0 (v11, no switch) | E3a (switch on) |
|---|---|---|
| nll \|dNLL\| | 6.27e-4 | 6.37e-4 |
| nll top-1 agreement | 98.447% | 98.424% |
| nll KL top-20 | 5.137e-3 | 5.137e-3 |
| greedy diverged | 14/20 | 14/20 |
| greedy hazard | 0.0096 | 0.0102 |

The patch served and changed nothing measurable, so it was removed. The
dominant noise is elsewhere. Prefill A/A swings of 5-15 nats recur at fixed
positions even in single-chunk prefill, where neither top-k nor the kpool ring
runs (prefill repeat, `evidence/e3b-av-tau0.1/notes.txt`). The likely
mechanism is a small run-to-run difference flipping a discrete choice such as
MoE top-8 routing (inferred, not traced).

### Staged A/A experiment

Both boots on the v13 image with no switch set (it serves like v11). Record
them under `evidence/<run>/`.

Tier 0's nll corpus is 1999-2000 tokens, or 2001-2002 with `[gMASK]<sop>`.
Each document is therefore one prefill chunk of at most 2048 rows on the
top-k-free path above, where the tiles should already land in order. e0 still
measured a 1.55% top-1 disagreement there. Stage 0 separates the regimes:

- **S**: prompts of at most 1900 tokens. The prefill is one chunk on the
  top-k-free path, and greedy output up to 148 tokens stays within 2048
  tokens of context. For S to be the in-order control, keep the prompts long
  (for example, the Tier-0 documents cut to 1900 tokens). Short prompts race
  like decode.
- **L**: prompts of 2100-2300 tokens. The prefill is a 2048-row chunk plus a
  52-252-row tail chunk (the racy regime), and uses the real top-k.

Neither set exists in `quality/data` yet. Cut both from the corpus sources
(`quality/data/build_corpus.py`). Score each set like Tier 0: `prompt_logprobs`
top-1 agreement, |dNLL| and KL, with L split at position 2048, plus greedy
divergence at 200 tokens (S at 148).

| Stage | Boot | Run | Question |
|---|---|---|---|
| 0 | v13, no switch | S and L twice, then the same on a second boot | Does within-boot noise sit in decode and the L tail (race), or also in S prefill (another source)? How much does the second boot add (autotune, JIT)? |

E3a already answers the first question in part: the index fix changed
nothing, and single-chunk prefill swings at fixed positions. So a source
outside the index compaction dominates, and Stage 0 now only sizes S against L
and the second boot.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_quality*.py'
```

A fake OpenAI server drives record/compare, the vision suite and the SSE
parser; PNGs are decoded back with zlib and every chunk CRC checked.
