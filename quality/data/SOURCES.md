# Vendored quality data: sources and licenses

These files are frozen. Do not edit, reorder or re-cut them: a Tier-0
reference is only comparable to a run over the same bytes (`tier0.py` checks
the corpus sha256). `build_corpus.py` records exactly how `corpus.jsonl` was
cut and can rebuild it.

## corpus.jsonl (Tier-0 teacher-forced NLL)

96 documents, 186,526 GLM tokens (nvidia 09b04e5 `tokenizer.json`, no special
tokens), 769 KB. Each document is at most 2,000 tokens, cut at a line or word
boundary. The eval prepends `[gMASK]<sop>` and does not score the first 16
document tokens.

| Domain | Docs | Tokens | Source | License |
|---|---:|---:|---|---|
| prose | 36 | 71,952 | Project Gutenberg, 12 English books x 3 passages at 25/50/75% of the body | Public domain (US) |
| multilingual | 21 | 41,880 | Project Gutenberg: fr, de, es, it, pt, fi, nl, zh (3 books), ja | Public domain (US) |
| code | 20 | 36,788 | CPython 3.12.3 `Lib/`, Go 1.22.0 `src/`, SQLite 3.45.0 `src/`, lodash 4.17.21, Rust 1.77.0 `library/` | PSF-2.0, BSD-3-Clause, public domain, MIT, MIT OR Apache-2.0 |
| math | 13 | 24,306 | GSM8K train (worked solutions, calculator annotations removed); MATH train (EleutherAI/hendrycks_math, LaTeX solutions) | MIT, MIT |
| chat | 6 | 11,600 | GSM8K train rendered in the GLM chat shape `<|user|>q<|assistant|><think>solution</think>The answer is N.` | MIT |

Each row names its exact source (eBook number and title, or the pinned raw
URL) in `source`, plus `license`.

Project Gutenberg books: English 1342, 2701, 1661, 84, 98, 345, 3300, 2009,
205, 1404, 145, 408; French 14155, 17489; German 22367, 2229; Spanish 2000;
Italian 45334; Portuguese 55752; Finnish 11940; Dutch 11024; Chinese 24264,
23962, 23950; Japanese 1982. Text was taken from
`https://www.gutenberg.org/cache/epub/<n>/pg<n>.txt`, the Project Gutenberg
header, footer and license were removed, and hard-wrapped lines were joined.
No Project Gutenberg trademark is used.

GSM8K train and MATH train rows never overlap the Tier-1 GSM8K test items.
These texts are almost certainly in the model's training data. Use only the
relative NLL between two serve configs; absolute perplexity means nothing here.

## tools50.json (Tier-0 tool-call JSON validity)

Written for this repo: 14 tool schemas and 50 prompts that each expect one
call, with the expected name and arguments (the exact-argument rate is
reported, not gated).

## tier1_ids.json (Tier-1 item selection)

Only item ids, drawn with seed 20260925 by `tier1.py fetch --resample`. The
datasets themselves are downloaded to `~/projects/data/glm53-evals/datasets/`
and never committed:

| Task | Source | License |
|---|---|---|
| IFEval | google-research `instruction_following_eval/data/input_data.jsonl` @26d8ccd | Apache-2.0 |
| GSM8K | openai/grade-school-math `test.jsonl` @3101c7d | MIT |
| MMLU-Pro | TIGER-Lab/MMLU-Pro test | MIT |
| BFCL | gorilla-llm/Berkeley-Function-Calling-Leaderboard `BFCL_v3_simple` / `BFCL_v3_multiple` | Apache-2.0 |
| ChartQA | HuggingFaceM4/ChartQA test | GPL-3.0 |
| OCRBench | echo840/OCRBench test | research use (see the dataset card) |
| MMMU | MMMU/MMMU validation | Apache-2.0 |

`quality/ifeval.py` ports the IFEval checks from google-research
(Apache-2.0, Copyright The Google Research Authors).

## License notices for vendored code excerpts

**CPython** (`Lib/heapq.py`, `bisect.py`, `textwrap.py`, `fractions.py`,
`statistics.py`, `functools.py`, `json/decoder.py`, `shlex.py` at v3.12.3):
Copyright (c) 2001-2024 Python Software Foundation; All Rights Reserved.
Used under the PSF License Agreement for Python 3.12
(https://docs.python.org/3.12/license.html).

**Go** (`src/sort/sort.go`, `src/container/heap/heap.go`,
`src/strings/builder.go`, `src/bufio/scan.go` at go1.22.0):

    Copyright (c) 2009 The Go Authors. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

       * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the
    distribution.
       * Neither the name of Google Inc. nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**SQLite** (`src/func.c`, `src/util.c` at version-3.45.0): public domain.

**lodash** (`lodash.js` at 4.17.21): Copyright OpenJS Foundation and other
contributors <https://openjsf.org/>, MIT License.

**Rust** (`library/core/src/iter/adapters/zip.rs`,
`library/alloc/src/collections/binary_heap/mod.rs`,
`library/core/src/str/pattern.rs` at 1.77.0): Copyright The Rust Project
Developers, dual-licensed MIT OR Apache-2.0; used under MIT.

**GSM8K**: Copyright (c) 2021 OpenAI, MIT License.
**MATH**: Copyright (c) 2021 Dan Hendrycks, MIT License.

MIT License text (applies to lodash, Rust under MIT, GSM8K, MATH):

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
