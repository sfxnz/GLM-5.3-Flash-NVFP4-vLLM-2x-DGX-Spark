# Vision smoke

The published recipe accepts an OpenAI `image_url` on `/v1/chat/completions`, does not return HTTP 400 `is not a multimodal model`, and answers synthetic images correctly (`quality/vision.py`).

## Sub-features

- `vision-image-url` posts stdlib-drawn PNGs as `image_url`.
- `vision-multimodal` fails if the engine says the serve is not a multimodal model.
- `vision-answers` asserts token accounting, colours, quadrants, circle counts, OCR, two-image order, and a short frame-list video (SKIP on HTTP 400).

## How to get to it (user POV)

- After `./run.sh` prints the ready URL, from the repo root: `python3 smoke_vision.py`.

## Driving it with verify-glm53

Preconditions:

- Doctor prints `status=ready`.
- Working directory is the repo root.
- `LANGUAGE_MODEL_ONLY` is `0` (the published default).

- **Vision completion.** Run `python3 smoke_vision.py`. Exit code `0`. Stdout ends with `VISION {"pass": true ...}`. Stderr does not contain `is not a multimodal model`.
- **Proof.** Save stdout/stderr plus a doctor dump under `artifacts/vision-smoke/<stamp>/`.

## Gotchas

- Images are drawn in code (no PIL). Every answer is asserted; see `quality/README.md`.
- Do not skip MM profiling into a max-size image+video dummy. That OOMs GB10 UMA.
- `--limit-mm-per-prompt` is `{"image":4,"video":1}`. This suite sends at most two images and one 8-frame video per request.
- Keep `chat_template.jinja`. The official Hub template always opens `<think>`.
