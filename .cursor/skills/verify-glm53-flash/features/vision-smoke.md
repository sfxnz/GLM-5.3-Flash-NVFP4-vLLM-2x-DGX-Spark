# Vision smoke

The published recipe accepts an OpenAI `image_url` on `/v1/chat/completions` and does not return HTTP 400 `is not a multimodal model`.

## Sub-features

- `vision-image-url` posts a tiny JPEG as `image_url`.
- `vision-multimodal` fails if the engine says the serve is not a multimodal model.
- `vision-content` requires non-empty `choices[0].message.content`.

## How to get to it (user POV)

- After `./run.sh` prints the ready URL, from the repo root: `python3 smoke_vision.py`.

## Driving it with verify-glm53

Preconditions:

- Doctor prints `status=ready`.
- Working directory is the repo root.
- `LANGUAGE_MODEL_ONLY` is `0` (the published default).

- **Vision completion.** Run `python3 smoke_vision.py`. Exit code `0`. Stdout is JSON with non-empty `content`. Stderr does not contain `is not a multimodal model`.
- **Proof.** Save stdout/stderr plus a doctor dump under `artifacts/vision-smoke/<stamp>/`.

## Gotchas

- The JPEG is a hardcoded 1×1 placeholder so clone-shape CI does not need PIL. Do not fail this smoke because the model did not say "red".
- Do not skip MM profiling into a max-size image+video dummy. That OOMs GB10 UMA.
- `--limit-mm-per-prompt` is `{"image":4,"video":1}`. This smoke sends one small image.
- Keep `chat_template.jinja`. The official Hub template always opens `<think>`.
