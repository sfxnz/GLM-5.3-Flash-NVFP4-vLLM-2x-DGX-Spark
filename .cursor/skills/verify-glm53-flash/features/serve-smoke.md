# Serve smoke

The published recipe answers a one-sentence hello on the OpenAI-compatible chat API with thinking off, proving the head rank is serving `LibertAIDAI/GLM-5.3-Flash-NVFP4`.

## Sub-features

- `smoke-models` lists the served model at `/v1/models`.
- `smoke-hello` completes “Say hello in one sentence.” with thinking off.
- `smoke-identity` returns that same model id on the completion.

## How to get to it (user POV)

- After `./run.sh` prints the ready URL, run the smoke `curl` in `README.md`.
- Equivalent: `.cursor/skills/verify-glm53-flash/scripts/smoke.sh`.

## Driving it with verify-glm53

Preconditions:

- Doctor prints `status=ready`.
- `GET http://127.0.0.1:8000/v1/models` includes `"id": "LibertAIDAI/GLM-5.3-Flash-NVFP4"` and `"max_model_len": 327680`.

- **Models list.** Run `curl -sf http://127.0.0.1:8000/v1/models`. The JSON `data[0].id` is `LibertAIDAI/GLM-5.3-Flash-NVFP4`.
- **Hello completion.** Run `.cursor/skills/verify-glm53-flash/scripts/smoke.sh`. Exit code `0`. HTTP status is `200`. `response.json` has non-empty `choices[0].message.content` and `"model": "LibertAIDAI/GLM-5.3-Flash-NVFP4"`.
- **Proof.** Keep the directory printed as `evidence=`. It contains `doctor.txt`, `request.json`, `response.json`, and `http_status.txt`.

## Gotchas

- Send `chat_template_kwargs.enable_thinking` false. Thinking-on can leave `message.content` empty. Even with thinking off, this model may prepend chain-of-thought into `content`; non-empty content is still a pass for this smoke.
- Do not send `stream: true` for this feature. The README smoke is a single JSON body.
- A 200 with `"object":"error"` or empty content is a fail.
- This feature does not prove tok/s or spec-decode acceptance. Use the decode bench for that.
- The live lab serve is shared. One 64-token completion is the whole drive; do not follow this feature with a full four-concurrency bench unless that feature was requested.
