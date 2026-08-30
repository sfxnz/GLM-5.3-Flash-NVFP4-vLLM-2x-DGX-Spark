# Quality probes

Thinking-off completions put the answer in `message.content` with no leaked `<think>`. A tools request with `tool_choice=auto` returns a parsed `tool_calls` entry. A unique-salt needle prompt reports prompt tokens, wall time, and prefill tok/s.

## Sub-features

- `thinking-off` runs `thinking_off_probe.py` and requires non-empty `content` with no `<think>` leak.
- `tool-call` runs `tool_call_probe.py` and requires a parsed `get_weather` tool call.
- `prefill-needle` runs `needle_probe.py --prompt-tokens N` with a unique `--salt` and prints `prefill_tok_s`.
- `count-greedy` runs `count_probe.py` and requires a long consecutive integer run.
- `hermes-tools` runs `hermes_probe.py` and requires a parsed tool call, then a `role=tool` follow-up with non-empty assistant `content` and no `<think>` leak.

## How to get to it (user POV)

From the repo root, with the serve ready:

```bash
python3 .cursor/skills/verify-glm53-flash/scripts/thinking_off_probe.py
python3 .cursor/skills/verify-glm53-flash/scripts/tool_call_probe.py
python3 .cursor/skills/verify-glm53-flash/scripts/hermes_probe.py
python3 .cursor/skills/verify-glm53-flash/scripts/count_probe.py
python3 .cursor/skills/verify-glm53-flash/scripts/needle_probe.py --prompt-tokens 8192
```

## Driving it with verify-glm53

Preconditions:

- Doctor prints `status=ready`.
- Working directory is the repo root.

- **Thinking off.** Run `python3 .cursor/skills/verify-glm53-flash/scripts/thinking_off_probe.py`. Exit `0`. Stdout has `leaked_think=0` and `content_chars` greater than 0.
- **Tool call.** Run `python3 .cursor/skills/verify-glm53-flash/scripts/tool_call_probe.py`. Exit `0`. Stdout has `n_tool_calls` greater than 0 and `names` containing `get_weather`.
- **Greedy count.** Run `python3 .cursor/skills/verify-glm53-flash/scripts/count_probe.py`. Exit `0`. `consecutive` is at least `--need` (default 80).
- **Prefill needle.** Run `python3 .cursor/skills/verify-glm53-flash/scripts/needle_probe.py --prompt-tokens 8192`. Exit `0`. Stdout contains `hit=1`, `prompt_tokens=` greater than 0, `wall_s=`, `ttft_s=`, and `prefill_tok_s=`. The `SUMMARY` JSON repeats those fields.
- **Hermes tools loop.** Run `python3 .cursor/skills/verify-glm53-flash/scripts/hermes_probe.py`. Exit `0`. Turn 1 has a parsed `get_weather` tool call. Turn 2 is HTTP 200 with non-empty `content` and `leaked_think=0`.
- **Proof.** Save each command's stdout plus a doctor dump.

## Gotchas

- Omit `--salt` to auto-generate one. Reusing a salt can let prefix cache fake prefill speed.
- Thinking-on can leave `message.content` empty. These probes send `enable_thinking: false`.
- `count_probe.py` is the lossless gate. `bench_decode.py --phase structured` is tok/s, not exact 1–200.
- A tools request that dumps glm47 XML into `content` without `message.tool_calls` is a fail. The recipe serves `--tool-call-parser glm47`.
- A tool follow-up that prefixes `</think>` onto `content` is a fail. Thinking-off seeds an empty `<think></think>` in `chat_template.jinja`. The live process only picks that up after a restart.
