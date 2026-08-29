#!/usr/bin/env bash
# README smoke: one /v1/chat/completions call. Writes evidence, does not start or stop the serve.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

FEATURE="${FEATURE:-serve-smoke}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ARTIFACTS/$FEATURE/$STAMP"
mkdir -p "$OUT"

if ! "$SCRIPT_DIR/doctor.sh" >"$OUT/doctor.txt"; then
  echo "doctor failed; not driving. see $OUT/doctor.txt" >&2
  cat "$OUT/doctor.txt" >&2
  exit 1
fi

cat >"$OUT/request.json" <<'JSON'
{
  "model": "LibertAIDAI/GLM-5.3-Flash-NVFP4",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false}
}
JSON

code="$(curl -sS -o "$OUT/response.json" -w '%{http_code}' --max-time 120 \
  -H 'Content-Type: application/json' \
  -d @"$OUT/request.json" \
  "http://127.0.0.1:${PORT}/v1/chat/completions")"
printf '%s\n' "$code" >"$OUT/http_status.txt"

python3 - "$OUT/response.json" "$code" <<'PY'
import json, sys
path, code = sys.argv[1], sys.argv[2]
if code != "200":
    raise SystemExit(f"HTTP {code}, expected 200")
body = json.load(open(path))
content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
model = body.get("model") or ""
if model != "LibertAIDAI/GLM-5.3-Flash-NVFP4":
    raise SystemExit(f"unexpected model {model!r}")
if not content:
    raise SystemExit("empty message content")
print(f"ok model={model} chars={len(content)}")
print(content)
PY

echo "evidence=$OUT"
