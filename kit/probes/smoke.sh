#!/usr/bin/env bash
# vendored from sfxnz/forge kit @ 6285f70
# README smoke: one /v1/chat/completions call ("Say hello in one sentence.", max_tokens 64, greedy,
# thinking off). Usage: kit/probes/smoke.sh <recipe-dir> <evidence-dir>
# Writes <evidence-dir>/smoke.txt (request, response, http status, verdict). Does not start or stop the serve.
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib.sh"
recipe_load "${1:?usage: kit/probes/smoke.sh <recipe-dir> <evidence-dir>}"
OUT="${2:?usage: kit/probes/smoke.sh <recipe-dir> <evidence-dir>}"
mkdir -p "$OUT"
report="$OUT/smoke.txt"

request="$(python3 - "$SERVED_NAME" "$CHAT_TEMPLATE_KWARGS" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": json.loads(sys.argv[2]),
}, indent=2))
PY
)"
response_tmp="$(mktemp)"
trap 'rm -f "$response_tmp"' EXIT
code="$(curl -sS -o "$response_tmp" -w '%{http_code}' --max-time 120 \
  -H 'Content-Type: application/json' \
  -d "$request" \
  "${API}/chat/completions" || true)"

{
  echo "--- request"
  printf '%s\n' "$request"
  echo "--- response"
  cat "$response_tmp"
  echo
  echo "http_status=$code"
} >"$report"

verdict="$(python3 - "$response_tmp" "$code" "$SERVED_NAME" <<'PY'
import json, sys
path, code, want = sys.argv[1:4]
if code != "200":
    print(f"verdict=FAIL reason=http_{code}"); raise SystemExit
try:
    body = json.load(open(path))
except Exception as exc:
    print(f"verdict=FAIL reason=bad_json:{exc}"); raise SystemExit
content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
model = body.get("model") or ""
if model != want:
    print(f"verdict=FAIL reason=unexpected_model:{model}")
elif not content:
    print("verdict=FAIL reason=empty_content")
else:
    print(f"ok model={model} chars={len(content)}")
    print("verdict=PASS")
PY
)"
printf '%s\n' "$verdict" | tee -a "$report"
echo "evidence=$report"
[[ "$verdict" == *"verdict=PASS"* ]]
