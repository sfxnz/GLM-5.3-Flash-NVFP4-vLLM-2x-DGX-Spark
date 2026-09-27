"""Fake OpenAI-compatible serve shared by the quality tests (no test cases here)."""
from __future__ import annotations

import json
import math
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOS_TEXT = "[gMASK]<sop>"
BOS_IDS = [154822, 154824]


class FakeServe:
    """Minimal OpenAI-compatible server. `chat_fn(body) -> dict` answers chat;
    completions returns deterministic prompt_logprobs, shifted by `nll_shift`."""

    def __init__(self, chat_fn=None, nll_shift: float = 0.0, swap_top: bool = False):
        self.chat_fn = chat_fn or (lambda body: {"content": "ok"})
        self.nll_shift = nll_shift
        self.swap_top = swap_top
        self.requests: list[tuple[str, dict]] = []
        serve = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, obj, stream: bool = False):
                data = obj if stream else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"data": [{"id": "fake-glm"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                serve.requests.append((self.path, body))
                if self.path == "/tokenize":
                    return self._send(200, {"count": len(body["prompt"].split())})
                if self.path == "/v1/completions":
                    return self._send(200, serve.completion(body))
                out = serve.chat_fn(body)
                if "http_error" in out:
                    return self._send(out["http_error"], {"error": {"message": out["message"]}})
                if body.get("stream"):
                    return self._send(200, serve.sse(out), stream=True)
                msg = {"content": out.get("content"), "reasoning": out.get("reasoning")}
                if out.get("tool_calls"):
                    msg["tool_calls"] = [{"type": "function", "function": c} for c in out["tool_calls"]]
                self._send(200, {"choices": [{"message": msg, "finish_reason": out.get("finish_reason", "stop"),
                                              "token_ids": out.get("token_ids")}],
                                 "usage": {"prompt_tokens": out.get("prompt_tokens", 10), "completion_tokens": 5}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()

    def completion(self, body: dict) -> dict:
        text = body["prompt"]
        assert text.startswith(BOS_TEXT)
        words = text[len(BOS_TEXT):].split()
        ids = BOS_IDS + [1000 + zlib.crc32(w.encode()) % 50 for w in words]
        k = body["prompt_logprobs"]
        plp = [None]
        for i in range(1, len(ids)):
            # A fixed distribution over 1000..1049 that rotates with position.
            logits = [-(((j - i) % 50) * 0.3) for j in range(50)]
            if self.swap_top:
                logits[i % 50], logits[(i + 1) % 50] = logits[(i + 1) % 50], logits[i % 50]
            z = math.log(sum(math.exp(x) for x in logits))
            lps = {1000 + j: logits[j] - z - self.nll_shift for j in range(50)}
            order = sorted(lps, key=lambda t: -lps[t])
            entry = {str(t): {"logprob": lps[t], "rank": r + 1, "decoded_token": "x"}
                     for r, t in enumerate(order[:k])}
            if 1000 <= ids[i] < 1050:
                entry[str(ids[i])] = {"logprob": lps[ids[i]], "rank": order.index(ids[i]) + 1, "decoded_token": "x"}
            plp.append(entry)
        return {"choices": [{"text": "", "prompt_token_ids": ids, "prompt_logprobs": plp,
                             "finish_reason": "length"}], "usage": {}}

    @staticmethod
    def sse(out: dict) -> bytes:
        evs = []
        text = out.get("content") or ""
        for i in range(0, len(text), 7):
            evs.append({"choices": [{"delta": {"content": text[i:i + 7]}}]})
        if out.get("reasoning"):
            evs.insert(0, {"choices": [{"delta": {"reasoning": out["reasoning"]}}]})
        for k, c in enumerate(out.get("tool_calls") or []):
            half = len(c["arguments"]) // 2
            evs.append({"choices": [{"delta": {"tool_calls": [{"index": k, "function": {
                "name": c["name"], "arguments": c["arguments"][:half]}}]}}]})
            evs.append({"choices": [{"delta": {"tool_calls": [{"index": k, "function": {
                "arguments": c["arguments"][half:]}}]}}]})
        evs.append({"choices": [{"delta": {}, "finish_reason": out.get("finish_reason", "stop")}]})
        evs.append({"choices": [], "usage": {"prompt_tokens": out.get("prompt_tokens", 10), "completion_tokens": 5}})
        return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in evs) + b"data: [DONE]\n\n"
