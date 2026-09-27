"""Shared stdlib helpers for the quality gates: HTTP client, SSE, paths."""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
# Large outputs (references, datasets, runs) live outside the repo.
EVALS_DIR = Path(os.environ.get("GLM53_EVALS_DIR", Path.home() / "projects/data/glm53-evals"))
BOX_TOKENS = ("<|begin_of_box|>", "<|end_of_box|>")
MAX_CONCURRENCY = 2  # the serve admits MAX_NUM_SEQS=2


class HTTPFailure(RuntimeError):
    def __init__(self, code: int, body: str):
        super().__init__(f"HTTP {code}: {body[:500]}")
        self.code = code
        self.body = body


def strip_box(text: str) -> str:
    for t in BOX_TOKENS:
        text = text.replace(t, "")
    return text


def base_url(url: str) -> str:
    """Accept http://host:port, .../v1 or .../v1/chat/completions."""
    url = url.rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/completions", "/v1"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


class Client:
    def __init__(self, url: str, model: str | None = None, timeout: float = 3600):
        self.url = base_url(url)
        self.timeout = timeout
        self._model = model

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self.get("/v1/models")["data"][0]["id"]
        return self._model

    def get(self, path: str):
        with urllib.request.urlopen(self.url + path, timeout=60) as r:
            body = r.read().decode()
        return body if path == "/metrics" else json.loads(body)

    def _request(self, path: str, body: dict):
        data = json.dumps({"model": self.model, **body}).encode()
        req = urllib.request.Request(self.url + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise HTTPFailure(exc.code, exc.read().decode("utf-8", "replace")) from exc

    def post(self, path: str, body: dict) -> dict:
        with self._request(path, body) as r:
            return json.loads(r.read().decode())

    def tokenize_count(self, text: str) -> int:
        return int(self.post("/tokenize", {"prompt": text})["count"])

    def chat(self, messages, *, stream: bool = False, **body) -> dict:
        """One chat completion. Returns a flat dict (same keys for both modes)."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"messages": messages, **body}
        t0 = time.time()
        if stream:
            body["stream"] = True
            body.setdefault("stream_options", {"include_usage": True})
            with self._request("/v1/chat/completions", body) as r:
                out = parse_sse(r)
        else:
            out = flatten_chat(self.post("/v1/chat/completions", body))
        out["s"] = round(time.time() - t0, 2)
        return out


def flatten_chat(resp: dict) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append({"name": fn.get("name"), "arguments": fn.get("arguments") or ""})
    return {"content": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "tool_calls": calls, "finish_reason": choice.get("finish_reason"),
            "token_ids": choice.get("token_ids"), "usage": resp.get("usage") or {}}


def parse_sse(lines) -> dict:
    """Accumulate an OpenAI chat SSE stream into the flatten_chat shape."""
    content, reasoning, calls = [], [], {}
    finish, usage, token_ids = None, {}, []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        ev = json.loads(payload)
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            content.append(d.get("content") or "")
            reasoning.append(d.get("reasoning") or d.get("reasoning_content") or "")
            token_ids += ch.get("token_ids") or []
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return {"content": "".join(content), "reasoning": "".join(reasoning),
            "tool_calls": [calls[k] for k in sorted(calls)], "finish_reason": finish,
            "token_ids": token_ids or None, "usage": usage}


def pmap(fn, items, workers: int = MAX_CONCURRENCY) -> list:
    with cf.ThreadPoolExecutor(max_workers=max(1, min(MAX_CONCURRENCY, workers))) as ex:
        return list(ex.map(fn, items))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def numbers(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text)]
