# vendored from sfxnz/forge kit @ 6f40808
"""Shared Python helpers for the kit: recipe.yaml access and one JSON POST.

Every kit script takes a recipe directory. The values it needs (port, served model name,
the chat-template kwargs that turn thinking off) come from recipe.yaml; an env var of the
same name as a serve.env key wins, matching run.sh's NAME="${NAME:-value}" rule.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import yaml


class Recipe:
    def __init__(self, recipe_dir: str | os.PathLike):
        self.dir = Path(recipe_dir).resolve()
        self.data = yaml.safe_load((self.dir / "recipe.yaml").read_text())

    def env(self, name: str, default: str = "") -> str:
        if os.environ.get(name):
            return os.environ[name]
        v = (self.data.get("serve") or {}).get("env", {}).get(name, default)
        return "" if v is None else str(v)

    @property
    def port(self) -> str:
        return self.env("PORT", "8000")

    @property
    def base_url(self) -> str:
        return ((self.data.get("bench") or {}).get("base_url") or f"http://127.0.0.1:{self.port}").rstrip("/")

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/v1/models"

    @property
    def served_name(self) -> str:
        return os.environ.get("SERVED_NAME") or str(self.data["model"]["served_name"])

    @property
    def chat_template_kwargs(self) -> dict:
        return dict((self.data.get("bench") or {}).get("chat_template_kwargs") or {})

    def probes(self) -> dict[str, object]:
        """recipe.yaml probes: list -> {name: params}. A bare string has params None."""
        out: dict[str, object] = {}
        for item in self.data.get("probes") or []:
            if isinstance(item, str):
                out[item] = None
            elif isinstance(item, dict) and len(item) == 1:
                (name, params), = item.items()
                out[name] = params
            else:
                raise ValueError(f"probes: entry must be a name or {{name: params}}, got {item!r}")
        return out


def post_json(url: str, body: dict, timeout: int = 120) -> tuple[int, dict]:
    """POST body, return (http_status, parsed_json). HTTP errors return their status and body."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw[:1500]}
        return exc.code, payload
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # No HTTP status: the serve is down or unreachable.
        return 0, {"error": str(exc)}
