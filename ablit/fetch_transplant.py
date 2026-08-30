#!/usr/bin/env python3
"""Fetch the published GLM-5.3 abliterated o_proj tensors for layers 15–45.

Downloads only the tensor byte ranges (~2.7 GB), not the full donor checkpoint.
Adapted from MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks at b5ab809.
Copyright (c) 2026 Mia's AI Lab, used under the MIT license in ../LICENSE.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(os.environ.get("ABLIT_DIR", REPO_ROOT / "ablit")) / "transplant"
DONOR = os.environ.get("ABLIT_DONOR", "dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4")
REVISION = os.environ.get("ABLIT_DONOR_REVISION", "main")
LAYERS = range(15, 46)
CHUNK = 8 * 1024 * 1024
RETRIES = 4

TOKEN = os.environ.get("HF_TOKEN", "")
if not TOKEN:
    token_file = Path.home() / ".cache/huggingface/token"
    if token_file.is_file():
        TOKEN = token_file.read_text().strip()


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def http(url: str, start: int | None = None, end: int | None = None):
    request = urllib.request.Request(url, headers=auth())
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end if end is not None else ''}")
    return urllib.request.urlopen(request, timeout=120)


def fetch_bytes(url: str, start: int, end: int, label: str) -> bytes:
    """Fetch one inclusive byte range, resuming short reads in memory."""
    expected = end - start + 1
    output = bytearray()
    failures = 0
    while len(output) < expected:
        try:
            with http(url, start + len(output), end) as response:
                if response.status != 206:
                    raise RuntimeError(f"expected HTTP 206, got {response.status}")
                before = len(output)
                while chunk := response.read(min(CHUNK, expected - len(output))):
                    output.extend(chunk)
                if len(output) == before:
                    raise RuntimeError(f"short read at {len(output)}/{expected}")
            failures = 0
        except Exception as exc:  # noqa: BLE001
            failures += 1
            if failures >= RETRIES:
                raise SystemExit(f"failed to fetch {label}: {exc}") from exc
            wait = failures * 3
            print(
                f"  retry {failures}/{RETRIES} for {label} at "
                f"{len(output)}/{expected} ({exc}); sleeping {wait}s",
                flush=True,
            )
            time.sleep(wait)
    return bytes(output)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_span(shard_url: str, key: str) -> tuple[int, int, dict]:
    header_size = int.from_bytes(fetch_bytes(shard_url, 0, 7, "safetensors header size"), "little")
    header = json.loads(
        fetch_bytes(shard_url, 8, 8 + header_size - 1, "safetensors header")
    )
    if key not in header:
        raise SystemExit(f"{key} is missing from its mapped donor shard")
    metadata = header[key]
    start, end = metadata["data_offsets"]
    return 8 + header_size + start, 8 + header_size + end - 1, metadata


def get_json(url: str) -> dict:
    with http(url) as response:
        return json.load(response)


def main() -> None:
    encoded_donor = urllib.parse.quote(DONOR, safe="/")
    encoded_revision = urllib.parse.quote(REVISION, safe="")
    donor_sha = get_json(
        f"https://huggingface.co/api/models/{encoded_donor}/revision/{encoded_revision}"
    )["sha"]
    base = f"https://huggingface.co/{encoded_donor}/resolve/{donor_sha}"
    index = get_json(f"{base}/model.safetensors.index.json")["weight_map"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "MANIFEST.json"
    manifest = {
        "donor": DONOR,
        "donor_revision": REVISION,
        "donor_sha": donor_sha,
        "method": "dealign-oproj-transplant",
        "layers": {},
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("donor_sha") == donor_sha:
            manifest = previous
            print("resuming verified donor revision")

    print(f"donor: {DONOR}@{donor_sha}")
    for layer in LAYERS:
        key = f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
        shard = index.get(key)
        if shard is None:
            raise SystemExit(f"donor index has no {key}")
        path = OUT_DIR / f"L{layer}.bin"
        metadata = {int(key): value for key, value in manifest["layers"].items()}
        if (
            path.is_file()
            and layer in metadata
            and path.stat().st_size == int(metadata[layer]["nbytes"])
            and file_sha256(path) == metadata[layer]["sha256"]
        ):
            print(f"L{layer}: already fetched ({path.stat().st_size / 1e6:.0f} MB)")
            continue

        shard_url = f"{base}/{shard}"
        start, end, tensor = tensor_span(shard_url, key)
        size = end - start + 1
        print(f"L{layer}: {tensor['dtype']} {tensor['shape']} ({size / 1e6:.0f} MB)")
        data = fetch_bytes(shard_url, start, end, f"layer {layer}")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        manifest["layers"][str(layer)] = {
            "shard": shard,
            "key": key,
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "nbytes": size,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(int(item["nbytes"]) for item in manifest["layers"].values())
    print(f"done: {len(manifest['layers'])}/31 tensors, {total / 1e9:.2f} GB -> {OUT_DIR}")


if __name__ == "__main__":
    main()
