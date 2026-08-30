"""Optional load-time GLM-5.3 o_proj transplant.

The launcher exposes one switch: ABLIT=1. Donor tensors are full BF16 weights;
each TP rank copies its input-dimension shard after the stock checkpoint loads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:
    logger = logging.getLogger("glm53_ablit")

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$")
_MTP_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mtp_block\.self_attn\.o_proj$")


class AblitError(RuntimeError):
    """The explicitly enabled transplant could not be applied safely."""


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise AblitError(f"{name} must be 0 or 1, got {value!r}")


def _tp_world() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


def _layers(spec: str) -> set[int]:
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            if lo > hi:
                raise AblitError(f"inverted ABLIT_LAYERS range: {part}")
            result.update(range(lo, hi + 1))
        else:
            result.add(int(part))
    if not result:
        raise AblitError("ABLIT_LAYERS is empty")
    return result


def _text_model(model: Any) -> Any:
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return model
    return getattr(language_model, "model", language_model)


def maybe_apply(model: Any) -> dict[str, Any] | None:
    """Apply the configured transplant, or do nothing when ABLIT is disabled."""
    if not _flag("ABLIT"):
        return None

    root = Path(os.environ.get("ABLIT_DIR", "/opt/glm53/ablit")) / "transplant"
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise AblitError(f"missing donor manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    metadata = {
        int(layer): info
        for layer, info in (manifest.get("layers") or manifest.get("tensors") or {}).items()
    }
    wanted = _layers(os.environ.get("ABLIT_LAYERS", "15-45"))
    world, rank = _tp_world(), _tp_rank()
    if world < 1 or rank not in range(world):
        raise AblitError(f"invalid tensor-parallel rank {rank}/{world}")
    edited: list[int] = []

    for name, module in _text_model(model).named_modules():
        match = _MTP_RE.search(name)
        if match is None:
            match = _LAYER_RE.search(name)
        if match is None:
            continue

        layer = int(match.group(1))
        if layer not in wanted:
            continue
        info = metadata.get(layer)
        if info is None:
            raise AblitError(f"donor manifest has no layer {layer}")
        if info.get("dtype") != "BF16":
            raise AblitError(f"donor layer {layer} is not BF16")

        path = root / f"L{layer}.bin"
        if not path.is_file():
            raise AblitError(f"donor layer {layer} is missing: {path}")
        raw = bytearray(path.read_bytes())
        if len(raw) != int(info["nbytes"]):
            raise AblitError(f"donor layer {layer} size mismatch")
        if hashlib.sha256(raw).hexdigest() != info.get("sha256"):
            raise AblitError(f"donor layer {layer} sha256 mismatch")
        donor = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(tuple(info["shape"]))
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            raise AblitError(f"{name} has no 2-D weight")

        local_in = weight.shape[1]
        if donor.shape[0] != weight.shape[0] or donor.shape[1] != local_in * world:
            raise AblitError(
                f"{name} shape {tuple(weight.shape)} does not match donor "
                f"{tuple(donor.shape)} at TP={world}"
            )
        donor = donor[:, rank * local_in : (rank + 1) * local_in]
        with torch.no_grad():
            weight.copy_(donor.to(device=weight.device, dtype=weight.dtype))
        edited.append(layer)

    if not edited:
        raise AblitError("ABLIT=1 matched no o_proj weights")
    logger.info("ablit: transplanted donor o_proj layers=%s TP=%d rank=%d", edited, world, rank)
    return {"edited_layers": edited, "tp_world": world, "tp_rank": rank}
