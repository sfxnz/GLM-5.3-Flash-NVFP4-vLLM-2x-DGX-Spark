#!/usr/bin/env python3
"""Small build-time check for the optional transplant and TP sharding."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from torch import nn

import ablit_runtime


class Attention(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.o_proj = nn.Linear(input_features, 4, bias=False, dtype=torch.bfloat16)


class Layer(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.self_attn = Attention(input_features)


class Model(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.layers = nn.ModuleList([Layer(input_features), Layer(input_features)])


def write_donor(root, donor):
    raw = donor.view(torch.uint8).numpy().tobytes()
    transplant = Path(root) / "transplant"
    transplant.mkdir()
    (transplant / "L1.bin").write_bytes(raw)
    (transplant / "MANIFEST.json").write_text(
        json.dumps(
            {
                "layers": {
                    "1": {
                        "dtype": "BF16",
                        "shape": list(donor.shape),
                        "nbytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                }
            }
        )
    )


donor = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.bfloat16)
with tempfile.TemporaryDirectory() as root:
    write_donor(root, donor)
    os.environ.update(ABLIT="1", ABLIT_DIR=root, ABLIT_LAYERS="1")

    model = Model(4)
    report = ablit_runtime.maybe_apply(model)
    assert report["edited_layers"] == [1]
    assert torch.equal(model.layers[1].self_attn.o_proj.weight, donor)

with tempfile.TemporaryDirectory() as root:
    write_donor(root, donor)
    os.environ.update(ABLIT="1", ABLIT_DIR=root, ABLIT_LAYERS="1")
    ablit_runtime._tp_world = lambda: 2
    ablit_runtime._tp_rank = lambda: 1

    model = Model(2)
    report = ablit_runtime.maybe_apply(model)
    assert report["tp_rank"] == 1
    assert torch.equal(model.layers[1].self_attn.o_proj.weight, donor[:, 2:])

os.environ["ABLIT"] = "0"
assert ablit_runtime.maybe_apply(Model(4)) is None

print("v12 ablit runtime self-check passed")
