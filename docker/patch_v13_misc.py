"""v13 misc runtime patches for the GLM-5.3-Flash serve image (on top of v11).

Every behaviour is opt-in through a GLM53_* environment variable and is off
by default. With none of them set, the patched image behaves like v11. See
docker/README-v13.md for each switch, its expected effect and how to validate
it on the Sparks.

  GLM53_ROUTER_FP32=1              MoE router logits via cuBLAS bf16xbf16->fp32
  GLM53_INDEXER_WS_FACTOR=<int>    DSA indexer prefill workspace factor (stock 40)

Usage: python3 patch_v13_misc.py [VLLM_ROOT]
VLLM_ROOT defaults to the image's site-packages vllm directory.

Every edit is an exact-substring replace. An edit whose replacement text is
already present counts as applied (so reruns are no-ops); an edit whose anchor
does not occur exactly once refuses the whole run before anything is written.
"""

import ast
import sys
from pathlib import Path

DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"

# --------------------------------------------------------------------------
# GLM53_ROUTER_FP32: models/glm5next/nvidia/model.py
# --------------------------------------------------------------------------
MODEL_EDITS = [
    (
        "model.py: import os",
        "from collections.abc import Iterable\nfrom typing import ClassVar, Literal\n",
        (
            "import os\nfrom collections.abc import Iterable\n"
            "from typing import ClassVar, Literal\n"
        ),
    ),
    (
        "model.py: router fp32 opt-in",
        """        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
""",
        """        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        # GLM53_ROUTER_FP32 (v13): GateLinear enables its cuBLAS bf16xbf16->fp32
        # tier (torch.mm out_dtype) only on SM90/SM100. On SM12x it falls to a
        # BF16 F.linear + .to(fp32), so the logits are BF16-rounded although
        # moe_router_dtype is float32. Opt in to the fp32-output GEMM here.
        if (
            os.environ.get("GLM53_ROUTER_FP32") == "1"
            and current_platform.is_cuda()
            and self.router_dtype == torch.float32
            and self.gate.weight.dtype == torch.bfloat16
            and self.gate.bias is None
            and not self.gate.allow_cublas_router_gemm
        ):
            self.gate.allow_cublas_router_gemm = True
            logger.info_once(
                "GLM53_ROUTER_FP32: MoE router GEMM uses cuBLAS bf16xbf16->fp32 "
                "(torch.mm out_dtype)"
            )
""",
    ),
]

# --------------------------------------------------------------------------
# GLM53_INDEXER_WS_FACTOR: v1/attention/backends/mla/indexer.py
# --------------------------------------------------------------------------
INDEXER_EDITS = [
    (
        "indexer.py: import os",
        "from dataclasses import dataclass\n",
        "import os\nfrom dataclasses import dataclass\n",
    ),
    (
        "indexer.py: workspace factor",
        """    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40
""",
        """    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    # GLM53_INDEXER_WS_FACTOR (v13): override the 40. The metadata builder's
    # chunk planner and the indexer op both size from this function, so they
    # stay consistent; 1 still fits one max_model_len request per chunk.
    ws_factor = os.environ.get("GLM53_INDEXER_WS_FACTOR", "").strip()
    if ws_factor:
        if not ws_factor.isdigit() or int(ws_factor) < 1:
            raise ValueError(
                f"GLM53_INDEXER_WS_FACTOR must be an integer >= 1, got {ws_factor!r}"
            )
        logger.info_once(
            "GLM53_INDEXER_WS_FACTOR=%s: indexer prefill buffer %d entries "
            "(stock factor 40: %d)",
            ws_factor,
            max_model_len * int(ws_factor),
            max_model_len * 40,
        )
        return max_model_len * int(ws_factor)
    return max_model_len * 40
""",
    ),
]

FILE_EDITS = {
    "models/glm5next/nvidia/model.py": MODEL_EDITS,
    "v1/attention/backends/mla/indexer.py": INDEXER_EDITS,
}

NEW_FILES: dict[str, str] = {}


def plan_edits(src: str, edits: list[tuple[str, str, str]]) -> tuple[str, int]:
    """Apply ``edits`` to ``src``; return (new_src, n_applied). Raise
    SystemExit if an anchor is missing or ambiguous."""
    applied = 0
    for what, old, new in edits:
        if src.count(new) == 1:
            continue  # already applied
        if src.count(old) != 1:
            raise SystemExit(
                f"refusing: anchor for '{what}' found {src.count(old)} times "
                "(expected 1); upstream source drifted"
            )
        src = src.replace(old, new)
        applied += 1
    return src, applied


def main(root: Path) -> list[Path]:
    if not (root / "__init__.py").is_file():
        raise SystemExit(f"refusing: {root} is not a vllm package directory")
    staged: dict[Path, str] = {}
    for rel, edits in FILE_EDITS.items():
        path = root / rel
        if not path.is_file():
            raise SystemExit(f"refusing: {path} missing")
        src, applied = plan_edits(path.read_text(), edits)
        ast.parse(src, filename=str(path))
        print(f"{rel}: {applied}/{len(edits)} edits to apply")
        if applied:
            staged[path] = src
    for rel, content in NEW_FILES.items():
        path = root / rel
        if path.is_file():
            if path.read_text() != content:
                raise SystemExit(f"refusing: {path} exists with other content")
            print(f"{rel}: present")
            continue
        ast.parse(content, filename=str(path))
        print(f"{rel}: new file")
        staged[path] = content
    # All anchors checked and every result parses: now write.
    for path, content in staged.items():
        path.write_text(content)
    print(f"v13 misc patches applied ({len(staged)} files written)")
    return list(staged)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
