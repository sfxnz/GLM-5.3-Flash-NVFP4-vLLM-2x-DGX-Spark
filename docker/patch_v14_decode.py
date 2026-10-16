"""v14 decode-step patches for the GLM-5.3-Flash serve image (on top of v13).

Every behaviour is opt-in through a GLM53_* environment variable and is off
by default. With none of them set, v14 runs v13's code.

  GLM53_ROUTER_DEDUPE=1     skip the model-side MoE router GEMM that the MoE
                            runner recomputes (42 GEMMs per step; exact)
  GLM53_HEADGATE_MM=1       DSA indexer head-gate as one bf16 x bf16 -> fp32
                            cuBLAS GEMM instead of an fp32 x fp32 gemv
  GLM53_LATE_MLA_PLAN=1     build the syncing sparse-MLA attention metadata
                            after the host-only KDA groups (host order only)

Usage: python3 patch_v14_decode.py [VLLM_ROOT]
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
# GLM53_ROUTER_DEDUPE: models/glm5next/nvidia/model.py
#
# Glm5NextMoE passes gate=self.gate to FusedMoEFactory, so the MoERunner holds
# the gate and _forward_impl computes router_logits itself (moe_runner.py,
# "If the Runner holds the gate"), overwriting the argument before any use.
# The model's own self.gate(hidden_states) is dead work: one 4096x288 GEMM
# plus a split-K reduce per MoE layer per step. With the switch on the model
# passes hidden_states as the (unread) router_logits placeholder.
# --------------------------------------------------------------------------
MODEL_EDITS = [
    (
        "model.py: router dedupe",
        """        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
""",
        """        if _GLM53_ROUTER_DEDUPE and self.experts.gate is self.gate:
            # GLM53_ROUTER_DEDUPE (v14): the runner holds the gate and
            # recomputes router_logits before reading them; skip ours.
            router_logits = hidden_states
        else:
            router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
""",
    ),
    (
        "model.py: router dedupe flag",
        "logger = init_logger(__name__)\n",
        (
            "logger = init_logger(__name__)\n\n"
            "_GLM53_ROUTER_DEDUPE = os.environ.get(\"GLM53_ROUTER_DEDUPE\") == \"1\"\n"
        ),
    ),
]

# --------------------------------------------------------------------------
# GLM53_HEADGATE_MM: models/glm5next/nvidia/attention.py
#
# The indexer head-gate is torch.mm(x.float(), W_fp32) with x [M, 4096] and
# W [4096, 32]. At decode M=8 cuBLAS picks gemmSN_NN<float> on a [1, 2, 1]
# grid: ~72 us for 0.5 MB, 11 layers per step. x and W are bf16 at the source
# (W is a slice of the bf16 wk_weights_proj weight), so their fp32 casts are
# exact; a bf16 x bf16 GEMM with fp32 output and fp32 accumulation multiplies
# the same values. Only the accumulation order can differ.
# --------------------------------------------------------------------------
ATTN_EDITS = [
    (
        "attention.py: headgate flag",
        "import torch\nimport torch.nn.functional as F\n",
        (
            "import os\n\nimport torch\nimport torch.nn.functional as F\n"
        ),
    ),
    (
        "attention.py: headgate flag value",
        "logger = init_logger(__name__)\n",
        (
            "logger = init_logger(__name__)\n\n"
            "_GLM53_HEADGATE_MM = os.environ.get(\"GLM53_HEADGATE_MM\") == \"1\"\n"
        ),
    ),
    (
        "attention.py: headgate mm",
        """        if getattr(self, "_wp_fp32", None) is None:
            self._wp_fp32 = (
                self.wk_weights_proj.weight.data[self.head_dim :, :]
                .t()
                .contiguous()
                .float()
            )
        weights = torch.mm(hidden_states.float(), self._wp_fp32)
""",
        """        if _GLM53_HEADGATE_MM and hidden_states.dtype == torch.bfloat16:
            # GLM53_HEADGATE_MM (v14): same bf16 operands, fp32 accumulate
            # and output, one tensor-core GEMM instead of an fp32 gemv.
            if getattr(self, "_wp_bf16", None) is None:
                self._wp_bf16 = self.wk_weights_proj.weight.data[
                    self.head_dim :, :
                ].t()
            weights = torch.mm(hidden_states, self._wp_bf16, out_dtype=torch.float32)
        else:
            if getattr(self, "_wp_fp32", None) is None:
                self._wp_fp32 = (
                    self.wk_weights_proj.weight.data[self.head_dim :, :]
                    .t()
                    .contiguous()
                    .float()
                )
            weights = torch.mm(hidden_states.float(), self._wp_fp32)
""",
    ),
]

# --------------------------------------------------------------------------
# GLM53_LATE_MLA_PLAN: v1/worker/gpu/attn_utils.py
#
# build_attn_metadata builds the KV-cache groups in index order. The sparse
# MLA group comes before the four KDA groups, and its SM90 builder replans
# with seq_lens[:num_reqs].cpu(), which waits for the GPU (the drafter graph of
# the step before). Every KDA builder then runs on the host while the GPU
# idles (~2.1 ms per verify step in the E5 F3 trace). The KDA builders read
# host tensors only, so building the sparse-MLA group last lets their work
# overlap the drafter graph. Each group's metadata is the same object either
# way; only the host order changes.
# --------------------------------------------------------------------------
ATTN_UTILS_EDITS = [
    (
        "attn_utils.py: late mla plan flag",
        "logger = init_logger(__name__)\n",
        (
            "logger = init_logger(__name__)\n\n"
            "_GLM53_LATE_MLA_PLAN = os.environ.get(\"GLM53_LATE_MLA_PLAN\") == \"1\"\n"
        ),
    ),
    (
        "attn_utils.py: import os",
        "from typing import Any, cast\n\nimport torch\n",
        "import os\nfrom typing import Any, cast\n\nimport torch\n",
    ),
    (
        "attn_utils.py: late mla plan order",
        "    for i in range(num_kv_cache_groups):\n",
        """    group_order = list(range(num_kv_cache_groups))
    if _GLM53_LATE_MLA_PLAN and not for_cudagraph_capture:
        # GLM53_LATE_MLA_PLAN (v14): the SM90 sparse-MLA builder syncs on
        # seq_lens.cpu(); build its group last so the host-only builders of
        # the other groups overlap the GPU instead of following the sync.
        group_order.sort(
            key=lambda g: any(
                type(ag.get_metadata_builder(0)).__name__
                == "FlashInferMLASparseSM90Builder"
                for ag in attn_groups[g]
            )
        )
    for i in group_order:
""",
    ),
]

FILE_EDITS = {
    "models/glm5next/nvidia/model.py": MODEL_EDITS,
    "models/glm5next/nvidia/attention.py": ATTN_EDITS,
    "v1/worker/gpu/attn_utils.py": ATTN_UTILS_EDITS,
}


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
    for path, content in staged.items():
        path.write_text(content)
    print(f"v14 decode patches applied ({len(staged)} files written)")
    return list(staged)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
