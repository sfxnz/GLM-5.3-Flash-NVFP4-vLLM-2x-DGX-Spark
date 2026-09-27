"""v13 determinism patch for the GLM-5.3-Flash serve image (on top of v11).

  GLM53_DETERMINISTIC_MLA_INDEX=1  canonical, run-to-run fixed order of the
                                   sparse-MLA kv indices

Off unless the variable is exactly "1". Off, every patched line takes the v11
path: the new kernel argument defaults to 0 and its branch is compiled out,
and both host-side helpers return their input unchanged.

Why: the sparse-MLA index conversion compacts each row's valid slots into
[0, valid_count) with an atomic slot allocator across 17 column tiles (the
2176-wide buffer is not a power of two, so it never gets the single-tile
path). The prefix order then depends on tile scheduling, and the FA2 MLA
kernel's online softmax and bf16 P rounding follow that order. On, one Triton
program owns each row (padded to the next power of two with masked loads), so
the prefix is the input column order. The kpool top-k pools are also sorted
per row before expansion, so the input order no longer depends on how the
top-k kernel emits its selection. CUDA-graph safe: no host sync, and every
shape is unchanged. The MLA kernel is not touched.

The same file lets LOGITS_FP32=1 (run.sh: --hf-overrides
'{"text_config":{"head_dtype":"float32"}}') boot. v11's LogitsProcessor
accepts a different head_dtype only for an UnquantizedEmbeddingMethod lm_head,
and ModelOpt gives the excluded lm_head UnquantizedLinearMethod, which is the
same plain BF16 weight. That branch runs only when head_dtype differs from the
model dtype, which never happens without the override.

Usage: python3 patch_v13_determinism.py [VLLM_ROOT]
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
# v1/attention/backends/mla/sparse_utils.py: one padded program per row
# --------------------------------------------------------------------------
SPARSE_UTILS_EDITS = [
    (
        "sparse_utils.py: switch",
        '''"""Utility functions for sparse MLA backends."""

import torch

from vllm.triton_utils import tl, triton
''',
        '''"""Utility functions for sparse MLA backends."""

import os

import torch

from vllm.triton_utils import tl, triton

# GLM53_DETERMINISTIC_MLA_INDEX (v13): see triton_convert_req_index_to_global_index.
_GLM53_DET_MLA_INDEX = os.environ.get("GLM53_DETERMINISTIC_MLA_INDEX") == "1"
''',
    ),
    (
        "sparse_utils.py: NUM_COLS kernel argument",
        """    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
""",
        """    out_stride0,
    out_stride1,
    # GLM53_DETERMINISTIC_MLA_INDEX (v13): the real row width when one program
    # owns a row padded to a power of two (BLOCK_N > NUM_COLS); 0 = not padded.
    # Only set together with COMPACT_TO_FRONT, whose stores are masked.
    NUM_COLS: tl.constexpr = 0,
):
    # program_id(0) -> token_id (row)
""",
    ),
    (
        "sparse_utils.py: padded lanes load as -1",
        """    tok = tl.load(ti_ptr)  # int32
""",
        """    if NUM_COLS > 0:
        # Padded lanes read as -1: invalid, so never counted or stored.
        tok = tl.load(ti_ptr, mask=indice_id < NUM_COLS, other=-1)
    else:
        tok = tl.load(ti_ptr)  # int32
""",
    ),
    (
        "sparse_utils.py: one padded tile per compacted row",
        """    single_tile, block_n, tiles_per_row, num_warps = _remap_tiling(
        NUM_TOPK_TOKENS, BLOCK_N, return_valid_counts
    )
""",
        """    single_tile, block_n, tiles_per_row, num_warps = _remap_tiling(
        NUM_TOPK_TOKENS, BLOCK_N, return_valid_counts
    )
    # GLM53_DETERMINISTIC_MLA_INDEX (v13): with several tiles per row the
    # atomic slot allocator below orders the compacted prefix by tile
    # scheduling, which changes run to run, and the sparse-MLA softmax follows
    # that order. One program per row, padded to a power of two, makes the
    # prefix the input column order. 16 warps at 4096 lanes keep the stock
    # 2048-lane single tile's 8 lanes per thread.
    num_cols = 0
    if _GLM53_DET_MLA_INDEX and return_valid_counts and not single_tile:
        num_cols = NUM_TOPK_TOKENS
        block_n = triton.next_power_of_2(NUM_TOPK_TOKENS)
        single_tile, tiles_per_row = True, 1
        num_warps = 16 if block_n > 2048 else 8
""",
    ),
    (
        "sparse_utils.py: pass NUM_COLS",
        """        out_stride1,
        num_warps=num_warps,
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out


def triton_filter_and_convert_dcp_index(
""",
        """        out_stride1,
        NUM_COLS=num_cols,
        num_warps=num_warps,
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out


def triton_filter_and_convert_dcp_index(
""",
    ),
]

# --------------------------------------------------------------------------
# model_executor/layers/sparse_attn_indexer_kpool.py: sorted pools per row
# --------------------------------------------------------------------------
INDEXER_EDITS = [
    (
        "sparse_attn_indexer_kpool.py: switch and pool sort",
        """logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
""",
        """logger = init_logger(__name__)

# GLM53_DETERMINISTIC_MLA_INDEX (v13): the top-k kernels leave the order of
# the selected pools unspecified. Sorting them per row makes the expanded
# token list ascending, so the sparse-MLA input order is canonical. The -1
# fill sorts first; expand maps it to -1 wherever it sits.
_GLM53_DET_MLA_INDEX = os.environ.get("GLM53_DETERMINISTIC_MLA_INDEX") == "1"
if _GLM53_DET_MLA_INDEX:
    logger.info(
        "GLM53_DETERMINISTIC_MLA_INDEX: kpool pools sorted per row, sparse-MLA "
        "index compaction in column order (one program per row)"
    )


def _glm53_canonical_pools(pool_topk: torch.Tensor) -> torch.Tensor:
    if not _GLM53_DET_MLA_INDEX:
        return pool_topk
    return torch.sort(pool_topk, dim=-1).values


RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
""",
    ),
    (
        "sparse_attn_indexer_kpool.py: prefill pools",
        """            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:
""",
        """            if index_kpool > 1:
                pool_ids = _glm53_canonical_pools(pool_topk).to(torch.int64)
                if positions is not None:
""",
    ),
    (
        "sparse_attn_indexer_kpool.py: decode pools",
        """        if index_kpool > 1:
            pool_ids = pool_topk.to(torch.int64)
            n = pool_topk.shape[0]
""",
        """        if index_kpool > 1:
            pool_ids = _glm53_canonical_pools(pool_topk).to(torch.int64)
            n = pool_topk.shape[0]
""",
    ),
]

# --------------------------------------------------------------------------
# model_executor/layers/logits_processor.py: LOGITS_FP32 on the ModelOpt lm_head
# --------------------------------------------------------------------------
LOGITS_EDITS = [
    (
        "logits_processor.py: accept the ModelOpt lm_head",
        """        if not isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError(
""",
        """        # GLM53 (v13, LOGITS_FP32): ModelOpt gives an excluded lm_head
        # UnquantizedLinearMethod, whose weight is the same plain BF16 tensor.
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        if not isinstance(
            lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        ):
            raise ValueError(
""",
    ),
]

FILE_EDITS = {
    "v1/attention/backends/mla/sparse_utils.py": SPARSE_UTILS_EDITS,
    "model_executor/layers/sparse_attn_indexer_kpool.py": INDEXER_EDITS,
    "model_executor/layers/logits_processor.py": LOGITS_EDITS,
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
    # All anchors checked and every result parses: now write.
    for path, content in staged.items():
        path.write_text(content)
    print(f"v13 determinism patches applied ({len(staged)} files written)")
    return list(staged)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
