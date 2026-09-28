"""v13 kpool tail-ring patch for the GLM-5.3-Flash serve image (on top of v11).

  GLM53_KPOOL_TAIL_FIX=1   size each request's indexer tail ring for the
                           speculative verify window, so rejected verify rows
                           never overwrite a committed slot

Off unless GLM53_KPOOL_TAIL_FIX=1: the tail block keeps index_kpool slots and
every address the kernels compute is v11's. docker/README-v13.md has the bug,
the bound, the proof and the GPU validation.

v11 keeps the open pool's raw indexer K and gate in a ring of index_kpool slots
per request, addressed pos % kpool. A verify step stashes all 1 + k rows in
position order, so rows past the accepted prefix overwrite the slots of
committed positions of the open pool, and the pool that later completes reads
rejected drafts. With the switch on, the tail block (the ring) has R slots
addressed pos % R, R the next power of two >= k + kpool - 1 (16 for k = 7).
The ring lives in the tail block's page, which v11 already pads to the indexer
page, so nothing is allocated. The prefill seed kernel addresses the block
through the tail view's real strides (v11 assumes a contiguous view, but the
view is carved out of the indexer tensor at the indexer's page stride).

The tail slot mapping (v1/attention/backends/mla/indexer.py) changes too:

- The V2 runner (SPEC=dflash2 and SPEC=mtp) builds no positions for the
  attention metadata, so v11's tail builder keeps the generic slots, which put
  every token at pos >= block_size into block 0: one ring shared by all
  running requests. With the switch the builder derives each token's position
  from seq_lens and query_start_loc and maps it into its request's own block.
- It writes the slots in place into the tail group's persistent slot-mapping
  row. A FULL CUDA graph (the uniform verify batch) replays the address it
  captured; v11's fresh clone would be read stale on replay.
- It maps the real tokens only. CUDA-graph padding keeps the slot kernel's -1.

Usage: python3 patch_v13_kpool_tail.py [VLLM_ROOT]
VLLM_ROOT defaults to the image's site-packages vllm directory.

Every edit is an exact-substring replace. An edit whose replacement text is
already present counts as applied (so reruns are no-ops); an edit whose anchor
does not occur exactly once refuses the whole run before anything is written.
No other v13 patch edits kpool_compress.py or attention.py. patch_v13_misc.py
and patch_v13_verify.py also edit indexer.py, at disjoint anchors, so each
pair applies in either order.
"""

import ast
import sys
from pathlib import Path

DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"

# --------------------------------------------------------------------------
# Kernels: models/glm5next/nvidia/ops/kpool_compress.py
# --------------------------------------------------------------------------
KPOOL_EDITS = [
    (
        "kpool_compress.py: import os",
        """from __future__ import annotations

import torch
""",
        """from __future__ import annotations

import os

import torch
""",
    ),
    (
        "kpool_compress.py: switch and ring size",
        """INDEX_HEAD_DIM = 128
""",
        """INDEX_HEAD_DIM = 128

# GLM53_KPOOL_TAIL_FIX (v13): off unless GLM53_KPOOL_TAIL_FIX=1.
GLM53_KPOOL_TAIL_FIX = os.environ.get("GLM53_KPOOL_TAIL_FIX") == "1"


def glm53_tail_ring_slots(kpool: int, num_speculative_tokens: int) -> int:
    \"\"\"Tail ring slots per request: kpool in v11, enlarged by the switch.

    A step writes up to 1 + k rows at consecutive positions P..P+k, each into
    slot pos % R, so row r overwrites the slot of position r - R. After the
    step the ring must still hold every committed position of the open pool,
    and the lowest of those can sit kpool - 2 below P (a = 0, P % kpool =
    kpool - 2). That needs R >= k + kpool - 1. R is the next power of two, so
    it divides the block size and the scheduler's LCM of the KV group block
    sizes stays the same. docker/README-v13.md (GLM53_KPOOL_TAIL_FIX) has the
    argument.
    \"\"\"
    if not GLM53_KPOOL_TAIL_FIX:
        return kpool
    return max(kpool, 1 << (num_speculative_tokens + kpool - 2).bit_length())
""",
    ),
    (
        "kpool_compress.py: seed kernel ring and strides",
        """    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    \"\"\"Copy token ``i``'s raw K + gate into its request's tail block.
""",
        """    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    RING: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    \"\"\"Copy token ``i``'s raw K + gate into its request's tail block.
""",
    ),
    (
        "kpool_compress.py: seed kernel addresses the ring",
        """    blk = t // KPOOL  # t >= 0 here, so trunc == floor
    ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
        tl.int64
    )
    # Match the torch semantics exactly: a negative ahead slot floors to a
    # block id that differs from every real block -> token is in the tail.
    # Only divide non-negative slots (Triton int div truncates, torch floors).
    if ahead >= 0 and ahead // KPOOL == blk:
        return
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, s, mask=m)
""",
        """    # GLM53_KPOOL_TAIL_FIX (v13): a tail block holds RING slots (v11: KPOOL),
    # TAIL_BLOCK_ELEMS apart, with the gate half KPOOL_HEAD after the K half.
    blk = t // RING  # t >= 0 here, so trunc == floor
    ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
        tl.int64
    )
    # Match the torch semantics exactly: a negative ahead slot floors to a
    # block id that differs from every real block -> token is in the tail.
    # Only divide non-negative slots (Triton int div truncates, torch floors).
    if ahead >= 0 and ahead // RING == blk:
        return
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    base = blk * TAIL_BLOCK_ELEMS + (t % RING) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(tail_ptr + base + KPOOL_HEAD + offs, s, mask=m)
""",
    ),
    (
        "kpool_compress.py: seed launch passes the ring layout",
        """    n = tslot.shape[0]
    if n == 0:
        return
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
""",
        """    n = tslot.shape[0]
    if n == 0:
        return
    if GLM53_KPOOL_TAIL_FIX:
        # GLM53_KPOOL_TAIL_FIX (v13): the tail view is carved out of the
        # indexer tensor at the indexer's page stride, so address its blocks
        # by the view's own strides.
        ring = tail_kv_cache.shape[2]
        block_elems, half_elems = tail_kv_cache.stride(0), tail_kv_cache.stride(1)
    else:
        # v11 addressing: a contiguous [num_blocks, 2, kpool, head_dim] view.
        ring, block_elems, half_elems = kpool, 2 * kpool * head_dim, kpool * head_dim
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        RING=ring,
        TAIL_BLOCK_ELEMS=block_elems,
        KPOOL_HEAD=half_elems,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
""",
    ),
    (
        "kpool_compress.py: decode kernel ring size",
        """    POOL_SIZE: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
""",
        """    POOL_SIZE: tl.constexpr,
    RING: tl.constexpr,  # tail ring slots per block (GLM53_KPOOL_TAIL_FIX)
    TAIL_BLOCK_ELEMS: tl.constexpr,
""",
    ),
    (
        "kpool_compress.py: decode stash slot",
        """        slot = safe_pos % POOL_SIZE
        phys_slot = safe_pos % POOL_SIZE
""",
        """        slot = safe_pos % POOL_SIZE
        # GLM53_KPOOL_TAIL_FIX (v13): the ring is addressed pos % RING (v11:
        # RING == POOL_SIZE), so a verify row never reaches a committed slot.
        phys_slot = safe_pos % RING
""",
    ),
    (
        "kpool_compress.py: decode tail block",
        """        block = tl.maximum(tail_slot, 0).to(tl.int64) // POOL_SIZE
""",
        """        block = tl.maximum(tail_slot, 0).to(tl.int64) // RING
""",
    ),
    (
        "kpool_compress.py: completion max pass reads pos % RING",
        """            max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % POOL_SIZE
""",
        """            max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % RING
""",
    ),
    (
        "kpool_compress.py: completion sum pass reads pos % RING",
        """            denom = tl.full((BLOCK_D,), 0.0, tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % POOL_SIZE
""",
        """            denom = tl.full((BLOCK_D,), 0.0, tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % RING
""",
    ),
    (
        "kpool_compress.py: decode launch checks and passes the ring size",
        """    assert tail_kv_cache.shape[2] == pool_size
""",
        """    assert tail_kv_cache.shape[2] >= pool_size  # ring slots (GLM53_KPOOL_TAIL_FIX)
""",
    ),
    (
        "kpool_compress.py: decode launch ring argument",
        """        POOL_SIZE=pool_size,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
""",
        """        POOL_SIZE=pool_size,
        RING=tail_kv_cache.shape[2],
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
""",
    ),
]

# --------------------------------------------------------------------------
# Tail spec: models/glm5next/nvidia/attention.py (Glm5NextTailCache)
# --------------------------------------------------------------------------
ATTENTION_EDITS = [
    (
        "attention.py: the tail block is the ring",
        """    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        # K + gate score packed into head_size (== 2*head_dim), head_size_v=0:
        # KpoolTailBackend.get_kv_cache_shape only consumes head_size and splits
        # it into [2, kpool, head_dim] (K | score halves), so the connectors'
        # non-MLA K/V half-split transfers K and score as separate halves.
        return KpoolTailSpec(
            block_size=self._index_kpool,
""",
        """    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        # GLM53_KPOOL_TAIL_FIX (v13): the tail block is the ring. It has
        # index_kpool slots unless the switch sizes it for the verify window.
        from vllm.models.glm5next.nvidia.ops.kpool_compress import (
            GLM53_KPOOL_TAIL_FIX,
            glm53_tail_ring_slots,
        )

        spec_cfg = vllm_config.speculative_config
        num_spec = spec_cfg.num_speculative_tokens if spec_cfg is not None else 0
        ring = glm53_tail_ring_slots(self._index_kpool, num_spec)
        if GLM53_KPOOL_TAIL_FIX:
            # The scheduler block size is the LCM of every KV group's block size.
            assert vllm_config.cache_config.block_size % ring == 0, (
                f"GLM53_KPOOL_TAIL_FIX: a {ring}-slot tail ring must divide "
                f"--block-size {vllm_config.cache_config.block_size}"
            )
            logger.info_once(
                "GLM53_KPOOL_TAIL_FIX: kpool tail ring %d slots (index_kpool=%d, "
                "k=%d)",
                ring,
                self._index_kpool,
                num_spec,
            )

        # K + gate score packed into head_size (== 2*head_dim), head_size_v=0:
        # KpoolTailBackend.get_kv_cache_shape only consumes head_size and splits
        # it into [2, kpool, head_dim] (K | score halves), so the connectors'
        # non-MLA K/V half-split transfers K and score as separate halves.
        return KpoolTailSpec(
            block_size=ring,
""",
    ),
]

# --------------------------------------------------------------------------
# Tail slots: v1/attention/backends/mla/indexer.py (KpoolTailMetadataBuilder)
# --------------------------------------------------------------------------
INDEXER_EDITS = [
    (
        "indexer.py: tail slot mapping takes seq_lens and in_place",
        """    num_reqs: int,
    kpool: int,
) -> torch.Tensor:
    \"\"\"Circular tail slots: every token of request r lands in r's own block.
""",
        """    num_reqs: int,
    kpool: int,
    seq_lens: torch.Tensor | None = None,
    in_place: bool = False,
) -> torch.Tensor:
    \"\"\"Circular tail slots: every token of request r lands in r's own block.
""",
    ),
    (
        "indexer.py: tail slots in place",
        """    Pure torch (no Triton, no device sync): the indexer op consumes the tail
    slot mapping on its eager break, so the returned tensor need not be the
    persistent ``BlockTables`` buffer.
    \"\"\"
    out = slot_mapping.clone()
""",
        """    Pure torch (no Triton, no device sync). v11 returns a clone, which only a
    PIECEWISE graph tolerates: the indexer op reads the tail slots from the
    forward context on its eager break. GLM53_KPOOL_TAIL_FIX (v13) passes
    ``in_place``: a FULL graph (the uniform verify batch) records the indexer
    op and replays the address it captured, so the slots go into
    ``slot_mapping`` itself, the tail group's persistent slot-mapping row.
    Only the switch passes ``positions`` None (the V2 runner builds none);
    token t of request r then sits at ``seq_lens[r] - (query_start_loc[r + 1]
    - t)``.
    \"\"\"
    out = slot_mapping if in_place else slot_mapping.clone()
""",
    ),
    (
        "indexer.py: positions from seq_lens",
        """    pos = positions[:num_actual_tokens].to(torch.int64)
""",
        """    if positions is None:
        # GLM53_KPOOL_TAIL_FIX (v13): no positions (V2 runner), see docstring.
        last = seq_lens[:num_reqs] - query_start_loc[1 : num_reqs + 1]
        pos = tokens + last.index_select(0, req)
    else:
        pos = positions[:num_actual_tokens].to(torch.int64)
""",
    ),
    (
        "indexer.py: tail builder reads the switch",
        """        # slot_mapping, which is rebuilt per step from the group's block table.
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
""",
        """        # slot_mapping, which is rebuilt per step from the group's block table.
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # GLM53_KPOOL_TAIL_FIX (v13): read where the ring size is decided.
        from vllm.models.glm5next.nvidia.ops.kpool_compress import (
            GLM53_KPOOL_TAIL_FIX,
        )

        self.glm53_tail_fix = GLM53_KPOOL_TAIL_FIX
""",
    ),
    (
        "indexer.py: tail builder maps V2 batches, in place",
        """        positions = common_attn_metadata.positions
        if positions is not None:
            # Circular per-request layout; the generic kernel output collapses
            # onto tail block 0 for pos >= kpool (see compute_... docstring).
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
            )
""",
        """        positions = common_attn_metadata.positions
        num_tokens = common_attn_metadata.num_actual_tokens
        if self.glm53_tail_fix:
            # GLM53_KPOOL_TAIL_FIX (v13): map without positions too. The V2
            # runner builds none, and its generic slots put every token at pos
            # >= block_size into block 0, one ring for all requests. Map the
            # real tokens only: CUDA-graph padding past query_start_loc[num_reqs]
            # keeps the slot kernel's -1, so it stashes nothing.
            num_tokens = int(
                common_attn_metadata.query_start_loc_cpu[common_attn_metadata.num_reqs]
            )
        if positions is not None or self.glm53_tail_fix:
            # Circular per-request layout; the generic kernel output collapses
            # onto tail block 0 for pos >= kpool (see compute_... docstring).
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                num_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
                seq_lens=common_attn_metadata.seq_lens,
                in_place=self.glm53_tail_fix,
            )
""",
    ),
]

FILE_EDITS = {
    "models/glm5next/nvidia/ops/kpool_compress.py": KPOOL_EDITS,
    "models/glm5next/nvidia/attention.py": ATTENTION_EDITS,
    "v1/attention/backends/mla/indexer.py": INDEXER_EDITS,
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
    print(f"v13 kpool tail patch applied ({len(staged)} files written)")
    return list(staged)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
