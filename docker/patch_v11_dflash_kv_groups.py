"""DFlash draft KV group for the GLM-5-Next bespoke KV cache layout.

The GLM-5-Next grouping (`_get_kv_cache_groups_glm5_next`) only admits
MLA/mamba/kpool specs, so a DFlash draft's SlidingWindowSpec layers make it
bail to the generic path, which cannot unify the indexer's 33 B/token page
with the draft's page and dies with NotImplementedError.

This patch carves the draft's uniform sliding-window layers into their own KV
group with dedicated per-layer tensors:

- The draft group adopts the attention group's block size. Every block id in
  the shared pool is charged the full per-block byte sum, so the draft's
  default 16-token blocks would burn sliding_window/16 ids per request and
  exhaust the pool; at the attention block size a request needs 1-2 ids.
- `_glm5_next_tensor_layout` learns to classify the draft group and returns
  it to all three consumers (pool bytes per block, tensor emission, max
  memory estimation), which add len(draft) * draft_page to the per-block sum.
"""

from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py")
s = p.read_text()


def swap(old: str, new: str, what: str) -> None:
    global s
    if s.count(old) != 1:
        raise SystemExit(f"unexpected source for {what}; refusing to patch")
    s = s.replace(old, new)


# 1. Builder: split draft sliding-window specs out of the attention specs.
swap(
    """    attn_specs = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, (MambaSpec, KpoolTailSpec))
    }
    if not mamba_specs or not all(
        type(s) is MLAAttentionSpec for s in attn_specs.values()
    ):
        return None""",
    """    attn_specs = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, (MambaSpec, KpoolTailSpec))
    }
    # DFlash draft layers (uniform sliding-window GQA). The target itself has
    # no SlidingWindowSpec layers, so this split is unambiguous.
    draft_specs = {
        k: v for k, v in attn_specs.items() if isinstance(v, SlidingWindowSpec)
    }
    attn_specs = {k: v for k, v in attn_specs.items() if k not in draft_specs}
    if not mamba_specs or not all(
        type(s) is MLAAttentionSpec for s in attn_specs.values()
    ):
        return None""",
    "glm5 group builder spec split",
)

# 2. Builder: construct the draft group and append it to the returned groups.
swap(
    """    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
    )""",
    """    draft_group = None
    if draft_specs:
        # Every pool block id costs the full per-block byte sum, so the
        # draft's block size trades tensor bytes against block-id burn: tiny
        # blocks (default 16) need sliding_window/16 ids per request and
        # exhaust the pool, while a full attention-sized block puts a page of
        # draft KV behind every target block id. A quarter block keeps a
        # 2048-token window at ~3 ids per request with a quarter of the
        # per-block tensor tax.
        attn_block = next(iter(mla_specs.values())).block_size
        draft_block = max(attn_block // 4, 16)
        resized_draft_specs: dict[str, KVCacheSpec] = {
            name: replace(spec, block_size=draft_block)
            for name, spec in draft_specs.items()
        }
        draft_uniform = UniformTypeKVCacheSpecs.from_specs(resized_draft_specs)
        assert draft_uniform is not None
        draft_group = KVCacheGroupSpec(list(resized_draft_specs), draft_uniform)

    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + ([draft_group] if draft_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
    )""",
    "glm5 group builder return",
)

# 3. Detector signature: two more fields for the draft group.
swap(
    """) -> (
    tuple[
        KVCacheGroupSpec,
        list[KVCacheGroupSpec],
        list[str],
        list[str],
        int,
        int,
        list[str],
        int,
    ]
    | None
):""",
    """) -> (
    tuple[
        KVCacheGroupSpec,
        list[KVCacheGroupSpec],
        list[str],
        list[str],
        int,
        int,
        list[str],
        int,
        list[str],
        int,
    ]
    | None
):""",
    "glm5 layout detector signature",
)

# 4. Detector: classify the draft group.
swap(
    """    for g in uniform_groups:
        group_inner = cast(UniformTypeKVCacheSpecs, g.kv_cache_spec).kv_cache_specs
        if all(type(s) is MLAAttentionSpec for s in group_inner.values()):
            attn_group = g
        elif all(isinstance(s, KpoolTailSpec) for s in group_inner.values()):
            tail_group = g""",
    """    draft_group: KVCacheGroupSpec | None = None
    for g in uniform_groups:
        group_inner = cast(UniformTypeKVCacheSpecs, g.kv_cache_spec).kv_cache_specs
        if all(type(s) is MLAAttentionSpec for s in group_inner.values()):
            attn_group = g
        elif all(isinstance(s, KpoolTailSpec) for s in group_inner.values()):
            tail_group = g
        elif all(isinstance(s, SlidingWindowSpec) for s in group_inner.values()):
            # DFlash draft group (uniform sliding-window GQA layers).
            draft_group = g""",
    "glm5 layout detector classification",
)

# 5. Detector: return the draft fields.
swap(
    """    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_pages.pop(),
        tail_names,
        tail_page,
    )""",
    """    draft_names: list[str] = []
    draft_page = 0
    if draft_group is not None:
        draft_names = list(draft_group.layer_names)
        draft_pages = {
            spec.page_size_bytes
            for spec in cast(
                UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
            ).kv_cache_specs.values()
        }
        if len(draft_pages) != 1:
            return None
        draft_page = draft_pages.pop()

    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_pages.pop(),
        tail_names,
        tail_page,
        draft_names,
        draft_page,
    )""",
    "glm5 layout detector return",
)

# 6. Pool bytes per block: charge draft tensors on every block id.
swap(
    """        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5
        return len(mla_names) * mla_page + len(idx_names) * idx_page""",
    """        _, _, mla_names, idx_names, mla_page, idx_page, _, _, dr_names, dr_page = (
            glm5
        )
        return (
            len(mla_names) * mla_page
            + len(idx_names) * idx_page
            + len(dr_names) * dr_page
        )""",
    "pool bytes per block",
)

# 7. Tensor emission: size the pool with draft pages and emit draft tensors.
swap(
    """        (
            _,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
        ) = glm5n""",
    """        (
            _,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
            draft_names,
            draft_page,
        ) = glm5n""",
    "tensor emission unpack",
)

swap(
    """        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        num_blocks = available_memory // per_block""",
    """        per_block = (
            len(mla_names) * mla_page
            + len(idx_names) * idx_page
            + len(draft_names) * draft_page
        )
        num_blocks = available_memory // per_block""",
    "tensor emission per-block",
)

swap(
    """            KVCacheTensor(
                size=idx_page * num_blocks,
                shared_by=(
                    [idx_names[i], tail_names[i]] if tail_names else [idx_names[i]]
                ),
            )
            for i in range(len(idx_names))
        ]""",
    """            KVCacheTensor(
                size=idx_page * num_blocks,
                shared_by=(
                    [idx_names[i], tail_names[i]] if tail_names else [idx_names[i]]
                ),
            )
            for i in range(len(idx_names))
        ] + [
            KVCacheTensor(size=draft_page * num_blocks, shared_by=[draft_name])
            for draft_name in draft_names
        ]""",
    "tensor emission draft tensors",
)

# 8. Max-memory estimator: account for draft block-id demand and pages.
swap(
    """        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
        ) = glm5n
        uniform_spec = attn_group.kv_cache_spec
        assert isinstance(uniform_spec, UniformTypeKVCacheSpecs)
        blocks_needed = uniform_spec.max_memory_usage_pages(vllm_config)""",
    """        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
            draft_names,
            draft_page,
        ) = glm5n
        uniform_spec = attn_group.kv_cache_spec
        assert isinstance(uniform_spec, UniformTypeKVCacheSpecs)
        blocks_needed = uniform_spec.max_memory_usage_pages(vllm_config)""",
    "estimator unpack",
)

swap(
    """        return blocks_needed * (len(mla_names) * mla_page + len(idx_names) * idx_page)""",
    """        if draft_names:
            # Sliding-window draft: window/block_size + 1 ids per request.
            draft_uniform = None
            for g in kv_cache_groups:
                if list(g.layer_names) == draft_names:
                    draft_uniform = g.kv_cache_spec
                    break
            assert isinstance(draft_uniform, UniformTypeKVCacheSpecs)
            blocks_needed += draft_uniform.max_memory_usage_pages(vllm_config)
        return blocks_needed * (
            len(mla_names) * mla_page
            + len(idx_names) * idx_page
            + len(draft_names) * draft_page
        )""",
    "estimator return",
)

p.write_text(s)
print("glm5next DFlash draft KV group applied")
