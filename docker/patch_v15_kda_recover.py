"""v15 KDA verify-without-state-stores patch for the GLM-5.3-Flash serve image.

  GLM53_KDA_RECOVER=1   the DFlash2 verify stores no KDA recurrent state; after
                        sampling, each row's accepted tokens are replayed from
                        the same initial state through the same kernel, which
                        stores only the states a later step reads (bit-exact)

Off unless GLM53_KDA_RECOVER=1: no buffer is allocated, the verify call keeps
v14's arguments and the commit hook returns at once. The switch needs the V2
model runner, pipeline parallel 1, speculative decoding, and mamba cache mode
none or align; the GDN metadata builder refuses at startup otherwise.
docker/README-v15-kda-recover.md has the argument, the risks and the checks.

Usage: python3 patch_v15_kda_recover.py [VLLM_ROOT]
VLLM_ROOT defaults to the image's site-packages vllm directory.

Every edit is an exact-substring replace. An edit whose replacement text is
already present counts as applied (so reruns are no-ops); an edit whose anchor
does not occur exactly once refuses the whole run before anything is written.
"""

import ast
import sys
from pathlib import Path

DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"

MODULE_REL = "models/glm5next/nvidia/ops/glm53_kda_recover.py"

# --------------------------------------------------------------------------
# New module: models/glm5next/nvidia/ops/glm53_kda_recover.py
# --------------------------------------------------------------------------
MODULE = '''"""GLM53_KDA_RECOVER (v15): KDA spec verify without per-token state stores.

Off unless GLM53_KDA_RECOVER=1.

The stock verify (fused_recurrent_kda with num_accepted_tokens) reads each
row's initial state from ssm_state_indices[row, num_accepted - 1] and stores
the state after every token t into ssm_state_indices[row, t]. At TP=2 that is
32 heads x 64 KiB fp32 per KDA layer and token. Later steps read two of those
columns: n - 1 (n = num_sampled; the next verify, and the align pre-copy) and,
in align mode, the column of the block boundary the step crossed (the align
post-copy). With the switch on:

  verify   the same kernel gets the index row [NULL x Q, src] with
           num_accepted = Q + 1. It reads the same initial state and stores
           nothing. The layer copies the token inputs (post-conv k and v, raw
           gate g, raw beta) into per-layer records.
  commit   after sampling, before the align post-copy, the same kernel runs
           each row's records from the same initial state with an index row
           that holds only column n - 1 and the boundary column.

Same kernel, same input values, same token order: every state it stores is
the state the stock verify stores in that column. Columns no later step reads
keep older contents. docker/README-v15-kda-recover.md has the argument.
"""

import os
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("GLM53_KDA_RECOVER") == "1"


@triton.jit
def _verify_index_kernel(
    state_indices,
    stride_row,
    num_accepted,
    out,
    Q: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Column Q is the element the stock kernel reads its initial state from,
    # ssm_state_indices + row * stride + num_accepted - 1. Columns 0..Q-1 are
    # NULL (the kernel skips ids <= 0), so the verify stores no state.
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    src = tl.load(state_indices + r * stride_row + tl.load(num_accepted + r) - 1)
    row = tl.where(cols == Q, src, 0)
    tl.store(out + r * (Q + 1) + cols, row, mask=cols < Q + 1)


@triton.jit
def _store_records_kernel(
    qkv,
    stride_qkv,
    g,
    stride_g,
    beta,
    stride_beta,
    rec_k,
    rec_v,
    rec_g,
    rec_beta,
    P: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    o = tl.arange(0, K)
    src = qkv + t * stride_qkv + h * K + o
    dst = (t * H + h) * K + o
    tl.store(rec_k + dst, tl.load(src + P))
    tl.store(rec_v + dst, tl.load(src + 2 * P))
    tl.store(rec_g + dst, tl.load(g + t * stride_g + h * K + o))
    tl.store(rec_beta + t * H + h, tl.load(beta + t * stride_beta + h))


@triton.jit
def _replay_index_kernel(
    state_indices,
    stride_row,
    verify_index,
    rows,
    num_sampled,
    idx_mapping,
    num_computed,
    out,
    Q: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ROWS: tl.constexpr,
    ALIGN_BLOCK: tl.constexpr,
):
    # Spec row r is batch row rows[r] (identity when every row verifies).
    # The boundary test is postprocess_mamba_fused_kernel's (V2,
    # PRECOMPUTED_NEW_COMPUTED): num_computed already holds the advanced count.
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    if HAS_ROWS:
        b = tl.load(rows + r)
    else:
        b = r
    req = tl.load(idx_mapping + b)
    live = req >= 0
    n = tl.maximum(tl.load(num_sampled + b), 1)
    keep = cols == n - 1
    if ALIGN_BLOCK > 0:
        new_computed = tl.load(num_computed + req, mask=live, other=0)
        running = new_computed - n + 1
        aligned = (new_computed // ALIGN_BLOCK) * ALIGN_BLOCK
        keep = keep | ((aligned >= running) & (cols == aligned - running))
    keep = keep & (cols < Q) & live
    row = tl.load(state_indices + r * stride_row + cols, mask=keep, other=0)
    src = tl.load(verify_index + r * (Q + 1) + Q)
    row = tl.where(cols == Q, tl.where(live, src, 0), row)
    tl.store(out + r * (Q + 1) + cols, row, mask=cols < Q + 1)


@dataclass
class Records:
    """One layer's verify inputs, token-major, as the stock kernel reads them."""

    k: torch.Tensor  # [tokens, H, K]
    v: torch.Tensor  # [tokens, H, K]
    g: torch.Tensor  # [tokens, H, K]
    beta: torch.Tensor  # [tokens, H]

    @classmethod
    def allocate(cls, tokens, heads, head_dim, dtype, device) -> "Records":
        def buf(*shape):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            k=buf(tokens, heads, head_dim),
            v=buf(tokens, heads, head_dim),
            g=buf(tokens, heads, head_dim),
            beta=buf(tokens, heads),
        )


def store_records(
    records: Records,
    qkv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    proj: int,
) -> None:
    """Copy the verify inputs: qkv [n, 3 * proj] post-conv, g [1, n, H, K]
    raw gate, beta [1, n, H] raw. Graph-safe: fixed shapes, no host sync."""
    n = qkv.shape[0]
    tokens, heads, head_dim = records.k.shape
    if (
        n > tokens
        or proj != heads * head_dim
        or qkv.stride(1) != 1
        or g.shape[1] != n
        or g.stride(-1) != 1
        or g.stride(-2) != head_dim
        or beta.shape[1] != n
        or beta.stride(-1) != 1
        or qkv.dtype != records.k.dtype
        or g.dtype != records.g.dtype
        or beta.dtype != records.beta.dtype
    ):
        raise RuntimeError("GLM53_KDA_RECOVER: verify inputs do not fit the records")
    if n == 0:
        return
    _store_records_kernel[(n, heads)](
        qkv,
        qkv.stride(0),
        g,
        g.stride(1),
        beta,
        beta.stride(1),
        records.k,
        records.v,
        records.g,
        records.beta,
        P=proj,
        H=heads,
        K=head_dim,
    )


class Recover:
    """One GDN metadata builder's (one KV-cache group's) recover state."""

    def __init__(self, layers, q, rows, tokens, align_block, dtype, device):
        self.layers = tuple(layers)
        self.q = q
        self.block = triton.next_power_of_2(q + 1)
        self.align_block = align_block
        self.verify_index = torch.zeros((rows, q + 1), dtype=torch.int32, device=device)
        self.replay_index = torch.zeros_like(self.verify_index)
        self.q_plus_1 = torch.full((rows,), q + 1, dtype=torch.int32, device=device)
        heads, head_dim = layers[0].local_num_heads, layers[0].head_dim
        for layer in self.layers:
            if getattr(layer, "glm53_rs_records", None) is None:
                layer.glm53_rs_records = Records.allocate(
                    tokens, layer.local_num_heads, layer.head_dim, dtype, device
                )
        self.out = torch.empty(
            (1, tokens, heads, head_dim), dtype=dtype, device=device
        )

    def prepare(self, state_indices, num_accepted, spec_mask_cpu) -> dict:
        """Builder side, outside any graph: the verify's index row and
        num_accepted. ``spec_mask_cpu`` is None when every row verifies."""
        n = state_indices.shape[0]
        if (
            state_indices.shape[1] != self.q
            or state_indices.stride(1) != 1
            or n > self.verify_index.shape[0]
            or num_accepted.shape[0] < n
        ):
            raise RuntimeError("GLM53_KDA_RECOVER: unexpected spec state indices")
        verify_index = self.verify_index[:n]
        if n:
            _verify_index_kernel[(n,)](
                state_indices,
                state_indices.stride(0),
                num_accepted,
                verify_index,
                Q=self.q,
                BLOCK=self.block,
            )
        rows = None
        if spec_mask_cpu is not None:
            from vllm.utils.torch_utils import async_tensor_h2d

            rows = async_tensor_h2d(
                spec_mask_cpu.nonzero(as_tuple=True)[0],
                device=state_indices.device,
                dtype=torch.int32,
            )
        return {
            "glm53_rs_verify_idx": verify_index,
            "glm53_rs_nacc": self.q_plus_1[:n],
            "glm53_rs_rows": rows,
            "glm53_rs": self,
        }

    def commit(self, md, idx_mapping, num_sampled, num_computed) -> None:
        """Replay this step's accepted tokens into the states later steps read.
        Same stream as the forward and the next step, no host sync."""
        from vllm.third_party.flash_linear_attention.ops.kda import (
            fused_recurrent_kda,
        )

        nsd = md.num_spec_decodes
        if nsd == 0:
            return
        if self.align_block and num_computed is None:
            raise RuntimeError("GLM53_KDA_RECOVER: align commit needs num_computed")
        state_indices = md.spec_state_indices_tensor
        rows = md.glm53_rs_rows
        _replay_index_kernel[(nsd,)](
            state_indices,
            state_indices.stride(0),
            md.glm53_rs_verify_idx,
            idx_mapping if rows is None else rows,
            num_sampled,
            idx_mapping,
            idx_mapping if num_computed is None else num_computed,
            self.replay_index,
            Q=self.q,
            BLOCK=self.block,
            HAS_ROWS=rows is not None,
            ALIGN_BLOCK=self.align_block,
        )
        cu_seqlens = md.spec_query_start_loc[: nsd + 1]
        replay_index = self.replay_index[:nsd]
        num_accepted = self.q_plus_1[:nsd]
        for layer in self.layers:
            rec = layer.glm53_rs_records
            k = rec.k.unsqueeze(0)
            # The keyword set is the verify call's in kda.py, so the stored
            # states come from the same kernel specialisation and arithmetic.
            fused_recurrent_kda(
                q=k,
                k=k,
                v=rec.v.unsqueeze(0),
                g=rec.g.unsqueeze(0),
                beta=rec.beta.unsqueeze(0),
                initial_state=layer.kv_cache[1],
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=replay_index,
                num_accepted_tokens=num_accepted,
                out=self.out,
                sigmoid_beta=True,
                a_log=layer.A_log,
                g_bias=layer.dt_bias,
                compute_gate=True,
                lower_bound=layer.kda_lower_bound,
            )


def make_builder_state(builder, layer_names, vllm_config, device):
    """GDNAttentionMetadataBuilder.__init__ hook: a Recover, or None when off."""
    if not ENABLED or builder.num_spec == 0:
        return None
    ctx = vllm_config.compilation_config.static_forward_context
    layers = [ctx[name] for name in layer_names]
    if not layers or not all(hasattr(x, "kda_lower_bound") for x in layers):
        raise RuntimeError("GLM53_KDA_RECOVER=1 covers GLM-5.3 KDA layers only")
    if not vllm_config.use_v2_model_runner:
        raise RuntimeError("GLM53_KDA_RECOVER=1 commits from the V2 model runner only")
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError("GLM53_KDA_RECOVER=1 needs pipeline_parallel_size=1")
    mode = vllm_config.cache_config.mamba_cache_mode
    if mode not in ("none", "align"):
        raise RuntimeError(f"GLM53_KDA_RECOVER=1 supports mamba cache modes none and align, not {mode}")
    for layer in layers:
        k = layer.head_dim
        if not layer.kda_safe_gate or k & (k - 1):
            raise RuntimeError("GLM53_KDA_RECOVER=1 needs the bounded gate and a power-of-two head dim")
    q = builder.num_spec + 1
    max_seqs = vllm_config.scheduler_config.max_num_seqs
    align_block = builder.kv_cache_spec.block_size if mode == "align" else 0
    if align_block and align_block < q:
        raise RuntimeError("GLM53_KDA_RECOVER=1 needs a mamba block of at least one verify window")
    return Recover(
        layers,
        q,
        rows=max(builder.decode_cudagraph_max_bs, max_seqs),
        tokens=max(builder.decode_cudagraph_max_bs, max_seqs * q),
        align_block=align_block,
        dtype=vllm_config.model_config.dtype,
        device=device,
    )


def commit_step(attn_metadata, idx_mapping, num_sampled, num_computed) -> None:
    """MambaHybridModelState.postprocess_state hook: one commit per GDN group."""
    if not isinstance(attn_metadata, dict):
        raise RuntimeError("GLM53_KDA_RECOVER: expected one attention metadata dict per step")
    seen = set()
    for md in attn_metadata.values():
        rs = getattr(md, "glm53_rs", None)
        if rs is None or id(md) in seen or md.num_spec_decodes == 0:
            continue
        seen.add(id(md))
        if isinstance(num_sampled, int):
            raise RuntimeError("GLM53_KDA_RECOVER: a verify step finished without sampling")
        rs.commit(md, idx_mapping, num_sampled, num_computed)
'''

# --------------------------------------------------------------------------
# models/glm5next/nvidia/kda.py: the verify call
#
# With the switch on, the builder sets glm53_rs_verify_idx (index rows
# [NULL x Q, src]) and glm53_rs_nacc (all Q + 1). The conv update keeps the
# real num_accepted_tokens; only the recurrent call changes its two index
# arguments. With the switch off both fields are None and the call is v14's.
# --------------------------------------------------------------------------
KDA_EDITS = [
    (
        "kda.py: import",
        "from vllm.model_executor.utils import (\n",
        "from vllm.models.glm5next.nvidia.ops import glm53_kda_recover\n"
        "from vllm.model_executor.utils import (\n",
    ),
    (
        "kda.py: verify without state stores",
        """            core_attn_out_spec, _ = fused_recurrent_kda(
                q=_rearr(q_spec),
                k=_rearr(k_spec),
                v=_rearr(v_spec),
                g=g1_spec,
                beta=beta_spec,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
""",
        """            # GLM53_KDA_RECOVER (v15): store no state here; record the inputs
            # and let postprocess_state replay the accepted tokens.
            rs_index = attn_metadata_narrowed.glm53_rs_verify_idx
            if rs_index is not None:
                glm53_kda_recover.store_records(
                    self.glm53_rs_records,
                    qkv_spec,
                    g1_spec,
                    beta_spec,
                    self.local_projection_size,
                )
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=_rearr(q_spec),
                k=_rearr(k_spec),
                v=_rearr(v_spec),
                g=g1_spec,
                beta=beta_spec,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
                ssm_state_indices=(
                    spec_state_indices_tensor if rs_index is None else rs_index
                ),
                num_accepted_tokens=(
                    num_accepted_tokens
                    if rs_index is None
                    else attn_metadata_narrowed.glm53_rs_nacc
                ),
""",
    ),
]

# --------------------------------------------------------------------------
# v1/attention/backends/gdn_attn.py: metadata fields and the builder hook
# --------------------------------------------------------------------------
GDN_EDITS = [
    (
        "gdn_attn.py: switch",
        "from vllm.v1.kv_cache_interface import MambaSpec\n",
        "from vllm.v1.kv_cache_interface import MambaSpec\n\n"
        "_GLM53_KDA_RECOVER = os.environ.get(\"GLM53_KDA_RECOVER\") == \"1\"\n",
    ),
    (
        "gdn_attn.py: import os",
        '"""Backend for GatedDeltaNet attention."""\n\nfrom dataclasses import dataclass\n',
        '"""Backend for GatedDeltaNet attention."""\n\nimport os\nfrom dataclasses import dataclass\n',
    ),
    (
        "gdn_attn.py: metadata fields",
        "    token_chunk_offset_ptr: torch.Tensor | None = None\n",
        """    token_chunk_offset_ptr: torch.Tensor | None = None

    # GLM53_KDA_RECOVER (v15): verify index rows [NULL x Q, src], their
    # num_accepted (Q + 1), spec row -> batch row (None: identity) and the
    # group's recover state. All None when the switch is off.
    glm53_rs_verify_idx: torch.Tensor | None = None
    glm53_rs_nacc: torch.Tensor | None = None
    glm53_rs_rows: torch.Tensor | None = None
    glm53_rs: object | None = None
""",
    ),
    (
        "gdn_attn.py: builder state",
        """        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
""",
        """        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.glm53_rs = None
        if _GLM53_KDA_RECOVER:
            from vllm.models.glm5next.nvidia.ops.glm53_kda_recover import (
                make_builder_state,
            )

            self.glm53_rs = make_builder_state(self, layer_names, vllm_config, device)
""",
    ),
    (
        "gdn_attn.py: build",
        """        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
""",
        """        glm53_rs_fields = {}
        if self.glm53_rs is not None and num_spec_decodes > 0:
            glm53_rs_fields = self.glm53_rs.prepare(
                spec_state_indices_tensor,
                num_accepted_tokens,
                None
                if num_prefills == 0 and num_decodes == 0
                else spec_sequence_masks_cpu,
            )
        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
""",
    ),
    (
        "gdn_attn.py: build fields",
        """            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata
""",
        """            token_chunk_offset_ptr=token_chunk_offset_ptr,
            **glm53_rs_fields,
        )
        return attn_metadata
""",
    ),
]

# --------------------------------------------------------------------------
# v1/worker/gpu/model_states/mamba_hybrid.py: keep the step's metadata, commit
# after sampling and before the align post-copy reads the states.
#
# patch_v15_fixes.py also edits this file. The "import os" edit is identical
# in both, and the switch is read in __init__ rather than next to their
# module-level flag, so the two patches apply and rerun in either order.
# --------------------------------------------------------------------------
MAMBA_HYBRID_EDITS = [
    (
        "mamba_hybrid.py: import os",
        "from dataclasses import dataclass\nfrom typing import Any\n",
        "import os\nfrom dataclasses import dataclass\nfrom typing import Any\n",
    ),
    (
        "mamba_hybrid.py: step slot",
        """        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
""",
        """        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # GLM53_KDA_RECOVER (v15): this step's attention metadata, committed
        # by postprocess_state.
        self._glm53_kda_recover = os.environ.get("GLM53_KDA_RECOVER") == "1"
        self._glm53_rs_step: dict[str, Any] | None = None
""",
    ),
    (
        "mamba_hybrid.py: keep step metadata",
        """        return build_attn_metadata(
            attn_groups=attn_groups,
""",
        """        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
""",
    ),
    (
        "mamba_hybrid.py: keep step metadata tail",
        """            rswa_prefix_lens=input_batch.prompt_lens,
        )

    def postprocess_state(
""",
        """            rswa_prefix_lens=input_batch.prompt_lens,
        )
        if self._glm53_kda_recover:
            self._glm53_rs_step = None if for_capture else attn_metadata
        return attn_metadata

    def postprocess_state(
""",
    ),
    (
        "mamba_hybrid.py: commit",
        """        num_reqs = idx_mapping.shape[0]
        if not num_reqs:
            return

""",
        """        num_reqs = idx_mapping.shape[0]
        if not num_reqs:
            return

        # GLM53_KDA_RECOVER (v15): fold the accepted tokens into the KDA
        # states before the align post-copy below reads them.
        step, self._glm53_rs_step = self._glm53_rs_step, None
        if step is not None:
            from vllm.models.glm5next.nvidia.ops.glm53_kda_recover import (
                commit_step,
            )

            commit_step(step, idx_mapping, num_sampled, num_computed_tokens)

""",
    ),
]

FILE_EDITS = {
    "models/glm5next/nvidia/kda.py": KDA_EDITS,
    "v1/attention/backends/gdn_attn.py": GDN_EDITS,
    "v1/worker/gpu/model_states/mamba_hybrid.py": MAMBA_HYBRID_EDITS,
}

NEW_FILES = {MODULE_REL: MODULE}


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
        if not path.parent.is_dir():
            raise SystemExit(f"refusing: {path.parent} missing")
        ast.parse(content, filename=str(path))
        print(f"{rel}: new file")
        staged[path] = content
    for path, content in staged.items():
        path.write_text(content)
    print(f"v15 kda recover patch applied ({len(staged)} files written)")
    return list(staged)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
