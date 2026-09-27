"""v13 adaptive-verification patch for the GLM-5.3-Flash serve image (on top of v11).

  GLM53_ADAPTIVE_VERIFY=1          verify only the drafter's confident prefix of
                                   each request, at fixed shapes (lossless)
  GLM53_ADAPTIVE_VERIFY_TAU=<f>    survival threshold in [0, 1] (default 0.1;
                                   0 turns the confidence rule off)
  GLM53_ADAPTIVE_VERIFY_MAX=<n>    fixed per-request verify cap in [1, k]
                                   (default k)

Off unless GLM53_ADAPTIVE_VERIFY=1: no hook is bound, nothing is allocated and
the served graphs are the v11 graphs. Only the V2 model runner creates the
state, i.e. SPEC=dflash2. docker/README-v13.md has the lossless argument, the
expected gain, the risks and the GPU validation.

Usage: python3 patch_v13_verify.py [VLLM_ROOT]
VLLM_ROOT defaults to the image's site-packages vllm directory.

Every edit is an exact-substring replace. An edit whose replacement text is
already present counts as applied (so reruns are no-ops); an edit whose anchor
does not occur exactly once refuses the whole run before anything is written.
The anchors are independent of patch_v13_census.py, which also edits the V2
model runner, and of patch_v13_misc.py, which also edits the MLA indexer
backend, so each pair applies in either order.
"""

import ast
import sys
from pathlib import Path

DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"

# --------------------------------------------------------------------------
# New module: v1/worker/gpu/spec_decode/glm53_adaptive_verify.py
# --------------------------------------------------------------------------
MODULE = '''"""GLM53_ADAPTIVE_VERIFY (v13): verify the drafter's confident prefix only.

Off unless GLM53_ADAPTIVE_VERIFY=1. Per request and step the verify width m is
the number of leading draft positions whose running product of DFlash2 top
probabilities (softmax over the selector candidates) is at least
GLM53_ADAPTIVE_VERIFY_TAU (default 0.1, 0 = off), clamped to
[1, GLM53_ADAPTIVE_VERIFY_MAX] (default k).

Shapes never change: a request still runs its 1 + n verify rows (n <= k
drafts). Row j (j = 0 the anchor, j = i the draft d_i) is masked when j > m.
One predicate, local_pos > verify_len[req_state], drives all three parts:

  rejection sampler  a masked row's draft id becomes -1, which v11's kernel
                     always rejects (greedy stores the argmax of the row
                     before it; sampled stops and draws from that row's
                     target logits). The step emits what k' = m speculation
                     emits.
  routed MoE         a masked row takes its anchor row's top-k ids with
                     weight 0, so it reads no expert the live rows do not.
  kpool tail ring    a masked row's tail slot becomes -1, so it does not
                     stash into the pos % index_kpool ring, where it would
                     overwrite the slot of a committed position.

The sampler reads rows 0..m only, and those rows depend on inputs 0..m only.
docker/README-v13.md has the full argument.

CUDA graphs: prepare() writes the persistent masked/anchor buffers before the
forward and before the attention metadata build, outside any graph; remap()
runs inside the graphs on fixed shapes; the tail-slot mask runs in the eager
metadata build; record() runs after the draft. No step syncs the host.
"""

import importlib
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _env_number(name, default, cast, lo, hi):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        value = None
    if value is None or not lo <= value <= hi:
        raise ValueError(f"{name}={raw!r}: want a {cast.__name__} in [{lo}, {hi}]")
    return value


def verify_width(scores: torch.Tensor, tau: float, cap: int) -> torch.Tensor:
    """[R, k, K] selector scores -> [R] verify width in [1, cap]."""
    top = torch.softmax(scores, dim=-1).amax(dim=-1)
    survival = torch.cumprod(top, dim=-1)
    return (survival >= tau).sum(dim=-1).clamp(1, cap)


def masked_rows(
    local_pos: torch.Tensor, req_state_idx: torch.Tensor, verify_len: torch.Tensor
) -> torch.Tensor:
    """Logit rows (= verify input rows) past their request's verify width."""
    return local_pos > verify_len[req_state_idx]


class AdaptiveVerify:
    def __init__(self, tau, cap, scores, max_num_reqs, max_num_tokens, device):
        self.tau = tau
        self.cap = cap
        self.scores = scores  # DFlash2 _selector_scores, or None when tau == 0
        self.verify_len = torch.full(
            (max_num_reqs,), cap, dtype=torch.int32, device=device
        )
        self.masked = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        # anchor[t] <= t holds for every t, so remap never gathers past the batch.
        self.anchor = torch.arange(max_num_tokens, dtype=torch.int64, device=device)

    def prepare(self, input_batch) -> None:
        """Before the target forward: mark this step's masked input rows."""
        masked = self.masked[: input_batch.num_tokens_after_padding]
        masked.zero_()
        if input_batch.num_draft_tokens == 0:
            return
        rows = input_batch.logits_indices
        local_pos = input_batch.expanded_local_pos
        masked[rows] = masked_rows(
            local_pos, input_batch.expanded_idx_mapping, self.verify_len
        )
        self.anchor[rows] = rows - local_pos

    def remap(self, topk_weights, topk_ids):
        """Router hook, inside the graphs: masked rows take the anchor's experts."""
        n = topk_ids.shape[0]
        masked = self.masked[:n].unsqueeze(1)
        return (
            topk_weights.masked_fill(masked, 0.0),
            torch.where(masked, topk_ids[self.anchor[:n]], topk_ids),
        )

    def mask_drafts(self, draft_sampled, req_state_idx, local_pos):
        """Rejection-sampler hook: a masked row's draft id becomes -1."""
        masked = masked_rows(local_pos, req_state_idx, self.verify_len)
        return torch.where(masked, -1, draft_sampled)

    def record(self, idx_mapping) -> None:
        """After the draft: verify widths for the drafts just proposed."""
        if self.tau > 0:
            width = verify_width(
                self.scores[: idx_mapping.shape[0]], self.tau, self.cap
            )
            self.verify_len[idx_mapping] = width.to(torch.int32)


def maybe_create_adaptive_verify(runner) -> AdaptiveVerify | None:
    """Return the state with its hooks bound, or None when the switch is off."""
    if os.environ.get("GLM53_ADAPTIVE_VERIFY") != "1":
        return None
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    k = runner.num_speculative_steps
    par = runner.parallel_config
    if (
        k < 1
        or type(runner.rejection_sampler) is not RejectionSampler
        or runner.model_state.num_new_sampled_tokens_per_step != 1
    ):
        raise ValueError(
            "GLM53_ADAPTIVE_VERIFY: needs draft-model speculative decoding with "
            "the V2 RejectionSampler and one bonus token per step"
        )
    if (
        par.pipeline_parallel_size > 1
        or par.data_parallel_size > 1
        or par.prefill_context_parallel_size > 1
        or par.use_ubatching
        or par.use_sequence_parallel_moe
        or getattr(runner.speculative_config, "enable_adaptive_verification", False)
    ):
        raise ValueError(
            "GLM53_ADAPTIVE_VERIFY: router rows must be the batch rows; PP, DP, "
            "PCP, DBO, sequence-parallel MoE and vLLM adaptive verification are "
            "not supported"
        )
    tau = _env_number("GLM53_ADAPTIVE_VERIFY_TAU", 0.1, float, 0.0, 1.0)
    cap = _env_number("GLM53_ADAPTIVE_VERIFY_MAX", k, int, 1, k)
    scores = getattr(runner.speculator, "_selector_scores", None)
    if tau > 0 and scores is None:
        raise ValueError(
            "GLM53_ADAPTIVE_VERIFY_TAU needs the DFlash2 drafter's selector "
            "scores; with another drafter set TAU=0 and MAX < k"
        )
    if tau == 0 and cap == k:
        raise ValueError(
            "GLM53_ADAPTIVE_VERIFY=1 with TAU=0 and MAX=k truncates nothing"
        )
    layers = [m for m in runner.model.modules() if isinstance(m, MoERunner)]
    if not layers:
        raise ValueError("GLM53_ADAPTIVE_VERIFY: the target model has no MoE layers")
    for layer in layers:
        if layer._quant_method.is_monolithic or not isinstance(
            layer.router, BaseRouter
        ):
            raise ValueError(
                f"GLM53_ADAPTIVE_VERIFY: {layer.layer_name} does not route "
                "through BaseRouter"
            )
    state = AdaptiveVerify(
        tau, cap, scores, runner.max_num_reqs, runner.max_num_tokens, runner.device
    )
    for layer in layers:
        layer.router.glm53_remap = state.remap
    runner.rejection_sampler.glm53_mask_drafts = state.mask_drafts
    mla_indexer = importlib.import_module("vllm.v1.attention.backends.mla.indexer")
    mla_indexer.GLM53_TAIL_STASH_MASK = state.masked
    logger.info(
        "GLM53_ADAPTIVE_VERIFY: tau=%g max=%d (k=%d); masked verify rows reuse "
        "their anchor's experts in %d MoE layers",
        tau,
        cap,
        k,
        len(layers),
    )
    return state
'''

# --------------------------------------------------------------------------
# Router hook: model_executor/layers/fused_moe/router/base_router.py
# --------------------------------------------------------------------------
ROUTER_EDITS = [
    (
        "base_router.py: remap attribute",
        """        self.capture_fn: Callable[[torch.Tensor], None] | None = None
""",
        """        self.capture_fn: Callable[[torch.Tensor], None] | None = None
        # GLM53_ADAPTIVE_VERIFY (v13): bound by glm53_adaptive_verify, else None.
        self.glm53_remap: Callable | None = None
""",
    ),
    (
        "base_router.py: remap masked verify rows before capture and EPLB",
        """        # Capture logical ids before EPLB mapping.
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
""",
        """        # GLM53_ADAPTIVE_VERIFY (v13): masked verify rows take their anchor
        # row's experts with weight 0, so they add no expert weight reads.
        if self.glm53_remap is not None:
            topk_weights, topk_ids = self.glm53_remap(topk_weights, topk_ids)

        # Capture logical ids before EPLB mapping.
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
""",
    ),
]

# --------------------------------------------------------------------------
# Forced rejection: v1/worker/gpu/spec_decode/rejection_sampler.py (V2)
# --------------------------------------------------------------------------
SAMPLER_EDITS = [
    (
        "rejection_sampler.py: mask attribute",
        """        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
""",
        """        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        # GLM53_ADAPTIVE_VERIFY (v13): bound by glm53_adaptive_verify, else None.
        self.glm53_mask_drafts = None
""",
    ),
    (
        "rejection_sampler.py: -1 for masked drafts after sampling params",
        """            expanded_local_pos,
        )
        sampled, num_sampled = rejection_sample(
""",
        """            expanded_local_pos,
        )
        if self.glm53_mask_drafts is not None:
            # GLM53_ADAPTIVE_VERIFY (v13): drafts past the request's verify
            # width become the -1 placeholder, which the kernel rejects.
            draft_sampled = self.glm53_mask_drafts(
                draft_sampled, expanded_idx_mapping, expanded_local_pos
            )
        sampled, num_sampled = rejection_sample(
""",
    ),
]

# --------------------------------------------------------------------------
# Wiring: v1/worker/gpu/model_runner.py (V2 model runner)
# --------------------------------------------------------------------------
RUNNER_EDITS = [
    (
        "model_runner.py: create after the samplers",
        """        if self.is_pooling_model and self.is_last_pp_rank:
            self.pooling_runner = PoolingRunner(self.model, self.vllm_config)
""",
        """        if self.is_pooling_model and self.is_last_pp_rank:
            self.pooling_runner = PoolingRunner(self.model, self.vllm_config)

        # GLM53_ADAPTIVE_VERIFY (v13): None unless GLM53_ADAPTIVE_VERIFY=1.
        from vllm.v1.worker.gpu.spec_decode.glm53_adaptive_verify import (
            maybe_create_adaptive_verify,
        )

        self._glm53_av = maybe_create_adaptive_verify(self)

""",
    ),
    (
        "model_runner.py: masks before the target forward",
        """            input_batch = self.prepare_inputs(
                scheduler_output, batch_req_state, batch_desc
            )
            block_tables, slot_mappings = self.prepare_attn(input_batch)
""",
        """            input_batch = self.prepare_inputs(
                scheduler_output, batch_req_state, batch_desc
            )
            block_tables, slot_mappings = self.prepare_attn(input_batch)
            if self._glm53_av is not None:
                self._glm53_av.prepare(input_batch)
""",
    ),
    (
        "model_runner.py: verify widths after the draft",
        """            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
""",
        """            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            if self._glm53_av is not None:
                self._glm53_av.record(input_batch.idx_mapping)
""",
    ),
]

# --------------------------------------------------------------------------
# Tail-ring stash: v1/attention/backends/mla/indexer.py (kpool tail group)
# --------------------------------------------------------------------------
INDEXER_EDITS = [
    (
        "indexer.py: tail-stash mask attribute",
        """def compute_kpool_tail_slot_mapping(
    slot_mapping: torch.Tensor,
""",
        """# GLM53_ADAPTIVE_VERIFY (v13): masked-row buffer bound by
# glm53_adaptive_verify, else None.
GLM53_TAIL_STASH_MASK: torch.Tensor | None = None


def compute_kpool_tail_slot_mapping(
    slot_mapping: torch.Tensor,
""",
    ),
    (
        "indexer.py: masked verify rows skip the tail-ring stash",
        """    out[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)
    return out
""",
        """    out[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)
    if GLM53_TAIL_STASH_MASK is not None:
        # GLM53_ADAPTIVE_VERIFY (v13): a masked verify row gets tail slot -1,
        # so the kernel skips its stash into the pos % kpool ring, where it
        # would overwrite the slot of a committed position in the open pool.
        out[:num_actual_tokens].masked_fill_(
            GLM53_TAIL_STASH_MASK[:num_actual_tokens], -1
        )
    return out
""",
    ),
]

FILE_EDITS = {
    "model_executor/layers/fused_moe/router/base_router.py": ROUTER_EDITS,
    "v1/worker/gpu/spec_decode/rejection_sampler.py": SAMPLER_EDITS,
    "v1/worker/gpu/model_runner.py": RUNNER_EDITS,
    "v1/attention/backends/mla/indexer.py": INDEXER_EDITS,
}

NEW_FILES = {
    "v1/worker/gpu/spec_decode/glm53_adaptive_verify.py": MODULE,
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
    print(f"v13 verify patch applied ({len(staged)} files written)")
    return list(staged)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
