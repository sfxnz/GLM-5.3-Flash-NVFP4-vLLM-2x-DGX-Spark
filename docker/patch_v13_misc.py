"""v13 misc runtime patches for the GLM-5.3-Flash serve image (on top of v11).

Every behaviour is opt-in through a GLM53_* environment variable and is off
by default. With none of them set, the patched image behaves like v11. See
docker/README-v13.md for each switch, its expected effect and how to validate
it on the Sparks.

  GLM53_ROUTER_FP32=1              MoE router logits via cuBLAS bf16xbf16->fp32
  GLM53_INDEXER_WS_FACTOR=<int>    DSA indexer prefill workspace factor (stock 40)
  GLM53_MHC_WARMUP=1               pre-compile Glm5Next mHC TileLang variants
  GLM53_KDA_TRIM=1                 skip 4 .contiguous() copies per KDA layer
  GLM53_SKIP_MTP_WEIGHTS=1         do not read MTP layer tensors when spec != mtp
  GLM53_DFLASH_PREFIX_CACHE_FIX=1  EAGLE block drop on the DFlash draft group only

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

# --------------------------------------------------------------------------
# GLM53_MHC_WARMUP: model_executor/warmup/{kernel_warmup,glm5next_mhc_warmup}.py
# --------------------------------------------------------------------------
KERNEL_WARMUP_EDITS = [
    (
        "kernel_warmup.py: Glm5Next mHC warmup hook",
        """    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        ),
    )
""",
        """    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        ),
    )

    # GLM53_MHC_WARMUP (v13): the DSv4 warmup above returns for Glm5Next.
    # No-op unless GLM53_MHC_WARMUP=1 (checked inside).
    from vllm.model_executor.warmup.glm5next_mhc_warmup import glm5next_mhc_warmup

    glm5next_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
    )
""",
    ),
]

GLM5NEXT_MHC_WARMUP = '''# SPDX-License-Identifier: Apache-2.0
"""Pre-compile Glm5Next mHC TileLang kernels before serving (GLM53_MHC_WARMUP=1).

v13 image layer (GLM-5.3-Flash 2x DGX Spark recipe). deepseek_v4_mhc_warmup
returns for every model_type other than deepseek_v4, so on GLM-5.3 the first
prefill whose token count lands in a new mHC kernel specialisation compiles
it (~5 s on each TP rank, inside the forward). The specialisation depends on
the token count only through the dispatch in kernels/mhc/tilelang.py:

- mhc_pre_tilelang (layer 0) and mhc_fused_post_pre_tilelang with more than
  16 tokens pick n_splits = compute_num_split(64, hc_mult * hidden,
  cdiv(num_tokens, 64)) when DeepGEMM is supported, else 1; n_splits is a
  static TileLang argument of mhc_pre_big_fuse[_with_norm]_tilelang.
- mhc_fused_post_pre_tilelang with 16 tokens or fewer runs mhc_fused_tilelang
  with (tile_n, n_splits) = (2, 8) below 8 tokens, else (3, 4).
- the non-DeepGEMM prenorm GEMM also branches at 128 and 1024 tokens.

This warmup runs the three mHC ops of one Glm5Next decoder layer once per
distinct specialisation up to max_num_batched_tokens, so all compiles happen
at boot, before the API is ready.
"""

import os
import time
from collections.abc import Callable

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_BLOCK = 64  # DeepGEMM block_m / block_k used by the mHC dispatch
_SMALL_FMA_MAX_TOKENS = 16


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def select_mhc_token_sizes(
    max_tokens: int,
    num_split_for_grid: Callable[[int], int],
    use_deep_gemm: bool,
) -> list[int]:
    """Smallest token count for each mHC kernel specialisation reachable at
    1..max_tokens tokens. ``num_split_for_grid(g)`` returns the n_splits the
    dispatch picks for cdiv(num_tokens, 64) == g."""
    candidates = {1, 7, 8, 16, 17, 127, 128, 1023, 1024, max_tokens}
    candidates.update(_BLOCK * g for g in range(1, _cdiv(max_tokens, _BLOCK) + 1))
    chosen: dict[tuple, int] = {}
    for t in sorted(c for c in candidates if 1 <= c <= max_tokens):
        big = (
            num_split_for_grid(_cdiv(t, _BLOCK))
            if use_deep_gemm
            else (1, t < 128, t >= 1024)
        )
        small = (t < 8) if t <= _SMALL_FMA_MAX_TOKENS else None
        chosen.setdefault((small, big), t)
    return sorted(chosen.values())


def _find_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if (
            module.__class__.__name__ == "Glm5NextDecoderLayer"
            and getattr(module, "mhc", False)
            and not getattr(module, "is_mtp_layer", False)
            and hasattr(module, "mhc_fused_post_pre_op")
        ):
            return module
    return None


def glm5next_mhc_warmup(model: torch.nn.Module, *, max_tokens: int) -> None:
    if os.environ.get("GLM53_MHC_WARMUP") != "1":
        return
    layer = _find_mhc_layer(model)
    if layer is None or max_tokens <= 0:
        logger.info("GLM53_MHC_WARMUP: no Glm5Next mHC layer found; skipped")
        return
    device = layer.hc_attn_fn.device
    if device.type != "cuda":
        return

    from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    n, hidden = int(layer.n), int(layer.hidden_size)
    sizes = select_mhc_token_sizes(
        max_tokens,
        lambda grid: compute_num_split(_BLOCK, n * hidden, grid),
        is_deep_gemm_supported(),
    )
    logger.info(
        "GLM53_MHC_WARMUP: compiling Glm5Next mHC TileLang variants for token "
        "sizes %s",
        sizes,
    )
    started = time.perf_counter()
    residual = torch.zeros(max(sizes), n, hidden, dtype=torch.bfloat16, device=device)
    attn_out = torch.zeros(max(sizes), hidden, dtype=torch.bfloat16, device=device)
    in_norm, post_norm = layer.input_layernorm, layer.post_attention_layernorm
    with torch.inference_mode():
        for t in sizes:
            # Same three ops, arguments and order as Glm5NextDecoderLayer.forward.
            post, comb, _ = layer.hc_pre(
                residual[:t],
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                norm_weight=in_norm.weight.data,
                norm_eps=in_norm.variance_epsilon,
            )
            res, post, comb, x = layer.hc_fused_post_pre(
                attn_out[:t],
                residual[:t],
                post,
                comb,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                norm_weight=post_norm.weight.data,
                norm_eps=post_norm.variance_epsilon,
            )
            layer.hc_post(x, res, post, comb)
        torch.accelerator.synchronize()
    logger.info(
        "GLM53_MHC_WARMUP: finished %d sizes in %.2f s",
        len(sizes),
        time.perf_counter() - started,
    )
'''

# --------------------------------------------------------------------------
# GLM53_KDA_TRIM: third_party/flash_linear_attention/ops/{fused_recurrent,kda}.py
# --------------------------------------------------------------------------
FUSED_RECURRENT_EDITS = [
    (
        "fused_recurrent.py: stride params",
        """    SAFE_GATE: tl.constexpr,  # bounded gate variant (only branch implemented)
    LOWER_BOUND: tl.constexpr,
):
""",
        """    SAFE_GATE: tl.constexpr,  # bounded gate variant (only branch implemented)
    LOWER_BOUND: tl.constexpr,
    # GLM53_KDA_TRIM (v13): per-token element strides of q/k/v/beta, read
    # only when STRIDED_QKVB. Each token's [H, K] block must be dense.
    stride_q_tok,
    stride_k_tok,
    stride_v_tok,
    stride_beta_tok,
    STRIDED_QKVB: tl.constexpr,
):
""",
    ),
    (
        "fused_recurrent.py: strided base pointers",
        """    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
""",
        """    if STRIDED_QKVB:
        p_q = q + bos * stride_q_tok + i_h * K + o_k
        p_k = k + bos * stride_k_tok + i_h * K + o_k
        p_v = v + bos * stride_v_tok + i_hv * V + o_v
    else:
        p_q = q + (bos * H + i_h) * K + o_k
        p_k = k + (bos * H + i_h) * K + o_k
        p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    elif STRIDED_QKVB:
        p_beta = beta + bos * stride_beta_tok + i_hv
    else:
        p_beta = beta + bos * HV + i_hv
""",
    ),
    (
        "fused_recurrent.py: strided pointer advance",
        """        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
""",
        """        if STRIDED_QKVB:
            p_q += stride_q_tok
            p_k += stride_k_tok
            p_v += stride_v_tok
        else:
            p_q += H * K
            p_k += H * K
            p_v += HV * V
        p_o += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        if STRIDED_QKVB and not IS_BETA_HEADWISE:
            p_beta += stride_beta_tok
        else:
            p_beta += HV * (V if IS_BETA_HEADWISE else 1)
""",
    ),
    (
        "fused_recurrent.py: GDN call site (dense)",
        """        SAFE_GATE=True,
        LOWER_BOUND=-5.0,
        num_warps=num_warps,
""",
        """        SAFE_GATE=True,
        LOWER_BOUND=-5.0,
        stride_q_tok=0,
        stride_k_tok=0,
        stride_v_tok=0,
        stride_beta_tok=0,
        STRIDED_QKVB=False,
        num_warps=num_warps,
""",
    ),
]

KDA_OPS_EDITS = [
    (
        "ops/kda.py: import os",
        "\nimport torch\nimport torch.nn as nn\n",
        "\nimport os\n\nimport torch\nimport torch.nn as nn\n",
    ),
    (
        "ops/kda.py: trim predicate",
        """BT_LIST_AUTOTUNE = [32, 64, 128]
""",
        """# GLM53_KDA_TRIM (v13): let fused_recurrent_kda read token-strided q/k/v/beta
# views (split() of the merged projection / conv output) instead of copying
# them with .contiguous(). The kernel loads the same elements either way.
_GLM53_KDA_TRIM = os.environ.get("GLM53_KDA_TRIM") == "1"


def _glm53_kda_trim_ok(q, k, v, beta) -> bool:
    return (
        _GLM53_KDA_TRIM
        and q.shape[0] == 1
        and beta is not None
        and beta.ndim == 3
        and beta.stride(-1) == 1
        and all(t.stride(-1) == 1 and t.stride(-2) == t.shape[-1] for t in (q, k, v))
    )


BT_LIST_AUTOTUNE = [32, 64, 128]
""",
    ),
    (
        "ops/kda.py: fwd signature",
        """    compute_gate: bool = False,
    lower_bound: float | None = -5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
""",
        """    compute_gate: bool = False,
    lower_bound: float | None = -5.0,
    strided_qkvb: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
""",
    ),
    (
        "ops/kda.py: dense output buffer",
        """    if out is None:
        o = torch.empty_like(k)
""",
        """    if out is None:
        o = torch.empty_like(k, memory_format=torch.contiguous_format)
""",
    ),
    (
        "ops/kda.py: kernel stride args",
        """        SAFE_GATE=True,
        LOWER_BOUND=lower_bound if lower_bound is not None else -5.0,
        num_warps=num_warps,
""",
        """        SAFE_GATE=True,
        LOWER_BOUND=lower_bound if lower_bound is not None else -5.0,
        stride_q_tok=q.stride(1),
        stride_k_tok=k.stride(1),
        stride_v_tok=v.stride(1),
        stride_beta_tok=beta.stride(1),
        STRIDED_QKVB=strided_qkvb,
        num_warps=num_warps,
""",
    ),
    (
        "ops/kda.py: wrapper skips .contiguous()",
        """    o, final_state = fused_recurrent_kda_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
""",
        """    trim = _glm53_kda_trim_ok(q, k, v, beta)
    o, final_state = fused_recurrent_kda_fwd(
        q=q if trim else q.contiguous(),
        k=k if trim else k.contiguous(),
        v=v if trim else v.contiguous(),
        g=g.contiguous(),
        beta=beta if trim else beta.contiguous(),
        strided_qkvb=trim,
""",
    ),
]

# --------------------------------------------------------------------------
# GLM53_SKIP_MTP_WEIGHTS: model.py + model_loader/ep_weight_filter.py
# --------------------------------------------------------------------------
MODEL_SKIP_MTP_EDITS = [
    (
        "model.py: skip MTP weights before read",
        """        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )
""",
        """        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

        # GLM53_SKIP_MTP_WEIGHTS (v13): load_weights drops the MTP layer(s)
        # only after the safetensors iterator has read them (13.84 GiB BF16 in
        # the nvidia pack). When the drafter is not MTP, hand their raw
        # checkpoint prefixes to the loader's pre-read filter instead.
        spec_cfg = vllm_config.speculative_config
        if os.environ.get("GLM53_SKIP_MTP_WEIGHTS") == "1" and (
            spec_cfg is None or spec_cfg.method != "mtp"
        ):
            from vllm.model_executor.model_loader import ep_weight_filter

            n_mtp = getattr(config, "num_nextn_predict_layers", 0) or 0
            skip = tuple(
                f"{root}layers.{config.num_hidden_layers + i}."
                for i in range(n_mtp)
                for root in ("model.language_model.", "model.", "")
            )
            if skip:
                ep_weight_filter.SKIP_NAME_PREFIXES = skip
                logger.info(
                    "GLM53_SKIP_MTP_WEIGHTS: not reading %s (speculative method=%s)",
                    skip,
                    getattr(spec_cfg, "method", None),
                )
""",
    ),
]

EP_FILTER_EDITS = [
    (
        "ep_weight_filter.py: skip-prefix table",
        '_EXPERT_ID_RE = re.compile(r"\\.experts\\.(\\d+)\\.")\n',
        (
            '_EXPERT_ID_RE = re.compile(r"\\.experts\\.(\\d+)\\.")\n'
            "\n"
            "# GLM53_SKIP_MTP_WEIGHTS (v13): raw checkpoint-name prefixes skipped\n"
            "# before the tensor is read. Empty unless Glm5NextModel fills it.\n"
            "SKIP_NAME_PREFIXES: tuple[str, ...] = ()\n"
        ),
    ),
    (
        "ep_weight_filter.py: skip-prefix check",
        '''    belong to the local rank and should be skipped during loading."""
    if local_expert_ids is None:
''',
        '''    belong to the local rank and should be skipped during loading."""
    if SKIP_NAME_PREFIXES and weight_name.startswith(SKIP_NAME_PREFIXES):
        return True
    if local_expert_ids is None:
''',
    ),
]

# --------------------------------------------------------------------------
# GLM53_DFLASH_PREFIX_CACHE_FIX: v1/core/kv_cache_coordinator.py
# --------------------------------------------------------------------------
COORDINATOR_EDITS = [
    (
        "kv_cache_coordinator.py: import os",
        "from abc import ABC, abstractmethod\n",
        "import os\nfrom abc import ABC, abstractmethod\n",
    ),
    (
        "kv_cache_coordinator.py: draft-group predicate",
        "logger = init_logger(__name__)\n",
        '''logger = init_logger(__name__)

# GLM53_DFLASH_PREFIX_CACHE_FIX (v13): port of tonyd2wild's
# patch_prefix_cache_draft_group.py (GLM-5.3 DFlash2 2x DGX Spark recipe).
_GLM53_PREFIX_CACHE_FIX = os.environ.get("GLM53_DFLASH_PREFIX_CACHE_FIX") == "1"


def _glm53_is_draft_swa_spec(spec) -> bool:
    """True only for the DFlash draft sliding-window group (patch_v11).

    Exact type on purpose: KpoolTailSpec subclasses SlidingWindowSpec. The
    draft group spec is a UniformTypeKVCacheSpecs of SlidingWindowSpec layers.
    """
    specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(specs, dict) and specs:
        spec = next(iter(specs.values()))
    return type(spec) is SlidingWindowSpec
''',
    ),
    (
        "kv_cache_coordinator.py: EAGLE fallback flags the draft group only",
        """        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))
""",
        """        if use_eagle and not self.eagle_group_ids:
            # GLM53_DFLASH_PREFIX_CACHE_FIX: only the DFlash draft window needs
            # the EAGLE last-block drop; the target MLA and KDA groups hold
            # complete KV/state for every cached token.
            draft_swa_ids = (
                {
                    i
                    for i, g in enumerate(kv_cache_config.kv_cache_groups)
                    if _glm53_is_draft_swa_spec(g.kv_cache_spec)
                }
                if _GLM53_PREFIX_CACHE_FIX
                else set()
            )
            self.eagle_group_ids = draft_swa_ids or set(
                range(len(kv_cache_config.kv_cache_groups))
            )
            if draft_swa_ids:
                logger.info_once(
                    "GLM53_DFLASH_PREFIX_CACHE_FIX: EAGLE block drop on draft KV "
                    "group(s) %s only",
                    sorted(draft_swa_ids),
                )
""",
    ),
    (
        "kv_cache_coordinator.py: draft window never shrinks the target hit",
        """                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
""",
        """                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                if _GLM53_PREFIX_CACHE_FIX and _glm53_is_draft_swa_spec(spec):
                    # The short draft window must not shrink the hit the target
                    # and KDA groups agreed on. A shorter draft hit is dropped
                    # (fresh draft pages); the target still verifies every
                    # draft token, so output stays lossless.
                    hit_ok = _new_hit_length >= curr_hit_length
                    for group_id, blocks in zip(group_ids, hit_blocks):
                        hit_blocks_by_group[group_id] = blocks if hit_ok else None
                        hit_length_by_group[group_id] = (
                            _new_hit_length if hit_ok else 0
                        )
                    continue
                curr_hit_length = _new_hit_length
""",
    ),
]

FILE_EDITS = {
    "models/glm5next/nvidia/model.py": MODEL_EDITS + MODEL_SKIP_MTP_EDITS,
    "v1/attention/backends/mla/indexer.py": INDEXER_EDITS,
    "model_executor/warmup/kernel_warmup.py": KERNEL_WARMUP_EDITS,
    "third_party/flash_linear_attention/ops/fused_recurrent.py": FUSED_RECURRENT_EDITS,
    "third_party/flash_linear_attention/ops/kda.py": KDA_OPS_EDITS,
    "model_executor/model_loader/ep_weight_filter.py": EP_FILTER_EDITS,
    "v1/core/kv_cache_coordinator.py": COORDINATOR_EDITS,
}

NEW_FILES = {
    "model_executor/warmup/glm5next_mhc_warmup.py": GLM5NEXT_MHC_WARMUP,
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
    print(f"v13 misc patches applied ({len(staged)} files written)")
    return list(staged)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
