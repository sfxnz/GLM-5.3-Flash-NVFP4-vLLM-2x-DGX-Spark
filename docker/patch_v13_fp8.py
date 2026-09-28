"""Opt-in weight-only Marlin (FP8/INT8 W8A16, NVFP4/INT4 W4A16) for the BF16 non-MoE linears.

Every verify step streams ~8.3 GB/rank of BF16 projection weights that the
nvidia pack (and the DFlash2 drafter) keep unquantized: KDA in/out projections,
MLA q_b/o, shared experts, the lm_head (read twice: DFlash2 shares it) and the
drafter itself. This patch lets the recipe store selected groups as FP8 e4m3
with one scale per output channel and run them through vLLM's Marlin FP8 GEMM
(BF16 activations, in-register dequant), halving those bytes; or as NVFP4
(E2M1 with an e4m3 scale per 16 elements and an fp32 tensor scale) through
vLLM's Marlin NVFP4 W4A16 GEMM, cutting them to 0.28x; or as symmetric INT8 /
INT4 with one BF16 scale per 128 (or 64) elements along K through vLLM's
GPTQ-Marlin GEMM (uint8b128 / uint4b8), 0.51x / 0.26x.

GLM53_FP8_W8A16, GLM53_NVFP4_W4A16, GLM53_INT8_W8A16 and GLM53_INT4_W4A16 are
comma lists of groups: draft, shared, mla, kda_o, kda_in, lm_head. A group may
be in one list only. GLM53_INT_GROUP_SIZE (64 or 128, default 128) sets the
INT group size. All four unset or empty skips the new code entirely (v11
behaviour). The swap runs at the end of
model_loader.utils.process_weights_after_loading, once per loaded model
(target, then drafter), and frees each BF16 weight it replaces.

Stays BF16 by construction: indexer, router gate, mHC, embeddings, kv_b
(absorbed into the MLA BMMs), fused_qkv_a, KDA f_b/g_b/conv, vision tower.

Usage: python3 patch_v13_fp8.py [VLLM_ROOT]
(default /usr/local/lib/python3.12/dist-packages/vllm). Exact-substring
anchors; refuses on drift before writing anything; re-running is a no-op.
"""

import py_compile
import sys
from pathlib import Path

ROOT = Path(
    sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm"
)
MODULE = "model_executor/layers/quantization/glm53_fp8_w8a16.py"

MODULE_SRC = '''\
# SPDX-License-Identifier: Apache-2.0
"""GLM53_{FP8,INT8}_W8A16 / GLM53_{NVFP4,INT4}_W4A16: opt-in weight-only Marlin for BF16 linears.

Installed by the GLM-5.3-Flash 2x DGX Spark recipe (docker/patch_v13_fp8.py).
model_loader.utils.process_weights_after_loading calls
apply_glm53_fp8_w8a16() only when one of the four variables is set.

Groups (comma list; a group may be in one of the four variables only):
  kda_in   KDA in_proj_qkvbfg_a (merged q|k|v|b|f_a|g_a)
  kda_o    KDA o_proj
  mla      MLA q_b_proj and o_proj
  shared   MoE shared_experts gate_up_proj and down_proj
  lm_head  target ParallelLMHead (the DFlash2 drafter shares this module)
  draft    every BF16 linear of the DFlash/DFlash2 drafter

FP8 (W8A16): symmetric e4m3, one scale per output channel of the per-rank
shard. The scale is amax/448 rounded to BF16, which is exactly the scale Marlin
applies, and W/scale is rounded once to e4m3.

NVFP4 (W4A16): the per-rank shard gets one fp32 global scale amax/(6*448), so
the largest block scale lands on the e4m3 maximum. Each 16-element block along
K tries the e4m3 block scales NVFP4_CODE_STEPS codes away from round(amax/6),
rounds W/scale to the nearest E2M1 value and keeps the lowest squared error.
Block scales stay in the e4m3 normal range, because Marlin zeroes subnormal
ones. The packing (low nibble = even k) and scale layout are what vLLM's
MarlinNvFp4LinearKernel takes from a ModelOpt NVFP4 checkpoint.

INT8 / INT4 (W8A16 / W4A16): symmetric round-to-nearest with one scale per
GLM53_INT_GROUP_SIZE elements along K of the per-rank shard, so no group
straddles a TP shard. The scale is ratio * amax / qmax (qmax 127 or 7) rounded
to BF16; INT_CLIP_RATIOS lists the ratios tried per group, and the lowest
squared error wins. Values are stored with bias 128 / 8 (uint8b128, uint4b8)
in GPTQ layout, then go through the same pad, gptq_marlin_repack and
marlin_permute_scales steps as vLLM's MarlinLinearKernel. A layer whose K is
not a multiple of the group size stays BF16.
"""

import os

import torch

ENV = "GLM53_FP8_W8A16"
ENV_NVFP4 = "GLM53_NVFP4_W4A16"
ENV_INT8 = "GLM53_INT8_W8A16"
ENV_INT4 = "GLM53_INT4_W4A16"
ENV_INT_GROUP = "GLM53_INT_GROUP_SIZE"
MODE_ENVS = {"fp8": ENV, "nvfp4": ENV_NVFP4, "int8": ENV_INT8, "int4": ENV_INT4}
INT_BITS = {"int8": 8, "int4": 4}
INT_GROUP_SIZES = (64, 128)
# Scale candidates per group, as fractions of amax / qmax; 1.0 first so ties keep it.
INT_CLIP_RATIOS = {"int8": (1.0,), "int4": (1.0, 0.95, 0.9, 0.85)}
GROUPS = ("draft", "shared", "mla", "kda_o", "kda_in", "lm_head")
FP8_MAX = 448.0
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E4M3_MIN_NORMAL_CODE, E4M3_MAX_CODE = 0x08, 0x7E  # 2^-6 and 448
# Block-scale candidates, in e4m3 codes from round(amax/6); 0 first so ties keep it.
NVFP4_CODE_STEPS = (0, -1, 1, -2)

_TARGET_MODELS = frozenset({"Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"})
_DRAFT_MODEL = "DFlashQwen3ForCausalLM"
# owner class name -> group -> attribute paths of the linears to swap
_TARGET_SITES = {
    "Glm5NextLinearAttention": {
        "kda_in": ("in_proj_qkvbfg_a",),
        "kda_o": ("o_proj",),
    },
    "Glm5NextMLAAttention": {"mla": ("q_b_proj", "o_proj")},
    "Glm5NextMoE": {
        "shared": ("shared_experts.gate_up_proj", "shared_experts.down_proj"),
    },
}


def parse_groups(value: str, env: str = ENV) -> list[str]:
    groups = list(dict.fromkeys(g.strip() for g in value.split(",") if g.strip()))
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise ValueError(
            f"{env}: unknown group(s) {unknown}; valid: {','.join(GROUPS)}"
        )
    return groups


def selected_modes(environ=os.environ) -> dict[str, str]:
    """group -> "fp8", "nvfp4", "int8" or "int4", from the four MODE_ENVS."""
    modes: dict[str, str] = {}
    for mode, env in MODE_ENVS.items():
        for group in parse_groups(environ.get(env, ""), env):
            if group in modes:
                raise ValueError(f"{group} set in both {MODE_ENVS[modes[group]]} and {env}")
            modes[group] = mode
    return modes


def int_group_size(environ=os.environ) -> int:
    value = environ.get(ENV_INT_GROUP, "").strip() or "128"
    if value not in {str(g) for g in INT_GROUP_SIZES}:
        raise ValueError(f"{ENV_INT_GROUP}={value!r}; valid: {INT_GROUP_SIZES}")
    return int(value)


def quantize_per_channel(
    weight: torch.Tensor, chunk_elems: int = 1 << 24
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric e4m3 quantization of an (N, K) weight.

    Returns (q, scale): q is float8_e4m3fn (N, K), scale is (N,) in
    weight.dtype, and weight ~= q.float() * scale.float()[:, None]. Rows go in
    chunks so the fp32 transient stays at chunk_elems * 4 bytes.
    """
    n, k = weight.shape
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty(n, dtype=weight.dtype, device=weight.device)
    rows = max(1, chunk_elems // k)
    for i in range(0, n, rows):
        w = weight[i : i + rows].float()
        s = (w.abs().amax(dim=1) / FP8_MAX).clamp_min(1e-12).to(weight.dtype)
        scale[i : i + rows] = s
        q[i : i + rows] = (
            (w / s.float().unsqueeze(1))
            .clamp_(-FP8_MAX, FP8_MAX)
            .to(torch.float8_e4m3fn)
        )
    return q, scale


def dequantize_per_channel(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float().unsqueeze(1)


def quantize_nvfp4(
    weight: torch.Tensor, chunk_elems: int = 1 << 22
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 quantization of an (N, K) weight, K a multiple of 16.

    Returns (packed, scale, global_scale): packed is uint8 (N, K/2) with
    element 2j in the low nibble, scale is float8_e4m3fn (N, K/16), and
    global_scale is a 0-dim fp32 tensor; weight ~= e2m1 * scale * global_scale.
    Rows go in chunks so the transients stay at about chunk_elems * 40 bytes.
    """
    n, k = weight.shape
    if k % 16:
        raise ValueError(f"NVFP4 needs K % 16 == 0, got K={k}")
    dev = weight.device
    lo, hi = torch.aminmax(weight)
    gscale = (torch.maximum(-lo.float(), hi.float()) / (6 * FP8_MAX)).clamp_min(1e-30)
    # Explicit fp32: the loader runs this under set_default_torch_dtype(bf16).
    grid = torch.tensor(E2M1_GRID, dtype=torch.float32, device=dev)
    mids = torch.tensor(E2M1_MIDS, dtype=torch.float32, device=dev)
    packed = torch.empty((n, k // 2), dtype=torch.uint8, device=dev)
    scale = torch.empty((n, k // 16), dtype=torch.uint8, device=dev)
    rows = max(1, chunk_elems // k)
    for i in range(0, n, rows):
        w = weight[i : i + rows].float().unflatten(1, (k // 16, 16))
        s0 = w.abs().amax(-1) / (6 * gscale)
        base = s0.clamp_(2**-6, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8).int()
        sign = (w < 0).to(torch.uint8) << 3
        best = None
        for step in NVFP4_CODE_STEPS:
            code = (base + step).clamp_(E4M3_MIN_NORMAL_CODE, E4M3_MAX_CODE)
            code = code.to(torch.uint8)
            s = (code.view(torch.float8_e4m3fn).float() * gscale).unsqueeze(-1)
            x = w / s
            idx = torch.bucketize(x.abs(), mids)
            err = (grid[idx].copysign(x) * s - w).square_().sum(-1)
            nib = idx.to(torch.uint8) | sign
            if best is None:
                best, best_code, best_nib = err, code, nib
            else:
                better = err < best
                best = torch.where(better, err, best)
                best_code = torch.where(better, code, best_code)
                best_nib = torch.where(better.unsqueeze(-1), nib, best_nib)
        nib = best_nib.flatten(1)
        packed[i : i + rows] = nib[:, 0::2] | (nib[:, 1::2] << 4)
        scale[i : i + rows] = best_code
    return packed, scale.view(torch.float8_e4m3fn), gscale


def dequantize_nvfp4(
    packed: torch.Tensor, scale: torch.Tensor, gscale: torch.Tensor
) -> torch.Tensor:
    nib = torch.stack((packed & 15, packed >> 4), -1).flatten(1)
    grid = torch.tensor(E2M1_GRID, dtype=torch.float32, device=packed.device)
    v = grid[(nib & 7).long()]
    v = torch.where(nib >= 8, -v, v).unflatten(1, (-1, 16))
    return (v * (scale.float() * gscale).unsqueeze(-1)).flatten(1)


def quantize_int(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    ratios: tuple[float, ...] = (1.0,),
    chunk_elems: int = 1 << 22,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group INT8/INT4 quantization of an (N, K) weight along K.

    Returns (qweight, scale) in GPTQ layout: qweight is int32 (K * bits / 32, N),
    element k of column n sits in bits [bits * (k % p), bits * (k % p + 1)) of
    row k // p (p = 32 / bits), stored as q + 2**(bits - 1); scale is
    (K / group_size, N) in weight.dtype, and weight ~= q * scale. Rows go in
    chunks so the fp32 transients stay at a few chunk_elems * 4 bytes.
    """
    n, k = weight.shape
    if k % group_size:
        raise ValueError(f"INT{bits} needs K % {group_size} == 0, got K={k}")
    pack, qmax = 32 // bits, 2 ** (bits - 1) - 1
    dev = weight.device
    qweight = torch.empty((k // pack, n), dtype=torch.int32, device=dev)
    scale = torch.empty((k // group_size, n), dtype=weight.dtype, device=dev)
    tiny = torch.finfo(weight.dtype).tiny  # keeps all-zero groups at q = 0
    rows = max(1, chunk_elems // k)
    for i in range(0, n, rows):
        w = weight[i : i + rows].float().unflatten(1, (k // group_size, group_size))
        amax = w.abs().amax(-1, keepdim=True)
        best = None
        for ratio in ratios:
            s = (amax * (ratio / qmax)).to(weight.dtype).float().clamp_min_(tiny)
            q = torch.round(w / s).clamp_(-qmax - 1, qmax)
            err = (q * s - w).square_().sum(-1, keepdim=True)
            if best is None:
                best, best_q, best_s = err, q, s
            else:
                better = err < best
                best = torch.where(better, err, best)
                best_q = torch.where(better, q, best_q)
                best_s = torch.where(better, s, best_s)
        u = (best_q.to(torch.int32) + qmax + 1).flatten(1).unflatten(1, (k // pack, pack))
        packed = u[..., 0].clone()
        for j in range(1, pack):
            packed |= u[..., j] << (bits * j)
        qweight[:, i : i + rows] = packed.T
        scale[:, i : i + rows] = best_s.flatten(1).T.to(weight.dtype)
    return qweight, scale


def dequantize_int(qweight: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """(N, K) fp32 weights from quantize_int's (qweight, scale)."""
    pack = 32 // bits
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=qweight.device)
    u = (qweight.unsqueeze(1) >> shifts.view(1, -1, 1)) & (2**bits - 1)
    q = u.flatten(0, 1) - 2 ** (bits - 1)
    group = q.shape[0] // scale.shape[0]
    return (q.float() * scale.float().repeat_interleave(group, 0)).T


class Fp8MarlinW8A16Method:
    """Replacement quant_method: Marlin FP8 weight-only GEMM, BF16 activations.

    Deliberately not a QuantizeMethodBase: the loader's post-load loops must
    skip it, because the weight is already in its final Marlin layout. apply()
    makes no host sync and allocates only through the caching allocator (the
    same ops as vLLM's MarlinFP8 linear kernel), so it is CUDA-graph safe.
    """

    def __init__(self, gemm) -> None:
        self._gemm = gemm

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        out = self._gemm(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=None,
        )
        return out if bias is None else out + bias


class Fp4MarlinW4A16Method(Fp8MarlinW8A16Method):
    """Marlin NVFP4 weight-only GEMM; the same call as MarlinNvFp4LinearKernel."""

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        out = self._gemm(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=None,
        )
        return out if bias is None else out + bias


class IntMarlinA16Method(Fp8MarlinW8A16Method):
    """GPTQ-Marlin INT8/INT4 weight-only GEMM; the call MarlinLinearKernel makes."""

    def __init__(self, gemm, wtype) -> None:
        super().__init__(gemm)
        self._wtype = wtype

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        out = self._gemm(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_zp=layer.weight_zp,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            wtype=self._wtype,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            is_k_full=True,
            bias=None,
        )
        return out if bias is None else out + bias


def _set_partition_sizes(layer: torch.nn.Module) -> None:
    n, k = layer.weight.shape
    sizes = (("output_size_per_partition", n), ("input_size_per_partition", k))
    for attr, size in sizes:
        if getattr(layer, attr, size) != size:
            raise RuntimeError(f"{ENV}: {attr}={getattr(layer, attr)} != {size}")
        setattr(layer, attr, size)


def quantize_layer_to_marlin_nvfp4(layer: torch.nn.Module) -> None:
    """Swap one bias-free BF16 (N, K) linear or LM head to Marlin NVFP4 in place."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        prepare_fp4_layer_for_marlin,
    )
    from vllm.model_executor.utils import replace_parameter

    _set_partition_sizes(layer)
    packed, scale, gscale = quantize_nvfp4(layer.weight.data)
    layer.params_dtype = layer.weight.dtype  # the Marlin scale/output dtype
    replace_parameter(layer, "weight", packed)  # drops the BF16 weight
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(gscale, requires_grad=False)
    prepare_fp4_layer_for_marlin(layer)
    layer.quant_method = Fp4MarlinW4A16Method(apply_fp4_marlin_linear)


def quantize_layer_to_marlin_fp8(layer: torch.nn.Module) -> None:
    """Swap one bias-free BF16 (N, K) linear or LM head to Marlin FP8 in place."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )
    from vllm.model_executor.utils import replace_parameter

    _set_partition_sizes(layer)
    q, scale = quantize_per_channel(layer.weight.data)
    replace_parameter(layer, "weight", q)  # drops the BF16 weight
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    layer.orig_dtype = scale.dtype
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    layer.quant_method = Fp8MarlinW8A16Method(apply_fp8_marlin_linear)


def quantize_layer_to_marlin_int(layer: torch.nn.Module, mode: str, group_size: int) -> None:
    """Swap one bias-free BF16 (N, K) linear or LM head to GPTQ-Marlin INT8/INT4 in place."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
        marlin_make_workspace_new,
        marlin_pad_qweight,
        marlin_pad_scales,
        marlin_padded_nk,
        marlin_permute_scales,
    )
    from vllm.model_executor.utils import replace_parameter
    from vllm.scalar_type import scalar_types

    _set_partition_sizes(layer)
    bits = INT_BITS[mode]
    n, k = layer.weight.shape
    qweight, scale = quantize_int(layer.weight.data, bits, group_size, INT_CLIP_RATIOS[mode])
    replace_parameter(layer, "weight", qweight)  # drops the BF16 weight
    del qweight
    pn, pk = marlin_padded_nk(n, k, group_size)
    empty = torch.empty(0, dtype=torch.int, device=scale.device)
    marlin_qweight = ops.gptq_marlin_repack(
        marlin_pad_qweight(layer.weight.data, n, k, pn, pk),
        perm=empty,
        size_k=pk,
        size_n=pn,
        num_bits=bits,
    )
    replace_parameter(layer, "weight", marlin_qweight)
    scale = marlin_pad_scales(scale, n, k, pn, pk, group_size)
    scale = marlin_permute_scales(scale, size_k=pk, size_n=pn, group_size=group_size)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    layer.workspace = marlin_make_workspace_new(scale.device)
    layer.weight_zp = layer.g_idx = layer.g_idx_sort_indices = empty  # no zp, no act-order
    wtype = scalar_types.uint8b128 if bits == 8 else scalar_types.uint4b8
    layer.quant_method = IntMarlinA16Method(apply_gptq_marlin_linear, wtype)


def _collect_sites(model, groups, linear_base):
    """Return (kind, [(group, name, layer)]); kind is target, draft or None."""
    mro = {c.__name__ for c in type(model).__mro__}
    sites = []
    if mro & _TARGET_MODELS:
        for name, module in model.named_modules():
            cls = type(module).__name__
            for group, paths in _TARGET_SITES.get(cls, {}).items():
                if group in groups:
                    for path in paths:
                        layer = module.get_submodule(path)
                        sites.append((group, f"{name}.{path}", layer))
            if cls == "ParallelLMHead" and "lm_head" in groups:
                sites.append(("lm_head", name, module))
        return "target", sites
    if _DRAFT_MODEL in mro:
        if "draft" in groups:
            for name, module in model.named_modules():
                if isinstance(module, linear_base):
                    sites.append(("draft", name, module))
        return "draft", sites
    return None, sites


def apply_glm53_fp8_w8a16(model: torch.nn.Module, target_device: torch.device) -> None:
    from vllm.logger import init_logger
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    logger = init_logger(__name__)
    modes = selected_modes()
    groups = list(modes)
    model_name = type(model).__name__
    kind, sites = _collect_sites(model, groups, LinearBase)
    if kind is None:
        logger.info("%s: %s is not a GLM-5.3 target or DFlash drafter; unchanged",
                    ENV, model_name)
        return
    wanted = {g for g in groups if (g == "draft") == (kind == "draft")}
    missing = wanted - {g for g, _, _ in sites}
    if missing:
        raise RuntimeError(f"{ENV}: no {sorted(missing)} layers found in {model_name}")
    if not sites:
        return
    inner = getattr(model, "model", None)
    if kind == "draft" and not hasattr(inner, "_num_attn_layers"):
        # The fused context-KV buffers would be rebuilt from packed weights.
        logger.warning("%s: drafter weights not loaded; draft left BF16", ENV)
        return
    if target_device.type != "cuda":
        raise RuntimeError(f"{ENV} needs a CUDA target device, got {target_device}")

    gsize = int_group_size() if set(INT_BITS) & set(modes.values()) else None
    reserved_before = torch.cuda.memory_reserved(target_device)
    plain = (UnquantizedLinearMethod, UnquantizedEmbeddingMethod)
    swap = {
        "fp8": quantize_layer_to_marlin_fp8,
        "nvfp4": quantize_layer_to_marlin_nvfp4,
        "int8": lambda layer: quantize_layer_to_marlin_int(layer, "int8", gsize),
        "int4": lambda layer: quantize_layer_to_marlin_int(layer, "int4", gsize),
    }
    stats: dict[str, list[int]] = {}
    skipped = []
    for group, name, layer in sites:
        mode = modes[group]
        w = layer.weight
        if (
            not isinstance(layer.quant_method, plain)
            or w.dtype != torch.bfloat16
            or w.dim() != 2
            or getattr(layer, "bias", None) is not None
            or (mode == "nvfp4" and w.shape[1] % 16)
            or (mode in INT_BITS and w.shape[1] % gsize)
        ):
            skipped.append(name)
            continue
        bf16_bytes = w.numel() * w.element_size()
        del w  # the swap below must drop the last reference to the BF16 weight
        swap[mode](layer)
        st = stats.setdefault(group, [0, 0, 0])
        st[0] += 1
        st[1] += bf16_bytes
        for attr in ("weight", "weight_scale", "weight_global_scale"):
            p = getattr(layer, attr, None)
            st[2] += 0 if p is None else p.numel() * p.element_size()
    torch.cuda.empty_cache()
    logger.info(
        "%s: %s %s; torch reserved %.2f -> %.2f GiB",
        "+".join(e for e in MODE_ENVS.values() if os.environ.get(e, "").strip())
        + ("" if gsize is None else f" (group {gsize})"),
        model_name,
        ", ".join(
            f"{g}[{modes[g]}]={n} layers {b / 2**20:.1f}->{f / 2**20:.1f} MiB"
            for g, (n, b, f) in stats.items()
        ),
        reserved_before / 2**30,
        torch.cuda.memory_reserved(target_device) / 2**30,
    )
    if skipped:
        logger.warning("%s: left %d selected layers as they were (not plain "
                       "bias-free BF16, NVFP4 with K %% 16, or INT with K %% "
                       "group size): %s", ENV, len(skipped), skipped[:8])
'''

HOOK_OLD = """    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)
"""
HOOK_NEW = HOOK_OLD + """
    # GLM53_{FP8,INT8}_W8A16 / GLM53_{NVFP4,INT4}_W4A16 (recipe patch_v13_fp8):
    # opt-in weight-only swap of selected BF16 linears. All unset or empty
    # keeps this function as it was.
    import os as _os

    if any(
        _os.environ.get(_v, "").strip()
        for _v in (
            "GLM53_FP8_W8A16",
            "GLM53_NVFP4_W4A16",
            "GLM53_INT8_W8A16",
            "GLM53_INT4_W4A16",
        )
    ):
        from vllm.model_executor.layers.quantization.glm53_fp8_w8a16 import (
            apply_glm53_fp8_w8a16,
        )

        apply_glm53_fp8_w8a16(model, target_device)
"""

ENV_OLD = """    for var in ray_noset_env_vars:
        factors[var] = normalize_value(os.getenv(var))
"""
ENV_NEW = ENV_OLD + """
    # GLM53_{FP8,INT8}_W8A16, GLM53_{NVFP4,INT4}_W4A16 and GLM53_INT_GROUP_SIZE
    # (recipe patch_v13_fp8) change the linears inside the torch.compile'd
    # drafter, so they must key the compile cache. Hashed only when set, so the
    # unset key matches v11.
    for _glm53 in (
        "GLM53_FP8_W8A16",
        "GLM53_NVFP4_W4A16",
        "GLM53_INT8_W8A16",
        "GLM53_INT4_W4A16",
        "GLM53_INT_GROUP_SIZE",
    ):
        if os.getenv(_glm53, "").strip():
            factors[_glm53] = normalize_value(os.getenv(_glm53))
"""

EDITS = [
    ("model_executor/model_loader/utils.py", HOOK_OLD, HOOK_NEW, "post-load hook"),
    ("envs.py", ENV_OLD, ENV_NEW, "compile-cache factor"),
]


def main() -> None:
    writes: dict[Path, str] = {}
    for rel, old, new, what in EDITS:
        path = ROOT / rel
        text = path.read_text()
        if text.count(new) == 1:
            print(f"{rel}: {what} already applied")
            continue
        if text.count(old) != 1:
            raise SystemExit(f"unexpected source for {what} in {rel}; refusing")
        writes[path] = text.replace(old, new)

    module = ROOT / MODULE
    if module.exists():
        if module.read_text() != MODULE_SRC:
            raise SystemExit(f"{MODULE} exists with other content; refusing")
        print(f"{MODULE}: already installed")
    else:
        writes[module] = MODULE_SRC

    for path, text in writes.items():
        path.write_text(text)
    for rel in [MODULE] + [rel for rel, *_ in EDITS]:
        py_compile.compile(str(ROOT / rel), doraise=True)
    print(f"patch_v13_fp8: {len(writes)} file(s) written under {ROOT}")


if __name__ == "__main__":
    main()
