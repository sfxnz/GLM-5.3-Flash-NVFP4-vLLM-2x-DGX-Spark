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

GLM53_WQ_DEQUANT_MIN_M=<rows> (unset = off) sends the swapped layers of the
groups in GLM53_WQ_DEQUANT_GROUPS (default kda_in; never the LM head) through
dequantize-then-cuBLAS when their input has at least that many rows (long
prefill chunks), instead of the Marlin GEMM, which is slower there.

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

GLM53_WQ_DEQUANT_MIN_M (a positive integer; unset or empty = off: nothing is
wrapped and no workspace is allocated): every swapped layer of the groups in
GLM53_WQ_DEQUANT_GROUPS (a comma list of the groups above; unset or empty =
kda_in), except the LM head, gets a MarlinDequantMethod; other swapped layers
keep the Marlin GEMM at every M. An input of at least that many rows (a long
prefill chunk, which runs eagerly) dequantizes the packed weight into one BF16
workspace shared by the whole process and runs F.linear (cuBLAS). Smaller
inputs keep the Marlin GEMM; decode and verify steps are captured at 16 rows
or fewer. Each workspace element is the BF16 rounding of the exact fp32
q * scale, the value dequantize_per_channel, dequantize_nvfp4 and
dequantize_int give. For INT groups it is also the BF16 product Marlin forms
in registers. The workspace is sized for the largest wrapped layer when its
model is swapped, so a drafter swap with draft listed may grow it once.
"""

import os
from typing import NamedTuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # the CPU quantizer tests run without triton
    triton = None

ENV = "GLM53_FP8_W8A16"
ENV_NVFP4 = "GLM53_NVFP4_W4A16"
ENV_INT8 = "GLM53_INT8_W8A16"
ENV_INT4 = "GLM53_INT4_W4A16"
ENV_INT_GROUP = "GLM53_INT_GROUP_SIZE"
ENV_DEQUANT_MIN_M = "GLM53_WQ_DEQUANT_MIN_M"
ENV_DEQUANT_GROUPS = "GLM53_WQ_DEQUANT_GROUPS"
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


def dequant_min_m(environ=os.environ) -> int | None:
    """GLM53_WQ_DEQUANT_MIN_M as a positive int, or None when unset or empty."""
    value = environ.get(ENV_DEQUANT_MIN_M, "").strip()
    if not value:
        return None
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"{ENV_DEQUANT_MIN_M}={value!r}; want a positive integer")
    return int(value)


def dequant_groups(environ=os.environ) -> list[str]:
    """GLM53_WQ_DEQUANT_GROUPS as a list of GROUPS; unset or empty = kda_in."""
    return parse_groups(environ.get(ENV_DEQUANT_GROUPS, ""), ENV_DEQUANT_GROUPS) or ["kda_in"]


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
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=qweight.device)
    u = (qweight.unsqueeze(1) >> shifts.view(1, -1, 1)) & (2**bits - 1)
    q = u.flatten(0, 1) - 2 ** (bits - 1)
    group = q.shape[0] // scale.shape[0]
    return (q.float() * scale.float().repeat_interleave(group, 0)).T


# GLM53_WQ_DEQUANT_MIN_M: BF16 weights straight from the Marlin layout.
DQ_FORMATS = {"int8": 0, "int4": 1, "fp8": 2, "nvfp4": 3}
_DQ_TABLES: dict = {}
_WORKSPACE = None  # the shared BF16 workspace (1-D)


def marlin_dequant_tables(weight_perm, scale_perm, nvfp4: bool = False):
    """int32 (tile, scol) tables that undo Marlin's weight and scale permutations.

    gptq_marlin_repack stores 16 rows of K per row, 16 x 16 tiles side by side,
    and permutes every 1024 elements (64 columns) by weight_perm (vLLM's
    get_weight_perm, which its repack test checks against). tile[k % 16, n % 64]
    is where element (k, n) lands inside its chunk. marlin_permute_scales
    permutes every len(scale_perm) scale columns; scol[n % len] is where
    column n lands. nvfp4_marlin_process_scales then swaps columns 1 and 2 of
    every 4.
    """
    perm = torch.as_tensor(weight_perm, dtype=torch.long).flatten()
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    j = torch.arange(64)
    tile = inv[(j // 16 * 256 + j % 16).unsqueeze(0) + 16 * torch.arange(16).unsqueeze(1)]
    sp = torch.as_tensor(scale_perm, dtype=torch.long)
    scol = torch.empty_like(sp)
    scol[sp] = torch.arange(sp.numel())
    if nvfp4:
        scol ^= ((scol ^ (scol >> 1)) & 1) * 3
    return tile.to(torch.int32), scol.to(torch.int32)


if triton is not None:

    @triton.jit
    def _marlin_dequant_kernel(
        q_ptr,  # repacked weight as bytes
        s_ptr,  # scales: BF16 bits as int16 (INT, FP8) or S0E5M3 codes (NVFP4)
        tile_ptr,
        scol_ptr,
        out_ptr,  # BF16 bits as int16, (N, K) row-major
        N,
        K,
        row_elems,
        s_stride,
        gscale,
        FMT: tl.constexpr,  # DQ_FORMATS
        GROUP: tl.constexpr,  # K elements per scale row; 0 = per channel
        SCHUNK: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
        k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        mask = (n < N) & (k < K)
        e = (k // 16) * row_elems + (n // 64) * 1024 + tl.load(tile_ptr + (k % 16) * 64 + n % 64)
        if FMT == 0 or FMT == 2:
            q = tl.load(q_ptr + e, mask=mask, other=0).to(tl.int32)
        else:
            q = (tl.load(q_ptr + e // 2, mask=mask, other=0).to(tl.int32) >> ((e % 2) * 4)) & 15
        col = (n // SCHUNK) * SCHUNK + tl.load(scol_ptr + n % SCHUNK)
        if GROUP > 0:
            s = tl.load(s_ptr + (k // GROUP) * s_stride + col, mask=mask, other=0).to(tl.int32)
        else:
            s = tl.load(s_ptr + col, mask=n < N, other=0).to(tl.int32)
        # Every product below is exact in fp32, except NVFP4's, which rounds
        # in dequantize_nvfp4's order: e2m1 * (block scale * global scale).
        if FMT == 3:
            # S0E5M3 code of block scale * sf * 2^7 -> fp32 block scale * sf.
            bs = ((((s >> 3) + 105) << 23) | ((s & 7) << 20)).to(tl.float32, bitcast=True)
            bs = tl.where(s == 0, 0.0, bs)
            m = q & 7
            mag = tl.where(
                m < 2,
                (m & 1).to(tl.float32) * 0.5,
                ((((m >> 1) + 126) << 23) | ((m & 1) << 22)).to(tl.float32, bitcast=True),
            )
            w = mag * (bs * gscale)  # gscale = global scale / sf
            w = tl.where(q >= 8, -w, w)
        elif FMT == 2:
            m = q & 0x7F
            mag = tl.where(
                m < 8,
                (m & 7).to(tl.float32) * 0.001953125,  # e4m3 subnormal: m * 2^-9
                ((m << 20) + (120 << 23)).to(tl.float32, bitcast=True),
            )
            # Marlin's scale carries 2^120 (fp8_fused_exponent_bias_into_scales).
            w = mag * ((s << 16) - (120 << 23)).to(tl.float32, bitcast=True)
            w = tl.where(q >= 128, -w, w)
        else:
            if FMT == 0:
                q = q - 128
            else:
                q = q - 8
            w = q.to(tl.float32) * (s << 16).to(tl.float32, bitcast=True)
        # fp32 -> BF16, round to nearest even (the CPU interpreter's cast truncates).
        b = w.to(tl.int32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
        tl.store(out_ptr + n * K + k, b.to(tl.int16), mask=mask)


class DequantSpec(NamedTuple):
    fmt: int
    bits: int
    group: int
    gscale: float
    tile: torch.Tensor
    scol: torch.Tensor


def _marlin_perms():
    from vllm.model_executor.layers.quantization.utils.marlin_utils import get_scale_perms
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        get_weight_perm,
    )

    return get_weight_perm, get_scale_perms()


def dequant_spec(layer: torch.nn.Module, mode: str, group_size: int | None) -> DequantSpec:
    """Kernel arguments for one swapped layer; tables are cached per mode and device."""
    dev = layer.weight.device
    bits = 4 if mode in ("int4", "nvfp4") else 8
    if (mode, dev) not in _DQ_TABLES:
        get_weight_perm, (scale_perm, scale_perm_single) = _marlin_perms()
        tables = marlin_dequant_tables(
            get_weight_perm(bits), scale_perm_single if mode == "fp8" else scale_perm,
            mode == "nvfp4")
        _DQ_TABLES[mode, dev] = tuple(t.to(dev) for t in tables)
    group = group_size if mode in INT_BITS else 16 if mode == "nvfp4" else 0
    # Marlin's global scale is global * 2^119 / sf (nvfp4_marlin_process_global_scale).
    gscale = float(layer.weight_global_scale) * 2.0**-119 if mode == "nvfp4" else 0.0
    return DequantSpec(DQ_FORMATS[mode], bits, group, gscale, *_DQ_TABLES[mode, dev])


def ensure_workspace(numel: int, device: torch.device) -> torch.Tensor:
    """The shared BF16 workspace, grown (never shrunk) to at least numel elements."""
    global _WORKSPACE
    if _WORKSPACE is None or _WORKSPACE.numel() < numel:
        _WORKSPACE = None  # free the old one first
        _WORKSPACE = torch.empty(numel, dtype=torch.bfloat16, device=device)
    return _WORKSPACE


def dequantize_marlin(
    layer: torch.nn.Module, spec: DequantSpec, out: torch.Tensor | None = None
) -> torch.Tensor:
    """(N, K) BF16 weight of a swapped layer, read from its Marlin tensors and
    written into out (default: the front of the shared workspace)."""
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    if out is None:
        out = _WORKSPACE[: n * k].view(n, k)
    w, s = layer.weight, layer.weight_scale
    _marlin_dequant_kernel[(triton.cdiv(n, 64), triton.cdiv(k, 64))](
        w.view(torch.uint8),
        s.view(torch.uint8 if spec.fmt == DQ_FORMATS["nvfp4"] else torch.int16),
        spec.tile,
        spec.scol,
        out.view(torch.int16),
        n,
        k,
        w.shape[1] * 32 // spec.bits,
        s.shape[1],
        spec.gscale,
        FMT=spec.fmt,
        GROUP=spec.group,
        SCHUNK=spec.scol.numel(),
        BLOCK_N=64,
        BLOCK_K=64,
    )
    return out


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


class MarlinDequantMethod:
    """GLM53_WQ_DEQUANT_MIN_M wrapper around a swapped layer's Marlin method.

    An input of at least min_m rows dequantizes the weight into the shared
    workspace and runs F.linear. Smaller inputs keep the Marlin GEMM, and so
    does anything torch.compile traces, so a compiled graph never branches on
    a symbolic row count. Captured decode and verify steps have fixed row
    counts of 16 or fewer, so the branch is constant per graph.
    """

    def __init__(self, inner, min_m: int, spec: DequantSpec) -> None:
        self._inner, self._min_m, self._spec = inner, min_m, spec

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        if torch.compiler.is_compiling() or x.numel() < self._min_m * x.shape[-1]:
            return self._inner.apply(layer, x, bias)
        return torch.nn.functional.linear(x, dequantize_marlin(layer, self._spec), bias)


def install_marlin_dequant(layers, min_m: int, group_size: int | None) -> int:
    """Wrap each (layer, mode) in MarlinDequantMethod and grow the shared
    workspace to the largest layer. Returns the workspace size in bytes."""
    numel = 0
    for layer, mode in layers:
        spec = dequant_spec(layer, mode, group_size)
        layer.quant_method = MarlinDequantMethod(layer.quant_method, min_m, spec)
        numel = max(numel, layer.output_size_per_partition * layer.input_size_per_partition)
    ws = ensure_workspace(numel, layers[0][0].weight.device)
    return ws.numel() * ws.element_size()


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


def _compact(layer: torch.nn.Module) -> None:
    """Copy a swapped layer's new tensors to the lowest free addresses.

    vLLM loads under max_split_size_mb:20, and run.sh sets
    expandable_segments:True, which maps memory in 20 MiB pages. The freed
    BF16 blocks are "oversized" for the new tensors, so each packed weight
    lands on freshly mapped pages among the swap's transients and keeps two
    partly used pages once those are unmapped. After empty_cache the only
    mapped free memory is page remainders: a copy either fills one or starts
    right after the last live block, so consecutive layers share pages.
    """
    torch.cuda.empty_cache()
    for attr in ("weight", "weight_scale", "weight_global_scale"):
        p = getattr(layer, attr, None)
        if p is not None:
            p.data = p.data.clone()


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
    min_m = dequant_min_m()
    dq_groups = dequant_groups() if min_m else []
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
    skipped, swapped = [], []
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
        _compact(layer)
        swapped.append((group, layer))
        st = stats.setdefault(group, [0, 0, 0])
        st[0] += 1
        st[1] += bf16_bytes
        for attr in ("weight", "weight_scale", "weight_global_scale"):
            p = getattr(layer, attr, None)
            st[2] += 0 if p is None else p.numel() * p.element_size()
    # The LM head only sees sampled rows (max_num_seqs * (k + 1) at most).
    big = [(layer, modes[g]) for g, layer in swapped if g in dq_groups and g != "lm_head"]
    if big:
        ws_bytes = install_marlin_dequant(big, min_m, gsize)
        logger.info("%s=%d (groups %s): %s %d layers dequantize at M >= %d; shared "
                    "BF16 workspace %.1f MiB", ENV_DEQUANT_MIN_M, min_m, ",".join(dq_groups),
                    model_name, len(big), min_m, ws_bytes / 2**20)
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
