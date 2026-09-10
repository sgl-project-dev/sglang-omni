# SPDX-License-Identifier: Apache-2.0
"""AuK packed QKV to FP32 Q/K RMSNorm and adjacent-pair RoPE.

Tile heads of each token, as in OmniVoice's fused attention prologue. AuK uses
adjacent even/odd rotary pairs and batch-dependent positions. Keep PyTorch
2.13's vector4 reduction and intermediate FP32 rounding; its CUDA autocast
RMSNorm runs in FP32 even though the packed projection is BF16.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.autotune(
    configs=[
        triton.Config({"HEAD_TILE": 4}, num_warps=4),
        triton.Config({"HEAD_TILE": 8}, num_warps=4),
    ],
    key=["H"],
)
@triton.jit(do_not_specialize=["S", "F_BATCH"])
def _norm_rope_kernel(
    QKV,
    QW,
    KW,
    FREQ,
    Q_OUT,
    K_OUT,
    S,
    H: tl.constexpr,
    F_BATCH,
    F_SEQ: tl.constexpr,
    EPS: tl.constexpr,
    HEAD_TILE: tl.constexpr,
):
    tile = tl.program_id(0)
    pos = tl.program_id(1)
    batch = tl.program_id(2)
    tiles_per_kind = tl.cdiv(H, HEAD_TILE)
    heads = (tile % tiles_per_kind) * HEAD_TILE + tl.arange(0, HEAD_TILE)
    is_k = tile // tiles_per_kind
    token = batch * S + pos
    groups = tl.arange(0, 16)
    base = (
        QKV
        + token * (3 * H * 64)
        + (heads[:, None] + is_k * H) * 64
        + groups[None, :] * 4
    )
    mask = heads[:, None] < H
    x0 = tl.load(base, mask, 0).to(tl.float32)
    x1 = tl.load(base + 1, mask, 0).to(tl.float32)
    x2 = tl.load(base + 2, mask, 0).to(tl.float32)
    x3 = tl.load(base + 3, mask, 0).to(tl.float32)
    # Torch's vector4 RMSNorm accumulates consecutive groups of four with FMA,
    # then uses a descending warp reduction. Do not use sum(x*x) over 64 lanes.
    acc = x0 * x0
    acc = tl.fma(x1, x1, acc)
    acc = tl.fma(x2, x2, acc)
    acc = tl.fma(x3, x3, acc)
    inv = tl.rsqrt(tl.sum(acc, axis=1) * (1.0 / 64) + EPS)
    x = tl.interleave(tl.interleave(x0, x2), tl.interleave(x1, x3))
    pair = tl.interleave(tl.interleave(x1, x3), tl.interleave(x0, x2))
    dims = tl.arange(0, 64)
    if is_k == 0:
        w = tl.load(QW + dims).to(tl.float32)
        wp = tl.load(QW + (dims ^ 1)).to(tl.float32)
    else:
        w = tl.load(KW + dims).to(tl.float32)
        wp = tl.load(KW + (dims ^ 1)).to(tl.float32)
    norm = (x * inv[:, None]) * w[None, :]
    pair_norm = (pair * inv[:, None]) * wp[None, :]
    pair_norm = tl.where(dims[None, :] % 2 == 0, -pair_norm, pair_norm)
    angle = tl.load(FREQ + batch * F_BATCH + pos * F_SEQ + dims).to(tl.float32)
    c = libdevice.cos(angle)
    s = libdevice.sin(angle)
    out = norm * c[None, :] + pair_norm * s[None, :]
    dst = batch * H * S * 64 + heads[:, None] * S * 64 + pos * 64 + dims[None, :]
    if is_k == 0:
        tl.store(Q_OUT + dst, out.to(Q_OUT.dtype.element_ty), mask)
    else:
        tl.store(K_OUT + dst, out.to(K_OUT.dtype.element_ty), mask)


def fused_norm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float = torch.finfo(torch.float32).eps,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous FP32 [B,H,S,64] Q/K from packed [B,S,3*H*64].

    The caller selects FP32 RMSNorm semantics, FP32 weights/angles and unit
    rotary scale. Angles may be shared [1,S,64] or request-specific [B,S,64].
    The packed projection remains available to the caller for its V view.
    """
    b, s, packed = qkv.shape
    h = packed // (3 * 64)
    q = torch.empty((b, h, s, 64), device=qkv.device, dtype=torch.float32)
    k = torch.empty_like(q)
    f = freqs[:, -s:]
    # Requests vary in length. Reuse the same compiled kernel and tile choice
    # instead of compiling and autotuning each new sequence length or batch.
    _norm_rope_kernel[lambda m: (2 * triton.cdiv(h, m["HEAD_TILE"]), s, b)](
        qkv,
        q_weight,
        k_weight,
        f,
        q,
        k,
        s,
        h,
        0 if f.shape[0] == 1 else f.stride(0),
        f.stride(1),
        eps,
        enable_fp_fusion=False,
    )
    return q, k
