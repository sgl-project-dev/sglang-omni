# SPDX-License-Identifier: Apache-2.0
"""Single-stream AuK rotary embedding with adjacent even/odd pairs."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=4),
    ],
    key=["N", "S", "H"],
)
@triton.jit
def _rope_kernel(
    Q,
    K,
    FREQ,
    Q_OUT,
    K_OUT,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    F_BATCH: tl.constexpr,
    F_SEQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    d = i % 64
    pos = i // 64 % S
    batch = i // (H * S * 64)
    angle = tl.load(FREQ + batch * F_BATCH + pos * F_SEQ + d, valid, 0).to(tl.float32)
    cos = libdevice.cos(angle)
    sin = libdevice.sin(angle)
    q = tl.load(Q + i, valid, 0).to(tl.float32)
    k = tl.load(K + i, valid, 0).to(tl.float32)
    q_pair = tl.load(Q + (i ^ 1), valid, 0).to(tl.float32)
    k_pair = tl.load(K + (i ^ 1), valid, 0).to(tl.float32)
    q_pair = tl.where(d % 2 == 0, -q_pair, q_pair)
    k_pair = tl.where(d % 2 == 0, -k_pair, k_pair)
    # PyTorch materializes both products before adding. Keep that rounding.
    q_out = q * cos + q_pair * sin
    k_out = k * cos + k_pair * sin
    tl.store(Q_OUT + i, q_out.to(Q_OUT.dtype.element_ty), valid)
    tl.store(K_OUT + i, k_out.to(K_OUT.dtype.element_ty), valid)


def fused_rope(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
    """Rotate contiguous FP32 [B,H,S,64] Q/K using FP32 [1 or B,S,64] angles."""
    batch, heads, seq_len, _ = q.shape
    freqs = freqs[:, -seq_len:]
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    _rope_kernel[lambda meta: (triton.cdiv(q.numel(), meta["BLOCK"]),)](
        q,
        k,
        freqs,
        q_out,
        k_out,
        q.numel(),
        seq_len,
        heads,
        0 if freqs.shape[0] == 1 else freqs.stride(0),
        freqs.stride(1),
        enable_fp_fusion=False,
    )
    return q_out, k_out
