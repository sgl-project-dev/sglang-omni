# SPDX-License-Identifier: Apache-2.0
"""Inference-only SnakeBeta data path with explicit eager BF16 round points.

Constants must already contain the original PyTorch BF16 expressions
``exp(alpha)`` and ``1.0 / (exp(beta) + eps)``. This module does not change
parameters, convolution geometry, or the caller's stream. Numerical equality
with a given CUDA/PyTorch runtime must be qualified before enabling it.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _get_triton_kernel():
    # Note (wenyao): the default model path must not import or initialize Triton.
    global _tl, _libdevice

    import triton
    import triton.language as _tl
    from triton.language.extra.cuda import libdevice as _libdevice

    @triton.jit
    def _kernel(
        x_ptr,
        alpha_ptr,
        inverse_ptr,
        output_ptr,
        numel,
        channels,
        time_steps,
        BLOCK: _tl.constexpr,
    ):
        offsets = _tl.program_id(0) * BLOCK + _tl.arange(0, BLOCK)
        mask = offsets < numel
        channel = (offsets // time_steps) % channels
        x = _tl.load(x_ptr + offsets, mask=mask, other=0).to(_tl.float32)
        alpha = _tl.load(alpha_ptr + channel, mask=mask, other=0).to(_tl.float32)
        inverse = _tl.load(inverse_ptr + channel, mask=mask, other=0).to(_tl.float32)

        # Note (wenyao): every eager BF16 intermediate rounds before the next op.
        scaled = (
            (x * alpha).to(_tl.bfloat16, fp_downcast_rounding="rtne").to(_tl.float32)
        )
        sine = (
            _libdevice.sin(scaled)
            .to(_tl.bfloat16, fp_downcast_rounding="rtne")
            .to(_tl.float32)
        )
        squared = (
            (sine * sine).to(_tl.bfloat16, fp_downcast_rounding="rtne").to(_tl.float32)
        )
        periodic = (
            (inverse * squared)
            .to(_tl.bfloat16, fp_downcast_rounding="rtne")
            .to(_tl.float32)
        )
        result = (x + periodic).to(_tl.bfloat16, fp_downcast_rounding="rtne")
        _tl.store(output_ptr + offsets, result, mask=mask)

    return _kernel


def fused_snake_beta(
    x: torch.Tensor,
    exp_alpha: torch.Tensor,
    inv_beta_safe: torch.Tensor,
) -> torch.Tensor:
    """Return a fresh contiguous NCT output; reject unsupported inputs explicitly."""
    values = (x, exp_alpha, inv_beta_safe)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("fused SnakeBeta expects three tensors")
    if torch.version.hip is not None or not all(value.is_cuda for value in values):
        raise ValueError("fused SnakeBeta requires NVIDIA CUDA tensors")
    if any(value.dtype != torch.bfloat16 for value in values):
        raise ValueError("fused SnakeBeta requires BF16 inputs and constants")
    if any(value.device != x.device for value in values):
        raise ValueError("fused SnakeBeta tensors must share one CUDA device")
    if x.ndim != 3 or any(size <= 0 for size in x.shape):
        raise ValueError("fused SnakeBeta requires a nonempty NCT tensor")
    expected = (1, x.shape[1], 1)
    if exp_alpha.shape != expected or inv_beta_safe.shape != expected:
        raise ValueError("fused SnakeBeta constants must have shape [1, C, 1]")
    if any(
        value.layout != torch.strided or not value.is_contiguous() for value in values
    ):
        raise ValueError("fused SnakeBeta requires contiguous strided tensors")
    if x.numel() > 2**31 - 1:
        raise ValueError("fused SnakeBeta exceeds the supported indexing range")
    if torch.is_grad_enabled() and any(value.requires_grad for value in values):
        raise RuntimeError("fused SnakeBeta does not implement autograd")

    kernel = _get_triton_kernel()
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    numel = x.numel()
    with torch.cuda.device(x.device):
        kernel[((numel + 255) // 256,)](
            x,
            exp_alpha,
            inv_beta_safe,
            output,
            numel,
            x.shape[1],
            x.shape[2],
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
