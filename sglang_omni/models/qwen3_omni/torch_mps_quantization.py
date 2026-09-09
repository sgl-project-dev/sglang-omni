# SPDX-License-Identifier: Apache-2.0
"""Native ATen MPS INT4/INT8 inference, without a dense weight cache.

Packing and affine checkpoint decoding use CPU Torch only. Forward uses the
native Metal kernels, never a CPU fallback or a materialized dense weight.
Float32 accumulations also avoid rounding an imported affine bias twice.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

_GROUP_SIZES = (32, 64, 128, 256)
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_MAX_LINEAR_ROWS = 4096


def _validate_affine(weight, scales, biases, bits, group_size):
    if bits not in (4, 8) or group_size not in _GROUP_SIZES:
        raise ValueError(
            "Affine quantization requires bits 4/8 and group_size 32/64/128/256"
        )
    if weight.ndim != 2 or weight.dtype != torch.uint32:
        raise ValueError("Affine weight must be a 2D packed uint32 tensor")
    if scales.dtype not in _DTYPES or biases.dtype != scales.dtype:
        raise ValueError("Affine scales/biases must have the same floating dtype")
    width = weight.shape[1] * (32 // bits)
    expected = (weight.shape[0], width // group_size)
    if width == 0 or width % group_size or tuple(scales.shape) != expected:
        raise ValueError(
            f"Affine scales shape must be {expected}, got {tuple(scales.shape)}"
        )
    if biases.shape != scales.shape:
        raise ValueError("Affine biases shape must match scales")
    if any(t.device.type != "cpu" for t in (weight, scales, biases)):
        raise ValueError("Decode affine checkpoint rows on CPU before device transfer")
    if not torch.isfinite(scales).all() or not torch.isfinite(biases).all():
        raise ValueError("Affine scales and biases must be finite")


def _affine_codes(weight: torch.Tensor, bits: int) -> torch.Tensor:
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)
    return (
        ((weight.to(torch.int64).unsqueeze(-1) >> shifts) & ((1 << bits) - 1))
        .reshape(weight.shape[0], -1)
        .to(torch.uint8)
    )


def dequantize_affine_rows(
    weight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """Decode a bounded row slice of MLX affine uint32 weights using pure Torch.

    MLX packs least-significant codes first and computes ``q * scale + bias``.
    Perform the arithmetic in float32 before the single cast to the source
    scales dtype (in particular, do not multiply in BF16 then add in BF16).
    """
    _validate_affine(weight, scales, biases, bits, group_size)
    if weight.shape[0] == 0:
        return scales.new_empty((0, weight.shape[1] * (32 // bits)))
    codes = _affine_codes(weight, bits).reshape(*scales.shape, group_size).float()
    return (
        torch.addcmul(biases.float().unsqueeze(-1), codes, scales.float().unsqueeze(-1))
        .reshape(weight.shape[0], -1)
        .to(scales.dtype)
    )


def _round_up(size, alignment):
    return math.ceil(size / alignment) * alignment


class MpsQuantizedLinear(nn.Module):
    """Packed native MPS linear; intentionally has no dense ``weight`` property."""

    def __init__(
        self,
        in_features,
        out_features,
        bits,
        group_size,
        packed_weight,
        scales_and_zeros,
        bias=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.padded_in_features = _round_up(in_features, max(128, group_size))
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scales_and_zeros", scales_and_zeros)
        self.register_buffer("bias", bias)

    @classmethod
    def _combine_rows(cls, parts, in_features, out_features):
        first = parts[0]
        return cls(
            in_features,
            out_features,
            first.bits,
            first.group_size,
            torch.cat([part.packed_weight for part in parts], dim=0),
            torch.cat(
                [part.scales_and_zeros for part in parts],
                dim=1 if first.bits == 4 else 0,
            ),
            None if first.bias is None else torch.cat([part.bias for part in parts]),
        )

    @staticmethod
    def _validate_target(bits, dtype, device):
        if bits not in (4, 8):
            raise ValueError("Native MPS quantization supports only bits=4 or bits=8")
        if dtype not in _DTYPES:
            raise ValueError(
                "Native MPS activations require float32, float16 or bfloat16"
            )
        if torch.device(device).type != "mps":
            raise ValueError("Native packed quantization requires an MPS device")
        operators = (
            ("_convert_weight_to_int4pack", "_weight_int4pack_mm")
            if bits == 4
            else ("_weight_int8pack_mm",)
        )
        for operator in operators:
            if not hasattr(torch.ops.aten, operator) or not (
                torch._C._dispatch_has_kernel_for_dispatch_key(
                    f"aten::{operator}", "MPS"
                )
            ):
                raise RuntimeError(
                    f"Native MPS INT{bits} requires a PyTorch build with "
                    f"the aten::{operator} MPS kernel; no CPU fallback is supported"
                )

    @classmethod
    def _from_codes(
        cls,
        codes,
        scales,
        zeros,
        *,
        in_features,
        out_features,
        bits,
        group_size,
        bias,
        device,
        dtype,
    ):
        cls._validate_target(bits, dtype, device)
        if bits == 4:
            byte_pairs = (codes[:, 0::2] << 4) | codes[:, 1::2]
            packed = torch.ops.aten._convert_weight_to_int4pack(
                byte_pairs.contiguous().to(device), 8
            )
            qparams = torch.stack((scales, zeros), dim=-1).transpose(0, 1).contiguous()
        else:
            packed = codes.contiguous().to(device)
            qparams = scales
        return cls(
            in_features,
            out_features,
            bits,
            group_size,
            packed,
            qparams.to(device=device, dtype=torch.float32),
            None if bias is None else bias.to(device=device, dtype=dtype),
        )

    @classmethod
    def from_float(
        cls,
        weight,
        *,
        bias=None,
        bits=4,
        group_size=64,
        device="mps",
        dtype=torch.float32,
    ):
        cls._validate_target(bits, dtype, device)
        if group_size not in _GROUP_SIZES:
            raise ValueError("INT4 group_size must be 32, 64, 128 or 256")
        if weight.ndim != 2 or min(weight.shape) < 1 or not weight.is_floating_point():
            raise ValueError("Linear weight must be a nonempty floating matrix")
        n, k = weight.shape
        if bias is not None and tuple(bias.shape) != (n,):
            raise ValueError(f"Linear bias must have shape {(n,)}")
        if n > _MAX_LINEAR_ROWS:
            # Vocabulary heads must not create multi-GB float/unpacking scratch.
            return cls._combine_rows(
                [
                    cls.from_float(
                        weight[start : start + _MAX_LINEAR_ROWS],
                        bias=(
                            None
                            if bias is None
                            else bias[start : start + _MAX_LINEAR_ROWS]
                        ),
                        bits=bits,
                        group_size=group_size,
                        device=device,
                        dtype=dtype,
                    )
                    for start in range(0, n, _MAX_LINEAR_ROWS)
                ],
                k,
                n,
            )
        weight = weight.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(weight).all():
            raise ValueError("Cannot quantize nonfinite linear weights")
        pn, pk = _round_up(n, 32 if bits == 8 else 8), _round_up(
            k, max(128, group_size)
        )
        padded = F.pad(weight, (0, pk - k, 0, pn - n))
        if bits == 8:
            scales = padded.abs().amax(dim=1) / 127
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            codes = (padded / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
            zeros = None
        else:
            grouped = padded.reshape(pn, -1, group_size)
            lower = grouped.amin(dim=-1)
            scales = (grouped.amax(dim=-1) - lower) / 15
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            codes = ((grouped - lower[..., None]) / scales[..., None]).round()
            codes = codes.clamp(0, 15).to(torch.uint8).reshape(pn, pk)
            zeros = lower + 8 * scales
        return cls._from_codes(
            codes,
            scales,
            zeros,
            in_features=k,
            out_features=n,
            bits=bits,
            group_size=group_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_affine(
        cls,
        weight,
        scales,
        biases,
        *,
        bits=4,
        group_size=64,
        bias=None,
        device="mps",
        dtype=torch.float32,
        target_bits=None,
    ):
        """Preserve source INT4 codes; INT8 requests requantize decoded source."""
        _validate_affine(weight, scales, biases, bits, group_size)
        target_bits = bits if target_bits is None else target_bits
        cls._validate_target(target_bits, dtype, device)
        n, k = weight.shape[0], weight.shape[1] * (32 // bits)
        if bias is not None and tuple(bias.shape) != (n,):
            raise ValueError(f"Linear bias must have shape {(n,)}")
        if n > _MAX_LINEAR_ROWS:
            return cls._combine_rows(
                [
                    cls.from_affine(
                        weight[start : start + _MAX_LINEAR_ROWS],
                        scales[start : start + _MAX_LINEAR_ROWS],
                        biases[start : start + _MAX_LINEAR_ROWS],
                        bias=(
                            None
                            if bias is None
                            else bias[start : start + _MAX_LINEAR_ROWS]
                        ),
                        bits=bits,
                        group_size=group_size,
                        target_bits=target_bits,
                        device=device,
                        dtype=dtype,
                    )
                    for start in range(0, n, _MAX_LINEAR_ROWS)
                ],
                k,
                n,
            )
        if bits != 4 or target_bits != 4:
            return cls.from_float(
                dequantize_affine_rows(
                    weight, scales, biases, bits=bits, group_size=group_size
                ),
                bias=bias,
                bits=target_bits,
                group_size=group_size,
                device=device,
                dtype=dtype,
            )
        n, k = weight.shape[0], weight.shape[1] * 8
        if n < 1 or (bias is not None and tuple(bias.shape) != (n,)):
            raise ValueError("Invalid affine linear rows or bias shape")
        pn, pk = _round_up(n, 8), _round_up(k, max(128, group_size))
        codes = F.pad(_affine_codes(weight, bits), (0, pk - k, 0, pn - n))
        padding = (0, pk // group_size - scales.shape[1], 0, pn - n)
        scales = F.pad(scales.float(), padding)
        zeros = F.pad(biases.float(), padding) + 8 * scales
        return cls._from_codes(
            codes,
            scales,
            zeros,
            in_features=k,
            out_features=n,
            bits=4,
            group_size=group_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states):
        if hidden_states.device.type != "mps" or hidden_states.dtype not in _DTYPES:
            raise ValueError("Packed linear requires floating MPS activations")
        if hidden_states.ndim < 1 or hidden_states.shape[-1] != self.in_features:
            raise ValueError(f"Expected last activation dimension {self.in_features}")
        shape = (*hidden_states.shape[:-1], self.out_features)
        if hidden_states.numel() == 0:
            return hidden_states.new_empty(shape)
        # The ATen Metal kernels interpret qparams in the activation dtype;
        # using float32 for both prevents silent reinterpretation of half data.
        x = hidden_states.reshape(-1, self.in_features).float()
        x = F.pad(x, (0, self.padded_in_features - self.in_features)).contiguous()
        qparams = self.scales_and_zeros.float()
        if self.bits == 4:
            result = torch.ops.aten._weight_int4pack_mm(
                x, self.packed_weight, self.group_size, qparams
            )
        else:
            result = torch.ops.aten._weight_int8pack_mm(x, self.packed_weight, qparams)
        result = result[:, : self.out_features]
        if self.bias is not None:
            result = result + self.bias.float()
        return result.to(hidden_states.dtype).reshape(shape)


class MpsQuantizedExperts(nn.Module):
    """Transformers 5.x routed experts, storing each projection independently."""

    def __init__(self, gate_projs, up_projs, down_projs, act_fn):
        super().__init__()
        if not (len(gate_projs) == len(up_projs) == len(down_projs)):
            raise ValueError("Expert projection counts must agree")
        self.num_experts = len(gate_projs)
        self.gate_projs = nn.ModuleList(gate_projs)
        self.up_projs = nn.ModuleList(up_projs)
        self.down_projs = nn.ModuleList(down_projs)
        self.act_fn = act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        if top_k_index.shape != top_k_weights.shape or top_k_index.ndim != 2:
            raise ValueError(
                "Expert indices and routing weights must be matching 2D tensors"
            )
        if top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError("Expert routing must have one row per token")
        # Transfer only routing IDs, not activations; inactive experts do no work.
        for expert in torch.unique(top_k_index).tolist():
            if not 0 <= expert < self.num_experts:
                raise ValueError(f"Invalid routed expert index {expert}")
            token, slot = torch.where(top_k_index == expert)
            selected = hidden_states[token]
            intermediate = self.act_fn(self.gate_projs[expert](selected))
            intermediate = intermediate * self.up_projs[expert](selected)
            current = self.down_projs[expert](intermediate)
            current = current * top_k_weights[token, slot, None]
            output.index_add_(0, token, current.to(output.dtype))
        return output
