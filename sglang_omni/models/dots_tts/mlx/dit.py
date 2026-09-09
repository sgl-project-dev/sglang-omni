# SPDX-License-Identifier: Apache-2.0
"""MLX port of the dots.tts flow-matching DiT.

Mirrors dots_tts.modules.backbone.dit: a modulated transformer whose
per-layer adaLN shift/scale/gate is produced from the timestep (+ speaker
condition g_cond), ending in a final normalized projection back to the
latent space. Used as the velocity field of the flow ODE.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from sglang_omni.models.dots_tts.mlx.config import TransformerConfig
from sglang_omni.models.dots_tts.mlx.layers import Mlp, MultiHeadAttention


class _GroupNormFree(nn.Module):
    """Layer normalization without affine parameters (elementwise_affine=False)."""

    def __init__(self, dims: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.dims = dims
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean(x * x, axis=-1, keepdims=True) - mean * mean
        return (x - mean) * mx.rsqrt(var + self.eps)


def _timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None, :]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class TimestepEmbedder(nn.Module):
    """Timestep -> hidden-size condition via a small SiLU MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        # note (guozhihao-224): a plain list keeps checkpoint parameter names
        # (nn.Sequential would nest a layers. prefix).
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        ]
        self.frequency_embedding_size = frequency_embedding_size

    def __call__(self, t: mx.array) -> mx.array:
        x = _timestep_embedding(t, self.frequency_embedding_size)
        for layer in self.mlp:
            x = layer(x)
        return x


class FinalLayer(nn.Module):
    """Modulated normalization then a linear back to the output dim."""

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        ]
        self.norm = _GroupNormFree(hidden_size, eps=1e-5)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        cond = c
        for layer in self.adaLN_modulation:
            cond = layer(cond)
        shift, scale = mx.split(cond, 2, axis=-1)
        x = x * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return self.linear(self.norm(x))


class DiTBlock(nn.Module):
    """Modulated transformer block (attention + GELU MLP)."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm1 = _GroupNormFree(config.hidden_size, eps=1e-5)
        self.norm2 = _GroupNormFree(config.hidden_size, eps=1e-5)
        self.attn = MultiHeadAttention(config)
        self.ffn = Mlp(config, activation="gelu_tanh")
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        ]

    def __call__(
        self,
        x: mx.array,
        c: mx.array,
        *,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        cond = c
        for layer in self.adaLN_modulation:
            cond = layer(cond)
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_ffn,
            scale_ffn,
            gate_ffn,
        ) = mx.split(cond, 6, axis=-1)
        gate_attn = gate_attn[:, None, :]
        gate_ffn = gate_ffn[:, None, :]

        h = self.norm1(x)
        h = h * (1.0 + scale_attn[:, None, :]) + shift_attn[:, None, :]
        x = x + gate_attn * self.attn(h, mask=mask)

        h = self.norm2(x)
        h = h * (1.0 + scale_ffn[:, None, :]) + shift_ffn[:, None, :]
        return x + gate_ffn * self.ffn(h)


class DiT(nn.Module):
    """Flow-matching velocity field over latent patches."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: TransformerConfig,
    ) -> None:
        super().__init__()
        if not config.modulation:
            raise ValueError("dots.tts DiT requires modulation=True")
        self.input_layer = nn.Linear(in_dim, config.hidden_size, bias=True)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.blocks = [DiTBlock(config) for _ in range(config.num_layers)]
        self.output_layer = FinalLayer(config.hidden_size, out_dim)

    def __call__(
        self,
        x: mx.array,
        timesteps: mx.array,
        *,
        g_cond: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        c = self.time_embedder(timesteps)
        if g_cond is not None:
            c = c + g_cond
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x, c, mask=mask)
        return self.output_layer(x, c)


__all__ = ["DiT", "TimestepEmbedder"]
