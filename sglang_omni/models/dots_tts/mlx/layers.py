# SPDX-License-Identifier: Apache-2.0
"""Shared MLX transformer building blocks for the dots.tts latent engine.

These mirror the torch dots_tts.modules.backbone.layers used by both the
semantic (patch) encoder and the DiT: multi-head attention with optional QK
RMSNorm and Qwen-style rotary, and the SiLU/GELU MLP. Dropout is inference-only
and omitted. The backbone Qwen2 uses its own attention in qwen.py.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention

from sglang_omni.models.dots_tts.mlx.config import TransformerConfig


class MultiHeadAttention(nn.Module):
    """Dot-product attention with optional QK norm and rotary."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        qk_norm: Optional[bool] = None,
        rotary_bias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        qk_norm = config.qk_norm if qk_norm is None else qk_norm
        rotary_bias = config.rotary_bias if rotary_bias is None else rotary_bias
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim()
        self.hidden_size = config.hidden_size
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.qkv_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.qkv_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.qkv_bias
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        eps = 1e-5 if config.norm_layer == "LayerNorm" else 1e-6
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.rope = (
            nn.RoPE(self.head_dim, traditional=False, base=config.rotary_theta)
            if rotary_bias
            else None
        )

    def __call__(
        self,
        x: mx.array,
        *,
        mask: Optional[mx.array] = None,
        offset: int = 0,
    ) -> mx.array:
        bsz, length, _ = x.shape

        queries = self.q_proj(x).reshape(bsz, length, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(bsz, length, self.num_heads, self.head_dim)
        values = self.v_proj(x).reshape(bsz, length, self.num_heads, self.head_dim)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)
        if self.rope is not None:
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
        queries = queries * self.scale

        # note (guozhihao-224): mx.fast SDPA does not broadcast the batch dim;
        # expand the (1, 1, L, S) mask for the DiT cond/uncond batch.
        if mask is not None and mask.shape[:2] != (bsz, self.num_heads):
            mask = mx.broadcast_to(mask, (bsz, self.num_heads, length, mask.shape[-1]))

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            mask=mask,
            scale=1.0,
            cache=None,
        )
        output = output.transpose(0, 2, 1, 3).reshape(bsz, length, self.hidden_size)
        return self.o_proj(output)


class Mlp(nn.Module):
    """Two-layer MLP with GELU-tanh (DiT) or SiLU (semantic encoder) act."""

    def __init__(self, config: TransformerConfig, *, activation: str = "silu") -> None:
        super().__init__()
        if activation not in {"silu", "gelu_tanh"}:
            raise ValueError(f"unsupported MLP activation {activation!r}")
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=True)
        self.fc2 = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=True)
        self._activation = activation

    @staticmethod
    def _gelu_tanh(x: mx.array) -> mx.array:
        inner = x * (2 / math.pi) ** 0.5 * (1.0 + 0.044715 * x * x)
        return 0.5 * x * (1.0 + mx.tanh(inner))

    def __call__(self, x: mx.array) -> mx.array:
        h = self.fc1(x)
        if self._activation == "silu":
            h = nn.silu(h)
        else:
            h = self._gelu_tanh(h)
        return self.fc2(h)


__all__ = ["Mlp", "MultiHeadAttention"]
