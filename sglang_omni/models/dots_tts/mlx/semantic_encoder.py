# SPDX-License-Identifier: Apache-2.0
"""MLX port of the dots.tts semantic (patch) encoder.

Mirrors dots_tts.modules.backbone.encoder (VAESemanticEncoder with its
SuperviseEncoder trunk). The torch path decodes patches incrementally with a KV
cache; the MLX engine recomputes the causal encoding over the accumulated patch
history, which yields the identical rows for the newest tokens (the causal mask
is triangular) at O(n^2) cost until an incremental port lands.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from sglang_omni.models.dots_tts.mlx.config import FlowHeadConfig, TransformerConfig
from sglang_omni.models.dots_tts.mlx.layers import Mlp, MultiHeadAttention


class _SuperviseEncoder(nn.Module):
    """Stack of transformer layers for the patch encoder."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = [_FreedomLayer(config) for _ in range(config.num_layers)]

    def __call__(
        self,
        x: mx.array,
        *,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class _FreedomLayer(nn.Module):
    """Pre-norm transformer layer (attention + SiLU MLP)."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        eps = 1e-5 if config.norm_layer == "LayerNorm" else 1e-6
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=eps)
        # note (guozhihao-224): the torch semantic encoder uses no qk_norm or
        # rotary; the checkpoint has no such weights.
        self.attn = MultiHeadAttention(config, qk_norm=False, rotary_bias=False)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.ffn = Mlp(config, activation="silu")

    def __call__(
        self,
        x: mx.array,
        *,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.attn_norm(x)
        x = x + self.attn(h, mask=mask)
        h = self.ffn_norm(x)
        return x + self.ffn(h)


class VAESemanticEncoder(nn.Module):
    """Patch encoder: causal conv downsample, transformer trunk, linear out."""

    def __init__(self, config: FlowHeadConfig, out_dim: int) -> None:
        super().__init__()
        patch_encoder = config.patch_encoder
        in_ds_rate = 2
        out_ds_rate = config.patch_size // in_ds_rate
        self.out_ds_rate = out_ds_rate
        if out_ds_rate != in_ds_rate:
            raise RuntimeError(
                "expected patch_size 4 for the MLX semantic encoder, got "
                f"{config.patch_size}"
            )

        self.ds_proj = nn.Conv1d(
            config.latent_dim,
            config.latent_dim,
            kernel_size=in_ds_rate,
            stride=in_ds_rate,
            bias=True,
        )
        self.in_proj = nn.Linear(
            config.latent_dim, patch_encoder.hidden_size, bias=True
        )
        self.encoder = _SuperviseEncoder(patch_encoder)
        self.out_proj = nn.Linear(
            patch_encoder.hidden_size * out_ds_rate, out_dim, bias=True
        )

    def downsample(self, x: mx.array) -> mx.array:
        """Causal stride-2 conv over the time axis (channels last)."""
        left_pad = mx.pad(x, ((0, 0), (1, 0), (0, 0)))
        return self.ds_proj(left_pad)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Encode downsampled patch tokens to LLM-size embeddings."""
        return self.in_proj_and_trunk(self.downsample(x), mask=mask)

    def in_proj_and_trunk(
        self,
        tokens: mx.array,
        *,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Project + transformer trunk + LLM projection over downsampled tokens."""
        h = self.in_proj(tokens)
        h = self.encoder(h, mask=mask)
        return self._project(h)

    def _project(self, z: mx.array) -> mx.array:
        bsz, length, hidden = z.shape
        d = self.out_ds_rate
        z = z.reshape(bsz, length // d, d * hidden)
        return self.out_proj(z)


__all__ = ["VAESemanticEncoder"]
