# SPDX-License-Identifier: Apache-2.0
"""MLX Qwen2 backbone for the dots.tts latent engine.

Ported from the Qwen3 text decoder in qwen3_asr/mlx/model.py (same repo),
stripped to the Qwen2 configuration: no QK norms, grouped-query attention
(GQA handled natively by mlx fast SDPA), and the mlx_lm KVCache the runner
advances one feedback step at a time.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache

from sglang_omni.models.dots_tts.mlx.config import BackboneConfig


class Qwen2Attention(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # note (guozhihao-224): mx.fast SDPA handles GQA natively; do not
        # expand K/V by hand.
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2Mlp(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        *,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask=mask, cache=cache)
        x = residual + h
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Mlp(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Model(nn.Module):
    """Qwen2 decoder over token ids or (feedback) input embeddings."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen2DecoderLayer(config) for _ in range(config.num_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[list[Optional[KVCache]]] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(inputs_embeds, cache[0])

        hidden = inputs_embeds
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        return self.norm(hidden)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]


__all__ = ["Qwen2Model"]
