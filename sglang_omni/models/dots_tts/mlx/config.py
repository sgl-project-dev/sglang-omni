# SPDX-License-Identifier: Apache-2.0
"""dots.tts MLX latent engine configuration.

Parses the dots checkpoint's top-level config.json and llm_config.json
into the fields the MLX latent engine needs. The torch path keeps the vendor
ModelConfig object around; the MLX engine re-homes this process and should
not import dots_tts (which pulls torch imports), so the configs are
lightweight dataclasses built from the raw JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"dots.tts config is missing {key!r}")
    return mapping[key]


@dataclass(frozen=True)
class TransformerConfig:
    """A shared attention/FFN config for the semantic encoder or the DiT."""

    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_hidden_size: int
    qkv_bias: bool = False
    qk_norm: bool = True
    rotary_bias: bool = True
    rotary_theta: float = 10000.0
    modulation: bool = False
    norm_layer: str = "RMSNorm"

    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class BackboneConfig:
    """Qwen2 backbone decoder fields from llm_config.json."""

    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    intermediate_size: int
    rope_theta: float
    rms_norm_eps: float

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "BackboneConfig":
        if raw.get("model_type") != "qwen2":
            raise ValueError(
                f"dots.tts llm_config must be a qwen2 config, got {raw.get('model_type')!r}"
            )
        hidden_size = int(_required(raw, "hidden_size"))
        num_heads = int(_required(raw, "num_attention_heads"))
        head_dim = int(raw.get("head_dim") or hidden_size // num_heads)
        return cls(
            hidden_size=hidden_size,
            num_layers=int(_required(raw, "num_hidden_layers")),
            num_attention_heads=num_heads,
            num_key_value_heads=int(_required(raw, "num_key_value_heads")),
            head_dim=head_dim,
            vocab_size=int(_required(raw, "vocab_size")),
            intermediate_size=int(_required(raw, "intermediate_size")),
            rope_theta=float(_required(raw, "rope_theta")),
            rms_norm_eps=float(_required(raw, "rms_norm_eps")),
        )


@dataclass(frozen=True)
class FlowHeadConfig:
    """Continuous-latent flow head fields from config.json."""

    latent_dim: int
    patch_size: int
    campplus_embedding_size: int
    xvec_max_audio_seconds: float
    patch_encoder: TransformerConfig
    dit: TransformerConfig
    flow_matching: bool
    fm_sigma: float

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "FlowHeadConfig":
        patch_encoder = _transformer(_required(raw, "PatchEncoder"))
        dit = _transformer(_required(raw, "DiT"))
        meanflow = raw.get("meanflow")
        flow_matching = meanflow is None or not bool(meanflow.get("enabled", False))
        return cls(
            latent_dim=int(_required(raw, "latent_dim")),
            patch_size=int(_required(raw, "patch_size")),
            campplus_embedding_size=int(_required(raw, "campplus_embedding_size")),
            xvec_max_audio_seconds=float(_required(raw, "xvec_max_audio_seconds")),
            patch_encoder=patch_encoder,
            dit=dit,
            flow_matching=flow_matching,
            fm_sigma=float(raw.get("fm_sigma", 0.0)),
        )


def _transformer(mapping: dict[str, Any]) -> TransformerConfig:
    return TransformerConfig(
        hidden_size=int(_required(mapping, "hidden_size")),
        num_layers=int(_required(mapping, "num_layers")),
        num_heads=int(_required(mapping, "num_heads")),
        ffn_hidden_size=int(_required(mapping, "ffn_hidden_size")),
        qkv_bias=bool(mapping.get("qkv_bias", False)),
        qk_norm=bool(mapping.get("qk_norm", False)),
        rotary_bias=bool(mapping.get("rotary_bias", False)),
        rotary_theta=float(mapping.get("rotary_theta", 10000.0)),
        modulation=bool(mapping.get("modulation", False)),
        norm_layer=str(mapping.get("norm_layer", "RMSNorm")),
    )


@dataclass(frozen=True)
class DotsMlxArgs:
    """Single config object the MLX engine constructs from.

    from_dict consumes the merged checkpoint config (dots config.json
    fields for the flow head + llm_config.json fields for the backbone),
    which is what mlx_lm.utils.load_model hands to ModelArgs.from_dict.
    """

    backbone: BackboneConfig
    flow: FlowHeadConfig
    latent_stats_path: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DotsMlxArgs":
        return cls(
            backbone=BackboneConfig.from_json(raw),
            flow=FlowHeadConfig.from_config(raw),
            latent_stats_path=str(_required(raw, "latent_stats_path")),
        )


__all__ = [
    "BackboneConfig",
    "DotsMlxArgs",
    "FlowHeadConfig",
    "TransformerConfig",
]
