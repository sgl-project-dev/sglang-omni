# SPDX-License-Identifier: Apache-2.0
"""MLX dots.tts latent engine model and per-request flow state machine.

Ports flow_head.py (the omni torch wrapper) and the flow-matching
full-compute DiT decode of dots_tts.modules.backbone.dit_inference. One
scheduler decode step equals one audio patch: the runner feeds the backbone the
semantic-encoder feedback embedding, runs the DiT ODE over the causally grown
flow history, and gets back the latent patch, the next feedback embedding, and
the EOS flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from sglang_omni.models.dots_tts.mlx.config import DotsMlxArgs
from sglang_omni.models.dots_tts.mlx.dit import DiT
from sglang_omni.models.dots_tts.mlx.qwen import Qwen2Model
from sglang_omni.models.dots_tts.mlx.semantic_encoder import VAESemanticEncoder


def _load_latent_stats(path: str) -> tuple[mx.array, mx.array]:
    """Global latent normalization stats (mean, var) from latent_stats.pt."""
    import torch

    stats = torch.load(path, map_location="cpu", weights_only=False)
    return mx.array(stats["mean"]), mx.array(stats["var"])


@dataclass
class DotsTTSFlowState:
    """Per-request continuous-latent AR state (MLX arrays)."""

    fm_history: list[mx.array] = field(default_factory=list)
    fm_cfg_history: list[mx.array] = field(default_factory=list)
    fm_seq_len: int = 0
    g_cond: Optional[mx.array] = None
    patch_history: list[mx.array] = field(default_factory=list)
    prompt_patches: Optional[mx.array] = None
    backbone_cache: Optional[list] = None
    rng_key: Optional[Any] = None
    decoded_patches: int = 0
    drop_regenerated_prompt_patch: bool = False
    suppress_first_eos_check: bool = False
    next_feedback: Optional[mx.array] = None


class DotsTTSMlxModel(nn.Module):
    """Backbone + patch encoder + DiT flow head loaded from the dots checkpoint."""

    def __init__(self, args: DotsMlxArgs) -> None:
        super().__init__()
        backbone_config = args.backbone
        flow_config = args.flow
        self.fm_hidden_size = flow_config.dit.hidden_size
        self.latent_patch_size = flow_config.patch_size
        self.latent_dim = flow_config.latent_dim

        self.backbone = Qwen2Model(backbone_config)
        self.patch_encoder = VAESemanticEncoder(
            flow_config, out_dim=backbone_config.hidden_size
        )
        self.hidden_proj = nn.Linear(
            backbone_config.hidden_size, self.fm_hidden_size, bias=True
        )
        self.latent_proj = nn.Linear(self.latent_dim, self.fm_hidden_size, bias=True)
        self.coordinate_proj = nn.Linear(
            self.latent_dim, self.fm_hidden_size, bias=True
        )
        # note (guozhihao-224): a plain list keeps checkpoint parameter names
        # (nn.Sequential would nest a layers. prefix).
        self.xvec_proj = [
            nn.Linear(
                flow_config.campplus_embedding_size, self.fm_hidden_size, bias=True
            ),
            nn.LayerNorm(self.fm_hidden_size),
        ]
        self.velocity_field_predictor = DiT(
            in_dim=self.fm_hidden_size,
            out_dim=self.latent_dim,
            config=flow_config.dit,
        )
        self.eos_proj = [
            nn.Linear(
                backbone_config.hidden_size, backbone_config.hidden_size, bias=True
            ),
            nn.SiLU(),
            nn.Linear(backbone_config.hidden_size, 2, bias=True),
        ]
        self._latent_mean, self._latent_var = _load_latent_stats(args.latent_stats_path)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Map checkpoint names onto this module's parameter tree.

        Follows the mlx_lm convention (mlx_lm.utils.load_model calls it
        before load_weights): the Qwen2 backbone lives under llm.model.*
        in the checkpoint and under backbone.* here, and the patch encoder
        stride-2 conv is stored torch-first [out, in, k] while mlx's Conv1d
        is channels-last [out, k, in].
        """
        sanitized = {}
        for name, tensor in weights.items():
            key = (
                "backbone." + name.removeprefix("llm.model.")
                if name.startswith("llm.model.")
                else name
            )
            if key == "patch_encoder.ds_proj.weight":
                tensor = mx.transpose(tensor, (0, 2, 1))
            sanitized[key] = tensor
        return sanitized

    def normalize(self, x: mx.array) -> mx.array:
        return (x - self._latent_mean) / mx.sqrt(self._latent_var)

    def denormalize(self, x: mx.array) -> mx.array:
        return x * mx.sqrt(self._latent_var) + self._latent_mean

    def __call__(
        self,
        embeds: mx.array,
        *,
        cache: Optional[list] = None,
    ) -> mx.array:
        """Backbone forward over input embeddings (token rows or feedback)."""
        return self.backbone(inputs_embeds=embeds, cache=cache)

    def speaker_condition(self, embedding: mx.array, scale: float) -> mx.array:
        """g_cond from the speaker embedding (xvec_proj chain)."""
        x = embedding * scale
        for layer in self.xvec_proj:
            x = layer(x)
        return x

    def eos_probability(self, hidden_last: mx.array) -> mx.array:
        """Two-class softmax EOS probability from the final hidden state."""
        x = hidden_last[:, -1, :]
        for layer in self.eos_proj:
            x = layer(x)
        return mx.softmax(x, axis=-1)[:, 1]


Model = DotsTTSMlxModel
ModelArgs = DotsMlxArgs

__all__ = ["DotsTTSFlowState", "DotsTTSMlxModel", "Model", "ModelArgs"]
