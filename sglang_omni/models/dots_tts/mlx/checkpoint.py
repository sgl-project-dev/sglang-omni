# SPDX-License-Identifier: Apache-2.0
"""dots.tts MLX model loading via the standard mlx_lm pipeline.

dots splits its config across config.json (flow head) and llm_config.json
(backbone), so the two files are merged before mlx_lm.utils.load_model —
which then discovers model*.safetensors, calls the model's sanitize,
and loads weights, exactly like the other MLX engines in this repo.
"""

from __future__ import annotations

import json
from pathlib import Path

from sglang_omni.models.dots_tts.mlx.config import DotsMlxArgs
from sglang_omni.models.dots_tts.mlx.model import DotsTTSMlxModel


def load_dots_mlx_model(checkpoint_dir: str) -> DotsTTSMlxModel:
    """Load the dots.tts MLX latent engine from a local checkpoint directory."""
    from mlx_lm.utils import load_model

    root = Path(checkpoint_dir)
    config = json.loads((root / "config.json").read_text())
    config.update(json.loads((root / "llm_config.json").read_text()))
    config["latent_stats_path"] = str(root / "latent_stats.pt")
    model, _ = load_model(
        root,
        get_model_classes=lambda config: (DotsTTSMlxModel, DotsMlxArgs),
        model_config=config,
        strict=True,
    )
    return model


__all__ = ["load_dots_mlx_model"]
