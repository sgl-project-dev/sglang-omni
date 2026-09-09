# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Tencent. All rights reserved.
# Derived from Tencent-Hunyuan/AuK; see LICENSE for the MIT permission notice.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.auk.dit import AuKDit


def fuse_hidden_states(hidden_states, layer_weights, layer_scale):
    d_llm = hidden_states.shape[-1]
    stacked = torch.stack(
        [F.layer_norm(h, [d_llm]) for h in hidden_states[:, 1:].unbind(1)], dim=1
    )
    weights = F.softmax(layer_weights, dim=0)
    return (stacked * weights[None, :, None, None]).sum(dim=1) * layer_scale


def build_time_grid(
    steps: int,
    sway_sampling_coef: float | None = None,
    t_grid: Sequence[float] | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if t_grid is not None:
        grid = torch.tensor(list(t_grid), device=device, dtype=torch.float32)
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError("t_grid must hold at least two time points")
        return grid
    if steps < 1:
        raise ValueError("AuK nfe must be positive")
    t = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)
    if sway_sampling_coef is not None:
        t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t


@dataclass
class AuKSampleItem:
    hidden_states: torch.Tensor  # [L, Nt, H], including the embedding layer
    text_mask: torch.Tensor  # [Nt]
    target_frames: int
    ref_latent: torch.Tensor | None = None  # [Np, D], including padded frames
    seed: int | None = None
    ref_length: int = 0


class AuKFlowMatching(nn.Module):
    """Serial velocity-field integration for AuK latent generation."""

    def __init__(self, transformer: AuKDit, num_llm_layers: int):
        super().__init__()
        self.transformer = transformer
        self.layer_weights = nn.Parameter(torch.zeros(num_llm_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))

    def fuse(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return fuse_hidden_states(hidden_states, self.layer_weights, self.layer_scale)

    @torch.no_grad()
    def sample(
        self,
        item: AuKSampleItem,
        *,
        steps: int,
        cfg_strength: float,
        sway_sampling_coef: float | None = None,
        t_grid: Sequence[float] | None = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        ref = (
            item.ref_latent.unsqueeze(0)
            if item.ref_latent is not None
            else torch.zeros(1, 0, self.transformer.latent_dim, device=device)
        )
        ref_mask = torch.arange(ref.shape[1], device=device)[None, :] < item.ref_length
        text = self.fuse(item.hidden_states.unsqueeze(0))
        text_mask = item.text_mask.unsqueeze(0)
        if item.seed is not None:
            torch.manual_seed(item.seed)
        y0 = torch.randn(
            item.target_frames,
            self.transformer.latent_dim,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)

        def fn(t, x):
            kwargs = dict(
                x=x,
                text=text,
                time=t,
                mask=None,
                c_mask=text_mask,
                ref=ref,
                ref_mask=ref_mask,
                cache=True,
            )
            if cfg_strength < 1e-5:
                return self.transformer(
                    **kwargs, drop_audio_cond=False, drop_text=False
                )
            pred = self.transformer(**kwargs, cfg_infer=True)
            v_cond, v_uncond = torch.chunk(pred, 2, dim=0)
            return v_cond + (v_cond - v_uncond) * cfg_strength

        t = build_time_grid(steps, sway_sampling_coef, t_grid, device=device)
        try:
            return integrate(fn, y0, t)[0]
        finally:
            self.transformer.clear_cache()


def integrate(fn, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Fixed-grid Euler integration, matching the released inference recipe."""
    y = y0
    for step in range(t.numel() - 1):
        y = y + (t[step + 1] - t[step]) * fn(t[step], y)
    return y
