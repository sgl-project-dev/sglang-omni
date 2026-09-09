# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.auk.dit import AuKDit


def fuse_hidden_states(
    hidden_states: torch.Tensor,
    layer_weights: torch.Tensor,
    layer_scale: torch.Tensor,
) -> torch.Tensor:
    d_llm = hidden_states.shape[-1]
    stacked = torch.stack(
        [F.layer_norm(h, [d_llm]) for h in hidden_states[:, 1:].unbind(1)], dim=1
    )
    weights = F.softmax(layer_weights, dim=0).to(stacked.dtype)
    return (stacked * weights[None, :, None, None]).sum(dim=1) * layer_scale.to(
        stacked.dtype
    )


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

    t = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)
    if sway_sampling_coef is not None:
        t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t


def lens_to_mask(lens: torch.Tensor, length: int) -> torch.Tensor:
    return torch.arange(length, device=lens.device)[None, :] < lens[:, None]


@dataclass
class AuKSampleItem:
    """One request handed to :meth:`AuKFlowMatching.sample_batch`."""

    hidden_states: torch.Tensor  # [L, Nt, H] all LLM layers, unpadded
    text_mask: torch.Tensor  # [Nt] bool
    target_frames: int
    ref_latent: torch.Tensor | None = None  # [Np, D]
    seed: int | None = None


class AuKFlowMatching(nn.Module):
    """Velocity field + ODE integration for AuK latent generation."""

    def __init__(self, transformer: AuKDit, num_llm_layers: int):
        super().__init__()
        self.transformer = transformer
        self.num_llm_layers = int(num_llm_layers)
        self.layer_weights = nn.Parameter(torch.zeros(self.num_llm_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))

    # -- conditioning ------------------------------------------------------- #

    def fuse(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fuse a packed ``[B, L, Nt, H]`` stack of LLM hidden states."""
        return fuse_hidden_states(hidden_states, self.layer_weights, self.layer_scale)

    # -- sampling ----------------------------------------------------------- #

    @torch.no_grad()
    def sample_batch(
        self,
        items: Sequence[AuKSampleItem],
        *,
        steps: int,
        cfg_strength: float,
        sway_sampling_coef: float | None = None,
        t_grid: Sequence[float] | None = None,
        method: str = "euler",
        max_duration: int = 65536,
    ) -> list[torch.Tensor]:
        """Integrate the flow for a batch, returning one [T, D] latent each."""
        if not items:
            return []

        device = next(self.parameters()).device
        dtype = torch.float32

        target_frames = torch.tensor(
            [max(int(item.target_frames), 1) for item in items],
            device=device,
            dtype=torch.long,
        ).clamp(max=max_duration)

        # --- reference (prompt) latents ---
        has_ref = any(item.ref_latent is not None for item in items)
        if has_ref:
            ref_lens = torch.tensor(
                [
                    int(item.ref_latent.shape[0]) if item.ref_latent is not None else 0
                    for item in items
                ],
                device=device,
                dtype=torch.long,
            )
            max_ref = max(
                int(item.ref_latent.shape[0])
                for item in items
                if item.ref_latent is not None
            )
            ref_latents = torch.zeros(
                len(items),
                max_ref,
                self.transformer.latent_dim,
                device=device,
                dtype=dtype,
            )
            for index, item in enumerate(items):
                if item.ref_latent is not None:
                    ref_latents[index, : item.ref_latent.shape[0]] = item.ref_latent.to(
                        device=device, dtype=dtype
                    )
            ref_mask = lens_to_mask(ref_lens, max_ref)
        else:
            ref_lens = torch.zeros(len(items), device=device, dtype=torch.long)
            ref_latents = torch.zeros(
                len(items), 0, self.transformer.latent_dim, device=device, dtype=dtype
            )
            ref_mask = torch.zeros(len(items), 0, dtype=torch.bool, device=device)

        # --- text conditioning ---
        text_lens = torch.tensor(
            [int(item.hidden_states.shape[1]) for item in items],
            device=device,
            dtype=torch.long,
        )
        max_text = int(text_lens.max().item())
        num_layers = int(items[0].hidden_states.shape[0])
        hidden_dim = int(items[0].hidden_states.shape[-1])
        hidden = torch.zeros(
            len(items), num_layers, max_text, hidden_dim, device=device, dtype=dtype
        )
        text_mask = torch.zeros(len(items), max_text, dtype=torch.bool, device=device)
        for index, item in enumerate(items):
            n = int(item.hidden_states.shape[1])
            hidden[index, :, :n] = item.hidden_states.to(device=device, dtype=dtype)
            text_mask[index, :n] = item.text_mask.to(device=device, dtype=torch.bool)

        text_embeds = self.fuse(hidden)  # [B, Nt, H]

        # --- initial noise (target-only ODE state) ---
        max_target = int(target_frames.max().item())
        if any(item.seed is not None for item in items):
            y0 = torch.zeros(
                len(items),
                max_target,
                self.transformer.latent_dim,
                device=device,
                dtype=dtype,
            )
            for index, item in enumerate(items):
                frames = int(target_frames[index])
                generator = (
                    torch.Generator(device=device).manual_seed(int(item.seed))
                    if item.seed is not None
                    else None
                )
                y0[index, :frames] = torch.randn(
                    frames,
                    self.transformer.latent_dim,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
        else:
            # No seeds anywhere: a single batched draw from the global RNG.
            y0 = torch.randn(
                len(items),
                max_target,
                self.transformer.latent_dim,
                device=device,
                dtype=dtype,
            )
        for index in range(len(items)):
            y0[index, int(target_frames[index]) :] = 0.0

        target_mask = lens_to_mask(target_frames, max_target)

        use_cfg = cfg_strength >= 1e-5

        def fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            if not use_cfg:
                return self.transformer(
                    x=x,
                    text=text_embeds,
                    time=t,
                    mask=target_mask,
                    c_mask=text_mask,
                    ref=ref_latents,
                    ref_mask=ref_mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
            pred = self.transformer(
                x=x,
                text=text_embeds,
                time=t,
                mask=target_mask,
                c_mask=text_mask,
                ref=ref_latents,
                ref_mask=ref_mask,
                cfg_infer=True,
                cache=True,
            )
            v_cond, v_uncond = torch.chunk(pred, 2, dim=0)
            return v_cond + (v_cond - v_uncond) * cfg_strength

        t = build_time_grid(steps, sway_sampling_coef, t_grid, device=device)
        try:
            y = integrate(fn, y0, t, method=method)
        finally:
            self.transformer.clear_cache()

        return [y[i, : int(target_frames[i])] for i in range(len(items))]


def integrate(
    fn,
    y0: torch.Tensor,
    t: torch.Tensor,
    *,
    method: str = "euler",
) -> torch.Tensor:
    """Fixed-step ODE integration (equivalent to torchdiffeq on a fixed grid)."""
    if method not in ("euler", "midpoint"):
        raise ValueError(f"Unsupported ODE method: {method!r}")
    y = y0
    for step in range(t.numel() - 1):
        dt = t[step + 1] - t[step]
        if method == "euler":
            derivative = fn(t[step], y)
        else:
            k1 = fn(t[step], y)
            derivative = fn(t[step] + dt / 2, y + (dt / 2) * k1)
        y = y + dt * derivative
    return y
