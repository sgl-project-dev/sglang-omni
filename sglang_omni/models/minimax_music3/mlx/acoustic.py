# SPDX-License-Identifier: Apache-2.0
"""Stage-facing native MLX acoustic decoder for MiniMax Music 3."""

from __future__ import annotations

import logging
from collections.abc import Callable

import mlx.core as mx
import numpy as np
import torch

from ..acoustic import _derive_seed
from ..chunking import overlap_mel_length
from ..serial_offload import get_coordinator
from .euler import denoise_chunk
from .loader import (
    MiniMaxMusic3MlxAcousticModel,
    load_mlx_acoustic_model,
    resolve_mlx_artifact,
)

logger = logging.getLogger(__name__)


class MiniMaxMusic3MlxAcousticDecoder:
    """Condition projection, Flow/DiT solve, and DAV decode on Metal."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        dit_steps: int = 30,
        dit_cfg_scale: float = 1.7,
        serial_offload: bool = False,
    ) -> None:
        if dit_steps < 1:
            raise ValueError("MiniMax Music 3 dit_steps must be positive")
        if dit_cfg_scale < 0:
            raise ValueError("MiniMax Music 3 dit_cfg_scale must be non-negative")
        self.serial_offload = bool(serial_offload)
        self._artifact = resolve_mlx_artifact(model_path, revision)
        self.model: MiniMaxMusic3MlxAcousticModel | None = None
        if not self.serial_offload:
            self.model = load_mlx_acoustic_model(
                model_path,
                revision,
                artifact=self._artifact,
            )
        self.device = "mps"
        self.dtype = (
            self.model.condition_encoder.proj.weight.dtype
            if self.model is not None
            else str(
                self._artifact[1].get("torch_dtype")
                or self._artifact[1].get("dtype")
                or "artifact"
            )
        )
        self.dit_steps = int(dit_steps)
        self.dit_cfg_scale = float(dit_cfg_scale)
        self.attention_backend = "mlx_sdpa"
        self.compile_acoustic = False
        self.cache_dit = False
        self.breakable_cuda_graph = False
        self._mlx_thread_stream = mx.new_thread_local_stream(mx.gpu)
        if self.serial_offload:
            get_coordinator().enable()

    def ensure_resident(self) -> None:
        if self.model is not None:
            return
        with mx.stream(self._mlx_thread_stream):
            self.model = load_mlx_acoustic_model(
                str(self._artifact[0]),
                artifact=self._artifact,
            )
        self.dtype = self.model.condition_encoder.proj.weight.dtype
        logger.info(
            "MiniMax Music 3 MLX acoustic loaded active=%.2fGiB cache=%.2fGiB",
            mx.get_active_memory() / 1024**3,
            mx.get_cache_memory() / 1024**3,
        )

    def release_residency(self) -> None:
        if not self.serial_offload:
            return
        mx.synchronize(self._mlx_thread_stream)
        self.model = None
        logger.info(
            "MiniMax Music 3 MLX acoustic released active=%.2fGiB cache=%.2fGiB",
            mx.get_active_memory() / 1024**3,
            mx.get_cache_memory() / 1024**3,
        )

    def decode_with_state(
        self,
        hidden: torch.Tensor,
        *,
        seed: int,
        chunk_idx: int,
        initial_latent: mx.array | None = None,
        initial_condition: mx.array | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> tuple[torch.Tensor, mx.array, mx.array]:
        if should_abort is not None and should_abort():
            raise InterruptedError("MiniMax Music 3 MLX acoustic generation aborted")
        self.ensure_resident()
        if self.model is None:
            raise RuntimeError("MiniMax Music 3 MLX acoustic model is not loaded")
        hidden_np = hidden.detach().to(device="cpu", dtype=torch.float32).numpy()
        with mx.stream(self._mlx_thread_stream):
            hidden_mx = mx.array(hidden_np)[None].astype(self.dtype)
            condition = self.model.condition_encoder(hidden_mx)
            key = mx.random.key(_derive_seed(int(seed), "dit", int(chunk_idx)))
            noise = mx.random.normal(
                (
                    1,
                    self.model.config.dit_in_channels,
                    condition.shape[1],
                ),
                key=key,
            ).astype(condition.dtype)
            latent, condition = denoise_chunk(
                self.model.transformer,
                noise,
                condition,
                num_inference_steps=self.dit_steps,
                guidance_scale=self.dit_cfg_scale,
                previous_latent=initial_latent,
                previous_condition=initial_condition,
                should_abort=should_abort,
            )
            if should_abort is not None and should_abort():
                raise InterruptedError(
                    "MiniMax Music 3 MLX acoustic generation aborted"
                )
            waveform = self.model.vocoder(latent)[0].astype(mx.float32)
            overlap = overlap_mel_length()
            start = max(0, latent.shape[-1] - 2 * overlap)
            end = max(start, latent.shape[-1] - overlap)
            latent_save = latent[..., start:end]
            condition_save = condition[:, start:end]
            mx.eval(waveform, latent_save, condition_save)

        wave_np = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
        wave = torch.from_numpy(wave_np).clamp(-1.0, 1.0)
        return wave, latent_save, condition_save


__all__ = ["MiniMaxMusic3MlxAcousticDecoder"]
