# SPDX-License-Identifier: Apache-2.0
"""AuK pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.models.auk import constants as C
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class AuKState(DeclarativeStateBase):
    """Per-request state for AuK generation/editing.

    Preprocessing fills the request fields, the engine fills ``latent``, and the
    vocoder replaces it with ``audio_samples`` on the way out.
    """

    sample_rate: int = wire(C.SAMPLE_RATE, codec="int")

    instruction: str = wire("", codec="str")
    # 24 kHz mono reference waveform, kept as numpy until the GPU stage.
    ref_audio: Any | None = None
    ref_seconds: float = wire(0.0, codec="float")

    gen_frames: int = wire(0, codec="int")
    nfe: int = wire(C.DEFAULT_NFE, codec="int")
    cfg_strength: float = wire(C.DEFAULT_CFG_STRENGTH, codec="float")
    sway_sampling_coef: float | None = None
    seed: int | None = None

    latent: Any | None = wire(None, codec="tensor_cpu")
    audio_samples: Any | None = wire(None, codec="tensor_cpu")

    @property
    def gen_seconds(self) -> float:
        return self.gen_frames * C.VAE_DOWNSAMPLE_RATE / self.sample_rate
