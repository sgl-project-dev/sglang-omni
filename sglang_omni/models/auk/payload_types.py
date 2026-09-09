# SPDX-License-Identifier: Apache-2.0
"""AuK pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.models.auk import constants as C
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class AuKState(DeclarativeStateBase):
    """Request fields passed from preprocessing to the terminal AuK engine."""

    sample_rate: int = wire(C.SAMPLE_RATE, codec="int")

    instruction: str = wire("", codec="str")
    ref_audio: Any | None = None
    qwen_audio: Any | None = None
    ref_seconds: float = wire(0.0, codec="float")

    gen_frames: int = wire(0, codec="int")
    seed: int | None = None

    @property
    def gen_seconds(self) -> float:
        return self.gen_frames * C.VAE_DOWNSAMPLE_RATE / self.sample_rate
