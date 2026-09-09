# SPDX-License-Identifier: Apache-2.0
"""AuK: CPU preprocessing followed by serial conditioning, sampling and decode."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.models.auk import constants as C
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.auk"

PREPROCESSING_STAGE = "preprocessing"
ENGINE_STAGE = "auk_engine"


class AuKPipelineConfig(PipelineConfig):
    """AuK speech generation / editing pipeline."""

    architecture: ClassVar[str] = "AuKForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("AuK", "AuK-Flash")

    required_speech_reference_count: ClassVar[int | None] = None

    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=8),
            next=ENGINE_STAGE,
        ),
        StageConfig(
            name=ENGINE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_auk_engine_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                text_encoder_path=C.DEFAULT_TEXT_ENCODER,
                nfe=C.DEFAULT_NFE,
                cfg_strength=C.DEFAULT_CFG_STRENGTH,
                sway_sampling_coef=C.DEFAULT_SWAY_SAMPLING_COEF,
                max_seconds=C.MAX_SECONDS,
            ),
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = AuKPipelineConfig
