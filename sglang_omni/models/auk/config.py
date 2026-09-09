# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for AuK: preprocessing -> auk_engine -> vocoder.

Neither the engine (Qwen2.5-Omni conditioning + flow-matching DiT) nor the
vocoder (BigVGAN-Flow VAE) is autoregressive, so neither needs an AR scheduler.
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.models.auk import constants as C
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.auk"

PREPROCESSING_STAGE = "preprocessing"
ENGINE_STAGE = "auk_engine"
VOCODER_STAGE = "vocoder"


class AuKPipelineConfig(PipelineConfig):
    """AuK speech generation / editing pipeline."""

    architecture: ClassVar[str] = "AuKForConditionalGeneration"
    # AuK and AuK-Flash share one architecture; the variant is picked by
    # ``config.yaml::model.name``.
    architecture_aliases: ClassVar[tuple[str, ...]] = ("AuK", "AuK-Flash")

    # Reference audio is optional (instruction-only TTS is supported).
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
                nfe=C.DEFAULT_NFE,
                cfg_strength=C.DEFAULT_CFG_STRENGTH,
                sway_sampling_coef=C.DEFAULT_SWAY_SAMPLING_COEF,
                max_seconds=C.MAX_SECONDS,
                max_batch_size=1,
                max_batch_wait_ms=0,
            ),
            gpu=0,
            next=VOCODER_STAGE,
        ),
        StageConfig(
            name=VOCODER_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(
                dtype="bfloat16",
                max_batch_size=8,
                max_batch_wait_ms=5,
            ),
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = AuKPipelineConfig
