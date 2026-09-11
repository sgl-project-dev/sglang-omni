# SPDX-License-Identifier: Apache-2.0
"""The models Intel XPU serves, shared by both XPU test lanes.

tests/unit_test/xpu/test_model_support.py and test_model_smoke.py both
parametrize over these entries, so a family cannot be covered by one lane and
forgotten by the other. Scope is what ``main`` serves on XPU: the Qwen3 set. Add
an entry when a family's XPU support lands.
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ROOTS_ENV = "MODEL_ROOTS"
ALLOW_DOWNLOAD_ENV = "OMNI_XPU_ALLOW_HF_DOWNLOAD"


@dataclass(frozen=True)
class XpuModel:
    """One XPU-served checkpoint: how to build its pipeline and how to ask it."""

    key: str
    package: str
    architecture: str
    # Key in the config module's ``Variants`` table, or None for its EntryClass.
    variant: str | None
    # Pinned in order, so a stage gained or lost is visible here rather than as a
    # placement failure on a live host.
    stages: tuple[str, ...]
    repo_id: str
    # The 30B-A3B thinker does not fit one 24 GB card, so Omni shards across eight.
    cards: int
    # Which request shape the serving lane sends; dispatch lives in the test module.
    kind: str
    checkpoint_env: str

    @property
    def pipeline_key(self) -> str:
        """Identity of the *pipeline*, which two checkpoints can share."""
        return f"{self.package}:{self.variant or 'default'}"

    @property
    def mirror_dirname(self) -> str:
        return self.repo_id.split("/")[-1]


_OMNI_TEXT_STAGES = (
    "preprocessing",
    "image_encoder",
    "audio_encoder",
    "mm_aggregate",
    "thinker",
    "decode",
)

_OMNI_SPEECH_STAGES = (
    "preprocessing",
    "image_encoder",
    "audio_encoder",
    "thinker",
    "decode",
    "talker_ar",
    "code2wav",
)


XPU_MODELS: tuple[XpuModel, ...] = (
    XpuModel(
        key="asr",
        package="qwen3_asr",
        architecture="Qwen3ASRForConditionalGeneration",
        variant=None,
        stages=("asr",),
        repo_id="Qwen/Qwen3-ASR-1.7B",
        cards=1,
        kind="transcription",
        checkpoint_env="OMNI_XPU_ASR_MODEL",
    ),
    XpuModel(
        key="tts",
        package="qwen3_tts",
        architecture="Qwen3TTSForConditionalGeneration",
        variant=None,
        stages=("preprocessing", "vocoder", "tts_engine"),
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        cards=1,
        kind="speech_cloned",
        checkpoint_env="OMNI_XPU_TTS_MODEL",
    ),
    XpuModel(
        key="tts-cv",
        package="qwen3_tts",
        architecture="Qwen3TTSForConditionalGeneration",
        variant=None,
        stages=("preprocessing", "vocoder", "tts_engine"),
        repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        cards=1,
        kind="speech_builtin_voice",
        checkpoint_env="OMNI_XPU_TTS_CV_MODEL",
    ),
    XpuModel(
        key="omni-text",
        package="qwen3_omni",
        architecture="Qwen3OmniMoeForConditionalGeneration",
        variant="text",
        stages=_OMNI_TEXT_STAGES,
        repo_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        cards=8,
        kind="chat",
        checkpoint_env="OMNI_XPU_OMNI_MODEL",
    ),
    XpuModel(
        key="omni-speech",
        package="qwen3_omni",
        architecture="Qwen3OmniMoeForConditionalGeneration",
        variant="speech",
        stages=_OMNI_SPEECH_STAGES,
        repo_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        cards=8,
        kind="chat_speech",
        checkpoint_env="OMNI_XPU_OMNI_MODEL",
    ),
)


def xpu_pipelines() -> tuple[XpuModel, ...]:
    """One entry per distinct pipeline: the TTS Base and CustomVoice entries share
    Qwen3TTSPipelineConfig, so weight-free tests need only run over it once."""
    seen: set[str] = set()
    unique: list[XpuModel] = []
    for model in XPU_MODELS:
        if model.pipeline_key in seen:
            continue
        seen.add(model.pipeline_key)
        unique.append(model)
    return tuple(unique)


def model_named(key: str) -> XpuModel:
    for model in XPU_MODELS:
        if model.key == key:
            return model
    known = ", ".join(model.key for model in XPU_MODELS)
    raise KeyError(f"no XPU model named {key!r}; known keys: {known}")
