# SPDX-License-Identifier: Apache-2.0
"""Request normalization for AuK (no GPU, no weights)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchaudio

from sglang_omni.models.auk.constants import (
    MAX_SECONDS,
    SAMPLE_RATE,
    VAE_DOWNSAMPLE_RATE,
)
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.request_builders import (
    AuKPreprocessingContext,
    build_auk_state,
    clear_auk_preprocessing_context,
    preprocess_auk_payload,
    set_auk_preprocessing_context,
)
from sglang_omni.proto import OmniRequest, StagePayload

FRAME_RATE = SAMPLE_RATE // VAE_DOWNSAMPLE_RATE


@pytest.fixture()
def context(tmp_path):
    config = AuKRuntimeConfig(
        model_path=str(tmp_path),
        vae={
            "target_sample_rate": SAMPLE_RATE,
            "downsample_rate": VAE_DOWNSAMPLE_RATE,
            "latent_dim": 64,
        },
    )
    set_auk_preprocessing_context(
        AuKPreprocessingContext(
            config=config, default_seconds=5.0, max_seconds=MAX_SECONDS
        )
    )
    yield config
    clear_auk_preprocessing_context()


def make_payload(inputs, params=None, metadata=None) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs=inputs, params=params or {}, metadata=metadata or {}
        ),
        data={},
    )


def write_wav(path, seconds: float, sample_rate: int = SAMPLE_RATE) -> None:
    samples = int(seconds * sample_rate)
    waveform = (0.1 * torch.randn(1, samples)).clamp(-1.0, 1.0)
    torchaudio.save(str(path), waveform, sample_rate)


def test_text_only_request_uses_default_duration(context):
    state = build_auk_state(make_payload("Say hello"), context)
    assert state.instruction == "Say hello"
    assert state.ref_audio is None
    assert state.gen_frames == 5 * FRAME_RATE
    assert state.sample_rate == SAMPLE_RATE


def test_instruction_is_read_from_inputs_and_params(context):
    state = build_auk_state(
        make_payload({"instruction": "Replace 'hi' with 'bye'"}), context
    )
    assert state.instruction == "Replace 'hi' with 'bye'"


def test_missing_instruction_is_rejected(context):
    with pytest.raises(ValueError, match="instruction"):
        build_auk_state(make_payload(""), context)


def test_gen_seconds_overrides_default(context):
    state = build_auk_state(make_payload("Say hello", {"gen_seconds": 2.0}), context)
    assert state.gen_frames == 2 * FRAME_RATE


def test_short_duration_rounds_up_to_one_frame(context):
    state = build_auk_state(make_payload("Say hello", {"gen_seconds": 0.001}), context)
    assert state.gen_frames == 1


def test_gen_seconds_is_clamped_to_max_seconds(context):
    state = build_auk_state(make_payload("Say hello", {"gen_seconds": 999.0}), context)
    assert state.gen_frames == MAX_SECONDS * FRAME_RATE


def test_invalid_gen_seconds_is_rejected(context):
    with pytest.raises(ValueError, match="gen_seconds"):
        build_auk_state(make_payload("Say hello", {"gen_seconds": 0.0}), context)


def test_reference_audio_defaults_to_source_length(context, tmp_path):
    path = tmp_path / "ref.wav"
    write_wav(path, seconds=1.0)
    state = build_auk_state(
        make_payload(
            "Say the following with the same voice: 'hello'",
            metadata={"tts_params": {"ref_audio": str(path)}},
        ),
        context,
    )
    assert state.ref_audio is not None
    assert state.ref_seconds == pytest.approx(1.0, abs=0.05)
    assert state.gen_frames == pytest.approx(1.0 * FRAME_RATE, abs=1)
    assert state.ref_audio.dtype == np.float32


def test_structured_reference_is_accepted(context, tmp_path):
    path = tmp_path / "ref.wav"
    write_wav(path, seconds=0.5)
    state = build_auk_state(
        make_payload(
            {"text": "Say hi", "references": [{"audio_path": str(path)}]},
            {"gen_seconds": 1.0},
        ),
        context,
    )
    assert state.ref_audio is not None
    assert state.gen_frames == 1 * FRAME_RATE


def test_raw_editing_duration_uses_complete_reference_frames(context, tmp_path):
    path = tmp_path / "ref.wav"
    write_wav(path, seconds=1.01)
    state = build_auk_state(
        make_payload({"instruction": "Remove noise", "audio": str(path)}), context
    )
    assert state.gen_frames == 50


def test_seed_is_forwarded(context):
    state = build_auk_state(
        make_payload("Say hello", {"seed": 42}),
        context,
    )
    assert state.seed == 42


@pytest.mark.parametrize("name", ["nfe", "cfg_strength", "sway_sampling_coef"])
def test_request_sampling_knobs_are_rejected(context, name):
    with pytest.raises(ValueError, match="server-level"):
        build_auk_state(make_payload("Say hello", {name: 1}), context)


def speech_payload(**kwargs):
    from sglang_omni.client.client import Client
    from sglang_omni.serve.protocol import CreateSpeechRequest
    from sglang_omni.serve.speech_service import SpeechRequestValidator

    request = CreateSpeechRequest(**kwargs)
    service = SpeechRequestValidator(default_model="tencent/AuK")
    generated = service.build_generate_request(request)
    return StagePayload(
        request_id="speech", request=Client._build_omni_request(generated), data={}
    )


def test_speech_instruct_tts_combines_instructions_and_input(context):
    payload = speech_payload(
        input="Welcome home.",
        instructions="warm, relaxed female voice",
        stage_params={"auk_engine": {"gen_seconds": 3.01}},
    )
    state = AuKState.from_dict(preprocess_auk_payload(payload).data)
    assert state.instruction == (
        'Generate speech based on the following description: "warm, relaxed female voice". '
        'The content to speak is: "Welcome home.".'
    )
    assert state.gen_frames == 151


def test_speech_zero_shot_tts_builds_auk_instruction(context, tmp_path):
    path = tmp_path / "ref.wav"
    write_wav(path, seconds=1)
    payload = speech_payload(
        input="Welcome home.",
        ref_audio=str(path),
        stage_params={"auk_engine": {"gen_seconds": 3}},
    )
    state = AuKState.from_dict(preprocess_auk_payload(payload).data)
    assert state.instruction == 'Say the following with the same voice: "Welcome home."'
    assert state.gen_frames == 3 * FRAME_RATE
    assert state.ref_seconds == pytest.approx(1)


def test_reference_tts_does_not_assume_reference_duration_equals_target_duration(
    context, tmp_path
):
    path = tmp_path / "ref.wav"
    write_wav(path, seconds=1)
    with pytest.raises(ValueError, match="gen_seconds"):
        preprocess_auk_payload(
            speech_payload(input="Welcome home.", ref_audio=str(path))
        )


def test_speech_without_description_uses_default_voice(context):
    state = build_auk_state(
        speech_payload(input="Hello.", stage_params={"auk_engine": {"gen_seconds": 2}}),
        context,
    )
    assert '"A clear, natural voice."' in state.instruction
    assert 'The content to speak is: "Hello.".' in state.instruction


def test_generate_preserves_raw_editing_instruction(context, tmp_path):
    from sglang_omni.client.client import Client
    from sglang_omni.serve.openai_api import _build_rollout_generate_request
    from sglang_omni.serve.protocol import RolloutGenerateRequest

    path = tmp_path / "ref.wav"
    write_wav(path, seconds=1.01)
    request = RolloutGenerateRequest(
        prompt="Remove the background noise.",
        metadata={"tts_params": {"ref_audio": str(path)}},
        output_modalities=["audio"],
    )
    generated = _build_rollout_generate_request(request)
    payload = StagePayload(
        request_id="editing", request=Client._build_omni_request(generated), data={}
    )
    state = AuKState.from_dict(preprocess_auk_payload(payload).data)
    assert state.instruction == request.prompt
    assert state.gen_frames == 50


def test_preprocess_payload_round_trips_state(context):
    payload = preprocess_auk_payload(make_payload("Say hello", {"gen_seconds": 1.0}))
    restored = AuKState.from_dict(payload.data)
    assert restored.instruction == "Say hello"
    assert restored.gen_frames == 1 * FRAME_RATE
    assert payload.request_id == "req-1"
