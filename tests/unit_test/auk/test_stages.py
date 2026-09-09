# SPDX-License-Identifier: Apache-2.0
"""Reference generation, result transport, and checkpoint sampling recipes."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.stages import (
    _EngineContext,
    _generate_one,
    create_auk_engine_executor,
)
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.proto import CompleteMessage, OmniRequest, StagePayload


@pytest.fixture
def context():
    vae = Mock(hop_size=480)
    vae.encoding_and_normalization.return_value = (
        torch.arange(64 * 51, dtype=torch.float32).reshape(1, 64, 51).transpose(1, 2),
        torch.tensor([50]),
    )
    vae.denormalize.side_effect = lambda latent: latent
    vae.inference_from_latents.side_effect = lambda latent: torch.full(
        (1, 1, latent.shape[-1] * 480), 0.25
    )
    encoder = Mock()
    encoder.encode.return_value = (
        torch.zeros(3, 6, 16),
        torch.ones(6, dtype=torch.bool),
    )
    flow = Mock()
    flow.sample.side_effect = lambda item, **kwargs: torch.zeros(item.target_frames, 64)
    return _EngineContext(
        config=AuKRuntimeConfig(model_path="stub"),
        encoder=encoder,
        vae=vae,
        flow=flow,
        device=torch.device("cpu"),
        compute_dtype=None,
        nfe=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1,
        t_grid=None,
        max_frames=1500,
    )


def test_reference_generation_survives_control_plane_transport(context):
    state = AuKState(
        instruction="Say hello",
        gen_frames=151,
        ref_audio=np.zeros(24001, dtype=np.float32),
    )
    payload = StagePayload(
        request_id="test", request=OmniRequest(inputs="hello"), data=state.to_dict()
    )
    result = _generate_one(context, payload)

    # Effective length counts complete frames; preserve the VAE's padded latent and layout.
    _, lengths = context.vae.encoding_and_normalization.call_args.args
    assert lengths.tolist() == [24000]
    item = context.flow.sample.call_args.args[0]
    expected = context.vae.encoding_and_normalization.return_value[0][0]
    torch.testing.assert_close(item.ref_latent, expected)
    assert item.ref_latent.stride() == (1, 51)
    assert item.ref_length == 50

    message = CompleteMessage(
        request_id=payload.request_id,
        from_stage="auk_engine",
        success=True,
        result=result.data,
    )
    restored = deserialize_message(serialize_message(message))
    assert restored.result == result.data
    assert restored.result["audio_waveform_shape"] == [151 * 480]
    waveform = np.frombuffer(restored.result["audio_waveform"], dtype=np.float32)
    np.testing.assert_array_equal(waveform, np.full(151 * 480, 0.25))
    assert restored.result["modality"] == "audio"
    assert restored.result["usage"]["completion_tokens"] == 151


@pytest.mark.parametrize("flash", [False, True])
def test_engine_uses_checkpoint_sampling_recipe(monkeypatch, flash):
    from sglang_omni.models.auk import stages

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    config = AuKRuntimeConfig(model_path="stub", name="AuK-Flash" if flash else "AuK")
    monkeypatch.setattr(stages, "make_runtime_config", lambda *args, **kwargs: config)
    encoder = Mock(num_hidden_layers=2)
    monkeypatch.setattr(stages, "AuKConditionEncoder", Mock(return_value=encoder))
    for name in (
        "AuKDit",
        "AuKFlowMatching",
        "BigVGANFlowVAE",
        "load_dit_weights",
        "load_vae_weights",
    ):
        monkeypatch.setattr(stages, name, Mock())
    captured = []
    monkeypatch.setattr(
        stages, "_generate_one", lambda ctx, payload: captured.append(ctx)
    )
    scheduler = create_auk_engine_executor("stub", device="cpu", nfe=8, cfg_strength=3)
    scheduler._fn(None)
    ctx = captured[0]
    assert ctx.nfe == (4 if flash else 8)
    assert ctx.cfg_strength == (0 if flash else 3)
    assert ctx.t_grid == (C.FLASH_T_GRID if flash else None)
    assert ctx.sway_sampling_coef == (None if flash else -1)
