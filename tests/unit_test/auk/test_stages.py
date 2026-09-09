# SPDX-License-Identifier: Apache-2.0
"""Serial AuK engine contracts, without downloading checkpoints."""

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
    _reference_latent,
    create_auk_engine_executor,
)
from sglang_omni.proto import OmniRequest, StagePayload


@pytest.fixture
def context():
    vae = Mock(hop_size=480)
    vae.encoding_and_normalization.return_value = (
        torch.randn(1, 64, 51).transpose(1, 2),
        torch.tensor([50]),
    )
    vae.denormalize.side_effect = lambda latent: latent
    vae.inference_from_latents.side_effect = lambda latent: torch.zeros(
        1, 1, latent.shape[-1] * 480
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


def test_terminal_engine_encodes_and_decodes_with_same_vae(context):
    state = AuKState(
        instruction="Say hello",
        gen_frames=151,
        ref_audio=np.zeros(24000, dtype=np.float32),
    )
    payload = StagePayload(
        request_id="test", request=OmniRequest(inputs="hello"), data=state.to_dict()
    )
    result = _generate_one(context, payload)
    context.vae.encoding_and_normalization.assert_called_once()
    context.vae.inference_from_latents.assert_called_once()
    item = context.flow.sample.call_args.args[0]
    assert item.ref_latent.shape == (51, 64)
    assert item.ref_length == 50
    assert item.ref_latent.device == context.device
    assert result.data["audio_waveform_shape"] == [151 * 480]
    assert result.data["modality"] == "audio"
    assert "latent" not in result.data
    assert result.data["usage"]["completion_tokens"] == 151


def test_reference_encoding_samples_posterior(context):
    audio = np.zeros(24001, dtype=np.float32)
    latent, length = _reference_latent(context, audio)
    assert length == 50
    context.vae.encode.assert_not_called()
    assert torch.equal(
        latent, context.vae.encoding_and_normalization.return_value[0][0]
    )
    assert latent.stride() == (1, 51)
    assert context.vae.encoding_and_normalization.call_args.args[1].tolist() == [24000]


@pytest.mark.parametrize("flash", [False, True])
def test_factory_is_serial_and_flash_matches_upstream_recipe(monkeypatch, flash):
    from sglang_omni.models.auk import stages

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    config = AuKRuntimeConfig(model_path="stub", name="AuK-Flash" if flash else "AuK")
    monkeypatch.setattr(stages, "make_runtime_config", lambda *args, **kwargs: config)
    encoder = Mock(num_hidden_layers=2)
    encoder_factory = Mock(return_value=encoder)
    monkeypatch.setattr(stages, "AuKConditionEncoder", encoder_factory)
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
    assert scheduler._batch_fn is None
    assert scheduler._max_batch_size == scheduler._max_concurrency == 1
    assert encoder_factory.call_args.args[0] == C.DEFAULT_TEXT_ENCODER
    stages.BigVGANFlowVAE.assert_called_once()
    assert ctx.nfe == (4 if flash else 8)
    assert ctx.cfg_strength == (0 if flash else 3)
    assert ctx.t_grid == (C.FLASH_T_GRID if flash else None)
    assert ctx.sway_sampling_coef == (None if flash else -1)
