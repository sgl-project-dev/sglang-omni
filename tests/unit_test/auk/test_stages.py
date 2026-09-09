# SPDX-License-Identifier: Apache-2.0
"""Batched stage hand-offs and checkpoint sampling recipes."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.flow_matching import request_generator
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.stages import (
    _condition_batch,
    _decode_batch,
    _sample_batch,
    create_auk_engine_executor,
)
from sglang_omni.models.auk.vae import BigVGANFlowVAE
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.proto import CompleteMessage, OmniRequest, StagePayload


def test_batched_generation_preserves_request_boundaries_and_serializes_audio():
    device = torch.device("cpu")
    vae = Mock(hop_size=480)
    vae.encoding_and_normalization.return_value = (
        torch.arange(64 * 51, dtype=torch.float32).reshape(1, 64, 51).transpose(1, 2),
        torch.tensor([50]),
    )
    vae.denormalize.side_effect = lambda latent: latent
    vae.inference_from_latents.side_effect = lambda latent: torch.full(
        (latent.shape[0], 1, latent.shape[-1] * 480), 0.25
    )
    encoder = Mock()
    encoder.encode_batch.return_value = [
        (torch.zeros(3, 6, 16), torch.ones(6, dtype=torch.bool)) for _ in range(3)
    ]
    flow = Mock()
    flow.fuse.side_effect = lambda hidden: hidden[:, 0]
    flow.sample_batch.side_effect = lambda items, **kwargs: [
        torch.zeros(item.target_frames, 64) for item in items
    ]
    payloads = [
        StagePayload(
            request_id=str(index),
            request=OmniRequest(inputs="hello"),
            data=AuKState(
                instruction="Say hello",
                gen_frames=frames,
                seed=11,
                ref_audio=np.zeros(24001, dtype=np.float32),
            ).to_dict(),
        )
        for index, frames in enumerate((151, 75, 151))
    ]

    conditioned = _condition_batch(payloads, encoder, vae, flow, device, "float32")
    assert vae.encoding_and_normalization.call_args.args[1].tolist() == [24000]
    assert torch.equal(
        vae.encoding_and_normalization.call_args.kwargs["generator"].get_state(),
        request_generator(11, device).get_state(),
    )
    state = AuKState.from_dict(conditioned[0].data)
    assert state.ref_length == 50
    assert state.ref_latent.stride() == (1, 51)
    sampled = _sample_batch(conditioned, flow, device, "float32", 1500, {})
    assert len(flow.sample_batch.call_args.args[0]) == 3
    results = _decode_batch(sampled, vae, device)

    assert [
        call.args[0].shape[0] for call in vae.inference_from_latents.call_args_list
    ] == [2, 1]
    for index, (frames, result) in enumerate(zip((151, 75, 151), results)):
        restored = deserialize_message(
            serialize_message(
                CompleteMessage(
                    request_id=result.request_id,
                    from_stage="decode",
                    success=True,
                    result=result.data,
                )
            )
        )
        assert restored.request_id == str(index)
        assert restored.result["audio_waveform_shape"] == [frames * 480]
        waveform = np.frombuffer(restored.result["audio_waveform"], dtype=np.float32)
        np.testing.assert_array_equal(waveform, np.full(frames * 480, 0.25))
        assert restored.result["usage"]["completion_tokens"] == frames


def test_reference_posterior_is_seeded_and_leaves_process_rng():
    vae = BigVGANFlowVAE.__new__(BigVGANFlowVAE)
    vae.hop_size = 4
    vae.global_mean = torch.zeros(1, 2)
    vae.global_log_std = torch.ones(1, 2)

    def audio_encoder(sample):
        frames = sample.size(-1) // vae.hop_size
        stats = torch.zeros(sample.size(0), 4, frames)
        stats[:, 2:] = -1.0
        return stats

    vae.audio_encoder = audio_encoder
    waveform = torch.zeros(1, 1, 16)
    lengths = torch.tensor([16])
    torch.manual_seed(0)
    before = torch.random.get_rng_state()
    first, _ = vae.encoding_and_normalization(
        waveform, lengths, generator=request_generator(123, waveform.device)
    )
    second, _ = vae.encoding_and_normalization(
        waveform, lengths, generator=request_generator(123, waveform.device)
    )
    other, _ = vae.encoding_and_normalization(
        waveform, lengths, generator=request_generator(456, waveform.device)
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("flash", [False, True])
def test_engine_uses_checkpoint_sampling_recipe(monkeypatch, flash):
    from sglang_omni.models.auk import stages

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    config = AuKRuntimeConfig(model_path="stub", name="AuK-Flash" if flash else "AuK")
    monkeypatch.setattr(stages, "make_runtime_config", lambda path: config)
    flow = Mock()
    flow.sample_batch.return_value = [torch.zeros(10, 64)]
    monkeypatch.setattr(stages, "_load_flow", lambda *args: flow)
    scheduler = create_auk_engine_executor("stub", device="cpu", nfe=8, cfg_strength=3)
    state = AuKState(
        gen_frames=10,
        conditioning=torch.zeros(6, 16),
        text_mask=torch.ones(6, dtype=torch.bool),
    )
    scheduler._fn(
        StagePayload(
            request_id="test", request=OmniRequest(inputs="hello"), data=state.to_dict()
        )
    )
    recipe = flow.sample_batch.call_args.kwargs
    assert recipe == dict(
        steps=4 if flash else 8,
        cfg_strength=0 if flash else 3,
        sway_sampling_coef=None if flash else -1,
        t_grid=C.FLASH_T_GRID if flash else None,
    )
