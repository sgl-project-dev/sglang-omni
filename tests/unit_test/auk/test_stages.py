# SPDX-License-Identifier: Apache-2.0
"""Stage-level behavior for AuK, exercised with tiny stand-in modules.

These tests cover the batching/slicing contracts of the engine and vocoder
stages without loading real weights: the DiT and VAE are the real modules at
toy sizes, and the Qwen conditioning encoder is a fake that emits hidden states
of the right shape.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.auk.dit import AuKDit
from sglang_omni.models.auk.flow_matching import AuKFlowMatching
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.stages import (
    _decode,
    _EngineContext,
    _generate,
    _VocoderContext,
    create_preprocessing_executor,
)
from sglang_omni.models.auk.vae import AuKVAEConfig, BigVGANFlowVAE
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.pipeline_state import load_state

LATENT_DIM = 8
TEXT_DIM = 16
NUM_LLM_LAYERS = 2
HOP = 20


def tiny_vae() -> BigVGANFlowVAE:
    config = AuKVAEConfig(
        upsample_rates=[5, 4],
        upsample_kernel_sizes=[10, 8],
        upsample_initial_channel=16,
        resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
        downsample_rates=[4, 5],
        downsample_channels=[12, 16, 32],
        latent_dim=LATENT_DIM,
        flow_hidden_channels=16,
    )
    vae = BigVGANFlowVAE(config)
    assert vae.hop_size == HOP
    return vae.eval()


def tiny_flow() -> AuKFlowMatching:
    dit = AuKDit(
        dim=32,
        heads=2,
        dim_head=16,
        num_layers=1,
        num_single_layers=1,
        latent_dim=LATENT_DIM,
        text_hidden_dim=TEXT_DIM,
    )
    return AuKFlowMatching(dit, num_llm_layers=NUM_LLM_LAYERS).eval()


class FakeEncoder:
    """Stands in for the frozen Qwen2.5-Omni Thinker."""

    num_hidden_layers = NUM_LLM_LAYERS

    def __init__(self, text_len: int = 6):
        self.text_len = text_len
        self.calls: list[int] = []

    def encode(self, messages, audios):
        self.calls.append(len(messages))
        hidden = [
            torch.randn(NUM_LLM_LAYERS + 1, self.text_len, TEXT_DIM) for _ in messages
        ]
        masks = [torch.ones(self.text_len, dtype=torch.bool) for _ in messages]
        return hidden, masks


def make_payload(request_id: str, state: AuKState) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=state.instruction),
        data=state.to_dict(),
    )


def make_state(**kwargs) -> AuKState:
    defaults = {
        "instruction": "Say hello",
        "sample_rate": 24000,
        "gen_frames": 6,
        "nfe": 2,
        "cfg_strength": 0.0,
        "sway_sampling_coef": None,
        "seed": None,
    }
    defaults.update(kwargs)
    return AuKState(**defaults)


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #


def test_preprocessing_executor_builds(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  name: AuK\n"
        "  vae:\n"
        "    target_sample_rate: 24000\n"
        "    downsample_rate: 480\n"
        "    latent_dim: 64\n",
        encoding="utf-8",
    )
    scheduler = create_preprocessing_executor(str(tmp_path), max_concurrency=2)
    assert scheduler is not None


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #


def test_generate_produces_requested_frame_counts():
    vae = tiny_vae()
    ctx = _EngineContext(
        config=AuKRuntimeConfig(model_path="stub"),
        encoder=FakeEncoder(),
        vae=vae,
        flow=tiny_flow(),
        device=torch.device("cpu"),
        compute_dtype=None,
        nfe=2,
        cfg_strength=0.0,
        sway_sampling_coef=None,
        t_grid=None,
        max_frames=64,
    )

    states = [make_state(gen_frames=5), make_state(gen_frames=9)]
    payloads = [make_payload(f"req-{i}", state) for i, state in enumerate(states)]
    results = _generate(ctx, payloads)

    assert len(results) == 2
    for payload, frames in zip(results, (5, 9)):
        state = load_state(payload, AuKState)
        assert state.latent.shape == (frames, LATENT_DIM)
        assert state.prompt_tokens == FakeEncoder().text_len
        assert state.completion_tokens == frames


def test_generate_without_reference_audio():
    vae = tiny_vae()
    ctx = _EngineContext(
        config=AuKRuntimeConfig(model_path="stub"),
        encoder=FakeEncoder(),
        vae=vae,
        flow=tiny_flow(),
        device=torch.device("cpu"),
        compute_dtype=None,
        nfe=2,
        cfg_strength=0.0,
        sway_sampling_coef=None,
        t_grid=None,
        max_frames=64,
    )
    state = make_state(gen_frames=4, ref_audio=None)
    results = _generate(ctx, [make_payload("req-0", state)])
    assert load_state(results[0], AuKState).latent.shape == (4, LATENT_DIM)


def test_generate_encodes_reference_audio_into_prompt_latent():
    vae = tiny_vae()
    encoder = FakeEncoder()
    ctx = _EngineContext(
        config=AuKRuntimeConfig(model_path="stub"),
        encoder=encoder,
        vae=vae,
        flow=tiny_flow(),
        device=torch.device("cpu"),
        compute_dtype=None,
        nfe=2,
        cfg_strength=0.0,
        sway_sampling_coef=None,
        t_grid=None,
        max_frames=64,
    )
    # 400 samples at 24 kHz -> 20 latent frames at the toy hop size.
    ref_audio = np.zeros(400, dtype=np.float32)
    state = make_state(gen_frames=3, ref_audio=ref_audio)
    results = _generate(ctx, [make_payload("req-0", state)])
    assert encoder.calls == [1]
    assert load_state(results[0], AuKState).latent.shape == (3, LATENT_DIM)


def test_generate_clamps_to_max_frames():
    vae = tiny_vae()
    ctx = _EngineContext(
        config=AuKRuntimeConfig(model_path="stub"),
        encoder=FakeEncoder(),
        vae=vae,
        flow=tiny_flow(),
        device=torch.device("cpu"),
        compute_dtype=None,
        nfe=2,
        cfg_strength=0.0,
        sway_sampling_coef=None,
        t_grid=None,
        max_frames=4,
    )
    state = make_state(gen_frames=100)
    results = _generate(ctx, [make_payload("req-0", state)])
    assert load_state(results[0], AuKState).latent.shape == (4, LATENT_DIM)


def test_generate_is_seed_reproducible():
    vae = tiny_vae()
    flow = tiny_flow()

    def run(seed: int) -> torch.Tensor:
        ctx = _EngineContext(
            config=AuKRuntimeConfig(model_path="stub"),
            encoder=FakeEncoder(),
            vae=vae,
            flow=flow,
            device=torch.device("cpu"),
            compute_dtype=None,
            nfe=2,
            cfg_strength=0.0,
            sway_sampling_coef=None,
            t_grid=None,
            max_frames=64,
        )
        state = make_state(gen_frames=5, seed=seed)
        payload = _generate(ctx, [make_payload("req-0", state)])[0]
        return load_state(payload, AuKState).latent

    assert torch.allclose(run(123), run(123))
    assert not torch.allclose(run(123), run(456))


# --------------------------------------------------------------------------- #
# vocoder
# --------------------------------------------------------------------------- #


def test_decode_batch_returns_waveform_payload():
    vae = tiny_vae()
    ctx = _VocoderContext(vae=vae, device=torch.device("cpu"))

    states = [
        make_state(gen_frames=5, latent=torch.randn(5, LATENT_DIM)),
        make_state(gen_frames=3, latent=torch.randn(3, LATENT_DIM)),
    ]
    payloads = [make_payload(f"req-{i}", state) for i, state in enumerate(states)]
    results = _decode(ctx, payloads)

    assert len(results) == 2
    for payload, frames in zip(results, (5, 3)):
        assert payload.data["modality"] == "audio"
        assert payload.data["sample_rate"] == 24000
        waveform = payload.data["audio_waveform"]
        shape = payload.data["audio_waveform_shape"]
        assert shape == [frames * HOP]
        assert len(waveform) == frames * HOP * 4  # float32
        # The latent is dropped once the waveform is attached.
        assert load_state(payload, AuKState).latent is None


def test_decode_requires_a_latent():
    vae = tiny_vae()
    ctx = _VocoderContext(vae=vae, device=torch.device("cpu"))
    payload = make_payload("req-0", make_state())
    with pytest.raises(RuntimeError, match="latent"):
        _decode(ctx, [payload])


def test_decode_rejects_malformed_latent():
    vae = tiny_vae()
    ctx = _VocoderContext(vae=vae, device=torch.device("cpu"))
    payload = make_payload("req-0", make_state(latent=torch.randn(LATENT_DIM)))
    with pytest.raises(ValueError, match=r"\[frames, channels\]"):
        _decode(ctx, [payload])
