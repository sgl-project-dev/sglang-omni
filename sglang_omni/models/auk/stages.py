# SPDX-License-Identifier: Apache-2.0 AND MIT
# Inference recipe adapted from Tencent-Hunyuan/AuK, Copyright (C) 2026 Tencent.
# See LICENSE for the upstream MIT permission notice.
"""Stage factories for the AuK pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.dit import AuKDit, AuKDitConfig
from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig, make_runtime_config
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.models.auk.reference_encode import AuKConditionEncoder, build_messages
from sglang_omni.models.auk.request_builders import (
    AuKPreprocessingContext,
    preprocess_auk_payload,
    set_auk_preprocessing_context,
)
from sglang_omni.models.auk.vae import AuKVAEConfig, BigVGANFlowVAE
from sglang_omni.models.auk.weight_loader import load_dit_weights, load_vae_weights
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage, load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_device_spec

logger = logging.getLogger(__name__)

_AUTOCAST_DTYPES: dict[str, torch.dtype | None] = {
    "float32": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
    default_seconds: float = C.DEFAULT_SECONDS,
    max_seconds: float = C.MAX_SECONDS,
) -> SimpleScheduler:
    """CPU stage: normalize the request and load the reference clip."""
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than zero")

    config = make_runtime_config(resolve_checkpoint(model_path))
    set_auk_preprocessing_context(
        AuKPreprocessingContext(
            config=config,
            default_seconds=float(default_seconds),
            max_seconds=float(max_seconds),
        )
    )
    return SimpleScheduler(
        preprocess_auk_payload,
        max_concurrency=max_concurrency,
    )


@dataclass
class _EngineContext:
    config: AuKRuntimeConfig
    encoder: AuKConditionEncoder
    vae: BigVGANFlowVAE
    flow: AuKFlowMatching
    device: torch.device
    compute_dtype: torch.dtype | None
    nfe: int
    cfg_strength: float
    sway_sampling_coef: float | None
    t_grid: tuple[float, ...] | None
    max_frames: int


def _autocast(ctx: _EngineContext):
    enabled = (
        ctx.compute_dtype is not None
        and ctx.device.type == current_platform.device_type
    )
    return torch.autocast(
        device_type=current_platform.device_type,
        dtype=ctx.compute_dtype,
        enabled=enabled,
    )


def _reference_latent(
    ctx: _EngineContext, ref_audio: Any
) -> tuple[torch.Tensor | None, int]:
    """VAE-encode the 24 kHz reference clip into the DiT's prompt latent."""
    if ref_audio is None:
        return None, 0
    waveform = np.asarray(ref_audio, dtype=np.float32).reshape(1, 1, -1)
    tensor = torch.from_numpy(waveform).to(ctx.device)
    with torch.autocast(device_type=current_platform.device_type, enabled=False):
        lengths = torch.tensor(
            [waveform.shape[-1] // ctx.vae.hop_size * ctx.vae.hop_size],
            device=ctx.device,
        )
        latent, latent_lengths = ctx.vae.encoding_and_normalization(tensor, lengths)
    length = int(latent_lengths[0])
    return latent[0], length


@torch.inference_mode()
def _generate_one(ctx: _EngineContext, payload: StagePayload) -> StagePayload:
    state = load_state(payload, AuKState)
    started = time.perf_counter()
    has_reference = state.ref_audio is not None and np.asarray(state.ref_audio).size > 0
    messages = build_messages(state.instruction, has_reference)
    frames = min(max(state.gen_frames, 1), ctx.max_frames)
    ref_latent, ref_length = _reference_latent(ctx, state.ref_audio)
    with _autocast(ctx):
        hidden, mask = ctx.encoder.encode(messages, state.qwen_audio)
        latent = ctx.flow.sample(
            AuKSampleItem(hidden, mask, frames, ref_latent, state.seed, ref_length),
            steps=ctx.nfe,
            cfg_strength=ctx.cfg_strength,
            sway_sampling_coef=ctx.sway_sampling_coef,
            t_grid=ctx.t_grid,
        )
    if not torch.isfinite(latent).all():
        raise RuntimeError("AuK generated latent contains NaN/Inf")
    with torch.autocast(device_type=ctx.device.type, enabled=False):
        denormalized = ctx.vae.denormalize(latent.unsqueeze(0)).permute(0, 2, 1)
        wav = ctx.vae.inference_from_latents(denormalized)[0]
    if not torch.isfinite(wav).all():
        raise RuntimeError("AuK generated audio contains NaN/Inf")
    wav = wav.float().cpu()
    state.ref_audio = None
    state.qwen_audio = None
    state.prompt_tokens = int(mask.sum())
    state.completion_tokens = int(latent.shape[0])
    state.engine_time_s = time.perf_counter() - started
    payload = store_state(payload, state)
    payload.data.update(
        audio_waveform_payload(wav, sample_rate=state.sample_rate, source_hint="AuK")
    )
    payload.data["sample_rate"] = state.sample_rate
    payload.data["modality"] = "audio"
    usage = build_usage(state)
    if usage is not None:
        payload.data["usage"] = usage
    return payload


def create_auk_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    text_encoder_path: str = C.DEFAULT_TEXT_ENCODER,
    nfe: int | None = None,
    cfg_strength: float | None = None,
    sway_sampling_coef: float | None = C.DEFAULT_SWAY_SAMPLING_COEF,
    max_seconds: float = C.MAX_SECONDS,
) -> SimpleScheduler:
    """Build the serial terminal engine, sharing one VAE for encode and decode."""
    if dtype not in _AUTOCAST_DTYPES:
        raise ValueError(
            f"Unsupported AuK engine dtype {dtype!r}; expected one of {sorted(_AUTOCAST_DTYPES)}"
        )
    checkpoint = resolve_checkpoint(model_path)
    config = make_runtime_config(checkpoint, text_encoder_path=text_encoder_path)
    resolved_device = torch.device(resolve_device_spec(device, gpu_id))

    encoder = AuKConditionEncoder(
        config.text_encoder_path,
        device=resolved_device,
        dtype=torch.bfloat16,
    )

    dit_config = AuKDitConfig.from_dict(config.arch)
    dit = AuKDit(
        **{**dit_config.__dict__, "latent_dim": config.latent_dim},
    )
    flow = AuKFlowMatching(dit, num_llm_layers=encoder.num_hidden_layers)
    load_dit_weights(flow, checkpoint)
    flow = flow.to(device=resolved_device, dtype=torch.float32).eval()

    vae = BigVGANFlowVAE(AuKVAEConfig.from_dict(config.vae_init_kwargs))
    load_vae_weights(vae, checkpoint)
    vae = vae.to(device=resolved_device).eval()
    vae.requires_grad_(False)

    if config.is_flash:
        nfe, cfg_strength, sway_sampling_coef = (
            C.FLASH_NFE,
            C.FLASH_CFG_STRENGTH,
            None,
        )
        t_grid: tuple[float, ...] | None = C.FLASH_T_GRID
        logger.info("AuK: detected AuK-Flash; locking to NFE=4, CFG=0")
    else:
        t_grid = None

    ctx = _EngineContext(
        config=config,
        encoder=encoder,
        vae=vae,
        flow=flow,
        device=resolved_device,
        compute_dtype=_AUTOCAST_DTYPES[dtype],
        nfe=int(nfe if nfe is not None else C.DEFAULT_NFE),
        cfg_strength=float(
            cfg_strength if cfg_strength is not None else C.DEFAULT_CFG_STRENGTH
        ),
        sway_sampling_coef=sway_sampling_coef,
        t_grid=t_grid,
        max_frames=config.seconds_to_frames(max_seconds),
    )

    return SimpleScheduler(lambda payload: _generate_one(ctx, payload))
