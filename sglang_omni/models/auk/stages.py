# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the AuK pipeline.

    preprocessing (CPU)  -> validate request, decode/resample reference audio
    auk_engine    (GPU)  -> Qwen2.5-Omni conditioning + flow-matching DiT
    vocoder       (GPU)  -> BigVGAN-Flow VAE decode to 24 kHz waveform

The engine and vocoder are plain batched ``SimpleScheduler`` stages (AuK is not
autoregressive).
"""

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
from sglang_omni.models.auk.reference_encode import (
    AuKConditionEncoder,
    build_messages,
    resample_to_qwen_rate,
)
from sglang_omni.models.auk.request_builders import (
    AuKPreprocessingContext,
    cleanup_prepared_auk_request,
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


def load_state_auk(payload: StagePayload) -> AuKState:
    return load_state(payload, AuKState)


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
        abort_callback=cleanup_prepared_auk_request,
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
    deterministic_reference_encode: bool = True


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


def _reference_latent(ctx: _EngineContext, ref_audio: Any) -> torch.Tensor | None:
    """VAE-encode the 24 kHz reference clip into the DiT's prompt latent."""
    if ref_audio is None:
        return None
    waveform = np.asarray(ref_audio, dtype=np.float32).reshape(1, 1, -1)
    tensor = torch.from_numpy(waveform).to(ctx.device)
    with torch.autocast(device_type=current_platform.device_type, enabled=False):
        if ctx.deterministic_reference_encode:
            latent = ctx.vae.encode(tensor)
        else:
            latent, _ = ctx.vae.encoding_and_normalization(tensor)
    length = min(latent.shape[1], waveform.shape[-1] // ctx.vae.hop_size)
    return latent[:, :length][0].contiguous()


def _prepare_item(ctx: _EngineContext, state: AuKState):
    has_reference = state.ref_audio is not None and np.asarray(state.ref_audio).size > 0
    messages = build_messages(state.instruction, has_reference_audio=has_reference)
    audio_16k = (
        resample_to_qwen_rate(
            np.asarray(state.ref_audio, dtype=np.float32), state.sample_rate
        )
        if has_reference
        else None
    )
    target_frames = int(min(max(int(state.gen_frames), 1), ctx.max_frames))
    return messages, audio_16k, _reference_latent(ctx, state.ref_audio), target_frames


def _generate(ctx: _EngineContext, payloads: list[StagePayload]) -> list[StagePayload]:
    states = [load_state_auk(payload) for payload in payloads]

    messages: list[list[dict[str, Any]]] = []
    audios: list[np.ndarray | None] = []
    ref_latents: list[torch.Tensor | None] = []
    target_frames: list[int] = []
    for state in states:
        message, audio, ref_latent, frames = _prepare_item(ctx, state)
        messages.append(message)
        audios.append(audio)
        ref_latents.append(ref_latent)
        target_frames.append(frames)

    started = time.perf_counter()
    hidden_states, text_masks = ctx.encoder.encode(messages, audios)
    encode_s = time.perf_counter() - started

    items = [
        AuKSampleItem(
            hidden_states=hidden,
            text_mask=mask,
            target_frames=frames,
            ref_latent=ref_latent,
            seed=state.seed,
        )
        for hidden, mask, frames, ref_latent, state in zip(
            hidden_states, text_masks, target_frames, ref_latents, states, strict=True
        )
    ]

    started = time.perf_counter()
    with _autocast(ctx):
        latents = ctx.flow.sample_batch(
            items,
            steps=ctx.nfe,
            cfg_strength=ctx.cfg_strength,
            sway_sampling_coef=ctx.sway_sampling_coef,
            t_grid=ctx.t_grid,
        )
    sample_s = time.perf_counter() - started

    results: list[StagePayload] = []
    for payload, state, latent, mask in zip(
        payloads, states, latents, text_masks, strict=True
    ):
        state.latent = latent.detach().to(torch.float32).cpu()
        state.prompt_tokens = int(mask.shape[0])
        state.completion_tokens = int(latent.shape[0])
        # Per-request attribution of the batch wall time (encode + ODE solve).
        state.engine_time_s = encode_s + sample_s
        results.append(store_state(payload, state))
    return results


async def _generate_one(ctx: _EngineContext, payload: StagePayload) -> StagePayload:
    return _generate(ctx, [payload])[0]


async def _generate_batch(
    ctx: _EngineContext, payloads: list[StagePayload]
) -> list[StagePayload]:
    return _generate(ctx, payloads)


def create_auk_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    text_encoder_path: str | None = None,
    nfe: int | None = None,
    cfg_strength: float | None = None,
    sway_sampling_coef: float | None = C.DEFAULT_SWAY_SAMPLING_COEF,
    max_seconds: float = C.MAX_SECONDS,
    deterministic_reference_encode: bool = True,
    max_batch_size: int = 1,
    max_batch_wait_ms: float = 0,
) -> SimpleScheduler:
    """Build the conditioning + flow-matching DiT stage."""
    if dtype not in _AUTOCAST_DTYPES:
        raise ValueError(
            f"Unsupported AuK engine dtype {dtype!r}; expected one of {sorted(_AUTOCAST_DTYPES)}"
        )
    checkpoint = resolve_checkpoint(model_path)
    config = make_runtime_config(checkpoint, text_encoder_path=text_encoder_path)
    resolved_device = torch.device(resolve_device_spec(device, gpu_id))

    encoder = AuKConditionEncoder(
        config.text_encoder_path or checkpoint,
        device=resolved_device,
        dtype=torch.bfloat16,
    )

    dit_config = AuKDitConfig.from_dict(config.arch)
    dit = AuKDit(
        **{**dit_config.__dict__, "latent_dim": config.latent_dim},
    )
    flow = AuKFlowMatching(dit, num_llm_layers=encoder.num_hidden_layers)
    load_dit_weights(flow, checkpoint)
    # The reference runs the DiT in fp32 under bf16 autocast.
    flow = flow.to(device=resolved_device, dtype=torch.float32).eval()

    vae = BigVGANFlowVAE(AuKVAEConfig.from_dict(config.vae_init_kwargs))
    load_vae_weights(vae, checkpoint)
    vae = vae.to(device=resolved_device).eval()
    vae.requires_grad_(False)

    if config.is_flash:
        # AuK-Flash is a DMD student: the 4-step grid is baked in and re-adding
        # CFG drives the amplitude into clipping.
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
        deterministic_reference_encode=deterministic_reference_encode,
    )

    return SimpleScheduler(
        lambda payload: _generate_one(ctx, payload),
        batch_compute_fn=lambda payloads: _generate_batch(ctx, payloads),
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


@dataclass
class _VocoderContext:
    vae: BigVGANFlowVAE
    device: torch.device


def _decode(ctx: _VocoderContext, payloads: list[StagePayload]) -> list[StagePayload]:
    states = [load_state_auk(payload) for payload in payloads]
    latents: list[torch.Tensor] = []
    for state in states:
        if state.latent is None:
            raise RuntimeError("AuK vocoder requires a latent from the engine stage")
        latent = torch.as_tensor(state.latent, dtype=torch.float32)
        if latent.ndim != 2:
            raise ValueError(
                f"AuK latent must be [frames, channels], got {tuple(latent.shape)}"
            )
        latents.append(latent)

    lengths = [int(latent.shape[0]) for latent in latents]
    max_length = max(lengths)
    channels = int(latents[0].shape[1])
    padded = torch.zeros(
        len(latents), max_length, channels, device=ctx.device, dtype=torch.float32
    )
    for index, latent in enumerate(latents):
        padded[index, : latent.shape[0]] = latent.to(ctx.device)

    denormalized = ctx.vae.denormalize(padded).permute(0, 2, 1)  # [B, D, T]
    with torch.autocast(device_type=current_platform.device_type, enabled=False):
        waveforms = ctx.vae.inference_from_latents(denormalized)  # [B, 1, T * hop]

    results: list[StagePayload] = []
    for index, (payload, state, length) in enumerate(
        zip(payloads, states, lengths, strict=True)
    ):
        samples = length * ctx.vae.hop_size
        wav = waveforms[index][:, :samples].detach().float().cpu()
        state.audio_samples = wav
        state.latent = None
        payload = store_state(payload, state)
        payload.data.update(
            audio_waveform_payload(
                wav, sample_rate=state.sample_rate, source_hint="AuK"
            )
        )
        payload.data["sample_rate"] = int(state.sample_rate)
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        results.append(payload)
    return results


async def _decode_one(ctx: _VocoderContext, payload: StagePayload) -> StagePayload:
    return _decode(ctx, [payload])[0]


async def _decode_batch(
    ctx: _VocoderContext, payloads: list[StagePayload]
) -> list[StagePayload]:
    return _decode(ctx, payloads)


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_batch_size: int = 8,
    max_batch_wait_ms: float = 5,
) -> SimpleScheduler:
    """Build the terminal VAE-decode stage."""
    if dtype not in _AUTOCAST_DTYPES:
        raise ValueError(
            f"Unsupported AuK vocoder dtype {dtype!r}; expected one of {sorted(_AUTOCAST_DTYPES)}"
        )
    checkpoint = resolve_checkpoint(model_path)
    config = make_runtime_config(checkpoint)
    resolved_device = torch.device(resolve_device_spec(device, gpu_id))

    vae = BigVGANFlowVAE(AuKVAEConfig.from_dict(config.vae_init_kwargs))
    load_vae_weights(vae, checkpoint)
    # The reference decodes in fp32; the VAE is causal so right padding is safe.
    vae = vae.to(device=resolved_device).eval()
    vae.requires_grad_(False)

    ctx = _VocoderContext(vae=vae, device=resolved_device)
    return SimpleScheduler(
        lambda payload: _decode_one(ctx, payload),
        batch_compute_fn=lambda payloads: _decode_batch(ctx, payloads),
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
