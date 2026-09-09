# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for AuK.

Normalizes the several shapes a client can send an instruction + optional
reference clip in, validates them, and loads/resamples the reference on CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.proto import StagePayload
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference

logger = logging.getLogger(__name__)


@dataclass
class AuKPreprocessingContext:
    config: AuKRuntimeConfig
    default_seconds: float = C.DEFAULT_SECONDS
    max_seconds: float = C.MAX_SECONDS


_CONTEXT: AuKPreprocessingContext | None = None


def set_auk_preprocessing_context(context: AuKPreprocessingContext) -> None:
    global _CONTEXT
    _CONTEXT = context


def clear_auk_preprocessing_context() -> None:
    global _CONTEXT
    _CONTEXT = None


def cleanup_prepared_auk_request(request_id: str) -> None:
    """Abort hook for the preprocessing scheduler (no per-request staging)."""
    del request_id


def _get_context() -> AuKPreprocessingContext:
    if _CONTEXT is None:
        raise RuntimeError("AuK preprocessing context is not initialized")
    return _CONTEXT


def _normalize_inputs(inputs: Any) -> tuple[str, list[dict[str, Any]], Any | None]:
    """Accept flat text, a dict payload, or a structured references list."""
    if isinstance(inputs, str):
        return inputs, [], None
    if not isinstance(inputs, dict):
        return (str(inputs) if inputs is not None else ""), [], None

    raw_references = inputs.get("references") or []
    if not isinstance(raw_references, list):
        raise ValueError("AuK references must be a list")
    if any(not isinstance(reference, dict) for reference in raw_references):
        raise ValueError("AuK references must be objects")
    references = [dict(reference) for reference in raw_references]

    text = str(
        inputs.get("instruction") or inputs.get("text") or inputs.get("input") or ""
    )
    ref_audio = inputs.get("ref_audio") or inputs.get("audio") or inputs.get("file")
    return text, references, ref_audio


def _resolve_reference(
    references: list[dict[str, Any]], fallback: Any | None
) -> Any | None:
    if fallback is not None:
        return fallback
    if not references:
        return None
    reference = references[0]
    return (
        reference.get("audio_path")
        or reference.get("ref_audio")
        or reference.get("audio")
        or audio_data_uri_from_reference(reference)
    )


def _resolve_float(raw: Any, default: float | None) -> float | None:
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AuK expected a number, got {raw!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"AuK expected a finite number, got {raw!r}")
    return value


def _resolve_seed(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError("AuK seed must be an integer")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AuK seed must be an integer, got {raw!r}") from exc


def build_auk_state(payload: StagePayload, config: AuKRuntimeConfig) -> AuKState:
    """Build the AuK state from an incoming request."""
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references, inline_ref = _normalize_inputs(inputs)
    instruction = str(
        tts_params.get("instruction") or params.get("instruction") or text
    ).strip()
    if not instruction:
        raise ValueError("AuK requires a natural-language instruction")

    ref_source = _resolve_reference(references, inline_ref) or tts_params.get(
        "ref_audio"
    )

    gen_seconds = _resolve_float(
        tts_params.get("gen_seconds", params.get("gen_seconds")), None
    )
    if gen_seconds is not None and gen_seconds <= 0:
        raise ValueError(f"AuK gen_seconds must be positive, got {gen_seconds}")

    clip_seconds = _resolve_float(
        tts_params.get("max_seconds", params.get("max_seconds")),
        _get_context().max_seconds if _CONTEXT is not None else C.MAX_SECONDS,
    )

    ref_audio: np.ndarray | None = None
    ref_seconds = 0.0
    if ref_source is not None:
        ref_audio = np.asarray(
            load_audio(
                ref_source, source_name="AuK", target_sample_rate=config.sample_rate
            ),
            dtype=np.float32,
        ).reshape(-1)
        ref_seconds = ref_audio.shape[-1] / float(config.sample_rate)

    if gen_seconds is None:
        # Default to the source length, or a short default when reference-free.
        gen_seconds = (
            ref_seconds
            if ref_seconds > 0
            else (_get_context().default_seconds if _CONTEXT else C.DEFAULT_SECONDS)
        )
    gen_seconds = float(min(max(gen_seconds, C.MIN_SECONDS), clip_seconds))

    return AuKState(
        sample_rate=config.sample_rate,
        instruction=instruction,
        ref_audio=ref_audio,
        ref_seconds=float(ref_seconds),
        gen_frames=config.seconds_to_frames(gen_seconds),
        nfe=int(
            tts_params.get("nfe", params.get("nfe", C.DEFAULT_NFE)) or C.DEFAULT_NFE
        ),
        cfg_strength=float(
            tts_params.get(
                "cfg_strength", params.get("cfg_strength", C.DEFAULT_CFG_STRENGTH)
            )
        ),
        sway_sampling_coef=_resolve_float(
            tts_params.get("sway_sampling_coef", params.get("sway_sampling_coef")),
            C.DEFAULT_SWAY_SAMPLING_COEF,
        ),
        seed=_resolve_seed(tts_params.get("seed", params.get("seed"))),
    )


def preprocess_auk_payload(payload: StagePayload) -> StagePayload:
    """Preprocessing-stage entry point: validate, load audio, size the output."""
    context = _get_context()
    state = build_auk_state(payload, context.config)
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )
