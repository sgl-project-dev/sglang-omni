# SPDX-License-Identifier: Apache-2.0
"""dots.tts MLX flow-matching decode engine.

The flow history is append-only: each step appends the projected hidden row and
the projected latent patch to two row lists (conditioned / null-conditioned).
MLX arrays are immutable, so the DiT input is rebuilt from the history each
step with lazy concatenation — which the full-compute velocity function needs
anyway.

decode_step mirrors flow_head.decode_next + the dit_inference
full-compute flow-matching path: one semantic-encoder feedback embedding feeds
the backbone, its hidden row is appended to the flow history, the DiT ODE with
classifier-free guidance is integrated t=0..1, and the denormalized latent
patch, the next feedback row, the EOS flag and the emit flag come back.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from sglang_omni.models.dots_tts.mlx.model import DotsTTSFlowState, DotsTTSMlxModel

_MASK_VALUE = -1.0e9


def start_request(
    model: DotsTTSMlxModel,
    *,
    prompt_latents: Optional[mx.array],
    speaker_embedding: Optional[mx.array],
    speaker_scale: float,
    seed: Optional[int],
) -> tuple[DotsTTSFlowState, Optional[mx.array]]:
    """Initialize request flow state, return semantic prompt LLM embeddings.

    Prompt latents are encoded by the semantic encoder (one LLM-size row per
    patch) and kept raw in the patch history for the feedback recompute; the
    normalized patch rows are kept as the FM-history prompt patches. The first
    decode is the regenerated prompt patch, so its emission and EOS check are
    suppressed.
    """
    state = DotsTTSFlowState()
    state.backbone_cache = model.backbone.make_cache()
    if seed is not None:
        state.rng_key = mx.random.key(seed)
    if speaker_embedding is not None:
        state.g_cond = model.speaker_condition(speaker_embedding, speaker_scale)

    if prompt_latents is None or prompt_latents.shape[-2] == 0:
        return state, None

    embeddings = _semantic_encode(model, prompt_latents)
    prompt_count = int(prompt_latents.shape[-2] // model.latent_patch_size)
    if int(embeddings.shape[1]) != prompt_count:
        raise RuntimeError(
            "dots.tts MLX prompt embeddings do not match the prompt patch count: "
            f"{int(embeddings.shape[1])} != {prompt_count}"
        )
    state.patch_history.append(prompt_latents)
    state.prompt_patches = model.normalize(prompt_latents).reshape(
        1, prompt_count, model.latent_patch_size, model.latent_dim
    )
    state.drop_regenerated_prompt_patch = True
    state.suppress_first_eos_check = True
    return state, embeddings


def build_prefill_embeds(
    model: DotsTTSMlxModel,
    schedule_ids: mx.array,
    *,
    prompt_span_positions: mx.array,
    prompt_embeddings: mx.array,
) -> mx.array:
    """Token embeddings with semantic rows spliced in at the prompt spans."""
    token_embeds = model.backbone.embed_tokens(schedule_ids)  # [T, H]
    rows = []
    cursor = 0
    for index in range(int(prompt_span_positions.shape[0])):
        position = int(prompt_span_positions[index].item())
        rows.append(token_embeds[cursor:position])
        rows.append(prompt_embeddings[0, index : index + 1])
        cursor = position + 1
    rows.append(token_embeds[cursor:])
    return mx.concatenate(rows, axis=0)[None, :, :]


def _semantic_encode(model: DotsTTSMlxModel, raw_latents: mx.array) -> mx.array:
    """Causal full-compute semantic encoding of a raw latent block."""
    tokens = model.patch_encoder.downsample(raw_latents)
    token_count = int(tokens.shape[1])
    return model.patch_encoder.in_proj_and_trunk(tokens, mask=_tril_mask(token_count))


def initialize_history(
    model: DotsTTSMlxModel,
    state: DotsTTSFlowState,
    *,
    hidden_states: mx.array,
    prompt_span_positions: mx.array,
    generation_schedule: mx.array,
    audio_span_token_ids: set[int],
    prefill_end: int,
    decoded_latent_patches: list[mx.array],
) -> None:
    """Append prefill hidden rows and prompt patch rows into the flow history."""
    prompt_patches = state.prompt_patches
    positions = (
        prompt_span_positions[:0] if prompt_patches is None else prompt_span_positions
    )
    if (
        prompt_patches is not None
        and int(positions.shape[0]) != prompt_patches.shape[1]
    ):
        raise RuntimeError("dots.tts MLX prompt spans do not match prompt latents")
    schedule = generation_schedule
    cursor = 0

    for prompt_index in range(int(positions.shape[0])):
        span_position = int(positions[prompt_index].item())
        if span_position > cursor:
            _append_hidden(
                model, state, hidden_states[:, span_position - 1 : span_position]
            )
        _append_patch(model, state, prompt_patches[:, prompt_index])
        next_position = span_position + 1
        if (next_position < int(schedule.shape[0])) and (
            int(schedule[next_position].item()) in audio_span_token_ids
        ):
            _append_hidden(
                model, state, hidden_states[:, span_position : span_position + 1]
            )
        cursor = next_position
    if prefill_end > cursor:
        _append_hidden(model, state, hidden_states[:, prefill_end - 1 : prefill_end])

    for patch in decoded_latent_patches:
        _append_patch(model, state, model.normalize(patch))
        _append_hidden(
            model,
            state,
            hidden_states[:, prefill_end : prefill_end + 1],
        )
    if decoded_latent_patches:
        state.drop_regenerated_prompt_patch = False
        state.suppress_first_eos_check = False


def _append_hidden(
    model: DotsTTSMlxModel, state: DotsTTSFlowState, hidden: mx.array
) -> None:
    projected = model.hidden_proj(hidden)
    # note (guozhihao-224): the null-conditioned branch is hidden_proj(0),
    # the bias row broadcast over the sequence.
    null_projected = mx.broadcast_to(
        model.hidden_proj.bias[None, None, :], projected.shape
    )
    state.fm_history.append(projected)
    state.fm_cfg_history.append(null_projected)
    state.fm_seq_len += int(projected.shape[1])


def _append_patch(
    model: DotsTTSMlxModel, state: DotsTTSFlowState, patch_rows: mx.array
) -> None:
    projected = model.latent_proj(patch_rows)
    state.fm_history.append(projected)
    state.fm_cfg_history.append(projected)
    state.fm_seq_len += int(projected.shape[1])


def decode_step(
    model: DotsTTSMlxModel,
    state: DotsTTSFlowState,
    *,
    hidden_last: mx.array,
    num_steps: int,
    ode_method: str,
    guidance_scale: float,
    eos_threshold: float,
    append_hidden: bool = True,
) -> tuple[mx.array, mx.array, bool, bool]:
    """One audio patch: EOS check, DiT ODE, history append, next feedback.

    Returns (latent_patch, feedback, finished, emit) — the latent patch is
    denormalized (data-space), feedback is the semantic embedding feeding
    the backbone next step. append_hidden=False marks the prefill step,
    where initialize_history already appended the last prefill hidden row
    (mirrors the torch runner's append_hidden arg).
    """
    if ode_method != "euler":
        raise ValueError(f"dots.tts MLX currently requires euler, got {ode_method!r}")

    should_check_eos = not (
        state.suppress_first_eos_check and state.decoded_patches == 0
    )
    finished = False
    if should_check_eos:
        probabilities = model.eos_probability(hidden_last)
        finished = bool((probabilities > eos_threshold).any().item())

    if append_hidden:
        _append_hidden(model, state, hidden_last)
    normalized_patch = _decode_flow_matching(
        model, state, nfe=num_steps, guidance=guidance_scale
    )
    _append_patch(model, state, normalized_patch)
    state.decoded_patches += 1
    emit = not state.drop_regenerated_prompt_patch
    state.drop_regenerated_prompt_patch = False

    # note (guozhihao-224): the semantic encoder consumes raw data-space
    # latents, so the decoded patch joins the raw history first.
    state.patch_history.append(model.denormalize(normalized_patch))
    feedback = _next_feedback(model, state)
    return model.denormalize(normalized_patch), feedback, finished, emit


def _next_feedback(model: DotsTTSMlxModel, state: DotsTTSFlowState) -> mx.array:
    """Semantic-encode the full raw history; take the newest patch's row."""
    if not state.patch_history:
        raise RuntimeError("dots.tts MLX feedback has no patch history")
    raw = mx.concatenate(state.patch_history, axis=1)
    embeddings = _semantic_encode(model, raw)
    return embeddings[:, -1:, :]


def _tril_mask(length: int) -> mx.array:
    rows = mx.arange(length)[:, None]
    cols = mx.arange(length)[None, :]
    valid = cols <= rows
    return mx.where(
        valid,
        mx.zeros((length, length)),
        mx.full((length, length), _MASK_VALUE),
    )[None, None, :, :]


def _decode_flow_matching(
    model: DotsTTSMlxModel,
    state: DotsTTSFlowState,
    *,
    nfe: int,
    guidance: float,
) -> mx.array:
    """Integrate the classifier-free-flow ODE from t=0 to 1 on the patch slot."""
    fm_seq_len = int(state.fm_seq_len)
    patch_size = int(model.latent_patch_size)
    total_len = fm_seq_len + patch_size
    latent_start = fm_seq_len  # patch slot starts right after the history

    input_sequence = _build_input(model, state.fm_history, fm_seq_len, total_len)
    cfg_sequence = _build_input(model, state.fm_cfg_history, fm_seq_len, total_len)
    mask = _decode_mask(total_len=total_len, fm_seq_len=fm_seq_len)

    z = _sample_noise(model, state)
    step_size = 1.0 / nfe
    for index in range(nfe):
        t = step_size * index
        timesteps = mx.full((2,), t)
        z_proj = model.coordinate_proj(z)
        x = mx.concatenate(
            [
                _splat(input_sequence, z_proj, latent_start),
                _splat(cfg_sequence, z_proj, latent_start),
            ],
            axis=0,
        )
        if state.g_cond is not None:
            g_cond_branches = mx.concatenate(
                [state.g_cond, mx.zeros_like(state.g_cond)], axis=0
            )
        else:
            g_cond_branches = mx.zeros((2, model.fm_hidden_size))
        pred = model.velocity_field_predictor(
            x, timesteps, g_cond=g_cond_branches, mask=mask
        )[:, latent_start:]
        cond = pred[:1]
        uncond = pred[1:]
        velocity = cond + guidance * (cond - uncond)
        z = z + step_size * velocity
    return z


def _build_input(
    model: DotsTTSMlxModel,
    history: list[mx.array],
    fm_seq_len: int,
    total_len: int,
) -> mx.array:
    # note (guozhihao-224): history is never empty (prefill seeds it) and the
    # patch slot always pads; concatenate the rows and the zero patch slot.
    rows = mx.concatenate(history, axis=1)
    pad = mx.zeros((1, total_len - fm_seq_len, model.fm_hidden_size))
    return mx.concatenate([rows, pad], axis=1)


def _splat(sequence: mx.array, z_proj: mx.array, latent_start: int) -> mx.array:
    return mx.concatenate([sequence[:, :latent_start, :], z_proj], axis=1)


def _sample_noise(model: DotsTTSMlxModel, state: DotsTTSFlowState) -> mx.array:
    if state.rng_key is None:
        return mx.random.normal((1, int(model.latent_patch_size), model.latent_dim))
    new_key, sample_key = mx.random.split(state.rng_key)
    state.rng_key = new_key
    return mx.random.normal(
        (1, int(model.latent_patch_size), model.latent_dim),
        key=sample_key,
    )


def _decode_mask(total_len: int, fm_seq_len: int) -> mx.array:
    """Additive attention bias mirroring _build_decode_mask."""
    q = mx.arange(total_len)[:, None]
    k = mx.arange(total_len)[None, :]
    latent_start = fm_seq_len
    block_start = max(0, fm_seq_len - 1)

    causal_prefix = (q < block_start) & (k <= q)
    tail_block = (
        (q >= block_start) & (q < fm_seq_len) & ((k < fm_seq_len) | (k >= latent_start))
    )
    patch_rows = (q >= latent_start) & ((k < fm_seq_len) | (k >= latent_start))
    ok = causal_prefix | tail_block | patch_rows
    return mx.where(
        ok,
        mx.zeros((total_len, total_len)),
        mx.full((total_len, total_len), _MASK_VALUE),
    )[None, None, :, :]


__all__ = [
    "build_prefill_embeds",
    "decode_step",
    "initialize_history",
    "start_request",
]
