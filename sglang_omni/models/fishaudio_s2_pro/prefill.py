# SPDX-License-Identifier: Apache-2.0
"""Rebuild Fish prefill VQ inputs from the final committed token prefix."""

from __future__ import annotations

import torch


def prefill_vq_inputs(data, *, device, num_codebooks, semantic_begin, semantic_end):
    """Return an immutable full-prefix mask and VQ rows in token-position order."""
    raw_mask = data.vq_mask_tokens
    committed = getattr(data, "output_codes", [])
    output_ids = list(getattr(data.req, "output_ids", []))
    if raw_mask is None and not committed and not output_ids:
        return None, None
    if raw_mask is None:
        mask = torch.zeros(
            len(data.req.origin_input_ids), dtype=torch.bool, device=device
        )
    else:
        mask = torch.as_tensor(raw_mask, device=device, dtype=torch.bool).reshape(-1)

    parts = data.vq_parts or []
    if any(part.ndim != 2 or part.shape[0] != num_codebooks for part in parts):
        raise ValueError("Fish reference VQ must have shape [codebooks, frames]")
    rows = (
        torch.cat([part.to(device=device) for part in parts], dim=1).T
        if parts
        else torch.empty((0, num_codebooks), dtype=torch.long, device=device)
    )
    if rows.shape[0] != int(mask.sum()):
        raise ValueError("Fish prompt VQ mask and reference frames are misaligned")
    if not committed and not output_ids:
        return mask, rows

    if mask.numel() != len(data.req.origin_input_ids):
        raise ValueError("Fish prompt VQ mask does not cover the original prompt")
    frames = (
        torch.cat(committed, dim=1).to(device=device)
        if committed
        else torch.empty((num_codebooks + 1, 0), dtype=torch.long, device=device)
    )
    if frames.ndim != 2 or frames.shape[0] != num_codebooks + 1:
        raise ValueError("Fish committed codec frames must include semantic IDs")

    # Prefill runs after drain. Validate against scheduler-accepted token IDs,
    # not device lookahead buffers or the bounded repetition-history window.
    frame_tokens = frames[0].tolist()
    cursor = 0
    tail_mask, tail_rows = [], []
    for token in output_ids:
        needs_vq = semantic_begin <= token <= semantic_end
        has_frame = cursor < len(frame_tokens) and frame_tokens[cursor] == token
        if needs_vq and not has_frame:
            raise ValueError(
                "Fish committed semantic token has no matching codec frame"
            )
        tail_mask.append(needs_vq)
        if has_frame:
            if needs_vq:
                tail_rows.append(frames[1:, cursor])
            cursor += 1
    if cursor != len(frame_tokens):
        raise ValueError(
            "Fish codec frames extend beyond or disagree with committed tokens"
        )
    mask = torch.cat((mask, torch.tensor(tail_mask, dtype=torch.bool, device=device)))
    if tail_rows:
        rows = torch.cat((rows, torch.stack(tail_rows)))
    return mask, rows


def build_prefill_input_embeds(runner, forward_batch, requests):
    input_ids = forward_batch.input_ids
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("Fish prefill expects tensor input_ids")
    model = runner.model
    embeds = model.get_embed_tokens()(input_ids)
    offset = 0
    for request in requests:
        data = request.data
        length = int(data.req.extend_range.length)
        if (
            data.vq_mask_tokens is None
            and not getattr(data, "output_codes", [])
            and not getattr(data.req, "output_ids", [])
        ):
            offset += length
            continue
        mask, rows = prefill_vq_inputs(
            data,
            device=input_ids.device,
            num_codebooks=model._vq_codes.shape[1],
            semantic_begin=runner._semantic_begin_id,
            semantic_end=runner._semantic_end_id,
        )
        if mask is not None:
            start = len(data.req.prefix_indices)
            end = start + length
            if end > mask.numel():
                raise ValueError("Fish prefill extends beyond the committed VQ input")
            selected = mask[start:end]
            if bool(selected.any()):
                first = int(mask[:start].sum())
                count = int(selected.sum())
                injected = model._audio_decoder.embed_text_dim(
                    embeds[offset : offset + length].unsqueeze(0),
                    rows[first : first + count],
                    selected.unsqueeze(0),
                )
                embeds[selected.nonzero(as_tuple=True)[0] + offset] = injected.to(
                    embeds.dtype
                )
        offset += length
    if offset != input_ids.numel():
        raise ValueError("Fish prefill request lengths do not cover input IDs")
    return embeds
