# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import types
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.audio_decoder import (
    FishQwen3AudioDecoder,
)
from sglang_omni.models.fishaudio_s2_pro.model_runner import FishS2ProModelRunner
from sglang_omni.models.fishaudio_s2_pro.prefill import prefill_vq_inputs


def fixture(reference=True, generated=7):
    model = SimpleNamespace(_vq_codes=torch.zeros((1, 2), dtype=torch.long))
    model.get_embed_tokens = lambda: lambda ids: ids.float().unsqueeze(1).repeat(1, 3)
    decoder = SimpleNamespace(
        codebook_offsets=torch.tensor([0, 100]),
        codebook_embeddings=torch.nn.Embedding.from_pretrained(
            torch.arange(600).float().reshape(200, 3)
        ),
        config=SimpleNamespace(num_codebooks=2),
    )
    decoder.embed_text_dim = types.MethodType(
        FishQwen3AudioDecoder.embed_text_dim, decoder
    )
    model._audio_decoder = decoder
    runner = object.__new__(FishS2ProModelRunner)
    runner.model = model
    runner._semantic_begin_id, runner._semantic_end_id = 100, 199
    prompt = [10] * 155
    mask = torch.zeros(155, dtype=torch.bool) if reference else None
    if reference:
        mask[1] = True
    output = [100 + i for i in range(generated)]
    frames = [torch.tensor([[token], [i], [i + 1]]) for i, token in enumerate(output)]
    data = SimpleNamespace(
        req=SimpleNamespace(
            origin_input_ids=prompt,
            output_ids=output,
            prefix_indices=[],
            extend_range=SimpleNamespace(length=155 + generated),
        ),
        vq_mask_tokens=mask,
        vq_parts=[torch.tensor([[3], [7]])] if reference else None,
        output_codes=frames,
        semantic_history_tokens=torch.tensor(output[-4:]),
        semantic_history_count=generated,
    )
    return runner, data


def expected(runner, data):
    tokens = data.req.origin_input_ids + data.req.output_ids
    result = runner.model.get_embed_tokens()(torch.tensor(tokens))
    rows = {}
    if data.vq_parts:
        rows[1] = data.vq_parts[0][:, 0]
    frame_by_token = {int(f[0, 0]): f[1:, 0] for f in data.output_codes}
    for i, token in enumerate(data.req.output_ids):
        if 100 <= token <= 199:
            rows[155 + i] = frame_by_token[token]
    decoder = runner.model._audio_decoder
    for index, codes in rows.items():
        result[index] = (
            result[index]
            + decoder.codebook_embeddings(codes + decoder.codebook_offsets).sum(0)
        ) / math.sqrt(3)
    return result


@pytest.mark.parametrize("reference", [True, False])
@pytest.mark.parametrize("generated", [7, 40])
@pytest.mark.parametrize("start", [0, 2, 154, 156])
def test_rebuild_and_partial_cached_prefix(reference, generated, start):
    runner, data = fixture(reference, generated)
    original = None if data.vq_mask_tokens is None else data.vq_mask_tokens.clone()
    want = expected(runner, data)
    # Separate extends can end inside the prompt or generated tail.
    for end in sorted(set([max(start + 1, 155), len(want)])):
        data.req.prefix_indices = list(range(start))
        data.req.extend_range.length = end - start
        batch = SimpleNamespace(
            input_ids=torch.tensor(
                (data.req.origin_input_ids + data.req.output_ids)[start:end]
            )
        )
        for _ in range(2):
            got = runner._build_prefill_input_embeds(
                batch, [SimpleNamespace(data=data)]
            )
            assert torch.equal(got, want[start:end])
        assert data.semantic_history_count == generated
        assert len(data.output_codes) == generated
        assert original is None or torch.equal(data.vq_mask_tokens, original)


def test_two_retractions_rederive_instead_of_appending():
    runner, data = fixture()
    for generated in (7, 12):
        while len(data.output_codes) < generated:
            i = len(data.output_codes)
            data.req.output_ids.append(100 + i)
            data.output_codes.append(torch.tensor([[100 + i], [i], [i + 1]]))
        data.req.extend_range.length = 155 + generated
        batch = SimpleNamespace(
            input_ids=torch.tensor(data.req.origin_input_ids + data.req.output_ids)
        )
        assert torch.equal(
            runner._build_prefill_input_embeds(batch, [SimpleNamespace(data=data)]),
            expected(runner, data),
        )
        assert len(data.vq_mask_tokens) == 155
        assert data.vq_parts[0].shape == (2, 1)


def test_control_and_eos_positions_do_not_consume_vq_rows():
    runner, data = fixture(False, 2)
    data.req.output_ids = [100, 9, 101, 2]
    data.req.extend_range.length = 159
    batch = SimpleNamespace(
        input_ids=torch.tensor(data.req.origin_input_ids + data.req.output_ids)
    )
    assert torch.equal(
        runner._build_prefill_input_embeds(batch, [SimpleNamespace(data=data)]),
        expected(runner, data),
    )


@pytest.mark.parametrize("fault", ["missing", "wrong_token", "overrun", "all_missing"])
def test_reject_uncommitted_or_misaligned_frames(fault):
    runner, data = fixture(False, 2)
    if fault == "missing":
        data.output_codes.pop()
    elif fault == "wrong_token":
        data.output_codes[0][0] = 199
    elif fault == "overrun":
        data.output_codes.append(torch.tensor([[102], [2], [3]]))
    else:
        data.output_codes.clear()
    with pytest.raises(ValueError, match="committed|codec"):
        prefill_vq_inputs(
            data, device="cpu", num_codebooks=2, semantic_begin=100, semantic_end=199
        )


def test_155_plus_7_rebuild_includes_generated_vq_embeddings():
    runner, data = fixture()
    batch = SimpleNamespace(
        input_ids=torch.tensor(data.req.origin_input_ids + data.req.output_ids)
    )
    got = runner._build_prefill_input_embeds(batch, [SimpleNamespace(data=data)])
    assert got.shape == (162, 3)
    assert torch.equal(got, expected(runner, data))
    text_only = runner.model.get_embed_tokens()(batch.input_ids)
    assert not torch.equal(got[155:], text_only[155:])
