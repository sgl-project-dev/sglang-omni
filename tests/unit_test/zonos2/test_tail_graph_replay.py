# SPDX-License-Identifier: Apache-2.0
"""The real per-frame tail is capturable and replayable on the host's accelerator.

test_tail_graph_capture.py proves the routing with stand-ins; this proves the
thing being routed actually records. Every op in ``sample_tts`` has to be legal
inside a capture -- ``torch.multinomial`` above all, since an RNG op that is not
graph-safe either refuses to capture or freezes the draw, and a frozen draw is a
TTS model that emits the same frame forever. Backend-neutral on purpose: the
graph API differs per accelerator, which is exactly what could break.
"""

from __future__ import annotations

import pytest
import torch

import sglang_omni.platforms as platforms
from sglang_omni.models.zonos2.sampler import sample_tts
from tests.unit_test.fixtures.accelerator import require_accelerator

pytestmark = pytest.mark.accelerator

N_CODEBOOKS = 9
AUDIO_VOCAB = 1026
TOP_K = 106


def _static_tail(bs: int, device: str):
    """The static buffers a captured tail reads and writes, ZONOS2's own shapes."""
    return {
        "logits": torch.zeros(bs, N_CODEBOOKS, AUDIO_VOCAB, device=device),
        "temperature": torch.full((bs,), 1.15, device=device),
        "top_k": torch.full((bs,), TOP_K, device=device, dtype=torch.long),
        "top_p": torch.zeros(bs, device=device),
        "min_p": torch.full((bs,), 0.18, device=device),
        "rep_pen": torch.full((bs,), 1.2, device=device),
        "codes": torch.zeros(bs, N_CODEBOOKS, device=device, dtype=torch.long),
    }


def test_the_tail_sampler_captures_and_redraws_on_replay() -> None:
    device_type = require_accelerator()
    device = f"{device_type}:0"
    backend = platforms.current_platform.get_device_graph_backend(torch.device(device))
    assert backend is not None, f"{device_type} names no model-owned graph backend"

    bs = 2
    buf = _static_tail(bs, device)
    module = torch.get_device_module(device)

    def tail() -> None:
        buf["codes"].copy_(
            sample_tts(
                buf["logits"],
                temperature=buf["temperature"],
                top_k=buf["top_k"],
                top_p=buf["top_p"],
                min_p=buf["min_p"],
                repetition_penalty=buf["rep_pen"],
                top_k_max=TOP_K,
                rep_token_ids=None,
                any_top_p=False,
                any_min_p=True,
            )
        )

    side = module.Stream()
    side.wait_stream(module.current_stream())
    with module.stream(side):
        for _ in range(3):
            tail()
    module.current_stream().wait_stream(side)
    module.synchronize()

    with backend.capture() as graph:
        tail()
    module.synchronize()

    for target in (11, 907):
        buf["logits"].fill_(0.0)
        buf["logits"][:, :, target] = 30.0
        graph.replay()
        module.synchronize()
        assert buf["codes"].unique().tolist() == [target]

    buf["logits"].fill_(0.0)
    draws = []
    for _ in range(8):
        graph.replay()
        module.synchronize()
        draws.append(buf["codes"].clone())
    assert all(int(d.min()) >= 0 and int(d.max()) < AUDIO_VOCAB for d in draws)
    assert any(
        not torch.equal(draws[0], d) for d in draws[1:]
    ), "replay froze the RNG: every frame drew the same codes"
