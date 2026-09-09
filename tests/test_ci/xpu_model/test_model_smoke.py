# SPDX-License-Identifier: Apache-2.0
"""Minimal functional smoke: every model Intel XPU serves answers one request.

Content is checked, not just status; the request shapes and their assertions live
in exercises.py. No WER, MOS, or throughput floors -- those need the eval extra
and belong in tests/test_model/.

    pytest tests/test_ci/xpu_model -v                     # every model with weights
    pytest tests/test_ci/xpu_model -v -k "asr or tts"     # a subset
    REQUIRE_FREE_GPU=0 pytest tests/test_ci/xpu_model -v  # ignore who else is on the card
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.test_ci.xpu_model.conftest import resolve_checkpoint
from tests.test_ci.xpu_model.exercises import EXERCISES
from tests.test_ci.xpu_model.model_table import XPU_MODELS, XpuModel

pytestmark = pytest.mark.xpu_serving


# --------------------------------------------------------------- serve topology


def _serve_plan(
    model: XpuModel, claim: Callable[[int], list[int]]
) -> tuple[list[str], dict[str, str]]:
    """The extra serve arguments and environment this topology needs.

    Placement uses stage-scoped overrides (``--thinker.tp_size``); the
    ``--thinker-tp-size`` spelling belongs to examples/run_omni.py and ``cli
    serve`` rejects it. A stage's gpu_id indexes into ZE_AFFINITY_MASK, not the
    host, so the mask is set here rather than inherited.
    """
    cards = claim(model.cards)
    mask = ",".join(str(card) for card in cards)

    if model.cards == 1:
        return [], {"ZE_AFFINITY_MASK": mask}

    # The whole group stays visible: every TP rank must see its peers for XCCL
    # discovery, and the stages address them by position within the mask.
    lanes = list(range(model.cards))
    gpu_list = ",".join(str(lane) for lane in lanes)
    env = {"SGLANG_OMNI_STARTUP_TIMEOUT": "1800", "ZE_AFFINITY_MASK": mask}

    if model.kind == "chat":
        # --text-only serves chat without the talker and vocoder.
        return (
            [
                "--text-only",
                "--thinker.tp_size",
                str(model.cards),
                "--thinker.gpu",
                f"[{gpu_list}]",
                "--thinker.process",
                "thinker",
            ],
            env,
        )

    # The talker and vocoder take the last two thinker cards, whose static
    # fractions drop to leave them room. code2wav takes 0.05 rather than the
    # pipeline's 0.02: the fraction is of the whole card, and 0.02 of 24 GB leaves
    # less than the capture keys need, so the vocoder graphs would decline.
    return (
        [
            "--image_encoder.gpu",
            str(lanes[0]),
            "--audio_encoder.gpu",
            str(lanes[0]),
            "--thinker.tp_size",
            str(model.cards),
            "--thinker.gpu",
            f"[{gpu_list}]",
            "--thinker.engine.mem_fraction_static",
            "0.55",
            "--talker_ar.gpu",
            str(lanes[-2]),
            "--talker_ar.engine.mem_fraction_static",
            "0.35",
            "--code2wav.gpu",
            str(lanes[-1]),
            "--code2wav.gpu_memory_fraction",
            "0.05",
        ],
        env,
    )


@pytest.mark.parametrize("model", XPU_MODELS, ids=[m.key for m in XPU_MODELS])
def test_the_model_serves_and_answers_on_xpu(
    model: XpuModel, xpu_host: None, serve_xpu_model, claim_cards
) -> None:
    """Serve ``model`` on the XPU, send one request, and check what came back."""
    checkpoint = resolve_checkpoint(model)
    extra_args, env = _serve_plan(model, claim_cards)

    with serve_xpu_model(model, checkpoint, extra_args, env) as base_url:
        verdict = EXERCISES[model.kind](base_url, checkpoint, model)

    print(f"\n{model.key}: {verdict}")
