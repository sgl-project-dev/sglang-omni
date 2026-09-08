# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the text-only thinker's fused expert weight loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
    Qwen3OmniThinkerForCausalLM,
    _expert_parameter_mapping,
    _load_expert_parameter,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "model.layers.0.mlp.experts.3.gate_proj.weight_scale",
            ("model.layers.0.mlp.experts.w13_weight_scale", "w1", 3),
        ),
        (
            "model.layers.0.mlp.experts.3.down_proj.weight_scale",
            ("model.layers.0.mlp.experts.w2_weight_scale", "w2", 3),
        ),
    ],
)
def test_expert_parameter_mapping_routes_group_scales(name, expected):
    mappings = [
        ("experts.w13_weight", "experts.3.gate_proj.weight", 3, "w1"),
        ("experts.w2_weight", "experts.3.down_proj.weight", 3, "w2"),
    ]

    assert _expert_parameter_mapping(name, mappings) == expected


def test_expert_parameter_loader_uses_positional_coordinates():
    calls: list[tuple[object, ...]] = []
    param = torch.nn.Parameter(torch.empty(1), requires_grad=False)
    param.weight_loader = lambda *args: calls.append(args)
    loaded = torch.tensor([127], dtype=torch.uint8)

    _load_expert_parameter(
        param,
        loaded,
        "model.layers.0.mlp.experts.w13_weight_scale",
        "w1",
        3,
    )

    assert calls == [
        (
            param,
            loaded,
            "model.layers.0.mlp.experts.w13_weight_scale",
            "w1",
            3,
        )
    ]


def test_thinker_routes_mxfp_expert_scales_to_fused_parameters():
    calls: list[tuple[str, torch.Tensor, str, int]] = []
    w13_scale = torch.nn.Parameter(torch.empty(1), requires_grad=False)
    w2_scale = torch.nn.Parameter(torch.empty(1), requires_grad=False)
    w13_scale.weight_loader = lambda _p, loaded, name, shard, expert: calls.append(
        (name, loaded, shard, expert)
    )
    w2_scale.weight_loader = lambda _p, loaded, name, shard, expert: calls.append(
        (name, loaded, shard, expert)
    )

    wrapper = object.__new__(Qwen3OmniThinkerForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(num_experts=1)
    wrapper.root_config = SimpleNamespace(quantization_config=None)
    wrapper.named_parameters = lambda: iter(
        (
            ("model.layers.0.mlp.experts.w13_weight_scale", w13_scale),
            ("model.layers.0.mlp.experts.w2_weight_scale", w2_scale),
        )
    )
    gate_scale = torch.tensor([127], dtype=torch.uint8)
    down_scale = torch.tensor([126], dtype=torch.uint8)

    wrapper.load_weights(
        iter(
            (
                (
                    "thinker.model.layers.0.mlp.experts.0.gate_proj.weight_scale",
                    gate_scale,
                ),
                (
                    "thinker.model.layers.0.mlp.experts.0.down_proj.weight_scale",
                    down_scale,
                ),
            )
        )
    )

    assert calls == [
        ("model.layers.0.mlp.experts.w13_weight_scale", gate_scale, "w1", 0),
        ("model.layers.0.mlp.experts.w2_weight_scale", down_scale, "w2", 0),
    ]
