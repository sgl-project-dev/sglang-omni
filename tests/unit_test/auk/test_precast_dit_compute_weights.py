# SPDX-License-Identifier: Apache-2.0

import copy
import inspect

import pytest
import torch
from torch import nn

from sglang_omni.models.auk import stages
from sglang_omni.models.auk.config import AuKPipelineConfig
from sglang_omni.models.auk.dit import AuKDit


def _model():
    torch.manual_seed(71)
    model = AuKDit(
        dim=32,
        heads=2,
        dim_head=16,
        latent_dim=8,
        text_hidden_dim=16,
        num_layers=1,
        num_single_layers=1,
        dropout=0.0,
    ).eval()
    for parameter in model.parameters():
        nn.init.uniform_(parameter, -0.2, 0.2)
    return model


def _inputs(with_ref):
    torch.manual_seed(73)
    inputs = {
        "x": torch.randn(1, 5, 8),
        "text": torch.randn(1, 3, 16),
        "time": torch.tensor(0.4),
        "mask": torch.ones(1, 5, dtype=torch.bool),
        "c_mask": torch.ones(1, 3, dtype=torch.bool),
        "cfg_infer": True,
    }
    if with_ref:
        inputs.update(
            ref=torch.randn(1, 2, 8),
            ref_mask=torch.ones(1, 2, dtype=torch.bool),
        )
    return inputs


def test_default_false_and_scope_are_explicit():
    engine = next(
        stage
        for stage in AuKPipelineConfig.model_fields["stages"].default
        if stage.name == "auk_engine"
    )
    assert engine.factory.precast_dit_compute_weights is False
    parameters = inspect.signature(stages._load_flow.__wrapped__).parameters
    assert "compute_weight_dtype" in parameters
    source = inspect.getsource(stages.create_auk_engine_executor)
    assert "flow.transformer" not in source
    assert '"bfloat16" if precast_dit_compute_weights else None' in source
    load_source = inspect.getsource(stages._load_flow)
    assert "flow.transformer" in load_source
    assert "_load_vae" not in load_source
    assert "AuKConditionEncoder" not in load_source


def test_precast_changes_only_dit_linear_and_conv_parameters():
    model = _model()
    before = {
        name: (value.dtype, value.shape) for name, value in model.state_dict().items()
    }
    targeted = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, (nn.Linear, nn.Conv1d))
        for parameter in (module.weight, module.bias)
        if parameter is not None
    }
    stages._precast_dit_compute_weights(model, torch.bfloat16)

    assert set(model.state_dict()) == set(before)
    for name, value in model.state_dict().items():
        assert value.shape == before[name][1]
    for parameter in model.parameters():
        expected = torch.bfloat16 if id(parameter) in targeted else torch.float32
        assert parameter.dtype == expected
    assert targeted
    assert any(
        parameter.dtype == torch.float32
        for name, parameter in model.named_parameters()
        if "norm" in name.lower()
    )


@pytest.mark.parametrize("with_ref", [False, True])
def test_precast_is_bit_exact_under_bf16_autocast(with_ref):
    eager = _model()
    precast = copy.deepcopy(eager)
    stages._precast_dit_compute_weights(precast, torch.bfloat16)
    inputs = _inputs(with_ref)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = eager(**inputs)
        actual = precast(**inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
