# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import SnakeBeta

from sglang_omni.models.qwen3_omni.components import fused_snake_beta, snake_beta


def _model(dtype=torch.float32, device="cpu"):
    model = (
        nn.Sequential(SnakeBeta(4), SnakeBeta(4)).to(device=device, dtype=dtype).eval()
    )
    with torch.no_grad():
        for index, module in enumerate(model):
            module.alpha.copy_(torch.linspace(-1, 1, 4, device=device) + index / 4)
            module.beta.copy_(torch.linspace(-0.5, 0.5, 4, device=device))
    return model


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_hoist_preserves_eager_values_and_checkpoint_parameters(dtype):
    model = _model(dtype)
    inputs = torch.linspace(-5, 5, 104, dtype=dtype).reshape(2, 4, 13)
    parameters = tuple(model.parameters())
    checkpoint = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        expected = model(inputs)

    assert snake_beta.install_code2wav_snake_beta(model, "hoist") == ("0", "1")

    with torch.no_grad():
        model._omni_snake_beta_guard()
        assert torch.equal(model(inputs), expected)
    assert all(left is right for left, right in zip(parameters, model.parameters()))
    assert model.state_dict().keys() == checkpoint.keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, checkpoint[name])


@pytest.mark.parametrize("implementation", ["eager", "unknown", None, True])
def test_installer_rejects_unknown_implementations(implementation):
    with pytest.raises(ValueError, match="requires 'hoist' or 'fused'"):
        snake_beta.install_code2wav_snake_beta(_model(), implementation)


@pytest.mark.parametrize(
    "change", ["training", "hook", "shared_module", "shared_parameter"]
)
def test_installer_rejects_unsupported_models_without_partial_installation(change):
    model = _model()
    if change == "training":
        model[1].train()
    elif change == "hook":
        model[1].register_forward_hook(lambda *args: None)
    elif change == "shared_module":
        model[1] = model[0]
    else:
        model[1].alpha = model[0].alpha

    with pytest.raises((ValueError, RuntimeError)):
        snake_beta.install_code2wav_snake_beta(model, "hoist")

    assert "_omni_snake_beta_guard" not in vars(model)
    assert all("_omni_snake_beta_state" not in vars(module) for module in model)
    assert all("forward" not in vars(module) for module in model)


def test_installer_rolls_back_when_binding_the_second_module_fails(monkeypatch):
    model = _model()
    original_bind = snake_beta.MethodType
    calls = 0

    def bind(function, module):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected binding failure")
        return original_bind(function, module)

    monkeypatch.setattr(snake_beta, "MethodType", bind)
    with pytest.raises(RuntimeError, match="injected binding failure"):
        snake_beta.install_code2wav_snake_beta(model, "hoist")
    assert "_omni_snake_beta_guard" not in vars(model)
    assert "_omni_snake_beta_modules" not in vars(model)
    for module in model:
        assert "_omni_snake_beta_state" not in vars(module)
        assert "forward" not in vars(module)
        assert module.forward.__func__ is SnakeBeta.forward


@pytest.mark.parametrize(
    "change", ["inplace", "replace_parameter", "dtype", "replace_module", "forward"]
)
def test_guard_rejects_parameter_or_module_changes_before_replay(change):
    model = _model()
    snake_beta.install_code2wav_snake_beta(model, "hoist")
    with torch.no_grad():
        model._omni_snake_beta_guard()
        if change == "inplace":
            model[0].alpha.add_(1)
        elif change == "replace_parameter":
            model[0].beta = nn.Parameter(model[0].beta.clone())
        elif change == "dtype":
            model.to(dtype=torch.float64)
        elif change == "replace_module":
            model[0] = SnakeBeta(4).eval()
        else:
            model[0].forward = lambda value: value
        with pytest.raises(RuntimeError, match="changed"):
            model._omni_snake_beta_guard()


def test_hoist_requires_inference_and_cannot_be_installed_twice():
    model = _model()
    snake_beta.install_code2wav_snake_beta(model, "hoist")
    with torch.enable_grad(), pytest.raises(RuntimeError, match="disabled gradients"):
        model(torch.zeros(1, 4, 2))
    with pytest.raises(ValueError, match="not installed"):
        snake_beta.install_code2wav_snake_beta(model, "hoist")


@pytest.mark.parametrize("shape", [(4, 2), (1, 0, 2), (1, 3, 2)])
def test_hoist_rejects_invalid_input_shapes(shape):
    model = _model()
    snake_beta.install_code2wav_snake_beta(model, "hoist")
    with torch.no_grad(), pytest.raises(ValueError, match="contiguous NCT"):
        model(torch.zeros(shape))


@pytest.mark.parametrize("kind", ["strided", "dtype", "non_tensor"])
def test_hoist_rejects_incompatible_inputs(kind):
    model = _model()
    snake_beta.install_code2wav_snake_beta(model, "hoist")
    inputs = {
        "strided": torch.zeros(1, 4, 4)[:, :, ::2],
        "dtype": torch.zeros(1, 4, 2, dtype=torch.float64),
        "non_tensor": [[[0, 0]] * 4],
    }
    with torch.no_grad(), pytest.raises(ValueError, match="contiguous NCT"):
        model(inputs[kind])


def test_fused_rejects_cpu_installation_without_mutating_model():
    model = _model(torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA BF16"):
        snake_beta.install_code2wav_snake_beta(model, "fused")
    assert all("forward" not in vars(module) for module in model)


def test_scheduler_checks_installed_state_before_graph_dispatch():
    # Note (wenyao): graph replay bypasses Python forwards, so the state guard
    # must run before graph dispatch.
    from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
        Code2WavScheduler,
    )

    model = _model()
    snake_beta.install_code2wav_snake_beta(model, "hoist")
    scheduler = object.__new__(Code2WavScheduler)
    scheduler._device = torch.device("cpu")
    scheduler._snake_beta_guard = model._omni_snake_beta_guard
    dispatched = []
    scheduler._cuda_graph_runner = SimpleNamespace(
        run=lambda *a, **k: dispatched.append(a)
    )
    with torch.no_grad():
        model[0].alpha.add_(1)
    with pytest.raises(RuntimeError, match="changed"):
        scheduler._forward_codes(torch.zeros(1, 4, 2), graph_eligible=True)
    assert dispatched == []


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA is required")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is required")
    return torch.device("cuda")


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "batch,length,scale", [(1, 2, 0.01), (16, 22, 1.0), (32, 129, 10.0)]
)
def test_fused_installed_model_matches_eager_bf16(cuda_device, batch, length, scale):
    model = _model(torch.bfloat16, cuda_device)
    generator = torch.Generator(device=cuda_device).manual_seed(20260905)
    inputs = (
        torch.randn(
            batch,
            4,
            length,
            device=cuda_device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * scale
    )
    with torch.no_grad():
        expected = model(inputs)
    snake_beta.install_code2wav_snake_beta(model, "fused")
    with torch.no_grad():
        actual = model(inputs)
    assert torch.equal(actual, expected)
    assert actual.data_ptr() != inputs.data_ptr()


@pytest.mark.accelerator
def test_fused_graph_replay_on_a_side_stream_matches_eager(cuda_device):
    model = _model(torch.bfloat16, cuda_device)
    eager = _model(torch.bfloat16, cuda_device)
    snake_beta.install_code2wav_snake_beta(model, "fused")
    inputs = torch.zeros(4, 4, 17, device=cuda_device, dtype=torch.bfloat16)
    stream = torch.cuda.Stream(device=cuda_device)
    stream.wait_stream(torch.cuda.current_stream(cuda_device))
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(2):
            model(inputs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = model(inputs)
        for value in (0.1, -3.0):
            inputs.fill_(value)
            model._omni_snake_beta_guard()
            graph.replay()
            expected = eager(inputs)
            stream.synchronize()
            assert torch.equal(actual, expected)


@pytest.mark.accelerator
@pytest.mark.parametrize("kind", ["dtype", "shape", "strided", "grad"])
def test_fused_rejects_unsupported_cuda_inputs(cuda_device, kind):
    inputs = torch.zeros(1, 4, 6, device=cuda_device, dtype=torch.bfloat16)
    alpha = inverse = torch.ones(1, 4, 1, device=cuda_device, dtype=torch.bfloat16)
    if kind == "dtype":
        inputs = inputs.float()
    elif kind == "shape":
        alpha = alpha.flatten()
    elif kind == "strided":
        inputs = inputs[:, :, ::2]
    else:
        inputs.requires_grad_(True)
    with torch.enable_grad(), pytest.raises((ValueError, RuntimeError)):
        fused_snake_beta.fused_snake_beta(inputs, alpha, inverse)


def _nested_snake_model():
    return nn.Sequential(
        nn.ModuleDict(
            {"left": nn.Sequential(SnakeBeta(4)), "right": nn.Sequential(SnakeBeta(4))}
        )
    ).eval()


@pytest.mark.parametrize("change", ["ancestor", "shadow", "leaf", "parameter"])
def test_guard_rejects_nested_changes_before_replay(change):
    model = _nested_snake_model()
    snake_beta.install_code2wav_snake_beta(model, "hoist")
    with torch.no_grad():
        model._omni_snake_beta_guard()
        if change == "ancestor":
            model[0] = nn.ModuleDict(
                {
                    "left": nn.Sequential(SnakeBeta(4)),
                    "right": nn.Sequential(SnakeBeta(4)),
                }
            ).eval()
        elif change == "shadow":
            object.__setattr__(model[0], "left", nn.Sequential(SnakeBeta(4)).eval())
        elif change == "leaf":
            model[0]["right"][0] = SnakeBeta(4).eval()
        else:
            model[0]["right"][0].beta.add_(1)
        with pytest.raises(RuntimeError, match="changed"):
            model._omni_snake_beta_guard()


def test_guard_accepts_equivalent_nested_registration_mapping():
    model = _nested_snake_model()
    snake_beta.install_code2wav_snake_beta(model, "hoist")
    model[0]._modules = dict(model[0]._modules)
    with torch.no_grad():
        model._omni_snake_beta_guard()


def test_guard_handles_root_snake_and_parameter_mutation():
    model = SnakeBeta(4).eval()
    assert snake_beta.install_code2wav_snake_beta(model, "hoist") == ("",)
    with torch.no_grad():
        model._omni_snake_beta_guard()
        model.alpha.add_(1)
        with pytest.raises(RuntimeError, match="changed"):
            model._omni_snake_beta_guard()


@pytest.mark.parametrize("nested", [False, True])
def test_installer_rejects_existing_hierarchy_state_without_mutation(nested):
    model = _nested_snake_model()
    target = model[0]["left"][0] if nested else model
    sentinel = object()
    vars(target)[snake_beta._LINKS] = sentinel
    with pytest.raises(ValueError, match="not installed"):
        snake_beta.install_code2wav_snake_beta(model, "hoist")
    assert vars(target)[snake_beta._LINKS] is sentinel
    assert "_omni_snake_beta_guard" not in vars(model)
    assert all("forward" not in vars(module) for module in model.modules())


def test_installer_rolls_back_hierarchy_state_on_guard_binding_failure(monkeypatch):
    model = _nested_snake_model()
    original_bind = snake_beta.MethodType

    def bind(function, module):
        if function is snake_beta._guard_model:
            raise RuntimeError("guard binding failed")
        return original_bind(function, module)

    monkeypatch.setattr(snake_beta, "MethodType", bind)
    with pytest.raises(RuntimeError, match="guard binding failed"):
        snake_beta.install_code2wav_snake_beta(model, "hoist")
    assert snake_beta._LINKS not in vars(model)
    assert "_omni_snake_beta_modules" not in vars(model)
    assert "_omni_snake_beta_guard" not in vars(model)
    for module in model.modules():
        assert "_omni_snake_beta_state" not in vars(module)
        assert "forward" not in vars(module)


def test_installer_rejects_preexisting_shadowed_hierarchy_without_mutation():
    model = _nested_snake_model()
    object.__setattr__(model[0], "left", nn.Sequential(SnakeBeta(4)).eval())
    with pytest.raises(RuntimeError, match="hierarchy"):
        snake_beta.install_code2wav_snake_beta(model, "hoist")
    assert snake_beta._LINKS not in vars(model)
    assert "_omni_snake_beta_guard" not in vars(model)
    for module in model.modules():
        assert "_omni_snake_beta_state" not in vars(module)
        assert "forward" not in vars(module)
