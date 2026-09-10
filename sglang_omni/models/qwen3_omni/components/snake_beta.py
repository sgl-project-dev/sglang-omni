# SPDX-License-Identifier: Apache-2.0
"""Opt-in inference-only constants for HF's Apache-2.0 SnakeBeta formula.

Install on a loaded, eval-mode model before capture. Parameters and checkpoint
keys stay unchanged. Hot reload, concurrent mutation and unsafe ``.data`` writes
are unsupported; the scheduler checks tracked state before each graph replay.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn

_STATE = "_omni_snake_beta_state"
_MODULES = "_omni_snake_beta_modules"
_LINKS = "_omni_snake_beta_links"
_GUARD = "_omni_snake_beta_guard"
_HOOKS = (
    "_forward_hooks",
    "_forward_pre_hooks",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)


@dataclass(frozen=True, eq=False)
class _State:
    implementation: str
    alpha_stamp: tuple
    beta_stamp: tuple
    exp_alpha: torch.Tensor
    inv_beta_safe: torch.Tensor


def _stamp(tensor: torch.Tensor) -> tuple:
    return (
        id(tensor),
        tensor._version,
        tensor.untyped_storage()._cdata,
        tensor.data_ptr(),
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tensor.stride(),
    )


def _plain_eval(module: nn.Module) -> None:
    if module.training:
        raise RuntimeError("SnakeBeta optimization requires eval mode")
    if (
        getattr(module, "_hf_hook", None) is not None
        or getattr(module, "_compiled_call_impl", None) is not None
    ):
        raise RuntimeError("SnakeBeta optimization forbids offload/compiled modules")
    if any(getattr(module, name, None) for name in _HOOKS):
        raise RuntimeError("SnakeBeta optimization forbids custom module hooks")


def _no_global_hooks() -> None:
    if any(
        getattr(nn.modules.module, name, None)
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        )
    ):
        raise RuntimeError("SnakeBeta optimization forbids global module hooks")


def _check_state(module: nn.Module, state: _State) -> None:
    _plain_eval(module)
    if (
        getattr(module.forward, "__func__", None) is not _forward
        or module.no_div_by_zero != 1e-9
        or _stamp(module.alpha) != state.alpha_stamp
        or _stamp(module.beta) != state.beta_stamp
    ):
        raise RuntimeError(
            "SnakeBeta static parameters or forward changed; rebuild model"
        )


def _guard_model(model: nn.Module) -> None:
    if torch.is_grad_enabled():
        raise RuntimeError("SnakeBeta optimization requires disabled gradients")
    _plain_eval(model)
    _no_global_hooks()
    for parent, name, child in vars(model)[_LINKS]:
        if getattr(parent, name, None) is not child:
            raise RuntimeError("SnakeBeta installed module changed; rebuild model")
    for _, module, state in vars(model)[_MODULES]:
        if vars(module).get(_STATE) is not state:
            raise RuntimeError("SnakeBeta installed module changed; rebuild model")
        _check_state(module, state)


def _forward(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled():
        raise RuntimeError("SnakeBeta optimization requires disabled gradients")
    state = vars(module)[_STATE]
    _check_state(module, state)
    if (
        type(hidden_states) is not torch.Tensor
        or hidden_states.ndim != 3
        or any(size < 1 for size in hidden_states.shape)
        or hidden_states.shape[1] != state.exp_alpha.shape[1]
        or hidden_states.layout != torch.strided
        or not hidden_states.is_contiguous()
        or hidden_states.device != state.exp_alpha.device
        or hidden_states.dtype != state.exp_alpha.dtype
    ):
        raise ValueError(
            "SnakeBeta requires contiguous NCT input matching loaded parameters"
        )
    if state.implementation == "fused":
        from .fused_snake_beta import fused_snake_beta

        return fused_snake_beta(hidden_states, state.exp_alpha, state.inv_beta_safe)
    # Note (wenyao): Keep all five HF input operations and their dtype roundings.
    return hidden_states + state.inv_beta_safe * torch.pow(
        torch.sin(hidden_states * state.exp_alpha), 2
    )


def install_code2wav_snake_beta(
    model: nn.Module, implementation: str
) -> tuple[str, ...]:
    """Atomically install ``hoist`` or CUDA-BF16 ``fused`` before graph capture.

    Cached tensors are private state, not Parameters or persistent buffers.
    Moving/reloading the model afterward must fail, never refresh captured state.
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import SnakeBeta

    if type(implementation) is not str or implementation not in ("hoist", "fused"):
        raise ValueError("SnakeBeta installer requires 'hoist' or 'fused'")
    _no_global_hooks()
    modules = tuple(model.named_modules(remove_duplicate=False))
    counts = Counter(id(module) for _, module in modules)
    for _, module in modules:
        _plain_eval(module)
        if any(
            name in vars(module)
            for name in (_STATE, _MODULES, _LINKS, _GUARD, "forward")
        ):
            raise ValueError(
                "SnakeBeta optimization requires unmodified, not installed modules"
            )
    selected = [(path, m) for path, m in modules if isinstance(m, SnakeBeta)]
    if not selected:
        raise ValueError("No HF SnakeBeta modules found")
    plan = []
    parameter_ids = set()
    for path, module in selected:
        if (
            type(module) is not SnakeBeta
            or counts[id(module)] != 1
            or set(module._parameters) != {"alpha", "beta"}
            or module._modules
            or module._buffers
            or type(module.in_features) is not int
            or module.in_features < 1
            or type(module.no_div_by_zero) is not float
            or module.no_div_by_zero != 1e-9
        ):
            raise ValueError("Unsupported or shared HF SnakeBeta module")
        for parameter in (module.alpha, module.beta):
            if (
                type(parameter) is not nn.Parameter
                or id(parameter) in parameter_ids
                or parameter.is_meta
                or parameter.device.type not in ("cpu", "cuda")
                or parameter.layout != torch.strided
                or parameter.dtype
                not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
                or parameter.shape != (module.in_features,)
                or not parameter.is_contiguous()
            ):
                raise ValueError(
                    "SnakeBeta requires loaded, unshared, contiguous floating parameters"
                )
            parameter_ids.add(id(parameter))
        if (
            module.alpha.device != module.beta.device
            or module.alpha.dtype != module.beta.dtype
        ):
            raise ValueError("SnakeBeta parameter device/dtype mismatch")
        if implementation == "fused" and (
            module.alpha.device.type != "cuda"
            or module.alpha.dtype != torch.bfloat16
            or torch.version.hip is not None
        ):
            raise ValueError("Fused SnakeBeta requires CUDA BF16 parameters")
        alpha_stamp, beta_stamp = _stamp(module.alpha), _stamp(module.beta)
        with torch.inference_mode(False), torch.no_grad():
            exp_alpha = torch.exp(module.alpha.unsqueeze(0).unsqueeze(-1))
            beta = torch.exp(module.beta.unsqueeze(0).unsqueeze(-1))
            # Note (wenyao): Preserve HF reciprocal/scalar-multiply roundings.
            inv_beta_safe = 1.0 / (beta + module.no_div_by_zero)
        plan.append(
            (
                path,
                module,
                _State(
                    implementation, alpha_stamp, beta_stamp, exp_alpha, inv_beta_safe
                ),
            )
        )
    for _, module, state in plan:
        if (
            _stamp(module.alpha) != state.alpha_stamp
            or _stamp(module.beta) != state.beta_stamp
        ):
            raise RuntimeError("SnakeBeta parameters changed during installation")
    links = {}
    for path, module, _ in plan:
        parent = model
        for name in path.split(".") if path else ():
            child = getattr(parent, name)
            links[id(parent), name] = (parent, name, child)
            parent = child
        if parent is not module:
            raise RuntimeError("SnakeBeta hierarchy changed during installation")
    bound = []
    try:
        for _, module, state in plan:
            bound.append(module)
            vars(module)[_STATE] = state
            module.forward = MethodType(_forward, module)
        vars(model)[_MODULES] = tuple(plan)
        vars(model)[_LINKS] = tuple(links.values())
        vars(model)[_GUARD] = MethodType(_guard_model, model)
    except BaseException:
        for module in bound:
            vars(module).pop(_STATE, None)
            vars(module).pop("forward", None)
        vars(model).pop(_MODULES, None)
        vars(model).pop(_LINKS, None)
        vars(model).pop(_GUARD, None)
        raise
    return tuple(path for path, _, _ in plan)
