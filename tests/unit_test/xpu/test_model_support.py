# SPDX-License-Identifier: Apache-2.0
"""Per-model contract for every family Intel XPU serves (no weights, no accelerator).

The serving lane (tests/test_ci/xpu_model/) needs checkpoints and cards, so it
cannot gate a PR; this covers the CPU-side wiring that a PR can break.

Registry membership matters more than it looks: registry.py logs and swallows a
failed model import, so a broken config module does not raise -- the architecture
vanishes and first surfaces as ``KeyError`` for whoever names the checkpoint.

Device resolution is asserted declaratively here; proving a factory *resolves* an
unset device needs the factory to run, which is
tests/unit_test/test_stage_device_contract.py.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.platforms import current_platform
from sglang_omni.utils.imports import import_string
from tests.test_ci.xpu_model.model_table import (
    XPU_MODELS,
    XpuModel,
    model_named,
    xpu_pipelines,
)

_PIPELINES = xpu_pipelines()
_PIPELINE_IDS = [model.pipeline_key for model in _PIPELINES]


def _config_cls(model: XpuModel) -> type:
    module = importlib.import_module(f"sglang_omni.models.{model.package}.config")
    if model.variant is None:
        return module.EntryClass
    variants = getattr(module, "Variants", {})
    assert model.variant in variants, (
        f"{model.package} exposes no {model.variant!r} variant; "
        f"Variants={sorted(variants)}"
    )
    return variants[model.variant]


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_the_config_module_imports_without_a_swallowed_error(model: XpuModel) -> None:
    """A direct import raises where the registry would only log and move on."""
    module = importlib.import_module(f"sglang_omni.models.{model.package}.config")
    assert hasattr(module, "EntryClass"), f"{model.package}.config has no EntryClass"


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_the_architecture_reaches_the_pipeline_config_registry(model: XpuModel) -> None:
    """An architecture missing here is a checkpoint no operator can serve."""
    registered = PIPELINE_CONFIG_REGISTRY.get_supported_archs()
    assert model.architecture in registered, (
        f"{model.architecture} is not registered, so a {model.package} checkpoint "
        "cannot be served. The registry swallows model import errors: import "
        f"sglang_omni.models.{model.package}.config directly to see the cause."
    )


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_the_pipeline_builds_the_stage_table_xpu_serves(model: XpuModel) -> None:
    """The topology is pinned, so a stage gained or lost surfaces here."""
    config = _config_cls(model)(model_path="unused")
    assert tuple(stage.name for stage in config.stages) == model.stages
    if config.entry_stage is not None:
        assert config.entry_stage in model.stages


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_no_stage_pins_a_device_this_host_cannot_serve(model: XpuModel) -> None:
    """Unset is intended -- the worker resolves it from the platform -- so the
    only other thing a stage may say is this platform's own device."""
    config = _config_cls(model)(model_path="unused")
    live = current_platform.device_type
    for stage in config.stages:
        device = getattr(stage.factory, "device", None)
        if device is None:
            continue
        assert str(device).split(":")[0] == live, (
            f"{model.package} stage {stage.name!r} pins device={device!r}, which "
            f"this host ({live}) cannot serve"
        )


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_every_stage_factory_resolves_to_a_callable(model: XpuModel) -> None:
    """A renamed or moved factory fails at stage launch, minutes into a load."""
    config = _config_cls(model)(model_path="unused")
    for stage in config.stages:
        assert stage.factory_path, f"stage {stage.name!r} declares no factory_path"
        factory = import_string(stage.factory_path)
        assert callable(factory), f"{stage.factory_path} is not callable"


@pytest.mark.parametrize("model", _PIPELINES, ids=_PIPELINE_IDS)
def test_a_checkpoint_resolves_to_the_family_that_serves_it(
    model: XpuModel, tmp_path: Path
) -> None:
    """``--model-path`` finds the family from checkpoint metadata alone."""
    from sglang_omni.config.manager import resolve_config_cls_for_model_path

    checkpoint = tmp_path / model.mirror_dirname
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": [model.architecture]})
    )

    resolved = resolve_config_cls_for_model_path(str(checkpoint))

    module = importlib.import_module(f"sglang_omni.models.{model.package}.config")
    # The resolver answers with the entry class; the operator selects a variant
    # afterwards, so this asserts the family and not the topology.
    assert resolved is module.EntryClass


def test_every_model_in_the_table_is_exercised_by_the_serving_lane() -> None:
    """The serving lane dispatches on ``kind`` and would raise KeyError on an
    unknown one, but only on a host with weights in an opt-in lane."""
    from tests.test_ci.xpu_model.exercises import EXERCISES

    uncovered = sorted(
        f"{model.key} (kind={model.kind})"
        for model in XPU_MODELS
        if model.kind not in EXERCISES
    )
    assert not uncovered, f"tests/test_ci/xpu_model has no exercise for: {uncovered}"


def test_the_code2wav_stage_takes_the_graph_policy_from_the_platform() -> None:
    """The policy has to travel to the stage for either answer to mean anything."""
    speech = _config_cls(model_named("omni-speech"))(model_path="unused")
    kwargs = speech.stage_factory_kwargs("code2wav")
    assert kwargs["enable_cuda_graph"] is current_platform.enable_code2wav_graph()
