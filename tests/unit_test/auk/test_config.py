# SPDX-License-Identifier: Apache-2.0
"""Pipeline config and registry wiring for AuK."""

from __future__ import annotations

import importlib

import pytest

from sglang_omni.models.auk.config import (
    ENGINE_STAGE,
    PREPROCESSING_STAGE,
    VOCODER_STAGE,
    AuKPipelineConfig,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_architecture_is_registered():
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("AuKForConditionalGeneration")
        is AuKPipelineConfig
    )


@pytest.mark.parametrize("alias", ["AuK", "AuK-Flash"])
def test_architecture_aliases_resolve_to_same_config(alias):
    assert PIPELINE_CONFIG_REGISTRY.get_config(alias) is AuKPipelineConfig


def test_stage_topology():
    config = AuKPipelineConfig(model_path="tencent/AuK")
    assert [stage.name for stage in config.stages] == [
        PREPROCESSING_STAGE,
        ENGINE_STAGE,
        VOCODER_STAGE,
    ]
    assert config.resolved_entry_stage == PREPROCESSING_STAGE
    assert config.terminal_stages == [VOCODER_STAGE]


def test_stage_graph_is_connected():
    config = AuKPipelineConfig(model_path="tencent/AuK")
    stages = {stage.name: stage for stage in config.stages}
    assert stages[PREPROCESSING_STAGE].next == ENGINE_STAGE
    assert stages[ENGINE_STAGE].next == VOCODER_STAGE
    assert stages[VOCODER_STAGE].terminal is True


def test_factory_paths_resolve():
    for stage in AuKPipelineConfig(model_path="tencent/AuK").stages:
        module_path, _, attr = stage.factory_path.partition(":")
        if not attr:
            module_path, _, attr = module_path.rpartition(".")
        module = importlib.import_module(module_path)
        assert callable(getattr(module, attr)), stage.factory_path
