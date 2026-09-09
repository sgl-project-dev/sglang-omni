# SPDX-License-Identifier: Apache-2.0
"""Pipeline config and registry wiring for AuK."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import huggingface_hub
import pytest

from sglang_omni.config import manager
from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.config import (
    ENGINE_STAGE,
    PREPROCESSING_STAGE,
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
    ]
    assert config.resolved_entry_stage == PREPROCESSING_STAGE
    assert config.terminal_stages == [ENGINE_STAGE]


def test_stage_graph_is_connected():
    config = AuKPipelineConfig(model_path="tencent/AuK")
    stages = {stage.name: stage for stage in config.stages}
    assert stages[PREPROCESSING_STAGE].next == ENGINE_STAGE
    assert stages[ENGINE_STAGE].terminal is True


def test_factory_paths_resolve():
    for stage in AuKPipelineConfig(model_path="tencent/AuK").stages:
        module_path, _, attr = stage.factory_path.partition(":")
        if not attr:
            module_path, _, attr = module_path.rpartition(".")
        module = importlib.import_module(module_path)
        assert callable(getattr(module, attr)), stage.factory_path


def test_pipeline_defaults_cover_former_example_yaml():
    stages = {
        stage.name: stage
        for stage in AuKPipelineConfig(model_path="tencent/AuK").stages
    }
    engine = stages[ENGINE_STAGE].factory
    assert stages[PREPROCESSING_STAGE].factory.max_concurrency == 8
    assert engine.dtype == "bfloat16"
    assert engine.text_encoder_path == C.DEFAULT_TEXT_ENCODER
    assert engine.nfe == C.DEFAULT_NFE
    assert engine.cfg_strength == C.DEFAULT_CFG_STRENGTH
    assert engine.sway_sampling_coef == C.DEFAULT_SWAY_SAMPLING_COEF
    assert engine.max_seconds == C.MAX_SECONDS


@pytest.mark.parametrize("name", ["AuK", "AuK-Flash"])
def test_local_checkpoint_yaml_resolves_pipeline(tmp_path, name):
    (tmp_path / "config.yaml").write_text(f"model:\n  name: {name}\n")
    assert manager.resolve_config_cls_for_model_path(str(tmp_path)) is AuKPipelineConfig


def test_unrelated_omegaconf_yaml_is_not_auk(tmp_path):
    (tmp_path / "config.yaml").write_text("model:\n  name: OtherModel\n")
    with pytest.raises(ValueError, match="Could not resolve model architecture"):
        manager.resolve_config_cls_for_model_path(str(tmp_path))


def test_local_weight_marker_resolves_without_yaml(tmp_path):
    (tmp_path / "auk_base.safetensors").write_bytes(b"")
    assert manager.resolve_config_cls_for_model_path(str(tmp_path)) is AuKPipelineConfig


def test_hub_config_yaml_resolves_without_snapshot(monkeypatch, tmp_path):
    def fail_snapshot(*args, **kwargs):
        raise AssertionError("architecture discovery must not download weights")

    def fail_auto_config(*args, **kwargs):
        raise OSError("no config.json")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail_snapshot)
    monkeypatch.setattr(
        manager, "AutoConfig", SimpleNamespace(from_pretrained=fail_auto_config)
    )

    def fake_hub_download(repo_id, filename, revision=None, **kwargs):
        if filename != "config.yaml":
            raise FileNotFoundError(filename)
        path = tmp_path / filename
        path.write_text("model:\n  name: AuK\n")
        return str(path)

    monkeypatch.setattr("sglang_omni.utils.hf.hf_hub_download", fake_hub_download)
    assert manager.resolve_config_cls_for_model_path("tencent/AuK") is AuKPipelineConfig
