# SPDX-License-Identifier: Apache-2.0
"""Checkpoint discovery and direct weight loading for AuK."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.auk.dit import AuKDit
from sglang_omni.models.auk.flow_matching import AuKFlowMatching
from sglang_omni.models.auk.vae import AuKVAEConfig, BigVGANFlowVAE
from sglang_omni.models.auk.weight_loader import (
    load_dit_weights,
    load_vae_weights,
    normalize_state_dict,
    resolve_vae_file,
    resolve_weight_file,
)


def test_normalize_state_dict_strips_wrappers():
    state_dict = {
        "ema_model.transformer.proj_out.weight": torch.zeros(1),
        "module.layer_weights": torch.zeros(2),
        "layer_scale": torch.ones(1),
    }
    normalized = normalize_state_dict(state_dict)
    assert sorted(normalized) == [
        "layer_scale",
        "layer_weights",
        "transformer.proj_out.weight",
    ]


def test_resolve_weight_file_prefers_named_export(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "auk_base.safetensors").write_bytes(b"")
    assert resolve_weight_file(str(tmp_path)).name == "auk_base.safetensors"


def test_resolve_weight_file_falls_back_to_any_safetensors(tmp_path):
    (tmp_path / "custom.safetensors").write_bytes(b"")
    assert resolve_weight_file(str(tmp_path)).name == "custom.safetensors"


def test_resolve_weight_file_ignores_vae(tmp_path):
    (tmp_path / "vae.safetensors").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        resolve_weight_file(str(tmp_path))


def test_resolve_vae_file(tmp_path):
    assert resolve_vae_file(str(tmp_path)) is None
    (tmp_path / "vae.safetensors").write_bytes(b"")
    assert resolve_vae_file(str(tmp_path)).name == "vae.safetensors"


def _tiny_flow() -> AuKFlowMatching:
    dit = AuKDit(
        dim=32,
        heads=2,
        dim_head=16,
        num_layers=1,
        num_single_layers=1,
        latent_dim=8,
        text_hidden_dim=16,
    )
    return AuKFlowMatching(dit, num_llm_layers=2)


def _save(state_dict: dict[str, torch.Tensor], path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(path))


def test_load_dit_weights_round_trip(tmp_path):
    flow = _tiny_flow()
    _save(flow.state_dict(), tmp_path / "auk_base.safetensors")

    report = load_dit_weights(flow, str(tmp_path))
    assert report.missing == 0
    assert report.unexpected == 0
    assert report.loaded == len(flow.state_dict())


def test_load_dit_weights_strips_ema_prefix(tmp_path):
    flow = _tiny_flow()
    state_dict = {f"ema_model.{k}": v for k, v in flow.state_dict().items()}
    _save(state_dict, tmp_path / "auk_base.safetensors")

    report = load_dit_weights(flow, str(tmp_path))
    assert report.missing == 0
    assert report.unexpected == 0


def test_load_dit_weights_rejects_missing_and_unexpected(tmp_path):
    flow = _tiny_flow()
    state_dict = dict(flow.state_dict())
    del state_dict["layer_weights"]
    state_dict["stale.key"] = torch.zeros(1)
    _save(state_dict, tmp_path / "auk_base.safetensors")

    with pytest.raises(RuntimeError, match="layer_weights"):
        load_dit_weights(flow, str(tmp_path))


@pytest.mark.parametrize("kind", ["missing", "unexpected"])
def test_load_vae_weights_rejects_mismatch(tmp_path, kind):
    vae = torch.nn.Linear(2, 2)
    state = dict(vae.state_dict())
    if kind == "missing":
        del state["bias"]
    else:
        state["stale"] = torch.zeros(1)
    _save(state, tmp_path / "vae.safetensors")
    with pytest.raises(RuntimeError):
        load_vae_weights(vae, str(tmp_path))


def test_load_vae_weights(tmp_path):
    config = AuKVAEConfig(
        upsample_rates=[5, 4],
        upsample_kernel_sizes=[10, 8],
        upsample_initial_channel=16,
        resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
        downsample_rates=[4, 5],
        downsample_channels=[12, 16, 32],
        latent_dim=8,
        flow_hidden_channels=16,
    )
    vae = BigVGANFlowVAE(config)
    _save(vae.state_dict(), tmp_path / "vae.safetensors")

    report = load_vae_weights(vae, str(tmp_path))
    assert report.missing == 0
    assert report.unexpected == 0


def test_load_vae_weights_requires_the_file(tmp_path):
    vae = BigVGANFlowVAE(
        AuKVAEConfig(
            upsample_rates=[5, 4],
            upsample_kernel_sizes=[10, 8],
            upsample_initial_channel=16,
            resblock_kernel_sizes=[3, 7],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
            downsample_rates=[4, 5],
            downsample_channels=[12, 16, 32],
            latent_dim=8,
            flow_hidden_channels=16,
        )
    )
    with pytest.raises(FileNotFoundError):
        load_vae_weights(vae, str(tmp_path))
