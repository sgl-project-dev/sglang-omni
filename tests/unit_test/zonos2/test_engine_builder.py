# SPDX-License-Identifier: Apache-2.0
"""ZONOS2's SGLang server-arg defaults.

fp8=True is a stage default (config.py stage_factory_kwargs), so it has to mean
"quantize if this device can" rather than "quantize or refuse to serve": a
platform whose SGLang build has no load-time FP8 weight quantizer keeps the bf16
experts and stays servable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.zonos2 import engine_builder as zonos2_engine


def _platform(*, supports_fp8: bool, compile: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        supports_online_fp8_quantization=lambda: supports_fp8,
        enable_zonos2_torch_compile=lambda: compile,
        device_type="cuda" if supports_fp8 else "xpu",
    )


def test_fp8_asks_for_quantization_where_the_device_can_do_it(monkeypatch) -> None:
    monkeypatch.setattr(zonos2_engine, "current_platform", _platform(supports_fp8=True))

    defaults = zonos2_engine.Zonos2EngineBuilder(fp8=True).generation_defaults(
        dtype="bfloat16"
    )

    assert defaults["quantization"] == "fp8"
    assert defaults["mem_fraction_static"] == 0.5


def test_fp8_degrades_to_bf16_experts_rather_than_failing_the_load(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(
        zonos2_engine, "current_platform", _platform(supports_fp8=False)
    )

    with caplog.at_level("INFO", logger=zonos2_engine.__name__):
        defaults = zonos2_engine.Zonos2EngineBuilder(fp8=True).generation_defaults(
            dtype="bfloat16"
        )

    assert "quantization" not in defaults
    assert "bf16 MoE experts" in caplog.text
    assert defaults["mem_fraction_static"] == 0.85


def test_a_larger_configured_static_pool_survives_the_bf16_floor(monkeypatch) -> None:
    """The floor raises the default; it must not lower an operator's own number."""
    monkeypatch.setattr(
        zonos2_engine, "current_platform", _platform(supports_fp8=False)
    )

    defaults = zonos2_engine.Zonos2EngineBuilder(
        fp8=True, mem_fraction_static=0.92
    ).generation_defaults(dtype="bfloat16")

    assert defaults["mem_fraction_static"] == 0.92


@pytest.mark.parametrize("compile_ok", [True, False])
def test_torch_compile_follows_the_platform(monkeypatch, compile_ok) -> None:
    monkeypatch.setattr(
        zonos2_engine,
        "current_platform",
        _platform(supports_fp8=True, compile=compile_ok),
    )

    defaults = zonos2_engine.Zonos2EngineBuilder().generation_defaults(dtype="bfloat16")

    assert defaults["enable_torch_compile"] is compile_ok


@pytest.mark.parametrize("supports_fp8", [True, False])
def test_fp8_off_never_names_a_quantization(monkeypatch, supports_fp8) -> None:
    monkeypatch.setattr(
        zonos2_engine, "current_platform", _platform(supports_fp8=supports_fp8)
    )

    defaults = zonos2_engine.Zonos2EngineBuilder(fp8=False).generation_defaults(
        dtype="bfloat16"
    )

    assert "quantization" not in defaults


def test_decode_graphs_stay_enabled_for_the_platform_backend_to_pick_up() -> None:
    """disable_cuda_graph=False is the precondition for the XPU decode graph.

    server_args_builder turns cuda_graph_backend_decode on from the platform hook
    only when graphs are not disabled outright, so this default is load-bearing on
    every accelerator, not just CUDA.
    """
    defaults = zonos2_engine.Zonos2EngineBuilder().generation_defaults(dtype="bfloat16")

    assert defaults["disable_cuda_graph"] is False
