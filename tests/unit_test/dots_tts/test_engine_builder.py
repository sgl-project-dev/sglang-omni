# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.dots_tts.engine_builder import DotsTTSEngineBuilder
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


def test_dots_engine_uses_shared_tts_builder() -> None:
    builder = DotsTTSEngineBuilder(optimize=True)

    assert isinstance(builder, TtsEngineBuilder)
    assert builder.optimize is True
    assert builder.generation_defaults(dtype="bfloat16")["max_running_requests"] == 16


def test_dots_engine_accepts_continuous_batching() -> None:
    DotsTTSEngineBuilder().adjust_overrides({"tp_size": 1, "max_running_requests": 16})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tp_size": 2, "max_running_requests": 16}, "does not implement TP"),
        (
            {
                "tp_size": 1,
                "max_running_requests": 16,
                "enable_torch_compile": True,
            },
            "backbone compile is disabled",
        ),
    ],
)
def test_dots_engine_rejects_unsupported_generation_modes(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DotsTTSEngineBuilder().adjust_overrides(overrides)


def test_extra_scheduler_callbacks_wire_tail_shutdown_logging() -> None:
    builder = DotsTTSEngineBuilder()
    assert builder.extra_scheduler_callbacks() == {}

    calls: list[int] = []
    builder._acoustic_tail = SimpleNamespace(log_graph_counters=lambda: calls.append(1))
    callback = builder.extra_scheduler_callbacks()["shutdown_callback"]
    callback()

    assert calls == [1]


def _mps_builder(**kwargs) -> DotsTTSEngineBuilder:
    builder = DotsTTSEngineBuilder(**kwargs)
    builder.device = "mps"
    return builder


def test_dots_engine_torch_mps_generation_defaults() -> None:
    defaults = _mps_builder(optimize=True).generation_defaults(dtype="bfloat16")

    assert defaults["max_running_requests"] == 1
    assert defaults["disable_cuda_graph"] is True
    assert defaults["enable_torch_compile"] is False
    assert defaults["attention_backend"] == "torch_native"
    assert defaults["sampling_backend"] == "pytorch"
    assert defaults["dtype"] == "float32"


def test_dots_engine_torch_mps_disables_optimize() -> None:
    builder = _mps_builder(optimize=True)
    builder.pre_infra_setup("stub")

    assert builder.optimize is False


def test_dots_engine_torch_mps_rejects_batched() -> None:
    with pytest.raises(ValueError, match="max_running_requests=1"):
        _mps_builder().adjust_overrides({"tp_size": 1, "max_running_requests": 16})


def test_dots_engine_torch_mps_degrades_cuda_graph_to_eager() -> None:
    overrides = {
        "tp_size": 1,
        "max_running_requests": 1,
        "disable_cuda_graph": False,
    }
    _mps_builder().adjust_overrides(overrides)

    assert overrides["disable_cuda_graph"] is True
    assert "enable_return_hidden_states" not in overrides


def test_dots_engine_torch_mps_accepts_single_eager_request() -> None:
    _mps_builder().adjust_overrides(
        {"tp_size": 1, "max_running_requests": 1, "disable_cuda_graph": True}
    )


def test_dots_engine_torch_mps_floors_mem_fraction() -> None:
    overrides = {
        "tp_size": 1,
        "max_running_requests": 1,
        "mem_fraction_static": 0.20,
    }
    _mps_builder().adjust_overrides(overrides)

    assert overrides["mem_fraction_static"] == 0.78


def test_dots_engine_mlx_selects_mlx_runner(monkeypatch) -> None:
    import sys
    import types

    stub = types.ModuleType("sglang_omni.models.dots_tts.mlx.runner")

    class FakeMlxRunner:
        def __init__(
            self, tp_worker: object, output_proc: object, *, checkpoint_dir: str
        ) -> None:
            self.tp_worker = tp_worker
            self.output_proc = output_proc
            self.checkpoint_dir = checkpoint_dir

    stub.DotsTTSMlxModelRunner = FakeMlxRunner
    monkeypatch.setitem(sys.modules, "sglang_omni.models.dots_tts.mlx.runner", stub)

    builder = DotsTTSEngineBuilder()
    monkeypatch.setattr(builder, "_use_mlx", lambda: True)
    builder.checkpoint_dir = "/tmp/stub-checkpoint"
    runner = builder.make_model_runner("worker", "proc")

    assert isinstance(runner, FakeMlxRunner)
    assert runner.tp_worker == "worker"
    assert runner.checkpoint_dir == builder.checkpoint_dir
    assert builder._model_runner is runner


def test_dots_engine_torch_mps_selects_mps_runner(monkeypatch) -> None:
    import sys
    import types

    stub = types.ModuleType("sglang_omni.models.dots_tts.torch_mps_runner")

    class FakeMpsRunner:
        def __init__(self, tp_worker: object, output_proc: object) -> None:
            self.tp_worker = tp_worker
            self.output_proc = output_proc

    stub.DotsTTSTorchMpsModelRunner = FakeMpsRunner
    monkeypatch.setitem(
        sys.modules, "sglang_omni.models.dots_tts.torch_mps_runner", stub
    )

    builder = _mps_builder()
    runner = builder.make_model_runner("worker", "proc")

    assert isinstance(runner, FakeMpsRunner)
    assert runner.tp_worker == "worker"
    assert builder._model_runner is runner
