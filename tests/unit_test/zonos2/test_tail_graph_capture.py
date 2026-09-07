# SPDX-License-Identifier: Apache-2.0
"""The per-frame tail graph is recorded by the platform, not by torch.cuda.

``capture_tail_graphs`` runs at startup on whatever accelerator the AR stage was
placed on, so both halves have to follow that device: the warmup stream dance
through the device module, and the graph object through the platform's graph
backend. Naming ``torch.cuda`` for either one kills startup on a non-CUDA build,
which no CUDA test would notice.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.platforms as platforms
from sglang_omni.models.zonos2.components.text_frontend import TTSSamplingParams
from sglang_omni.models.zonos2.payload_types import N_CODEBOOKS
from sglang_omni.models.zonos2.sglang_model import Zonos2SGLangModel


class _RecordingStream:
    def __init__(self, log: list[str], name: str) -> None:
        self._log = log
        self._name = name

    def wait_stream(self, other) -> None:
        self._log.append(f"{self._name}.wait_stream({other._name})")


class _RecordingDeviceModule:
    """Stands in for the accelerator module of the device the model reports."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.side = _RecordingStream(log, "side")
        self.current = _RecordingStream(log, "current")

    def Stream(self, *args, **kwargs):
        del args, kwargs
        self._log.append("Stream()")
        return self.side

    def current_stream(self, *args, **kwargs):
        del args, kwargs
        return self.current

    @contextlib.contextmanager
    def stream(self, stream):
        self._log.append(f"enter stream({stream._name})")
        yield
        self._log.append("exit stream")

    def synchronize(self, *args, **kwargs) -> None:
        del args, kwargs
        self._log.append("synchronize")


class _RecordingGraphBackend:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.graphs: list[object] = []

    @contextlib.contextmanager
    def capture(self, **kwargs):
        self._log.append(f"capture({sorted(kwargs)})")
        graph = SimpleNamespace(index=len(self.graphs))
        self.graphs.append(graph)
        yield graph
        self._log.append("captured")


def _harness(log: list[str], device: torch.device):
    """A model stand-in exposing only what capture_tail_graphs reads."""
    model = SimpleNamespace(
        device=device,
        dtype=torch.float32,
        n_codebooks=N_CODEBOOKS,
        audio_vocab=8,
        config=SimpleNamespace(dim=4),
    )
    model._tail_compute = lambda bs: log.append(f"compute({bs})")
    model.capture_tail_graphs = Zonos2SGLangModel.capture_tail_graphs.__get__(model)
    return model


def test_capture_records_each_bucket_through_the_platform_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    device = torch.device("cpu")
    backend = _RecordingGraphBackend(log)
    module = _RecordingDeviceModule(log)

    monkeypatch.setattr(
        platforms.current_platform,
        "get_device_graph_backend",
        lambda dev: backend if dev == device else None,
    )
    monkeypatch.setattr(
        torch, "get_device_module", lambda dev: module if dev == device else None
    )

    model = _harness(log, device)
    model.capture_tail_graphs([1, 2], TTSSamplingParams())

    assert log == [
        "Stream()",
        "side.wait_stream(current)",
        "enter stream(side)",
        "compute(1)",
        "compute(1)",
        "compute(1)",
        "compute(2)",
        "compute(2)",
        "compute(2)",
        "exit stream",
        "current.wait_stream(side)",
        "synchronize",
        "capture([])",
        "compute(1)",
        "captured",
        "capture([])",
        "compute(2)",
        "captured",
        "synchronize",
    ]
    assert model._tail_buckets == [1, 2]
    assert model._tail_graphs == {1: backend.graphs[0], 2: backend.graphs[1]}
    assert model._cg["hidden"].shape == (2, 4)
    assert model._cg["codes"].shape == (2, N_CODEBOOKS)


def test_capture_refuses_a_platform_that_records_no_model_owned_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """frame_graph on a graph-less platform must fail loudly at startup.

    The runner arms the graph path on ``_tail_buckets`` being non-empty, so a
    silent return here would send every frame into replaying a graph that was
    never recorded.
    """
    log: list[str] = []
    device = torch.device("cpu")
    monkeypatch.setattr(
        platforms.current_platform, "get_device_graph_backend", lambda dev: None
    )

    model = _harness(log, device)
    with pytest.raises(RuntimeError, match="frame_graph"):
        model.capture_tail_graphs([1], TTSSamplingParams())

    assert log == [], "no compute may run once capture is known to be impossible"
    assert model._tail_graphs == {}
