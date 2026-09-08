# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.utils import device_graph


def test_device_graph_rejects_cpu() -> None:
    with pytest.raises(RuntimeError, match="require CUDA or MUSA"):
        device_graph._backend(torch.device("cpu"))


def test_musa_graph_context_uses_relaxed_capture(monkeypatch) -> None:
    calls = []

    class FakeMusa:
        @contextmanager
        def graph(self, graph_obj, **kwargs):
            calls.append((graph_obj, kwargs))
            yield

    monkeypatch.setattr(torch, "musa", FakeMusa(), raising=False)
    monkeypatch.setattr(
        device_graph.torch,
        "device",
        lambda value: SimpleNamespace(type=str(value)),
    )
    with device_graph.graph(object(), device="musa"):
        pass

    assert calls[0][1]["capture_error_mode"] == "relaxed"
