# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import threading
import time
from types import ModuleType

import pytest

from benchmarks import runtime_metrics_npu
from benchmarks.runtime_metrics_npu import (
    NpuResourceMonitor,
    NpuResourceSample,
    collect_npu_environment_fingerprint,
    summarize_npu_resource_samples,
)


USAGES = """
| HBM Capacity(MB)       | 65536 |
| HBM Usage Rate(%)      | 84    |
| Aicore Usage Rate(%)   | 73    |
| NPU Utilization(%)     | 75    |
"""
MEMORY = "| HBM Temperature(C) | 51 |"
POWER = "| NPU Real-time Power(W) | 286.5 |"


def _sample(**overrides) -> NpuResourceSample:
    values = {
        "elapsed_s": 1.0,
        "npu_id": 0,
        "chip_id": 0,
        "chip_count": 1,
        "hbm_capacity_mb": 65536,
        "hbm_usage_percent": 84.0,
        "aicore_usage_percent": 73.0,
        "npu_util_percent": 75.0,
        "hbm_temp_c": 51.0,
        "power_w": 286.5,
        "system_cpu_percent": 12.0,
    }
    values.update(overrides)
    return NpuResourceSample(**values)


def test_label_parser_accepts_table_and_colon_formats() -> None:
    assert runtime_metrics_npu._label_values(USAGES, ("HBM Capacity",)) == [
        65536.0
    ]
    assert runtime_metrics_npu._label_values(
        "HBM Usage Rate: 42%", ("HBM Usage Rate",)
    ) == [42.0]


def test_summarize_npu_samples_reports_required_metrics() -> None:
    result = summarize_npu_resource_samples(
        [_sample(elapsed_s=0.0), _sample(elapsed_s=1.0, hbm_usage_percent=90.0)],
        interval_s=1.0,
        npu_id=0,
        chip_id=0,
    )
    assert result["available"] is True
    assert result["hbm_usage_percent"]["max"] == 90.0
    assert result["aicore_usage_percent"]["mean"] == 73.0
    assert len(result["raw_samples"]) == 2
    assert result["error"] is None


def test_summarize_npu_samples_fails_closed_without_hbm_or_utilization() -> None:
    result = summarize_npu_resource_samples(
        [
            _sample(
                hbm_usage_percent=None,
                aicore_usage_percent=None,
                npu_util_percent=None,
            )
        ],
        interval_s=1.0,
        npu_id=0,
        chip_id=0,
    )
    assert result["available"] is False
    assert "missing required NPU metrics" in result["error"]


def test_summarize_npu_samples_reports_empty_monitor() -> None:
    result = summarize_npu_resource_samples(
        [], interval_s=1.0, npu_id=2, chip_id=0
    )
    assert result == {
        "available": False,
        "npu_id": 2,
        "chip_id": 0,
        "sample_interval_s": 1.0,
        "samples": 0,
        "error": "no NPU resource samples were collected",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"npu_id": -1},
        {"chip_id": -1},
        {"interval_s": 0},
    ],
)
def test_npu_monitor_rejects_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        NpuResourceMonitor(**kwargs)


def test_npu_monitor_refuses_overlapping_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()
    lock.acquire()
    monkeypatch.setattr(runtime_metrics_npu, "_NPU_SMI_LOCK", lock)
    result = NpuResourceMonitor().start().stop()
    assert result["available"] is False
    assert result["error"] == "another NPU resource monitor is still active"


def test_npu_monitor_collects_on_background_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller = threading.get_ident()
    sampling_threads: list[int] = []

    def fake_run(*args: str) -> str:
        sampling_threads.append(threading.get_ident())
        if "usages" in args:
            return USAGES
        if "memory" in args:
            return MEMORY
        return POWER

    psutil = ModuleType("psutil")
    psutil.cpu_percent = lambda interval=None: 12.0
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    monkeypatch.setattr(runtime_metrics_npu, "_run_npu_smi", fake_run)

    monitor = NpuResourceMonitor(interval_s=0.01).start()
    deadline = time.monotonic() + 1.0
    while not monitor.samples and time.monotonic() < deadline:
        time.sleep(0.01)
    result = monitor.stop()

    assert result["available"] is True
    assert result["hbm_capacity_mb"]["max"] == 65536.0
    assert sampling_threads
    assert caller not in sampling_threads
    assert len(set(sampling_threads)) == 1


def test_environment_fingerprint_does_not_embed_raw_npu_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "| Name | Ascend910_9382 |\n| Health | OK |\n| HBM Capacity(MB) | 65536 |"
    monkeypatch.setattr(runtime_metrics_npu, "_run_npu_smi", lambda *_args: output)
    result = collect_npu_environment_fingerprint([0])
    assert result["available"] is True
    assert result["cards"][0]["chip_name"] == "Ascend910_9382"
    assert result["cards"][0]["raw_sha256"]
    assert output not in str(result)
