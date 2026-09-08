# SPDX-License-Identifier: Apache-2.0
"""Fail-closed Ascend NPU resource sampling for performance gates."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

_NPU_SMI_LOCK = threading.Lock()
_NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"


@dataclass(frozen=True)
class NpuResourceSample:
    elapsed_s: float
    npu_id: int
    chip_id: int
    chip_count: int
    hbm_capacity_mb: int | None
    hbm_usage_percent: float | None
    aicore_usage_percent: float | None
    npu_util_percent: float | None
    hbm_temp_c: float | None
    power_w: float | None
    system_cpu_percent: float | None


def _run_npu_smi(*args: str) -> str | None:
    try:
        return subprocess.run(
            ("npu-smi", *args),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _label_values(text: str, labels: Iterable[str]) -> list[float]:
    alternatives = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(
        rf"(?:{alternatives})\s*(?:\([^\r\n|]*\))?\s*(?:[:|=]\s*)?{_NUMBER}",
        re.IGNORECASE,
    )
    return [float(match.group(1)) for match in pattern.finditer(text or "")]


def _parse_per_chip(text: str, pattern: re.Pattern[str]) -> list[int]:
    return [int(match.group(1)) for match in pattern.finditer(text or "")]


def _parse_per_chip_float(text: str, pattern: re.Pattern[str]) -> list[float]:
    return [float(match.group(1)) for match in pattern.finditer(text or "")]


def _pick(values: list[float], chip_id: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values[chip_id] if chip_id < len(values) else None


def _optional_series_summary(values: list[float | None]) -> dict[str, float] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return {
        "min": min(present),
        "max": max(present),
        "mean": sum(present) / len(present),
        "end": present[-1],
    }


def summarize_npu_resource_samples(
    samples: list[NpuResourceSample],
    *,
    interval_s: float,
    npu_id: int,
    chip_id: int,
    error: str | None = None,
) -> dict[str, Any]:
    if not samples:
        return {
            "available": False,
            "npu_id": npu_id,
            "chip_id": chip_id,
            "sample_interval_s": interval_s,
            "samples": 0,
            "error": error or "no NPU resource samples were collected",
        }

    required_missing = []
    if not any(sample.hbm_usage_percent is not None for sample in samples):
        required_missing.append("hbm_usage_percent")
    if not any(
        sample.aicore_usage_percent is not None
        or sample.npu_util_percent is not None
        for sample in samples
    ):
        required_missing.append("aicore_usage_percent or npu_util_percent")
    if required_missing and error is None:
        error = "missing required NPU metrics: " + ", ".join(required_missing)

    return {
        "available": error is None,
        "npu_id": npu_id,
        "chip_id": chip_id,
        "chip_count": max(sample.chip_count for sample in samples),
        "sample_interval_s": interval_s,
        "samples": len(samples),
        "duration_s": samples[-1].elapsed_s,
        "hbm_capacity_mb": _optional_series_summary(
            [
                (
                    float(sample.hbm_capacity_mb)
                    if sample.hbm_capacity_mb is not None
                    else None
                )
                for sample in samples
            ]
        ),
        "hbm_usage_percent": _optional_series_summary(
            [sample.hbm_usage_percent for sample in samples]
        ),
        "aicore_usage_percent": _optional_series_summary(
            [sample.aicore_usage_percent for sample in samples]
        ),
        "npu_util_percent": _optional_series_summary(
            [sample.npu_util_percent for sample in samples]
        ),
        "hbm_temp_c": _optional_series_summary(
            [sample.hbm_temp_c for sample in samples]
        ),
        "power_w": _optional_series_summary([sample.power_w for sample in samples]),
        "system_cpu_percent": _optional_series_summary(
            [sample.system_cpu_percent for sample in samples]
        ),
        "raw_samples": [asdict(sample) for sample in samples],
        "error": error,
    }


class NpuResourceMonitor:
    """Sample one explicitly selected Ascend NPU chip in a background thread."""

    def __init__(
        self,
        npu_id: int = 0,
        chip_id: int = 0,
        interval_s: float = 1.0,
    ) -> None:
        if npu_id < 0:
            raise ValueError("npu_id must be >= 0")
        if chip_id < 0:
            raise ValueError("chip_id must be >= 0")
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.npu_id = npu_id
        self.chip_id = chip_id
        self.interval_s = interval_s
        self.samples: list[NpuResourceSample] = []
        self.error: str | None = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._psutil: Any = None

    def start(self) -> "NpuResourceMonitor":
        if not _NPU_SMI_LOCK.acquire(blocking=False):
            self.error = "another NPU resource monitor is still active"
            self._ready_event.set()
            return self
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name=f"benchmark-npu-monitor-{self.npu_id}-{self.chip_id}",
            daemon=True,
        )
        try:
            self._thread.start()
        except BaseException:
            _NPU_SMI_LOCK.release()
            raise
        if not self._ready_event.wait(timeout=max(10.0, self.interval_s * 5)):
            self.error = "NPU resource sampler initialization timed out"
        return self

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self.interval_s * 5))
            if self._thread.is_alive() and self.error is None:
                self.error = "NPU resource sampler did not stop before timeout"
        return summarize_npu_resource_samples(
            list(self.samples),
            interval_s=self.interval_s,
            npu_id=self.npu_id,
            chip_id=self.chip_id,
            error=self.error,
        )

    def _run(self) -> None:
        try:
            try:
                import psutil

                self._psutil = psutil
                psutil.cpu_percent(interval=None)
                self._sample_once()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self._ready_event.set()
            if self.error is not None:
                return
            while not self._stop_event.wait(self.interval_s):
                self._sample_once()
                if self.error is not None:
                    return
            self._sample_once()
        finally:
            _NPU_SMI_LOCK.release()

    def _sample_once(self) -> None:
        try:
            selector = ("-i", str(self.npu_id), "-c", str(self.chip_id))
            usages = _run_npu_smi("info", "-t", "usages", *selector)
            memory = _run_npu_smi("info", "-t", "memory", *selector)
            power = _run_npu_smi("info", "-t", "power", *selector)
            if usages is None or memory is None or power is None:
                raise RuntimeError("one or more npu-smi sampling commands failed")

            capacities = _label_values(usages, ("HBM Capacity",))
            hbm_usage = _label_values(
                usages, ("HBM Usage Rate", "HBM Usage", "Memory Usage Rate")
            )
            aicore_usage = _label_values(
                usages, ("Aicore Usage Rate", "AI Core Usage Rate")
            )
            npu_util = _label_values(
                usages, ("NPU Utilization", "NPU Usage Rate")
            )
            temperatures = _label_values(
                memory, ("HBM Temperature", "HBM Temp")
            )
            powers = _label_values(
                power, ("NPU Real-time Power", "NPU Power", "Power")
            )
            capacity = _pick(capacities, self.chip_id)
            sample = NpuResourceSample(
                elapsed_s=time.perf_counter() - self._started_at,
                npu_id=self.npu_id,
                chip_id=self.chip_id,
                chip_count=max(1, len(capacities)),
                hbm_capacity_mb=int(capacity) if capacity is not None else None,
                hbm_usage_percent=_pick(hbm_usage, self.chip_id),
                aicore_usage_percent=_pick(aicore_usage, self.chip_id),
                npu_util_percent=_pick(npu_util, self.chip_id),
                hbm_temp_c=_pick(temperatures, self.chip_id),
                power_w=_pick(powers, self.chip_id),
                system_cpu_percent=float(self._psutil.cpu_percent(interval=None)),
            )
            self.samples.append(sample)
            if sample.hbm_usage_percent is None or (
                sample.aicore_usage_percent is None
                and sample.npu_util_percent is None
            ):
                raise RuntimeError(
                    "npu-smi output omitted required HBM/utilization metrics"
                )
        except Exception as exc:
            if self.error is None:
                self.error = f"{type(exc).__name__}: {exc}"


def collect_npu_environment_fingerprint(
    npu_ids: list[int] | None = None,
) -> dict[str, Any]:
    ids = npu_ids or [0]
    cards = []
    for npu_id in ids:
        if npu_id < 0:
            raise ValueError("npu_ids must contain only non-negative values")
        output = _run_npu_smi("info", "-i", str(npu_id))
        if output is None:
            cards.append({"npu_id": npu_id, "available": False})
            continue
        chip_name = _first_label(output, ("Name", "Chip Name", "NPU Name"))
        health = _first_label(output, ("Health",))
        capacities = _label_values(output, ("HBM Capacity",))
        cards.append(
            {
                "npu_id": npu_id,
                "available": True,
                "chip_name": chip_name,
                "health": health,
                "hbm_capacity_mb": int(capacities[0]) if capacities else None,
                "raw_sha256": __import__("hashlib").sha256(output.encode()).hexdigest(),
            }
        )
    version = _run_npu_smi("info", "-t", "board", "-i", str(ids[0]))
    return {
        "available": bool(cards) and all(card["available"] for card in cards),
        "platform": platform.platform(),
        "ascend_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "driver_info_sha256": (
            __import__("hashlib").sha256(version.encode()).hexdigest()
            if version
            else None
        ),
        "cards": cards,
    }


def _first_label(text: str, labels: Iterable[str]) -> str | None:
    alternatives = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"(?:{alternatives})\s*(?:[:|=]\s*)?([^|\r\n]+)",
        text or "",
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


__all__ = [
    "NpuResourceMonitor",
    "NpuResourceSample",
    "collect_npu_environment_fingerprint",
    "summarize_npu_resource_samples",
]
