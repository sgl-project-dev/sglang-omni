# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sglang_omni.profiler.event_recorder import diag_emit, get_recorder
from sglang_omni.scheduling.pre_lm_encoder import PreLMEncoderService, QueueEntry

_STOP = object()


class _StubEncoder(PreLMEncoderService[SimpleNamespace, list[int], int]):
    def __init__(self) -> None:
        super().__init__(worker_name="test-encoder-diag")

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join(timeout=2)

    def _next_batch(self) -> tuple[list[QueueEntry[SimpleNamespace]], bool]:
        entry = self._queue.get()
        if entry is _STOP:
            return [], True
        return [entry], False

    def encode_batch(self, items: list[SimpleNamespace]) -> list[int]:
        return [item.value * 2 for item in items]

    def split_embeddings(
        self,
        items: list[SimpleNamespace],
        encoded: list[int],
    ) -> list[int]:
        del items
        return encoded

    def attach_embedding(self, item: SimpleNamespace, embedding: int) -> None:
        item.embedding = embedding


@pytest.fixture(autouse=True)
def _reset_recorder():
    recorder = get_recorder()
    if recorder.is_active():
        recorder.stop()
    yield
    if recorder.is_active():
        recorder.stop()


def _read_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def test_diag_emit_disabled_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SGLANG_OMNI_ENCODER_DIAG", raising=False)
    recorder = get_recorder()
    path = recorder.start("diag-disabled", str(tmp_path), "asr")

    diag_emit(
        request_id="request-1",
        stage="asr",
        event_name="encoder_enqueue",
    )
    recorder.stop()

    assert _read_events(path) == []


def test_diag_emit_enabled_adds_monotonic_clock(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SGLANG_OMNI_ENCODER_DIAG", "true")
    recorder = get_recorder()
    path = recorder.start("diag-enabled", str(tmp_path), "asr")

    diag_emit(
        request_id="request-1",
        stage="asr",
        event_name="encoder_enqueue",
        metadata={"value": 3},
    )
    recorder.stop()

    event = _read_events(path)[0]
    assert event["metadata"]["value"] == 3
    assert event["metadata"]["clock"] == "CLOCK_MONOTONIC"
    assert isinstance(event["metadata"]["monotonic_ns"], int)


def test_encoder_events_are_env_gated_and_request_correlated(
    monkeypatch,
    tmp_path,
) -> None:
    recorder = get_recorder()
    path = recorder.start("encoder-diag", str(tmp_path), "asr")
    monkeypatch.delenv("SGLANG_OMNI_ENCODER_DIAG", raising=False)
    disabled = _StubEncoder()
    try:
        assert (
            disabled._submit(SimpleNamespace(request_id="disabled", value=2)).result(
                timeout=2
            )
            == 4
        )
    finally:
        disabled.close()

    monkeypatch.setenv("SGLANG_OMNI_ENCODER_DIAG", "1")
    enabled = _StubEncoder()
    try:
        assert (
            enabled._submit(SimpleNamespace(request_id="request-1", value=3)).result(
                timeout=2
            )
            == 6
        )
    finally:
        enabled.close()
        recorder.stop()

    events = _read_events(path)
    assert [event["event_name"] for event in events] == [
        "encoder_enqueue",
        "encoder_batch_start",
        "encoder_encode_return",
        "encoder_batch_finish",
    ]
    assert {event["request_id"] for event in events} == {"request-1"}
    for event in events:
        assert isinstance(event["metadata"]["monotonic_ns"], int)
    assert events[1]["metadata"]["batch_size"] == 1
    assert events[2]["metadata"]["elapsed_ms"] >= 0
    assert events[3]["metadata"]["error_class"] is None
