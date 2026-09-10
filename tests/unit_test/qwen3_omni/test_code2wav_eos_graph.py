# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.qwen3_omni.components import code2wav_scheduler as mod
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavCudaGraphRunner,
    Code2WavRunResult,
    GraphKey,
)
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel, make_qwen_payload
from tests.unit_test.qwen3_omni.test_code2wav_batching import _FakeGraphRunner
from tests.unit_test.qwen3_omni.test_code2wav_cuda_graph import (
    _codes,
    _FakeCudaBackend,
    _FakeModel,
)

BASE_KEYS = mod._batched_graph_keys(10, 25, 8)
EXTRA_KEYS = tuple(
    GraphKey(1, n) for n in range(1, 36) if GraphKey(1, n) not in BASE_KEYS
)


class _MemoryBackend(_FakeCudaBackend):
    def __init__(self, *, before=(100, 120), after=(140, 400), **kwargs):
        super().__init__(**kwargs)
        self.before = before
        self.after = after
        self.snapshots = 0

    def memory_stats(self, device):
        del device
        allocated, reserved = self.before if self.snapshots == 0 else self.after
        self.snapshots += 1
        return dict(
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            max_reserved_bytes=reserved,
            free_bytes=1000 - reserved,
            total_bytes=1000,
        )


def _baseline():
    model = _FakeModel()
    model.config = SimpleNamespace(num_quantizers=16)
    backend = _MemoryBackend()
    runner = Code2WavCudaGraphRunner.build(
        model,
        device="cuda:0",
        num_quantizers=16,
        total_gpu_memory_fraction=0.5,
        graph_keys=BASE_KEYS,
        device_api=backend,
    )
    assert runner.stats()["build"]["published_graph_count"] == 16
    return model, backend, runner


def _optional(monkeypatch, model, baseline, backend, *, max_frames=35):
    captured = []

    class _Builder:
        @classmethod
        def build(cls, model, **kwargs):
            captured.append(kwargs)
            return Code2WavCudaGraphRunner.build(model, device_api=backend, **kwargs)

    monkeypatch.setattr(mod, "Code2WavCudaGraphRunner", _Builder)
    optional = mod._build_eos_cuda_graph_runner(
        model,
        baseline=baseline,
        device=torch.device("cuda:0"),
        total_gpu_memory_fraction=0.5,
        max_frames=max_frames,
    )
    return optional, captured


def test_optional_pool_captures_only_missing_keys_and_charges_reserved_baseline(
    monkeypatch,
):
    model, baseline_backend, baseline = _baseline()
    before = baseline.stats()
    graph_objects = dict(baseline._graphs)
    backend = _MemoryBackend(before=(140, 400), after=(180, 501))
    optional, calls = _optional(monkeypatch, model, baseline, backend)
    assert len(calls) == 1
    assert calls[0]["graph_keys"] == EXTRA_KEYS
    assert len(EXTRA_KEYS) == 31
    assert calls[0]["graph_memory_budget_bytes"] == 120
    assert optional.stats()["memory"]["graph_budget_bytes"] == 120
    assert optional.stats()["build"]["published_graph_count"] == 31
    assert baseline.stats() == before
    assert baseline._graphs == graph_objects
    assert baseline._pool is not optional._pool
    assert baseline._capture_stream is not optional._capture_stream
    assert backend.capture_calls == 31
    assert backend.warmup_iterations == [3] * 31
    for generation in (1, 2):
        for key in (*BASE_KEYS, *EXTRA_KEYS):
            target = baseline if key in BASE_KEYS else optional
            target_backend = baseline_backend if key in BASE_KEYS else backend
            codes = _codes(target_backend, key.batch_size, key.frames)
            codes.add_(generation)
            expected = model(codes).clone()
            replay = target.run(codes)
            assert replay.execution_mode == "cuda_graph"
            assert torch.equal(replay.output, expected)


@pytest.mark.parametrize("failure", ["oom", "equivalence", "replay", "budget"])
def test_optional_startup_failure_preserves_baseline_graphs_and_batching(
    monkeypatch, failure
):
    model, baseline_backend, baseline = _baseline()
    before = baseline.stats()
    graphs = dict(baseline._graphs)
    backend = _MemoryBackend(
        before=(140, 400),
        after=(180, 540) if failure == "budget" else (180, 501),
        capture_error_at=0 if failure == "oom" else None,
        corrupt_at=0 if failure == "equivalence" else None,
        replay_error_at=0 if failure == "replay" else None,
    )
    optional, _ = _optional(monkeypatch, model, baseline, backend)
    assert optional.stats()["enabled"] is False
    assert optional.stats()["build"]["published_graph_count"] == 0
    assert baseline.stats() == before
    assert baseline._graphs == graphs
    assert baseline.available_batch_sizes(35) == (8, 4, 2, 1)
    assert baseline.run(_codes(baseline_backend, 8, 35)).execution_mode == "cuda_graph"
    assert baseline.run(_codes(baseline_backend, 1, 30)).execution_mode == "cuda_graph"
    assert optional.available_batch_sizes(34) == ()


def test_existing_keys_only_does_not_build_optional_pool(monkeypatch):
    model, _, baseline = _baseline()
    optional, calls = _optional(
        monkeypatch, model, baseline, _MemoryBackend(), max_frames=0
    )
    assert optional is None
    assert calls == []


@pytest.mark.parametrize("reason", ["baseline_disabled", "no_remaining_graph_budget"])
def test_unavailable_baseline_or_budget_never_captures_optional(monkeypatch, reason):
    model, _, baseline = _baseline()
    stats = baseline.stats()
    if reason == "baseline_disabled":
        stats["enabled"] = False
    else:
        stats["memory"]["graph_footprint_bytes"] = 400
    monkeypatch.setattr(baseline, "stats", lambda: stats)
    optional, calls = _optional(monkeypatch, model, baseline, _MemoryBackend())
    assert optional is None
    assert calls == []


@pytest.mark.parametrize("cap", [-1, True, 1.5])
def test_runner_rejects_invalid_memory_cap_before_capture(cap):
    backend = _MemoryBackend()
    with pytest.raises(ValueError, match="nonnegative integer"):
        Code2WavCudaGraphRunner.build(
            _FakeModel(),
            device="cuda:0",
            num_quantizers=16,
            total_gpu_memory_fraction=0.5,
            graph_keys=EXTRA_KEYS,
            graph_memory_budget_bytes=cap,
            device_api=backend,
        )
    assert backend.capture_calls == 0


class _RecordingModel(FakeCode2WavModel):
    def __init__(self):
        super().__init__(total_upsample=4, output_deficit=1)
        self.inputs = []

    def __call__(self, codes):
        self.inputs.append(codes.clone())
        return super().__call__(codes)


class _BorrowedRunner(_FakeGraphRunner):
    def __init__(self, model, keys):
        super().__init__(model, keys)
        self.buffers = {}

    def run(self, codes, *, eligible):
        result = super().run(codes, eligible=eligible)
        if result.execution_mode != "cuda_graph":
            return result
        buffer = self.buffers.setdefault(result.key, torch.empty_like(result.output))
        buffer.copy_(result.output)
        return Code2WavRunResult(
            buffer, result.execution_mode, result.key, result.fallback_reason
        )


def _scheduler(*, enabled=True, optional=True):
    model = _RecordingModel()
    baseline = _BorrowedRunner(model, BASE_KEYS)
    extra = _BorrowedRunner(model, EXTRA_KEYS if optional else ())
    scheduler = mod.Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=25,
        enable_batching=True,
        batch_ceiling=8,
        enable_cuda_graph=True,
        enable_eos_cuda_graph=enabled,
        _cuda_graph_runner=baseline,
        _eos_cuda_graph_runner=extra,
    )
    return scheduler, model, baseline, extra


def _state(window_frames, *, offset=0):
    start = 52 if window_frames > 25 else 0
    context = min(start, 25)
    end = start + window_frames - context
    return mod.Code2WavStreamState(
        chunks=[
            torch.tensor([(i + offset) % 2048, (3 * i + offset) % 2048])
            for i in range(end)
        ],
        emitted=start,
        checked=end,
        stream_enabled=True,
        audio_parts=[np.arange(3, dtype=np.float32)] if start else [],
    )


@pytest.mark.parametrize("window_frames", range(1, 48))
def test_final_replay_preserves_exact_context_tail_samples_and_saved_output(
    window_frames,
):
    snapshots = []
    for enabled in (False, True):
        scheduler, model, baseline, optional = _scheduler(enabled=enabled)
        state = _state(window_frames)
        start = state.emitted
        original_parts = len(state.audio_parts)
        expected_codes = torch.stack(
            state.chunks[start - min(start, 25) :]
        ).T.unsqueeze(0)
        waveform = scheduler.decode_delta("a", state, is_final=True)
        saved = waveform.clone()
        stored = state.audio_parts[-1].copy()
        assert torch.equal(model.inputs[-1], expected_codes)
        assert state.emitted == len(state.chunks)
        assert len(state.audio_parts) == original_parts + 1
        assert len(waveform) == min(
            window_frames * 4 - 1, (len(state.chunks) - start) * 4
        )
        assert scheduler.decode_delta("a", state, is_final=True) is None
        if enabled and GraphKey(1, window_frames) in EXTRA_KEYS:
            assert optional.calls[-1] == ((1, 2, window_frames), True, "cuda_graph")
            assert baseline.calls == []
        else:
            expected_mode = (
                "cuda_graph"
                if enabled and GraphKey(1, window_frames) in BASE_KEYS
                else "eager"
            )
            assert baseline.calls[-1][2] == expected_mode
            assert optional.calls == []
        scheduler.decode_delta("b", _state(window_frames, offset=19), is_final=True)
        assert torch.equal(waveform, saved)
        np.testing.assert_array_equal(state.audio_parts[-1], stored)
        snapshots.append((saved, model.inputs[0]))
    assert torch.equal(snapshots[0][0], snapshots[1][0])
    assert torch.equal(snapshots[0][1], snapshots[1][1])


def test_nonfinal_never_routes_optional_and_disabled_optional_preserves_fallback():
    scheduler, _, baseline, optional = _scheduler()
    scheduler.decode_delta("a", _state(34), is_final=False)
    assert optional.calls == []
    assert baseline.calls[-1] == ((1, 2, 34), True, "eager")
    scheduler, _, baseline, optional = _scheduler(optional=False)
    for frames in (30, 34, 35, 36):
        scheduler.decode_delta(str(frames), _state(frames), is_final=True)
    assert optional.calls == []
    assert [call[2] for call in baseline.calls] == [
        "cuda_graph",
        "eager",
        "cuda_graph",
        "eager",
    ]


def test_stream_done_emits_owned_tail_before_terminal_result():
    scheduler, _, _, optional = _scheduler()
    state = _state(34)
    scheduler._stream_states["a"] = state
    scheduler._stream_payloads["a"] = make_qwen_payload(request_id="a")
    messages = scheduler.on_stream_done("a")
    assert [message.type for message in messages] == ["stream", "result"]
    assert len(optional.calls) == 1
    assert state.emitted == len(state.chunks)
    assert scheduler.decode_delta("a", state, is_final=True) is None


@pytest.mark.parametrize("enabled", [False, True])
def test_factory_leaves_baseline_matrix_unchanged_and_plumbs_eos_flag(
    monkeypatch, enabled, caplog
):
    caplog.set_level("INFO", logger=mod.__name__)
    model = _RecordingModel()
    model.config = SimpleNamespace(num_quantizers=2)
    monkeypatch.setattr(mod, "load_code2wav_model", lambda *a, **k: model)
    builds = []

    class _Builder:
        @classmethod
        def build(cls, model, **kwargs):
            builds.append(kwargs)
            runner = _FakeGraphRunner(model, kwargs["graph_keys"])
            stats = runner.stats()
            stats.update(
                build={"published_graph_count": len(runner._keys)},
                memory={
                    "stage_budget_bytes": 500,
                    "loaded_model_footprint_bytes": 100,
                    "graph_footprint_bytes": 280,
                },
            )
            runner.stats = lambda: stats
            return runner

    monkeypatch.setattr(mod, "Code2WavCudaGraphRunner", _Builder)
    scheduler = mod.create_code2wav_scheduler(
        "fake",
        device="cpu",
        stream_chunk_size=10,
        left_context_size=25,
        enable_batching=True,
        batch_ceiling=8,
        enable_cuda_graph=True,
        enable_eos_cuda_graph=enabled,
        total_gpu_memory_fraction=0.5,
    )
    assert builds[0]["graph_keys"] == BASE_KEYS
    assert "graph_memory_budget_bytes" not in builds[0]
    assert len(builds) == (2 if enabled else 1)
    assert scheduler._enable_eos_cuda_graph is enabled
    assert scheduler._chunk_aligned_dispatch is True
    if enabled:
        assert builds[1]["graph_keys"] == EXTRA_KEYS
        assert builds[1]["graph_memory_budget_bytes"] == 120
    baseline_messages = [
        m
        for m in caplog.messages
        if m.startswith("Code2Wav device graph startup stats=")
    ]
    optional_messages = [
        m
        for m in caplog.messages
        if m.startswith("Code2Wav optional EOS graph startup stats=")
    ]
    assert len(baseline_messages) == 1
    assert len(optional_messages) == int(enabled)
    if enabled:
        report = json.loads(optional_messages[0].split("stats=", 1)[1])
        assert report["baseline_published_graph_count"] == 16
        assert report["optional_published_keys"] == report["optional_requested_keys"]
        assert len(report["optional_published_keys"]) == 31


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_eos_cuda_graph": True},
        {"eos_cuda_graph_max_frames": -1},
        {"eos_cuda_graph_max_frames": 49},
        {"eos_cuda_graph_max_frames": True},
    ],
)
def test_factory_rejects_invalid_opt_in_before_loading(monkeypatch, kwargs):
    monkeypatch.setattr(
        mod, "load_code2wav_model", lambda *a, **k: pytest.fail("loaded model")
    )
    with pytest.raises(ValueError):
        mod.create_code2wav_scheduler("fake", device="cpu", **kwargs)
