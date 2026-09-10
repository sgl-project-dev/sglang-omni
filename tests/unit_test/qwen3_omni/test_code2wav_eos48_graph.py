# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.qwen3_omni.components import code2wav_scheduler as mod
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import GraphKey
from tests.unit_test.qwen3_omni.test_code2wav_cuda_graph import _codes
from tests.unit_test.qwen3_omni.test_code2wav_eos_graph import (
    BASE_KEYS,
    _baseline,
    _BorrowedRunner,
    _MemoryBackend,
    _optional,
    _RecordingModel,
    _state,
)

EXTRA48 = tuple(
    GraphKey(1, frames)
    for frames in range(1, 49)
    if GraphKey(1, frames) not in BASE_KEYS
)


def test_wider_single_optional_pool_covers_contiguous_range_with_same_budget(
    monkeypatch,
):
    model, baseline_backend, baseline = _baseline()
    before = baseline.stats()
    baseline_graphs = dict(baseline._graphs)
    backend = _MemoryBackend(before=(140, 400), after=(180, 501))
    optional, calls = _optional(monkeypatch, model, baseline, backend, max_frames=48)
    assert len(calls) == 1
    assert calls[0]["graph_keys"] == EXTRA48
    assert len(EXTRA48) == 44
    assert calls[0]["graph_memory_budget_bytes"] == 120
    assert optional.stats()["build"]["published_graph_count"] == 44
    assert optional.stats()["memory"]["graph_budget_bytes"] == 120
    assert backend.capture_calls == 44
    assert backend.warmup_iterations == [3] * 44
    assert baseline.stats() == before
    assert baseline._graphs == baseline_graphs
    assert optional._pool is not baseline._pool
    assert optional._capture_stream is not baseline._capture_stream
    assert {
        frames
        for frames in range(1, 50)
        if baseline.available_batch_sizes(frames)
        or optional.available_batch_sizes(frames)
    } == set(range(1, 49))
    for offset in (1, 19, 1):
        for key in (*BASE_KEYS, *EXTRA48):
            target = baseline if key in BASE_KEYS else optional
            target_backend = baseline_backend if key in BASE_KEYS else backend
            codes = _codes(target_backend, key.batch_size, key.frames)
            codes.add_(offset)
            expected = model(codes).clone()
            replay = target.run(codes)
            assert replay.execution_mode == "cuda_graph"
            assert torch.equal(replay.output, expected)


@pytest.mark.parametrize("failure", ["oom", "equivalence", "replay", "budget"])
def test_failure_at_expanded_tier_does_not_publish_partial_optional_coverage(
    monkeypatch, failure
):
    model, baseline_backend, baseline = _baseline()
    before = baseline.stats()
    graphs = dict(baseline._graphs)
    last_key = len(EXTRA48) - 1
    backend = _MemoryBackend(
        before=(140, 400),
        after=(180, 540) if failure == "budget" else (180, 501),
        capture_error_at=last_key if failure == "oom" else None,
        corrupt_at=last_key if failure == "equivalence" else None,
        replay_error_at=last_key if failure == "replay" else None,
    )
    optional, calls = _optional(monkeypatch, model, baseline, backend, max_frames=48)
    assert calls[0]["graph_keys"] == EXTRA48
    assert optional.stats()["enabled"] is False
    assert optional.stats()["build"]["published_graph_count"] == 0
    assert all(optional.available_batch_sizes(frames) == () for frames in range(1, 49))
    assert baseline.stats() == before
    assert baseline._graphs == graphs
    assert baseline.available_batch_sizes(35) == (8, 4, 2, 1)
    assert baseline.run(_codes(baseline_backend, 8, 35)).execution_mode == "cuda_graph"


def _factory(monkeypatch, *, enabled=True, cap=48):
    model = _RecordingModel()
    model.config = SimpleNamespace(num_quantizers=2)
    monkeypatch.setattr(mod, "load_code2wav_model", lambda *a, **k: model)
    builds = []
    runners = []

    class _Builder:
        @classmethod
        def build(cls, model, **kwargs):
            builds.append(kwargs)
            runner = _BorrowedRunner(model, kwargs["graph_keys"])
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
            runners.append(runner)
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
        eos_cuda_graph_max_frames=cap,
        total_gpu_memory_fraction=0.5,
    )
    return scheduler, model, builds, runners


@pytest.mark.parametrize("cap", [36, 48])
def test_factory_accepts_expanded_cap_and_keeps_baseline_matrix(monkeypatch, cap):
    scheduler, _, builds, _ = _factory(monkeypatch, cap=cap)
    assert builds[0]["graph_keys"] == BASE_KEYS
    assert "graph_memory_budget_bytes" not in builds[0]
    assert len(builds) == 2
    assert builds[1]["graph_keys"] == tuple(key for key in EXTRA48 if key.frames <= cap)
    assert builds[1]["graph_memory_budget_bytes"] == 120
    assert scheduler._enable_eos_cuda_graph is True
    assert scheduler._chunk_aligned_dispatch is True


@pytest.mark.parametrize("frames", range(36, 50))
def test_expanded_final_windows_preserve_exact_inputs_samples_and_owned_output(
    monkeypatch, frames
):
    snapshots = []
    for enabled in (False, True):
        scheduler, model, _, runners = _factory(monkeypatch, enabled=enabled)
        state = _state(frames)
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
        assert len(waveform) == min(frames * 4 - 1, (len(state.chunks) - start) * 4)
        assert scheduler.decode_delta("a", state, is_final=True) is None
        if enabled and frames <= 48:
            assert runners[1].calls[-1] == ((1, 2, frames), True, "cuda_graph")
            assert runners[0].calls == []
        else:
            assert runners[0].calls[-1][2] == "eager"
            assert len(runners) == 1 or runners[1].calls == []
        scheduler.decode_delta("b", _state(frames, offset=19), is_final=True)
        assert torch.equal(waveform, saved)
        np.testing.assert_array_equal(state.audio_parts[-1], stored)
        snapshots.append((saved, model.inputs[0]))
    assert torch.equal(snapshots[0][0], snapshots[1][0])
    assert torch.equal(snapshots[0][1], snapshots[1][1])


def test_expansion_still_excludes_nonfinal_and_batched_requests(monkeypatch):
    scheduler, _, _, runners = _factory(monkeypatch)
    scheduler.decode_delta("a", _state(48), is_final=False)
    assert runners[1].calls == []
    assert runners[0].calls[-1] == ((1, 2, 48), True, "eager")
    batched = torch.arange(2 * 2 * 48).reshape(2, 2, 48)
    _, execution = scheduler._forward_codes(batched, graph_eligible=True, is_final=True)
    assert execution["execution_mode"] == "eager"
    assert execution["fallback_reason"] == "key_miss"
    assert runners[1].calls == []
    assert runners[0].calls[-1] == ((2, 2, 48), True, "eager")
