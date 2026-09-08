# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)

from sglang_omni.model_runner.model_worker import (
    ModelWorker,
    _DecodeCudaGraphUsage,
    _PrefillCudaGraphUsage,
)


def _forward_batch(
    num_tokens: int,
    *,
    forward_mode: ForwardMode = ForwardMode.EXTEND,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        batch_size=num_tokens,
        forward_mode=forward_mode,
    )


def test_prefill_cuda_graph_usage_instances_do_not_share_buckets() -> None:
    first = _PrefillCudaGraphUsage()
    second = _PrefillCudaGraphUsage()

    first.replay_buckets[16] += 1

    assert first.replay_buckets == {16: 1}
    assert second.replay_buckets == {}


def test_model_worker_reports_actual_prefill_graph_replays_by_bucket(
    monkeypatch,
) -> None:
    prefill_runner = object.__new__(PrefillCudaGraphRunner)
    prefill_runner.capture_num_tokens = [16, 32]
    prefill_runner.backend = SimpleNamespace()
    prefill_runner.buffer_registry = SimpleNamespace(
        has_slot=lambda name: name == "input_embeds"
    )
    outcomes = iter(
        [
            SimpleNamespace(
                logits_output="graph-16",
                can_run_graph=True,
                expert_distribution_metrics=None,
            ),
            SimpleNamespace(
                logits_output="eager",
                can_run_graph=False,
                expert_distribution_metrics=None,
            ),
            SimpleNamespace(
                logits_output="graph-32",
                can_run_graph=True,
                expert_distribution_metrics=None,
            ),
            # Decode graphs must not be reported as prefill replays.
            SimpleNamespace(
                logits_output="decode-graph",
                can_run_graph=True,
                expert_distribution_metrics=None,
            ),
            # TARGET_VERIFY is an extend-like mode but belongs to the decode
            # graph runner and must not consume a stale prefill bucket.
            SimpleNamespace(
                logits_output="target-verify-graph",
                can_run_graph=True,
                expert_distribution_metrics=None,
            ),
        ]
    )

    def forward(*, forward_batch: object) -> object:
        del forward_batch
        return next(outcomes)

    runner = SimpleNamespace(
        forward=forward,
        decode_cuda_graph_runner=None,
        prefill_cuda_graph_runner=prefill_runner,
    )
    worker = object.__new__(ModelWorker)
    worker.dllm_algorithm = None
    worker.model_runner = runner
    worker._prefill_cuda_graph_usage = _PrefillCudaGraphUsage()
    worker._decode_cuda_graph_usage = _DecodeCudaGraphUsage()
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_model",
        lambda: SimpleNamespace(model_path="model", load_format="auto"),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_serving",
        lambda: SimpleNamespace(weight_version=None),
    )
    worker.server_args = SimpleNamespace(
        tp_size=1,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="disabled"),
            prefill=SimpleNamespace(
                backend="breakable",
                bs=[16, 32],
            ),
        ),
    )
    worker.tp_rank = 0
    worker.model_arch_override = None
    ModelWorker.forward_batch_generation(worker, _forward_batch(5))
    ModelWorker.forward_batch_generation(worker, _forward_batch(40))
    ModelWorker.forward_batch_generation(worker, _forward_batch(31))
    ModelWorker.forward_batch_generation(
        worker,
        _forward_batch(1, forward_mode=ForwardMode.DECODE),
    )
    ModelWorker.forward_batch_generation(
        worker,
        _forward_batch(1, forward_mode=ForwardMode.TARGET_VERIFY),
    )
    ModelWorker.record_custom_prefill_eager(worker)

    stats = ModelWorker.model_info(worker)["prefill_cuda_graph"]

    assert stats["backend"] == "breakable"
    assert stats["capture_num_tokens"] == [16, 32]
    assert stats["runner"] == "PrefillCudaGraphRunner"
    assert stats["backend_runner"] == "SimpleNamespace"
    assert stats["input_embeds_slot"] is True
    assert stats["replay_count"] == 2
    assert stats["standard_eager_count"] == 1
    assert stats["custom_eager_count"] == 1
    assert stats["replay_buckets"] == {"16": 1, "32": 1}
    assert json.loads(json.dumps(stats)) == stats


def test_model_worker_reports_actual_decode_graph_replays_by_bucket(
    monkeypatch,
) -> None:
    # SGLang classifies DECODE as a CUDA-graph execution mode even when a
    # particular batch falls back to eager; can_run_graph distinguishes those
    # two outcomes for usage accounting.
    assert ForwardMode.DECODE.is_decode()
    assert ForwardMode.DECODE.is_cuda_graph()

    decode_runner = SimpleNamespace(
        capture_bs=[1, 2, 4],
        compile_bs=[1, 2, 4],
        enable_torch_compile=True,
        bs=2,
        backend=SimpleNamespace(),
    )
    outcomes = iter(
        [
            SimpleNamespace(
                logits_output="graph-2",
                can_run_graph=True,
                expert_distribution_metrics=None,
            ),
            SimpleNamespace(
                logits_output="eager",
                can_run_graph=False,
                expert_distribution_metrics=None,
            ),
        ]
    )

    runner = SimpleNamespace(
        forward=lambda *, forward_batch: next(outcomes),
        decode_cuda_graph_runner=decode_runner,
        prefill_cuda_graph_runner=None,
    )
    worker = object.__new__(ModelWorker)
    worker.dllm_algorithm = None
    worker.model_runner = runner
    worker._prefill_cuda_graph_usage = _PrefillCudaGraphUsage()
    worker._decode_cuda_graph_usage = _DecodeCudaGraphUsage()
    worker.server_args = SimpleNamespace(
        tp_size=1,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="full"),
            prefill=SimpleNamespace(backend="disabled"),
        ),
    )
    worker.tp_rank = 0
    worker.model_arch_override = None
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_model",
        lambda: SimpleNamespace(model_path="model", load_format="auto"),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_serving",
        lambda: SimpleNamespace(weight_version=None),
    )

    ModelWorker.forward_batch_generation(
        worker,
        _forward_batch(2, forward_mode=ForwardMode.DECODE),
    )
    ModelWorker.forward_batch_generation(
        worker,
        _forward_batch(3, forward_mode=ForwardMode.DECODE),
    )

    stats = ModelWorker.model_info(worker)["decode_cuda_graph"]

    assert stats == {
        "backend": "full",
        "runner": "SimpleNamespace",
        "backend_runner": "SimpleNamespace",
        "capture_bs": [1, 2, 4],
        "torch_compile_enabled": True,
        "compile_bs": [1, 2, 4],
        "replay_count": 1,
        "standard_eager_count": 1,
        "replay_buckets": {"2": 1},
    }
    assert json.loads(json.dumps(stats)) == stats


def test_model_worker_exposes_encoder_graph_runner_info() -> None:
    expected = {
        "enabled": True,
        "captured_graph_count": 2,
        "replay_count": 7,
    }
    encoder_runner = SimpleNamespace(model_info=lambda: expected)
    worker = object.__new__(ModelWorker)
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(_encoder_graph_runner=encoder_runner)
    )

    assert ModelWorker._encoder_cuda_graph_info(worker) is expected
