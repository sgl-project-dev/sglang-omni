# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import contextmanager

import pytest

from benchmarks.eval import benchmark_tts_seedtts as tts
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics


@pytest.mark.parametrize(
    "model, options, phases",
    [
        ("tencent/AuK", [], ["generate", "transcribe"]),
        ("tencent/AuK-Flash", [], ["generate", "transcribe"]),
        ("tencent/AuK@revision", [], ["generate", "transcribe"]),
        ("tencent/AuK", ["--generate-only"], ["generate"]),
        ("tencent/AuK", ["--transcribe-only"], ["transcribe"]),
        ("tencent/AuK", ["--generate-only", "--use-existing-server"], ["generate"]),
        ("fishaudio/s2-pro", [], ["generate", "transcribe"]),
    ],
)
def test_shared_server_lifecycle(monkeypatch, model, options, phases):
    events = []
    servers = []

    @contextmanager
    def server(**kwargs):
        servers.append(kwargs)
        events.append("start")
        yield
        events.append("stop")

    async def generate(config):
        assert config.port == 18280
        assert config.max_samples == 2
        if model.startswith("tencent/AuK"):
            assert config.model == model
            assert config.concurrency == config.warmup == 1
        events.append("generate")

    def transcribe(config, **kwargs):
        assert kwargs["asr_router_port"] == 18280
        events.append("transcribe")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            model,
            "--port",
            "18280",
            "--max-samples",
            "2",
            *options,
        ],
    )
    monkeypatch.setattr(tts, "managed_omni_server", server)
    monkeypatch.setattr(tts, "benchmark", generate)
    monkeypatch.setattr(tts, "run_tts_seedtts_transcribe", transcribe)
    tts.main()

    if "--use-existing-server" in options:
        assert events == phases
        assert not servers
    else:
        assert events == [
            event for phase in phases for event in ("start", phase, "stop")
        ]
        if "generate" in phases:
            if model.startswith("tencent/AuK"):
                assert "max_running_requests" not in servers[0]
                assert "cuda_graph_max_bs" not in servers[0]
                assert servers[0]["server_config"].endswith("examples/configs/auk.yaml")
            else:
                assert servers[0]["max_running_requests"] == 64
                assert servers[0]["cuda_graph_max_bs"] == 64


def test_filtered_wer_mean_keeps_exactly_50_percent_and_excludes_failures():
    metrics = calculate_wer_metrics(
        [
            SampleOutput(is_success=True, wer=0, hits=10),
            SampleOutput(is_success=True, wer=0.5, hits=1, deletions=1),
            SampleOutput(is_success=True, wer=0.75, hits=1, deletions=3),
            SampleOutput(is_success=False),
        ],
        "en",
    )
    assert metrics["wer_below_50_per_sample_mean"] == 0.25
    assert metrics["wer_below_50_corpus"] == pytest.approx(1 / 12)
    assert metrics["n_above_50_pct_wer"] == 1
    assert metrics["evaluated"] == 3
    assert metrics["skipped"] == 1


def test_auk_explicit_options_override_model_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            "tencent/AuK",
            "--generate-only",
            "--use-existing-server",
            "--max-concurrency",
            "3",
            "--warmup",
            "0",
            "--seed",
            "7",
            "--output-dir",
            "custom-results",
            "--server-config",
            "custom.yaml",
        ],
    )

    async def generate(config):
        assert config.concurrency == 3
        assert config.warmup == 0
        assert config.seed == 7
        assert config.output_dir == "custom-results"
        assert config.server_config == "custom.yaml"

    monkeypatch.setattr(tts, "benchmark", generate)
    tts.main()
