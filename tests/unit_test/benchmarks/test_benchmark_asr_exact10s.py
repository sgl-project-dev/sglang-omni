# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

from benchmarks.benchmarker.data import RequestResult
from benchmarks.eval import benchmark_asr_exact10s
from benchmarks.eval.benchmark_asr_exact10s import (
    _aggregate_repeat_metrics,
    _parse_concurrencies,
    _split_manifest,
    _validate_args,
)
from benchmarks.manifest.exact10s import TARGET_FRAMES, validate_clip_duration_with_ref


def _write_wav(path: Path, fill: int) -> Path:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(bytes([fill]) * TARGET_FRAMES * 2)
    return path


def _sample(tmp_path: Path, sample_id: str, fill: int):
    return validate_clip_duration_with_ref(
        _write_wav(tmp_path / f"{sample_id}.wav", fill),
        "hello world",
        sample_id,
    )


def _args(**overrides):
    values = {
        "port": 8000,
        "repeats": 1,
        "warmup_samples": 0,
        "max_samples": 1,
        "min_distinct_audio": 1,
        "request_timeout_s": 120.0,
        "monitor_interval_s": 1.0,
        "npu_id": 0,
        "npu_chip_id": 0,
        "hard_gate": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("value", ["", "0", "1,-2", "1,a", "2,2"])
def test_exact10s_cli_rejects_invalid_concurrencies(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_concurrencies(value)


def test_exact10s_cli_parses_concurrencies() -> None:
    assert _parse_concurrencies("1,2,70") == (1, 2, 70)


@pytest.mark.parametrize("field", ["npu_id", "npu_chip_id"])
def test_exact10s_cli_rejects_negative_npu_id(field: str) -> None:
    with pytest.raises(ValueError, match="NPU identifiers"):
        _validate_args(_args(**{field: -1}))


def test_hard_gate_rejects_client_loop_fresh_process_substitute() -> None:
    with pytest.raises(ValueError, match="restart the server"):
        _validate_args(
            _args(
                hard_gate=True,
                repeats=3,
                warmup_samples=70,
                max_samples=700,
                min_distinct_audio=770,
            )
        )


def test_exact10s_warmup_carves_disjoint_manifest(tmp_path: Path) -> None:
    samples = [_sample(tmp_path, f"s{index}", index + 1) for index in range(3)]
    warmup, measured = _split_manifest(
        samples,
        warmup_count=1,
        measured_count=2,
    )
    assert [sample.sample_id for sample in warmup] == ["s0"]
    assert [sample.sample_id for sample in measured] == ["s1", "s2"]


def test_exact10s_warmup_rejects_duplicate_audio_overlap(tmp_path: Path) -> None:
    first = _sample(tmp_path, "first", 1)
    duplicate = validate_clip_duration_with_ref(
        first.wav_path,
        "same audio",
        "duplicate",
    )
    with pytest.raises(ValueError, match="overlap"):
        _split_manifest([first, duplicate], warmup_count=1, measured_count=1)


def test_aggregate_retains_failure_latency_and_invalidates_repeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = [_sample(tmp_path, "ok", 1), _sample(tmp_path, "bad", 2)]
    outputs = [
        RequestResult(
            request_id="ok",
            text="hello world",
            is_success=True,
            latency_s=0.1,
            audio_duration_s=10.0,
            rtf=0.01,
        ),
        RequestResult(
            request_id="bad",
            is_success=False,
            latency_s=120.0,
            audio_duration_s=10.0,
            rtf=12.0,
            error="timeout after 120.000s",
        ),
    ]
    monkeypatch.setattr(
        benchmark_asr_exact10s,
        "build_asr_eval_results",
        lambda *_args, **_kwargs: {
            "summary": {"corpus_wer": 0.0, "evaluated": 1, "skipped": 1}
        },
    )
    summary, rows = _aggregate_repeat_metrics(
        samples,
        outputs,
        120.0,
        lang="en",
        model_name="model",
        concurrency=2,
    )
    assert summary["valid"] is False
    assert summary["failed_count"] == 1
    assert summary["timeout_count"] == 1
    assert summary["latency_p50_s"] == 0.1
    assert summary["latency_p95_s"] == 120.0
    assert rows[1]["timeout"] is True
    assert rows[1]["latency_s"] == 120.0


def test_aggregate_rejects_and_retains_duplicate_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = _sample(tmp_path, "one", 1)
    outputs = [
        RequestResult(
            request_id="one",
            text="hello world",
            is_success=True,
            latency_s=0.1,
            audio_duration_s=10.0,
            rtf=0.01,
        ),
        RequestResult(
            request_id="one",
            text="hello world",
            is_success=True,
            latency_s=0.2,
            audio_duration_s=10.0,
            rtf=0.02,
        ),
    ]
    monkeypatch.setattr(
        benchmark_asr_exact10s,
        "build_asr_eval_results",
        lambda *_args, **_kwargs: {
            "summary": {"corpus_wer": 0.0, "evaluated": 1, "skipped": 0}
        },
    )
    summary, _ = _aggregate_repeat_metrics(
        [sample],
        outputs,
        0.2,
        lang="en",
        model_name="model",
        concurrency=1,
    )
    assert summary["valid"] is False
    assert summary["duplicate_result_count"] == 1


def test_latency_starts_after_multipart_body_is_sent(tmp_path: Path) -> None:
    sample = _sample(tmp_path, "one", 1)

    async def run_test() -> RequestResult:
        async def transcribe(request: web.Request) -> web.Response:
            await request.read()
            await asyncio.sleep(0.05)
            return web.json_response({"text": "hello world"})

        app = web.Application()
        app.router.add_post("/v1/audio/transcriptions", transcribe)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            sockets = site._server.sockets
            port = sockets[0].getsockname()[1]
            outputs, _ = await benchmark_asr_exact10s.run_exact10s_once(
                [sample],
                host="127.0.0.1",
                port=port,
                model_name="model",
                concurrency=1,
                request_timeout_s=2.0,
            )
            return outputs[0]
        finally:
            await runner.cleanup()

    result = asyncio.run(run_test())
    assert result.is_success is True
    assert result.error == ""
    assert result.latency_s >= 0.04


def test_repeat_invalidates_npu_monitor_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = _sample(tmp_path, "one", 1)

    class FakeMonitor:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            return self

        def stop(self):
            return {"available": False, "error": "npu-smi failed"}

    async def fake_run(*_args, **_kwargs):
        return [
            RequestResult(
                request_id="one",
                text="hello world",
                is_success=True,
                latency_s=0.1,
                audio_duration_s=10.0,
                rtf=0.01,
            )
        ], 0.1

    monkeypatch.setattr(benchmark_asr_exact10s, "NpuResourceMonitor", FakeMonitor)
    monkeypatch.setattr(benchmark_asr_exact10s, "run_exact10s_once", fake_run)
    monkeypatch.setattr(
        benchmark_asr_exact10s,
        "build_asr_eval_results",
        lambda *_args, **_kwargs: {
            "summary": {"corpus_wer": 0.0, "evaluated": 1, "skipped": 0}
        },
    )
    args = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model="model",
        request_timeout_s=120.0,
        npu_id=0,
        npu_chip_id=0,
        monitor_interval_s=1.0,
        lang="en",
        save_raw_dir=str(tmp_path / "raw"),
    )
    result = asyncio.run(
        benchmark_asr_exact10s._run_one_repeat(
            args, [sample], 1, 1, warmup_samples=[]
        )
    )
    assert result.valid is False
    assert result.invalid_reasons == ("NPU monitor failure: npu-smi failed",)
