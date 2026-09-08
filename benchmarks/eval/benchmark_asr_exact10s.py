# SPDX-License-Identifier: Apache-2.0
"""Strict exact-10-second Qwen3-ASR latency benchmark for Ascend NPU."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.manifest.exact10s import (
    Exact10sSample,
    _audio_bytes_hash,
    fingerprint_manifest,
    load_exact10s_manifest,
)
from benchmarks.runtime_metrics import collect_benchmark_provenance
from benchmarks.runtime_metrics_npu import (
    NpuResourceMonitor,
    collect_npu_environment_fingerprint,
)
from benchmarks.tasks.asr import build_asr_eval_results

DEFAULT_CONCURRENCIES = (1, 2, 8, 16, 32, 64, 70)
HARD_GATE_MEASURED_SAMPLES = 700
HARD_GATE_WARMUP_SAMPLES = 70


@dataclass(frozen=True)
class Exact10sRepeatResult:
    concurrency: int
    repeat: int
    valid: bool
    invalid_reasons: tuple[str, ...]
    wall_clock_s: float
    evaluated: int
    total: int
    skipped: int
    corpus_wer: float | None
    throughput_samples_per_s: float
    rtfx: float
    latency_mean_s: float | None
    latency_p50_s: float | None
    latency_median_s: float | None
    latency_p90_s: float | None
    latency_p95_s: float | None
    latency_p99_s: float | None
    latency_max_s: float | None
    rtf_mean: float | None
    rtf_p95: float | None
    failed_count: int
    timeout_count: int
    missing_result_count: int
    duplicate_result_count: int
    unexpected_result_count: int
    npu_resources: dict[str, Any]
    raw_jsonl: str


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _parse_concurrencies(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(token.strip()) for token in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "concurrencies must be comma-separated integers"
        ) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("concurrencies must contain positive integers")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("concurrencies must not contain duplicates")
    return parsed


def _split_manifest(
    samples: Sequence[Exact10sSample],
    *,
    warmup_count: int,
    measured_count: int | None,
) -> tuple[list[Exact10sSample], list[Exact10sSample]]:
    if warmup_count < 0:
        raise ValueError("warmup_count must be >= 0")
    if measured_count is not None and measured_count <= 0:
        raise ValueError("measured_count must be > 0")
    warmup = list(samples[:warmup_count])
    measured = list(samples[warmup_count:])
    if measured_count is not None:
        measured = measured[:measured_count]
    if not measured:
        raise ValueError("no measured samples remain after the warmup split")
    warm_hashes = {_audio_bytes_hash(sample.wav_path) for sample in warmup}
    measured_hashes = {_audio_bytes_hash(sample.wav_path) for sample in measured}
    overlap = warm_hashes & measured_hashes
    if overlap:
        raise ValueError(
            f"warmup and measured partitions overlap by {len(overlap)} audio inputs"
        )
    return warmup, measured


async def _send_exact10_sample(
    session: aiohttp.ClientSession,
    api_url: str,
    model_name: str,
    sample: Exact10sSample,
    request_timeout_s: float,
) -> RequestResult:
    result = RequestResult(
        request_id=sample.sample_id,
        audio_duration_s=sample.duration_s,
    )
    trace_context: dict[str, float | None] = {"body_sent_at": None}
    request_started_at = time.perf_counter()
    try:
        audio_bytes = Path(sample.wav_path).read_bytes()
        body, boundary = _multipart_body(
            model_name=model_name,
            language=sample.language,
            filename=Path(sample.wav_path).name,
            audio_bytes=audio_bytes,
        )

        async def timed_body():
            yield body
            # An async-iterable payload resumes only after aiohttp's stream
            # writer has accepted the yielded chunk. This timestamp therefore
            # follows the complete multipart body write, unlike the
            # on_request_chunk_sent trace signal, which fires before write().
            trace_context["body_sent_at"] = time.perf_counter()

        timeout = aiohttp.ClientTimeout(total=request_timeout_s)
        async with session.post(
            api_url,
            data=timed_body(),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            timeout=timeout,
        ) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
            else:
                payload = await response.json()
                text = payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    result.error = "HTTP 200 response omitted non-empty text"
                else:
                    result.text = text
                    result.is_success = True
    except asyncio.TimeoutError:
        result.error = f"timeout after {request_timeout_s:.3f}s"
    except (OSError, aiohttp.ClientError, json.JSONDecodeError, ValueError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        completed_at = time.perf_counter()
        body_sent_at = trace_context["body_sent_at"]
        if body_sent_at is None:
            result.latency_s = completed_at - request_started_at
            suffix = "request body completion was not observed"
            result.error = f"{result.error}; {suffix}" if result.error else suffix
            result.is_success = False
        else:
            result.latency_s = completed_at - body_sent_at
        result.rtf = result.latency_s / sample.duration_s
    return result


def _multipart_body(
    *, model_name: str, language: str, filename: str, audio_bytes: bytes
) -> tuple[bytes, str]:
    for name, value in (
        ("model_name", model_name),
        ("language", language),
        ("filename", filename),
    ):
        if "\r" in value or "\n" in value or (name == "filename" and '"' in value):
            raise ValueError(f"unsafe multipart {name}")
    boundary = f"sglang-omni-exact10-{uuid.uuid4().hex}"
    chunks = []
    for name, value in (
        ("model", model_name),
        ("language", language),
        ("response_format", "json"),
    ):
        chunks.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    chunks.extend(
        (
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode(),
            audio_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        )
    )
    return b"".join(chunks), boundary


async def run_exact10s_once(
    samples: Sequence[Exact10sSample],
    *,
    host: str,
    port: int,
    model_name: str,
    concurrency: int,
    request_timeout_s: float,
) -> tuple[list[RequestResult], float]:
    semaphore = asyncio.Semaphore(concurrency)
    api_url = f"http://{host}:{port}/v1/audio/transcriptions"
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    connector = aiohttp.TCPConnector(limit=concurrency)
    started_at = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
    ) as session:

        async def bounded(sample: Exact10sSample) -> RequestResult:
            async with semaphore:
                return await _send_exact10_sample(
                    session,
                    api_url,
                    model_name,
                    sample,
                    request_timeout_s,
                )

        outputs = await asyncio.gather(*(bounded(sample) for sample in samples))
    return outputs, time.perf_counter() - started_at


def _raw_record(sample: Exact10sSample, result: RequestResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_kind": "expected_result",
        "sample_id": sample.sample_id,
        "language": sample.language,
        "audio_sha256": _audio_bytes_hash(sample.wav_path),
        "duration_s": sample.duration_s,
        "success": result.is_success,
        "latency_s": result.latency_s,
        "rtf": result.rtf,
        "timeout": result.error.startswith("timeout after"),
        "error": result.error,
        "reference_text": sample.ref_text,
        "hypothesis_text": result.text,
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _aggregate_repeat_metrics(
    samples: Sequence[Exact10sSample],
    outputs: Sequence[RequestResult],
    wall_clock_s: float,
    *,
    lang: str,
    model_name: str,
    concurrency: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_by_id = {sample.sample_id: sample for sample in samples}
    output_groups: dict[str, list[RequestResult]] = {}
    for output in outputs:
        output_groups.setdefault(output.request_id, []).append(output)
    output_by_id = {request_id: group[0] for request_id, group in output_groups.items()}
    duplicate_count = sum(max(0, len(group) - 1) for group in output_groups.values())
    unexpected = [
        output
        for output in outputs
        if output.request_id not in expected_by_id
    ]
    complete_outputs: list[RequestResult] = []
    raw_rows: list[dict[str, Any]] = []
    missing_count = 0
    for sample in samples:
        output = output_by_id.get(sample.sample_id)
        if output is None:
            missing_count += 1
            output = RequestResult(
                request_id=sample.sample_id,
                audio_duration_s=sample.duration_s,
                latency_s=wall_clock_s,
                rtf=wall_clock_s / sample.duration_s,
                error="missing request result",
            )
        complete_outputs.append(output)
        raw_rows.append(_raw_record(sample, output))
        for duplicate in output_groups.get(sample.sample_id, [])[1:]:
            duplicate_row = _raw_record(sample, duplicate)
            duplicate_row["record_kind"] = "duplicate_result"
            raw_rows.append(duplicate_row)
    for output in unexpected:
        raw_rows.append(
            {
                "schema_version": 1,
                "sample_id": output.request_id,
                "record_kind": "unexpected_result",
                "success": output.is_success,
                "latency_s": output.latency_s,
                "rtf": output.rtf,
                "timeout": output.error.startswith("timeout after"),
                "error": output.error,
                "hypothesis_text": output.text,
            }
        )

    scored = build_asr_eval_results(
        list(samples),
        complete_outputs,
        wall_clock_s,
        lang,
        model_path=model_name,
        concurrency=concurrency,
    )
    latencies = [output.latency_s for output in complete_outputs]
    rtfs = [output.rtf for output in complete_outputs]
    failed = sum(not output.is_success for output in complete_outputs)
    timeouts = sum(
        output.error.startswith("timeout after") for output in complete_outputs
    )
    skipped = int(scored["summary"].get("skipped", 0) or 0)
    evaluated = int(scored["summary"].get("evaluated", len(samples) - failed - skipped))
    summary = {
        "valid": (
            failed == 0
            and missing_count == 0
            and skipped == 0
            and duplicate_count == 0
            and not unexpected
        ),
        "invalid_reasons": [
            reason
            for condition, reason in (
                (failed, f"{failed} failed request(s)"),
                (missing_count, f"{missing_count} missing request result(s)"),
                (duplicate_count, f"{duplicate_count} duplicate request result(s)"),
                (len(unexpected), f"{len(unexpected)} unexpected request result(s)"),
                (skipped, f"{skipped} unscoreable request(s)"),
            )
            if condition
        ],
        "evaluated": evaluated,
        "total": len(samples),
        "skipped": skipped,
        "failed_count": failed,
        "timeout_count": timeouts,
        "missing_result_count": missing_count,
        "duplicate_result_count": duplicate_count,
        "unexpected_result_count": len(unexpected),
        "corpus_wer": scored["summary"].get("corpus_wer"),
        "throughput_samples_per_s": len(samples) / wall_clock_s,
        "rtfx": sum(sample.duration_s for sample in samples) / wall_clock_s,
        "latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "latency_p90_s": _percentile(latencies, 0.90),
        "latency_p95_s": _percentile(latencies, 0.95),
        "latency_p99_s": _percentile(latencies, 0.99),
        "latency_max_s": max(latencies) if latencies else None,
        "rtf_mean": statistics.fmean(rtfs) if rtfs else None,
        "rtf_p95": _percentile(rtfs, 0.95),
    }
    return summary, raw_rows


async def _run_one_repeat(
    args: argparse.Namespace,
    samples: Sequence[Exact10sSample],
    concurrency: int,
    repeat: int,
    *,
    warmup_samples: Sequence[Exact10sSample],
) -> Exact10sRepeatResult:
    if warmup_samples:
        warm_outputs, _ = await run_exact10s_once(
            warmup_samples,
            host=args.host,
            port=args.port,
            model_name=args.model,
            concurrency=min(concurrency, len(warmup_samples)),
            request_timeout_s=args.request_timeout_s,
        )
        warm_failures = sum(not output.is_success for output in warm_outputs)
        if warm_failures:
            raise RuntimeError(f"warmup failed for {warm_failures} request(s)")

    monitor = NpuResourceMonitor(
        npu_id=args.npu_id,
        chip_id=args.npu_chip_id,
        interval_s=args.monitor_interval_s,
    ).start()
    try:
        outputs, wall_clock_s = await run_exact10s_once(
            samples,
            host=args.host,
            port=args.port,
            model_name=args.model,
            concurrency=concurrency,
            request_timeout_s=args.request_timeout_s,
        )
    finally:
        npu_resources = monitor.stop()

    summary, raw_rows = _aggregate_repeat_metrics(
        samples,
        outputs,
        wall_clock_s,
        lang=args.lang,
        model_name=args.model,
        concurrency=concurrency,
    )
    if not npu_resources.get("available") or npu_resources.get("error"):
        summary["valid"] = False
        summary["invalid_reasons"].append(
            f"NPU monitor failure: {npu_resources.get('error') or 'unavailable'}"
        )
    summary["invalid_reasons"] = tuple(summary["invalid_reasons"])
    raw_path = Path(args.save_raw_dir) / f"conc{concurrency}-repeat{repeat}.jsonl"
    _write_jsonl(raw_path, raw_rows)
    return Exact10sRepeatResult(
        concurrency=concurrency,
        repeat=repeat,
        wall_clock_s=wall_clock_s,
        npu_resources=npu_resources,
        raw_jsonl=str(raw_path.resolve()),
        **summary,
    )


def _aggregate_repeats(
    concurrency: int, results: Sequence[Exact10sRepeatResult]
) -> dict[str, Any]:
    metric_names = (
        "corpus_wer",
        "throughput_samples_per_s",
        "rtfx",
        "latency_mean_s",
        "latency_p50_s",
        "latency_median_s",
        "latency_p90_s",
        "latency_p95_s",
        "latency_p99_s",
        "latency_max_s",
        "rtf_mean",
        "rtf_p95",
    )
    metrics = {}
    for name in metric_names:
        values = [
            getattr(result, name)
            for result in results
            if getattr(result, name) is not None
        ]
        metrics[name] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": statistics.fmean(values) if values else None,
        }
    return {
        "concurrency": concurrency,
        "repeats": len(results),
        "valid": all(result.valid for result in results),
        "failed_total": sum(result.failed_count for result in results),
        "timeout_total": sum(result.timeout_count for result in results),
        "missing_result_total": sum(result.missing_result_count for result in results),
        "duplicate_result_total": sum(
            result.duplicate_result_count for result in results
        ),
        "unexpected_result_total": sum(
            result.unexpected_result_count for result in results
        ),
        "metrics": metrics,
        "per_repeat": [asdict(result) for result in results],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True, help="Strict exact10 JSONL manifest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--lang", default="en")
    parser.add_argument(
        "--concurrencies", type=_parse_concurrencies, default=DEFAULT_CONCURRENCIES
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--min-distinct-audio", type=int, default=1)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--npu-id", type=int, default=0)
    parser.add_argument("--npu-chip-id", type=int, default=0)
    parser.add_argument("--monitor-interval-s", type=float, default=1.0)
    parser.add_argument("--hard-gate", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-raw-dir", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--launch-command")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in 1..65535")
    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")
    if args.warmup_samples < 0:
        raise ValueError("warmup-samples must be >= 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be > 0")
    if args.min_distinct_audio <= 0:
        raise ValueError("min-distinct-audio must be > 0")
    if args.request_timeout_s <= 0 or args.monitor_interval_s <= 0:
        raise ValueError("timeouts and monitor intervals must be > 0")
    if args.npu_id < 0 or args.npu_chip_id < 0:
        raise ValueError("NPU identifiers must be >= 0")
    if args.hard_gate:
        if args.repeats != 1:
            raise ValueError(
                "hard-gate repeats must be 1; restart the server for each "
                "process repeat"
            )
        if args.warmup_samples < HARD_GATE_WARMUP_SAMPLES:
            raise ValueError(
                f"hard gate requires at least {HARD_GATE_WARMUP_SAMPLES} warmup samples"
            )
        if args.max_samples != HARD_GATE_MEASURED_SAMPLES:
            raise ValueError(
                "hard gate requires exactly "
                f"{HARD_GATE_MEASURED_SAMPLES} measured samples"
            )
        if args.min_distinct_audio < (
            HARD_GATE_WARMUP_SAMPLES + HARD_GATE_MEASURED_SAMPLES
        ):
            raise ValueError(
                "hard gate requires at least 770 distinct total audio inputs"
            )


async def main_async(args: argparse.Namespace) -> int:
    _validate_args(args)
    samples = load_exact10s_manifest(args.meta)
    manifest_info = fingerprint_manifest(
        samples,
        min_distinct_count=args.min_distinct_audio,
    )
    warmup, measured = _split_manifest(
        manifest_info.samples,
        warmup_count=args.warmup_samples,
        measured_count=args.max_samples,
    )
    if args.hard_gate and len(measured) != HARD_GATE_MEASURED_SAMPLES:
        raise ValueError(
            "hard gate measured partition must contain exactly 700 samples"
        )
    languages = {sample.language for sample in measured}
    if languages != {args.lang}:
        raise ValueError(
            f"measured manifest languages {sorted(languages)} do not match "
            f"--lang {args.lang}"
        )

    aggregates = []
    for concurrency in args.concurrencies:
        repeats = []
        for repeat in range(1, args.repeats + 1):
            repeats.append(
                await _run_one_repeat(
                    args,
                    measured,
                    concurrency,
                    repeat,
                    warmup_samples=warmup,
                )
            )
        aggregates.append(_aggregate_repeats(concurrency, repeats))

    payload = {
        "schema_version": 1,
        "benchmark": "qwen3-asr-exact10s",
        "valid": all(aggregate["valid"] for aggregate in aggregates),
        "manifest": {
            "sha256": manifest_info.sha256,
            "total_count": manifest_info.total_count,
            "distinct_audio_count": manifest_info.distinct_audio_count,
            "warmup_count": len(warmup),
            "measured_count": len(measured),
            "measured_sha256": fingerprint_manifest(measured).sha256,
        },
        "concurrencies": list(args.concurrencies),
        "aggregates": aggregates,
        "npu_environment": collect_npu_environment_fingerprint([args.npu_id]),
        "provenance": collect_benchmark_provenance(
            model_id=args.model,
            model_revision=args.model_revision,
            dataset_id=str(Path(args.meta).resolve()),
            dataset_revision=args.dataset_revision,
            launch_command=args.launch_command,
            server_config={
                "host": args.host,
                "port": args.port,
                "hard_gate": args.hard_gate,
            },
            evaluation_input_sha256=fingerprint_manifest(measured).sha256,
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if payload["valid"] else 1


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
