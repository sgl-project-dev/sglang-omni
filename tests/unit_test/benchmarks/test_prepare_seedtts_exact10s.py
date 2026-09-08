# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.manifest.exact10s import (
    TARGET_FRAMES,
    fingerprint_manifest,
    load_exact10s_manifest,
)
from benchmarks.manifest.prepare_seedtts_exact10s import (
    FFMPEG_RESAMPLE_FILTER,
    build_exact10s_corpus,
)


def _source(
    tmp_path: Path,
    sample_id: str,
    frames: int,
    fill: int,
    *,
    sample_rate: int = 16_000,
) -> SampleInput:
    path = tmp_path / f"{sample_id}.wav"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes([fill]) * frames * 2)
    return SampleInput(
        sample_id=sample_id,
        ref_text=f"text {sample_id}",
        ref_audio=str(path),
        target_text="unused",
    )


def test_builder_creates_exact_distinct_manifest_and_provenance(
    tmp_path: Path,
) -> None:
    sources = [
        _source(tmp_path, f"s{index}", 70_000, index + 1)
        for index in range(3)
    ]
    output = tmp_path / "derived"
    provenance = build_exact10s_corpus(
        sources,
        output,
        total_clips=2,
        warmup_clips=1,
    )
    samples = load_exact10s_manifest(output / "manifest.jsonl")
    info = fingerprint_manifest(samples, min_distinct_count=2)

    assert len(samples) == 2
    assert all(sample.num_samples == TARGET_FRAMES for sample in samples)
    assert info.sha256 == provenance["manifest_sha256"]
    assert provenance["warmup_count"] == 1
    assert provenance["measured_count"] == 1
    assert provenance["minimum_speech_fraction"] == 0.8
    stored = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert stored["source_membership"][0]["source_sample_ids"] == ["s0", "s1"]


def test_builder_is_deterministic_across_output_directories(tmp_path: Path) -> None:
    sources = [
        _source(tmp_path, f"s{index}", 70_000, index + 1)
        for index in range(3)
    ]
    first = build_exact10s_corpus(
        list(reversed(sources)),
        tmp_path / "first",
        total_clips=2,
        warmup_clips=1,
    )
    second = build_exact10s_corpus(
        sources,
        tmp_path / "second",
        total_clips=2,
        warmup_clips=1,
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_builder_refuses_existing_output(tmp_path: Path) -> None:
    sources = [
        _source(tmp_path, f"s{index}", 70_000, index + 1)
        for index in range(3)
    ]
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_exact10s_corpus(
            sources,
            output,
            total_clips=2,
            warmup_clips=1,
        )


def test_builder_rejects_low_speech_occupancy(tmp_path: Path) -> None:
    sources = [
        _source(tmp_path, f"s{index}", 10_000, index + 1)
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="speech occupancy"):
        build_exact10s_corpus(
            sources,
            tmp_path / "derived",
            total_clips=2,
            warmup_clips=1,
        )


def test_builder_excludes_overlong_sources_before_count_gate(
    tmp_path: Path,
) -> None:
    sources = [
        _source(tmp_path, "short", 140_000, 1),
        _source(tmp_path, "long", TARGET_FRAMES + 1, 2),
    ]
    with pytest.raises(ValueError, match="usable"):
        build_exact10s_corpus(
            sources,
            tmp_path / "derived",
            total_clips=2,
            warmup_clips=1,
        )


def test_builder_resamples_24khz_sources_with_pinned_ffmpeg_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> SimpleNamespace:
        calls.append(command)
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.PIPE
        if command[-1] == "-version":
            return SimpleNamespace(
                returncode=0,
                stdout=b"ffmpeg version test-build\nconfiguration test\n",
                stderr=b"",
            )
        source_name = Path(command[command.index("-i") + 1]).stem
        fill = 1 if source_name == "s0" else 2
        return SimpleNamespace(
            returncode=0,
            stdout=bytes([fill]) * 70_000 * 2,
            stderr=b"",
        )

    monkeypatch.setattr(
        "benchmarks.manifest.prepare_seedtts_exact10s.shutil.which",
        lambda executable: "/usr/bin/ffmpeg",
    )
    monkeypatch.setattr(
        "benchmarks.manifest.prepare_seedtts_exact10s.subprocess.run",
        fake_run,
    )
    sources = [
        _source(
            tmp_path,
            f"s{index}",
            105_000,
            index + 1,
            sample_rate=24_000,
        )
        for index in range(2)
    ]

    provenance = build_exact10s_corpus(
        sources,
        tmp_path / "derived",
        total_clips=2,
        warmup_clips=1,
    )

    assert provenance["source_sample_rate_counts"] == {"24000": 2}
    assert provenance["resampled_source_count"] == 2
    assert provenance["resampler"]["backend"] == "ffmpeg-swresample"
    assert provenance["resampler"]["version_first_line"] == (
        "ffmpeg version test-build"
    )
    assert len(provenance["resampler"]["version_output_sha256"]) == 64
    assert provenance["resampler"]["command_template"][-1] == "pipe:1"
    resample_calls = [command for command in calls if "-i" in command]
    assert len(resample_calls) == 2
    assert all(
        command[command.index("-af") + 1] == FFMPEG_RESAMPLE_FILTER
        for command in resample_calls
    )


def test_builder_rejects_unapproved_source_sample_rate(tmp_path: Path) -> None:
    sources = [
        _source(
            tmp_path,
            f"s{index}",
            70_000,
            index + 1,
            sample_rate=22_050,
        )
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="sample rate must be one of"):
        build_exact10s_corpus(
            sources,
            tmp_path / "derived",
            total_clips=2,
            warmup_clips=1,
        )


def test_builder_best_fit_avoids_cyclic_greedy_occupancy_failure(
    tmp_path: Path,
) -> None:
    sources = [
        _source(tmp_path, "s0", 125_000, 1),
        _source(tmp_path, "s1", 40_000, 2),
        _source(tmp_path, "s2", 10_000, 3),
        _source(tmp_path, "s3", 11_000, 4),
    ]

    provenance = build_exact10s_corpus(
        sources,
        tmp_path / "derived",
        total_clips=2,
        warmup_clips=1,
    )

    assert provenance["total_count"] == 2
    assert provenance["source_membership"][0]["source_sample_ids"] == [
        "s0",
        "s2",
        "s3",
    ]
    assert provenance["source_membership"][0]["speech_frames"] == 146_000


def test_builder_uses_alternate_partner_for_duplicate_source_audio(
    tmp_path: Path,
) -> None:
    sources = [
        _source(tmp_path, "a0", 70_000, 1),
        _source(tmp_path, "a1", 70_000, 1),
        _source(tmp_path, "z0", 70_000, 2),
        _source(tmp_path, "z1", 65_000, 3),
    ]

    provenance = build_exact10s_corpus(
        sources,
        tmp_path / "derived",
        total_clips=2,
        warmup_clips=1,
    )

    assert provenance["distinct_source_audio_count"] == 3
    assert provenance["source_membership"][0]["source_sample_ids"] == [
        "a0",
        "z0",
    ]
    assert provenance["source_membership"][1]["source_sample_ids"] == [
        "a1",
        "z1",
    ]
