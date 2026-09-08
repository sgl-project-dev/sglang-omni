from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from benchmarks.manifest.exact10s import (
    TARGET_FRAMES,
    Exact10sSample,
    exact10s_input_sha256,
    fingerprint_manifest,
    load_exact10s_from_directory,
    load_exact10s_manifest,
    select_unique_audio_samples,
    validate_clip_duration_with_ref,
)


def _write_wav(
    path: Path,
    *,
    frames: int = TARGET_FRAMES,
    channels: int = 1,
    sample_rate: int = 16_000,
    sample_width: int = 2,
    fill: int = 1,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes([fill]) * frames * channels * sample_width)
    return path


@pytest.mark.parametrize(
    "frames", [TARGET_FRAMES - 1, TARGET_FRAMES, TARGET_FRAMES + 1]
)
def test_validate_accepts_one_frame_tolerance(tmp_path: Path, frames: int) -> None:
    sample = validate_clip_duration_with_ref(
        _write_wav(tmp_path / "ok.wav", frames=frames), "hello", "sample"
    )
    assert sample.num_samples == frames
    assert sample.duration_s == pytest.approx(frames / 16_000)
    assert sample.ref_audio == sample.wav_path
    assert sample.target_text == "hello"


@pytest.mark.parametrize("frames", [TARGET_FRAMES - 2, TARGET_FRAMES + 2])
def test_validate_rejects_duration_outside_tolerance(
    tmp_path: Path, frames: int
) -> None:
    with pytest.raises(ValueError, match="must contain"):
        validate_clip_duration_with_ref(
            _write_wav(tmp_path / "bad.wav", frames=frames), "hello"
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"channels": 2}, "mono"),
        ({"sample_rate": 8_000}, "16000 Hz"),
        ({"sample_width": 1}, "PCM16"),
    ],
)
def test_validate_rejects_wrong_pcm_format(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_clip_duration_with_ref(
            _write_wav(tmp_path / "bad.wav", **kwargs), "hello"
        )


def test_validate_rejects_non_wav(tmp_path: Path) -> None:
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not a wav")
    with pytest.raises(ValueError, match="RIFF/WAVE"):
        validate_clip_duration_with_ref(path, "hello")


def test_manifest_fingerprint_is_full_and_tracks_pcm_and_text(tmp_path: Path) -> None:
    first = validate_clip_duration_with_ref(
        _write_wav(tmp_path / "one.wav", fill=1), "one", "one"
    )
    same = exact10s_input_sha256([first])
    assert len(same) == 64
    assert same == exact10s_input_sha256([first])

    changed_pcm = validate_clip_duration_with_ref(
        _write_wav(tmp_path / "two.wav", fill=2), "one", "one"
    )
    changed_text = Exact10sSample(**{**first.__dict__, "ref_text": "two"})
    assert exact10s_input_sha256([changed_pcm]) != same
    assert exact10s_input_sha256([changed_text]) != same


def test_manifest_reports_and_can_reject_duplicate_audio(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "one.wav")
    first = validate_clip_duration_with_ref(path, "one", "one")
    second = validate_clip_duration_with_ref(path, "two", "two")

    info = fingerprint_manifest([first, second], validate_unique=False)
    assert info.distinct_audio is False
    assert info.distinct_audio_count == 1
    with pytest.raises(ValueError, match="duplicate audio"):
        fingerprint_manifest([first, second])


def test_manifest_enforces_minimum_distinct_count(tmp_path: Path) -> None:
    sample = validate_clip_duration_with_ref(
        _write_wav(tmp_path / "one.wav"), "one", "one"
    )
    with pytest.raises(ValueError, match="at least 2"):
        fingerprint_manifest([sample], min_distinct_count=2)


def test_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    first = validate_clip_duration_with_ref(
        _write_wav(tmp_path / "one.wav", fill=1), "one", "same"
    )
    second = validate_clip_duration_with_ref(
        _write_wav(tmp_path / "two.wav", fill=2), "two", "same"
    )
    with pytest.raises(ValueError, match="sample_id"):
        fingerprint_manifest([first, second])


def test_select_unique_audio_requires_requested_count(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "one.wav")
    samples = [
        validate_clip_duration_with_ref(path, "one", "one"),
        validate_clip_duration_with_ref(path, "two", "two"),
    ]
    assert [sample.sample_id for sample in select_unique_audio_samples(samples)] == [
        "one"
    ]
    with pytest.raises(ValueError, match="requested 2"):
        select_unique_audio_samples(samples, 2)


def test_jsonl_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    _write_wav(tmp_path / "audio" / "one.wav")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "one",
                "wav_path": "audio/one.wav",
                "ref_text": "hello",
                "language": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_exact10s_manifest(manifest)
    assert samples[0].sample_id == "one"
    assert Path(samples[0].wav_path).is_absolute()


def test_jsonl_manifest_rejects_relative_escape(tmp_path: Path) -> None:
    _write_wav(tmp_path.parent / "escape.wav")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"wav_path": "../escape.wav", "ref_text": "hello"}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        load_exact10s_manifest(manifest)


def test_directory_loader_fails_closed_on_missing_reference(tmp_path: Path) -> None:
    _write_wav(tmp_path / "one.wav")
    with pytest.raises(ValueError, match="missing reference"):
        load_exact10s_from_directory(tmp_path, {})


def test_directory_loader_uses_relative_reference_key(tmp_path: Path) -> None:
    _write_wav(tmp_path / "nested" / "one.wav")
    samples = load_exact10s_from_directory(
        tmp_path, {"nested/one.wav": "hello"}
    )
    assert samples[0].sample_id == "nested/one.wav"
