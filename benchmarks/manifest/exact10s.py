# SPDX-License-Identifier: Apache-2.0
"""Strict exact-10-second PCM WAV manifests for ASR performance gates."""

from __future__ import annotations

import hashlib
import json
import os
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
TARGET_FRAMES = 160_000
FRAME_TOLERANCE = 1
_HASH_CHUNK_FRAMES = 64 * 1024


@dataclass(frozen=True)
class Exact10sSample:
    sample_id: str
    wav_path: str
    duration_s: float
    num_samples: int
    ref_text: str
    language: str = "en"

    @property
    def ref_audio(self) -> str:
        """Compatibility with the shared ASR benchmark sample protocol."""
        return self.wav_path

    @property
    def target_text(self) -> str:
        return self.ref_text


@dataclass(frozen=True)
class ManifestInfo:
    samples: tuple[Exact10sSample, ...]
    total_count: int
    sha256: str
    duration_min_s: float
    duration_max_s: float
    distinct_audio: bool
    distinct_audio_count: int
    language_counts: dict[str, int]


@dataclass(frozen=True)
class _WavInfo:
    channels: int
    sample_rate: int
    sample_width_bytes: int
    num_frames: int
    compression_type: str


def _read_wav_header(path: str | os.PathLike[str]) -> _WavInfo:
    """Read and validate the effective WAV stream metadata.

    ``wave`` walks RIFF chunks instead of assuming a 44-byte header. Reading
    all frames also detects truncated payloads whose declared frame count looks
    valid.
    """
    try:
        with wave.open(os.fspath(path), "rb") as wav_file:
            info = _WavInfo(
                channels=wav_file.getnchannels(),
                sample_rate=wav_file.getframerate(),
                sample_width_bytes=wav_file.getsampwidth(),
                num_frames=wav_file.getnframes(),
                compression_type=wav_file.getcomptype(),
            )
            payload = wav_file.readframes(info.num_frames)
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"invalid RIFF/WAVE file: {path}") from exc

    if info.compression_type != "NONE":
        raise ValueError(f"WAV must use uncompressed PCM: {path}")
    if info.channels != CHANNELS:
        raise ValueError(f"WAV must be mono, got {info.channels} channels: {path}")
    if info.sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"WAV must be {SAMPLE_RATE} Hz, got {info.sample_rate}: {path}"
        )
    if info.sample_width_bytes != SAMPLE_WIDTH_BYTES:
        raise ValueError(
            "WAV must be PCM16, got "
            f"{info.sample_width_bytes * 8}-bit samples: {path}"
        )
    expected_bytes = info.num_frames * info.channels * info.sample_width_bytes
    if len(payload) != expected_bytes:
        raise ValueError(
            f"truncated WAV payload: expected {expected_bytes} bytes, "
            f"read {len(payload)}: {path}"
        )
    return info


def clip_duration_s_from_header(
    path: str | os.PathLike[str],
) -> tuple[float, int]:
    info = _read_wav_header(path)
    return info.num_frames / info.sample_rate, info.num_frames


def validate_clip_duration_with_ref(
    path: str | os.PathLike[str],
    ref_text: str,
    sample_id: str | None = None,
    *,
    language: str = "en",
) -> Exact10sSample:
    resolved = Path(path).resolve(strict=True)
    duration_s, num_frames = clip_duration_s_from_header(resolved)
    if abs(num_frames - TARGET_FRAMES) > FRAME_TOLERANCE:
        raise ValueError(
            f"WAV must contain {TARGET_FRAMES} +/- {FRAME_TOLERANCE} frames, "
            f"got {num_frames}: {resolved}"
        )
    normalized_id = (sample_id or resolved.stem).strip()
    if not normalized_id:
        raise ValueError("sample_id must be non-empty")
    if not isinstance(ref_text, str) or not ref_text.strip():
        raise ValueError(f"reference text must be non-empty: {normalized_id}")
    if not isinstance(language, str) or not language.strip():
        raise ValueError(f"language must be non-empty: {normalized_id}")
    return Exact10sSample(
        sample_id=normalized_id,
        wav_path=str(resolved),
        duration_s=duration_s,
        num_samples=num_frames,
        ref_text=ref_text,
        language=language.strip(),
    )


def validate_clip_duration(path: str | os.PathLike[str]) -> Exact10sSample:
    """Validate audio only; use a non-empty placeholder reference."""
    return validate_clip_duration_with_ref(path, "<unscored>")


def _audio_bytes_hash(path: str | os.PathLike[str]) -> str:
    """Hash decoded PCM frame bytes while validating the complete WAV."""
    _read_wav_header(path)
    digest = hashlib.sha256()
    with wave.open(os.fspath(path), "rb") as wav_file:
        remaining = wav_file.getnframes()
        total = 0
        while remaining:
            chunk = wav_file.readframes(min(remaining, _HASH_CHUNK_FRAMES))
            if not chunk:
                break
            digest.update(chunk)
            frames = len(chunk) // (wav_file.getnchannels() * wav_file.getsampwidth())
            total += frames
            remaining -= frames
        if total != wav_file.getnframes():
            raise ValueError(
                f"truncated WAV while hashing: expected {wav_file.getnframes()} "
                f"frames, read {total}: {path}"
            )
    return digest.hexdigest()


def _update_length_prefixed(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def fingerprint_manifest(
    samples: Iterable[Exact10sSample],
    namespace: str = "exact10s",
    validate_unique: bool = True,
    min_distinct_count: int | None = None,
) -> ManifestInfo:
    frozen = tuple(samples)
    if not frozen:
        raise ValueError("exact-10-second manifest must not be empty")
    if min_distinct_count is not None and min_distinct_count <= 0:
        raise ValueError("min_distinct_count must be greater than zero")

    ids = [sample.sample_id for sample in frozen]
    if len(set(ids)) != len(ids):
        raise ValueError("manifest sample_id values must be unique")

    digest = hashlib.sha256()
    _update_length_prefixed(digest, f"{namespace}-manifest-v1")
    checked_samples: list[Exact10sSample] = []
    audio_hashes: list[str] = []
    for sample in frozen:
        # Revalidate so callers cannot manufacture a dataclass that bypasses
        # the file-format and duration contract.
        checked = validate_clip_duration_with_ref(
            sample.wav_path,
            sample.ref_text,
            sample.sample_id,
            language=sample.language,
        )
        checked_samples.append(checked)
        audio_hash = _audio_bytes_hash(checked.wav_path)
        audio_hashes.append(audio_hash)
        for value in (
            checked.sample_id,
            checked.language,
            checked.ref_text,
            str(checked.num_samples),
            audio_hash,
        ):
            _update_length_prefixed(digest, value)

    distinct_count = len(set(audio_hashes))
    distinct = distinct_count == len(frozen)
    if validate_unique and not distinct:
        raise ValueError(
            "manifest contains duplicate audio: "
            f"{distinct_count}/{len(frozen)} distinct"
        )
    if min_distinct_count is not None and distinct_count < min_distinct_count:
        raise ValueError(
            f"manifest requires at least {min_distinct_count} distinct audio files, "
            f"found {distinct_count}"
        )

    durations = [sample.duration_s for sample in checked_samples]
    return ManifestInfo(
        samples=tuple(checked_samples),
        total_count=len(frozen),
        sha256=digest.hexdigest(),
        duration_min_s=min(durations),
        duration_max_s=max(durations),
        distinct_audio=distinct,
        distinct_audio_count=distinct_count,
        language_counts=dict(Counter(sample.language for sample in checked_samples)),
    )


def exact10s_input_sha256(samples: Iterable[Exact10sSample]) -> str:
    """Return the full 64-character aggregate SHA-256 hex digest."""
    return fingerprint_manifest(samples).sha256


def select_unique_audio_samples(
    samples: Iterable[Exact10sSample],
    max_samples: int | None = None,
) -> list[Exact10sSample]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be greater than zero")
    selected: list[Exact10sSample] = []
    seen: set[str] = set()
    for sample in samples:
        audio_hash = _audio_bytes_hash(sample.wav_path)
        if audio_hash in seen:
            continue
        seen.add(audio_hash)
        selected.append(sample)
        if max_samples is not None and len(selected) == max_samples:
            break
    if max_samples is not None and len(selected) < max_samples:
        raise ValueError(
            f"requested {max_samples} distinct audio files, found {len(selected)}"
        )
    return selected


def load_exact10s_manifest(path: str | os.PathLike[str]) -> list[Exact10sSample]:
    """Load a JSONL manifest with paths relative to the manifest directory."""
    manifest_path = Path(path).resolve(strict=True)
    base = manifest_path.parent
    samples: list[Exact10sSample] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
                wav_value = record["wav_path"]
                ref_text = record["ref_text"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"invalid manifest record at line {line_number}"
                ) from exc
            candidate = Path(wav_value)
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (base / candidate).resolve()
            )
            if not candidate.is_absolute():
                try:
                    resolved.relative_to(base)
                except ValueError as exc:
                    raise ValueError(
                        f"manifest path escapes its directory at line {line_number}"
                    ) from exc
            samples.append(
                validate_clip_duration_with_ref(
                    resolved,
                    ref_text,
                    record.get("sample_id"),
                    language=record.get("language", "en"),
                )
            )
    if not samples:
        raise ValueError("exact-10-second manifest contains no records")
    return samples


def load_exact10s_from_directory(
    dir_path: str | os.PathLike[str],
    ref_texts: Mapping[str, str],
    *,
    language: str = "en",
) -> list[Exact10sSample]:
    """Load all WAVs below a directory, failing on any missing/invalid row."""
    root = Path(dir_path).resolve(strict=True)
    wav_paths = sorted(root.rglob("*.wav"))
    if not wav_paths:
        raise ValueError(f"no WAV files found under {root}")
    samples: list[Exact10sSample] = []
    for wav_path in wav_paths:
        relative = wav_path.relative_to(root).as_posix()
        ref_text = ref_texts.get(relative)
        if ref_text is None:
            ref_text = ref_texts.get(wav_path.stem)
        if ref_text is None:
            raise ValueError(f"missing reference text for {relative}")
        samples.append(
            validate_clip_duration_with_ref(
                wav_path,
                ref_text,
                sample_id=relative,
                language=language,
            )
        )
    return samples
