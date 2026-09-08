# SPDX-License-Identifier: Apache-2.0
"""Build a deterministic exact-10-second corpus from a pinned SeedTTS snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.manifest.exact10s import (
    CHANNELS,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    TARGET_FRAMES,
    Exact10sSample,
    fingerprint_manifest,
    validate_clip_duration_with_ref,
)

DEFAULT_TOTAL_CLIPS = 770
DEFAULT_WARMUP_CLIPS = 70
DEFAULT_SILENCE_MS = 100
MIN_SPEECH_FRACTION = 0.80
SUPPORTED_SOURCE_SAMPLE_RATES = (SAMPLE_RATE, 24_000)
FFMPEG_TIMEOUT_S = 120
FFMPEG_RESAMPLE_FILTER = (
    "aresample=16000:filter_size=32:phase_shift=10:linear_interp=0:"
    "exact_rational=1:dither_method=none"
)


@dataclass(frozen=True)
class _SourcePcm:
    sample_id: str
    ref_text: str
    pcm: bytes
    num_frames: int
    source_sample_rate: int
    resampled: bool
    pcm_sha256: str


class _InsufficientSpeechOccupancy(ValueError):
    pass


class _NoMoreDistinctCompositions(ValueError):
    pass


class _FfmpegResampler:
    def __init__(self, executable: str = "ffmpeg") -> None:
        self.requested_executable = executable
        self.executable: str | None = None
        self._identity: dict | None = None

    def _resolve(self) -> str:
        if self.executable is None:
            self.executable = shutil.which(self.requested_executable)
            if self.executable is None:
                raise RuntimeError(
                    "ffmpeg is required to resample pinned 24 kHz SeedTTS audio"
                )
        return self.executable

    def identity(self) -> dict:
        if self._identity is None:
            executable = self._resolve()
            result = subprocess.run(
                [executable, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=FFMPEG_TIMEOUT_S,
            )
            if result.returncode != 0 or not result.stdout:
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg -version failed: {stderr}")
            version_text = result.stdout.decode("utf-8", errors="replace")
            self._identity = {
                "backend": "ffmpeg-swresample",
                "resolved_executable": executable,
                "version_first_line": version_text.splitlines()[0],
                "version_output_sha256": hashlib.sha256(result.stdout).hexdigest(),
                "command_template": self.command_template(),
            }
        return dict(self._identity)

    def command_template(self) -> list[str]:
        return [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "<input-wav>",
            "-map_metadata",
            "-1",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-af",
            FFMPEG_RESAMPLE_FILTER,
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ]

    def resample(self, wav_path: str, sample_id: str) -> bytes:
        # Query and freeze the backend identity before accepting any output.
        self.identity()
        command = self.command_template()
        command[command.index("<input-wav>")] = wav_path
        command[0] = self._resolve()
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=FFMPEG_TIMEOUT_S,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg resampling failed for {sample_id}: {stderr}"
            )
        if not result.stdout or len(result.stdout) % SAMPLE_WIDTH_BYTES:
            raise RuntimeError(
                f"ffmpeg produced invalid PCM16 output for {sample_id}"
            )
        return result.stdout


def _read_source(
    sample: SampleInput,
    *,
    resampler: _FfmpegResampler,
) -> _SourcePcm:
    try:
        with wave.open(sample.ref_audio, "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("source must use uncompressed PCM")
            if wav_file.getnchannels() != CHANNELS:
                raise ValueError("source must be mono")
            source_sample_rate = wav_file.getframerate()
            if source_sample_rate not in SUPPORTED_SOURCE_SAMPLE_RATES:
                supported = ", ".join(
                    str(rate) for rate in SUPPORTED_SOURCE_SAMPLE_RATES
                )
                raise ValueError(f"source sample rate must be one of: {supported}")
            if wav_file.getsampwidth() != SAMPLE_WIDTH_BYTES:
                raise ValueError("source must be PCM16")
            num_frames = wav_file.getnframes()
            pcm = wav_file.readframes(num_frames)
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"invalid source WAV for {sample.sample_id}") from exc
    expected_bytes = num_frames * CHANNELS * SAMPLE_WIDTH_BYTES
    if len(pcm) != expected_bytes:
        raise ValueError(f"truncated source WAV for {sample.sample_id}")
    if num_frames <= 0:
        raise ValueError(f"source {sample.sample_id} must not be empty")
    resampled = source_sample_rate != SAMPLE_RATE
    if resampled:
        pcm = resampler.resample(sample.ref_audio, sample.sample_id)
        num_frames = len(pcm) // (CHANNELS * SAMPLE_WIDTH_BYTES)
    ref_text = sample.ref_text.strip()
    if not ref_text:
        raise ValueError(f"empty source reference for {sample.sample_id}")
    return _SourcePcm(
        sample.sample_id,
        ref_text,
        pcm,
        num_frames,
        source_sample_rate,
        resampled,
        hashlib.sha256(pcm).hexdigest(),
    )


def _compose_clip(
    sources: Sequence[_SourcePcm],
    start_index: int,
    *,
    silence_frames: int,
    variant: int = 0,
) -> tuple[bytes, str, list[str], int]:
    anchor = sources[start_index]
    candidate_by_pcm_hash: dict[str, int] = {}
    for index, source in enumerate(sources):
        if index == start_index or source.pcm_sha256 == anchor.pcm_sha256:
            continue
        existing = candidate_by_pcm_hash.get(source.pcm_sha256)
        if existing is None or (index - start_index) % len(sources) < (
            existing - start_index
        ) % len(sources):
            candidate_by_pcm_hash[source.pcm_sha256] = index
    candidate_indices = list(candidate_by_pcm_hash.values())
    plans: list[tuple[int, ...]] = [(start_index,)]

    single_capacity = TARGET_FRAMES - anchor.num_frames - silence_frames
    fitting_singles = [
        index
        for index in candidate_indices
        if sources[index].num_frames <= single_capacity
    ]
    plans.extend((start_index, index) for index in fitting_singles)

    pair_capacity = TARGET_FRAMES - anchor.num_frames - 2 * silence_frames
    if pair_capacity >= 0 and len(candidate_indices) >= 2:
        by_frames = sorted(
            candidate_indices,
            key=lambda index: (sources[index].num_frames, sources[index].sample_id),
        )
        left = 0
        right = len(by_frames) - 1
        best_pair: tuple[int, int] | None = None
        best_pair_frames = -1
        while left < right:
            left_index = by_frames[left]
            right_index = by_frames[right]
            pair_frames = (
                sources[left_index].num_frames + sources[right_index].num_frames
            )
            if pair_frames > pair_capacity:
                right -= 1
                continue
            pair = tuple(
                sorted(
                    (left_index, right_index),
                    key=lambda index: (index - start_index) % len(sources),
                )
            )
            if pair_frames > best_pair_frames or (
                pair_frames == best_pair_frames and pair < (best_pair or pair)
            ):
                best_pair = pair
                best_pair_frames = pair_frames
            left += 1
        if best_pair is not None:
            plans.append((start_index, *best_pair))

    qualifying_plans = [
        plan
        for plan in plans
        if sum(sources[index].num_frames for index in plan) / TARGET_FRAMES
        >= MIN_SPEECH_FRACTION
    ]
    qualifying_plans.sort(
        key=lambda plan: (
            -sum(sources[index].num_frames for index in plan),
            len(plan),
            tuple((index - start_index) % len(sources) for index in plan[1:]),
        )
    )
    if not qualifying_plans:
        raise _InsufficientSpeechOccupancy(
            f"anchor {anchor.sample_id} cannot reach "
            f"{MIN_SPEECH_FRACTION:.1%} speech occupancy"
        )
    if variant >= len(qualifying_plans):
        raise _NoMoreDistinctCompositions(
            f"anchor {anchor.sample_id} exhausted "
            f"{len(qualifying_plans)} qualifying compositions"
        )
    selected_indices = qualifying_plans[variant]
    selected = [sources[index] for index in selected_indices]
    occupied = sum(source.num_frames for source in selected) + silence_frames * (
        len(selected) - 1
    )

    speech_frames = sum(source.num_frames for source in selected)
    silence = b"\0" * silence_frames * SAMPLE_WIDTH_BYTES
    chunks: list[bytes] = []
    for index, source in enumerate(selected):
        if index:
            chunks.append(silence)
        chunks.append(source.pcm)
    chunks.append(b"\0" * (TARGET_FRAMES - occupied) * SAMPLE_WIDTH_BYTES)
    pcm = b"".join(chunks)
    if len(pcm) != TARGET_FRAMES * SAMPLE_WIDTH_BYTES:
        raise AssertionError("derived PCM length does not match exact10 contract")
    return (
        pcm,
        " ".join(source.ref_text for source in selected),
        [source.sample_id for source in selected],
        speech_frames,
    )


def _write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm)


def _hash_files(root: Path, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_digest = hashlib.sha256()
        file_size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                file_digest.update(chunk)
                file_size += len(chunk)
        digest.update(file_size.to_bytes(8, "big"))
        digest.update(file_digest.digest())
    return digest.hexdigest()


def build_exact10s_corpus(
    source_samples: Sequence[SampleInput],
    output_root: str | os.PathLike[str],
    *,
    total_clips: int = DEFAULT_TOTAL_CLIPS,
    warmup_clips: int = DEFAULT_WARMUP_CLIPS,
    silence_ms: int = DEFAULT_SILENCE_MS,
    source_identity: dict | None = None,
    ffmpeg_executable: str = "ffmpeg",
) -> dict:
    if total_clips <= 0:
        raise ValueError("total_clips must be > 0")
    if not 0 < warmup_clips < total_clips:
        raise ValueError("warmup_clips must be between zero and total_clips")
    if silence_ms < 0:
        raise ValueError("silence_ms must be >= 0")
    if len(source_samples) < total_clips:
        raise ValueError(
            f"need at least {total_clips} source samples, got {len(source_samples)}"
        )

    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        ids = [sample.sample_id for sample in source_samples]
        if len(ids) != len(set(ids)):
            raise ValueError("source sample IDs must be unique")
        resampler = _FfmpegResampler(ffmpeg_executable)
        all_sources = sorted(
            (
                _read_source(sample, resampler=resampler)
                for sample in source_samples
            ),
            key=lambda item: item.sample_id,
        )
        sources = [
            source for source in all_sources if source.num_frames <= TARGET_FRAMES
        ]
        excluded_too_long = [
            source.sample_id
            for source in all_sources
            if source.num_frames > TARGET_FRAMES
        ]
        if len(sources) < total_clips:
            raise ValueError(
                f"need at least {total_clips} usable <=10-second sources, "
                f"found {len(sources)}"
            )
        silence_frames = round(SAMPLE_RATE * silence_ms / 1000)
        records = []
        exact_samples: list[Exact10sSample] = []
        source_membership = []
        skipped_low_occupancy_anchor_ids = []
        skipped_duplicate_derived_anchor_ids = []
        derived_pcm_hashes: set[str] = set()
        for anchor_index, anchor in enumerate(sources):
            if len(records) == total_clips:
                break
            variant = 0
            pcm_hash: str | None = None
            while True:
                try:
                    pcm, ref_text, source_ids, speech_frames = _compose_clip(
                        sources,
                        anchor_index,
                        silence_frames=silence_frames,
                        variant=variant,
                    )
                except _InsufficientSpeechOccupancy:
                    skipped_low_occupancy_anchor_ids.append(anchor.sample_id)
                    pcm_hash = None
                    break
                except _NoMoreDistinctCompositions:
                    skipped_duplicate_derived_anchor_ids.append(anchor.sample_id)
                    pcm_hash = None
                    break
                pcm_hash = hashlib.sha256(pcm).hexdigest()
                if pcm_hash not in derived_pcm_hashes:
                    break
                variant += 1
            if pcm_hash is None:
                continue
            derived_pcm_hashes.add(pcm_hash)
            output_index = len(records)
            relative_path = Path("audio") / f"exact10-{output_index:04d}.wav"
            wav_path = temp_root / relative_path
            _write_wav(wav_path, pcm)
            sample_id = f"seedtts-exact10-{output_index:04d}"
            exact_sample = validate_clip_duration_with_ref(
                wav_path,
                ref_text,
                sample_id,
                language="en",
            )
            exact_samples.append(exact_sample)
            records.append(
                {
                    "sample_id": sample_id,
                    "wav_path": relative_path.as_posix(),
                    "ref_text": ref_text,
                    "language": "en",
                }
            )
            source_membership.append(
                {
                    "sample_id": sample_id,
                    "source_sample_ids": source_ids,
                    "speech_frames": speech_frames,
                }
            )

        if len(records) != total_clips:
            raise ValueError(
                f"could build only {len(records)} distinct exact10 clips from "
                f"{len(sources)} usable sources; speech occupancy failures="
                f"{len(skipped_low_occupancy_anchor_ids)}, duplicate outputs="
                f"{len(skipped_duplicate_derived_anchor_ids)}"
            )

        info = fingerprint_manifest(
            exact_samples,
            min_distinct_count=total_clips,
        )
        manifest_path = temp_root / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        source_sample_rate_counts = Counter(
            source.source_sample_rate for source in all_sources
        )
        distinct_source_audio_count = len(
            {source.pcm_sha256 for source in all_sources}
        )
        resampled_source_count = sum(source.resampled for source in all_sources)
        provenance = {
            "schema_version": 2,
            "generator": "benchmarks.manifest.prepare_seedtts_exact10s",
            "source": source_identity or {},
            "source_sample_count": len(source_samples),
            "distinct_source_audio_count": distinct_source_audio_count,
            "source_sample_rate_counts": {
                str(rate): count
                for rate, count in sorted(source_sample_rate_counts.items())
            },
            "resampled_source_count": resampled_source_count,
            "resampler": (
                resampler.identity() if resampled_source_count else None
            ),
            "usable_source_count": len(sources),
            "excluded_too_long_source_ids": excluded_too_long,
            "skipped_low_occupancy_anchor_ids": (
                skipped_low_occupancy_anchor_ids
            ),
            "skipped_duplicate_derived_anchor_ids": (
                skipped_duplicate_derived_anchor_ids
            ),
            "total_count": info.total_count,
            "warmup_count": warmup_clips,
            "measured_count": total_clips - warmup_clips,
            "silence_ms_between_utterances": silence_ms,
            "minimum_speech_fraction": MIN_SPEECH_FRACTION,
            "manifest_sha256": info.sha256,
            "duration_min_s": info.duration_min_s,
            "duration_max_s": info.duration_max_s,
            "distinct_audio_count": info.distinct_audio_count,
            "source_membership": source_membership,
        }
        (temp_root / "provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temp_root.replace(destination)
        return provenance
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="en", choices=("en",))
    parser.add_argument("--revision", default=SEEDTTS_DATASET_REVISION)
    parser.add_argument("--total-clips", type=int, default=DEFAULT_TOTAL_CLIPS)
    parser.add_argument("--warmup-clips", type=int, default=DEFAULT_WARMUP_CLIPS)
    parser.add_argument("--silence-ms", type=int, default=DEFAULT_SILENCE_MS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.revision != SEEDTTS_DATASET_REVISION:
        raise ValueError(
            f"revision must remain pinned to {SEEDTTS_DATASET_REVISION}"
        )
    snapshot_root = Path(args.snapshot_root).resolve(strict=True)
    parquet_files = sorted((snapshot_root / "data").glob(f"{args.split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("pinned snapshot contains no English Parquet")
    source_samples = load_seedtts_samples(
        str(snapshot_root),
        split=args.split,
    )
    source_identity = {
        "dataset_id": SEEDTTS_DATASET_ID,
        "revision": args.revision,
        "split": args.split,
        "parquet_files": [path.name for path in parquet_files],
        "parquet_set_sha256": _hash_files(snapshot_root, parquet_files),
    }
    provenance = build_exact10s_corpus(
        source_samples,
        args.output_root,
        total_clips=args.total_clips,
        warmup_clips=args.warmup_clips,
        silence_ms=args.silence_ms,
        source_identity=source_identity,
    )
    print(json.dumps(provenance, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
