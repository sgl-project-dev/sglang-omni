# SPDX-License-Identifier: Apache-2.0
"""Tests for video audio extraction."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
import pytest

from sglang_omni.preprocessing import video
from sglang_omni.serve.openai_errors import is_bad_request_error


def _write_video_with_audio(path: Path) -> None:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=10:duration=0.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=8000:duration=0.2",
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-y",
            str(path),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )


def test_extract_audio_from_path_decodes_resamples_and_downmixes(
    tmp_path: Path,
) -> None:
    media = tmp_path / "audio.mp4"
    _write_video_with_audio(media)

    audio = video._extract_audio_from_path(media, 16_000)

    assert audio is not None
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert 3_000 <= audio.size <= 4_500
    assert np.max(np.abs(audio)) > 0.01


def test_extract_audio_from_path_returns_none_without_audio(monkeypatch) -> None:
    class Container:
        def __init__(self) -> None:
            self.streams = [SimpleNamespace(type="video")]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(video.av, "open", lambda _path: Container())

    assert video._extract_audio_from_path(Path("silent.mp4"), 16_000) is None


@pytest.mark.parametrize(
    ("error", "bad_request"),
    [
        (video.av.error.InvalidDataError(1094995529, "broken stream"), True),
        (video.av.error.EOFError(541478725, "broken stream"), True),
        (RuntimeError("broken stream"), False),
        (MemoryError("broken stream"), False),
        (PermissionError("broken stream"), False),
    ],
)
@pytest.mark.parametrize("stage", ["open", "decode"])
def test_extract_audio_from_path_surfaces_decode_failure(
    monkeypatch, error, bad_request, stage
) -> None:
    class Container:
        def __init__(self) -> None:
            self.audio_stream = SimpleNamespace(type="audio", index=2)
            self.streams = [SimpleNamespace(type="video", index=0), self.audio_stream]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, stream):
            assert stream is self.audio_stream
            raise error

    def open_media(_path):
        if stage == "open":
            raise error
        return Container()

    monkeypatch.setattr(video.av, "open", open_media)

    with pytest.raises(video.VideoDecodeError, match="broken stream") as exc_info:
        video._extract_audio_from_path(Path("broken.mp4"), 16_000)

    assert exc_info.value.__cause__ is error
    assert is_bad_request_error(exc_info.value) is bad_request


def test_extract_audio_from_path_rejects_corrupt_media(tmp_path: Path) -> None:
    media = tmp_path / "corrupt.mp4"
    media.write_bytes(b"not an mp4 file")

    with pytest.raises(video.VideoDecodeError, match="Invalid media data") as exc_info:
        video._extract_audio_from_path(media, 16_000)

    assert isinstance(exc_info.value.__cause__, video.av.error.InvalidDataError)
    assert is_bad_request_error(exc_info.value)


def test_extract_audio_from_path_rejects_empty_audio_stream(monkeypatch) -> None:
    class Container:
        streams = [SimpleNamespace(type="audio")]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, _stream):
            return iter(())

    monkeypatch.setattr(video.av, "open", lambda _path: Container())

    with pytest.raises(video.VideoDecodeError, match="decoded no samples"):
        video._extract_audio_from_path(Path("empty.mp4"), 16_000)
