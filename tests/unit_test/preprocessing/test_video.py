# SPDX-License-Identifier: Apache-2.0
"""Tests for video audio extraction."""

from __future__ import annotations

import asyncio
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
import pytest

from sglang_omni.preprocessing import video
from sglang_omni.preprocessing.resource_connector import run_media_io
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


@pytest.mark.parametrize("backend", ["torchvision", "decord"])
@pytest.mark.parametrize(
    ("error", "bad_request"),
    [
        (video.av.error.InvalidDataError(1094995529, "broken stream"), True),
        (video.av.error.EOFError(541478725, "broken stream"), True),
        (RuntimeError("reader unavailable"), False),
        (MemoryError("allocation failed"), False),
        (PermissionError("permission denied"), False),
    ],
)
def test_video_reader_error_classification(
    tmp_path, monkeypatch, backend, error, bad_request
):
    """Reader fallback must distinguish invalid media from backend failures."""
    path = tmp_path / "valid.mp4"
    _write_video_with_audio(path)

    def fail(_item):
        raise error

    monkeypatch.setattr(video.qwen_vision, "get_video_reader_backend", lambda: backend)
    monkeypatch.setattr(
        video.qwen_vision,
        "VIDEO_READER_BACKENDS",
        {"torchvision": fail, "decord": fail},
    )
    with pytest.raises(video.VideoDecodeError) as caught:
        video.load_video_path(path)
    assert caught.value.__cause__ is error
    assert is_bad_request_error(caught.value) is bad_request


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_video_loads_cancel_and_await_siblings(cancel):
    """Both a failed video and request cancellation must finish sibling cleanup."""
    started = asyncio.Event()
    cleaned = asyncio.Event()

    class Connector:
        async def fetch_video_async(self, url, **_kwargs):
            if url.endswith("bad"):
                await started.wait()
                raise video.VideoDecodeError("broken video")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

    urls = ["https://example/slow"]
    if not cancel:
        urls.append("https://example/bad")
    task = asyncio.create_task(
        video.ensure_video_list_async(urls, resource_connector=Connector())
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else video.VideoDecodeError):
        await asyncio.wait_for(task, timeout=5)
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_cancelled_decoder_is_drained_before_returning():
    """A cancelled awaiter must not leave its decoder thread using request resources."""
    started = asyncio.Event()
    released = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()

    def decode():
        loop.call_soon_threadsafe(started.set)
        released.wait(timeout=5)
        finished.set()

    task = asyncio.create_task(run_media_io(decode))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert finished.is_set()
    finally:
        released.set()
