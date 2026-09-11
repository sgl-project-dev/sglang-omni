# SPDX-License-Identifier: Apache-2.0
"""What one request per model family asks for, and what the answer must contain.

Not a test module: the unit lane imports EXERCISES from here to prove every
family in the table has a request shaped for it, and pytest must not collect it.
"""

from __future__ import annotations

import base64
import io
import json
import wave
from array import array
from pathlib import Path

import pytest
import requests

from tests.test_ci.xpu_model.conftest import REQUEST_TIMEOUT, SPEECH_REQUEST_TIMEOUT
from tests.test_ci.xpu_model.model_table import XpuModel

_DATA = Path(__file__).resolve().parents[2] / "data"

REF_AUDIO = _DATA / "query_to_cars.wav"
REF_TEXT = "How many cars are there in the picture?"
# A content word from REF_TEXT separates a working decode from confident babble
# without pulling in an edit-distance library.
REF_KEYWORD = "car"

SPOKEN_TEXT = "The quick brown fox jumps over the lazy dog."
# ~2.5 s of speech, so 1 s is below any plausible rendering and above a truncated one.
MIN_SPEECH_SECONDS = 1.0

CV_VOICE = "Vivian"


# ------------------------------------------------------------------- assertions


def assert_audible_speech(raw: bytes, label: str) -> float:
    """Assert ``raw`` is a decodable WAV holding audible frames; return seconds.

    A broken vocoder still returns 200 with an undecodable container, a header
    with no frames, or frames that are all zeros.
    """
    assert raw, f"{label}: empty response body"
    try:
        with wave.open(io.BytesIO(raw)) as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            width = handle.getsampwidth()
            payload = handle.readframes(frames)
    except wave.Error as exc:
        pytest.fail(f"{label}: response is not a decodable WAV ({exc}); {raw[:64]!r}")

    assert frames > 0, f"{label}: WAV header declares no frames"
    assert rate > 0, f"{label}: WAV declares no sample rate"
    seconds = frames / rate
    assert seconds >= MIN_SPEECH_SECONDS, (
        f"{label}: {seconds:.2f}s of audio is too short for {SPOKEN_TEXT!r}; "
        "the utterance was truncated"
    )

    typecode = {1: "b", 2: "h", 4: "i"}.get(width)
    if typecode is None:  # pragma: no cover - no XPU vocoder emits these
        return seconds
    samples = array(typecode)
    samples.frombytes(payload[: len(payload) - len(payload) % samples.itemsize])
    peak = max((abs(sample) for sample in samples), default=0)
    full_scale = float(1 << (8 * width - 1))
    assert peak / full_scale > 0.01, (
        f"{label}: {seconds:.2f}s of audio peaks at {peak / full_scale:.4f} of "
        "full scale, which is silence"
    )
    return seconds


def _message_content(body: dict, label: str) -> str:
    assert body.get("choices"), f"{label}: no choices in response: {body}"
    content = body["choices"][0].get("message", {}).get("content")
    assert content, f"{label}: no assistant content: {json.dumps(body)[:400]}"
    return content


# --------------------------------------------------------------------- exercises


def _exercise_transcription(base_url: str, checkpoint: str, model: XpuModel) -> str:
    with REF_AUDIO.open("rb") as handle:
        response = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": (REF_AUDIO.name, handle.read(), "audio/wav")},
            data={"model": checkpoint},
            timeout=REQUEST_TIMEOUT,
        )
    assert response.status_code == 200, f"http {response.status_code}: {response.text}"
    heard = response.json().get("text", "")
    assert heard.strip(), f"empty transcript for {REF_TEXT!r}"
    assert REF_KEYWORD in heard.lower(), (
        f"transcript {heard!r} does not contain {REF_KEYWORD!r}, so the decode "
        f"did not recover {REF_TEXT!r}"
    )
    return f"heard {heard.strip()!r}"


def _speak(base_url: str, payload: dict, label: str) -> str:
    response = requests.post(
        f"{base_url}/v1/audio/speech",
        json=payload,
        timeout=SPEECH_REQUEST_TIMEOUT,
    )
    assert response.status_code == 200, f"http {response.status_code}: {response.text}"
    seconds = assert_audible_speech(response.content, label)
    return f"{seconds:.2f}s of audible speech"


def _exercise_speech_cloned(base_url: str, checkpoint: str, model: XpuModel) -> str:
    # The Base checkpoint clones a voice, so ref_audio/ref_text are required.
    return _speak(
        base_url,
        {
            "model": checkpoint,
            "input": SPOKEN_TEXT,
            "voice": "default",
            "ref_audio": str(REF_AUDIO),
            "ref_text": REF_TEXT,
            "response_format": "wav",
        },
        model.key,
    )


def _exercise_speech_builtin_voice(
    base_url: str, checkpoint: str, model: XpuModel
) -> str:
    return _speak(
        base_url,
        {
            "model": checkpoint,
            "input": SPOKEN_TEXT,
            "voice": CV_VOICE,
            "task_type": "CustomVoice",
            "response_format": "wav",
        },
        model.key,
    )


def _exercise_chat(base_url: str, checkpoint: str, model: XpuModel) -> str:
    # One correct short answer: a thinker decoding garbage still returns content.
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": checkpoint,
            "messages": [
                {"role": "user", "content": "Answer with the number only: 17 plus 25?"}
            ],
            "max_tokens": 32,
            "temperature": 0,
        },
        timeout=REQUEST_TIMEOUT,
    )
    assert response.status_code == 200, f"http {response.status_code}: {response.text}"
    content = _message_content(response.json(), model.key)
    assert "42" in content, f"expected 42 in the answer, got {content.strip()!r}"
    return f"answered {content.strip()[:40]!r}"


def _exercise_chat_speech(base_url: str, checkpoint: str, model: XpuModel) -> str:
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": checkpoint,
            "messages": [{"role": "user", "content": f"Say exactly: {SPOKEN_TEXT}"}],
            "modalities": ["text", "audio"],
            "audio": {"format": "wav"},
            "max_tokens": 64,
        },
        timeout=SPEECH_REQUEST_TIMEOUT,
    )
    assert response.status_code == 200, f"http {response.status_code}: {response.text}"
    body = response.json()
    audio = body["choices"][0].get("message", {}).get("audio") or {}
    data = audio.get("data")
    assert data, f"no audio in response: {json.dumps(body)[:400]}"
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    seconds = assert_audible_speech(base64.b64decode(data), model.key)
    return f"talker and vocoder produced {seconds:.2f}s of audible speech"


EXERCISES = {
    "transcription": _exercise_transcription,
    "speech_cloned": _exercise_speech_cloned,
    "speech_builtin_voice": _exercise_speech_builtin_voice,
    "chat": _exercise_chat,
    "chat_speech": _exercise_chat_speech,
}
