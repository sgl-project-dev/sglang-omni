# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest


@pytest.mark.asyncio
async def test_cancel_terminates_visible_response_before_ack_and_fences_old_data():
    from sglang_omni.serve.realtime.control import Cancelled
    from sglang_omni.serve.realtime.output import (
        ResponseFinished,
        ResponseStarted,
        TextDelta,
    )
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    class Adapter:
        async def open(self, sid, cfg, emit):
            pass

        async def cancel(self):
            pass

        async def close(self):
            pass

    runtime = SessionRuntime("mock", Capabilities(), Adapter, RuntimeLimits())
    await runtime.update({}, "open")
    runtime.output.clear()
    runtime.output_bytes = 0
    await runtime.emit(ResponseStarted("r"), 0)
    created, _ = runtime.output.popleft()
    runtime.sent(created)
    await runtime.emit(ResponseFinished("r", "i", "", False, "completed", "stop"), 0)
    await runtime.cancel("cancel")
    await runtime.emit(TextDelta("r", "i", "late"), 0)
    control = [env for env, _ in runtime.output if env.control]
    assert isinstance(control[-2].event, ResponseFinished)
    assert control[-2].event.status == "cancelled"
    assert isinstance(control[-1].event, Cancelled)
    await runtime.close("test")


@pytest.mark.asyncio
async def test_cancel_requires_completed_unit_receipt():
    from sglang_omni.proto.session import SessionRef
    from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
    from sglang_omni.serve.realtime.runtime import Unit

    class Client:
        async def open_session(self, *args, **kwargs):
            return SessionRef(kwargs["session_id"])

        async def append_session(self, *args):
            pass

        async def session_outputs(self, ref):
            await closed.wait()
            if False:
                yield None

        async def abort_session(self, ref):
            return SessionRef(ref.session_id, ref.incarnation, ref.epoch + 1)

        async def close_session(self, ref):
            closed.set()

    closed = asyncio.Event()
    adapter = CoordinatorAdapter(
        Client(),
        stages=["mock"],
        request_builder=lambda cfg: None,
        output_converter=lambda out: [],
        atomic_consumption=True,
    )
    await adapter.open("test", {}, lambda *args: None)
    process = asyncio.create_task(adapter.process(Unit(0, 0, b"\0\0" * 8, 8), 0))
    await asyncio.sleep(0)
    adapter.local_cleanup_timeout = 0.02
    with pytest.raises(RuntimeError, match="receipt is missing"):
        await adapter.cancel()
    process.cancel()
    await asyncio.gather(process, return_exceptions=True)
    await adapter.close()


@pytest.mark.asyncio
async def test_native_completion_before_receipt_observation_is_consumed():
    from sglang_omni.proto.session import OutputChunk, SessionRef
    from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
    from sglang_omni.serve.realtime.runtime import Unit

    release, closed = asyncio.Event(), asyncio.Event()

    class Client:
        async def open_session(self, *args, **kwargs):
            return SessionRef(kwargs["session_id"])

        async def append_session(self, *args):
            pass

        async def session_outputs(self, ref):
            await release.wait()
            yield OutputChunk(ref, 0, 0, "audio", 0, 0.5, None, kind="input_done")
            await closed.wait()

        async def abort_session(self, ref):
            release.set()  # Note (Junnan Li): already-completed receipt survives abort, still old epoch
            return SessionRef(ref.session_id, 1, 1)

        async def close_session(self, ref):
            closed.set()

    adapter = CoordinatorAdapter(
        Client(),
        stages=["mock"],
        request_builder=lambda cfg: None,
        output_converter=lambda out: [],
        atomic_consumption=True,
    )
    await adapter.open("test", {}, lambda *args: None)
    task = asyncio.create_task(adapter.process(Unit(0, 0, b"\0\0" * 8, 8), 0))
    await asyncio.sleep(0)
    await adapter.cancel()
    assert await task == 8
    await adapter.close()


@pytest.mark.asyncio
async def test_ack_does_not_count_queued_audio():
    from sglang_omni.serve.realtime.output import AudioDelta, ResponseStarted
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        ProtocolError,
        RuntimeLimits,
        SessionRuntime,
    )

    class Adapter:
        async def open(self, *args):
            pass

        async def close(self):
            pass

    runtime = SessionRuntime(
        "mock", Capabilities(output_modalities=("audio",)), Adapter, RuntimeLimits()
    )
    await runtime.update({}, "open")
    await runtime.emit(ResponseStarted("r"), 0)
    await runtime.emit(AudioDelta("r", "i", b"\0\0" * 160), 0)
    with pytest.raises(ProtocolError):
        await runtime.playback_ack(0, "r", "i", 0, 1)
    await runtime.close("test")


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_type", ["response.created", "response.done"])
async def test_cancel_during_stalled_lifecycle_send(blocked_type):
    import json

    from sglang_omni.serve.realtime.output import ResponseFinished, ResponseStarted
    from sglang_omni.serve.realtime.protocol import SharedRealtimeSession
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    class Adapter:
        async def open(self, *args):
            pass

        async def cancel(self):
            pass

        async def close(self):
            pass

    class Socket:
        def __init__(self):
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.frames = []

        async def send_text(self, raw):
            frame = json.loads(raw)
            if frame["type"] == blocked_type:
                self.blocked.set()
                await self.release.wait()
            self.frames.append(frame)
            if frame["type"] == "sglang.response.cancelled":
                self.cancelled.set()

        async def close(self):
            pass

    runtime = SessionRuntime("mock", Capabilities(), Adapter, RuntimeLimits())
    await runtime.update({}, "open")
    runtime.output.clear()
    runtime.output_bytes = 0
    await runtime.emit(ResponseStarted("r"), 0)
    if blocked_type == "response.done":
        await runtime.emit(
            ResponseFinished("r", "i", "", False, "completed", "stop"), 0
        )
    socket = Socket()
    session = SharedRealtimeSession(socket, runtime)
    sender = asyncio.create_task(session._send())
    await socket.blocked.wait()
    await runtime.cancel("cancel")
    socket.release.set()
    await asyncio.wait_for(socket.cancelled.wait(), 1)
    kinds = [e["type"] for e in socket.frames]
    assert kinds.count("response.done") == 1
    assert (
        kinds.index("response.created")
        < kinds.index("response.done")
        < kinds.index("sglang.response.cancelled")
    )
    await runtime.close("test")
    await sender


@pytest.mark.asyncio
async def test_hot_modality_update_preserves_active_response_and_queued_projection():
    from sglang_omni.serve.realtime.output import ResponseStarted, TextDelta
    from sglang_omni.serve.realtime.projection import project_output
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
        Unit,
    )

    class Adapter:
        async def open(self, *args):
            pass

        async def close(self):
            pass

    runtime = SessionRuntime(
        "mock",
        Capabilities(output_modalities=("text", "audio")),
        Adapter,
        RuntimeLimits(),
    )
    await runtime.update({"output_modalities": ["audio"]}, "open")
    await runtime.emit(ResponseStarted("old"), 0)
    await runtime.emit(TextDelta("old", "i", "before"), 0)
    await runtime.update({"output_modalities": ["text"]}, "switch")
    await runtime.emit(TextDelta("old", "i", "after"), 0)
    active_unit = Unit(0, 0, b"", 0, output_modalities=("audio",))
    await runtime.emit(ResponseStarted("late"), 0, active_unit)
    await runtime.emit(TextDelta("late", "k", "separate reader"), 0, active_unit)
    await runtime.emit(ResponseStarted("new"), 0)
    await runtime.emit(TextDelta("new", "j", "new"), 0)
    projected = [
        project_output(e.event, output_modalities=list(e.output_modalities))["type"]
        for e, _ in runtime.output
        if isinstance(e.event, TextDelta)
    ]
    assert projected == [
        "response.output_audio_transcript.delta",
        "response.output_audio_transcript.delta",
        "response.output_audio_transcript.delta",
        "response.output_text.delta",
    ]
    await runtime.close("test")
