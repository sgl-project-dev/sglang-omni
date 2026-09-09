# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest


@pytest.mark.asyncio
async def test_cleanup_timeout_has_no_false_zero_held():
    from sglang_omni.serve.realtime.control import Closed, Failure
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    class Adapter:
        async def open(self, *args):
            pass

        async def close(self):
            await asyncio.Event().wait()

    runtime = SessionRuntime(
        "mock", Capabilities(), Adapter, RuntimeLimits(cleanup_timeout_s=0.01)
    )
    await runtime.update({}, "open")
    await runtime.close("test")
    events = [env.event for env, _ in runtime.output]
    assert any(isinstance(e, Failure) and e.code == "cleanup_timeout" for e in events)
    assert not any(isinstance(e, Closed) for e in events)
    assert runtime.worker.done()


@pytest.mark.asyncio
async def test_output_overflow_failure_has_bounded_deliverable_terminal():
    from sglang_omni.serve.realtime.control import Closed, Failure
    from sglang_omni.serve.realtime.output import ResponseStarted, TextDelta
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    class Adapter:
        async def open(self, sid, cfg, emit):
            self.emit = emit

        async def process(self, unit, epoch):
            await self.emit(ResponseStarted("r"), epoch)
            for _ in range(10):
                await self.emit(TextDelta("r", "i", "chunk"), epoch)
            return unit.real_samples

        async def close(self):
            pass

    runtime = SessionRuntime(
        "mock", Capabilities(), Adapter, RuntimeLimits(max_output_events=3)
    )
    await runtime.update({}, "open")
    await runtime.append(b"\0\0" * 320, 0, 0, "append")
    await asyncio.sleep(0.02)
    assert runtime.state == "CLOSED"
    events = [env.event for env, _ in runtime.output]
    assert len(events) <= 2
    assert isinstance(events[0], Failure) and isinstance(events[1], Closed)


@pytest.mark.asyncio
async def test_close_preserves_visible_response_terminals():
    from sglang_omni.serve.realtime.control import Closed
    from sglang_omni.serve.realtime.output import ResponseFinished, ResponseStarted
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    class Adapter:
        async def open(self, *args):
            pass

        async def close(self):
            pass

    runtime = SessionRuntime("mock", Capabilities(), Adapter, RuntimeLimits())
    await runtime.update({}, "open")
    await runtime.emit(ResponseStarted("r"), 0)
    runtime.before_send(runtime.output[-1][0])
    await runtime.close("client_closed", "close")
    events = [env.event for env, _ in runtime.output]
    assert isinstance(events[-1], Closed)
    assert sum(isinstance(e, ResponseFinished) for e in events) == 1
    assert isinstance(events[-2], ResponseFinished)


@pytest.mark.asyncio
async def test_explicit_close_drains_stalled_sender_before_socket_close():
    import json

    from sglang_omni.serve.realtime.output import ResponseStarted
    from sglang_omni.serve.realtime.protocol import SharedRealtimeSession
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    class Adapter:
        async def open(self, *args):
            pass

        async def close(self):
            pass

    class Socket:
        def __init__(self):
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.frames = []

        async def receive(self):
            await self.blocked.wait()
            return {
                "type": "websocket.receive",
                "text": json.dumps({"type": "session.close", "event_id": "close"}),
            }

        async def send_text(self, raw):
            event = json.loads(raw)
            if event["type"] == "response.created":
                self.blocked.set()
                await self.release.wait()
            self.frames.append(event)

        async def close(self):
            self.frames.append({"type": "socket.closed"})

    runtime = SessionRuntime("mock", Capabilities(), Adapter, RuntimeLimits())
    await runtime.update({}, "open")
    await runtime.emit(ResponseStarted("r"), 0)
    socket = Socket()
    session = SharedRealtimeSession(socket, runtime)
    task = asyncio.create_task(session.run())
    await socket.blocked.wait()
    await asyncio.sleep(0.02)
    assert not task.done()
    socket.release.set()
    await asyncio.wait_for(task, 1)
    kinds = [event["type"] for event in socket.frames]
    assert (
        kinds.index("response.done")
        < kinds.index("session.closed")
        < kinds.index("socket.closed")
    )


@pytest.mark.asyncio
async def test_close_during_delayed_admission_cannot_resurrect_or_leak():
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    started, release = asyncio.Event(), asyncio.Event()

    class Adapter:
        allocated = False

        async def open(self, *args):
            started.set()
            await release.wait()
            self.allocated = True

        async def close(self):
            self.allocated = False

    adapter = Adapter()
    runtime = SessionRuntime("mock", Capabilities(), lambda: adapter, RuntimeLimits())
    opening = asyncio.create_task(runtime.update({}, "open"))
    await started.wait()
    closing = asyncio.create_task(runtime.close("session_timeout"))
    await asyncio.sleep(0.01)
    release.set()
    await asyncio.gather(opening, closing)
    assert runtime.state == "CLOSED"
    assert not adapter.allocated
    assert runtime.worker is None or runtime.worker.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_failure", ["timeout", "exception"])
async def test_failed_close_stops_active_runtime_worker(close_failure):
    from sglang_omni.serve.realtime.control import Closed, Failure
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    entered = asyncio.Event()

    class Adapter:
        async def open(self, *args):
            pass

        async def process(self, *args):
            entered.set()
            await asyncio.Event().wait()

        async def close(self):
            if close_failure == "exception":
                raise RuntimeError("release unacknowledged")
            await asyncio.Event().wait()

    runtime = SessionRuntime(
        "mock", Capabilities(), Adapter, RuntimeLimits(cleanup_timeout_s=0.02)
    )
    await runtime.update({}, "open")
    await runtime.append(b"\0\0" * 320, 0, 0, "append")
    await entered.wait()
    await runtime.close("client_closed")
    assert runtime.worker.done()
    events = [env.event for env, _ in runtime.output]
    assert any(isinstance(e, Failure) and e.code == "cleanup_timeout" for e in events)
    assert not any(isinstance(e, Closed) for e in events)


@pytest.mark.asyncio
async def test_shared_manager_preserves_transcription_intent(monkeypatch):
    from sglang_omni.config import AudioChunkingConfig, RealtimeTranscriptionConfig
    from sglang_omni.serve.realtime.manager import RealtimeSessionManager
    from sglang_omni.serve.realtime.protocol import SharedRealtimeSession
    from sglang_omni.serve.realtime.transcription_session import (
        RealtimeTranscriptionSession,
    )

    monkeypatch.setattr(RealtimeTranscriptionSession, "_new_vad", lambda *args: None)
    manager = RealtimeSessionManager(
        client=object(),
        model_name="asr",
        audio_chunking=AudioChunkingConfig(),
        transcription_config=RealtimeTranscriptionConfig(strategy_cls=object),
    )
    from starlette.websockets import WebSocket, WebSocketState

    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    websocket = WebSocket({"type": "websocket"}, receive, send)
    await websocket.accept()
    conversation = manager.open(object())
    transcription = manager.open(websocket, intent="transcription")
    assert isinstance(conversation, SharedRealtimeSession)
    assert isinstance(transcription, RealtimeTranscriptionSession)
    with pytest.raises(ValueError, match="Realtime intent"):
        manager.open(object(), intent="unknown")
    await manager.close(conversation.session_id)
    await manager.close(transcription.session_id)
    assert manager.active_sessions() == []
    assert websocket.application_state == WebSocketState.DISCONNECTED
    assert sent[-1]["type"] == "websocket.close"


@pytest.mark.asyncio
@pytest.mark.parametrize("during_unit", [False, True])
async def test_coordinator_output_eof_fails_pending_and_future_units(during_unit):
    from sglang_omni.proto.session import SessionRef
    from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
    from sglang_omni.serve.realtime.runtime import Unit

    release = asyncio.Event()
    appended = asyncio.Event()
    events = []

    class Client:
        async def open_session(self, *args, **kwargs):
            return SessionRef(kwargs["session_id"])

        async def append_session(self, *args):
            appended.set()

        async def session_outputs(self, ref):
            await release.wait()
            if False:
                yield None

        async def close_session(self, ref):
            release.set()

    async def emit(event, *args):
        events.append(event)

    adapter = CoordinatorAdapter(
        Client(),
        stages=["mock"],
        request_builder=lambda cfg: None,
        output_converter=lambda output: [],
        atomic_consumption=True,
    )
    await adapter.open("eof", {}, emit)
    task = None
    try:
        unit = Unit(0, 0, b"\0\0" * 8, 8)
        if during_unit:
            task = asyncio.create_task(adapter.process(unit, 0))
            await appended.wait()
        release.set()
        await adapter.reader
        if task is None:
            task = asyncio.create_task(adapter.process(unit, 0))
        with pytest.raises(RuntimeError, match="output stream closed"):
            await asyncio.wait_for(task, 0.2)
        with pytest.raises(RuntimeError, match="output stream closed"):
            await asyncio.wait_for(adapter.process(unit, 0), 0.2)
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await adapter.close()
