# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest


@pytest.mark.asyncio
async def test_turn_computation_has_typed_sink_and_no_socket(monkeypatch):
    from sglang_omni.serve.realtime import session
    from sglang_omni.serve.realtime.output import TextDelta

    seen = []

    async def emit(event):
        seen.append(event)

    engine = session.TurnBasedSession(
        client=object(), model_name="mock", emit=emit, enable_vad=False
    )
    assert not hasattr(engine, "websocket")
    await engine.emit_output(TextDelta("r", "i", "hello"))
    assert seen == [TextDelta("r", "i", "hello")]
    await engine.append_audio(b"\0\0" * 16)
    assert engine.audio_buffer.num_samples == 16
    await engine.teardown()


@pytest.mark.asyncio
async def test_shared_vad_server_cancel_uses_runtime_epoch(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from sglang_omni.serve.realtime.adapters import TurnBasedAdapterFactory
    from sglang_omni.serve.realtime.control import Cancelled
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )
    from sglang_omni.serve.realtime.vad import VADEvent

    class Detector:
        def process(self, pcm):
            return [
                SimpleNamespace(event_type=VADEvent.SPEECH_STARTED, sample_offset=0)
            ]

        def reset(self):
            pass

    monkeypatch.setattr(
        "sglang_omni.serve.realtime.session.build_turn_detector",
        lambda config, model: SimpleNamespace(
            detector=Detector(),
            effective_config={"type": "server_vad", "interrupt_response": True},
        ),
    )
    runtime = SessionRuntime(
        "mock",
        Capabilities(interaction="turn_based"),
        TurnBasedAdapterFactory(object(), "mock"),
        RuntimeLimits(),
    )
    await runtime.update(
        {"audio": {"input": {"turn_detection": {"type": "server_vad"}}}}, "open"
    )
    engine = runtime.adapter.engine
    engine.active_response_has_audio = True
    engine.cancel_active_response = AsyncMock()
    await runtime.append(b"\0\0" * 320, 0, 0, "append")
    await asyncio.sleep(0.02)
    assert runtime.epoch == 1
    assert engine.cancel_active_response.await_count == 0
    assert any(
        isinstance(env.event, Cancelled) and env.event.client_event_id is None
        for env, _ in runtime.output
    )
    await runtime.close("test")


@pytest.mark.asyncio
async def test_queued_turn_starts_in_new_epoch_after_cancel():
    import contextvars
    from types import SimpleNamespace

    from sglang_omni.serve.realtime.adapters import TurnBasedAdapterFactory
    from sglang_omni.serve.realtime.output import TextDelta
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    class Client:
        async def completion_stream(self, request, **kwargs):
            yield SimpleNamespace(
                modality="text", text="new turn", finish_reason="stop", usage=None
            )

        async def abort(self, request_id):
            pass

    runtime = SessionRuntime(
        "mock",
        Capabilities(interaction="turn_based"),
        TurnBasedAdapterFactory(Client(), "mock"),
        RuntimeLimits(),
    )
    await runtime.update({"audio": {"input": {"turn_detection": None}}}, "open")
    engine = runtime.adapter.engine
    engine.audio_buffer.append_bytes(b"\0\0" * 80)
    payload = engine.audio_buffer.to_full_wav_data_uri()
    engine.speech_idle.clear()
    engine.queued_audio_bytes = len(payload)
    await engine.response_queue.put(("item", payload, contextvars.copy_context()))
    engine.queue_drainer = asyncio.create_task(engine.drain_queue())
    await asyncio.sleep(0)
    await runtime.cancel("cancel")
    engine.speech_idle.set()
    for _ in range(100):
        if any(isinstance(env.event, TextDelta) for env, _ in runtime.output):
            break
        await asyncio.sleep(0.001)
    assert any(
        isinstance(env.event, TextDelta) and env.epoch == 1 for env, _ in runtime.output
    )
    await runtime.close("test")


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", ["response", "transcription", "response_ledger"])
async def test_shared_turn_preserves_context_limit_reason(overflow):
    from types import SimpleNamespace

    from sglang_omni.serve.realtime.adapters import TurnBasedAdapterFactory
    from sglang_omni.serve.realtime.control import Closed, Failure, UnitCompleted
    from sglang_omni.serve.realtime.output import ResponseFinished, ResponseStarted
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    class Client:
        calls = 0

        async def completion_stream(self, request, **kwargs):
            self.calls += 1
            text = (
                "x" * 17
                if overflow == "response"
                or (overflow == "transcription" and self.calls == 2)
                else "ok"
            )
            yield SimpleNamespace(
                modality="text", text=text, finish_reason="stop", usage=None
            )

        async def abort(self, *args):
            pass

    runtime = SessionRuntime(
        "mock",
        Capabilities(interaction="turn_based"),
        TurnBasedAdapterFactory(Client(), "mock"),
        RuntimeLimits(max_history_chars=16, max_responses=1, cleanup_timeout_s=0.1),
    )
    await runtime.update({"audio": {"input": {"turn_detection": None}}}, "open")
    if overflow == "response_ledger":
        await runtime.emit(ResponseStarted("prior"), 0)
        await runtime.emit(
            ResponseFinished("prior", "item", "", False, "completed", "stop"), 0
        )
    await runtime.append(b"\0\0" * 80, 0, 0, "append")
    await runtime.end("end")
    for _ in range(100):
        if runtime.state == "CLOSED":
            break
        await asyncio.sleep(0.005)
    events = [env.event for env, _ in runtime.output]
    assert any(
        isinstance(e, Failure) and e.code == "context_limit" and e.fatal for e in events
    )
    assert any(isinstance(e, Closed) and e.reason == "context_limit" for e in events)
    assert not any(isinstance(e, UnitCompleted) for e in events)
    assert runtime.worker.done()
    engine = runtime.adapter.engine
    assert engine.queue_drainer is None or engine.queue_drainer.done()
    await runtime.close("test")


@pytest.mark.asyncio
async def test_turn_abort_failure_stops_drainer_response_and_runtime_tasks():
    from sglang_omni.serve.realtime.adapters import TurnBasedAdapterFactory
    from sglang_omni.serve.realtime.control import Closed, Failure
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    entered = asyncio.Event()

    class Client:
        async def completion_stream(self, *args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
            if False:
                yield None

        async def abort(self, *args):
            raise RuntimeError("engine release unacknowledged")

    runtime = SessionRuntime(
        "mock",
        Capabilities(interaction="turn_based"),
        TurnBasedAdapterFactory(Client(), "mock"),
        RuntimeLimits(cleanup_timeout_s=0.1),
    )
    await runtime.update({"audio": {"input": {"turn_detection": None}}}, "open")
    await runtime.append(b"\0\0" * 80, 0, 0, "append")
    await runtime.end("end")
    await entered.wait()
    engine = runtime.adapter.engine
    tasks = [
        runtime.worker,
        engine.queue_drainer,
        engine.active_task,
        engine.active_response_task,
    ]
    await runtime.close("client_closed")
    assert all(task.done() for task in tasks if task is not None)
    events = [env.event for env, _ in runtime.output]
    assert any(isinstance(e, Failure) and e.code == "cleanup_timeout" for e in events)
    assert not any(isinstance(e, Closed) for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [16, 25])
async def test_queued_turns_cannot_exceed_cumulative_history_budget(budget):
    from types import SimpleNamespace

    from sglang_omni.serve.realtime.adapters import TurnBasedAdapterFactory
    from sglang_omni.serve.realtime.control import Closed, Failure
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )
    from sglang_omni.serve.realtime.vad import VADEvent

    release = asyncio.Event()
    entered = asyncio.Event()
    request_history_sizes = []

    class Client:
        async def completion_stream(self, request, **kwargs):
            request_history_sizes.append(
                sum(len(message.content) for message in request.messages[1:-1])
            )
            entered.set()
            await release.wait()
            yield SimpleNamespace(
                modality="text", text="abcdefghij", finish_reason="stop", usage=None
            )

        async def abort(self, *args):
            pass

    class Detector:
        position = 0

        def process(self, pcm):
            start = self.position
            self.position += len(pcm) // 2
            return [
                SimpleNamespace(
                    event_type=VADEvent.SPEECH_STARTED, sample_offset=start
                ),
                SimpleNamespace(
                    event_type=VADEvent.SPEECH_STOPPED, sample_offset=self.position
                ),
            ]

        def reset(self):
            pass

    class History(list):
        peak = 0

        def record(self):
            self.peak = max(self.peak, sum(len(item.text) for item in self))

        def append(self, item):
            super().append(item)
            self.record()

        def extend(self, items):
            super().extend(items)
            self.record()

    runtime = SessionRuntime(
        "mock",
        Capabilities(interaction="turn_based"),
        TurnBasedAdapterFactory(Client(), "mock"),
        RuntimeLimits(max_history_chars=budget, cleanup_timeout_s=0.1),
    )
    await runtime.update({"audio": {"input": {"turn_detection": None}}}, "open")
    engine = runtime.adapter.engine
    engine.vad = Detector()
    history = History()
    engine.conversation = history
    await runtime.append(b"\0\0" * 960, 0, 0, "append")
    for _ in range(100):
        if runtime.consumed_samples == 960:
            break
        await asyncio.sleep(0.002)
    assert runtime.consumed_samples == 960
    await asyncio.wait_for(entered.wait(), 1)
    assert engine.response_queue.qsize() == 2
    tasks = [
        runtime.worker,
        engine.queue_drainer,
        engine.active_task,
        engine.active_response_task,
    ]
    await runtime.end("end")
    release.set()
    for _ in range(100):
        if runtime.state == "CLOSED":
            break
        await asyncio.sleep(0.005)
    events = [env.event for env, _ in runtime.output]
    assert any(
        isinstance(e, Failure) and e.code == "context_limit" and e.fatal for e in events
    )
    assert any(isinstance(e, Closed) and e.reason == "context_limit" for e in events)
    assert history.peak <= budget
    assert request_history_sizes and max(request_history_sizes) <= budget
    assert all(task.done() for task in tasks if task is not None)
    assert not runtime.drained.is_set()
    await runtime.close("test")


@pytest.mark.asyncio
async def test_response_builder_checks_history_at_dispatch_boundary():
    from sglang_omni.serve.realtime.output import ContextLimitError
    from sglang_omni.serve.realtime.session import ConversationItem, TurnBasedSession

    async def emit(event):
        pass

    engine = TurnBasedSession(
        client=object(),
        model_name="mock",
        emit=emit,
        enable_vad=False,
        max_text_chars=16,
    )
    engine.conversation = [ConversationItem("user", "x" * 17)]
    with pytest.raises(ContextLimitError):
        engine.build_response_request("audio")
    await engine.teardown()


@pytest.mark.asyncio
async def test_turn_adapter_cancel_waits_for_history_without_aborting():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from sglang_omni.serve.realtime.adapters import TurnBasedAdapterFactory

    release = asyncio.Event()
    history = []

    async def turn():
        await release.wait()
        history.append("completed reply")

    adapter = TurnBasedAdapterFactory(object(), "mock")()
    active = asyncio.create_task(turn())
    abort = AsyncMock()
    adapter.engine = SimpleNamespace(active_task=active, cancel_active_response=abort)
    cancel = asyncio.create_task(adapter.cancel())
    await asyncio.sleep(0)
    assert adapter.output_epoch == 1 and not cancel.done()
    assert not active.cancelled()
    release.set()
    await cancel
    assert history == ["completed reply"]
    abort.assert_not_awaited()
