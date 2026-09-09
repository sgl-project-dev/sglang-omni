# SPDX-License-Identifier: Apache-2.0
import asyncio

import httpx
import pytest
import websockets

from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.runtime import Capabilities, RuntimeLimits
from tests.unit_test.fixtures.realtime_websocket import (
    Producer,
    append,
    endpoint,
    recv,
    send,
    until,
)


@pytest.mark.asyncio
async def test_unknown_progress_failure_closes_only_affected_connection():
    class Unknown(Producer):
        async def cancel(self):
            raise RuntimeError("interrupted native media consumption is unknown")

    producer = Unknown()
    producer.release.clear()
    async with endpoint(producer) as (_, url, _, app):
        async with websockets.connect(url) as ws:
            await recv(ws)
            await send(ws, "session.update", session={})
            await recv(ws)
            await append(ws, 0, 320)
            await recv(ws)
            await producer.started.wait()
            await send(ws, "response.cancel", "cancel")
            error = await recv(ws)
            assert (
                error["sglang"]["fatal"] is True
                and error["error"]["code"] == "internal"
            )
            assert (await recv(ws))["type"] == "session.closed"
        assert producer.closed == 1
        for _ in range(100):
            if not app.state.realtime_manager.sessions:
                break
            await asyncio.sleep(0.005)
        for _ in range(100):
            if not app.state.realtime_manager.sessions:
                break
            await asyncio.sleep(0.01)
        assert not app.state.realtime_manager.sessions


@pytest.mark.asyncio
async def test_typed_context_limit_closes_with_explicit_reason():
    from sglang_omni.serve.realtime.output import TurnFailure

    class Limited(Producer):
        async def process(self, unit, epoch):
            await self.emit(
                TurnFailure(
                    "server_error", "context_limit", "native context exhausted"
                ),
                epoch,
            )
            return unit.real_samples

    async with endpoint(Limited()) as (_, url, _, _):
        async with websockets.connect(url) as ws:
            await recv(ws)
            await send(ws, "session.update", session={})
            await recv(ws)
            await append(ws, 0, 320)
            _, events = await until(ws, "session.closed")
            assert events[-1]["reason"] == "context_limit"
            assert any(
                e.get("error", {}).get("code") == "context_limit" for e in events
            )


@pytest.mark.asyncio
async def test_capabilities_unready_is_503_without_allocation():
    async with endpoint() as (http, _, producer, app):
        app.state.client.health = lambda: {"running": False}
        async with httpx.AsyncClient() as client:
            response = await client.get(http + "/v1/realtime/capabilities")
        assert response.status_code == 503
        assert producer.opened == 0


@pytest.mark.asyncio
async def test_created_connections_exhaust_capacity_before_model_allocation():
    producer = Producer()
    deployment = RealtimeDeployment(Capabilities(), lambda: producer, max_connections=1)
    async with endpoint(deployment=deployment) as (_, url, _, app):
        async with websockets.connect(url) as first:
            await recv(first)
            with pytest.raises(websockets.exceptions.InvalidStatus) as error:
                async with websockets.connect(url):
                    pass
            assert error.value.response.status_code == 503
            assert len(app.state.realtime_manager.sessions) == 1
            assert producer.opened == 0


@pytest.mark.asyncio
async def test_coordinator_cleanup_failure_stops_reader_but_keeps_native_owner(
    tmp_path, monkeypatch
):
    from sglang_omni.client import Client
    from sglang_omni.proto import OmniRequest
    from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
    from sglang_omni.serve.realtime.control import Closed, Failure
    from sglang_omni.serve.realtime.runtime import SessionRuntime
    from tests.unit_test.fixtures.session_pipeline import pipeline

    async with pipeline(tmp_path) as (coordinator, _, _):
        adapter = CoordinatorAdapter(
            Client(coordinator),
            stages=["source", "sink"],
            request_builder=lambda cfg: OmniRequest(inputs=None, params={"delay": 0.2}),
            output_converter=lambda output: [],
            atomic_consumption=True,
        )
        runtime = SessionRuntime(
            "mock",
            Capabilities(),
            lambda: adapter,
            RuntimeLimits(cleanup_timeout_s=0.2),
        )
        await runtime.update({}, "open")
        await runtime.append(b"\0\0" * 320, 0, 0, "append")
        for _ in range(100):
            if adapter.active is not None:
                break
            await asyncio.sleep(0.001)
        assert adapter.active is not None
        original = coordinator._session_command

        async def fail_close(session, op, **kwargs):
            if op == "close":
                raise TimeoutError("native release unacknowledged")
            return await original(session, op, **kwargs)

        monkeypatch.setattr(coordinator, "_session_command", fail_close)
        await runtime.close("client_closed")
        assert runtime.worker.done() and adapter.reader.done()
        retained = coordinator._sessions[runtime.session_id]
        assert (
            retained.cleanup_error is not None
        )  # Note (Junnan Li): native budget is quarantined, not released
        events = [env.event for env, _ in runtime.output]
        assert any(
            isinstance(e, Failure) and e.code == "cleanup_timeout" for e in events
        )
        assert not any(isinstance(e, Closed) for e in events)
        monkeypatch.setattr(coordinator, "_session_command", original)
