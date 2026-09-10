# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import SessionLimits, TimedChunk
from tests.unit_test.fixtures.session_pipeline import block_async_call, chunk, pipeline


@pytest.mark.asyncio
async def test_timeout_cancel_noop_waits_before_close(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None, {"ignore_cancel": True, "delay": 0.3}),
            stages=["source", "sink"],
        )
        coordinator._sessions[ref.session_id].limits = SessionLimits(
            command_timeout_s=0.1
        )
        output = coordinator.session_outputs(ref)
        await coordinator.append_session(ref, chunk(0))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(output), 5)
        # The close timeout quarantines ownership until process teardown; late
        # non-preemptible work may finish, but capacity is not falsely returned.
        assert coordinator._sessions[ref.session_id].cleanup_error is not None
        await asyncio.sleep(0.4)
        log = []
        while not events.empty():
            log.append(events.get(timeout=1))
        finished = next(i for i, e in enumerate(log) if e[:2] == ("finished", "sink"))
        # A timed-out queued close may be dropped by request abort. If it ran,
        # it must follow completion; otherwise scheduler.stop owns reclamation.
        close_positions = [i for i, e in enumerate(log) if e[:2] == ("close", "sink")]
        assert all(i > finished for i in close_positions)
        assert not any(e[:2] == ("close", "source") for e in log)


@pytest.mark.asyncio
async def test_open_timeout_quarantines_and_worker_shutdown_releases(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        with pytest.raises(TimeoutError):
            await coordinator.open_session(
                OmniRequest(None, {"open_delay": 0.3}),
                stages=["source", "sink"],
                session_id="slow-open",
                limits=SessionLimits(command_timeout_s=0.08),
            )
        assert coordinator._sessions["slow-open"].cleanup_error is not None
        await asyncio.sleep(0.4)
        with pytest.raises(RuntimeError, match="capacity remains reserved"):
            await coordinator.close_session(coordinator._sessions["slow-open"].ref)


@pytest.mark.asyncio
async def test_worker_failure_wakes_output_and_fails_session(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"]
        )
        output = coordinator.session_outputs(ref)
        waiting = asyncio.create_task(anext(output))
        await asyncio.sleep(0)
        processes[-1].kill()
        processes[-1].expected_exitcode = -9
        await asyncio.to_thread(processes[-1].join, 5)
        await coordinator.fail_pending_requests("session worker exited")
        with pytest.raises(RuntimeError, match="worker exited"):
            await asyncio.wait_for(waiting, 5)
        with pytest.raises(RuntimeError, match="worker exited"):
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])


@pytest.mark.asyncio
async def test_public_subset_shutdown_closes_owners_despite_full_admission(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        refs = [
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])
            for _ in range(3)
        ]
        with pytest.raises(QueueFullError):
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])
        await coordinator.shutdown_stages([])
        assert len(coordinator._sessions) == 3
        coordinator.max_in_flight = 0
        await coordinator.shutdown_stages(["sink"])
        assert not coordinator._sessions
        assert processes[0].is_alive()
        await asyncio.to_thread(processes[1].join, 5)
        assert processes[1].exitcode == 0
        with pytest.raises(ValueError, match="unregistered owner"):
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])
        for ref in refs:
            await coordinator.close_session(ref)


@pytest.mark.asyncio
async def test_partial_open_releases_previously_opened_owner(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        with pytest.raises(RuntimeError, match="open failed"):
            await coordinator.open_session(
                OmniRequest(None, {"fail_open": "sink"}), stages=["source", "sink"]
            )
        assert not coordinator._sessions
        log = [await asyncio.to_thread(events.get, True, 1) for _ in range(3)]
        assert [entry[1] for entry in log if entry[0] == "close"] == ["source"]


@pytest.mark.asyncio
async def test_cancel_failure_closes_owners_in_reverse_order(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None, {"cannot_abort": True}), stages=["source", "sink"]
        )
        with pytest.raises(RuntimeError, match="cannot be retained"):
            await coordinator.abort_session(ref)
        assert not coordinator._sessions
        log = [await asyncio.to_thread(events.get, True, 1) for _ in range(5)]
        assert [entry[1] for entry in log if entry[0] == "close"] == ["sink", "source"]


@pytest.mark.asyncio
async def test_pending_input_limit_rejects_extra_unit(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None),
            stages=["source", "sink"],
            limits=SessionLimits(max_pending_chunks=1),
        )
        await coordinator.append_session(ref, chunk(0))
        with pytest.raises(QueueFullError):
            await coordinator.append_session(ref, chunk(1))
        await coordinator.close_session(ref)


@pytest.mark.asyncio
async def test_input_sequence_rejects_a_gap(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"]
        )
        await coordinator.append_session(ref, chunk(0))
        with pytest.raises(ValueError, match="contiguous"):
            await coordinator.append_session(ref, chunk(2))
        await coordinator.append_session(ref, chunk(1, eos=True))
        await coordinator.close_session(ref)


@pytest.mark.asyncio
async def test_output_overflow_closes_session(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None, {"cadence": 3}),
            stages=["source", "sink"],
            limits=SessionLimits(max_output_chunks=1),
        )
        state = coordinator._sessions[ref.session_id]
        await coordinator.append_session(ref, chunk(0))
        async with asyncio.timeout(5):
            while ref.session_id in coordinator._sessions:
                await asyncio.sleep(0.01)
        assert isinstance(state.error, QueueFullError)


@pytest.mark.asyncio
async def test_idle_timeout_closes_session_and_wakes_reader(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None),
            stages=["source", "sink"],
            limits=SessionLimits(idle_timeout_s=0.3),
        )
        output = coordinator.session_outputs(ref)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(output), 5)
        assert ref.session_id not in coordinator._sessions


@pytest.mark.asyncio
async def test_cross_modality_order_and_rejected_input_retry(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None),
            stages=["source", "sink"],
            limits=SessionLimits(max_pending_chunks=1),
        )
        output = coordinator.session_outputs(ref)
        await coordinator.append_session(ref, chunk(0, eos=True))
        text = TimedChunk("text", 0, 0, 1, "hello", eos=True)
        with pytest.raises(QueueFullError):
            await coordinator.append_session(ref, text)
        for kind in ("data", "input_done"):
            result = await asyncio.wait_for(anext(output), 5)
            assert (result.kind, result.input_seq) == (kind, 0)
        assert await coordinator.append_session(ref, text) == 1
        with pytest.raises(ValueError, match="contiguous"):
            await coordinator.append_session(ref, text)
        for kind in ("data", "input_done"):
            result = await asyncio.wait_for(anext(output), 5)
            assert (result.kind, result.input_seq) == (kind, 1)
        with pytest.raises(ValueError, match="EOS"):
            await coordinator.append_session(ref, chunk(2))
        await output.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["close", "shutdown", "idle"])
async def test_closing_rejects_input_before_cleanup(tmp_path, monkeypatch, trigger):
    async with pipeline(tmp_path) as (coordinator, _, _):
        ref = await coordinator.open_session(
            OmniRequest(None),
            stages=["source", "sink"],
            limits=SessionLimits(idle_timeout_s=0.2 if trigger == "idle" else 300),
        )
        entered, release, _ = block_async_call(
            monkeypatch, coordinator, "_cleanup_session"
        )
        task = None
        if trigger == "close":
            task = asyncio.create_task(coordinator.close_session(ref))
        elif trigger == "shutdown":
            task = asyncio.create_task(coordinator.shutdown_stages(["sink"]))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            with pytest.raises(RuntimeError, match="closing"):
                await coordinator.append_session(ref, chunk(0))
        finally:
            release.set()
            await asyncio.wait_for(
                task if task is not None else coordinator.close_session(ref), 5
            )
