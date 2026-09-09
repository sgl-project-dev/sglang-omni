# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import TimedChunk
from tests.unit_test.fixtures.session_pipeline import chunk, pipeline


@pytest.mark.asyncio
async def test_interleaved_cadences_input_during_output_eos_and_disconnect(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        a = await coordinator.open_session(
            OmniRequest(None, {"cadence": 3, "delay": 0.05}), stages=["source", "sink"]
        )
        b = await coordinator.open_session(
            OmniRequest(None, {"emit_every": 2}), stages=["source", "sink"]
        )
        output_a = coordinator.session_outputs(a)
        output_b = coordinator.session_outputs(b)
        await coordinator.append_session(a, chunk(0))
        first = await asyncio.wait_for(anext(output_a), 5)
        assert first.payload == [1, 0]
        # Accept while unit zero still emits its remaining two outputs.
        await coordinator.append_session(a, chunk(1, eos=True))
        await coordinator.append_session(b, chunk(0))
        receipt = await asyncio.wait_for(anext(output_b), 5)
        assert receipt.kind == "input_done"
        await coordinator.append_session(b, chunk(1, eos=True))
        other = await asyncio.wait_for(anext(output_b), 5)
        assert other.payload == [2, 0] and other.eos
        rest = [await asyncio.wait_for(anext(output_a), 5) for _ in range(7)]
        assert [item.seq for item in [first, *rest]] == list(range(8))
        assert [item.input_seq for item in rest] == [0, 0, 0, 1, 1, 1, 1]
        assert rest[-1].eos
        with pytest.raises(ValueError, match="EOS"):
            await coordinator.append_session(a, chunk(2))
        await output_a.aclose()
        await output_b.aclose()
        assert not coordinator._sessions
        assert not coordinator._requests


@pytest.mark.asyncio
async def test_configured_singleton_list_route_with_three_stages(tmp_path):
    async with pipeline(tmp_path, stage_count=3, list_next=True) as (
        coordinator,
        events,
        processes,
    ):
        ref = await coordinator.open_session(
            OmniRequest(None), stages=["source", "middle", "sink"]
        )
        output = coordinator.session_outputs(ref)
        await coordinator.append_session(ref, chunk(0, eos=True))
        data = await asyncio.wait_for(anext(output), 5)
        assert data.kind == "data" and data.payload == [1, 0]
        receipt = await asyncio.wait_for(anext(output), 5)
        assert receipt.kind == "input_done" and receipt.eos
        await output.aclose()


@pytest.mark.asyncio
async def test_replica_owner_survives_units_abort_and_scoped_shutdown(tmp_path):
    async with pipeline(tmp_path, replicated=True) as (coordinator, events, processes):
        first = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="first"
        )
        second = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="second"
        )
        first_output = coordinator.session_outputs(first)
        second_output = coordinator.session_outputs(second)

        async def unit(ref, output, seq):
            await coordinator.append_session(ref, chunk(seq))
            data = await asyncio.wait_for(anext(output), 5)
            receipt = await asyncio.wait_for(anext(output), 5)
            assert data.kind == "data" and receipt.kind == "input_done"
            assert receipt.input_seq == seq

        await unit(first, first_output, 0)
        await unit(second, second_output, 0)
        first = await coordinator.abort_session(first)
        await unit(first, first_output, 1)
        await unit(second, second_output, 1)
        await coordinator.shutdown_stages(["sink@r0"])
        assert processes[0].is_alive() and processes[2].is_alive()
        await unit(second, second_output, 2)
        await first_output.aclose()
        await second_output.aclose()
        log = []
        while not events.empty():
            log.append(events.get(timeout=1))
        first_owners = {
            event[1] for event in log if event[0] == "append" and event[2] == "first"
        }
        second_owners = {
            event[1] for event in log if event[0] == "append" and event[2] == "second"
        }
        assert first_owners == {"source", "sink@r0"}
        assert second_owners == {"source", "sink@r1"}
        assert ("abort", "sink@r0", "first") in log
        assert ("abort", "sink@r1", "first") not in log


@pytest.mark.asyncio
async def test_clear_preserves_input_clock_and_copies_discarded_payload(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None, {"cadence": 2, "delay": 0.1}), stages=["source", "sink"]
        )
        output = coordinator.session_outputs(ref)
        await coordinator.append_session(ref, chunk(0))
        await asyncio.wait_for(anext(output), 5)
        payload = {"values": [1]}
        await coordinator.append_session(ref, TimedChunk("audio", 20, 20, 1, payload))
        payload["values"].append(2)
        discarded = await coordinator.clear_session_input(ref)
        assert [c.seq for c in discarded] == [1]
        assert discarded[0].payload == {"values": [1]}
        assert sum(c.duration_ms for c in discarded) == 20
        with pytest.raises(ValueError, match="contiguous"):
            await coordinator.append_session(ref, chunk(1))
        await coordinator.append_session(ref, chunk(2, eos=True))
        receipts = []
        while len(receipts) < 2:
            item = await asyncio.wait_for(anext(output), 5)
            if item.kind == "input_done":
                receipts.append(item)
        assert [r.input_seq for r in receipts] == [0, 2]
        assert receipts[-1].eos
        await output.aclose()


@pytest.mark.asyncio
async def test_open_inputs_are_not_relayed_by_later_commands(tmp_path, monkeypatch):
    from dataclasses import asdict

    import msgpack

    from sglang_omni.admission import QueueFullError
    from sglang_omni.proto.session import SESSION_METADATA_KEY, SessionLimits

    async with pipeline(tmp_path) as (coordinator, events, processes):
        submitted = []
        submit = coordinator._submit_request

        async def record(request_id, request, **kwargs):
            submitted.append(
                (request.metadata[SESSION_METADATA_KEY]["op"], request.inputs)
            )
            return await submit(request_id, request, **kwargs)

        monkeypatch.setattr(coordinator, "_submit_request", record)
        request = OmniRequest(b"initial audio")
        unit = TimedChunk("audio", 0, 80, 0, b"unit")
        limit = len(msgpack.packb(asdict(unit), use_bin_type=True))
        ref = await coordinator.open_session(
            request,
            stages=["source", "sink"],
            limits=SessionLimits(max_chunk_bytes=limit),
        )
        output = coordinator.session_outputs(ref)
        with pytest.raises(QueueFullError):
            await coordinator.append_session(
                ref, TimedChunk("audio", 0, 80, 0, b"units")
            )
        await coordinator.append_session(ref, unit)
        assert (await asyncio.wait_for(anext(output), 5)).kind == "data"
        assert (await asyncio.wait_for(anext(output), 5)).kind == "input_done"
        ref = await coordinator.abort_session(ref)
        await output.aclose()
        assert [value for op, value in submitted if op == "open"] == [
            request.inputs
        ] * 2
        later = [(op, value) for op, value in submitted if op != "open"]
        assert {op for op, _ in later} == {"append", "abort", "close"}
        assert all(value is None for _, value in later)
        assert request.inputs == b"initial audio"
