# SPDX-License-Identifier: Apache-2.0
"""Session value validation and stage state ownership without worker processes."""
from dataclasses import asdict

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import (
    ResourceUsage,
    SessionLimits,
    SessionRef,
    TimedChunk,
)
from sglang_omni.scheduling.session import SessionHooks, SessionScheduler


class Hooks(SessionHooks):
    def __init__(self, name, events):
        self.name, self.events = name, events

    def open(self, ref, request):
        self.events.put(("open", self.name, ref.session_id))
        return {"id": ref.session_id}

    def close(self, state):
        self.events.put(("close", self.name, state["id"]))


def test_timing_and_capacity_validation():
    for bad in [-1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            TimedChunk("audio", bad, 1, 0, b"")
    with pytest.raises(ValueError):
        SessionLimits(max_pending_chunks=0)


def test_open_usage_failure_releases_state():
    import queue

    from sglang_omni.proto import StagePayload
    from sglang_omni.proto.session import SESSION_METADATA_KEY

    class BrokenUsage(Hooks):
        def usage(self, state):
            raise RuntimeError("usage failed")

    events = queue.Queue()
    scheduler = SessionScheduler(BrokenUsage("source", events))
    request = OmniRequest(
        None,
        metadata={
            SESSION_METADATA_KEY: {"op": "open", "ref": asdict(SessionRef("one"))}
        },
    )
    with pytest.raises(RuntimeError, match="usage failed"):
        scheduler._compute(StagePayload("one-open", request, {}))
    assert not scheduler._sessions
    assert events.get_nowait()[0] == "open"
    assert events.get_nowait()[0] == "close"


def test_stage_capacity_is_aggregate_and_unknown_commands_fail():
    import queue

    from sglang_omni.proto import StagePayload
    from sglang_omni.proto.session import SESSION_METADATA_KEY

    class SizedHooks(Hooks):
        def usage(self, state):
            return ResourceUsage(bytes=2)

    scheduler = SessionScheduler(SizedHooks("source", queue.Queue()), max_state_bytes=3)

    def invoke(sid, op):
        request = OmniRequest(
            None,
            metadata={SESSION_METADATA_KEY: {"op": op, "ref": asdict(SessionRef(sid))}},
        )
        return scheduler._compute(StagePayload(sid + op, request, {}))

    invoke("one", "open")
    with pytest.raises(QueueFullError):
        invoke("two", "open")
    assert list(scheduler._sessions) == [("one", 1)]
    with pytest.raises(ValueError, match="unknown session operation"):
        invoke("one", "invalid")
    scheduler.stop()
    assert not scheduler._sessions
