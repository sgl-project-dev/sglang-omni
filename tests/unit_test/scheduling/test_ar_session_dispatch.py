# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from sglang_omni.proto.session import TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionBridge
from tests.unit_test.fixtures.ar_session import TestAdapter, bridge, data, payload


def test_lifecycle_bypasses_builder_and_queue_capacity(monkeypatch):
    from tests.unit_test.pipeline.test_scheduler import _construct_omni_scheduler

    s = _construct_omni_scheduler(monkeypatch)
    b = ARSessionBridge(s, TestAdapter())
    s._session_bridge = b
    s.session_controller = bridge().scheduler.session_controller
    s.tree_cache = s.session_controller.tree_cache
    s._request_builder = lambda _: pytest.fail("lifecycle reached model builder")
    s._waiting_queue_is_full = lambda: True
    s.process_input_requests([payload("open")])
    assert s.outbox.get_nowait().data.data == {"opened": True}
    s.process_input_requests([payload("close", "close")])
    assert s.outbox.get_nowait().data.data == {"closed": True}
    assert not b.owners


def test_output_budget_includes_flush_and_terminal_only(monkeypatch):
    import sglang_omni.scheduling.sglang_backend.ar_session as module

    b = bridge()
    chunk = TimedChunk("text", 0, 0, 0, "hello")
    b.adapter = TestAdapter(
        stream=lambda *args: [chunk], flush=lambda *args: [chunk, chunk]
    )
    b.command(payload("open"))
    b.accept(payload("append"))
    monkeypatch.setattr(module, "get_active_stage", lambda: "encoder")
    assert list(b.messages("r", data(), object())) == []
    monkeypatch.setattr(module, "get_active_stage", lambda: "ar")
    assert len(list(b.messages("r", data(), object()))) == 1
    with pytest.raises(Exception, match="queue"):
        list(b.messages("r", data(), flush=True))


def test_ordinary_embeddings_and_sidecars_unchanged_with_bridge(monkeypatch):
    from tests.unit_test.pipeline.test_scheduler import _construct_omni_scheduler

    s = _construct_omni_scheduler(monkeypatch)
    s._session_bridge = ARSessionBridge(s, TestAdapter())
    p = payload("append")
    p.request.metadata.clear()
    d = data()
    original_req = d.req
    marker = object()
    d.prefill_input_embeds = marker
    d.decode_input_embeds = [marker]
    d.model_inputs = {"marker": marker}
    d.capture_model_output_keys = ("hidden",)
    s._request_builder = lambda _: d
    s._request_kv_capacity_error = lambda req: None
    s.process_input_requests([p])
    assert s.waiting_queue == [original_req]
    assert d.prefill_input_embeds is marker
    assert d.decode_input_embeds == [marker]
    assert d.model_inputs == {"marker": marker}
    assert d.capture_model_output_keys == ("hidden",)


def test_stream_conversion_without_ordinary_builder(monkeypatch):
    import queue

    import sglang_omni.scheduling.sglang_backend.ar_session as module
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    b = bridge()
    b.command(payload("open"))
    b.accept(payload("append"))
    chunk = TimedChunk("text", 0, 0, 0, "converted")
    b.adapter = TestAdapter(stream=lambda *args: [chunk])
    s = object.__new__(OmniScheduler)
    s._session_bridge = b
    s._stream_output_builder = None
    s._aborted_request_ids = set()
    s._first_emit_done = set()
    s.outbox = queue.Queue()
    monkeypatch.setattr(module, "get_active_stage", lambda: "ar")
    sched = SimpleNamespace(requests=[SimpleNamespace(request_id="r", data=data())])
    output = SimpleNamespace(outputs={"r": object()})
    s._emit_stream_output(sched, output)
    assert s.outbox.get_nowait().data == asdict(chunk)
    s._aborted_request_ids.add("r")
    s._emit_stream_output(sched, output)
    assert s.outbox.empty()


def test_retained_sessions_block_direct_cache_flush_and_weight_reset(monkeypatch):
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler, _Upstream

    s = object.__new__(OmniScheduler)
    s._session_bridge = SimpleNamespace(owners={"held": object()})
    monkeypatch.setattr(_Upstream, "flush_cache", lambda *args, **kwargs: True)
    assert s.flush_cache() is False
    assert (
        s._run_weight_update_with_lifecycle(
            {}, lambda _: pytest.fail("weight reset ran"), {}
        )["success"]
        is False
    )
