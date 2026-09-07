# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import ModelRunnerOutput


def _fake_model(n: int, hidden: int, code_groups: int) -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_buffer=torch.zeros(n, hidden, dtype=torch.float32),
        _feedback_mask=torch.zeros(n, dtype=torch.bool),
        _output_codes=torch.stack(
            [torch.tensor([i, i + 100], dtype=torch.long) for i in range(n)]
        )[:, :code_groups],
        _output_embeds=torch.stack(
            [torch.full((hidden,), float(i * 7 + 1)) for i in range(n)]
        ),
    )


def _runner(model: SimpleNamespace) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner._feedback_enabled = True
    runner._code2wav_target = "code2wav"
    runner._request_is_aborted = None
    runner._decode_prepared_rows = None
    runner._lookahead_launch_count = 0
    runner._lookahead_resolve_count = 0
    runner._outbox = SimpleNamespace(sent=[])
    runner._outbox.put = runner._outbox.sent.append
    return runner


def _data(
    feedback: torch.Tensor | None,
    text: torch.Tensor | None,
    *,
    thinker_done: bool = False,
    pad: torch.Tensor | None = None,
    stage_payload: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_queue=deque([feedback]) if feedback is not None else deque(),
        pending_text_queue=deque([text]) if text is not None else deque(),
        thinker_chunks_done=thinker_done,
        tts_pad_embed=pad,
        stage_payload=stage_payload,
    )


def _req_wrap(data: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(data=data)


def _sched_batch(n: int) -> SimpleNamespace:
    return SimpleNamespace(reqs=[SimpleNamespace(rid=f"r{i}") for i in range(n)])


def test_row_ownership_survives_prep_then_emit() -> None:
    n, hidden, code_groups = 3, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    requests = [_req_wrap(_data(feedbacks[i], texts[i])) for i in range(n)]
    schedule_batch = _sched_batch(n)

    runner._write_feedback_buffers(requests)

    assert torch.equal(model._feedback_mask, torch.ones(n, dtype=torch.bool))
    for i in range(n):
        assert torch.equal(model._feedback_buffer[i], feedbacks[i] + texts[i])

    runner._emit_code_chunks_and_feedback(
        schedule_batch=schedule_batch, requests=requests
    )

    sent = runner._outbox.sent
    assert [m.request_id for m in sent] == [f"r{i}" for i in range(n)]
    for i, msg in enumerate(sent):
        assert msg.target == "code2wav"
        assert msg.metadata == {"stream": False}
        assert torch.equal(msg.data, model._output_codes[i])
        fb_queue = requests[i].data.pending_feedback_queue
        assert len(fb_queue) == 1
        assert torch.equal(fb_queue[0], model._output_embeds[i])


def test_sparse_feedback_row_stays_unwritten() -> None:
    n, hidden, code_groups = 3, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    feedbacks = [torch.full((hidden,), float(i + 1)) for i in range(n)]
    texts = [torch.full((hidden,), float(10 * (i + 1))) for i in range(n)]
    requests = [
        _req_wrap(_data(feedbacks[0], texts[0])),
        _req_wrap(_data(feedbacks[1], None, thinker_done=False)),
        _req_wrap(_data(feedbacks[2], texts[2])),
    ]

    runner._write_feedback_buffers(requests)

    assert model._feedback_mask.tolist() == [True, False, True]
    assert torch.equal(model._feedback_buffer[1], torch.zeros(hidden))
    assert torch.equal(model._feedback_buffer[0], feedbacks[0] + texts[0])
    assert torch.equal(model._feedback_buffer[2], feedbacks[2] + texts[2])


def test_stale_mask_cannot_leak_into_reused_slot() -> None:
    # Note (wenyao): forward-side mask reset (talker.py:422) needs a real forward; integration-level only
    n, hidden, code_groups = 2, 3, 2
    model = _fake_model(n, hidden, code_groups)
    model._feedback_mask[:n] = True
    runner = _runner(model)

    feedback1 = torch.full((hidden,), 5.0)
    text1 = torch.full((hidden,), 50.0)
    requests = [
        _req_wrap(_data(torch.full((hidden,), 1.0), None, thinker_done=False)),
        _req_wrap(_data(feedback1, text1)),
    ]

    runner._write_feedback_buffers(requests)

    assert model._feedback_mask.tolist() == [False, True]
    assert torch.equal(model._feedback_buffer[0], torch.zeros(hidden))
    assert torch.equal(model._feedback_buffer[1], feedback1 + text1)


def test_row_ownership_tracks_current_batch_order_across_steps() -> None:
    n, hidden, code_groups = 2, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    request_data = {
        "r0": _data(
            torch.full((hidden,), 1.0),
            torch.full((hidden,), 10.0),
        ),
        "r1": _data(
            torch.full((hidden,), 2.0),
            torch.full((hidden,), 20.0),
        ),
    }
    request_data["r0"].pending_text_queue.extend(
        [torch.full((hidden,), 11.0), torch.full((hidden,), 12.0)]
    )
    request_data["r1"].pending_text_queue.append(torch.full((hidden,), 21.0))

    previous_feedback = {
        "r0": torch.full((hidden,), 1.0),
        "r1": torch.full((hidden,), 2.0),
    }
    text_by_request = {
        "r0": [10.0, 11.0, 12.0],
        "r1": [20.0, 21.0],
    }
    step_orders = [("r0", "r1"), ("r1", "r0"), ("r0",)]
    expected_messages: list[tuple[str, torch.Tensor]] = []
    expected_pending_feedback: dict[str, torch.Tensor] = {}

    for step, order in enumerate(step_orders):
        requests = [_req_wrap(request_data[rid]) for rid in order]
        schedule_batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid=rid) for rid in order],
            output_ids=None,
        )
        expected_inputs = [
            previous_feedback[rid] + torch.full((hidden,), text_by_request[rid].pop(0))
            for rid in order
        ]

        runner._write_feedback_buffers(requests)

        assert model._feedback_mask.tolist() == [True] * len(order) + [False] * (
            n - len(order)
        )
        for row, expected in enumerate(expected_inputs):
            assert torch.equal(model._feedback_buffer[row], expected)

        # Match the real forward, which consumes and clears the active mask.
        model._feedback_mask[: len(order)] = False
        tokens = torch.tensor(
            [step * 10 + int(rid[-1]) for rid in order], dtype=torch.long
        )
        codes = torch.stack(
            [
                torch.tensor(
                    [step * 100 + int(rid[-1]), step * 100 + int(rid[-1]) + 1000],
                    dtype=torch.long,
                )
                for rid in order
            ]
        )
        embeds = torch.stack(
            [
                torch.full((hidden,), float(step * 100 + int(rid[-1]) + 1))
                for rid in order
            ]
        )
        model._output_codes[: len(order)] = codes
        model._output_embeds[: len(order)] = embeds

        result = SimpleNamespace()
        runner._stage_token_ids(result, tokens)
        runner._emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
        )

        emitted = runner._outbox.sent[-len(order) :]
        assert [message.request_id for message in emitted] == list(order)
        for row, rid in enumerate(order):
            assert torch.equal(emitted[row].data, codes[row])
            assert torch.equal(request_data[rid].pending_feedback_queue[0], embeds[row])
            previous_feedback[rid] = embeds[row].clone()
            expected_messages.append((rid, codes[row].clone()))
            expected_pending_feedback[rid] = embeds[row].clone()

        assert len(runner._outbox.sent) == len(expected_messages)
        for message, (expected_rid, expected_code) in zip(
            runner._outbox.sent, expected_messages
        ):
            assert message.request_id == expected_rid
            assert torch.equal(message.data, expected_code)
        for rid, expected_feedback in expected_pending_feedback.items():
            pending_feedback = request_data[rid].pending_feedback_queue
            assert len(pending_feedback) == 1
            assert torch.equal(pending_feedback[0], expected_feedback)

        model_runner_output = ModelRunnerOutput(
            outputs={},
            can_run_cuda_graph=False,
            host_token_ids=runner._resolve_host_token_ids(result),
        )
        batch_result = OmniScheduler._make_batch_result(model_runner_output)
        assert batch_result.next_token_ids is model_runner_output.host_token_ids
        assert batch_result.next_token_ids.tolist() == tokens.tolist()


def test_make_batch_result_requires_declared_host_token_ids() -> None:
    malformed_output = SimpleNamespace(next_token_ids=None, can_run_cuda_graph=False)

    with pytest.raises(AttributeError, match="host_token_ids"):
        OmniScheduler._make_batch_result(malformed_output)


class _FakeReq:
    def __init__(self, rid: str, finished: bool, retracted: bool = False) -> None:
        self.rid = rid
        self._finished = finished
        self.is_retracted = retracted

    def finished(self) -> bool:
        return self._finished


def _lookahead_request(rid: str, hidden: int = 3) -> SimpleNamespace:
    req = _FakeReq(rid, finished=False)
    req.to_finish = None
    req._omni_terminal_claimed = False
    req.custom_logit_processor = None
    req.return_logprob = False
    req.sampling_params = SimpleNamespace(
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.5,
        sampling_seed=17,
        min_new_tokens=0,
        max_new_tokens=1024,
    )
    data = _data(None, torch.full((hidden,), 10.0))
    data.pending_text_queue.extend(
        [torch.full((hidden,), 11.0), torch.full((hidden,), 12.0)]
    )
    data.req = req
    req._omni_data = data
    data.decode_input_embeds = []
    return _req_wrap(data)


def _lookahead_runner(n: int = 2):
    model = _fake_model(n, hidden=3, code_groups=2)
    model._sampled_token_ids = torch.arange(n, dtype=torch.long)
    model._decode_prep_rids = None
    model.prepare_events = []

    def prepare(requests):
        model.prepare_events.append("prepare")
        model._decode_prep_rids = [request.data.req.rid for request in requests]

    def invalidate():
        model.prepare_events.append("invalidate")
        model._decode_prep_rids = None

    model.prepare_decode_buffers = prepare
    model.invalidate_decode_buffers = invalidate
    runner = _runner(model)
    aborted = set()
    scheduler = object.__new__(OmniScheduler)
    scheduler._aborted_request_ids = aborted
    runner._request_is_aborted = scheduler.is_request_aborted
    requests = [_lookahead_request(f"r{i}") for i in range(n)]
    batch = SimpleNamespace(reqs=[request.data.req for request in requests])
    return runner, requests, batch, aborted


def _prime_decode_rows(runner, requests, batch) -> None:
    for request in requests:
        request.data.pending_feedback_queue.append(torch.ones(3))
    runner.before_decode(None, batch, requests)


def _make_row_inactive(kind, req, data, aborted) -> None:
    if kind == "finished":
        req._finished = True
    elif kind == "retracted":
        req.is_retracted = True
    elif kind == "pending_abort":
        req.to_finish = object()
    elif kind == "terminal_claimed":
        req._omni_terminal_claimed = True
    elif kind == "published_abort":
        aborted.add(req.rid)
    elif kind == "replaced_data":
        req._omni_data = SimpleNamespace(req=req)
    elif kind == "replaced_owner":
        data.req = _FakeReq(req.rid, finished=False)
    else:
        raise AssertionError(kind)


def test_lookahead_snapshots_survive_next_launch_and_feedback_is_ready_before_resolve():
    runner, requests, batch, _ = _lookahead_runner()
    model = runner.model
    first_codes = model._output_codes.clone()
    first_embeds = model._output_embeds.clone()
    first_tokens = model._sampled_token_ids.clone()
    first_result = SimpleNamespace()
    first = runner.post_decode_launch(first_result, None, requests)

    assert runner._outbox.sent == []
    assert runner.is_decode_batch_ready(
        SimpleNamespace(
            reqs=batch.reqs, forward_mode=SimpleNamespace(is_decode=lambda: True)
        )
    )
    runner.before_decode(None, batch, requests, is_lookahead=True)
    assert model._lookahead_prep is True
    assert torch.equal(model._feedback_buffer, first_embeds + 10.0)
    assert all(not request.data.pending_feedback_queue for request in requests)
    for row, request in enumerate(requests):
        assert torch.equal(
            request.data.decode_input_embeds[0], first_embeds[row] + 10.0
        )

    model._output_codes.add_(1000)
    model._output_embeds.add_(100.0)
    model._sampled_token_ids.add_(50)
    second_codes = model._output_codes.clone()
    second_embeds = model._output_embeds.clone()
    second_tokens = model._sampled_token_ids.clone()
    second_result = SimpleNamespace()
    second = runner.post_decode_launch(second_result, None, requests)
    pending_second = [request.data.pending_feedback_queue[0] for request in requests]

    runner.post_decode_resolve(first, first_result, None, batch, requests)
    assert torch.equal(first_result.next_token_ids, first_tokens)
    assert [message.request_id for message in runner._outbox.sent] == ["r0", "r1"]
    for row, message in enumerate(runner._outbox.sent):
        assert torch.equal(message.data, first_codes[row])
    for row, request in enumerate(requests):
        assert len(request.data.pending_feedback_queue) == 1
        assert request.data.pending_feedback_queue[0] is pending_second[row]
        assert torch.equal(pending_second[row], second_embeds[row])

    model._output_codes.zero_()
    model._output_embeds.zero_()
    model._sampled_token_ids.zero_()
    runner.post_decode_resolve(second, second_result, None, batch, requests)
    assert torch.equal(second_result.next_token_ids, second_tokens)
    assert [message.request_id for message in runner._outbox.sent] == [
        "r0",
        "r1",
        "r0",
        "r1",
    ]
    for row in range(2):
        assert torch.equal(runner._outbox.sent[row].data, first_codes[row])
        assert torch.equal(runner._outbox.sent[row + 2].data, second_codes[row])
        assert len(requests[row].data.pending_feedback_queue) == 1
    assert runner._lookahead_launch_count == runner._lookahead_resolve_count == 2


@pytest.mark.parametrize(
    "inactive",
    [
        "finished",
        "retracted",
        "pending_abort",
        "terminal_claimed",
        "published_abort",
        "replaced_data",
        "replaced_owner",
    ],
)
def test_lookahead_resolve_discards_inactive_row_before_codec_emission(inactive):
    runner, requests, batch, aborted = _lookahead_runner(3)
    data = requests[1].data
    saved_history = torch.full((3,), 8.0)
    data.decode_input_embeds.append(saved_history)
    result = SimpleNamespace()
    launch = runner.post_decode_launch(result, None, requests)
    owned_feedback = data.pending_feedback_queue[0]
    earlier, later = torch.full((3,), -1.0), torch.full((3,), -2.0)
    data.pending_feedback_queue.appendleft(earlier)
    data.pending_feedback_queue.append(later)
    _make_row_inactive(inactive, batch.reqs[1], data, aborted)
    if inactive == "published_abort":
        assert batch.reqs[1].to_finish is None
        assert not batch.reqs[1].finished()

    runner.post_decode_resolve(launch, result, None, batch, requests)

    assert all(message.request_id != "r1" for message in runner._outbox.sent)
    assert len(data.decode_input_embeds) == 1
    assert data.decode_input_embeds[0] is saved_history
    assert len(data.pending_feedback_queue) == 2
    assert data.pending_feedback_queue[0] is earlier
    assert data.pending_feedback_queue[1] is later
    assert all(
        feedback is not owned_feedback for feedback in data.pending_feedback_queue
    )
    assert [message.request_id for message in runner._outbox.sent] == ["r0", "r2"]
    for message, row in zip(runner._outbox.sent, (0, 2)):
        assert torch.equal(message.data, runner.model._output_codes[row])


def test_lookahead_resolve_preserves_single_frame_codec_emission():
    runner, requests, batch, _ = _lookahead_runner(1)
    result = SimpleNamespace()
    snapshot = runner.post_decode_launch(result, None, requests)
    runner.post_decode_resolve(snapshot, result, None, batch, requests)
    assert len(runner._outbox.sent) == 1
    message = runner._outbox.sent[0]
    assert message.request_id == requests[0].data.req.rid
    assert message.data.ndim == 1
    assert torch.equal(message.data, snapshot.codes[0])


def test_lookahead_resolve_changed_row_count_fails_closed_and_cleans_owned_feedback():
    runner, requests, batch, _ = _lookahead_runner()
    launch = runner.post_decode_launch(SimpleNamespace(), None, requests)
    with pytest.raises(RuntimeError, match="captured row count"):
        runner.post_decode_resolve(
            launch,
            SimpleNamespace(),
            None,
            SimpleNamespace(reqs=batch.reqs[:1]),
            requests[:1],
        )
    assert runner._outbox.sent == []
    assert all(not request.data.pending_feedback_queue for request in requests)


def test_lookahead_resolve_reordered_rows_does_not_emit_or_reassign_feedback():
    runner, requests, batch, _ = _lookahead_runner()
    launch = runner.post_decode_launch(SimpleNamespace(), None, requests)
    runner.post_decode_resolve(
        launch,
        SimpleNamespace(),
        None,
        SimpleNamespace(reqs=list(reversed(batch.reqs))),
        list(reversed(requests)),
    )
    assert runner._outbox.sent == []
    assert all(not request.data.pending_feedback_queue for request in requests)


def test_lookahead_retract_prefill_decode_replays_consumed_input_without_duplicate_feedback():
    runner, requests, batch, _ = _lookahead_runner(1)
    model, data = runner.model, requests[0].data
    prompt = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data.prefill_input_embeds = prompt
    first_result = SimpleNamespace()
    first = runner.post_decode_launch(first_result, None, requests)
    consumed = model._output_embeds[0].clone() + data.pending_text_queue[0]
    runner.before_decode(None, batch, requests, is_lookahead=True)
    assert len(data.decode_input_embeds) == 1
    saved_history = data.decode_input_embeds[0]
    next_text = data.pending_text_queue[0]
    model._output_codes.add_(1000)
    model._output_embeds.add_(100.0)
    second_result = SimpleNamespace()
    second = runner.post_decode_launch(second_result, None, requests)
    discarded_feedback = data.pending_feedback_queue[0]
    runner.post_decode_resolve(first, first_result, None, batch, requests)
    batch.reqs[0].is_retracted = True
    runner.post_decode_resolve(second, second_result, None, batch, requests)

    assert not data.pending_feedback_queue
    assert len(data.decode_input_embeds) == 1
    assert data.decode_input_embeds[0] is saved_history
    assert torch.equal(saved_history, consumed)
    assert data.pending_text_queue[0] is next_text
    assert len(runner._outbox.sent) == 1
    replay = runner._projected_prefill_slice(
        sched_req=requests[0],
        prefix_len=0,
        extend_len=3,
        device=torch.device("cpu"),
    )
    assert torch.equal(replay, torch.cat([prompt, consumed.unsqueeze(0)]))
    assert data.pending_text_queue[0] is next_text
    assert len(data.pending_text_queue) == 2
    assert not data.pending_feedback_queue

    def prefill_code_predictor(codes, hidden):
        model._output_codes[:] = torch.tensor([[77, 177]])
        model._output_embeds.fill_(30.0)

    batch.reqs[0].is_retracted = False
    model.code_predictor_forward = prefill_code_predictor
    prefill_result = SimpleNamespace(
        next_token_ids=torch.tensor([77]),
        logits_output=SimpleNamespace(hidden_states=torch.zeros(1, 3)),
    )
    runner.post_prefill(prefill_result, None, batch, requests)
    assert len(data.pending_feedback_queue) == 1
    assert data.pending_feedback_queue[0] is not discarded_feedback
    runner.before_decode(None, batch, requests)
    assert model._lookahead_prep is False
    assert len(data.decode_input_embeds) == 2
    assert data.decode_input_embeds[0] is saved_history
    assert torch.equal(data.decode_input_embeds[1], torch.full((3,), 30.0) + next_text)
    assert not data.pending_feedback_queue
    assert len(data.pending_text_queue) == 1
    assert [message.data.tolist() for message in runner._outbox.sent] == [
        [0, 100],
        [77, 177],
    ]


def test_lookahead_eligibility_allows_seeded_repetition():
    runner, requests, batch, _ = _lookahead_runner()
    _prime_decode_rows(runner, requests, batch)
    assert runner.lookahead_eligible(batch) is True


@pytest.mark.parametrize(
    "change",
    [
        "reorder",
        "new_rid",
        "same_rid_new_request",
        "same_request_new_data",
        "smaller_batch",
        "larger_batch",
        "unprepared",
        "unbound_abort",
    ],
)
def test_lookahead_eligibility_rejects_changed_rows_or_missing_contract(change):
    runner, requests, batch, _ = _lookahead_runner()
    _prime_decode_rows(runner, requests, batch)
    assert runner.lookahead_eligible(batch)
    if change == "reorder":
        batch.reqs.reverse()
    elif change == "new_rid":
        batch.reqs[0] = _lookahead_request("new").data.req
    elif change == "same_rid_new_request":
        batch.reqs[0] = _lookahead_request(batch.reqs[0].rid).data.req
    elif change == "same_request_new_data":
        batch.reqs[0]._omni_data = SimpleNamespace(req=batch.reqs[0])
    elif change == "smaller_batch":
        batch.reqs.pop()
    elif change == "larger_batch":
        batch.reqs.append(_lookahead_request("new").data.req)
    elif change == "unprepared":
        runner._decode_prepared_rows = None
    elif change == "unbound_abort":
        runner._request_is_aborted = None
    assert runner.lookahead_eligible(batch) is False


@pytest.mark.parametrize(
    "inactive",
    [
        "finished",
        "retracted",
        "pending_abort",
        "terminal_claimed",
        "published_abort",
        "replaced_data",
        "replaced_owner",
    ],
)
def test_lookahead_eligibility_rejects_inactive_rows(inactive):
    runner, requests, batch, aborted = _lookahead_runner()
    _prime_decode_rows(runner, requests, batch)
    _make_row_inactive(inactive, batch.reqs[0], requests[0].data, aborted)
    assert runner.lookahead_eligible(batch) is False


@pytest.mark.parametrize(
    "unsupported",
    [
        "frequency_penalty",
        "presence_penalty",
        "custom_logit_processor",
        "return_logprob",
        "intermediate_minimum",
        "fixed_length_minimum",
    ],
)
def test_lookahead_eligibility_rejects_host_history_dependent_sampling(unsupported):
    runner, requests, batch, _ = _lookahead_runner()
    _prime_decode_rows(runner, requests, batch)
    req = batch.reqs[0]
    if unsupported in ("frequency_penalty", "presence_penalty"):
        setattr(req.sampling_params, unsupported, 0.1)
    elif unsupported == "custom_logit_processor":
        req.custom_logit_processor = object()
    elif unsupported == "return_logprob":
        req.return_logprob = True
    elif unsupported == "fixed_length_minimum":
        req.sampling_params.min_new_tokens = req.sampling_params.max_new_tokens
    else:
        req.sampling_params.min_new_tokens = 10
    assert runner.lookahead_eligible(batch) is False


def test_lookahead_launch_requires_bound_abort_predicate():
    runner, requests, _, _ = _lookahead_runner()
    runner._request_is_aborted = None
    with pytest.raises(RuntimeError, match="scheduler abort predicate"):
        runner.post_decode_launch(SimpleNamespace(), None, requests)
    assert all(not request.data.pending_feedback_queue for request in requests)


@pytest.mark.parametrize("replacement", ["request", "data"])
def test_before_decode_invalidates_sampling_cache_for_reused_request_id(replacement):
    runner, requests, batch, _ = _lookahead_runner(1)
    _prime_decode_rows(runner, requests, batch)
    old_req = batch.reqs[0]
    new_request = _lookahead_request(old_req.rid)
    if replacement == "data":
        new_request.data.req = old_req
        old_req._omni_data = new_request.data
    new_requests = [new_request]
    new_batch = SimpleNamespace(reqs=[new_request.data.req])
    _prime_decode_rows(runner, new_requests, new_batch)
    assert runner.model.prepare_events == ["prepare", "invalidate", "prepare"]
    assert runner.lookahead_eligible(new_batch)


def _resolve_scheduler(result: SimpleNamespace) -> tuple[OmniScheduler, list]:
    scheduler = object.__new__(OmniScheduler)
    captured: list = []
    scheduler._run_batch_resolve = (
        lambda batch, sched_output, pending_step, skip_rids=(): result
    )
    scheduler.process_batch_result = lambda batch, res: captured.append(
        ([r.rid for r in batch.reqs], res.next_token_ids)
    )
    return scheduler, captured


def test_overrun_drop_keeps_reqs_and_tokens_index_aligned() -> None:
    reqs = [
        _FakeReq("r0", finished=False),
        _FakeReq("r1", finished=True),
        _FakeReq("r2", finished=False),
        _FakeReq("r3", finished=True),
    ]
    batch = SimpleNamespace(reqs=list(reqs))
    result = SimpleNamespace(next_token_ids=torch.tensor([100, 101, 102, 103]))
    scheduler, captured = _resolve_scheduler(result)

    sched_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id=req.rid) for req in reqs]
    )
    scheduler._resolve_and_process(batch, sched_output, None)

    assert len(captured) == 1
    rids, tokens = captured[0]
    assert rids == ["r0", "r2"]
    assert tokens.tolist() == [100, 102]


def test_overrun_drop_retracted_row_is_dropped() -> None:
    reqs = [
        _FakeReq("r0", finished=False),
        _FakeReq("r1", finished=False, retracted=True),
        _FakeReq("r2", finished=False),
    ]
    batch = SimpleNamespace(reqs=list(reqs))
    result = SimpleNamespace(next_token_ids=torch.tensor([10, 11, 12]))
    scheduler, captured = _resolve_scheduler(result)

    sched_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id=req.rid) for req in reqs]
    )
    scheduler._resolve_and_process(batch, sched_output, None)

    rids, tokens = captured[0]
    assert rids == ["r0", "r2"]
    assert tokens.tolist() == [10, 12]


def test_overrun_drop_noop_keeps_full_alignment() -> None:
    reqs = [_FakeReq(f"r{i}", finished=False) for i in range(3)]
    batch = SimpleNamespace(reqs=list(reqs))
    result = SimpleNamespace(next_token_ids=torch.tensor([7, 8, 9]))
    scheduler, captured = _resolve_scheduler(result)

    sched_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id=req.rid) for req in reqs]
    )
    scheduler._resolve_and_process(batch, sched_output, None)

    rids, tokens = captured[0]
    assert rids == ["r0", "r1", "r2"]
    assert tokens.tolist() == [7, 8, 9]


def test_overrun_drop_all_finished_skips_process() -> None:
    reqs = [_FakeReq("r0", finished=True), _FakeReq("r1", finished=True)]
    batch = SimpleNamespace(reqs=list(reqs))
    result = SimpleNamespace(next_token_ids=torch.tensor([1, 2]))
    scheduler, captured = _resolve_scheduler(result)

    sched_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id=req.rid) for req in reqs]
    )
    scheduler._resolve_and_process(batch, sched_output, None)

    assert captured == []
    assert batch.reqs == []


@pytest.mark.parametrize("removed", [0, 1])
@pytest.mark.parametrize("finished", [False, True])
def test_immediate_abort_survivors_keep_launch_token_rows(removed, finished):
    reqs = [_FakeReq(f"r{i}", finished=False) for i in range(3)]
    reqs[2]._finished = finished
    batch = SimpleNamespace(reqs=[req for i, req in enumerate(reqs) if i != removed])
    result = SimpleNamespace(next_token_ids=torch.tensor([100, 200, 300]))
    scheduler, captured = _resolve_scheduler(result)
    observed_skips = set()

    def resolve(batch, sched_output, pending_step, skip_rids=()):
        observed_skips.update(skip_rids)
        return result

    scheduler._run_batch_resolve = resolve
    launch = SimpleNamespace(
        requests=[SimpleNamespace(request_id=req.rid) for req in reqs]
    )
    scheduler._resolve_and_process(batch, launch, None)
    kept = [i for i in range(3) if i != removed and not reqs[i].finished()]
    assert captured[0][0] == [f"r{i}" for i in kept]
    assert captured[0][1].tolist() == [[100, 200, 300][i] for i in kept]
    assert f"r{removed}" in observed_skips
    if finished:
        assert "r2" in observed_skips


def test_default_off_sync_publishes_codec_before_feedback():
    runner = _runner(_fake_model(1, hidden=3, code_groups=2))
    data = _data(None, None)
    observed = []
    runner._outbox.put = lambda message: observed.append(
        len(data.pending_feedback_queue)
    )
    runner._emit_code_chunks_and_feedback(
        schedule_batch=_sched_batch(1), requests=[_req_wrap(data)]
    )
    assert observed == [0]
    assert len(data.pending_feedback_queue) == 1


def test_async_resolve_preserves_staged_host_token_ids():
    device_ids = torch.tensor([10, 20])
    staged_ids = torch.tensor([10, 20])
    output = ModelRunnerOutput(
        outputs={}, next_token_ids=device_ids, host_token_ids=staged_ids
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler._model_runner = SimpleNamespace(execute_resolve=lambda pending: output)
    emitted = []
    scheduler._emit_stream_output = lambda *args, **kwargs: emitted.append(
        (args, kwargs)
    )
    batch, launch, pending = object(), object(), object()
    result = scheduler._run_batch_resolve(batch, launch, pending, skip_rids={"aborted"})
    assert result.next_token_ids is staged_ids
    assert emitted == [((launch, output), {"skip_rids": {"aborted"}})]
