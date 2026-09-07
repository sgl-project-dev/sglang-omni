# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker runner with FIFO text/feedback decode handoff."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.scheduling.messages import OutgoingMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TalkerLookaheadRow:
    request_id: str
    request: Any
    data: Any
    feedback: torch.Tensor


@dataclass(frozen=True)
class _TalkerLookaheadSnapshot:
    codes: torch.Tensor
    rows: tuple[_TalkerLookaheadRow, ...]


class QwenTalkerModelRunner(ModelRunner):

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        outbox: Any,
        *,
        code2wav_target: str = "code2wav",
        feedback_enabled: bool = True,
        request_is_aborted: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox = outbox
        self._code2wav_target = code2wav_target
        self._feedback_enabled = bool(feedback_enabled)
        self._lookahead_launch_count = 0
        self._lookahead_resolve_count = 0
        self._request_is_aborted = request_is_aborted
        self._decode_prepared_rows: tuple[tuple[Any, Any], ...] | None = None

    def execute(self, scheduler_output: Any):
        return super().execute(scheduler_output)

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del schedule_batch
        composed = self._compose_prefill_embeds(forward_batch, requests)
        if composed is None:
            return
        input_embeds, input_embeds_are_projected = composed
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=input_embeds,
                input_embeds_are_projected=input_embeds_are_projected,
            ),
        )

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del forward_batch
        del schedule_batch
        self.model._lookahead_prep = bool(is_lookahead)
        if not self._feedback_enabled:
            return

        if not self._requests_ready_for_decode(requests):
            raise RuntimeError(
                "Talker decode reached model runner without ready feedback/text input"
            )

        rows = None
        if self._request_is_aborted is not None:
            rows = tuple((sched_req.data.req, sched_req.data) for sched_req in requests)
            previous = self._decode_prepared_rows
            if previous is not None and not self._same_prepared_rows(previous, rows):
                self.model.invalidate_decode_buffers()
        self.model.prepare_decode_buffers(requests)
        self._decode_prepared_rows = rows
        self._write_feedback_buffers(requests)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        # Note (Xuesong): Do not clear data.prefill_input_embeds: decode retract may requeue
        # the Req for another prefill pass and Req.input_embeds is None.
        if not self._feedback_enabled:
            return

        if result.next_token_ids is None:
            return
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        talker_hidden = result.logits_output.hidden_states
        if isinstance(talker_hidden, torch.Tensor) and talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        self.model.code_predictor_forward(layer0_codes, talker_hidden)
        self._stage_token_ids(result, result.next_token_ids)
        self._emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
        )

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if not self._feedback_enabled:
            return

        batch_size = len(requests)
        result.next_token_ids = self.model._sampled_token_ids[:batch_size].clone()
        self._stage_token_ids(result, result.next_token_ids)
        self._emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
        )

    def _emit_code_chunks_and_feedback(
        self,
        *,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        bs = len(requests)
        # Note (wenyao): snapshots must outlive the next graph's fixed-buffer writes.
        codes_snap = self.model._output_codes[:bs].detach().clone()
        embeds_snap = self.model._output_embeds[:bs].detach().clone()
        guarded = self._request_is_aborted is not None
        for idx, sched_req in enumerate(requests):
            req = schedule_batch.reqs[idx]
            data = sched_req.data
            if not guarded:
                self._emit_codec_row(req.rid, data, codes_snap[idx])
                data.pending_feedback_queue.append(embeds_snap[idx])
                continue
            if not self._request_row_is_live(req, data):
                continue
            feedback = embeds_snap[idx]
            data.pending_feedback_queue.append(feedback)
            if guarded and not self._request_row_is_live(req, data):
                self._discard_launch_feedback(
                    _TalkerLookaheadRow(req.rid, req, data, feedback)
                )
                continue
            self._emit_codec_row(req.rid, data, codes_snap[idx])

    def _emit_codec_row(
        self, request_id: str, data: Any, code_chunk: torch.Tensor
    ) -> None:
        self._outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=code_chunk,
                target=self._code2wav_target,
                metadata={"stream": self._is_streaming(data)},
            )
        )

    @staticmethod
    def _same_prepared_rows(
        left: tuple[tuple[Any, Any], ...], right: tuple[tuple[Any, Any], ...]
    ) -> bool:
        return len(left) == len(right) and all(
            left_req is right_req and left_data is right_data
            for (left_req, left_data), (right_req, right_data) in zip(left, right)
        )

    def _request_row_is_live(self, req: Any, data: Any) -> bool:
        # Note (wenyao): the abort set is published before Req.to_finish is updated.
        is_aborted = self._request_is_aborted
        return (
            is_aborted is not None
            and req is not None
            and data is not None
            and not is_aborted(req.rid)
            and req._omni_data is data
            and data.req is req
            and not req.finished()
            and not self._req_is_retracted(req)
            and req.to_finish is None
            and not req._omni_terminal_claimed
        )

    @staticmethod
    def _discard_launch_feedback(row: _TalkerLookaheadRow) -> None:
        queue = row.data.pending_feedback_queue
        for index, candidate in enumerate(queue):
            if candidate is row.feedback:
                del queue[index]
                break
        # Note (wenyao): consumed inputs remain in history for re-prefill replay.

    def post_decode_launch(
        self, result: Any, forward_batch: Any, requests: list
    ) -> Any:
        if not self._feedback_enabled:
            return super().post_decode_launch(result, forward_batch, requests)
        if self._request_is_aborted is None:
            raise RuntimeError(
                "Talker lookahead requires the scheduler abort predicate"
            )
        bs = len(requests)
        result.next_token_ids = self.model._sampled_token_ids[:bs].clone()
        self._stage_token_ids(result, result.next_token_ids)
        codes_snap = self.model._output_codes[:bs].detach().clone()
        embeds_snap = self.model._output_embeds[:bs].detach().clone()
        rows = []
        for idx, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            feedback = embeds_snap[idx]
            row = _TalkerLookaheadRow(req.rid, req, data, feedback)
            # Note (wenyao): the next launch consumes feedback before this one resolves.
            if self._request_row_is_live(req, data):
                data.pending_feedback_queue.append(feedback)
            rows.append(row)
        self._lookahead_launch_count += 1
        return _TalkerLookaheadSnapshot(codes_snap, tuple(rows))

    def post_decode_resolve(
        self,
        launch_buf: Any,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if not self._feedback_enabled:
            return super().post_decode_resolve(
                launch_buf, result, forward_batch, schedule_batch, requests
            )
        if launch_buf is None:
            return
        if not isinstance(launch_buf, _TalkerLookaheadSnapshot):
            raise TypeError("Talker lookahead requires its launch-owned snapshot")
        if len(requests) != len(launch_buf.rows) or len(schedule_batch.reqs) != len(
            launch_buf.rows
        ):
            for row in launch_buf.rows:
                self._discard_launch_feedback(row)
            raise RuntimeError(
                "Talker lookahead resolve changed the captured row count"
            )
        for idx, row in enumerate(launch_buf.rows):
            current = requests[idx]
            if (
                current.data is not row.data
                or schedule_batch.reqs[idx] is not row.request
                or row.request.rid != row.request_id
                or not self._request_row_is_live(row.request, row.data)
            ):
                self._discard_launch_feedback(row)
                continue
            self._emit_codec_row(row.request_id, row.data, launch_buf.codes[idx])
        self._lookahead_resolve_count += 1
        if self._lookahead_resolve_count % 256 == 0:
            logger.info(
                "talker lookahead launches=%d resolves=%d query_hit=%d query_miss=%d",
                self._lookahead_launch_count,
                self._lookahead_resolve_count,
                self._async_query_hit,
                self._async_query_miss,
            )

    def lookahead_eligible(self, batch: Any) -> bool:
        if not self._feedback_enabled or self._request_is_aborted is None:
            return False
        prev_rids = self.model._decode_prep_rids
        prepared = self._decode_prepared_rows
        if (
            prev_rids is None
            or prepared is None
            or len(prev_rids) != len(batch.reqs)
            or len(prepared) != len(batch.reqs)
        ):
            return False
        for req, previous_rid, (prepared_req, data) in zip(
            batch.reqs, prev_rids, prepared
        ):
            if (
                req.rid != previous_rid
                or req is not prepared_req
                or not self._request_row_is_live(req, data)
            ):
                return False
            sp = req.sampling_params
            if sp.frequency_penalty != 0.0 or sp.presence_penalty != 0.0:
                return False
            if req.custom_logit_processor is not None or req.return_logprob:
                return False
            # Note (wenyao): Minimum-token constraints read host lengths that lag lookahead.
            if sp.min_new_tokens != 0:
                return False
        return True

    @staticmethod
    def _is_streaming(data: Any) -> bool:
        stage_payload = data.stage_payload
        return bool(
            stage_payload is not None
            and (stage_payload.request.params or {}).get("stream", False)
        )

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return False

    def is_decode_batch_ready(self, schedule_batch: Any) -> bool:
        if not self._feedback_enabled or not schedule_batch.forward_mode.is_decode():
            return True
        return all(
            self._data_has_next_decode_input(getattr(req, "_omni_data", None))
            for req in schedule_batch.reqs
        )

    def _compose_prefill_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> tuple[torch.Tensor, bool] | None:
        """Assemble prefill rows and preserve whether they are projected."""
        projected_flags = [
            bool(req.data.input_embeds_are_projected) for req in requests
        ]
        has_tensor_requests = any(
            req.data.prefill_input_embeds is not None for req in requests
        )
        if not any(projected_flags) and not has_tensor_requests:
            return None

        has_projected_requests = any(projected_flags)
        if has_projected_requests and not all(projected_flags):
            raise RuntimeError(
                "Talker projected and unprojected prefill requests cannot be "
                "batched together"
            )

        parts: list[torch.Tensor] = []
        for sched_req in requests:
            req = sched_req.data.req
            prefix_len = len(req.prefix_indices)
            extend_len = int(req.extend_range.length)
            part = self._projected_prefill_slice(
                sched_req=sched_req,
                prefix_len=prefix_len,
                extend_len=extend_len,
                device=forward_batch.input_ids.device,
            )
            if part is not None and part.shape[0] > 0:
                parts.append(part)
        if not parts:
            return None
        input_embeds = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

        expected_rows = int(forward_batch.input_ids.shape[0])
        if input_embeds.shape[0] != expected_rows:
            raise RuntimeError(
                "Talker prefill embeds must align with forward input_ids: "
                f"got {input_embeds.shape[0]} rows for {expected_rows} input ids"
            )
        return (
            input_embeds.to(
                device=forward_batch.input_ids.device,
                dtype=self.model.activation_dtype,
            ),
            has_projected_requests,
        )

    @staticmethod
    def _projected_prefill_slice(
        *,
        sched_req: Any,
        prefix_len: int,
        extend_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if extend_len <= 0:
            return None

        data = sched_req.data
        req = data.req
        end = prefix_len + extend_len
        tensor = data.prefill_input_embeds
        if tensor is not None:
            prompt_len = int(tensor.shape[0])
            dtype = tensor.dtype
            embed_device = tensor.device
            parts = QwenTalkerModelRunner._prefill_prompt_parts_from_tensor(
                tensor=tensor,
                prefix_len=prefix_len,
                end=end,
            )
        else:
            embeds = req.input_embeds
            if not embeds:
                return None
            prompt_len = len(embeds)
            dtype = torch.float32
            embed_device = device
            parts = QwenTalkerModelRunner._prefill_prompt_parts_from_list(
                embeds=embeds,
                prefix_len=prefix_len,
                end=end,
                device=device,
            )

        if end > prompt_len:
            generated = QwenTalkerModelRunner._generated_prefill_slice(
                sched_req=sched_req,
                gen_start=max(prefix_len, prompt_len) - prompt_len,
                gen_end=end - prompt_len,
                device=embed_device,
                dtype=dtype,
            )
            if generated is not None:
                parts.append(generated)

        if not parts:
            return None
        return torch.cat(parts, dim=0)

    @staticmethod
    def _prefill_prompt_parts_from_tensor(
        *,
        tensor: torch.Tensor,
        prefix_len: int,
        end: int,
    ) -> list[torch.Tensor]:
        prompt_len = int(tensor.shape[0])
        start = min(prefix_len, prompt_len)
        stop = min(end, prompt_len)
        return [tensor[start:stop]] if stop > start else []

    @staticmethod
    def _prefill_prompt_parts_from_list(
        *,
        embeds: list,
        prefix_len: int,
        end: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        prompt_len = len(embeds)
        start = min(prefix_len, prompt_len)
        stop = min(end, prompt_len)
        if stop <= start:
            return []
        return [
            torch.as_tensor(
                embeds[start:stop],
                device=device,
                dtype=torch.float32,
            )
        ]

    @staticmethod
    def _generated_prefill_slice(
        *,
        sched_req: Any,
        gen_start: int,
        gen_end: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if gen_end <= gen_start:
            return None

        data = sched_req.data
        history = QwenTalkerModelRunner._decode_input_history(data)
        while len(history) < gen_end:
            combined = QwenTalkerModelRunner._take_next_decode_input_embed(
                sched_req=sched_req,
                device=device,
                dtype=dtype,
            )
            if combined is None:
                raise RuntimeError(
                    "Cannot replay retracted talker decode tokens: missing "
                    "feedback/text input embeds for generated-token prefill"
                )
            QwenTalkerModelRunner._append_decode_input_history(data, combined)

        rows = [
            QwenTalkerModelRunner._decode_row(row, device=device, dtype=dtype)
            for row in history[gen_start:gen_end]
        ]
        if not rows:
            return None
        return torch.stack(rows, dim=0)

    def _write_feedback_buffers(self, requests: list) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return

        feedback_buffer = self.model._feedback_buffer
        feedback_mask = self.model._feedback_mask
        feedback_mask[:batch_size] = False

        rows: list[int] = []
        embeds: list[torch.Tensor] = []
        for row_idx, sched_req in enumerate(requests):
            combined = self._take_next_decode_input_embed(
                sched_req=sched_req,
                device=feedback_buffer.device,
                dtype=feedback_buffer.dtype,
            )
            if combined is None:
                continue
            self._append_decode_input_history(sched_req.data, combined)
            rows.append(row_idx)
            embeds.append(combined)
        if not rows:
            return
        embeds_stacked = torch.stack(embeds, dim=0)
        if len(rows) == batch_size:
            # Note (wenyao): dense steady state: rows is exactly range(batch_size),
            # so slice-assign and skip the per-frame pageable index H2D
            feedback_buffer[:batch_size] = embeds_stacked
            feedback_mask[:batch_size] = True
            return
        rows_t = torch.tensor(rows, dtype=torch.long, device=feedback_buffer.device)
        feedback_buffer[rows_t] = embeds_stacked
        feedback_mask[rows_t] = True

    @staticmethod
    def _data_has_next_decode_input(data: Any) -> bool:
        if data is None:
            return False
        pending_feedback_queue = getattr(data, "pending_feedback_queue", None)
        if not pending_feedback_queue:
            return False
        pending_text_queue = getattr(data, "pending_text_queue", None)
        if pending_text_queue:
            return True
        return bool(
            data.thinker_chunks_done
            and getattr(data, "tts_pad_embed", None) is not None
        )

    def _requests_ready_for_decode(self, requests: list) -> bool:
        return all(
            self._data_has_next_decode_input(sched_req.data) for sched_req in requests
        )

    @staticmethod
    def _pop_left(queue: Any) -> torch.Tensor | None:
        if not queue:
            return None
        if hasattr(queue, "popleft"):
            return queue.popleft()
        if isinstance(queue, list):
            return queue.pop(0)
        return None

    @staticmethod
    def _peek_left(queue: Any) -> torch.Tensor | None:
        if not queue:
            return None
        if isinstance(queue, list):
            return queue[0]
        if hasattr(queue, "__getitem__"):
            return queue[0]
        return None

    @staticmethod
    def _decode_input_history(data: Any) -> list[torch.Tensor]:
        history = getattr(data, "decode_input_embeds", None)
        if history is None:
            history = []
            data.decode_input_embeds = history
        return history

    @staticmethod
    def _append_decode_input_history(data: Any, row: torch.Tensor) -> None:
        QwenTalkerModelRunner._decode_input_history(data).append(row.detach())

    @staticmethod
    def _decode_row(
        row: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        row = row.reshape(-1)
        if row.device != device or row.dtype != dtype:
            raise RuntimeError(
                "Talker decode rows must already match the feedback buffer "
                f"device/dtype, got {row.device}/{row.dtype}, "
                f"expected {device}/{dtype}"
            )
        return row

    @staticmethod
    def _combine_feedback_with_next_text(
        *,
        data: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        pending_feedback_queue = getattr(data, "pending_feedback_queue", None)
        feedback = QwenTalkerModelRunner._peek_left(pending_feedback_queue)
        if feedback is None:
            return None

        combined = QwenTalkerModelRunner._decode_row(
            feedback,
            device=device,
            dtype=dtype,
        )
        next_text = QwenTalkerModelRunner._peek_left(
            getattr(data, "pending_text_queue", None)
        )
        if next_text is None:
            if not data.thinker_chunks_done:
                return None
            next_text = data.tts_pad_embed

        return combined + QwenTalkerModelRunner._decode_row(
            next_text,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _take_next_decode_input_embed(
        *,
        sched_req: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        data = sched_req.data
        combined = QwenTalkerModelRunner._combine_feedback_with_next_text(
            data=data,
            device=device,
            dtype=dtype,
        )
        if combined is None:
            return None

        QwenTalkerModelRunner._pop_left(getattr(data, "pending_feedback_queue", None))
        if getattr(data, "pending_text_queue", None):
            QwenTalkerModelRunner._pop_left(data.pending_text_queue)
        return combined
