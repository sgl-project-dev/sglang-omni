# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker scheduler policy on top of the generic OmniScheduler."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from sglang_omni.models.qwen3_omni.config import MIN_PARTIAL_START_CHUNKS
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.profiler.event_recorder import get_recorder as _get_event_recorder
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)


def configure_talker_server_args(
    server_args: Any,
    *,
    feedback_enabled: bool = True,
) -> bool:
    """Apply talker-specific scheduler/runtime defaults.

    Returns whether CUDA graphs were requested so the caller can capture them
    after the model worker is constructed.
    """

    want_cuda_graph = not bool(server_args.disable_cuda_graph)
    overrides = {
        "disable_radix_cache": True,
        "chunked_prefill_size": 0,
    }
    if feedback_enabled:
        overrides["disable_overlap_schedule"] = True
    override_server_args(server_args, "qwen3_omni.talker", **overrides)
    return want_cuda_graph


class QwenTalkerScheduler(OmniScheduler):
    """Talker scheduler with Qwen-specific request and decode readiness."""

    _chunk_wait_steps: int = 0
    _critical_gate_episode: dict[str, Any] | None = None
    _critical_gate_id: int = 0

    def __init__(
        self,
        *args: Any,
        enable_partial_start: bool = False,
        partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
        im_end_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if partial_start_min_chunks < MIN_PARTIAL_START_CHUNKS:
            raise ValueError(
                f"partial_start_min_chunks must be >= {MIN_PARTIAL_START_CHUNKS}, "
                f"got {partial_start_min_chunks}"
            )
        self._enable_partial_start = bool(enable_partial_start)
        self._partial_start_min_chunks = int(partial_start_min_chunks)
        self._im_end_token_id = im_end_token_id

    def get_new_batch_prefill(self, running_batch):
        plan = super().get_new_batch_prefill(running_batch)
        if plan.batch_to_run is not None and plan.batch_to_run.forward_mode.is_extend():
            histogram = getattr(self, "_prefill_batch_histogram", None)
            if histogram is None:
                histogram = self._prefill_batch_histogram = {}
            size = len(plan.batch_to_run.reqs)
            histogram[size] = histogram.get(size, 0) + 1
        return plan

    def _admin_model_info(self):
        response = super()._admin_model_info()
        response["data"]["talker_prefill_batching"] = {
            "target_requests": self.prefill_coalesce_requests,
            "max_wait_ms": self.prefill_coalesce_wait_s * 1000,
            "when_idle": self.prefill_coalesce_when_idle,
            # Note (wenyao): MessagePack's strict map keys reject integer histogram keys.
            "batch_histogram": {
                str(size): count
                for size, count in getattr(self, "_prefill_batch_histogram", {}).items()
            },
        }
        return response

    def _profile_decode_gate(self, batch: Any, *, blocked: bool) -> None:
        # Note (wenyao): Per-poll profiling repeats the same blocked state,
        # so each wait episode records only its boundary events.
        run_id = _get_event_recorder().active_run_id()
        episode = self._critical_gate_episode
        if episode is not None and episode["run_id"] != run_id:
            self._critical_gate_episode = episode = None
        if blocked:
            if episode is not None:
                return
            request_ids = []
            missing_feedback = []
            missing_text = []
            ready_count = 0
            for req in batch.reqs:
                rid = getattr(req, "rid", None)
                request_ids.append(rid)
                data = getattr(req, "_omni_data", None)
                feedback = getattr(data, "pending_feedback_queue", None)
                text = getattr(data, "pending_text_queue", None)
                feedback_ready = feedback is not None and len(feedback) > 0
                text_ready = (text is not None and len(text) > 0) or (
                    getattr(data, "thinker_chunks_done", False)
                    and getattr(data, "tts_pad_embed", None) is not None
                )
                if not feedback_ready:
                    missing_feedback.append(rid)
                if not text_ready:
                    missing_text.append(rid)
                ready_count += int(feedback_ready and text_ready)
            self._critical_gate_id += 1
            metadata = {
                "gate_id": self._critical_gate_id,
                "participant_request_ids": request_ids,
                "missing_feedback_request_ids": missing_feedback,
                "missing_text_request_ids": missing_text,
                "ready_count": ready_count,
                "batch_size": len(request_ids),
            }
            episode = {
                "run_id": run_id,
                "start_ns": time.perf_counter_ns(),
                "wait_steps": self._chunk_wait_steps,
                "metadata": metadata,
            }
            self._critical_gate_episode = episode
            _emit_event(
                request_id=request_ids[0] if request_ids else "",
                stage=None,
                event_name="talker_decode_gate_blocked",
                metadata=metadata,
            )
        elif episode is not None:
            self._critical_gate_episode = None
            metadata = episode["metadata"]
            _emit_event(
                request_id=(metadata["participant_request_ids"] or [""])[0],
                stage=None,
                event_name="talker_decode_gate_released",
                metadata={
                    **metadata,
                    "elapsed_wall_ns": time.perf_counter_ns() - episode["start_ns"],
                    "deferred_steps": self._chunk_wait_steps - episode["wait_steps"],
                    "releasing_request_ids": [
                        getattr(req, "rid", None) for req in batch.reqs
                    ],
                },
            )

    def _count_usable_prefetched_chunks(self, prefetched: list[Any]) -> int:
        im_end = self._im_end_token_id
        if im_end is None or not prefetched:
            return len(prefetched)
        metadata = getattr(prefetched[-1], "metadata", None) or {}
        token_id = metadata.get("token_id")
        if token_id is not None and int(token_id) == int(im_end):
            return len(prefetched) - 1
        return len(prefetched)

    def _is_request_build_ready(
        self,
        payload: Any,
        *,
        pending_stream_done: bool,
    ) -> bool:
        if pending_stream_done:
            return True
        if not self._enable_partial_start:
            return False
        prefetched = getattr(payload, "prefetched_chunks", None) or []
        return (
            self._count_usable_prefetched_chunks(prefetched)
            >= self._partial_start_min_chunks
        )

    def _initialize_request_stream_state(self, req_data: Any, payload: Any) -> None:
        del req_data, payload
        return None

    def _should_recheck_deferred_request_on_stream_chunk(
        self, request_id: str, chunk: Any
    ) -> bool:
        del request_id, chunk
        return self._enable_partial_start

    def _is_batch_ready_to_run(self, batch: Any) -> bool:
        if (
            batch is not None
            and batch.forward_mode.is_decode()
            and self._model_runner is not None
            and hasattr(self._model_runner, "is_decode_batch_ready")
        ):
            ready = self._model_runner.is_decode_batch_ready(batch)
            if _get_event_recorder().is_active():
                self._profile_decode_gate(batch, blocked=not ready)
            if not ready:
                self._chunk_wait_steps += 1
                logger.debug(
                    "Deferring decode batch until talker feedback/text input is ready"
                )
                return False
        return True

    def get_next_batch_to_run(self) -> Any | None:
        batch = super().get_next_batch_to_run()
        if batch is not None and not self._is_batch_ready_to_run(batch):
            self._rollback_decode_prep_after_skip(batch)
            return None
        return batch

    def _rollback_decode_prep_after_skip(self, batch: Any) -> None:
        # Note(Chenchen Hong, Xuesong): This is talker-only. It does not fully
        # invert prepare_for_decode; talker disables overlap/spec/Mamba/hisparse,
        # and the penalizer's cumulate scatter_ is idempotent under the talker's
        # own SamplingBatchInfo. Zero the req_to_token_pool cell that
        # alloc_for_decode wrote at (req_pool_indices, pre-increment seq_lens);
        # seq_lens_sum stays untouched (always None after prepare_for_decode,
        # recomputed at the next forward).
        if not batch.forward_mode.is_decode():
            return
        if batch.out_cache_loc is not None:
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            batch.out_cache_loc = None
        for req in batch.reqs:
            req.decode_batch_idx -= 1
            req.kv_committed_len -= 1
            req.kv.kv_allocated_len -= 1
        batch.seq_lens.sub_(1)
        batch.seq_lens_cpu.sub_(1)
        batch.orig_seq_lens.sub_(1)
        batch.req_to_token_pool.req_to_token[batch.req_pool_indices, batch.seq_lens] = 0

    def self_check_during_idle(self) -> None:
        if self.running_batch is not None and not self.running_batch.is_empty():
            return
        if self.waiting_queue:
            return
        super().self_check_during_idle()

    @staticmethod
    def _append_stream_chunk_default(req_data: Any, chunk: Any) -> None:
        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if pending_text_queue is None:
            pending_text_queue = deque()
            req_data.pending_text_queue = pending_text_queue
        pending_text_queue.append(getattr(chunk, "data", chunk))

    def _mark_stream_done(self, req_data: Any) -> None:
        if self._stream_done_handler is None:
            req_data.thinker_chunks_done = True
            return
        self._stream_done_handler(req_data)
