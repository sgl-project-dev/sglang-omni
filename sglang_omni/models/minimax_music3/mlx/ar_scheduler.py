# SPDX-License-Identifier: Apache-2.0
"""Whole-request MLX AR scheduler with the CUDA pipeline's stream contract."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoTokenizer

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

from ..chunking import chunk_windows
from ..payload_types import MiniMaxMusic3State
from ..prompt import validate_tokenizer_ids
from ..serial_offload import get_coordinator
from .ar import generate_frame_hiddens
from .loader import (
    MiniMaxMusic3MlxARModel,
    load_mlx_ar_model,
    resolve_mlx_artifact,
)

logger = logging.getLogger(__name__)


def _build_text_pair(
    prompt: str,
    model: MiniMaxMusic3MlxARModel,
    tokenizer: object,
) -> mx.array:
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    if input_ids.shape[1] > 5_000:
        raise ValueError(
            f"MiniMax Music 3 prompt has {input_ids.shape[1]} tokens; "
            "the maximum is 5000"
        )
    conditional = mx.array(input_ids.astype("int32"))
    unconditional = conditional
    if unconditional.shape[1] > 3:
        middle = mx.full(
            (1, unconditional.shape[1] - 3),
            model.config.audio_cfg_token_id,
            dtype=mx.int32,
        )
        unconditional = mx.concatenate(
            [unconditional[:, :1], middle, unconditional[:, -2:]], axis=1
        )
    return mx.concatenate([conditional, unconditional], axis=0)


class MiniMaxMusic3MlxARScheduler(SimpleScheduler):
    """Generate MLX frame hiddens and stream transport-safe CPU chunks."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        serial_offload: bool = False,
    ) -> None:
        self._serial_offload = bool(serial_offload)
        self._artifact = resolve_mlx_artifact(model_path, revision)
        self.model: MiniMaxMusic3MlxARModel | None = None
        if not self._serial_offload:
            self.model = load_mlx_ar_model(
                model_path,
                revision,
                artifact=self._artifact,
            )
        tokenizer_dir = Path(self._artifact[0]) / "tokenizer"
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(
                "MiniMax Music 3 MLX artifact must include its tokenizer directory"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            trust_remote_code=False,
        )
        validate_tokenizer_ids(self.tokenizer)
        self._abort_events: dict[str, threading.Event] = {}
        self._events_lock = threading.Lock()
        self._mlx_thread_stream = mx.new_thread_local_stream(mx.gpu)
        if self._serial_offload:
            get_coordinator().enable()
        super().__init__(self._generate, max_concurrency=1)

    def _abort_event(self, request_id: str) -> threading.Event:
        with self._events_lock:
            return self._abort_events.setdefault(request_id, threading.Event())

    def _generate(self, payload: StagePayload) -> StagePayload:
        state = MiniMaxMusic3State.from_dict(payload.data)
        if state.prompt is None:
            raise ValueError("MiniMax Music 3 preprocessing did not build a prompt")
        abort_event = self._abort_event(payload.request_id)
        started = time.perf_counter()
        coordinator = get_coordinator()
        handed_off = False
        try:
            if self._serial_offload:
                coordinator.acquire_ar(
                    payload.request_id,
                    should_abort=abort_event.is_set,
                )
                self._load_model()
            if self.model is None:
                raise RuntimeError("MiniMax Music 3 MLX AR model is not loaded")
            pending: list[OutgoingMessage] = []
            with mx.stream(self._mlx_thread_stream):
                text_ids = _build_text_pair(state.prompt, self.model, self.tokenizer)
                hidden = generate_frame_hiddens(
                    self.model.language_model,
                    self.model.rvq_depth_decoder,
                    self.model.config,
                    text_ids,
                    max_frames=state.max_audio_frames,
                    seed=state.seed,
                    should_abort=abort_event.is_set,
                )
                mx.eval(hidden)
                generated_frames = int(hidden.shape[1])
                for window in chunk_windows(generated_frames):
                    if abort_event.is_set():
                        raise InterruptedError("MiniMax Music 3 MLX generation aborted")
                    chunk = hidden[:, window.start : window.end].astype(mx.float16)
                    mx.eval(chunk)
                    chunk_np = np.asarray(chunk, dtype=np.float16)
                    if self._serial_offload:
                        chunk_np = np.array(chunk_np, copy=True, order="C")
                    else:
                        chunk_np = np.ascontiguousarray(chunk_np)
                    transport = torch.from_numpy(chunk_np)
                    pending.append(
                        OutgoingMessage(
                            request_id=payload.request_id,
                            type="stream",
                            data=transport,
                            metadata={
                                "stream": True,
                                "modality": "ttm_hidden",
                                "chunk_idx": window.index,
                                "start_frame": window.start,
                                "end_frame": window.end,
                                "is_final": window.is_last,
                                "seed": state.seed,
                            },
                        )
                    )
                    del chunk, chunk_np, transport
                del hidden, text_ids

            state.generated_frames = generated_frames
            state.finish_reason = (
                "length" if generated_frames >= state.max_audio_frames else "stop"
            )
            state.prompt = None
            state.caption = ""
            state.lyrics = ""
            result = store_state(payload, state)
            if self._serial_offload:
                self._release_model()
                coordinator.begin_dit_handoff(payload.request_id)
                handed_off = True
            for message in pending:
                self.outbox.put(message)
            logger.info(
                "MiniMax Music 3 MLX AR done request=%s frames=%d elapsed=%.1fs",
                payload.request_id,
                generated_frames,
                time.perf_counter() - started,
            )
            return result
        except BaseException:
            if self._serial_offload and not handed_off:
                try:
                    self._release_model()
                except BaseException as cleanup_error:
                    coordinator.fail_closed(cleanup_error)
                    raise
                coordinator.cancel_ar(payload.request_id)
            raise
        finally:
            with self._events_lock:
                self._abort_events.pop(payload.request_id, None)

    def _load_model(self) -> None:
        if self.model is not None:
            return
        with mx.stream(self._mlx_thread_stream):
            self.model = load_mlx_ar_model(
                str(self._artifact[0]),
                artifact=self._artifact,
            )
        logger.info(
            "MiniMax Music 3 MLX AR loaded active=%.2fGiB cache=%.2fGiB",
            mx.get_active_memory() / 1024**3,
            mx.get_cache_memory() / 1024**3,
        )

    def _release_model(self) -> None:
        mx.synchronize(self._mlx_thread_stream)
        self.model = None
        logger.info(
            "MiniMax Music 3 MLX AR released active=%.2fGiB cache=%.2fGiB",
            mx.get_active_memory() / 1024**3,
            mx.get_cache_memory() / 1024**3,
        )

    def abort(self, request_id: str) -> None:
        with self._events_lock:
            event = self._abort_events.get(request_id)
            if event is not None:
                event.set()
        super().abort(request_id)


__all__ = ["MiniMaxMusic3MlxARScheduler", "_build_text_pair"]
