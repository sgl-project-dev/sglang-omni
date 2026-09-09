# SPDX-License-Identifier: Apache-2.0
"""Omni ModelRunner that drives the dots.tts MLX latent engine.

The scheduler keeps its ordinary token loop (one decode step == one audio
patch); this runner owns the MLX flow state per request and converts patches
back to Torch at the emission boundary so the omni request-data plumbing
(latent_patches / latest_latent_patch streaming, apply_latent_result)
and the Torch-MPS vocoder stage see the usual lower-bound tensors.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import numpy as np
import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.dots_tts.mlx import flow as f
from sglang_omni.models.dots_tts.mlx.model import DotsTTSFlowState
from sglang_omni.models.dots_tts.request_builders import DotsTTSSGLangRequestData


def _to_mx(tensor: Optional[torch.Tensor]) -> Optional[mx.array]:
    if tensor is None:
        return None
    return mx.array(tensor.detach().cpu().numpy())


def _to_torch(array: mx.array, device: torch.device) -> torch.Tensor:
    mx.eval(array)
    return torch.as_tensor(np.asarray(array), device=device).float()


class DotsTTSMlxModelRunner(ModelRunner):
    """Token loop over the MLX latent engine with omni data plumbing.

    MLX streams are thread-local, while the scheduler runs decode on its own
    thread (same pattern as the other MLX engine runners): a thread-local
    stream is created here and every engine step runs under its context, so
    lazy builds and eager readbacks resolve on the scheduler thread.
    """

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        *,
        checkpoint_dir: str,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self._checkpoint_dir = checkpoint_dir
        self._flow: dict[str, DotsTTSFlowState] = {}
        self._model: Any = None
        self._mlx_stream = mx.new_thread_local_stream(mx.gpu)

    def lookahead_eligible(self, batch: Any) -> bool:
        del batch
        return False

    def _mlx_model(self) -> Any:
        if self._model is None:
            # note (guozhihao-224): build lazily on the scheduler thread under
            # the stream context; cross-thread array evaluation has no stream.
            from sglang_omni.models.dots_tts.mlx.checkpoint import load_dots_mlx_model

            self._model = load_dots_mlx_model(self._checkpoint_dir)
        return self._model

    @staticmethod
    def _one_request(requests: list) -> Any:
        if len(requests) != 1:
            raise RuntimeError("dots.tts MLX currently requires max_running_requests=1")
        return requests[0]

    def _result(self, data: DotsTTSSGLangRequestData, device: torch.device) -> Any:
        from sglang.srt.managers.utils import GenerationBatchResult

        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor(
                [data.control_token_id], dtype=torch.long, device=device
            ),
            can_run_cuda_graph=False,
        )

    @torch.inference_mode()
    def custom_prefill_forward(self, forward_batch, schedule_batch, requests) -> Any:
        del forward_batch, schedule_batch
        with mx.stream(self._mlx_stream):
            return self._prefill_impl(requests)

    def _prefill_impl(self, requests: list) -> Any:
        request = self._one_request(requests)
        data: DotsTTSSGLangRequestData = request.data
        state_data = data.state
        model = self._mlx_model()

        state, prompt_embeddings = f.start_request(
            model,
            prompt_latents=_to_mx(state_data.prompt_latents),
            speaker_embedding=_to_mx(state_data.speaker_embedding),
            speaker_scale=float(state_data.speaker_scale),
            seed=state_data.seed,
        )
        self._flow[request.request_id] = state

        prefill_ids = mx.array(data.generation_schedule[0, : data.prefill_end].numpy())
        prompt_span_positions = mx.array(data.prompt_span_positions.numpy())
        if prompt_embeddings is None:
            embeds = model.backbone.embed_tokens(prefill_ids)[None, :, :]
        else:
            embeds = f.build_prefill_embeds(
                model,
                prefill_ids,
                prompt_span_positions=prompt_span_positions,
                prompt_embeddings=prompt_embeddings,
            )
        hidden = model(embeds, cache=state.backbone_cache)
        f.initialize_history(
            model,
            state,
            hidden_states=hidden,
            prompt_span_positions=prompt_span_positions,
            generation_schedule=mx.array(data.generation_schedule[0].numpy()),
            audio_span_token_ids=set(state_data.audio_span_token_ids),
            prefill_end=int(data.prefill_end),
            decoded_latent_patches=[
                _to_mx(patch) for patch in data.decoded_latent_patches
            ],
        )
        # note (guozhihao-224): the last prefill hidden row is already in the
        # history; the first flow step appends none (torch runner parity).
        self._run_flow_step(
            request, data, state, hidden[:, -1:, :], append_hidden=False
        )
        return self._result(data, self.device)

    @torch.inference_mode()
    def custom_decode_forward(self, forward_batch, schedule_batch, requests) -> Any:
        del forward_batch, schedule_batch
        with mx.stream(self._mlx_stream):
            return self._decode_impl(requests)

    def _decode_impl(self, requests: list) -> Any:
        request = self._one_request(requests)
        data: DotsTTSSGLangRequestData = request.data
        state = self._flow.get(request.request_id)
        if state is None:
            raise RuntimeError(
                f"dots.tts MLX decode has no flow state for {request.request_id}"
            )
        if state.next_feedback is None:
            raise RuntimeError("dots.tts MLX decode has no feedback embedding")
        hidden = self._mlx_model()(state.next_feedback, cache=state.backbone_cache)
        self._run_flow_step(request, data, state, hidden[:, -1:, :])
        return self._result(data, self.device)

    def _run_flow_step(
        self,
        request: Any,
        data: DotsTTSSGLangRequestData,
        state: DotsTTSFlowState,
        hidden_last: mx.array,
        *,
        append_hidden: bool = True,
    ) -> None:
        state_data = data.state
        latent, feedback, finished, emit = f.decode_step(
            self._mlx_model(),
            state,
            hidden_last=hidden_last,
            num_steps=int(state_data.num_steps),
            ode_method=str(state_data.ode_method),
            guidance_scale=float(state_data.guidance_scale),
            eos_threshold=float(state_data.eos_threshold),
            append_hidden=append_hidden,
        )
        state.next_feedback = feedback
        if emit:
            torch_latent = _to_torch(latent, self.device)
            data.decoded_latent_patches.append(torch_latent)
            data.latent_patches.append(torch_latent)
            if bool(state_data.stream):
                data.latest_latent_patch = torch_latent
        if finished and request.data.req.finished_reason is None:
            from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

            request.data.req.finished_reason = FINISH_MATCHED_TOKEN(
                data.control_token_id
            )

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        del req_data
        self._flow.pop(request_id, None)

    def reset_request(self, request_id: str) -> None:
        self._flow.pop(request_id, None)


__all__ = ["DotsTTSMlxModelRunner"]
