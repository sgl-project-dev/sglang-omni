# SPDX-License-Identifier: Apache-2.0
"""Torch/MPS runner for dots.tts.

SGLang's eager extend path hardcodes a bfloat16() cast on projected
input embeddings (ModelRunner._extend_forward_kwargs) that mirrors the
BF16 models it targets; on MPS with the float32 correctness profile it turns
every QKV linear into a mixed-dtype matmul, which Metal aborts. Following the
Qwen3-ASR Torch/MPS pattern, the MPS runner owns the backbone forward instead
of SGLang's: it swaps the SGLang Qwen2 for the pinned Hugging Face
implementation (self-contained past_key_values, no SGLang KV pool or
attention-backend state) and runs prefill/decode through the same
before_* / post_* hooks as the CUDA path, so the flow head is shared.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner

logger = logging.getLogger(__name__)


def install_torch_mps_backbone(model: Any, checkpoint_dir: str) -> None:
    """Replace SGLang's Qwen2 with the pinned HF Torch implementation.

    MPS serves the single-request float32 profile; HF's Qwen2Model keeps its
    own past_key_values and never touches SGLang's KV pool or eager
    forward path. The flow head is left as SGLang loaded it (same device and
    dtype), matching the Qwen3-ASR install_torch_mps_language_model split.
    """
    from safetensors import safe_open
    from transformers import Qwen2Config, Qwen2Model

    root = Path(checkpoint_dir)
    config = Qwen2Config(**json.loads((root / "llm_config.json").read_text()))
    old = model.qwen2
    device = next(old.parameters()).device
    dtype = next(old.parameters()).dtype

    # note (guozhihao-224): build on CPU, not meta — Qwen2 rotary buffers are
    # non-persistent, so a meta module would keep them meta and .to would fail.
    backbone = Qwen2Model(config)
    expected = set(backbone.state_dict())
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(root / "model.safetensors", framework="pt") as handle:
        for name in handle.keys():
            if name.startswith("llm.model."):
                key = name.removeprefix("llm.model.")
                if key in expected:
                    state_dict[key] = handle.get_tensor(name)
    if not state_dict:
        raise RuntimeError("dots.tts checkpoint has no llm.model.* backbone weights")
    missing = expected - state_dict.keys()
    if missing:
        raise ValueError(
            "dots.tts Torch MPS backbone checkpoint is incomplete: "
            f"{sorted(missing)[:10]}"
        )
    backbone.load_state_dict(state_dict, strict=True, assign=True)

    del model.qwen2
    gc.collect()
    torch.mps.empty_cache()
    model.qwen2 = backbone.eval().to(device=device, dtype=dtype)
    logger.info("Installed dots.tts HF Qwen2 backbone on %s (%s)", device, dtype)


class DotsTTSTorchMpsModelRunner(DotsTTSModelRunner):
    """Shared flow-head recurrence with an HF Qwen2 owned backbone forward."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._past_key_values: dict[str, Any] = {}

    def lookahead_eligible(self, batch: Any) -> bool:
        del batch
        return False

    @staticmethod
    def _one_request(requests: list[Any]) -> Any:
        if len(requests) != 1:
            raise RuntimeError(
                "dots.tts Torch MPS currently requires max_running_requests=1"
            )
        return requests[0]

    def _next_result(self, hidden: torch.Tensor) -> Any:
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.utils import GenerationBatchResult

        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=None,
                hidden_states=hidden,
            ),
            next_token_ids=None,
            can_run_cuda_graph=False,
        )

    @torch.inference_mode()
    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        del schedule_batch
        request = self._one_request(requests)
        embeds = forward_batch.input_embeds
        if embeds is None:
            raise RuntimeError("dots.tts Torch MPS prefill has no input embeddings")
        output = self.model.qwen2(
            inputs_embeds=embeds.unsqueeze(0),
            use_cache=True,
        )
        self._past_key_values[request.request_id] = output.past_key_values
        return self._next_result(output.last_hidden_state)

    @torch.inference_mode()
    def custom_decode_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        del schedule_batch
        request = self._one_request(requests)
        embeds = forward_batch.input_embeds
        if embeds is None:
            raise RuntimeError("dots.tts Torch MPS decode has no input embeddings")
        try:
            past_key_values = self._past_key_values[request.request_id]
        except KeyError as exc:
            raise RuntimeError(
                f"dots.tts Torch MPS decode has no cache for {request.request_id}"
            ) from exc
        output = self.model.qwen2(
            inputs_embeds=embeds.unsqueeze(1),
            past_key_values=past_key_values,
            use_cache=True,
        )
        self._past_key_values[request.request_id] = output.past_key_values
        return self._next_result(output.last_hidden_state)

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        self._past_key_values.pop(request_id, None)
        super().on_request_finished(request_id, req_data)

    def reset_request(self, request_id: str) -> None:
        self._past_key_values.pop(request_id, None)
        super().reset_request(request_id)


__all__ = ["DotsTTSTorchMpsModelRunner", "install_torch_mps_backbone"]
