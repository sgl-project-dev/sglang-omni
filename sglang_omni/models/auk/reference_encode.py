# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Tencent. All rights reserved.
# Derived from Tencent-Hunyuan/AuK; see LICENSE for the MIT permission notice.
"""AuK conditioning encoder: frozen Qwen2.5-Omni-3B Thinker.

Turns a chat-style message list (instruction + optional reference audio) into
the packed all-layer hidden states the sampler fuses. The Thinker runs frozen
with ``output_hidden_states=True``; the unused vision tower is dropped.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from sglang_omni.models.auk.constants import (
    NO_PROMPT_AUDIO_MARKER,
)

logger = logging.getLogger(__name__)


def build_messages(instruction: str, has_reference_audio: bool) -> list[dict[str, Any]]:
    """Build the single-turn ChatML message list AuK is trained on.

    The reference-free (instruction-only TTS) path appends the
    ``|<no_prompt_audio>|`` marker, matching the released inference code.
    """
    text = instruction
    if not has_reference_audio and not text.endswith(NO_PROMPT_AUDIO_MARKER):
        text = text + NO_PROMPT_AUDIO_MARKER

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if has_reference_audio:
        # The placeholder is what the chat template expands to the audio tokens;
        # the waveform itself is passed through the processor's ``audio=`` arg.
        content.append({"type": "audio", "audio": None})
    return [{"role": "user", "content": content}]


class AuKConditionEncoder:
    """Frozen Qwen2.5-Omni Thinker used as AuK's instruction/reference encoder."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype

        logger.info("AuK: loading Qwen2.5-Omni encoder from %s", model_path)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype
        )
        # Keep the multimodal Thinker (text + audio); drop the unused vision tower.
        visual = getattr(model, "visual", None)
        if visual is not None:
            del model.visual
            model.visual = None
        model.requires_grad_(False)
        model.eval()
        # AukInfer casts the enclosing CFM (including Qwen) to FP32 and uses
        # autocast for conditioning and sampling.
        self.model = model.to(device=self.device, dtype=torch.float32)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.model.config.text_config.num_hidden_layers)

    def _process_one(
        self, messages: list[dict[str, Any]], audio: np.ndarray | None
    ) -> dict[str, torch.Tensor]:
        formatted = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        kwargs: dict[str, Any] = {
            "text": formatted,
            "padding": True,
            "return_tensors": "pt",
        }
        if audio is not None:
            kwargs["audio"] = [audio]
        return self.processor(**kwargs)

    @torch.no_grad()
    def encode(
        self,
        messages: list[dict[str, Any]],
        audio: np.ndarray | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one request into all-layer ``[L, Nt, H]`` states and its mask."""
        inputs = self._process_one(messages, audio)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        outputs = self.model(**inputs, output_hidden_states=True)
        return (
            torch.stack(outputs.hidden_states, dim=1)[0],
            inputs["attention_mask"][0].bool(),
        )
