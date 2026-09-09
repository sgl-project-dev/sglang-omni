# SPDX-License-Identifier: Apache-2.0
"""AuK conditioning encoder: frozen Qwen2.5-Omni-3B Thinker.

Turns a chat-style message list (instruction + optional reference audio) into
the packed all-layer hidden states the sampler fuses. The Thinker runs frozen
with ``output_hidden_states=True``; the unused vision tower is dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from sglang_omni.models.auk.constants import (
    NO_PROMPT_AUDIO_MARKER,
    QWEN_AUDIO_SAMPLE_RATE,
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


def resample_to_qwen_rate(waveform: np.ndarray, source_sample_rate: int) -> np.ndarray:
    """Resample mono float audio to the 16 kHz rate Qwen2.5-Omni expects."""
    if source_sample_rate == QWEN_AUDIO_SAMPLE_RATE:
        return np.ascontiguousarray(waveform, dtype=np.float32)
    import torchaudio

    tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
    resampled = torchaudio.transforms.Resample(
        source_sample_rate, QWEN_AUDIO_SAMPLE_RATE
    )(tensor)
    return np.ascontiguousarray(resampled.squeeze(0).numpy(), dtype=np.float32)


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
        self.model = model.to(self.device)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.model.config.text_config.num_hidden_layers)

    def _apply_chat_template(
        self, messages_batch: Sequence[list[dict[str, Any]]]
    ) -> list[str]:
        return self.processor.apply_chat_template(
            messages_batch, tokenize=False, add_generation_prompt=True
        )

    def _process_one(
        self, messages: list[dict[str, Any]], audio: np.ndarray | None
    ) -> dict[str, torch.Tensor]:
        formatted = self._apply_chat_template([messages])
        kwargs: dict[str, Any] = {"text": formatted, "return_tensors": "pt"}
        if audio is not None:
            kwargs["audio"] = [audio]
        return self.processor(**kwargs)

    def _process_batch(
        self,
        messages_batch: Sequence[list[dict[str, Any]]],
        audios: Sequence[np.ndarray | None],
    ) -> dict[str, torch.Tensor]:
        formatted = self._apply_chat_template(list(messages_batch))
        kwargs: dict[str, Any] = {
            "text": formatted,
            "padding": True,
            "return_tensors": "pt",
        }
        has_audio = [audio is not None for audio in audios]
        if all(has_audio):
            kwargs["audio"] = list(audios)
        elif not any(has_audio):
            return self.processor(**kwargs)
        else:
            # Mixed audio/no-audio batches are not supported by one processor
            # call; fall back to encoding each request on its own and padding.
            return {}
        return self.processor(**kwargs)

    @torch.no_grad()
    def encode(
        self,
        messages_batch: Sequence[list[dict[str, Any]]],
        audios: Sequence[np.ndarray | None],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return ``(hidden_states, masks)`` per request.

        ``hidden_states[i]`` is ``[L, Nt, H]`` — every Thinker layer (index 0 is
        the embedding output, dropped later by the layer fusion) for the
        unpadded token sequence. ``masks[i]`` is a bool ``[Nt]`` attention mask.
        """
        if not messages_batch:
            return [], []
        if len(messages_batch) != len(audios):
            raise ValueError("messages_batch and audios must have the same length")

        inputs = self._process_batch(messages_batch, audios)
        if not inputs:
            return self._encode_sequentially(messages_batch, audios)

        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        outputs = self.model(**inputs, output_hidden_states=True)
        stacked = torch.stack(outputs.hidden_states, dim=1)  # [B, L, Nt, H]
        masks = inputs["attention_mask"].bool()

        return self._unbind(stacked, masks)

    def _encode_sequentially(
        self,
        messages_batch: Sequence[list[dict[str, Any]]],
        audios: Sequence[np.ndarray | None],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        result_hidden: list[torch.Tensor] = []
        result_masks: list[torch.Tensor] = []
        for messages, audio in zip(messages_batch, audios):
            inputs = self._process_one(messages, audio)
            inputs = {
                k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)
            }
            outputs = self.model(**inputs, output_hidden_states=True)
            stacked = torch.stack(outputs.hidden_states, dim=1)[0]  # [L, Nt, H]
            mask = inputs["attention_mask"].bool()[0]
            length = int(mask.sum().item())
            result_hidden.append(stacked[:, :length].contiguous())
            result_masks.append(mask[:length].contiguous())
        return result_hidden, result_masks

    @staticmethod
    def _unbind(
        stacked: torch.Tensor, masks: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        hidden: list[torch.Tensor] = []
        out_masks: list[torch.Tensor] = []
        for index in range(stacked.shape[0]):
            mask = masks[index]
            length = int(mask.sum().item())
            # Left padding would shift the valid window; AuK's processor pads right.
            hidden.append(stacked[index][:, :length].contiguous())
            out_masks.append(mask[:length].contiguous())
        return hidden, out_masks
