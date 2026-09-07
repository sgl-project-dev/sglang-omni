# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

TENSOR_IMPORT_ERROR = "torch is not installed"
RUNNER_IMPORT_ERROR = "tensor dependencies are not installed"
torch = None
if importlib.util.find_spec("torch") is not None:
    try:
        import torch

        from sglang_omni.models.qwen3_omni.components import talker_prefill
        from sglang_omni.models.qwen3_omni.pending_text_queue import (
            PendingTextTensorQueue,
        )

        TENSOR_IMPORT_ERROR = ""
    except ModuleNotFoundError as exc:
        TENSOR_IMPORT_ERROR = str(exc)
if not TENSOR_IMPORT_ERROR:
    try:
        from sglang_omni.models.qwen3_omni.talker_model_runner import (
            QwenTalkerModelRunner,
        )

        RUNNER_IMPORT_ERROR = ""
    except ModuleNotFoundError as exc:
        RUNNER_IMPORT_ERROR = str(exc)


@unittest.skipIf(bool(TENSOR_IMPORT_ERROR), TENSOR_IMPORT_ERROR)
class AssistantProjectionCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.device = "cpu"
        self.dtype = torch.float32
        self.lookup_calls = 0
        self.patch = patch.object(
            talker_prefill, "load_thinker_embedding_rows", self.load_rows
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def load_rows(self, path, ids):
        del path
        self.lookup_calls += 1
        return torch.tensor(
            [[i, i + 1, i + 2, i + 3] for i in ids], dtype=torch.float32
        )

    def builder(self, size=0):
        class Projection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.arange(12, dtype=torch.float32).view(3, 4) / 17
                )
                self.bias = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
                self.calls = 0

            def forward(self, value):
                self.calls += 1
                return torch.nn.functional.silu(
                    torch.nn.functional.linear(value, self.weight, self.bias)
                )

        projection = Projection().to(device=self.device, dtype=self.dtype)
        model = SimpleNamespace(
            text_projection=projection,
            model=SimpleNamespace(
                codec_embedding=SimpleNamespace(weight=projection.weight)
            ),
            activation_dtype=self.dtype,
            _text_projection_cache_epoch=1,
        )
        token_names = (
            "tts_bos_token_id",
            "tts_eos_token_id",
            "tts_pad_token_id",
            "im_start_token_id",
            "im_end_token_id",
            "system_token_id",
            "user_token_id",
            "assistant_token_id",
            "codec_bos_id",
            "codec_nothink_id",
            "codec_think_bos_id",
            "codec_think_eos_id",
            "codec_pad_id",
        )
        return talker_prefill.TalkerPrefillBuilder(
            model=model,
            model_path=self.tmp.name,
            audio_token_id=None,
            image_token_id=None,
            video_token_id=None,
            assistant_projection_cache_size=size,
            **dict.fromkeys(token_names, 31),
        )

    @staticmethod
    def chunk(token_id):
        return SimpleNamespace(metadata={"token_id": token_id}, data=None)

    def test_default_off_recomputes_exact_result(self):
        b = self.builder()
        first = b.project_assistant_chunk(self.chunk(2))
        second = b.project_assistant_chunk(self.chunk(2))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(b._model.text_projection.calls, 2)
        self.assertEqual(len(b._assistant_projection_cache), 0)

    def test_hit_skips_both_embedding_lookup_path_and_projection(self):
        b = self.builder(2)
        with patch.object(
            b, "_load_prompt_token_embeddings", wraps=b._load_prompt_token_embeddings
        ) as load:
            first = b.project_assistant_chunk(self.chunk(2))
            second = b.project_assistant_chunk(self.chunk(2))
            self.assertEqual(load.call_count, 1)
        self.assertIs(first, second)
        self.assertFalse(second.requires_grad)
        self.assertEqual(b._model.text_projection.calls, 1)
        baseline = self.builder().project_assistant_chunk(self.chunk(2))
        self.assertTrue(torch.equal(second, baseline))

    def test_lru_is_bounded_and_eviction_keeps_queued_rows_alive(self):
        b = self.builder(2)
        saved = b.project_assistant_chunk(self.chunk(1))
        expected = saved.clone()
        queue = PendingTextTensorQueue.from_tensor(saved)
        for token in (2, 1, 3):
            b.project_assistant_chunk(self.chunk(token))
        self.assertEqual(list(b._assistant_projection_cache), [1, 3])
        b.project_assistant_chunk(self.chunk(4))
        self.assertEqual(list(b._assistant_projection_cache), [3, 4])
        self.assertTrue(torch.equal(queue.popleft(), expected))
        self.assertFalse(queue)

    def test_hidden_tensor_fallback_never_caches(self):
        b = self.builder(2)
        x = torch.arange(4, device=self.device, dtype=self.dtype)
        first = b.project_assistant_chunk(SimpleNamespace(metadata={}, data=x))
        second = b.project_assistant_chunk(SimpleNamespace(metadata={}, data=x + 1))
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(b._model.text_projection.calls, 2)
        self.assertFalse(b._assistant_projection_cache)

    def test_weight_version_and_loader_epoch_each_invalidate(self):
        b = self.builder(2)
        first = b.project_assistant_chunk(self.chunk(2))
        with torch.no_grad():
            b._model.text_projection.bias.add_(1)
        second = b.project_assistant_chunk(self.chunk(2))
        self.assertFalse(torch.equal(first, second))
        # Note (wenyao): .data-based loaders change weights without advancing
        # tensor._version, so cache invalidation also needs the model epoch.
        b._model.text_projection.bias.data.add_(1)
        b._model._text_projection_cache_epoch += 1
        third = b.project_assistant_chunk(self.chunk(2))
        self.assertFalse(torch.equal(second, third))
        self.assertEqual(b._model.text_projection.calls, 3)

    def test_projection_replacement_and_autocast_invalidate(self):
        b = self.builder(2)
        b.project_assistant_chunk(self.chunk(2))
        replacement = self.builder()._model.text_projection
        b._model.text_projection = replacement
        plain = b.project_assistant_chunk(self.chunk(2))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            cast = b.project_assistant_chunk(self.chunk(2))
            self.assertIs(cast, b.project_assistant_chunk(self.chunk(2)))
        self.assertEqual(replacement.calls, 2)
        self.assertEqual(plain.dtype, torch.float32)
        self.assertEqual(cast.dtype, torch.bfloat16)

    def test_inference_and_grad_contexts_do_not_share_entries(self):
        b = self.builder(2)
        b.project_assistant_chunk(self.chunk(2))
        with torch.no_grad():
            b.project_assistant_chunk(self.chunk(2))
        with torch.inference_mode():
            first = b.project_assistant_chunk(self.chunk(2))
            self.assertIs(first, b.project_assistant_chunk(self.chunk(2)))
        self.assertEqual(b._model.text_projection.calls, 3)

    def test_second_thread_bypasses_without_touching_owner_entries(self):
        b = self.builder(2)
        owner = b.project_assistant_chunk(self.chunk(2))
        result = []
        worker = threading.Thread(
            target=lambda: result.append(b.project_assistant_chunk(self.chunk(2)))
        )
        worker.start()
        worker.join()
        self.assertEqual(len(result), 1)
        self.assertIsNot(owner, result[0])
        self.assertTrue(torch.equal(owner, result[0]))
        self.assertIs(owner, b.project_assistant_chunk(self.chunk(2)))

    @unittest.skipIf(bool(RUNNER_IMPORT_ERROR), RUNNER_IMPORT_ERROR)
    def test_real_feedback_consumer_does_not_mutate_shared_cached_row(self):
        b = self.builder(2)
        row = b.project_assistant_chunk(self.chunk(2))
        expected = row.clone()
        for value in (3, 7):
            data = SimpleNamespace(
                pending_feedback_queue=[torch.full_like(row, value)],
                pending_text_queue=PendingTextTensorQueue.from_tensor(row),
                thinker_chunks_done=False,
            )
            combined = QwenTalkerModelRunner._take_next_decode_input_embed(
                sched_req=SimpleNamespace(data=data), device=row.device, dtype=row.dtype
            )
            self.assertTrue(torch.equal(combined, expected + value))
            self.assertFalse(data.pending_text_queue)
            self.assertFalse(data.pending_feedback_queue)
            self.assertTrue(torch.equal(row, expected))

    def test_invalid_capacity_is_rejected(self):
        for size in (-1, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.builder(size)


@pytest.mark.accelerator
class AssistantProjectionCacheCudaTests(AssistantProjectionCacheTests):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.device = "cuda:0"
        self.dtype = torch.bfloat16

    def test_projection_replacement_and_autocast_invalidate(self):
        self.skipTest("CPU autocast case is covered by the CPU class")

    def test_second_thread_bypasses_without_touching_owner_entries(self):
        self.skipTest("CPU thread case is covered by the CPU class")

    def test_stream_change_recomputes_and_lru_eviction_keeps_output_alive(self):
        b = self.builder(2)
        producer = torch.cuda.Stream()
        alternate = torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            first = b.project_assistant_chunk(self.chunk(2))
            self.assertIs(first, b.project_assistant_chunk(self.chunk(2)))
        # Note (wenyao): The raw embedding cache predates projected-token caching
        # and still needs the producer stream to finish before reuse.
        alternate.wait_stream(producer)
        with torch.cuda.stream(alternate):
            second = b.project_assistant_chunk(self.chunk(2))
            retained = second.clone()
            b.project_assistant_chunk(self.chunk(3))
            b.project_assistant_chunk(self.chunk(4))
        alternate.synchronize()
        self.assertIsNot(first, second)
        self.assertTrue(torch.equal(first, retained))
        self.assertEqual(b._model.text_projection.calls, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
