# SPDX-License-Identifier: Apache-2.0
"""Semantic readback must not publish unfinished or reusable codec buffers."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("graphed", [False, True])
def test_codec_snapshots_after_early_semantic_readback(batch_size, prefill, graphed):
    device = torch.device("cuda")
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    codes = torch.zeros(batch_size, 3, device=device, dtype=torch.long)
    embeds = torch.zeros(batch_size, 4, device=device)
    static_ids = torch.arange(1, batch_size + 1, device=device)

    def predict():
        # Keep the output asynchronous so host token readiness cannot be used
        # as a substitute for the codec's own completion event.
        torch.cuda._sleep(10_000_000)
        codes.copy_(static_ids[:, None].expand(-1, 3))
        embeds.copy_(static_ids[:, None].expand(-1, 4))

    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        graph = None
        if graphed:
            predict()
            producer.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=producer):
                predict()

        def forward(layer0_codes, hidden, semantic_positions):
            static_ids.copy_(layer0_codes.flatten())
            if graph is None:
                predict()
            else:
                graph.replay()

        model = SimpleNamespace(
            config=SimpleNamespace(codec_eos_token_id=99),
            _output_codes=codes,
            _output_embeds=embeds,
            code_predictor_forward=forward,
        )
        runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
        runner.model = model
        runner._token_id_host_bufs = None
        runner._token_id_host_slot = 0
        snapshots = []
        retained_requests = []
        for step in range(3):
            # Changing membership/size exercises pinned-buffer reuse and growth.
            n = 1 if step == 0 else batch_size
            ids = torch.arange(1 + step * 10, 1 + step * 10 + n, device=device)
            if step == 2:
                ids[-1] = 99
            result = SimpleNamespace(
                next_token_ids=ids,
                logits_output=SimpleNamespace(
                    hidden_states=torch.zeros(n, 4, device=device)
                ),
            )
            batch = SimpleNamespace(
                forward_mode=SimpleNamespace(is_decode=lambda: not prefill),
                positions=torch.zeros(n, device=device, dtype=torch.long),
                seq_lens=torch.ones(n, device=device, dtype=torch.long),
            )

            # Our predictor has a fixed graph shape; pad live IDs as a real
            # graph runner does, then expose only the live rows to collection.
            def live_forward(layer0_codes, hidden, semantic_positions):
                padded = torch.zeros(batch_size, device=device, dtype=torch.long)
                padded[:n].copy_(layer0_codes.flatten())
                forward(padded, hidden, semantic_positions)

            model.code_predictor_forward = live_forward
            runner._collect_codes(result, batch, None, [])
            host_ids = runner._resolve_host_token_ids(result).tolist()
            expected = list(range(1 + step * 10, 1 + step * 10 + n))
            if step == 2:
                expected[-1] = 99
            assert host_ids == expected
            requests = [
                SimpleNamespace(
                    request_id=str(i),
                    data=SimpleNamespace(
                        output_codes=[], pending_feedback_queue=deque()
                    ),
                )
                for i in range(n)
            ]
            retained_requests.extend(requests)
            outputs = {
                str(i): SimpleNamespace(data=token) for i, token in enumerate(host_ids)
            }
            runner.post_process_outputs(
                result, SimpleNamespace(requests=requests), outputs
            )
            for request, token in zip(requests, expected):
                data = request.data
                if token == 99:
                    assert not data.output_codes
                    assert not data.pending_feedback_queue
                    continue
                with torch.cuda.stream(consumer):
                    consumer.wait_event(data.codes_ready_event)
                    observed_codes = data.output_codes[0].clone()
                    observed_embed = data.pending_feedback_queue[0].clone()
                snapshots.append((observed_codes, observed_embed, token))
        # Overwrite reusable graph outputs before inspecting saved snapshots.
        codes.fill_(-1)
        embeds.fill_(-1)
    producer.synchronize()
    consumer.synchronize()
    for observed_codes, observed_embed, token in snapshots:
        assert observed_codes.cpu().tolist() == [token] * 3
        assert observed_embed.cpu().tolist() == [token] * 4
