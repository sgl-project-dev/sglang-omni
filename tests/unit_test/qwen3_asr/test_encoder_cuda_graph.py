# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_asr import encoder_cuda_graph, sglang_model
from sglang_omni.models.qwen3_asr.encoder_cuda_graph import (
    Qwen3ASREncoderLayerStackGraphRunner,
    _capture_defer_calls_diagnostic_value,
    _capture_state_delta,
    build_buckets,
    window_lens_from_token_counts,
)
from sglang_omni.platforms import current_platform


def test_build_buckets_rejects_bad_limits():
    with pytest.raises(ValueError):
        build_buckets(0, 780)


def _plan_only_runner(max_batch=8, max_tokens_per_clip=780):
    r = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    r._max_seqlen = 104
    r._max_windows_for = lambda b: max_batch + b // 104 + 1
    raw = build_buckets(max_batch, max_tokens_per_clip)
    r._buckets = raw[:-1] + (raw[-1] + r._max_windows_for(raw[-1]),)
    return r


@pytest.mark.parametrize("total,windows", [(65, 1), (260, 4), (6240, 64), (104, 1)])
def test_plan_invariants(total, windows):
    r = _plan_only_runner()
    bucket_size, dummies = r._plan(total, windows)
    assert total + sum(dummies) == bucket_size
    assert all(1 <= d <= r._max_seqlen for d in dummies)
    assert windows + len(dummies) == r._max_windows_for(bucket_size)
    assert r._plan(r._buckets[-1] + 1, 1) is None


def test_get_audio_feature_routing(monkeypatch):
    monkeypatch.setattr(sglang_model, "eager_preamble", lambda *a: torch.zeros(65, 4))
    tower = torch.nn.Linear(4, 4)
    tower.dtype = torch.float32
    tower.forward = lambda feats, feature_lens: SimpleNamespace(
        last_hidden_state=torch.full((1, 65, 8), 7.0)
    )
    model = object.__new__(sglang_model.Qwen3ASRForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.audio_tower = tower
    item = SimpleNamespace(
        feature=torch.zeros(1, 128, 500),
        feature_attention_mask=None,
        model_specific_data={"num_audio_tokens": 65},
    )
    get = sglang_model.Qwen3ASRForConditionalGeneration.get_audio_feature

    model._encoder_graph_runner = SimpleNamespace(
        tokens_per_window=104, run=lambda h, w: torch.ones(65, 8)
    )
    assert torch.equal(get(model, [item]), torch.ones(1, 65, 8))

    recorded = []
    model._encoder_graph_runner = SimpleNamespace(
        tokens_per_window=104,
        run=lambda h, w: None,
        record_capture_release_eager_output=lambda output: recorded.append(output),
    )
    assert torch.equal(get(model, [item]), torch.full((1, 65, 8), 7.0))
    assert len(recorded) == 1
    assert torch.equal(recorded[0], torch.full((1, 65, 8), 7.0))


def test_layer_stack_forwards_precomputed_attention_metadata():
    seen = {}

    class Attention:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return kwargs["x"]

    def identity_linear(hidden_states):
        return hidden_states, None

    layer = SimpleNamespace(
        self_attn_layer_norm=lambda hidden_states: hidden_states,
        self_attn=Attention(),
        final_layer_norm=lambda hidden_states: hidden_states,
        fc1=identity_linear,
        activation_fn=lambda hidden_states: hidden_states,
        fc2=identity_linear,
    )
    tower = SimpleNamespace(
        layers=[layer],
        ln_post=lambda hidden_states: hidden_states,
        proj1=identity_linear,
        act=lambda hidden_states: hidden_states,
        proj2=identity_linear,
    )
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._tower = tower
    runner._max_seqlen = 104
    hidden_states = torch.zeros(8, 4)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    attention_metadata = object()
    runner._capture_attention_metadata = attention_metadata

    output = runner._layer_stack(hidden_states, cu_seqlens)

    assert torch.equal(output, hidden_states)
    assert seen["cu_seqlens"] is cu_seqlens
    assert seen["max_seqlen"] == 104
    assert seen["forward_metadata"] is attention_metadata


def test_npu_capture_all_defers_until_real_window_signature():
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._is_npu = True
    runner._buckets = (128, 256)
    runner._graphs = {}
    runner._failed = set()

    runner.capture_all()

    assert runner._graphs == {}
    assert runner._failed == set()


def test_npu_capture_materializes_sequence_boundaries_on_host():
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._is_npu = True
    runner._device = torch.device("meta")

    cu_seqlens = runner._make_static_cu([4, 3, 1])

    assert cu_seqlens.device.type == "cpu"
    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, 4, 7, 8]


def test_npu_replay_admits_multiple_signatures_with_global_bound():
    replayed = []
    captured = []
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._is_npu = True
    runner._max_seqlen = 8
    runner._failed = set()
    runner._graphs = {}
    runner._npu_signature_capacity = 2
    runner._npu_signature_capacity_reported = False
    runner._reported_replays = set()
    runner._replay_count = 0
    runner._replay_buckets = Counter()
    runner._eager_fallback_reasons = Counter()
    runner._plan = lambda total, windows: (8, [8 - total])

    def capture(bucket_size, *, window_lens=None):
        captured.append((bucket_size, window_lens))
        return SimpleNamespace(
            hidden_states=torch.zeros(8, 2),
            cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
            attention_metadata=None,
            graph=SimpleNamespace(replay=lambda: replayed.append(True)),
            output=torch.zeros(8, 2),
        )

    runner._capture = capture
    hidden_states = torch.ones(4, 2)

    assert runner.run(hidden_states, [4]) is not None
    assert runner.run(hidden_states, [4]) is not None
    assert captured == [(8, (4, 4))]
    assert len(replayed) == 2
    assert runner._reported_replays == {(8, (4, 4))}

    # A second exact host-side layout in the same token bucket gets its own
    # graph, so neither replay can observe stale operator parameters.
    assert runner.run(hidden_states, [2, 2]) is not None
    assert captured == [(8, (4, 4)), (8, (2, 2, 4))]
    assert len(replayed) == 3

    # Graph memory remains globally bounded. A third unseen signature falls
    # back explicitly rather than evicting live graph memory or growing it.
    assert runner.run(hidden_states, [1, 3]) is None
    assert captured == [(8, (4, 4)), (8, (2, 2, 4))]
    assert runner._eager_fallback_reasons == {"npu_signature_capacity": 1}


def test_npu_capture_only_diagnostic_captures_without_replay(monkeypatch):
    replayed = []
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._is_npu = True
    runner._max_seqlen = 8
    runner._failed = set()
    runner._graphs = {}
    runner._npu_signature_capacity = 2
    runner._npu_signature_capacity_reported = False
    runner._reported_replays = set()
    runner._replay_count = 0
    runner._replay_buckets = Counter()
    runner._eager_fallback_reasons = Counter()
    runner._plan = lambda total, windows: (8, [8 - total])
    runner._capture = lambda *args, **kwargs: SimpleNamespace(
        hidden_states=torch.zeros(8, 2),
        cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
        attention_metadata=None,
        graph=SimpleNamespace(replay=lambda: replayed.append(True)),
        output=torch.zeros(8, 2),
    )
    monkeypatch.setenv("SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_ONLY", "1")

    assert runner.run(torch.ones(4, 2), [4]) is None
    assert len(runner._graphs) == 1
    assert replayed == []
    assert runner._eager_fallback_reasons == {"diagnostic_capture_only": 1}


def test_npu_capture_defer_diagnostic_keeps_first_signature_eager(monkeypatch):
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._is_npu = True
    runner._max_seqlen = 8
    runner._failed = set()
    runner._graphs = {}
    runner._npu_signature_capacity = 2
    runner._npu_signature_capacity_reported = False
    runner._reported_replays = set()
    runner._replay_count = 0
    runner._replay_buckets = Counter()
    runner._eager_fallback_reasons = Counter()
    runner._diagnostic_capture_defer_remaining = 1
    runner._diagnostic_capture_defer_configured = 1
    runner._diagnostic_capture_deferred_count = 0
    runner._diagnostic_run_count = 0
    runner._diagnostic_first_capture = None
    runner._plan = lambda total, windows: (8, [8 - total])
    captures = []
    runner._capture = lambda *args, **kwargs: captures.append((args, kwargs)) or SimpleNamespace(
        hidden_states=torch.zeros(8, 2),
        cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
        attention_metadata=None,
        graph=SimpleNamespace(replay=lambda: None),
        output=torch.zeros(8, 2),
    )

    assert runner.run(torch.ones(4, 2), [4]) is None
    assert captures == []
    assert runner._diagnostic_capture_deferred_count == 1
    assert runner._diagnostic_capture_defer_remaining == 0
    assert runner._diagnostic_run_count == 1
    assert runner._eager_fallback_reasons == {}

    assert runner.run(torch.ones(4, 2), [4]) is not None
    assert len(captures) == 1
    assert runner._diagnostic_first_capture == {
        "run_index": 2,
        "bucket_size": 8,
        "window_count": 2,
    }


def test_npu_capture_defer_diagnostic_env_validation(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES", raising=False)
    assert _capture_defer_calls_diagnostic_value() == 0
    monkeypatch.setenv("SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES", "2")
    assert _capture_defer_calls_diagnostic_value() == 2
    monkeypatch.setenv("SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES", "-1")
    with pytest.raises(ValueError, match="non-negative integer"):
        _capture_defer_calls_diagnostic_value()


def test_npu_capture_release_diagnostic_drops_live_graph(monkeypatch):
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._is_npu = True
    runner._device = torch.device("meta")
    runner._max_seqlen = 8
    runner._failed = set()
    runner._graphs = {}
    runner._npu_signature_capacity = 8
    runner._npu_signature_capacity_reported = False
    runner._reported_replays = set()
    runner._replay_count = 0
    runner._replay_buckets = Counter()
    runner._eager_fallback_reasons = Counter()
    runner._diagnostic_capture_release_count = 0
    runner._diagnostic_released_keys = set()
    runner._plan = lambda total, windows: (8, [8 - total])
    graph = object()
    capture_calls = []

    def capture(bucket, window_lens=None):
        capture_calls.append((bucket, window_lens))
        return SimpleNamespace(
            graph=graph,
            hidden_states=None,
            cu_seqlens=None,
            attention_metadata=None,
            output=None,
        )

    runner._capture = capture
    synchronize_calls = []
    monkeypatch.setattr(
        encoder_cuda_graph.torch.cuda,
        "synchronize",
        lambda device=None: synchronize_calls.append(device),
    )
    monkeypatch.setenv("SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE", "1")

    result = runner.run(torch.empty(7, 4), [7])

    assert result is None
    assert runner._graphs == {}
    assert runner._diagnostic_capture_release_count == 1
    assert runner._eager_fallback_reasons == {"diagnostic_capture_released": 1}
    assert synchronize_calls == [runner._device, runner._device]

    assert runner.run(torch.empty(7, 4), [7]) is None
    assert len(capture_calls) == 1
    assert runner._diagnostic_capture_release_count == 1
    assert runner._eager_fallback_reasons == {"diagnostic_capture_released": 2}


def test_capture_release_parity_records_match_and_mismatch():
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._diagnostic_pending_reference = torch.tensor([[1.0, 2.0]])
    runner._diagnostic_parity_count = 0
    runner._diagnostic_parity_match_count = 0
    runner._diagnostic_parity_mismatch_count = 0
    runner._diagnostic_last_parity = None

    runner.record_capture_release_eager_output(torch.tensor([[[1.0, 2.01]]]))

    assert runner._diagnostic_pending_reference is None
    assert runner._diagnostic_parity_count == 1
    assert runner._diagnostic_parity_match_count == 1
    assert runner._diagnostic_parity_mismatch_count == 0
    assert runner._diagnostic_last_parity["shape_match"] is True
    assert runner._diagnostic_last_parity["allclose"] is True
    assert runner._diagnostic_last_parity["reference_nonfinite"] == 0
    assert runner._diagnostic_last_parity["actual_nonfinite"] == 0

    runner._diagnostic_pending_reference = torch.tensor([[1.0, 2.0]])
    runner.record_capture_release_eager_output(torch.tensor([[[1.0, 3.0]]]))

    assert runner._diagnostic_parity_count == 2
    assert runner._diagnostic_parity_match_count == 1
    assert runner._diagnostic_parity_mismatch_count == 1
    assert runner._diagnostic_last_parity["allclose"] is False
    assert runner._diagnostic_last_parity["max_abs"] == 1.0


def test_capture_release_parity_passes_real_hidden_state_to_capture(monkeypatch):
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._is_npu = True
    runner._device = torch.device("meta")
    runner._max_seqlen = 8
    runner._failed = set()
    runner._graphs = {}
    runner._npu_signature_capacity = 8
    runner._npu_signature_capacity_reported = False
    runner._reported_replays = set()
    runner._replay_count = 0
    runner._replay_buckets = Counter()
    runner._eager_fallback_reasons = Counter()
    runner._diagnostic_capture_release_count = 0
    runner._diagnostic_released_keys = set()
    runner._plan = lambda total, windows: (8, [8 - total])
    captured = {}

    def capture(bucket, **kwargs):
        captured.update(bucket=bucket, **kwargs)
        return SimpleNamespace(
            graph=object(),
            hidden_states=None,
            cu_seqlens=None,
            attention_metadata=None,
            output=None,
        )

    runner._capture = capture
    monkeypatch.setattr(encoder_cuda_graph.torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setenv("SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE", "1")
    monkeypatch.setenv("SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE_PARITY", "1")
    hidden_states = torch.ones(7, 4)

    assert runner.run(hidden_states, [7]) is None
    assert captured["bucket"] == 8
    assert captured["window_lens"] == (7, 1)
    assert captured["diagnostic_hidden_states"] is hidden_states
    assert captured["diagnostic_total"] == 7


def test_npu_runner_captures_into_private_graph_pool(monkeypatch):
    pool = object()
    monkeypatch.setattr(
        encoder_cuda_graph.torch.cuda, "graph_pool_handle", lambda: pool
    )
    monkeypatch.setattr(
        encoder_cuda_graph.current_platform, "is_npu", lambda: True
    )
    tower = SimpleNamespace(
        parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]),
        config=SimpleNamespace(n_window=4, n_window_infer=8),
    )

    runner = Qwen3ASREncoderLayerStackGraphRunner(
        tower, buckets=(128,), max_batch_size=1
    )

    assert runner._graph_pool is pool


def test_non_npu_runner_uses_default_graph_pool(monkeypatch):
    monkeypatch.setattr(
        encoder_cuda_graph.current_platform, "is_npu", lambda: False
    )
    tower = SimpleNamespace(
        parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]),
        config=SimpleNamespace(n_window=4, n_window_infer=8),
    )

    runner = Qwen3ASREncoderLayerStackGraphRunner(
        tower, buckets=(128,), max_batch_size=1
    )

    assert runner._graph_pool is None


def test_capture_state_delta_is_stable_and_bounded():
    before = {
        "fused_ops": {"decoder.norm": {"forward": "native"}},
        "tensor_metadata": {"parameter:weight": {"version": 0}},
        "module_training": {"<root>": False},
        "runtime": {"memory_allocated": 1},
    }
    after = {
        "fused_ops": {"decoder.norm": {"forward": "native"}},
        "tensor_metadata": {"parameter:weight": {"version": 1}},
        "module_training": {"<root>": False},
        "runtime": {"memory_allocated": 2},
    }

    delta = _capture_state_delta(before, after)

    assert len(delta["before_digest"]) == 64
    assert len(delta["after_digest"]) == 64
    assert delta["sections"]["fused_ops"] == {"count": 0, "first_keys": []}
    assert delta["sections"]["tensor_metadata"] == {
        "count": 1,
        "first_keys": ["parameter:weight"],
    }
    assert delta["sections"]["runtime"] == {
        "count": 1,
        "first_keys": ["memory_allocated"],
    }


def test_encoder_graph_model_info_reports_replay_and_fallbacks():
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._is_npu = True
    runner._buckets = (128, 256)
    runner._graphs = {(128, (64, 64)): object()}
    runner._failed = {(256, (128, 128))}
    runner._npu_signature_capacity = 8
    runner._replay_count = 3
    runner._replay_buckets = Counter({128: 3})
    runner._eager_fallback_reasons = Counter({"npu_signature_mismatch": 2})

    assert runner.model_info() == {
        "enabled": True,
        "npu_lazy_signature_capture": True,
        "npu_signature_capacity": 8,
        "npu_signature_count": 1,
        "configured_buckets": [128, 256],
        "captured_graph_count": 1,
        "captured_buckets": {"128": 1},
        "capture_failure_count": 1,
        "diagnostic_capture_release_count": 0,
        "diagnostic_capture_release_parity": {
            "count": 0,
            "match_count": 0,
            "mismatch_count": 0,
            "last": None,
        },
        "diagnostic_capture_release_state": {"count": 0, "last": None},
        "diagnostic_capture_defer": {
            "configured": 0,
            "deferred_count": 0,
            "remaining": 0,
            "run_count": 0,
            "first_capture": None,
        },
        "replay_count": 3,
        "replay_buckets": {"128": 3},
        "eager_fallback_count": 2,
        "eager_fallback_reasons": {"npu_signature_mismatch": 2},
    }


@pytest.fixture
def asr_server_args():
    from sglang.srt.runtime_context import get_context

    mm_attention_backend = "aiter_attn" if current_platform.is_rocm() else "triton_attn"
    with get_context().override_server_args(
        model_path="Qwen/Qwen3-ASR-1.7B", mm_attention_backend=mm_attention_backend
    ):
        yield


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_matches_eager_tower(asr_server_args):
    from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.distributed.parallel_state import model_parallel_is_initialized

    from sglang_omni.models.qwen3_asr.audio_lengths import qwen3_asr_num_audio_tokens
    from sglang_omni.models.qwen3_asr.encoder_cuda_graph import eager_preamble

    if not model_parallel_is_initialized():
        init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29601",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
    from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    cfg = Qwen3OmniMoeAudioEncoderConfig(
        d_model=64,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        output_dim=32,
        num_mel_bins=128,
        n_window=50,
        n_window_infer=800,
    )
    torch.manual_seed(0)
    tower = Qwen3OmniMoeAudioEncoder(cfg).cuda().eval().to(torch.bfloat16)
    with torch.no_grad():
        for name, prm in tower.named_parameters():
            if prm.dim() >= 2:
                prm.normal_(0.0, 0.02)
            elif "weight" in name:
                prm.fill_(1.0)
            else:
                prm.zero_()

    runner = Qwen3ASREncoderLayerStackGraphRunner(
        tower, buckets=build_buckets(4, 104), max_batch_size=4
    )
    runner.capture_all()
    assert runner._graphs and not runner._failed

    # [500] and [450] share a bucket, so the second replay also checks that
    # stale rows from the first never leak into the output.
    for frame_lens in ([500], [450], [300, 500, 120], [800, 800, 800, 800]):
        feats = (torch.randn(128, sum(frame_lens), device="cuda") * 0.05).to(
            torch.bfloat16
        )
        lens = torch.tensor(frame_lens, device="cuda", dtype=torch.long)
        with torch.no_grad():
            ref = tower(feats, feature_lens=lens).last_hidden_state.squeeze(0)
        wl = window_lens_from_token_counts(
            [qwen3_asr_num_audio_tokens(f) for f in frame_lens],
            tokens_per_window=runner.tokens_per_window,
        )
        out = runner.run(eager_preamble(tower, feats, lens), wl)
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < 3e-2, f"{frame_lens}: max|diff|={diff}"

    assert (
        runner.run(
            torch.zeros(9999, 64, device="cuda", dtype=torch.bfloat16),
            [104] * 96 + [15],
        )
        is None
    )
