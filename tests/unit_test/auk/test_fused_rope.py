# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from sglang_omni.models.auk.dit import Attention

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None,
        reason="requires NVIDIA CUDA",
    ),
]


@pytest.mark.parametrize(
    "batch,seq_len,heads,shared",
    [
        (2, 329, 24, True),
        (2, 539, 24, True),
        (6, 539, 24, False),
        (2, 731, 24, True),
        (1, 221, 24, True),
        (1, 1, 2, True),
        (32, 539, 24, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_norm_rope_matches_eager(batch, seq_len, heads, shared, dtype):
    from sglang_omni.models.auk.fused_qk_norm_rope import fused_norm_rope

    torch.manual_seed(1234)
    qkv = torch.randn(batch, seq_len, heads * 192, device="cuda", dtype=dtype)
    q_weight = torch.randn(64, device="cuda")
    k_weight = torch.randn(64, device="cuda")
    freqs = torch.randn(1 if shared else batch, seq_len + 3, 64, device="cuda") * 500
    q, k, _ = [
        t.view(batch, seq_len, heads, 64).transpose(1, 2) for t in qkv.chunk(3, -1)
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q = torch.nn.functional.rms_norm(q, (64,), q_weight)
        k = torch.nn.functional.rms_norm(k, (64,), k_weight)
    expected = Attention._apply_rope(q, k, (freqs, 1.0))
    actual = fused_norm_rope(qkv, q_weight, k_weight, freqs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("head_dim", [16, 64])
@pytest.mark.parametrize("with_rope", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_attention_matches_eager(head_dim, with_rope, autocast):
    from sglang_omni.models.auk.fused_qk_norm_rope import fused_norm_rope

    torch.manual_seed(7)
    attn = Attention(dim=128, heads=2, dim_head=head_dim).cuda().eval()
    attn.requires_grad_(False)
    x = torch.randn(3, 19, 128, device="cuda")
    mask = (
        torch.arange(19, device="cuda")[None, :]
        < torch.tensor([11, 19, 7], device="cuda")[:, None]
    )
    rope = (torch.randn(3, 19, head_dim, device="cuda"), 1.0) if with_rope else None
    with (
        patch(
            "sglang_omni.models.auk.fused_qk_norm_rope.fused_norm_rope",
            wraps=fused_norm_rope,
        ) as fused,
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast),
    ):
        # Grad mode retains the eager implementation; frozen weights avoid
        # changing SDPA's backward requirements between the two calls.
        with torch.enable_grad():
            expected = attn(x, mask=mask, rope=rope)
        assert fused.call_count == 0
        with torch.inference_mode():
            actual = attn(x, mask=mask, rope=rope)
        assert fused.call_count == int(head_dim == 64 and with_rope)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("model_name", ["AuK", "AuK-Flash"])
def test_engine_sampling_matches_eager(monkeypatch, model_name):
    from sglang_omni.config.runtime import resolve_stage_typed_kwargs
    from sglang_omni.models.auk import stages
    from sglang_omni.models.auk.config import ENGINE_STAGE, AuKPipelineConfig
    from sglang_omni.models.auk.dit import AuKDit
    from sglang_omni.models.auk.flow_matching import AuKFlowMatching
    from sglang_omni.models.auk.fused_qk_norm_rope import fused_norm_rope
    from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
    from sglang_omni.models.auk.payload_types import AuKState
    from sglang_omni.proto import OmniRequest, StagePayload
    from sglang_omni.scheduling.pipeline_state import load_state

    torch.manual_seed(42)
    flow = (
        AuKFlowMatching(
            AuKDit(
                dim=128,
                heads=2,
                dim_head=64,
                latent_dim=8,
                text_hidden_dim=16,
                num_layers=1,
                num_single_layers=2,
            ),
            num_llm_layers=2,
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    # AuK zero-initializes the gates and final projection. Nonzero weights
    # ensure an attention regression can affect the returned latents.
    for parameter in flow.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    states = [
        AuKState(
            conditioning=torch.randn(text, 16),
            text_mask=torch.ones(text, dtype=torch.bool),
            gen_frames=frames,
            ref_latent=torch.randn(ref, 8) if ref else None,
            ref_length=ref,
            seed=index,
        )
        for index, (text, frames, ref) in enumerate([(5, 19, 4), (8, 11, 0)])
    ]
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages, "make_runtime_config", lambda path: AuKRuntimeConfig(path, model_name)
    )
    monkeypatch.setattr(stages, "_load_flow", lambda *args: flow)
    config = AuKPipelineConfig(model_path="stub")
    engine = next(stage for stage in config.stages if stage.name == ENGINE_STAGE)
    scheduler = stages.create_auk_engine_executor(
        "stub", **resolve_stage_typed_kwargs(engine)
    )
    torch.cuda.synchronize()

    def run():
        payloads = [
            StagePayload(
                request_id=str(index),
                request=OmniRequest(inputs="test"),
                data=state.to_dict(),
            )
            for index, state in enumerate(states)
        ]
        return [
            load_state(payload, AuKState).latent
            for payload in scheduler._batch_fn(payloads)
        ]

    with patch(
        "sglang_omni.models.auk.fused_qk_norm_rope.fused_norm_rope",
        wraps=fused_norm_rope,
    ) as fused:
        with ExitStack() as stack:
            for block in flow.transformer.single_transformer_blocks:
                # An explicit FP32 epsilon equals the autocast reference's
                # default while selecting the existing eager branch.
                stack.enter_context(
                    patch.object(
                        block.attn.q_norm, "eps", torch.finfo(torch.float32).eps
                    )
                )
            expected = run()
        assert fused.call_count == 0
        actual = run()
        steps = stages.C.FLASH_NFE if model_name == "AuK-Flash" else engine.factory.nfe
        assert fused.call_count == steps * len(
            flow.transformer.single_transformer_blocks
        )
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
