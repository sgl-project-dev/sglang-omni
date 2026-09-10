# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang_omni.models.auk.dit import Attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)


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
    torch.manual_seed(7)
    attn = Attention(dim=128, heads=2, dim_head=head_dim).cuda().eval()
    attn.requires_grad_(False)
    x = torch.randn(3, 19, 128, device="cuda")
    mask = (
        torch.arange(19, device="cuda")[None, :]
        < torch.tensor([11, 19, 7], device="cuda")[:, None]
    )
    rope = (torch.randn(3, 19, head_dim, device="cuda"), 1.0) if with_rope else None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        # Grad mode retains the eager implementation; frozen weights avoid
        # changing SDPA's backward requirements between the two calls.
        with torch.enable_grad():
            expected = attn(x, mask=mask, rope=rope)
        with torch.inference_mode():
            actual = attn(x, mask=mask, rope=rope)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
