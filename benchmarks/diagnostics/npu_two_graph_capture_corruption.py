"""910C-050: NPU two-graph capture interaction probe.

Sixteen server-side gates (910C-032 .. 910C-049) localized a corruption in
which enabling the Qwen3-ASR encoder CUDA graph garbles compiled decode,
while every Python-observable state stays clean (910C-048) and a private
graph pool does not help (910C-049). The bounded working hypothesis is:

    one graph capture retroactively corrupts an already-captured,
    correctly-replaying graph in the same process, via process-global NPU
    runtime state.

This is a deliberately synthetic, fully exportable probe: it uses two
independent graphs and no model weights, SGLang service, HTTP, benchmark
client, or private data. A positive result is vendor-actionable evidence of a
runtime-level two-graph interaction. A negative result is *inconclusive* for
the Qwen3-ASR failure because this probe does not include Qwen3's compiled
decode, KV-cache, attention, stream, or graph-pool lifecycle.

Run one arm per fresh process. Mixing the two arms in one process would allow
the first arm to contaminate the control and invalidate its interpretation.

Main arm (decode-first, then encoder):
  1. capture graph A over a decode-like op sequence on a fixed static input;
  2. replay A, record its static-output hash, confirm it matches the eager
     reference (baseline healthy);
  3. capture graph B over a distinct encoder-like op sequence;
  4. replay A again on the identical static input and re-hash its static
     output;
  5. verdict: step-4 mismatch with step-2 match == one capture corrupts a
     resident graph.

Control arm (encoder-first, then decode): captures B before A. Since A is
then captured after B, the arm instead replays B before and after capturing
a *second* decode-like graph, to report whether corruption is order-dependent
(second-capture-breaks-first) or specific to the encoder-like sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys


def _require_npu():
    try:
        import torch
        import torch_npu  # noqa: F401  (registers the NPU backend)
    except Exception as exc:  # pragma: no cover - import guard
        print(json.dumps({"error": f"torch/torch_npu import failed: {exc}"}))
        sys.exit(2)
    if not torch.npu.is_available():
        print(json.dumps({"error": "NPU device not available"}))
        sys.exit(2)
    return torch


def _versions(torch) -> dict:
    out = {
        "torch": torch.__version__,
        "npu_device_count": int(torch.npu.device_count()),
    }
    try:
        import torch_npu

        out["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
    except Exception:
        out["torch_npu"] = "unavailable"
    try:
        out["cann"] = torch.version.cann  # type: ignore[attr-defined]
    except Exception:
        out["cann"] = "unavailable"
    return out


def _hash(t) -> str:
    return hashlib.sha256(
        t.detach().float().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def _decode_like(torch, x, w_qkv, w_o, w_norm, b_norm):
    """Decode-like: fused qkv projection, scaled attention-ish matmul, norm."""
    qkv = x @ w_qkv
    q, k, v = qkv.chunk(3, dim=-1)
    attn = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    attn = attn.softmax(dim=-1)
    ctx = attn @ v
    out = ctx @ w_o
    return torch.nn.functional.layer_norm(out + x, out.shape[-1:], w_norm, b_norm)


def _encoder_like(torch, x, w_conv, w1, w2):
    """Encoder-like: depthwise-ish conv + two-layer feed-forward stack."""
    y = torch.nn.functional.conv1d(
        x.transpose(1, 2),
        w_conv,
        padding=w_conv.shape[-1] // 2,
        groups=x.shape[-1],
    ).transpose(1, 2)
    y = torch.nn.functional.gelu(y @ w1)
    return y @ w2


class _CapturedGraph:
    """A captured graph plus the static output buffer it writes on replay."""

    def __init__(self, torch, fn):
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):  # warmup on a side stream per capture rules
                fn()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            self.static_out = fn()

    def replayed_hash(self, torch) -> str:
        self.graph.replay()
        torch.npu.synchronize()
        return _hash(self.static_out)


def _stable_replay_hashes(
    graph: _CapturedGraph, torch, replays: int
) -> set[str]:
    """Replay repeatedly so a one-off match is not treated as a healthy graph."""
    return {graph.replayed_hash(torch) for _ in range(replays)}


def _build_ops(torch, d=256, encoder_channels=64):
    dev = "npu"
    a_in = torch.randn(8, d, device=dev)
    w_qkv = torch.randn(d, 3 * d, device=dev) * 0.02
    w_o = torch.randn(d, d, device=dev) * 0.02
    w_norm = torch.ones(d, device=dev)
    b_norm = torch.zeros(d, device=dev)
    # [batch, time, channels] becomes [batch, channels, time] for depthwise
    # conv1d, then returns to [batch, time, channels] for the MLP.
    b_in = torch.randn(4, 64, encoder_channels, device=dev)
    w_conv = torch.randn(encoder_channels, 1, 5, device=dev) * 0.02
    w1 = torch.randn(encoder_channels, encoder_channels * 4, device=dev) * 0.02
    w2 = torch.randn(encoder_channels * 4, encoder_channels, device=dev) * 0.02

    fn_a = lambda: _decode_like(  # noqa: E731
        torch, a_in, w_qkv, w_o, w_norm, b_norm
    )
    fn_b = lambda: _encoder_like(torch, b_in, w_conv, w1, w2)  # noqa: E731
    return fn_a, fn_b


def _run_main_decode_first(torch, replays: int) -> dict:
    """Capture decode graph A, verify, then capture encoder graph B, re-check A."""
    torch.manual_seed(0)
    fn_a, fn_b = _build_ops(torch)

    eager_ref_hash = _hash(fn_a())

    graph_a = _CapturedGraph(torch, fn_a)
    pre_hashes = _stable_replay_hashes(graph_a, torch, replays)
    pre_hash = next(iter(pre_hashes))

    # Trigger: capture a second, distinct encoder-like graph into the process.
    graph_b = _CapturedGraph(torch, fn_b)
    _ = graph_b  # retained to mirror the live-resident-graph condition

    post_hashes = _stable_replay_hashes(graph_a, torch, replays)
    post_hash = next(iter(post_hashes))

    return {
        "arm_order": "decode-first",
        "replays": replays,
        "a_baseline_matches_eager": bool(pre_hash == eager_ref_hash),
        "a_replay_stable_before_other_capture": bool(len(pre_hashes) == 1),
        "a_output_changed_after_encoder_capture": bool(pre_hash != post_hash),
        "a_replay_stable_after_encoder_capture": bool(len(post_hashes) == 1),
        "a_post_capture_matches_eager": bool(post_hash == eager_ref_hash),
        "eager_hash": eager_ref_hash,
        "pre_hash": pre_hash,
        "post_hash": post_hash,
    }


def _run_control_encoder_first(torch, replays: int) -> dict:
    """Capture encoder graph B first, then decode graph A, then a 2nd decode graph.

    Reports whether a later capture corrupts an earlier graph regardless of the
    op family (order dependence), by replaying B before/after a second decode
    capture.
    """
    torch.manual_seed(0)
    fn_a, fn_b = _build_ops(torch)

    eager_ref_hash = _hash(fn_b())
    graph_b = _CapturedGraph(torch, fn_b)
    b_pre_hashes = _stable_replay_hashes(graph_b, torch, replays)
    b_pre = next(iter(b_pre_hashes))

    graph_a1 = _CapturedGraph(torch, fn_a)
    _ = graph_a1
    # A second decode-family capture after B was already resident.
    graph_a2 = _CapturedGraph(torch, fn_a)
    _ = graph_a2

    b_post_hashes = _stable_replay_hashes(graph_b, torch, replays)
    b_post = next(iter(b_post_hashes))

    return {
        "arm_order": "encoder-first",
        "replays": replays,
        "b_baseline_matches_eager": bool(b_pre == eager_ref_hash),
        "b_replay_stable_before_later_captures": bool(len(b_pre_hashes) == 1),
        "b_output_changed_after_later_captures": bool(b_pre != b_post),
        "b_replay_stable_after_later_captures": bool(len(b_post_hashes) == 1),
        "b_post_capture_matches_eager": bool(b_post == eager_ref_hash),
        "eager_hash": eager_ref_hash,
        "b_pre_hash": b_pre,
        "b_post_hash": b_post,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument(
        "--arm",
        choices=("decode-first", "encoder-first"),
        default="decode-first",
        help="Run exactly one arm. Execute the other arm in a new process.",
    )
    args = parser.parse_args()
    if args.replays < 2:
        parser.error("--replays must be at least 2")

    torch = _require_npu()
    result: dict = {
        "versions": _versions(torch),
        "synthetic_probe_limitations": (
            "Does not include Qwen3-ASR compiled decode, attention/KV cache, "
            "or production stream/pool lifecycle; a negative result is inconclusive."
        ),
    }
    if args.arm == "decode-first":
        main_arm = _run_main_decode_first(torch, args.replays)
        reproduced = bool(
            main_arm["a_baseline_matches_eager"]
            and main_arm["a_replay_stable_before_other_capture"]
            and main_arm["a_output_changed_after_encoder_capture"]
            and main_arm["a_replay_stable_after_encoder_capture"]
        )
        result.update({
            "arm": main_arm,
            "verdict": {
                "runtime_interaction_reproduced": reproduced,
                "negative_result_is_inconclusive": not reproduced,
            },
        })
        exit_code = 0 if reproduced else 1
    else:
        control_arm = _run_control_encoder_first(torch, args.replays)
        result.update({
            "arm": control_arm,
            "verdict": {
                "later_capture_changed_resident_graph": bool(
                    control_arm["b_baseline_matches_eager"]
                    and control_arm["b_replay_stable_before_later_captures"]
                    and control_arm["b_output_changed_after_later_captures"]
                    and control_arm["b_replay_stable_after_later_captures"]
                ),
                "negative_result_is_inconclusive": not control_arm[
                    "b_output_changed_after_later_captures"
                ],
            },
        })
        # A healthy control is a valid diagnostic result, not a failed run.
        exit_code = 0
    print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
