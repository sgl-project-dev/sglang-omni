# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for the Qwen3-ASR audio encoder layer stack.

The chunk/conv front end reads each clip's length, so its shapes and control
flow change from request to request — we leave it on the eager path. What
we capture is the 24-layer transformer stack and the output projection that
follow. By then the batch is packed as [total_tokens, hidden], so capture
buckets only need to track total token count.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.layers.attention.vision import VisionAttentionMetadata

from sglang_omni.platforms import current_platform

logger = logging.getLogger(__name__)


def _capture_only_diagnostic_enabled() -> bool:
    return os.getenv("SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _capture_release_diagnostic_enabled() -> bool:
    return os.getenv(
        "SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _capture_release_parity_diagnostic_enabled() -> bool:
    return os.getenv(
        "SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE_PARITY", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _capture_release_state_snapshot_diagnostic_enabled() -> bool:
    return os.getenv(
        "SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE_STATE_SNAPSHOT", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _capture_defer_calls_diagnostic_value() -> int:
    """Return the opt-in count of first-seen signatures kept eager.

    This gate exists only to compare a real request before and immediately
    after its first encoder graph capture while preserving the same compiled
    generation process. It deliberately does not count as an eager fallback:
    the caller requested this diagnostic control path.
    """
    raw = os.getenv("SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES", "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES must be a non-negative integer"
        ) from exc
    if value < 0:
        raise ValueError(
            "SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES must be a non-negative integer"
        )
    return value


def _callable_label(value: Any) -> str | None:
    if value is None:
        return None
    function = getattr(value, "__func__", value)
    module = getattr(function, "__module__", type(function).__module__)
    qualname = getattr(function, "__qualname__", type(function).__qualname__)
    return f"{module}.{qualname}"


def _snapshot_digest(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _snapshot_section_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    changed = sorted(
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    )
    return {"count": len(changed), "first_keys": changed[:8]}


def _capture_state_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    sections = ("fused_ops", "tensor_metadata", "module_training", "runtime")
    return {
        "before_digest": _snapshot_digest(before),
        "after_digest": _snapshot_digest(after),
        "sections": {
            section: _snapshot_section_delta(
                before.get(section, {}), after.get(section, {})
            )
            for section in sections
        },
    }


def build_buckets(max_batch: int, max_tokens_per_clip: int) -> tuple[int, ...]:
    """Return token-count bucket sizes for encoder CUDA-graph capture.

    Each captured graph pads the packed [total_tokens, hidden] encoder
    input up to one of these sizes. The caller supplies two deployment
    limits:
    1. max_batch: pre_lm_max_batch_size on the per-LM encoder service
      (maximum clips in one encode batch).
    2. max_tokens_per_clip: encoder output tokens for the longest clip the
      pipeline admits; derived from AudioChunkingConfig.max_audio_clip_s
      via qwen3_asr_num_audio_tokens.

    Buckets are power-of-two sizes from 128 up to max_batch * max_tokens_per_clip.
    """
    if max_batch < 1 or max_tokens_per_clip < 1:
        raise ValueError(
            f"build_buckets needs positive limits, got max_batch={max_batch} "
            f"max_tokens_per_clip={max_tokens_per_clip}"
        )
    ceiling = int(max_batch) * int(max_tokens_per_clip)
    buckets: list[int] = []
    step = 128
    while step < ceiling:
        buckets.append(step)
        step *= 2
    buckets.append(ceiling)
    return tuple(buckets)


@dataclass
class _CapturedGraph:
    graph: torch.cuda.CUDAGraph
    hidden_states: torch.Tensor  # [bucket, hidden] static input
    cu_seqlens: torch.Tensor  # static boundaries; host-resident on Ascend
    attention_metadata: VisionAttentionMetadata | None
    output: torch.Tensor  # [bucket, output_dim] static result


class Qwen3ASREncoderLayerStackGraphRunner:
    """Captures the audio tower's transformer stack (post-conv) per bucket.

    The chunk/conv front end stays eager; callers hand over the packed
    hidden_states it produced plus the window boundaries, and get back the
    projected embeddings. max_seqlen is the architectural cap on tokens per
    attention window; materializing it once on the host removes the per-layer
    sync inside VisionAttention.
    """

    def __init__(
        self,
        audio_tower: Any,
        *,
        buckets: tuple[int, ...],
        max_batch_size: int,
    ) -> None:
        self._tower = audio_tower
        param = next(audio_tower.parameters())
        self._device = param.device
        self._dtype = param.dtype
        self._is_npu = current_platform.is_npu()
        # NPU: capture into a private graph pool so encoder capture does not
        # share the device default pool (and its driver-side allocator state)
        # with the SGLang decode graphs already resident in it.
        self._graph_pool = (
            torch.cuda.graph_pool_handle() if self._is_npu else None
        )
        cfg = audio_tower.config

        chunk_tokens = _get_feat_extract_output_lengths_int(cfg.n_window * 2)
        self._max_seqlen = chunk_tokens * (cfg.n_window_infer // (cfg.n_window * 2))
        self._max_windows_for = lambda bucket_size: (
            max_batch_size + bucket_size // self._max_seqlen + 1
        )
        top = buckets[-1]
        self._buckets = buckets[:-1] + (top + self._max_windows_for(top),)
        self._graphs: dict[Hashable, _CapturedGraph] = {}
        self._failed: set[Hashable] = set()
        # Ascend fused attention consumes actual_seq_lengths as a host-side
        # operator parameter. It cannot safely change that list by copying a
        # device tensor before replay. Real encoder batches can nevertheless
        # produce several window layouts for the same token bucket (most
        # commonly one per encoder batch size), so retain a bounded global set
        # of exact signatures rather than pinning one signature per bucket.
        # max_batch_size is both the number of normal batch-size shapes and a
        # deployment-controlled hard bound on graph memory growth.
        self._npu_signature_capacity = max_batch_size
        self._npu_signature_capacity_reported = False
        self._reported_replays: set[Hashable] = set()
        self._replay_count = 0
        self._replay_buckets: Counter[int] = Counter()
        self._eager_fallback_reasons: Counter[str] = Counter()
        self._diagnostic_capture_release_count = 0
        self._diagnostic_released_keys: set[Hashable] = set()
        self._diagnostic_pending_reference: torch.Tensor | None = None
        self._diagnostic_parity_count = 0
        self._diagnostic_parity_match_count = 0
        self._diagnostic_parity_mismatch_count = 0
        self._diagnostic_last_parity: dict[str, Any] | None = None
        self._diagnostic_model_root: torch.nn.Module | None = None
        self._diagnostic_pending_capture_state: dict[str, Any] | None = None
        self._diagnostic_state_snapshot_count = 0
        self._diagnostic_last_capture_state: dict[str, Any] | None = None
        self._diagnostic_capture_defer_remaining = (
            _capture_defer_calls_diagnostic_value() if self._is_npu else 0
        )
        self._diagnostic_capture_defer_configured = (
            self._diagnostic_capture_defer_remaining
        )
        self._diagnostic_capture_deferred_count = 0
        self._diagnostic_run_count = 0
        self._diagnostic_first_capture: dict[str, Any] | None = None
        self._capture_attention_metadata: VisionAttentionMetadata | None = None
        if self._is_npu:
            logger.info(
                "[qwen3-asr] encoder graph runner created pid=%d "
                "defer_captures=%d",
                os.getpid(),
                self._diagnostic_capture_defer_configured,
            )

    @property
    def tokens_per_window(self) -> int:
        return self._max_seqlen

    def set_diagnostic_model_root(self, model: torch.nn.Module) -> None:
        """Attach the complete model only for opt-in capture-state diagnostics."""
        self._diagnostic_model_root = model

    def _capture_state_snapshot(self) -> dict[str, Any]:
        """Collect bounded, non-content state around one NPU graph capture."""
        from sglang.kernels.fused_op import BaseFusedOp

        root = getattr(self, "_diagnostic_model_root", None) or self._tower
        fused_ops: dict[str, Any] = {}
        module_training: dict[str, bool] = {}
        for name, module in root.named_modules():
            path = name or "<root>"
            module_training[path] = bool(module.training)
            if isinstance(module, BaseFusedOp):
                fused_ops[path] = {
                    "type": f"{type(module).__module__}.{type(module).__qualname__}",
                    "is_torch_compile": bool(module.is_torch_compile),
                    "forward": _callable_label(module._forward_method),
                    "original_forward": _callable_label(
                        module._original_forward_method
                    ),
                    "compiled_native": _callable_label(module._compiled_native),
                }

        tensor_metadata: dict[str, Any] = {}
        for kind, named_tensors in (
            ("parameter", root.named_parameters()),
            ("buffer", root.named_buffers()),
        ):
            for name, tensor in named_tensors:
                try:
                    pointer = int(tensor.data_ptr())
                except Exception:
                    pointer = None
                tensor_metadata[f"{kind}:{name}"] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "data_ptr": pointer,
                    "version": int(getattr(tensor, "_version", -1)),
                    "requires_grad": bool(tensor.requires_grad),
                }

        runtime: dict[str, Any] = {
            "grad_enabled": bool(torch.is_grad_enabled()),
            "capture_only": _capture_only_diagnostic_enabled(),
            "capture_release": _capture_release_diagnostic_enabled(),
            "capture_release_parity": _capture_release_parity_diagnostic_enabled(),
        }
        try:
            from sglang.srt.distributed.device_communicators import pynccl_allocator

            pool = getattr(pynccl_allocator, "_graph_pool_id", None)
            runtime["symmetric_graph_pool"] = (
                None
                if pool is None
                else f"{type(pool).__module__}.{type(pool).__qualname__}:{pool}"
            )
        except Exception as exc:
            runtime["symmetric_graph_pool"] = f"unavailable:{type(exc).__name__}"

        device_module = torch.get_device_module(self._device)
        for field in ("memory_allocated", "memory_reserved"):
            fn = getattr(device_module, field, None)
            try:
                runtime[field] = int(fn(self._device)) if callable(fn) else None
            except Exception as exc:
                runtime[field] = f"unavailable:{type(exc).__name__}"
        try:
            stream = device_module.current_stream(self._device)
            runtime["current_stream"] = {
                "type": f"{type(stream).__module__}.{type(stream).__qualname__}",
                "handle": getattr(stream, "cuda_stream", None),
            }
        except Exception as exc:
            runtime["current_stream"] = f"unavailable:{type(exc).__name__}"
        try:
            runtime["stream_capturing"] = bool(torch.cuda.is_current_stream_capturing())
        except Exception as exc:
            runtime["stream_capturing"] = f"unavailable:{type(exc).__name__}"

        return {
            "fused_ops": fused_ops,
            "tensor_metadata": tensor_metadata,
            "module_training": module_training,
            "runtime": runtime,
        }

    def _record_capture_state_after_release(self) -> None:
        pending = getattr(self, "_diagnostic_pending_capture_state", None)
        if pending is None:
            return
        post_release = self._capture_state_snapshot()
        self._diagnostic_pending_capture_state = None
        transitions = {
            "warmup": _capture_state_delta(
                pending["pre_warmup"], pending["pre_graph_capture"]
            ),
            "capture": _capture_state_delta(
                pending["pre_graph_capture"], pending["post_graph_capture"]
            ),
            "release": _capture_state_delta(
                pending["post_graph_capture"], post_release
            ),
        }
        self._diagnostic_state_snapshot_count = (
            getattr(self, "_diagnostic_state_snapshot_count", 0) + 1
        )
        self._diagnostic_last_capture_state = transitions
        logger.info(
            "[qwen3-asr] encoder capture-state deltas warmup=%s capture=%s release=%s",
            transitions["warmup"],
            transitions["capture"],
            transitions["release"],
        )
        # memory_allocated/memory_reserved deltas are dominated by the graph's
        # static-buffer allocation and are expected; the diagnostic value is in
        # fused_ops dispatch, tensor metadata version/data_ptr, module training,
        # stream/pool identity changes.

    def capture_all(self) -> None:
        """Capture every bucket up front. A failed bucket stays eager."""
        if self._is_npu:
            # Synthetic bucket layouts are not valid replay metadata for
            # VisionAscendAttention: its sequence boundaries are captured as
            # host operator parameters. Capture lazily from the first real
            # layout observed for each bucket instead.
            logger.info(
                "[qwen3-asr] deferring NPU encoder graph capture until real "
                "window signatures are available"
            )
            return
        for bucket_size in self._buckets:
            if bucket_size in self._graphs or bucket_size in self._failed:
                continue
            try:
                self._graphs[bucket_size] = self._capture(bucket_size)
            except Exception as exc:
                logger.warning(
                    "[qwen3-asr] encoder graph capture failed for bucket=%d: %s; "
                    "bucket stays eager",
                    bucket_size,
                    exc,
                )
                self._failed.add(bucket_size)

    def _layer_stack(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """The computation we capture: 24 layers + ln_post + proj chain."""
        tower = self._tower
        h = hidden_states
        for layer in tower.layers:
            residual = h
            h = layer.self_attn_layer_norm(h)
            attention_kwargs = dict(
                max_seqlen=self._max_seqlen,
                forward_metadata=self._capture_attention_metadata,
            )
            h = layer.self_attn(x=h, cu_seqlens=cu_seqlens, **attention_kwargs)
            h = residual + h
            residual = h
            h = layer.final_layer_norm(h)
            h, _ = layer.fc1(h)
            h = layer.activation_fn(h)
            h, _ = layer.fc2(h)
            h = residual + h
        h = tower.ln_post(h)
        h = tower.proj1(h)[0]
        h = tower.act(h)
        return tower.proj2(h)[0]

    def _make_static_cu(self, sizes: list[int]) -> torch.Tensor:
        bounds = [0]
        for size in sizes:
            bounds.append(bounds[-1] + size)
        return torch.tensor(
            bounds,
            dtype=torch.int32,
            device="cpu" if self._is_npu else self._device,
        )

    def _capture(
        self,
        bucket_size: int,
        *,
        window_lens: tuple[int, ...] | None = None,
        diagnostic_hidden_states: torch.Tensor | None = None,
        diagnostic_total: int | None = None,
    ) -> _CapturedGraph:
        """Record one graph for a bucket-sized packed input."""
        device, dtype = self._device, self._dtype
        d_model = self._tower.ln_post.normalized_shape[0]
        static_hs = torch.zeros(bucket_size, d_model, device=device, dtype=dtype)
        if diagnostic_hidden_states is not None:
            if diagnostic_total is None:
                diagnostic_total = int(diagnostic_hidden_states.shape[0])
            if not 0 < diagnostic_total <= bucket_size:
                raise ValueError(
                    "encoder capture parity needs a valid real-token count, got "
                    f"total={diagnostic_total} bucket={bucket_size}"
                )
            static_hs[:diagnostic_total].copy_(diagnostic_hidden_states)

        max_windows = self._max_windows_for(bucket_size)
        if self._is_npu:
            if not window_lens or sum(window_lens) != bucket_size:
                raise ValueError(
                    "NPU encoder graph capture requires an exact window "
                    f"signature for bucket {bucket_size}"
                )
            sizes = list(window_lens)
        else:
            base, rem = divmod(bucket_size, max_windows)
            sizes = [base + 1] * rem + [base] * (max_windows - rem)
        # VisionAscendAttention turns cumulative sequence lengths into the
        # host-side actual_seq_lengths op parameter. Keeping this tensor on the
        # NPU made `.to("cpu")` synchronize the captured stream (ACL 107030).
        # The signature is immutable for this graph, so materialize it on the
        # host before capture. CUDA/ROCm retain their mutable device tensor.
        static_cu = self._make_static_cu(sizes)
        attention_metadata = None
        if current_platform.is_rocm() or self._is_npu:
            # AITER needs mutable device metadata without a per-layer sync.
            # Ascend needs immutable host metadata so no device-to-host copy is
            # issued from its captured stream.
            attention_metadata = VisionAttentionMetadata(
                cu_seqlens=static_cu,
                seq_lens=static_cu[1:] - static_cu[:-1],
                max_seqlen=self._max_seqlen,
            )
        self._capture_attention_metadata = attention_metadata

        capture_state = (
            self._is_npu
            and _capture_release_diagnostic_enabled()
            and _capture_release_state_snapshot_diagnostic_enabled()
            and getattr(self, "_diagnostic_state_snapshot_count", 0) == 0
        )
        pre_warmup_state = self._capture_state_snapshot() if capture_state else None

        def run_once() -> torch.Tensor:
            with torch.no_grad():
                return self._layer_stack(static_hs, static_cu)

        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        warmup_out = None
        with torch.cuda.stream(side):
            for _ in range(3):
                warmup_out = run_once()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        pre_graph_capture_state = (
            self._capture_state_snapshot() if capture_state else None
        )
        diagnostic_reference = None
        if diagnostic_hidden_states is not None:
            assert warmup_out is not None and diagnostic_total is not None
            normalized = warmup_out.squeeze(0) if warmup_out.dim() == 3 else warmup_out
            diagnostic_reference = normalized[:diagnostic_total].detach().clone()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=self._graph_pool, capture_error_mode="thread_local"
        ):
            static_out = run_once()
        torch.cuda.synchronize(device)
        post_graph_capture_state = (
            self._capture_state_snapshot() if capture_state else None
        )
        if capture_state:
            assert (
                pre_warmup_state is not None
                and pre_graph_capture_state is not None
                and post_graph_capture_state is not None
            )
            self._diagnostic_pending_capture_state = {
                "pre_warmup": pre_warmup_state,
                "pre_graph_capture": pre_graph_capture_state,
                "post_graph_capture": post_graph_capture_state,
            }
        if diagnostic_reference is not None:
            self._diagnostic_pending_reference = diagnostic_reference
        logger.info(
            "[qwen3-asr] captured encoder layer-stack graph bucket=%d "
            "windows=%d out=%s",
            bucket_size,
            len(sizes),
            tuple(static_out.shape),
        )
        return _CapturedGraph(
            graph=graph,
            hidden_states=static_hs,
            cu_seqlens=static_cu,
            attention_metadata=attention_metadata,
            output=static_out,
        )

    def run(
        self, hidden_states: torch.Tensor, window_lens: list[int]
    ) -> torch.Tensor | None:
        """Replay the recorded graph for a batch of hidden states."""
        self._diagnostic_run_count = getattr(self, "_diagnostic_run_count", 0) + 1
        total = int(hidden_states.shape[0])
        if not window_lens or sum(window_lens) != total:
            return self._fallback("invalid_window_layout")
        if max(window_lens) > self._max_seqlen:
            return self._fallback("window_too_large")

        plan = self._plan(total, len(window_lens))
        if plan is None:
            return self._fallback("no_bucket")
        bucket_size, dummy_sizes = plan
        effective_window_lens = tuple(window_lens + dummy_sizes)
        graph_key: Hashable = (
            (bucket_size, effective_window_lens) if self._is_npu else bucket_size
        )
        if (
            self._is_npu
            and _capture_release_diagnostic_enabled()
            and graph_key in getattr(self, "_diagnostic_released_keys", set())
        ):
            return self._fallback("diagnostic_capture_released")
        if graph_key in self._failed:
            return self._fallback("capture_failed")

        entry = self._graphs.get(graph_key)
        if entry is None:
            if self._is_npu and getattr(
                self, "_diagnostic_capture_defer_remaining", 0
            ):
                self._diagnostic_capture_defer_remaining -= 1
                self._diagnostic_capture_deferred_count = getattr(
                    self, "_diagnostic_capture_deferred_count", 0
                ) + 1
                logger.info(
                    "[qwen3-asr] diagnostic deferring first encoder graph "
                    "capture for bucket=%d windows=%d remaining=%d",
                    bucket_size,
                    len(effective_window_lens),
                    self._diagnostic_capture_defer_remaining,
                )
                return None
            if self._is_npu and len(self._graphs) >= self._npu_signature_capacity:
                if not self._npu_signature_capacity_reported:
                    logger.warning(
                        "[qwen3-asr] NPU encoder graph signature capacity=%d "
                        "exhausted; unseen signatures stay eager",
                        self._npu_signature_capacity,
                    )
                    self._npu_signature_capacity_reported = True
                return self._fallback("npu_signature_capacity")
            try:
                if getattr(self, "_diagnostic_first_capture", None) is None:
                    self._diagnostic_first_capture = {
                        "run_index": int(self._diagnostic_run_count),
                        "bucket_size": int(bucket_size),
                        "window_count": len(effective_window_lens),
                    }
                capture_kwargs: dict[str, Any] = {
                    "window_lens": (effective_window_lens if self._is_npu else None)
                }
                if (
                    self._is_npu
                    and _capture_release_diagnostic_enabled()
                    and _capture_release_parity_diagnostic_enabled()
                ):
                    capture_kwargs.update(
                        diagnostic_hidden_states=hidden_states,
                        diagnostic_total=total,
                    )
                entry = self._capture(bucket_size, **capture_kwargs)
            except Exception as exc:
                logger.warning(
                    "[qwen3-asr] encoder graph capture failed for bucket=%d: %s; "
                    "bucket stays eager",
                    bucket_size,
                    exc,
                )
                self._failed.add(graph_key)
                return self._fallback("capture_failed")
            self._graphs[graph_key] = entry

        if self._is_npu and _capture_release_diagnostic_enabled():
            # Diagnostic only: distinguish an irreversible capture-time side
            # effect from interference caused by a live NPUGraph/static-buffer
            # owner. Synchronize capture completion, drop every strong
            # reference owned by this runner, and collect before the eager
            # encoder path resumes.
            torch.cuda.synchronize(self._device)
            released = self._graphs.pop(graph_key)
            self._diagnostic_capture_release_count = (
                getattr(self, "_diagnostic_capture_release_count", 0) + 1
            )
            if not hasattr(self, "_diagnostic_released_keys"):
                self._diagnostic_released_keys = set()
            self._diagnostic_released_keys.add(graph_key)
            del entry, released
            gc.collect()
            torch.cuda.synchronize(self._device)
            self._record_capture_state_after_release()
            return self._fallback("diagnostic_capture_released")

        if self._is_npu and _capture_only_diagnostic_enabled():
            return self._fallback("diagnostic_capture_only")

        entry.hidden_states[:total].copy_(hidden_states)
        if not self._is_npu:
            bounds = [0]
            for size in window_lens + dummy_sizes:
                bounds.append(bounds[-1] + size)
            cu = torch.tensor(bounds, dtype=torch.int32)
            entry.cu_seqlens.copy_(cu, non_blocking=True)
            if entry.attention_metadata is not None:
                entry.attention_metadata.seq_lens.copy_(
                    cu[1:] - cu[:-1], non_blocking=True
                )
        entry.graph.replay()
        self._replay_count += 1
        self._replay_buckets[bucket_size] += 1
        if graph_key not in self._reported_replays:
            logger.info(
                "[qwen3-asr] replayed encoder layer-stack graph bucket=%d windows=%d",
                bucket_size,
                len(effective_window_lens),
            )
            self._reported_replays.add(graph_key)
        out = entry.output
        if out.dim() == 3:  # attention backends emit [1, tokens, dim]
            out = out.squeeze(0)
        return out[:total].clone()

    def record_capture_release_eager_output(self, output: torch.Tensor) -> None:
        """Compare post-release eager encoder output with pre-capture output.

        This diagnostic is deliberately evaluated only after the normal full
        audio-tower fallback has run.  It distinguishes encoder-output
        corruption from a process/device state change that first becomes
        visible in the subsequent compiled decode path.
        """
        reference = getattr(self, "_diagnostic_pending_reference", None)
        if reference is None:
            return
        self._diagnostic_pending_reference = None
        actual = (
            output.squeeze(0) if output.dim() == 3 and output.shape[0] == 1 else output
        )
        shape_match = tuple(reference.shape) == tuple(actual.shape)
        reference_nonfinite = int((~torch.isfinite(reference)).sum().item())
        actual_nonfinite = int((~torch.isfinite(actual)).sum().item())
        allclose = False
        max_abs = None
        mean_abs = None
        if shape_match:
            diff = (actual.float() - reference.float()).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
            allclose = bool(
                torch.allclose(actual.float(), reference.float(), rtol=1e-3, atol=3e-2)
            )

        self._diagnostic_parity_count = getattr(self, "_diagnostic_parity_count", 0) + 1
        counter_name = (
            "_diagnostic_parity_match_count"
            if allclose
            else "_diagnostic_parity_mismatch_count"
        )
        setattr(self, counter_name, getattr(self, counter_name, 0) + 1)
        self._diagnostic_last_parity = {
            "shape_match": shape_match,
            "allclose": allclose,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "reference_nonfinite": reference_nonfinite,
            "actual_nonfinite": actual_nonfinite,
        }
        logger.info(
            "[qwen3-asr] encoder capture-release parity shape_match=%s "
            "allclose=%s max_abs=%s mean_abs=%s reference_nonfinite=%d "
            "actual_nonfinite=%d",
            shape_match,
            allclose,
            max_abs,
            mean_abs,
            reference_nonfinite,
            actual_nonfinite,
        )

    def _fallback(self, reason: str) -> None:
        self._eager_fallback_reasons[reason] += 1

    def model_info(self) -> dict[str, Any]:
        captured_buckets = Counter(
            int(key[0] if isinstance(key, tuple) else key) for key in self._graphs
        )
        return {
            "enabled": True,
            "npu_lazy_signature_capture": self._is_npu,
            "npu_signature_capacity": (
                int(self._npu_signature_capacity) if self._is_npu else None
            ),
            "npu_signature_count": len(self._graphs) if self._is_npu else None,
            "configured_buckets": [int(bucket) for bucket in self._buckets],
            "captured_graph_count": len(self._graphs),
            "captured_buckets": {
                str(bucket): int(count)
                for bucket, count in sorted(captured_buckets.items())
            },
            "capture_failure_count": len(self._failed),
            "diagnostic_capture_release_count": int(
                getattr(self, "_diagnostic_capture_release_count", 0)
            ),
            "diagnostic_capture_release_parity": {
                "count": int(getattr(self, "_diagnostic_parity_count", 0)),
                "match_count": int(getattr(self, "_diagnostic_parity_match_count", 0)),
                "mismatch_count": int(
                    getattr(self, "_diagnostic_parity_mismatch_count", 0)
                ),
                "last": getattr(self, "_diagnostic_last_parity", None),
            },
            "diagnostic_capture_release_state": {
                "count": int(getattr(self, "_diagnostic_state_snapshot_count", 0)),
                "last": getattr(self, "_diagnostic_last_capture_state", None),
            },
            "diagnostic_capture_defer": {
                "configured": int(
                    getattr(self, "_diagnostic_capture_defer_configured", 0)
                ),
                "deferred_count": int(
                    getattr(self, "_diagnostic_capture_deferred_count", 0)
                ),
                "remaining": int(
                    getattr(self, "_diagnostic_capture_defer_remaining", 0)
                ),
                "run_count": int(getattr(self, "_diagnostic_run_count", 0)),
                "first_capture": getattr(self, "_diagnostic_first_capture", None),
            },
            "replay_count": int(self._replay_count),
            "replay_buckets": {
                str(bucket): int(count)
                for bucket, count in sorted(self._replay_buckets.items())
            },
            "eager_fallback_count": int(sum(self._eager_fallback_reasons.values())),
            "eager_fallback_reasons": {
                reason: int(count)
                for reason, count in sorted(self._eager_fallback_reasons.items())
            },
        }

    def _plan(self, total: int, real_windows: int) -> tuple[int, list[int]] | None:
        """Pick a bucket and the dummy-window sizes that absorb its padding."""

        for bucket_size in self._buckets:
            if bucket_size < total:
                continue
            slots = self._max_windows_for(bucket_size) - real_windows
            pad = bucket_size - total
            if slots < 0:
                continue
            if slots == 0:
                if pad == 0:
                    return bucket_size, []
                continue
            if not (slots <= pad <= slots * self._max_seqlen):
                continue
            base, rem = divmod(pad, slots)
            sizes = [base + 1] * rem + [base] * (slots - rem)
            return bucket_size, sizes
        return None


def _get_feat_extract_output_lengths_int(frames: int) -> int:
    """Compute conv output length for a mel-frame count."""
    from .audio_lengths import qwen3_asr_num_audio_tokens

    return int(qwen3_asr_num_audio_tokens(frames))


def eager_preamble(
    tower: Any, input_features: torch.Tensor, feature_lens: torch.Tensor
) -> torch.Tensor:

    import torch.nn.functional as F

    chunk_width = tower.n_window * 2
    chunk_num = torch.ceil(feature_lens / chunk_width).long()
    chunk_lengths = torch.tensor(
        [chunk_width] * chunk_num.sum(),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % chunk_width
    chunk_lengths[chunk_lengths == 0] = chunk_width

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = torch.nn.utils.rnn.pad_sequence(
        chunk_list, batch_first=True
    ).transpose(1, 2)

    feature_lens_after_cnn = _get_feat_extract_output_lengths_tensor(chunk_lengths)
    max_len_after_cnn = (
        int(feature_lens_after_cnn.max().item())
        if feature_lens_after_cnn.numel()
        else 0
    )
    idx = torch.arange(max_len_after_cnn, device=padded_feature.device)
    padded_mask_after_cnn = idx.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)

    padded_feature = padded_feature.unsqueeze(1)
    if padded_feature.size(0) <= tower.conv_chunksize:
        padded_embed = F.gelu(tower.conv2d1(padded_feature))
        padded_embed = F.gelu(tower.conv2d2(padded_embed))
        padded_embed = F.gelu(tower.conv2d3(padded_embed))
    else:
        padded_embeds = []
        for chunk in padded_feature.split(tower.conv_chunksize, dim=0):
            x = F.gelu(tower.conv2d1(chunk))
            x = F.gelu(tower.conv2d2(x))
            x = F.gelu(tower.conv2d3(x))
            padded_embeds.append(x)
        padded_embed = torch.cat(padded_embeds, dim=0)

    b, c, f, t = padded_embed.size()
    padded_embed = tower.conv_out(
        padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    )[0]
    positional_embedding = (
        tower.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
        .unsqueeze(0)
        .to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding
    return padded_embed[padded_mask_after_cnn]


def _get_feat_extract_output_lengths_tensor(
    input_lengths: torch.Tensor,
) -> torch.Tensor:
    leave = input_lengths % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def window_lens_from_token_counts(
    token_counts: list[int], *, tokens_per_window: int
) -> list[int]:
    out: list[int] = []
    for count in token_counts:
        full, rem = divmod(int(count), tokens_per_window)
        out.extend([tokens_per_window] * full)
        if rem:
            out.append(rem)
    return out
