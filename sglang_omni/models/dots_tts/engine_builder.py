# SPDX-License-Identifier: Apache-2.0
"""dots.tts SGLang engine builder."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

logger = logging.getLogger(__name__)

_MPS_MEM_FRACTION_FLOOR = 0.78


class DotsTTSEngineBuilder(TtsEngineBuilder):
    model_name = "dots.tts"
    context_length = 2048

    def __init__(
        self,
        *,
        optimize: bool = True,
        num_steps: int = 4,
        max_audio_patches: int = 500,
        max_running_requests: int = 16,
    ) -> None:
        from sglang_omni.models.dots_tts.hf_config import DOTS_TTS_MODEL_ARCH_OVERRIDE

        self.model_arch_override = DOTS_TTS_MODEL_ARCH_OVERRIDE
        self.optimize = bool(optimize)
        self.num_steps = int(num_steps)
        self.max_audio_patches = int(max_audio_patches)
        self.max_running_requests = int(max_running_requests)
        if min(self.num_steps, self.max_audio_patches, self.max_running_requests) <= 0:
            raise ValueError("dots.tts batching limits must be positive")
        self._model_runner: Any | None = None
        self._acoustic_tail: Any | None = None
        # Filled by SGLangGenerationEngineBuilder.build before the infra is created.
        self.checkpoint_dir: str | None = None
        self.device: str | None = None

    def _uses_torch_mps(self) -> bool:
        # note (guozhihao-224): the Torch/MPS profile is the non-MLX Apple path.
        return self._on_apple() and not self._use_mlx()

    def _on_apple(self) -> bool:
        import torch

        return self.device is not None and torch.device(self.device).type == "mps"

    @staticmethod
    def _use_mlx() -> bool:
        from sglang.srt.utils.tensor_bridge import use_mlx

        return bool(use_mlx())

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from sglang_omni.models.dots_tts.compat import import_dots_solver_deps
        from sglang_omni.models.dots_tts.hf_config import register_dots_tts_hf_config

        register_dots_tts_hf_config()
        if self._on_apple():
            # note (guozhihao-224): the DiT compile hook and acoustic-tail
            # graphs are CUDA-only; Apple serves the eager single-request path.
            self.optimize = False
            import_dots_solver_deps()
        del checkpoint_dir

    def customize_server_args(self, server_args: Any) -> None:
        # The compiled DiT path only serves max_running_requests=1; the batched
        # tail is eager, so skip the process-global compile policy otherwise.
        # The policy must exist before SGLang builds the model; applying it in
        # setup_model nests Dynamo under FX.
        if self.optimize and int(server_args.max_running_requests) == 1:
            from sglang_omni.models.dots_tts.stages import _configure_optimized_kernels

            _configure_optimized_kernels()

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        if self._on_apple():
            if self._use_mlx():
                # note (guozhihao-224): only idle torch weights sit on Metal
                # under MLX; bf16 halves them.
                dtype = "bfloat16"
            else:
                # note (guozhihao-224): Metal aborts on mixed-dtype matmuls;
                # the Torch/MPS profile runs one eager fp32 request.
                dtype = "float32"
            return {
                "disable_cuda_graph": True,
                "disable_overlap_schedule": True,
                "disable_radix_cache": True,
                "enable_torch_compile": False,
                "max_running_requests": 1,
                "chunked_prefill_size": 0,
                # note (guozhihao-224): unified memory must also hold the
                # weights; start at the viability floor, not the canonical
                # dedicated-card 0.20.
                "mem_fraction_static": _MPS_MEM_FRACTION_FLOOR,
                "attention_backend": "torch_native",
                "sampling_backend": "pytorch",
                "dtype": dtype,
                "trust_remote_code": False,
            }
        return {
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": 0,
            "mem_fraction_static": 0.20,
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("dots.tts base support does not implement TP")
        requested = int(
            overrides.get("max_running_requests", self.max_running_requests)
        )
        if self._on_apple():
            if requested != 1:
                raise ValueError(
                    "dots.tts Apple profiles currently require max_running_requests=1"
                )
            # note (guozhihao-224): the canonical 0.20 assumes a dedicated
            # card; unified memory must also hold the weights.
            requested_fraction = float(overrides.get("mem_fraction_static", 0.20))
            if requested_fraction < _MPS_MEM_FRACTION_FLOOR:
                logger.info(
                    "dots.tts Apple: raising mem_fraction_static from "
                    "%.2f to %.2f (unified memory must fit weights + KV)",
                    requested_fraction,
                    _MPS_MEM_FRACTION_FLOOR,
                )
                overrides["mem_fraction_static"] = _MPS_MEM_FRACTION_FLOOR
        if requested <= 0:
            raise ValueError("dots.tts max_running_requests must be positive")
        self.max_running_requests = requested
        overrides["disable_radix_cache"] = True
        overrides["chunked_prefill_size"] = 0
        if bool(overrides.get("enable_torch_compile", False)):
            raise ValueError(
                "dots.tts uses its DiT compile path; SGLang backbone compile is disabled"
            )
        if not bool(overrides.get("disable_cuda_graph", True)):
            if self._on_apple():
                # note (guozhihao-224): no CUDA graph lifecycle on Metal;
                # degrade loudly so the canonical serving configs still boot.
                logger.info(
                    "dots.tts Apple: forcing disable_cuda_graph=True "
                    "(no CUDA graph lifecycle on Metal)"
                )
                overrides["disable_cuda_graph"] = True
            else:
                # note (luojiaxuan): the decode graph must be captured with hidden states (FULL);
                # its can_run gate requires an exact hidden-mode match with the
                # acoustic tail's per-step request.
                overrides["enable_return_hidden_states"] = True

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        model = model_worker.model_runner.model
        if self._uses_torch_mps():
            # note (guozhihao-224): SGLang casts input embeddings to bf16,
            # which the fp32 MPS profile cannot satisfy; own the backbone
            # forward with the pinned HF Qwen2.
            from sglang_omni.models.dots_tts.torch_mps_runner import (
                install_torch_mps_backbone,
            )

            install_torch_mps_backbone(model, checkpoint_dir)
        del device, gpu_id
        max_running_requests = int(server_args.max_running_requests)
        if not bool(server_args.disable_cuda_graph):
            from sglang_omni.scheduling.generation_batch_policy import (
                get_decode_cuda_graph_max_bs,
            )

            # note (luojiaxuan): installed before init_cuda_graphs so capture bakes the buffer
            # address into the decode graph. Generously sized: rows are tiny
            # (hidden_size elements) and capture may pad above
            # max_running_requests.
            model.enable_graph_feedback(
                max(
                    max_running_requests,
                    int(get_decode_cuda_graph_max_bs(server_args) or 0),
                    256,
                )
            )
        model.flow.optimize = self.optimize and max_running_requests == 1
        model.eval()
        if max_running_requests > 1:
            model.flow.init_batched_tail(
                num_slots=max_running_requests,
                nfe=self.num_steps,
                max_audio_patches=self.max_audio_patches,
                optimize=self.optimize,
            )
            self._acoustic_tail = model.flow.batched_tail
        if max_running_requests == 1:
            tail_backend = (
                "compiled single-request DiT/semantic encoder"
                if model.flow.optimize
                else "eager single-request DiT/semantic encoder"
            )
        else:
            tail_backend = model.flow.batched_tail.backend
        logger.info(
            "dots.tts latent engine backend: %s (optimize=%s, "
            "max_running_requests=%d, num_steps=%d)",
            tail_backend,
            self.optimize,
            max_running_requests,
            self.num_steps,
        )
        logger.info(
            "dots.tts backbone decode: %s",
            (
                "SGLang CUDA graph with model-owned feedback buffer"
                if not bool(server_args.disable_cuda_graph)
                else "eager"
            ),
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        if self._use_mlx():
            from sglang_omni.models.dots_tts.mlx.runner import DotsTTSMlxModelRunner

            if self.checkpoint_dir is None:
                raise RuntimeError(
                    "dots.tts MLX runner requires checkpoint_dir, which "
                    "SGLangGenerationEngineBuilder.build sets before infra creation"
                )
            self._model_runner = DotsTTSMlxModelRunner(
                model_worker, output_proc, checkpoint_dir=self.checkpoint_dir
            )
            return self._model_runner
        if self._uses_torch_mps():
            from sglang_omni.models.dots_tts.torch_mps_runner import (
                DotsTTSTorchMpsModelRunner,
            )

            self._model_runner = DotsTTSTorchMpsModelRunner(model_worker, output_proc)
            return self._model_runner
        from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner

        self._model_runner = DotsTTSModelRunner(model_worker, output_proc)
        return self._model_runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.dots_tts.request_builders import (
            apply_latent_result,
            build_sglang_dots_tts_request,
        )

        def _build_request(payload: Any) -> Any:
            data = build_sglang_dots_tts_request(payload)
            model.flow.validate_request(
                num_steps=data.state.num_steps,
                ode_method=data.state.ode_method,
                prompt_patch_count=int(data.prompt_span_positions.numel()),
                total_span_count=int(data.span_positions.numel()),
            )
            return data

        return _build_request, apply_latent_result

    def make_abort_callback(self) -> Any | None:
        assert self._model_runner is not None
        return self._model_runner.reset_request

    def extra_scheduler_callbacks(self) -> dict[str, Any]:
        if self._acoustic_tail is None:
            return {}
        return {"shutdown_callback": self._acoustic_tail.log_graph_counters}

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        from sglang_omni.models.dots_tts.request_builders import build_stream_output

        return {
            "stream_output_builder": build_stream_output,
            "enable_async_decode": False,
        }


__all__ = ["DotsTTSEngineBuilder"]
