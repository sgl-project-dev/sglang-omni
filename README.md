<div align="center">

<img src="https://raw.githubusercontent.com/sgl-project/sglang-omni/main/docs/_static/image/sgl-omni-logo.svg" alt="SGLang-Omni logo" width="400"></img>

### High-performance serving for teams deploying speech, audio, and omni models with streaming, multi-stage execution, and OpenAI-compatible APIs

<p>
<a href="https://pypi.org/project/sglang-omni/"><img src="https://img.shields.io/pypi/v/sglang-omni?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
<a href="https://github.com/sgl-project/sglang-omni/stargazers"><img src="https://img.shields.io/github/stars/sgl-project/sglang-omni?style=for-the-badge&logo=github&label=stars" alt="GitHub stars"></a>
<a href="https://github.com/sgl-project/sglang-omni/blob/main/LICENSE"><img src="https://img.shields.io/github/license/sgl-project/sglang-omni?style=for-the-badge" alt="license"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-closed-raw/sgl-project/sglang-omni?style=for-the-badge&label=closed%20issues" alt="closed issues"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-raw/sgl-project/sglang-omni?style=for-the-badge&label=open%20issues" alt="open issues"></a>
<a href="https://deepwiki.com/sgl-project/sglang-omni"><img src="https://img.shields.io/badge/Ask-DeepWiki-087fca?style=for-the-badge" alt="Ask DeepWiki"></a>
</p>

<p>
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://sgl-project.github.io/sglang-omni/"><b>Documentation</b></a> |
<a href="#quick-start"><b>Quick Start</b></a> |
<a href="./docs/cookbook/"><b>Cookbook</b></a> |
<a href="https://github.com/sgl-project/sglang"><b>SGLang</b></a> |
<a href="https://slack.sglang.io"><b>Join Slack</b></a>
</p>

<p>
⭐ <b><a href="https://github.com/sgl-project/sglang-omni/stargazers">Star SGLang-Omni</a> to help more builders discover open infrastructure for multimodal and speech serving!</b>
</p>

</div>

--------------------------------------------------------------------------------

## Quick Start

This minimal path serves [Higgs Audio v3](./docs/cookbook/higgs_tts.md) on one
NVIDIA GPU and writes a generated WAV file. For Docker, system prerequisites,
or source installation, see [Installation](./docs/get_started/installation.md).

Install SGLang-Omni in an active Python 3.12 environment:

```bash
uv pip install --prerelease=allow "sglang-omni==0.1.3"
```

Start the server:

```bash
sgl-omni serve \
  --model-path bosonai/higgs-audio-v3-tts-4b \
  --port 8000
```

Send a matching speech request:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "Hello from SGLang-Omni."
  }' \
  --output output.wav
```

Next: [TTS](./docs/basic_usage/tts.md) ·
[ASR](./docs/cookbook/qwen3_asr.md) ·
[Omni](./docs/basic_usage/qwen3_omni.md) ·
[Supported models](./docs/supported_models.md)

## News

- [2026/09] 🐧 Day-0 support for [AuK](https://huggingface.co/tencent/AuK) and [AuK-Flash](https://huggingface.co/tencent/AuK-Flash): text + voice instructions → 24 kHz speech on `/v1/audio/speech`, audio + editing instructions → edited speech on `/generate`. \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/auk.html)\]
- [2026/09] 🚀 SGLang-Omni **v0.1.4** is on [PyPI](https://pypi.org/project/sglang-omni/). Install with `uv pip install --prerelease=allow "sglang-omni==0.1.4"`. \[[Installation](https://sgl-project.github.io/sglang-omni/get_started/installation.html)\]
- [2026/08] 🎵 Day-0 support for [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3): lyrics + caption → 32 kHz stereo song on `/v1/audio/speech`. \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/minimax_music3.html)\]
- [2026/08] 🚀 TTS architecture refactor: shared pipeline state, engine construction, reference encoding, capability metadata, and vocoder scheduling. \[[Roadmap](https://github.com/sgl-project/sglang-omni/issues/985)\] \[[Blog](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-refactor.md)\]
- [2026/06] 🔥 MOSS-TTS Local Transformer v1.5 on SGLang-Omni with native-streaming 48 kHz speech. \[[Blog](https://lmsys.org/blog/2026-06-17-moss-tts-local-v15/)\] \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html)\]
- [2026/06] 🔥 Higgs Audio v3 TTS for real-time, controllable speech. \[[Blog](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)\] \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)\]

## Why SGLang-Omni

- **Serve complete model pipelines.** Run preprocessing, encoders,
  autoregressive engines, talkers, codecs, and vocoders as one managed request
  lifecycle instead of assembling separate services.
- **Build responsive speech experiences.** Stream generated audio, transcript
  deltas, chat completions, and realtime sessions through OpenAI-compatible
  HTTP, SSE, and WebSocket interfaces.
- **Optimize each stage for its workload.** Use continuous batching and SGLang
  execution for autoregressive stages while lightweight schedulers handle
  preprocessing, acoustic tails, and streaming vocoders.
- **Scale placement deliberately.** Assign stages and process replicas across
  GPUs and processes, or colocate qualified pipelines with explicit memory
  budgets.
- **Operate a unified serving surface.** Route requests across workers with
  capability-aware selection, bounded admission, health checks, and readiness
  signals.
- **Extend shared model contracts.** Reuse pipeline configuration, request
  mapping, scheduling, transport, and response machinery when adding a model
  family.

## Supported Workloads

| Workload | Representative models | API / output | Streaming |
|---|---|---|---|
| Omni | [Qwen3-Omni](./docs/cookbook/qwen3_omni.md), [Ming-Omni](./docs/cookbook/ming_omni.md) | `/v1/chat/completions`; text and optional audio | Model-dependent |
| Text-to-speech | [Qwen3-TTS](./docs/cookbook/qwen3_tts.md), [MOSS-TTS](./docs/cookbook/moss_tts.md), [Higgs Audio](./docs/cookbook/higgs_tts.md) | `/v1/audio/speech`; audio | Model-dependent |
| ASR and diarization | [Qwen3-ASR](./docs/cookbook/qwen3_asr.md), [Fun-ASR](./docs/cookbook/fun_asr.md), [MOSS-Transcribe-Diarize](./docs/cookbook/moss_transcribe_diarize.md) | `/v1/audio/transcriptions`; text and structured segments | Model-dependent |
| Music generation | [MiniMax Music 3](./docs/cookbook/minimax_music3.md) | `/v1/audio/speech`; audio | No |

See the [supported-model and qualification matrix](./docs/supported_models.md)
for the complete list, validated hardware, endpoints, and support status.

## Performance

SGLang-Omni qualifies performance per model, hardware, workload, and traffic
shape instead of publishing one project-wide speedup. Current optimization
work includes:

- CUDA Graph capture for decode and supported prefill paths, asynchronous
  decoding, request-build overlap, and prefill coalescing;
- stage-level batching for encoders, autoregressive engines, acoustic tails,
  and vocoders;
- streaming chunk policies tuned for time to first audio, cadence, and playback
  continuity;
- host/device utilization, process replication, stage placement, and
  same-GPU or cross-GPU transport.

See the [benchmark methodology](./docs/benchmarks/methodology.md),
[Qwen3-ASR concurrency profile](./docs/developer_reference/qwen3_asr_concurrency_profile.md),
[MPS/DP qualification](./docs/basic_usage/mps_dp.md), and
[relay benchmarks](./docs/benchmarks/relay.md) for scoped evidence and
reproducible commands.

## Key Features

- **OpenAI-compatible serving:** speech generation, batch speech, uploaded
  voices, transcription, translation, multimodal chat, and realtime sessions.
  See the [Speech API](./docs/user_guide/serving/speech_api.md) and
  [Transcription API](./docs/user_guide/serving/transcription_api.md).
- **Streaming speech and text:** HTTP PCM, transcription and chat SSE, stateful
  speech WebSocket sessions, and bidirectional realtime interaction. See
  [Streaming](./docs/user_guide/advanced_features/streaming.md).
- **Multi-stage deployment:** declarative topologies, tensor parallel stages,
  process-level replicas, colocation, and disaggregated placement. See
  [Stage placement](./docs/user_guide/deployment/stage_placement.md).
- **Scheduling and admission:** workload-specific schedulers, bounded request
  queues, continuous batching, and model-local resource pools. See
  [Admission control](./docs/user_guide/advanced_features/admission_control.md).
- **Routing and health:** one capability-aware endpoint for worker selection,
  liveness, readiness, and lifecycle management. See the
  [Omni router](./docs/basic_usage/omni_router.md).
- **Model integration:** shared stage, model-runner, request-builder, and
  vocoder contracts for adding new speech and multimodal pipelines. See the
  [developer guide](./docs/developer_reference/main.md).

## Architecture Overview

The control plane owns pipeline topology, request registration, routing,
stream completion, cancellation, and stage lifecycle. The data plane moves
typed payloads between stages through process-local dispatch, CUDA IPC, shared
memory, or configured cross-node relay transport according to placement.

Each stage owns a scheduler matched to its execution pattern. Autoregressive
stages compose with [SGLang](https://github.com/sgl-project/sglang) for model
execution and continuous batching; preprocessing, encoders, decoders, and
vocoders use pipeline-native scheduling and exchange only the payloads required
by downstream stages.

Read the [pipeline lifecycle](./docs/developer_reference/pipeline.md),
[communication design](./docs/developer_reference/communication.md), and
[configuration reference](./docs/developer_reference/config.md) for the full
runtime contract.

## Hardware Support

Hardware status describes documented implementations, not theoretical minimum
memory requirements. Model-level validation remains in the
[supported-model matrix](./docs/supported_models.md).

| Backend | Status | Documented scope |
|---|---|---|
| NVIDIA CUDA | Supported | Primary backend with checked-in single- and multi-GPU model profiles. See [Installation](./docs/get_started/installation.md). |
| Apple Silicon | Experimental | Qwen3-ASR runs through native MLX or Torch MPS on macOS arm64. Install with [`install.sh`](./install.sh) and follow the [Qwen3-ASR guide](./docs/cookbook/qwen3_asr.md#apple-silicon-mlx). |
| Intel GPU (XPU) | Experimental | Qwen3-ASR and Qwen3-TTS serve on one XPU; Qwen3-Omni uses multi-XPU tensor parallelism. See [Intel XPU installation](./docs/get_started/installation_xpu.md). |

## Documentation and Community

| Area | Links |
|---|---|
| Get started | [Installation](./docs/get_started/installation.md) · [Apple Silicon](./docs/get_started/installation.md#macos-apple-silicon) · [Quick Start](#quick-start) · [Supported models](./docs/supported_models.md) |
| Task guides | [TTS](./docs/basic_usage/tts.md) · [ASR](./docs/cookbook/qwen3_asr.md) · [Omni](./docs/basic_usage/qwen3_omni.md) |
| APIs and deployment | [Speech API](./docs/user_guide/serving/speech_api.md) · [Transcription API](./docs/user_guide/serving/transcription_api.md) · [Stage placement](./docs/user_guide/deployment/stage_placement.md) · [Router](./docs/basic_usage/omni_router.md) |
| Benchmarks | [Methodology](./docs/benchmarks/methodology.md) · [Relay](./docs/benchmarks/relay.md) |
| Development and contributing | [Developer guide](./docs/developer_reference/main.md) · [TTS model integration](./docs/developer_reference/tts_model_integration.md) · [Issue tracker](https://github.com/sgl-project/sglang-omni/issues) |
| Community | [SGLang Slack](https://slack.sglang.io) · [SGLang](https://github.com/sgl-project/sglang) · [Blog](https://lmsys.org/blog/) |

SGLang-Omni welcomes contributors working on inference systems, kernels,
scheduling, inter-stage communication, model integration, benchmarking, and
deployment. Organizations interested in supporting SGLang-Omni, TTS, or omni
model serving can contact Chenyang Zhao at
[zhaochenyang@lmsys.org](mailto:zhaochenyang@lmsys.org).

## Acknowledgments

SGLang-Omni builds on the SGLang ecosystem and on open model work from the TTS,
speech, and omni-model communities. We thank the model teams, systems
contributors, and partner organizations helping make open multimodal serving
faster, more reliable, and easier to extend.
