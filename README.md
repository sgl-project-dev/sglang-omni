<div align="center">

<img src="https://raw.githubusercontent.com/sgl-project/sglang-omni/main/docs/_static/image/sgl-omni-logo.svg" alt="SGLang-Omni logo" width="400"></img>

### High-performance serving for speech and omni models

<p>
<a href="https://pypi.org/project/sglang-omni/"><img src="https://img.shields.io/pypi/v/sglang-omni?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
<a href="https://github.com/sgl-project/sglang-omni/stargazers"><img src="https://img.shields.io/github/stars/sgl-project/sglang-omni?style=for-the-badge&logo=github&label=stars" alt="GitHub stars"></a>
<a href="https://github.com/sgl-project/sglang-omni/blob/main/LICENSE"><img src="https://img.shields.io/github/license/sgl-project/sglang-omni?style=for-the-badge" alt="license"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-closed-raw/sgl-project/sglang-omni?style=for-the-badge&label=closed%20issues" alt="closed issues"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-raw/sgl-project/sglang-omni?style=for-the-badge&label=open%20issues" alt="open issues"></a>
<a href="https://deepwiki.com/sgl-project/sglang-omni"><img src="https://img.shields.io/badge/Ask-DeepWiki-087fca?style=for-the-badge" alt="Ask DeepWiki"></a>
</p>

<p>
<a href="https://sgl-project.github.io/sglang-omni/"><b>Documentation</b></a> |
<a href="#getting-started"><b>Quick Start</b></a> |
<a href="./docs/supported_models.md"><b>Models</b></a> |
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://slack.sglang.io"><b>Join Slack</b></a>
</p>

<p>
⭐ <b><a href="https://github.com/sgl-project/sglang-omni/stargazers">Star SGLang-Omni</a> to help more builders discover open infrastructure for multimodal and speech serving!</b>
</p>

</div>

--------------------------------------------------------------------------------

## News

- [2026/08] 🎵 Day-0 support for [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3): lyrics + caption → 32 kHz stereo song on `/v1/audio/speech`. \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/minimax_music3.html)\]
- [2026/08] 🚀 SGLang-Omni **v0.1.3** is on [PyPI](https://pypi.org/project/sglang-omni/). Install with `uv pip install --prerelease=allow "sglang-omni==0.1.3"`. \[[Installation](https://sgl-project.github.io/sglang-omni/get_started/installation.html)\]
- [2026/08] 🚀 TTS architecture refactor: shared pipeline state, engine construction, reference encoding, capability metadata, and vocoder scheduling. \[[Roadmap](https://github.com/sgl-project/sglang-omni/issues/985)\] \[[Blog](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-refactor.md)\]
- [2026/06] 🔥 MOSS-TTS Local Transformer v1.5 on SGLang-Omni with native-streaming 48 kHz speech. \[[Blog](https://lmsys.org/blog/2026-06-17-moss-tts-local-v15/)\] \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html)\]
- [2026/06] 🔥 Higgs Audio v3 TTS for real-time, controllable speech. \[[Blog](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)\] \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)\]

## Getting Started

This path assumes you are inside the recommended NVIDIA CUDA container from
[Installation](./docs/get_started/installation.md), which provides UCX,
FlashAttention, SGLang, and CUDA. For a manual installation, complete those
prerequisites before installing the package.

Install SGLang-Omni in an active Python 3.12 environment:

```bash
uv pip install --prerelease=allow sglang-omni
```

Start a [Higgs Audio v3](./docs/cookbook/higgs_tts.md) server:

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

## About

SGLang-Omni is a high-performance serving runtime for speech, audio, and omni
models. It manages complete model pipelines behind OpenAI-compatible APIs so
applications can serve multimodal chat, speech generation, transcription,
diarization, and music generation through one runtime.

- **Multi-stage serving:** Coordinate preprocessing, encoders, autoregressive
  engines, talkers, codecs, vocoders, and aggregators as one request lifecycle.
- **Stage-specialized scheduling:** Use SGLang execution and continuous
  batching for autoregressive stages while other stages use workload-specific
  schedulers.
- **Streaming and OpenAI-compatible APIs:** Stream speech, transcripts, chat
  completions, and realtime sessions through HTTP, SSE, and WebSocket APIs.
- **Flexible placement and scaling:** Colocate or distribute stages and process
  replicas across GPUs with explicit resource and transport configuration.

SGLang-Omni orchestrates heterogeneous model stages independently, allowing
autoregressive stages to use [SGLang](https://github.com/sgl-project/sglang)
while preprocessing, codecs, vocoders, and other stages use workload-specific
scheduling. Stages can be colocated or distributed across processes and GPUs.
See the [pipeline lifecycle](./docs/developer_reference/pipeline.md),
[communication design](./docs/developer_reference/communication.md), and
[stage placement guide](./docs/user_guide/deployment/stage_placement.md).

## Supported Workloads

| Workload | Representative models | API / output | Streaming |
|---|---|---|---|
| Omni | [Qwen3-Omni](./docs/cookbook/qwen3_omni.md), [Ming-Omni](./docs/cookbook/ming_omni.md) | `/v1/chat/completions`; text and optional audio | Model-dependent |
| Text-to-speech | [Qwen3-TTS](./docs/cookbook/qwen3_tts.md), [MOSS-TTS](./docs/cookbook/moss_tts.md), [Higgs Audio](./docs/cookbook/higgs_tts.md) | `/v1/audio/speech`; audio | Model-dependent |
| ASR and diarization | [Qwen3-ASR](./docs/cookbook/qwen3_asr.md), [Fun-ASR](./docs/cookbook/fun_asr.md), [MOSS-Transcribe-Diarize](./docs/cookbook/moss_transcribe_diarize.md) | `/v1/audio/transcriptions`; text and structured segments | Model-dependent |
| Music generation | [MiniMax Music 3](./docs/cookbook/minimax_music3.md) | `/v1/audio/speech`; audio | No |

See the [supported-model matrix and configuration evidence](./docs/supported_models.md)
for the complete list, endpoints, maturity, and audited qualification scope.

## Performance

Performance is qualified per model, hardware, workload, and traffic shape
rather than summarized as a project-wide speedup. SGLang-Omni includes
optimizations across CUDA Graphs, stage batching, asynchronous execution,
streaming, and multi-stage placement. See the
[benchmark methodology](./docs/benchmarks/methodology.md),
[configuration evidence](./docs/supported_models.md#configuration-evidence),
and model cookbooks for the audited scope and reproducible commands.

## Hardware Support

Hardware status describes documented implementations, not theoretical minimum
memory requirements. Model-level validation remains in the
[supported-model matrix](./docs/supported_models.md).

| Backend | Status | Documented scope |
|---|---|---|
| NVIDIA CUDA | Supported | Primary backend with checked-in single- and multi-GPU model profiles. See [Installation](./docs/get_started/installation.md). |
| Apple Silicon | Experimental | Qwen3-ASR runs through native MLX or Torch MPS on macOS arm64. Install with [`install.sh`](./install.sh) and follow the [Qwen3-ASR guide](./docs/cookbook/qwen3_asr.md#apple-silicon-mlx). |
| Intel GPU (XPU) | Experimental | Qwen3-ASR and Qwen3-TTS serve on one XPU; Qwen3-Omni uses multi-XPU tensor parallelism. See [Intel XPU installation](./docs/get_started/installation_xpu.md). |

## Documentation

[Installation](./docs/get_started/installation.md) ·
[Supported models](./docs/supported_models.md) ·
[Cookbook](./docs/cookbook/) ·
[Speech API](./docs/user_guide/serving/speech_api.md) ·
[Transcription API](./docs/user_guide/serving/transcription_api.md) ·
[Streaming](./docs/user_guide/advanced_features/streaming.md) ·
[Deployment](./docs/user_guide/deployment/stage_placement.md) ·
[Benchmarks](./docs/benchmarks/methodology.md) ·
[Developer guide](./docs/developer_reference/main.md)

## Community

SGLang-Omni welcomes contributors working on inference systems, kernels,
scheduling, inter-stage communication, model integration, benchmarking, and
deployment. Join the [SGLang Slack](https://slack.sglang.io), read the
[project blog](https://lmsys.org/blog/), or open an
[issue](https://github.com/sgl-project/sglang-omni/issues).

Organizations interested in supporting SGLang-Omni, TTS, or omni model serving
can contact Chenyang Zhao at
[zhaochenyang@lmsys.org](mailto:zhaochenyang@lmsys.org).

## Acknowledgments

SGLang-Omni builds on the SGLang ecosystem and on open model work from the TTS,
speech, and omni-model communities. We thank the model teams, systems
contributors, and partner organizations helping make open multimodal serving
faster, more reliable, and easier to extend.
