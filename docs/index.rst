SGLang-Omni
=======================

SGLang-Omni is a high-performance serving framework for omni and multimodal models, built on top of `SGLang <https://github.com/sgl-project/sglang>`_. It is designed to orchestrate multi-stage pipelines with low latency and OpenAI-compatible APIs.

Modern omni models — such as speech-output LLMs and multimodal generation systems — decompose into heterogeneous stages with fundamentally different computational profiles: a compute-bound thinker, a memory-bound talker, a latency-sensitive codec. SGLang-Omni is built around a **computation-centric design**: each stage runs its own independent scheduler tuned to its bottleneck, communicates through a shared inbox/outbox abstraction, and transfers tensors via zero-copy shared memory. This prevents any single stage from degrading the others and allows new models to plug into the framework by declaring a pipeline topology rather than building an inference system from scratch.

About
-----

Core features:

- **Multi-Stage Pipeline**: Flexible framework for orchestrating preprocessing, AR engine, codec, and vocoder stages across processes and GPUs.
- **Native SGLang Integration**: Leverages SGLang's RadixAttention, continuous batching, and CUDA Graph optimizations for the AR backbone.
- **OpenAI-Compatible Server**: Drop-in ``/v1/audio/speech``, ``/v1/audio/transcriptions``, ``/v1/audio/translations``, and ``/v1/chat/completions`` endpoints with real-time streaming support.
- **Broad Model Support**: TTS (Higgs, Fish S2-Pro, Voxtral, Qwen3-TTS, MOSS-TTS / Local, Ming-Omni-TTS, dots.tts, ZONOS2), Music (MiniMax Music 3), ASR (Qwen3-ASR, Fun-ASR, ARK-ASR, Whisper, MOSS-Transcribe-Diarize), Omni (Qwen3-Omni, Ming-Omni), and LLaDA2.0-Uni.

Supported Models
----------------

See the centralized `supported-model and qualification matrix <supported_models.html>`_
for task, endpoint, pipeline, streaming, validated hardware, status, and
cookbook links.


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/installation.md
   get_started/installation_npu.md
   get_started/installation_xpu.md
   get_started/installation_cpu.md
   get_started/release_notes.md


.. toctree::
   :maxdepth: 1
   :caption: Supported Models

   supported_models.md


.. toctree::
   :maxdepth: 1
   :caption: Cookbook

   cookbook/higgs_tts.md
   cookbook/voxtral_tts.md
   cookbook/fishaudio_s2_pro.md
   cookbook/qwen3_tts.md
   cookbook/ming_tts.md
   cookbook/moss_tts.md
   cookbook/moss_tts_local.md
   cookbook/dots_tts.md
   cookbook/minimax_music3.md
   cookbook/zonos2.md
   cookbook/qwen3_asr.md
   cookbook/fun_asr.md
   cookbook/arkasr.md
   cookbook/moss_transcribe_diarize.md
   cookbook/whisper_asr.md
   cookbook/qwen3_omni.md
   cookbook/ming_omni.md
   cookbook/llada2_uni.md
   cookbook/fun_cosyvoice3.md
   cookbook/auk.md

.. toctree::
   :maxdepth: 1
   :caption: User Guide: Serving

   user_guide/serving/speech_api.md
   user_guide/serving/transcription_api.md
   basic_usage/qwen3_omni.md
   basic_usage/audio_translations.md
   basic_usage/tts.md
   basic_usage/process_topology_migration.md
   basic_usage/omni_router.md


.. toctree::
   :maxdepth: 1
   :caption: User Guide: Advanced Features

   user_guide/advanced_features/streaming.md
   user_guide/advanced_features/admission_control.md
   user_guide/advanced_features/deterministic_inference.md
   basic_usage/mps_dp.md


.. toctree::
   :maxdepth: 1
   :caption: User Guide: Deployment

   user_guide/deployment/stage_placement.md
   basic_usage/tts_process_topology.md


.. toctree::
   :maxdepth: 1
   :caption: Benchmarks

   benchmarks/methodology.md
   benchmarks/relay.md


.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   STYLE_GUIDE.md
   developer_reference/main.md
   developer_reference/apiserver_design.md
   developer_reference/pipeline.md
   developer_reference/adding_parameters.md
   developer_reference/communication.md
   developer_reference/reference_encode_service.md
   developer_reference/profiler.md
   developer_reference/qwen3_asr_concurrency_profile.md
   developer_reference/rl_admin_control.md
   developer_reference/bump_version.md


.. toctree::
   :maxdepth: 1
   :caption: References

   developer_reference/config.md
