# Supported models

This page is the central user-facing view of model support. The model matrix
describes model-family capabilities and maturity. The configuration-evidence
table records validation only for the exact checkpoint, launch configuration,
and hardware named in that row. Evidence from one row does not qualify another
checkpoint or profile.

## Maturity and validation

Maturity describes the maintenance expectation:

- **Experimental**: an implementation exists but is not regularly qualified.
- **Supported**: the documented configuration is maintained and expected to
  work.

Validation describes the evidence recorded for an exact configuration:

- **Not recorded**: no recurring CI or performance qualification is documented.
- **Profile available**: a checked-in launch profile exists, but this table does
  not claim a completed runtime qualification.
- **Manually validated**: the exact configuration has a recorded validation,
  but is not covered by a recurring gate.
- **CI tested**: recurring model CI covers the documented configuration.
- **Performance qualified**: correctness and performance were measured under a
  defined benchmark configuration.

## Model matrix

| Model | Task | Endpoint | Pipeline | Streaming contract | Maturity | Cookbook |
|---|---|---|---|---|---|---|
| Higgs Audio v3 | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | Audio output; see cookbook | Supported | [Higgs TTS](./cookbook/higgs_tts.md) |
| Fish Audio S2-Pro | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | Audio output; see cookbook | Supported | [Fish Audio S2-Pro](./cookbook/fishaudio_s2_pro.md) |
| Voxtral-4B-TTS | TTS | `/v1/audio/speech` | preprocessing → TTS generation → vocoder | Audio output; see cookbook | Supported | [Voxtral TTS](./cookbook/voxtral_tts.md) |
| Qwen3-TTS | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | HTTP PCM or WebSocket audio output; Base checkpoints only | Supported | [Qwen3-TTS](./cookbook/qwen3_tts.md) |
| Fun-CosyVoice3 | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | No | Experimental | [Fun-CosyVoice3](./cookbook/fun_cosyvoice3.md) |
| MOSS-TTS v1.5 | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | Audio output; see cookbook | Supported | [MOSS-TTS](./cookbook/moss_tts.md) |
| MOSS-TTS Local v1.5 | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | Audio output; see cookbook | Supported | [MOSS-TTS Local](./cookbook/moss_tts_local.md) |
| Ming-Omni-TTS | TTS | `/v1/audio/speech` | preprocessing → reference encode → TTS engine → audio decode | No | Supported | [Ming-Omni-TTS](./cookbook/ming_tts.md) |
| dots.tts | TTS | `/v1/audio/speech` | preprocessing → backbone/acoustic tail → AudioVAE | Audio output; see cookbook | Supported | [dots.tts](./cookbook/dots_tts.md) |
| ZONOS2 | TTS | `/v1/audio/speech` | preprocessing → speaker encode → TTS engine → vocoder | Audio output; see cookbook | Supported | [ZONOS2](./cookbook/zonos2.md) |
| MiniMax Music 3 | Music | `/v1/audio/speech` | autoregressive music engine → DIT/DAC decode | No | Supported | [MiniMax Music 3](./cookbook/minimax_music3.md) |
| Qwen3-ASR | ASR | `/v1/audio/transcriptions` | audio preprocessing → ASR engine → response formatting | SSE transcript output; uploaded-file input | Supported | [Qwen3-ASR](./cookbook/qwen3_asr.md) |
| Fun-ASR-Nano | ASR | `/v1/audio/transcriptions` | audio preprocessing → ASR engine → response formatting | SSE transcript output; uploaded-file input | Supported | [Fun-ASR-Nano](./cookbook/fun_asr.md) |
| ARK-ASR-3B | ASR | `/v1/audio/transcriptions` | audio preprocessing → ASR engine → response formatting | SSE transcript output; uploaded-file input | Supported | [ARK-ASR-3B](./cookbook/arkasr.md) |
| MOSS-Transcribe-Diarize | ASR + diarization | `/v1/audio/transcriptions` | audio encoder → language model → structured transcript | SSE transcript output; uploaded-file input | Supported | [MOSS-Transcribe-Diarize](./cookbook/moss_transcribe_diarize.md) |
| Whisper | ASR / translation | `/v1/audio/transcriptions`, `/v1/audio/translations` | audio preprocessing → ASR engine → response formatting | SSE transcript output; uploaded-file input | Experimental | [Whisper ASR](./cookbook/whisper_asr.md) |
| Qwen3-Omni | Omni | `/v1/chat/completions`, `/v1/realtime` | multimodal preprocessing/encoders → thinker → optional talker/code2wav | Chat SSE and realtime WebSocket | Supported | [Qwen3-Omni](./cookbook/qwen3_omni.md) |
| Ming-Omni | Omni | `/v1/chat/completions` | multimodal preprocessing/encoders → thinker → optional talker | Model-dependent; see cookbook | Supported | [Ming-Omni](./cookbook/ming_omni.md) |
| LLaDA2.0-Uni | Multimodal generation | `/v1/chat/completions` | preprocessing → image encoder → thinker → decode | No | Experimental | [LLaDA2.0-Uni](./cookbook/llada2_uni.md) |

## Configuration evidence

This table starts with the configurations audited by the reference migrations
and the Higgs Quick Start. A missing row means qualification evidence has not
yet been recorded here; it does not mean the model is unsupported.

| Model | Checkpoint | Configuration | Hardware | Validation | Evidence |
|---|---|---|---|---|---|
| Higgs Audio v3 | `bosonai/higgs-audio-v3-tts-4b` | Model-derived default | Not yet recorded | Not recorded | [Cookbook](./cookbook/higgs_tts.md) |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | [Default profile](../examples/configs/qwen3_tts_1_7b.yaml) | Not yet recorded | Profile available | [Checked-in profile](../examples/configs/qwen3_tts_1_7b.yaml) |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (CI does not pin a model revision) | Two router workers using the model-derived pipeline plus tuned per-worker CI overrides | 2× H100 (one per worker) | CI tested | [TTS CI preset](../tests/test_model/tts_ci_config.py), [router fixture](../tests/test_model/test_tts_ci.py), [H100 workflow](../.github/workflows/test-tts-ci.yaml) |
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` (CI does not pin a model revision) | Two router workers using the model-derived default | 2× H100 (one per worker) | CI tested | [ASR CI preset](../tests/test_model/asr_ci_config.py), [router fixture](../tests/test_model/test_asr_ci_seedtts.py), [H100 workflow](../.github/workflows/test-asr-ci.yaml) |
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` | [RTX 4090 profile](../examples/configs/qwen3_asr_rtx4090.yaml) | RTX 4090 24 GB | Profile available | [Checked-in profile](../examples/configs/qwen3_asr_rtx4090.yaml) |
| Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Two router workers using the [H100 BF16 colocated profile](../examples/configs/qwen3_omni_colocated_h100_bf16.yaml) plus 32,768-token sequence overrides | 2× H100 (one per worker) | CI tested | [H100 workflow](../.github/workflows/test-qwen3-omni-ci.yaml), [CI fixture](../tests/test_model/conftest.py) |
| Qwen3-Omni | `marksverdhei/Qwen3-Omni-30B-A3B-FP8` | Two router workers using the [H100 FP8 colocated profile](../examples/configs/qwen3_omni_colocated_h100_fp8.yaml) plus 32,768-token sequence overrides | 2× H100 (one per worker) | CI tested | [H100 workflow](../.github/workflows/test-qwen3-omni-ci.yaml), [CI fixture](../tests/test_model/conftest.py) |
| Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | [H20 colocated profile](../examples/configs/qwen3_omni_colocated_h20.yaml) | 1× H20 | Profile available | [Checked-in profile](../examples/configs/qwen3_omni_colocated_h20.yaml) |
| Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | [H200 colocated profile](../examples/configs/qwen3_omni_colocated_h200.yaml) | 1× H200 | Profile available | [Checked-in profile](../examples/configs/qwen3_omni_colocated_h200.yaml) |

The recurring `higgs` TTS CI preset covers `bosonai/higgs-tts-3-4b`, not
`bosonai/higgs-audio-v3-tts-4b`, so it is not used as validation evidence for
the Higgs Audio v3 row.

## Reference-model capabilities

The reference migrations establish task-specific fields without making the
primary matrix excessively wide.

### TTS

| Model | Voice cloning | Text-only | Streaming | Languages | Output format |
|---|---|---|---|---|---|
| Qwen3-TTS | Base checkpoints | CustomVoice and VoiceDesign checkpoints | Base checkpoints only | 10 | 24 kHz audio |

### ASR

| Model | Language hints | Long audio | Streaming | Translation | Diarization | Timestamps |
|---|---|---|---|---|---|---|
| Qwen3-ASR | 30 languages plus Chinese aliases | Up to 1 hour non-streaming | SSE transcript output; uploaded-file input up to 1,200 seconds | No | No | Chunk-level in `verbose_json` |

### Omni

| Model | Text input | Image | Audio | Video | Text output | Audio output | Streaming |
|---|---|---|---|---|---|---|---|
| Qwen3-Omni | Yes | Yes | Yes | Yes | Yes | Speech pipeline | Yes |

## Maintaining this page

Update the model registry or pipeline configuration first when support changes.
Update CI qualification from the model CI definition, hardware claims from a
checked-in validated profile or qualification report, and benchmark status from
reproducible artifacts. See the [documentation guide](./STYLE_GUIDE.md) for the
full source-of-truth policy and new-model checklist.
