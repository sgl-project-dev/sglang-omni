# Supported models

This page is the central user-facing view of model support. Maturity and
validation apply to the documented configuration, not every theoretical
checkpoint, precision, or hardware combination. "Not yet recorded" means the
project has not documented a validated hardware configuration; it is not a
claim that the model is unsupported.

## Maturity and validation

Maturity describes the maintenance expectation:

- **Experimental**: an implementation exists but is not regularly qualified.
- **Supported**: the documented configuration is maintained and expected to
  work.

Validation describes the evidence recorded for that configuration:

- **Not recorded**: no recurring CI or performance qualification is documented.
- **CI tested**: recurring model CI covers the documented configuration.
- **Performance qualified**: correctness and performance were measured under a
  defined benchmark configuration.

## Model matrix

| Model | Task | Endpoint | Pipeline | Streaming | Validated hardware | Maturity | Validation | Cookbook |
|---|---|---|---|---|---|---|---|---|
| Higgs Audio v3 | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | Yes | 1× H100 | Supported | CI tested | [Higgs TTS](./cookbook/higgs_tts.md) |
| Fish Audio S2-Pro | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | Yes | Not yet recorded | Supported | Not recorded | [Fish Audio S2-Pro](./cookbook/fishaudio_s2_pro.md) |
| Voxtral-4B-TTS | TTS | `/v1/audio/speech` | preprocessing → TTS generation → vocoder | Yes | 1× H200 | Supported | Performance qualified | [Voxtral TTS](./cookbook/voxtral_tts.md) |
| Qwen3-TTS | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | Partial | 1× H100 for the 1.7B Base CI configuration | Supported | CI tested | [Qwen3-TTS](./cookbook/qwen3_tts.md) |
| Fun-CosyVoice3 | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | No | Not yet recorded | Experimental | Not recorded | [Fun-CosyVoice3](./cookbook/fun_cosyvoice3.md) |
| MOSS-TTS v1.5 | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | Yes | RTX 4090 24 GB; 32 GB qualification profile; 1× H200 benchmark | Supported | Performance qualified | [MOSS-TTS](./cookbook/moss_tts.md) |
| MOSS-TTS Local v1.5 | TTS | `/v1/audio/speech` | preprocessing → TTS engine → vocoder | Yes | 1× H100 per CI worker; 2× H100 benchmark | Supported | CI tested | [MOSS-TTS Local](./cookbook/moss_tts_local.md) |
| Ming-Omni-TTS | TTS | `/v1/audio/speech` | preprocessing → reference encode → TTS engine → audio decode | No | 1× H200 | Supported | Performance qualified | [Ming-Omni-TTS](./cookbook/ming_tts.md) |
| dots.tts | TTS | `/v1/audio/speech` | preprocessing → backbone/acoustic tail → AudioVAE | Yes | 1× H100 | Supported | Performance qualified | [dots.tts](./cookbook/dots_tts.md) |
| ZONOS2 | TTS | `/v1/audio/speech` | preprocessing → speaker encode → TTS engine → vocoder | Yes | 1× H100 | Supported | Performance qualified | [ZONOS2](./cookbook/zonos2.md) |
| MiniMax Music 3 | Music | `/v1/audio/speech` | autoregressive music engine → DIT/DAC decode | No | 1× H200 | Supported | Performance qualified | [MiniMax Music 3](./cookbook/minimax_music3.md) |
| Qwen3-ASR | ASR | `/v1/audio/transcriptions` | audio preprocessing → ASR engine → response formatting | Yes | RTX 4090 24 GB; 1× H100 CI | Supported | CI tested | [Qwen3-ASR](./cookbook/qwen3_asr.md) |
| Fun-ASR-Nano | ASR | `/v1/audio/transcriptions` | audio preprocessing → ASR engine → response formatting | Yes | 1× H100 | Supported | CI tested | [Fun-ASR-Nano](./cookbook/fun_asr.md) |
| ARK-ASR-3B | ASR | `/v1/audio/transcriptions` | audio preprocessing → ASR engine → response formatting | Yes | Not yet recorded | Supported | Not recorded | [ARK-ASR-3B](./cookbook/arkasr.md) |
| MOSS-Transcribe-Diarize | ASR + diarization | `/v1/audio/transcriptions` | audio encoder → language model → structured transcript | Yes | 1× H100 | Supported | CI tested | [MOSS-Transcribe-Diarize](./cookbook/moss_transcribe_diarize.md) |
| Whisper | ASR / translation | `/v1/audio/transcriptions`, `/v1/audio/translations` | audio preprocessing → ASR engine → response formatting | Yes | 1× H100 CI; 1× H200 benchmark | Experimental | CI tested | [Whisper ASR](./cookbook/whisper_asr.md) |
| Qwen3-Omni | Omni | `/v1/chat/completions`, `/v1/realtime` | multimodal preprocessing/encoders → thinker → optional talker/code2wav | Yes | 1× H20, H100, or H200 colocated profiles | Supported | CI tested | [Qwen3-Omni](./cookbook/qwen3_omni.md) |
| Ming-Omni | Omni | `/v1/chat/completions` | multimodal preprocessing/encoders → thinker → optional talker | Partial | 4× H100 thinker + 1× H100 talker | Supported | Performance qualified | [Ming-Omni](./cookbook/ming_omni.md) |
| LLaDA2.0-Uni | Multimodal generation | `/v1/chat/completions` | preprocessing → image encoder → thinker → decode | No | Not yet recorded | Experimental | Not recorded | [LLaDA2.0-Uni](./cookbook/llada2_uni.md) |

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
| Qwen3-ASR | 30 languages plus Chinese aliases | Up to 1 hour non-streaming | Up to 1,200 seconds | No | No | Chunk-level in `verbose_json` |

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
