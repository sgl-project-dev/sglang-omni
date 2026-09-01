# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) is a multilingual
audio transcription model served through the OpenAI-compatible transcription
API.

## Overview

| Item | Value |
|---|---|
| Task | ASR |
| Checkpoint(s) | `Qwen/Qwen3-ASR-1.7B` |
| Endpoint(s) | `/v1/audio/transcriptions` |
| Pipeline | audio preprocessing → ASR engine → response formatting |
| Input / output | One uploaded audio file → text, JSON, or verbose JSON transcript |
| Streaming | SSE transcript output; complete uploaded-file input, up to 1,200 seconds |
| Validated hardware | H100; RTX 4090 24 GB |

Qwen3-ASR does not support `/v1/audio/translations`; that route returns HTTP
400. See the [audio translation matrix](../basic_usage/audio_translations.md)
for models that support it.

## Prerequisites

Follow [Installation](../get_started/installation.md). No additional
model-specific package is required.

## Deploy

Qwen3-ASR runs one ASR stage on one GPU:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --port 8000
```

## Send a request

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

See the [Transcription API](../user_guide/serving/transcription_api.md) for
shared request fields, response formats, usage, and errors.

## Capabilities

### Language hints

When `language` is omitted, Qwen3-ASR detects the spoken language. You can pass
a case-insensitive code or canonical name for these 30 languages:

| Codes | Canonical names |
|---|---|
| `ar`, `yue`, `zh`, `cs`, `da`, `nl`, `en`, `fil`, `fi`, `fr` | Arabic, Cantonese, Chinese, Czech, Danish, Dutch, English, Filipino, Finnish, French |
| `de`, `el`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `mk`, `ms` | German, Greek, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Macedonian, Malay |
| `fa`, `pl`, `pt`, `ro`, `ru`, `es`, `sv`, `th`, `tr`, `vi` | Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Vietnamese |

The legacy `cn` and regional `zh-*` spellings map to Chinese. Unsupported hints
return HTTP 400. The model recognizes additional Chinese dialects, but they are
not separate forced hints; use `Chinese` or `zh`.

### Long audio

The current Qwen3-ASR model accepts at most 1,200 seconds of audio in one
request, so we transcribe longer uploads in chunks: we split the audio, run
each chunk as its own engine request, and join the transcripts back in
order. The behavior follows two kinds of values.

The scheduling policy is yours to tune, with dotted flags or the matching
YAML keys:

| Name | Default | Meaning |
|---|---|---|
| `--audio_chunking.max_audio_clip_s` | `30` | Longest clip we send to the engine in one request, and therefore the chunk length. It sits well below the model's native 1,200s on purpose: shorter chunks batch better, and the output-token budget scales with clip length on its own. Capped at the native clip limit. |
| `--audio_chunking.max_concurrent_chunks` | `8` | How many chunks of one request run in the engine at once. A per-request cap so one long upload can't crowd out everyone else's requests. |
| `--audio_chunking.max_total_audio_s` | `3600` | Upper limit on the whole upload; you get HTTP 400 above it. This is a memory guard: we keep the decoded waveform in memory while its chunks run. |

The model properties are ClassVars on `Qwen3ASRPipelineConfig`; no
configuration path reaches them:

| Name | Value | Meaning |
|---|---|---|
| `allow_audio_chunking` | `true` | Qwen3-ASR transcribes an isolated chunk correctly, so chunking is on. |
| `max_native_clip_s` | `1200` | Longest clip the model takes as one request (its native limit). Streaming cannot chunk, so this is the streaming cutoff. |
| `min_tail_s` | `0.5` | Shortest final chunk worth transcribing; if the tail would be shorter, we move the previous cut earlier to absorb it. This matches the model's own minimum input length. |

Note: Raising `audio_chunking.max_audio_clip_s` also resizes the encoder CUDA-graph bucket
ladder, which is derived from the chunk length: a longer chunk means more and
larger captured graphs, and their static buffers stay resident for the life of
the server (roughly 6.6 KB per token of ladder ceiling; at 1,200s the ceiling
is 124,800 tokens). Budget for that when you raise the flag on small GPUs.

`verbose_json` returns one segment per chunk with chunk-level start and end
times, not word timestamps. Formats without a readable duration fall back to
the non-chunked path.

### Streaming

Streaming does not currently use long-audio chunking, so uploads above 1,200
seconds return HTTP 400. Use non-streaming mode for longer files. See
[Streaming](../user_guide/advanced_features/streaming.md) for the shared SSE
event contract.

## Configuration

The checked-in `examples/configs/qwen3_asr_rtx4090.yaml` profile keeps BF16,
limits the stage to 16 running requests, and sets `mem_fraction_static` to
`0.65`; it was validated on one 24 GB RTX 4090. This is not a minimum-memory
claim.

The default `auto` dtype follows the BF16 checkpoint configuration. Pass
`--asr.factory.dtype float16` only when you intentionally need FP16.
Asynchronous decode overlaps the next GPU forward with host work; disable it with
`--asr.factory.enable_async_decode false`, or move the batch-size crossover at
which it engages with `--asr.factory.async_decode_min_batch_size`. Per-stage
config files and dotted CLI overrides follow the shared
[configuration contract](../developer_reference/config.md); command-line
overrides take precedence over the checked-in profile.

`prompt` is accepted for OpenAI compatibility but Qwen3-ASR ignores it. Audio
is resampled to 16 kHz before transcription.

## Limitations

- The endpoint accepts one uploaded file per request.
- `/v1/audio/translations` is unsupported.
- Streaming is limited to 1,200 seconds and does not use long-audio chunking.
- Timestamps are chunk-level; the model does not emit word timestamps.
- `prompt` does not affect transcription.

## Benchmark

Run the Seed-TTS ASR benchmark against the deployed server:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --concurrencies 1,2,4,8,16,32,64 \
  --repeats 3 \
  --warmup
```

See the
[Qwen3-ASR concurrency profile](../developer_reference/qwen3_asr_concurrency_profile.md)
for the measured tuning study and bottleneck decomposition, and follow the
[benchmark methodology](../benchmarks/methodology.md) when publishing results.

## Related documentation

- [Transcription API](../user_guide/serving/transcription_api.md)
- [Streaming](../user_guide/advanced_features/streaming.md)
- [Admission control](../user_guide/advanced_features/admission_control.md)
- [Benchmark methodology](../benchmarks/methodology.md)
- [Audio translation support](../basic_usage/audio_translations.md)
- [MPS/DP deployment](../basic_usage/mps_dp.md)
- [Supported models](../supported_models.md)
- [Qwen3-ASR concurrency profile](../developer_reference/qwen3_asr_concurrency_profile.md)
