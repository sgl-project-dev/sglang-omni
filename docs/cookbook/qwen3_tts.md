# Qwen3-TTS

[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) is a
discrete multi-codebook text-to-speech family with voice cloning, 10-language
generation, and 24 kHz audio output.

## Overview

| Item | Value |
|---|---|
| Task | TTS |
| Checkpoint(s) | `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base`, plus CustomVoice and VoiceDesign variants |
| Endpoint(s) | `/v1/audio/speech` |
| Pipeline | preprocessing → TTS engine → vocoder |
| Input / output | Text and optional reference audio → 24 kHz audio |
| Streaming | HTTP PCM or WebSocket audio output; Base, CustomVoice, and VoiceDesign |
| Validated hardware | H100 |

`12Hz` is the codec frame rate, not the playback sample rate.

## Prerequisites

Install SGLang-Omni by following [Installation](../get_started/installation.md).
Qwen3-TTS uses the upstream `qwen-tts` package and the system `sox` binary:

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

Keep `--no-deps` on both commands. Resolving `qwen-tts` would replace the
project's Transformers 5.12 / SGLang 0.5.19 stack with Transformers 4.57.3;
resolving `sox` can upgrade NumPy beyond the `numba==0.65.1` ceiling. Do not add
`onnxruntime`, which is already a project dependency and can trigger the same
NumPy conflict.

SGLang-Omni applies the required Transformers compatibility shim from
`sglang_omni/models/qwen3_tts/compat.py`. If an upstream API change produces a
`TypeError`, report it instead of installing `qwen-tts`'s Transformers pin.

## Deploy

Serve the 1.7B Base checkpoint with its checked-in default configuration:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --port 8000
```

First startup can take several minutes while the TTS engine captures CUDA
Graphs.

## Send a request

Base checkpoints clone a voice from `references[0]`. Include the reference
transcript to use in-context-learning mode, which gives better speaker
similarity than speaker-embedding-only mode.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice": "default",
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

`ref_audio` and `ref_text` are shorthand for the first reference object's
`audio_path` and `text` fields.

## Capabilities

### Checkpoint modes

| Mode | Conditioning | Streaming |
|---|---|---|
| Base | Reference audio; transcript recommended | Yes |
| CustomVoice | Checkpoint speaker selected by `voice` | Yes |
| VoiceDesign | Text plus non-empty `instructions` | Yes |

### CustomVoice checkpoints

CustomVoice generates speech with built-in speakers through the same pipeline. Use it without reference audio; omit `ref_audio`, `ref_text`, `references`, and `x_vector_only_mode`. Omit `task_type` or set it to `CustomVoice`.

| Checkpoint | Config | Instruction guidance |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `examples/configs/qwen3_tts_0_6b_customvoice.yaml` | Accepted for backward compatibility, but not recommended |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `examples/configs/qwen3_tts_1_7b_customvoice.yaml` | Supported |

Both released checkpoints provide `Serena`, `Vivian`, `Uncle_Fu`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`, `Eric`, and `Dylan`. Speaker matching is case-insensitive; an omitted or `default` voice selects `Vivian`. `GET /v1/audio/voices` lists `default` and the served checkpoint's speakers. Unknown speakers or supplied cloning fields return HTTP 400; uploaded reference voices are not used for CustomVoice synthesis.

Both sizes support buffered speech, batch requests, incremental HTTP PCM output, and WebSocket audio output. HTTP streaming requires `stream=true` with `response_format="pcm"`; WebSocket sessions use `stream_audio=true` with `response_format="pcm"`.

**0.6B instruction compatibility:** SGLang-Omni continues to pass optional `instructions` into the 0.6B prompt, preserving existing behavior. The released 0.6B model does not provide reliable instruction control; omit this field or use 1.7B when style control is needed.

**Eric/Dylan language behavior:** For both sizes, `language: Auto` selects Eric's Sichuan dialect token or Dylan's Beijing dialect token. An explicit language takes precedence: `language: Chinese` keeps the Chinese language token. This preserves existing SGLang-Omni behavior and differs from the QwenLM/Qwen3-TTS Python wrapper (`qwen-tts` 0.1.1), which also selects dialect tokens for `Chinese`. This is a conditioning choice, not a guarantee that the speaker's accent disappears.

Start the 1.7B checkpoint with its matching config:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000
```

Then select a built-in speaker in the request. For 0.6B, use its model/config pair from the table and omit `instructions`.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "SGLang-Omni serves Qwen CustomVoice.",
    "voice": "Ryan",
    "language": "English",
    "instructions": "Speak clearly and calmly."
  }' \
  --output custom-voice.wav
```

### Language hints

`language` defaults to `auto`. You can explicitly select Chinese, English,
Japanese, Korean, German, French, Russian, Portuguese, Spanish, or Italian.
Use an explicit hint for short or code-switched input when automatic detection
is unreliable.

### Streaming

All three task types (Base/reference-cloning, CustomVoice and VoiceDesign) use
true incremental codec and vocoder streaming, for both this HTTP endpoint and
`/v1/audio/speech/stream` WebSocket sessions with `stream_audio=true`. Pass
`"stream_codec_output": false` on a request, or launch with
`--preprocessing.factory.stream_codec_output false`, to restore whole-utterance
decoding.

Streamed CustomVoice output on validated voice/language pairs (currently
Ryan/English with default sampling) withholds the model's silent bootstrap
codec frame, removing about 80 ms of leading silence from the first chunk. The
frame still feeds the vocoder, so every later sample is unchanged, and a
runtime silence check emits the audio unmodified whenever the first frame is
not actually silent. Opt out per request with
`"suppress_bootstrap_silence": false` or per deployment with
`--vocoder.factory.suppress_bootstrap_silence false`.

When `initial_codec_chunk_frames` is omitted, Qwen3-TTS ramps its first chunks
`1 -> 2 -> 4` codec frames before the steady stride, so first audio leaves after a
single AR step while the playback cushion is rebuilt within four chunks. Pass an
explicit value to trade continuity against time-to-first-audio.
Utterances that finish in fewer than the first chunk's generated codec frames never reach the
first chunk, so their audio arrives complete in a single final flush.

See [Streaming](../user_guide/advanced_features/streaming.md) for shared transport
and framing contracts.

### Deterministic inference

Both Base sizes expose opt-in deterministic inference. It is disabled by
default because it serializes preprocessing and vocoder work. See
[Deterministic inference](../user_guide/advanced_features/deterministic_inference.md)
for the enablement and evidence contract.

## Configuration

The 0.6B Base checkpoint uses the same pipeline and request format through
`examples/configs/qwen3_tts_0_6b.yaml`. CustomVoice and VoiceDesign use their
own checked-in configs. See [TTS model usage](../basic_usage/tts.md) for those
launch commands and their text-only request fields.

### Prefill Admission Coalescing

Under concurrent load, the `tts_engine` stage can coalesce prefill admission:
instead of admitting each prepared request into its own prefill batch, the
scheduler can briefly hold admission so that multiple ready requests are
prefilled together.

A prefill step has a largely fixed scheduler cost, so fuller batches can reduce
prefill overhead. The end-to-end benefit depends on whether that saving
outweighs the extra admission delay and any resulting reduction in decode
occupancy.

Coalescing is **off by default** and opt-in through the `tts_engine` factory
configuration:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --tts_engine.factory.prefill_coalesce_requests 2 \
  --tts_engine.factory.prefill_coalesce_wait_ms 30 \
  --port 8000
```

or per-stage in YAML:

```yaml
stages:
  tts_engine:
    factory:
      prefill_coalesce_requests: 2
      prefill_coalesce_wait_ms: 30.0
```

The gate engages only when `prefill_coalesce_requests >= 2`. Once engaged,
admission is released as soon as any of the following holds:

- decode is idle, so a ready request can start immediately;
- the waiting queue reaches `prefill_coalesce_requests`;
- the oldest waiting request has waited `prefill_coalesce_wait_ms`.

`prefill_coalesce_wait_ms` is therefore an upper bound on the added admission
wait. Admission may be released earlier if the target queue size is reached.

The values above are an example for the Qwen3-TTS workload and are not intended
as universal defaults. Match both `prefill_coalesce_requests` and
`prefill_coalesce_wait_ms` to the workload you actually serve. Coalescing is
most useful when natural prefill batches are small and a short hold can increase
batching without materially reducing decode occupancy. If the wait is too long,
the reduced decode occupancy can offset the prefill savings.

Leave coalescing disabled for latency-sensitive traffic or workloads where the
added wait does not produce enough additional batching.

### Process topology

By default all three stages share one process. Per-request reference
preprocessing (speech-tokenizer encode, speaker embedding, prompt embedding)
then competes with the AR scheduler and the vocoder for the same interpreter,
which caps single-replica throughput once concurrency passes ~32. Moving the
preprocessing stage to its own process removes that contention: the stage
loads a prompt-only frontend (embedding tables, text projection, predictor
codec embeddings, speaker encoder) plus the speech tokenizer, together about
2.2 GB of GPU memory for the extra process on a 1.7B checkpoint, and ships the
prepared prompt tensors to the engine through the payload. Every GPU stage must
then declare a memory fraction, and the engine's static fraction has to agree
with the one it declares:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --preprocessing.process tts_frontend \
  --preprocessing.gpu 0 \
  --preprocessing.gpu_memory_fraction 0.05 \
  --tts_engine.gpu_memory_fraction 0.75 \
  --tts_engine.engine.mem_fraction_static 0.75 \
  --vocoder.gpu_memory_fraction 0.12
```

`--vocoder.process vocoder` composes with it (lower `tts_engine` to 0.72 and
give the vocoder 0.15). Six SeedTTS samples at a fixed seed produced identical
PCM in both layouts; the extra process costs its CUDA context plus the frontend
weights.

### Codec decoding defaults

Streaming decodes run on the stateful incremental codec by default: each
follow-up chunk decodes only its fresh frames against per-stream state held in
a preallocated arena, steady-state cohorts replay CUDA graphs whose decode step
is `torch.compile`d, and the follow-up workers collect for 4 ms. Startup spends
about a minute compiling the steady shapes. The left-context decoder remains
available as a rollback:

```yaml
stages:
  vocoder:
    factory:
      enable_stateful_codec_decoder: false
```

`incremental_codec_cuda_graph`, `incremental_codec_compile` and
`followup_batch_wait_ms` are the individual switches. Measured on one H100
80GB at 20 requests per second, three client seeds of roughly 1200 requests
each: the default path holds 0.6% to 2.3% of streams underrun against 20.9%
for the left-context decoder, with first playable audio at 55 to 58 ms
against 82 to 89 ms.

### First-audio chunk ramp

For latency-sensitive deployments the whole early chunk schedule can be
configured server-side with `stream_chunk_ramp` on the vocoder stage: entry
`i` sizes streaming decode chunk `i + 1` in codec frames, and past the ramp
the steady stride takes over, so `[2, 4, 8]` yields a
`2 -> 4 -> 8 -> 8 -> ...` schedule. Set it through a pipeline config file:

```yaml
config_cls: Qwen3TTSPipelineConfig
model_path: Qwen/Qwen3-TTS-12Hz-0.6B-Base
stages:
  vocoder:
    factory:
      stream_chunk_ramp: [2, 4, 8]
```

```bash
python -m sglang_omni.cli serve --config qwen3_tts_ramp.yaml
```

Smaller early chunks lower time-to-first-audio but start playback with less
buffered audio, so the continuity cost grows with concurrency. The default
`[1, 2, 4]` suits low concurrency; prefer `[2, 4, 8]` at moderate concurrency
and `[4, 8]` for saturated serving. The ramp is mutually
exclusive with the legacy `initial_chunk_frames` /
`stream_initial_followup_stride` options, its first entry must not exceed the
steady stride, and a per-request `initial_codec_chunk_frames` still overrides
only the first chunk.

### Breakable prefill CUDA graphs

CustomVoice defaults to the breakable prefill backend with the shared token
ladder through 512 plus a one-token bucket. Base and VoiceDesign use eager
prefill. Opt out with `--tts_engine.engine.cuda_graph_backend_prefill disabled`.
Raising `cuda_graph_max_bs_prefill` regrows the default ladder; an explicit
`cuda_graph_bs_prefill` list is preserved. Capture adds startup work.

## Generation Parameters

Non-streaming responses set `X-Finish-Reason` to `stop` after codec EOS or
`length` at `max_new_tokens`. A `length` response is decodable but may contain
an incomplete utterance.

For the complete shared request and response contract, see the
[Speech API](../user_guide/serving/speech_api.md).

## Limitations

- Base checkpoints need a reference clip for natural output; without one,
  speech is typically robotic.
- Omitting the reference transcript uses speaker-embedding-only mode and
  usually reduces cloning quality.
- `language: auto` can misdetect short or code-switched inputs.
- The 0.6B Base checkpoint has shown rare repetition loops up to
  `max_new_tokens`. Lower that limit or raise `repetition_penalty` when this
  occurs; the 1.7B checkpoint is less prone.

## Benchmark

Run the Seed-TTS benchmark against the deployed server:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --stream \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --port 8000
```

Follow the [benchmark methodology](../benchmarks/methodology.md) when
publishing results.

See the [CustomVoice benchmark](../benchmarks/qwen3_tts_customvoice.md) for
the recorded H200 EN/ZH quality, latency, and playback-continuity results.

## Related documentation

- [TTS serving and request fields](../basic_usage/tts.md)
- [Speech API](../user_guide/serving/speech_api.md)
- [Streaming](../user_guide/advanced_features/streaming.md)
- [Admission control](../user_guide/advanced_features/admission_control.md)
- [Deterministic inference](../user_guide/advanced_features/deterministic_inference.md)
- [TTS process topology](../basic_usage/tts_process_topology.md)
- [MPS/DP and Qwen3-TTS weight-sharing status](../basic_usage/mps_dp.md)
- [Supported models](../supported_models.md)
- [TTS model integration](../developer_reference/tts_model_integration.md)
