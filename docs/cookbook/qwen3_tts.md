# Qwen3-TTS

[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) is a
discrete multi-codebook text-to-speech family with voice cloning, 10-language
generation, and 24 kHz audio output.

## At a glance

| Item | Value |
|---|---|
| Task | TTS |
| Checkpoints | `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base`, plus CustomVoice and VoiceDesign variants |
| Endpoint | `/v1/audio/speech` |
| Pipeline | preprocessing → TTS engine → vocoder |
| Input | Text; Base checkpoints also require reference audio |
| Output | 24 kHz audio |
| Streaming | HTTP PCM or WebSocket audio output; Base checkpoints only |
| Maturity | Supported |
| Qualified checkpoint | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (recurring CI does not pin a model revision) |
| Qualified configuration | Two router workers using the model-derived pipeline plus the tuned per-worker CI overrides below |
| Evidence hardware | 2× H100 (one per worker) |
| Validation | CI tested |
| Evidence | [TTS CI preset](../../tests/test_model/tts_ci_config.py), [router fixture](../../tests/test_model/test_tts_ci.py), and [H100 workflow](../../.github/workflows/test-tts-ci.yaml) |

`12Hz` is the codec frame rate, not the playback sample rate.

## Install

Install SGLang-Omni by following [Installation](../get_started/installation.md).
Qwen3-TTS uses the upstream `qwen-tts` package and the system `sox` binary:

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

Keep `--no-deps` on both commands. Resolving `qwen-tts` would replace the
project's Transformers 5.12 / SGLang 0.5.18 stack with Transformers 4.57.3;
resolving `sox` can upgrade NumPy beyond the `numba==0.65.1` ceiling. Do not add
`onnxruntime`, which is already a project dependency and can trigger the same
NumPy conflict.

SGLang-Omni applies the required Transformers compatibility shim from
`sglang_omni/models/qwen3_tts/compat.py`. If an upstream API change produces a
`TypeError`, report it instead of installing `qwen-tts`'s Transformers pin.

## Deploy

### Default configuration

Serve the 1.7B Base checkpoint with its checked-in default configuration:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --port 8000
```

First startup can take several minutes while the TTS engine captures CUDA
Graphs.

This default launch is not the tuned configuration used by recurring CI.

### CI-qualified per-worker configuration

Each of the two `qwen3-tts` CI router workers adds the following topology,
concurrency, CUDA Graph, compile, and memory overrides to the model-derived
pipeline configuration. The checked-in YAML below selects the same pipeline
class and checkpoint, but CI does not pass that file directly.

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --tts_engine.engine.max_running_requests 64 \
  --tts_engine.engine.cuda_graph_max_bs 64 \
  --tts_engine.engine.torch_compile_max_bs 64 \
  --vocoder.process vocoder \
  --tts_engine.gpu_memory_fraction 0.85 \
  --vocoder.gpu_memory_fraction 0.10 \
  --port 8000
```

The [TTS CI preset](../../tests/test_model/tts_ci_config.py) is the source of
truth for these overrides. The [router fixture](../../tests/test_model/test_tts_ci.py)
launches two workers, each using one H100. Recurring CI does not qualify the
single-worker default command above.

### Other available configurations

The 0.6B Base checkpoint uses the same pipeline and request format, but is not
covered by the recurring 1.7B CI preset:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000
```

CustomVoice and VoiceDesign use their own checked-in configs. See
[TTS model usage](../basic_usage/tts.md) for those launch commands and their
text-only request fields.

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

## Model capabilities

### Checkpoint modes

| Mode | Conditioning | Streaming |
|---|---|---|
| Base | Reference audio; transcript recommended | Yes |
| CustomVoice | Checkpoint speaker selected by `voice` | No |
| VoiceDesign | Text plus non-empty `instructions` | No |

### Language hints

`language` defaults to `auto`. You can explicitly select Chinese, English,
Japanese, Korean, German, French, Russian, Portuguese, Spanish, or Italian.
Use an explicit hint for short or code-switched input when automatic detection
is unreliable.

### Streaming

Base checkpoints support the shared HTTP PCM stream and stateful speech
WebSocket. CustomVoice and VoiceDesign remain non-streaming. See
[Streaming](../user_guide/advanced_features/streaming.md) for the transport and
framing contracts.

When `initial_codec_chunk_frames` is omitted, Base checkpoints use 8 frames for
the first vocoder chunk. A smaller value lowers time to first audio but can
increase playback underruns. An explicit `0` uses the steady-state stride from
the first chunk. Utterances shorter than the initial threshold arrive in the
final flush.

### Deterministic inference

Both Base sizes expose the opt-in deterministic mode described in
[Deterministic inference](../user_guide/advanced_features/deterministic_inference.md).
That guide scopes the recorded batch-invariance evidence. The mode is disabled
by default because its serialized preprocessing and vocoder work reduce
throughput.

## Model-specific configuration

Qwen3-TTS defaults to 16 running requests, a waiting-queue depth of 16, four
request-build workers, and a pending-build depth of 16. See
[Admission control](../user_guide/advanced_features/admission_control.md) before
changing the running, queue, KV, or CUDA Graph limits together.

#### First-audio chunk ramp

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
buffered audio, so the continuity cost grows with concurrency: keep
`[2, 4, 8]` to low concurrency, prefer `[4, 8]` up to moderate concurrency,
and keep the default schedule for saturated serving. The ramp is mutually
exclusive with the legacy `initial_chunk_frames` /
`stream_initial_followup_stride` options, its first entry must not exceed the
steady stride, and a per-request `initial_codec_chunk_frames` still overrides
only the first chunk.

## Generation Parameters

Non-streaming responses set `X-Finish-Reason` to `stop` after codec EOS or
`length` at `max_new_tokens`. A `length` response is decodable but may contain
an incomplete utterance.

For the complete shared request and response contract, see the
[Speech API](../user_guide/serving/speech_api.md).

## Known limitations

- Base checkpoints need a reference clip for natural output; without one,
  speech is typically robotic.
- Omitting the reference transcript uses speaker-embedding-only mode and
  usually reduces cloning quality.
- `language: auto` can misdetect short or code-switched inputs.
- The 0.6B Base checkpoint has shown rare repetition loops up to
  `max_new_tokens`. Lower that limit or raise `repetition_penalty` when this
  occurs; the 1.7B checkpoint is less prone.

## Benchmark

The canonical Seed-TTS benchmark exercises correctness, latency, throughput,
streaming, and overload behavior:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --stream \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --port 8000
```

Use benchmark artifacts for current performance numbers instead of treating a
cookbook snapshot as a release guarantee. Follow the
[benchmark methodology](../benchmarks/methodology.md) when publishing results.

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
