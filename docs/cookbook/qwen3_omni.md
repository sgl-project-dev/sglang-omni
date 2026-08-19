# Qwen3-Omni

[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
accepts text, image, audio, and video and returns text or text plus 24 kHz audio.

## At a glance

| Item | Value |
|---|---|
| Task | Omni |
| Checkpoint | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| Endpoints | `/v1/chat/completions`, `/v1/realtime` |
| Text pipeline | preprocessing/encoders → multimodal aggregate → thinker → decode |
| Speech pipeline | text pipeline + talker AR → code2wav |
| Input | Text, image, audio, video |
| Output | Text; optional audio in speech mode |
| Streaming | Text and audio |
| Validated hardware | Single-GPU H20, H100, and H200 colocated profiles |
| Support status | CI tested |

## Install

Follow [Installation](../get_started/installation.md). No additional
model-specific package is required.

## Deploy

Use the selector to generate a launch command for text-only or speech output,
topology, tensor parallelism, and precision:

```{raw} html
<div id="sgl-server-gen-mount"></div>
```

For a one-GPU speech worker, choose the checked-in profile for your hardware:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_colocated_h100_bf16.yaml \
  --colocate \
  --port 8008
```

Equivalent H20 and H200 profiles are
`examples/configs/qwen3_omni_colocated_h20.yaml` and
`examples/configs/qwen3_omni_colocated_h200.yaml`. Use
`qwen3_omni_colocated_h100_fp8.yaml` for the validated H100 FP8 checkpoint.

## Send a request

This example combines image and text input and requests text plus audio. Use a
speech-mode server for audio output.

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

## Model capabilities

- Text-only mode runs the six-stage thinker pipeline. Speech mode adds the
  talker and code2wav stages for an eight-stage pipeline.
- Any supported input modality can produce text. Speech mode can additionally
  return audio with `modalities: ["text", "audio"]`.
- Native BF16, native FP8, and an AutoRound INT4 thinker with BF16
  talker/code2wav are supported in the documented topologies.
- The speech pipeline supports the shared streaming chat and server-VAD
  realtime transports.
- Disaggregated thinker TP=1 or TP=2 is supported. Colocated speech requires
  thinker TP=1 and explicit per-stage memory budgets.

See [Omni model usage](../basic_usage/qwen3_omni.md) for complete modality
examples, realtime events, model-specific placement measurements, precision
details, and sampling fields. See [Streaming](../user_guide/advanced_features/streaming.md)
and [Stage placement](../user_guide/deployment/stage_placement.md) for the shared
contracts.

## Known limitations

- A text-only server accepts `modalities: ["text", "audio"]` but returns no
  audio; use a speech-mode server when audio output is required.
- Use an empty message `content` when the request's semantic input is entirely
  in `images`, `audios`, or `videos`. Non-empty content is processed as an
  additional text input.
- Colocated speech does not support thinker TP=2. Use disaggregated placement.
- Requests are rejected when prompt tokens or prompt plus requested output meet
  or exceed the model context length.

## Benchmark

Use the benchmark matching the modality being qualified. For example, run MMMU
for image-plus-text input and text output:

```bash
python benchmarks/eval/benchmark_omni_mmmu.py \
  --model qwen3-omni \
  --host localhost \
  --port 8008
```

Qwen3-Omni CI also covers speech, MMSU, Video-MME, and Video-AMME paths with
separate benchmark entry points. Follow the
[benchmark methodology](../benchmarks/methodology.md) when publishing results.

## Related documentation

- [Omni model usage](../basic_usage/qwen3_omni.md)
- [Omni router](../basic_usage/omni_router.md)
- [Streaming](../user_guide/advanced_features/streaming.md)
- [Stage placement](../user_guide/deployment/stage_placement.md)
- [Benchmark methodology](../benchmarks/methodology.md)
- [Pipeline architecture](../developer_reference/pipeline.md)
- [Supported models](../supported_models.md)
