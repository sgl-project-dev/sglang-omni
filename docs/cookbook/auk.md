# AuK

[AuK](https://huggingface.co/tencent/AuK) supports instruction-driven speech generation and editing. SGLang-Omni runs CPU preprocessing followed by one serial GPU engine and returns 24 kHz mono audio. Batching, streaming, and CUDA graphs are not implemented for this pipeline.

The released `tencent/AuK` checkpoint uses:

| Component | Configuration |
|---|---|
| Conditioner | Frozen Qwen2.5-Omni-3B Thinker, text and audio only |
| DiT | Flux-style MMDiT: 10 double-stream blocks, 20 single-stream blocks, dim=1536, 24 heads |
| VAE | Shared reference encoder and audio decoder; 50 Hz, 64-channel latents |

Architecture settings come from the checkpoint's `config.yaml`.

## Start the Server

Follow [Installation](../get_started/installation.md), then run from the repository root:

```bash
sgl-omni serve --config examples/configs/auk.yaml --port 8000
```

The server downloads AuK and the separate `Qwen/Qwen2.5-Omni-3B` encoder as needed. To use a local encoder, set `stages.auk_engine.factory.text_encoder_path` in the YAML.

For AuK-Flash, change `model_path` to `tencent/AuK-Flash`. It fixes inference to the released four-step time grid with CFG disabled, ignoring the factory's `nfe`, `cfg_strength`, and `sway_sampling_coef` values.

## Speech Generation

`/v1/audio/speech` takes the text in `input`. Without reference audio, `instructions` describes the voice and defaults to `A clear, natural voice.` An explicit target duration is required in this mode:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Welcome home.",
    "instructions": "warm, relaxed female voice",
    "stage_params": {"auk_engine": {"gen_seconds": 3}},
    "seed": 1234,
    "response_format": "wav"
  }' --output speech.wav
```

For voice cloning, provide `ref_audio` or one structured reference. The model uses a same-voice instruction and ignores the voice description. Local paths are resolved on the server; HTTP(S), file URLs, and audio data URLs are also accepted.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Welcome home.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "seed": 1234,
    "response_format": "wav"
  }' --output speech.wav
```

When `gen_seconds` is omitted, voice cloning requires the reference transcript (`ref_text`, or `references[0].text`). Target duration is estimated as:

```text
target_seconds = reference_seconds × UTF8_bytes(input) / UTF8_bytes(ref_text)
```

Explicit `gen_seconds` takes priority and must be positive. Target duration rounds up to 20 ms frames and is capped at 30 seconds by default. To change the cap, set `max_seconds` in **both** `stages.preprocessing.factory` and `stages.auk_engine.factory`.

## Speech Editing

`/generate` accepts a raw AuK instruction in `prompt` and returns JSON. Set `output_modalities` to `["audio"]` and supply reference audio through `metadata.tts_params.ref_audio`:

```bash
curl http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Remove the background noise.",
    "metadata": {"tts_params": {"ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"}},
    "output_modalities": ["audio"]
  }'
```

Override duration with `stage_params.auk_engine.gen_seconds`. Otherwise, editing uses the source's complete 20 ms frames, subject to the duration cap. Raw requests without reference audio or explicit duration default to 5 seconds.

## Sampling

Base AuK uses Euler integration with factory defaults `nfe=32`, `cfg_strength=2.0`, and `sway_sampling_coef=-1.0`. Configure these under `stages.auk_engine.factory`; request overrides of these settings and `max_seconds` are rejected. Qwen and DiT use BF16 autocast by default; the VAE runs in FP32.

`seed` controls target noise only. Reference VAE posterior sampling uses the process RNG, so a request seed alone does not make voice cloning deterministic. Multiple structured references are rejected.

## SeedTTS Evaluation

The standard benchmark detects `tencent/AuK` and `tencent/AuK-Flash`, selects the AuK server config, and defaults to the full English dataset, concurrency 1, one warmup, and seed 1234. It estimates duration from the reference audio and transcript, then automatically starts and stops the TTS and ASR servers:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.benchmark_tts_seedtts \
  --model tencent/AuK --output-dir results/auk_en
```

Use `--max-samples` and `--sample-offset` for a subset. `--generate-only` and `--transcribe-only` run individual phases; add `--use-existing-server` to either mode to use a running server. Explicit CLI options override the AuK defaults.

`wer_results.json` includes full sample mean WER, `wer_below_50_per_sample_mean` (excluding samples strictly above 50%), and `n_above_50_pct_wer`. Corpus WER is reported separately and is word-weighted.

## Upstream Parity

The checkpoint test compares reference latents, fused Qwen conditioning, generated latents, and waveforms with upstream, with and without reference audio. It aligns both process RNG and request seed. Install `torchdiffeq` and `qwen-omni-utils` in addition to the serving dependencies, and use a GPU with memory for both implementations:

```bash
git clone https://github.com/Tencent-Hunyuan/AuK.git /tmp/AuK
git -C /tmp/AuK checkout d9f30ffe4231dbc90b48cc83a35d310fece0b060
AUK_UPSTREAM_SOURCE=/tmp/AuK \
AUK_PARITY_CHECKPOINT=tencent/AuK \
python -m pytest tests/test_model/test_auk_parity.py -v
```

Set `AUK_PARITY_CHECKPOINT=tencent/AuK-Flash` to test Flash. `AUK_QWEN_CHECKPOINT` optionally selects a local encoder. The test skips unless both `AUK_UPSTREAM_SOURCE` and `AUK_PARITY_CHECKPOINT` are set.

## Attribution

The implementation derives from [Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK) at the revision above. Its MIT notice is preserved in `sglang_omni/models/auk/LICENSE`. The VAE source also retains NVIDIA and alias-free-torch attribution.
