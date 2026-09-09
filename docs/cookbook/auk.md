# AuK

The pipeline runs CPU preprocessing followed by one serial GPU engine:
Qwen2.5-Omni conditioning, stochastic VAE reference encoding, DiT sampling,
and decoding with the same VAE. It returns a complete 24 kHz waveform.

```bash
sgl-omni serve --config examples/configs/auk.yaml
```

The config downloads `tencent/AuK` and the separate
`Qwen/Qwen2.5-Omni-3B` checkpoint. Set
`auk_engine.factory.text_encoder_path` to use a local Qwen checkpoint.
Use `model_path: tencent/AuK-Flash` for the distilled model; it always uses
the released four-step time grid with CFG disabled.

## Speech API

`input` is the text to speak; `instructions` describes the voice. Without
instructions, the description defaults to `A clear, natural voice.`
With `ref_audio` (or one structured reference), AuK uses the upstream
same-voice TTS instruction instead of the voice description.

TTS requires an explicit target duration in seconds. Reference duration is
independent of target duration. Durations are rounded up to latent frames
(20 ms) and bounded by the configured maximum (30 seconds by default).

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

`/generate` accepts a raw AuK instruction in `prompt`, with duration in
`stage_params.auk_engine.gen_seconds` and reference audio in
`metadata.tts_params.ref_audio`. Set `output_modalities` to `["audio"]`.
Raw editing requests with reference audio may omit the duration to use the
source duration, as upstream does.
`nfe`, `cfg_strength`, and `sway_sampling_coef` are server factory settings;
request overrides are rejected. Engine batching and streaming are unsupported.

As upstream, `seed` controls target noise. Reference posterior sampling uses
the process RNG and is stochastic; a request seed alone does not make reference
conditioning deterministic.

## Parity Test

The test compares real upstream and serving outputs at the reference latent,
fused Qwen conditioning, generated latent, and waveform boundaries. It lowers
speech requests through `SpeechRequestValidator` and `Client` before running
the terminal engine, covering both reference and instruction-only TTS.
Both runs start with the same process RNG state as well as the same request seed.

Install the repository dependencies plus `torchdiffeq` and `qwen-omni-utils`,
then run on a GPU with enough memory for both model instances:

```bash
git clone https://github.com/Tencent-Hunyuan/AuK.git /tmp/AuK
git -C /tmp/AuK checkout d9f30ffe4231dbc90b48cc83a35d310fece0b060
AUK_UPSTREAM_SOURCE=/tmp/AuK \
AUK_PARITY_CHECKPOINT=tencent/AuK \
python -m pytest tests/test_model/test_auk_parity.py -v
```

Repeat with `AUK_PARITY_CHECKPOINT=tencent/AuK-Flash` for the distilled recipe.
`AUK_QWEN_CHECKPOINT` optionally selects a local encoder checkpoint. The test
skips unless both required environment variables are set.

Validated on 2026-09-09 with an NVIDIA H200, PyTorch 2.13.0 and Transformers
5.12.1: all four cases (AuK/AuK-Flash, with/without reference) passed, with
zero maximum absolute error at every compared boundary. This checks the
serving port against upstream under the same installed dependency versions.

## Attribution

The model implementation derives from Tencent-Hunyuan/AuK at the revision above.
Its MIT copyright and permission notice are preserved in the model directory's
`LICENSE` file.
The VAE also retains NVIDIA attribution and the Apache-2.0 attribution for
alias-free-torch components.
