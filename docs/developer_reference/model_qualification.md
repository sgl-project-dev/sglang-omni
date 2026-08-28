# Model qualification evidence

This maintainer-facing catalog records exact model, configuration, hardware,
and validation evidence. The user-facing [supported-model matrix](../supported_models.md)
describes model families and accelerator support without duplicating these
implementation details.

A row qualifies only the checkpoint, launch configuration, and hardware named
in that row. A checked-in profile without a completed runtime record is useful
deployment input, but it is not validation evidence.

## Validation types

- **Not recorded**: no recurring CI or completed manual validation is linked.
- **Profile available**: a checked-in launch profile exists, but no completed
  runtime validation is claimed.
- **Manually validated**: current-main documentation records validation of the
  exact configuration, but it is not covered by a recurring gate.
- **CI tested**: recurring model CI covers the documented configuration.
- **Performance qualified**: correctness and performance were measured under a
  defined, reproducible benchmark configuration.

## Configuration evidence

| Model | Exact checkpoint / revision | Configuration and material overrides | Hardware | Validation type | CI / workflow / report evidence |
|---|---|---|---|---|---|
| Higgs Audio v3 | `bosonai/higgs-audio-v3-tts-4b` | Model-derived default | Not recorded | Not recorded | [Cookbook](../cookbook/higgs_tts.md) |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | [Default profile](../../examples/configs/qwen3_tts_1_7b.yaml) | Not recorded | Profile available | [Checked-in profile](../../examples/configs/qwen3_tts_1_7b.yaml) |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`; CI does not pin a model revision | Two router workers; each uses `tts_engine.engine.{max_running_requests,cuda_graph_max_bs,torch_compile_max_bs}=64`, a separate vocoder process, and TTS-engine/vocoder GPU fractions of `0.85`/`0.10` | 2× H100, one per worker | CI tested | [TTS CI preset](../../tests/test_model/tts_ci_config.py), [router fixture](../../tests/test_model/test_tts_ci.py), [H100 workflow](../../.github/workflows/test-tts-ci.yaml) |
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B`; CI does not pin a model revision | Two router workers using the model-derived default | 2× H100, one per worker | CI tested | [ASR CI preset](../../tests/test_model/asr_ci_config.py), [router fixture](../../tests/test_model/test_asr_ci_seedtts.py), [H100 workflow](../../.github/workflows/test-asr-ci.yaml) |
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` | [RTX 4090 profile](../../examples/configs/qwen3_asr_rtx4090.yaml) | RTX 4090 24 GB | Manually validated | [Checked-in validated profile](../../examples/configs/qwen3_asr_rtx4090.yaml) |
| Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct`; CI does not pin a model revision | Two router workers using the [H100 BF16 colocated profile](../../examples/configs/qwen3_omni_colocated_h100_bf16.yaml) plus `preprocessing.factory.max_seq_len=32768` and `thinker.factory.max_seq_len=32768` | 2× H100, one per worker | CI tested | [H100 workflow](../../.github/workflows/test-qwen3-omni-ci.yaml), [CI fixture](../../tests/test_model/conftest.py) |
| Qwen3-Omni | `marksverdhei/Qwen3-Omni-30B-A3B-FP8`; CI does not pin a model revision | Two router workers using the [H100 FP8 colocated profile](../../examples/configs/qwen3_omni_colocated_h100_fp8.yaml) plus `preprocessing.factory.max_seq_len=32768` and `thinker.factory.max_seq_len=32768` | 2× H100, one per worker | CI tested | [H100 workflow](../../.github/workflows/test-qwen3-omni-ci.yaml), [CI fixture](../../tests/test_model/conftest.py) |
| Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | [H20 colocated profile](../../examples/configs/qwen3_omni_colocated_h20.yaml) | 1× H20 | Profile available | [Checked-in profile](../../examples/configs/qwen3_omni_colocated_h20.yaml) |
| Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | [H200 colocated profile](../../examples/configs/qwen3_omni_colocated_h200.yaml) | 1× H200 | Profile available | [Checked-in profile](../../examples/configs/qwen3_omni_colocated_h200.yaml) |

The recurring `higgs` TTS CI preset covers `bosonai/higgs-tts-3-4b`, not
`bosonai/higgs-audio-v3-tts-4b`, so it is not validation evidence for the
Higgs Audio v3 row.

## Maintenance

Update this page only from current-main evidence. Link recurring CI from the
model preset and workflow, manual validation from an explicit checked-in record
or durable report, and performance qualification from a reproducible benchmark
artifact. Do not infer validation from a backend abstraction, model
registration, or an unvalidated checked-in profile.
