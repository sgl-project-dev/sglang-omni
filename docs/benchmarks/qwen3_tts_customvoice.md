# Qwen3-TTS CustomVoice benchmark

## 1.7B CustomVoice

Qwen3-TTS-12Hz-1.7B-CustomVoice on the full Seed-TTS-Eval EN and ZH splits, concurrency 16, with 16 warmup requests per language/mode and `max_new_tokens=2048`. EN used Ryan/English and ZH used Vivian/Chinese, without reference audio or instructions. WER/CER was scored with Qwen3-ASR-1.7B at concurrency 32. Hardware: 1× H200 141 GB, BF16, TP1. Sampling overrides and seed were unset.

The server used `--tts_engine.engine.max_running_requests 64`, `--tts_engine.engine.cuda_graph_max_bs 64`, `--tts_engine.engine.torch_compile_max_bs 64`, `--vocoder.process vocoder`, `--tts_engine.gpu_memory_fraction 0.85`, and `--vocoder.gpu_memory_fraction 0.10`; `torch.compile` remained disabled. Streaming used the default `1 -> 2 -> 4` chunk ramp without a request-level override. Each language/mode was measured once, in non-streaming EN/ZH then streaming EN/ZH order on the same warmed server. The target GPU had no external GPU process during timed windows; host CPU, memory, and I/O were shared with another profiling task.

| Metric | Non-streaming EN | Non-streaming ZH | Streaming EN | Streaming ZH |
|---|---:|---:|---:|---:|
| Samples | 1088 | 2020 | 1088 | 2020 |
| Corpus WER/CER | 1.608% | 0.984% | 2.085% | 0.927% |
| Corpus WER/CER (excl. >50% outliers) | 1.359% | 0.984% | 1.454% | 0.927% |
| Samples above 50% WER/CER | 3 | 0 | 4 | 0 |
| UTMOS | 4.1723 | 3.1824 | 4.1500 | 3.1789 |
| QPS | 14.788 | 13.307 | 10.098 | 8.333 |
| Latency mean (s) | 1.075 | 1.198 | 1.573 | 1.915 |
| RTF mean | 0.2335 | 0.2079 | 0.3380 | 0.3332 |
| TTFA mean (s) | N/A | N/A | 0.1213 | 0.1047 |

Corpus WER (EN) / CER (ZH) is total edit distance divided by total reference words / characters and includes every sample. The filtered row excludes samples whose own WER/CER exceeds 50% and recomputes the corpus rate. UTMOS is the mean predicted audio-quality score, not a listening-test score. These independently sampled runs are not a paired comparison of streaming and non-streaming quality; the Base result also uses different conditioning and a different ASR evaluator.

TTFA measures arrival of the first PCM payload, not the first audible speech; its mean payload duration was 80 ms in both languages. All 3,108 streaming requests were continuity-scored: 98.99% EN and 93.76% ZH had no playback underrun longer than 50 ms. Maximum underrun was 396.1 ms EN and 2072.0 ms ZH. The default ramp therefore does not guarantee uninterrupted playback at concurrency 16; see [First-audio chunk ramp](../cookbook/qwen3_tts.md#first-audio-chunk-ramp) for the buffering tradeoff.
