# Admission control

Admission control bounds the work accepted by a pipeline before overload turns
into unbounded queue growth, excessive memory use, or unstable latency.

## Where limits apply

SGLang-Omni can reject or defer work at several layers:

| Layer | Responsibility |
|---|---|
| API/coordinator | Bound requests tracked by the whole pipeline |
| Request-build path | Bound preprocessing and request-construction backlog |
| Generation waiting queue | Bound requests waiting to enter the SGLang scheduler |
| Generation running set | Bound requests eligible to run in the AR stage |
| Model-local pools | Bound fixed slots, acoustic state, or other model resources |

A request can pass one layer and still wait or fail at a later model-specific
resource boundary. The model cookbook documents those additional limits.

## Generation-stage controls

The shared serve CLI exposes:

| Flag | Behavior |
|---|---|
| `--max-running-requests` | Override the generation stage's concurrent running slots |
| `--max-queued-requests` | Override waiting capacity before fast rejection |
| `--max-total-tokens` | Cap the generation-stage KV pool |
| `--cuda-graph-max-bs` | Set the largest captured decode batch size |

The first three values are related but not interchangeable. Running slots do
not reserve enough KV for every request's maximum length, and raising running
capacity does not raise the queue limit or CUDA Graph range automatically.

These generic flags target the pipeline's declared `generation` role. A
multi-engine pipeline can expose additional role-specific flags. For example,
Qwen3-Omni uses `--max-running-requests` for its talker and
`--thinker-max-running-requests` for its thinker.

## Queue behavior

`max_queued_requests` covers more than the visible SGLang waiting deque. The
shared scheduler counts requests that are building, awaiting admission,
backlogged before build, deferred, or already waiting. This prevents the
request-build path from becoming an unbounded queue ahead of the generation
stage.

When the bounded queue is full, the request is fast-rejected. Speech serving
maps queue saturation to HTTP 503. Other endpoints use their shared serving
error mapping. Clients should treat overload as retryable only with bounded
backoff and should not immediately replay the same burst.

Approximate pipeline capacity is often described as:

```text
running slots + queued requests
```

This is an admission ceiling, not a guarantee that every admitted request can
run simultaneously. KV length, model-local pools, stage placement, and request
shape can reduce effective concurrency.

## Choose limits

1. Start from the checked-in model configuration.
2. Select the largest running set that fits the validated KV and model-local
   memory budgets.
3. Keep a bounded queue large enough to absorb normal bursts without hiding
   sustained overload behind long waits.
4. Ensure compiled or captured batch ranges cover the intended running set
   when the model requires them.
5. Measure latency percentiles, failure counts, queue rejections, throughput,
   and memory under the expected arrival pattern.
6. Record the complete launch configuration with benchmark artifacts.

Closed-loop concurrency tests hold at most a fixed number of in-flight
requests; they do not model sustained overload. Use an open-loop or sustained
overshoot workload when validating rejection and recovery behavior.

## Model-specific defaults

Defaults belong to pipeline configuration, not this page. For example,
Qwen3-TTS currently declares 16 running and 16 queued requests, while other
pipelines use different limits or model-local slots. Keep those values in the
model cookbook only when they explain a model-specific operational choice.

## Related documentation

- [Qwen3-TTS cookbook](../../cookbook/qwen3_tts.md)
- [Qwen3-ASR cookbook](../../cookbook/qwen3_asr.md)
- [Stage placement](../deployment/stage_placement.md)
- [Benchmark methodology](../../benchmarks/methodology.md)
