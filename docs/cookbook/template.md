# Model cookbook template

Copy this template when adding a model cookbook. Remove optional sections that
do not apply. Replace every placeholder before publishing the page.

````markdown
# Model name

One sentence describing the model and its primary use.

## At a glance

| Item | Value |
|---|---|
| Task | TTS / ASR / Omni / Music / Generation |
| Checkpoint(s) | `organization/model` |
| Endpoint(s) | `/v1/...` |
| Pipeline | preprocessing → engine → vocoder |
| Input | ... |
| Output | ... |
| Streaming | Yes / No / Partial |
| Validated hardware | Measured or CI-tested configuration |
| Maturity | Supported / Experimental |
| Validation | Not recorded / CI tested / Performance qualified |

## Install

Follow the shared installation guide, then install only model-specific
dependencies.

## Deploy

### Recommended configuration

Prefer a checked-in configuration.

```bash
sgl-omni serve \
  --model-path organization/model \
  --config examples/configs/model.yaml \
  --port 8000
```

### Other validated configurations

Include only meaningful alternatives that have been validated, such as a
consumer-GPU, multi-GPU, or memory-conservative configuration.

## Send a request

Provide one minimal working request. Include curl and Python only when both add
value.

```bash
curl ...
```

## Model capabilities

Include only relevant model-specific subsections, for example voice cloning,
language hints, streaming, long audio, diarization, multimodal input, or voice
design.

## Model-specific configuration

Document behavior that differs from shared runtime defaults. Do not copy the
complete server configuration reference.

## Known limitations

- List concrete unsupported or constrained behavior.

## Benchmark

Provide the canonical benchmark command and link to benchmark methodology or a
qualification report.

```bash
python -m benchmarks.eval.example ...
```

## Related documentation

- API or serving guide
- Runtime feature guide
- Deployment guide
- Benchmark methodology
- Runnable examples
- Developer documentation
````
