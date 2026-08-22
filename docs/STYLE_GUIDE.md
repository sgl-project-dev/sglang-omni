# Documentation guide

SGLang-Omni follows SGLang's task-oriented documentation structure while
treating multimodal pipelines, stage placement, streaming behavior, and
hardware qualification as first-class concepts.

This guide defines where information belongs and how to write model
documentation. Apply it to new pages and to existing pages when they are
materially updated. Do not reorganize unrelated legacy pages only to match the
physical layout below.

## Information architecture

The user-facing documentation has these sections:

| Section | Question it answers |
|---|---|
| Get started | How do I install SGLang-Omni and send one request? |
| User guide: serving | How does a public API behave? |
| User guide: advanced features | How does a reusable runtime capability work? |
| User guide: deployment | How do I place and size a multi-stage pipeline? |
| Supported models | Which models and capabilities are qualified? |
| Cookbook | How do I deploy and use one model? |
| Benchmarks | How do we measure correctness and performance? |
| Developer guide | How does the system work, and how do I extend it? |
| References | What does a configuration field, CLI flag, or error mean? |

The directory structure may migrate incrementally. Navigation and content
ownership define the contract even when a legacy page still has an older path.

## Content boundaries

### Get started

Keep installation, the first successful request, supported platforms, and
release notes here. Do not turn this section into a complete feature guide.

### Serving

Document endpoint fields, response formats, streaming protocols, and generic
error semantics here. A cookbook should mention only model-specific endpoint
behavior, such as an unsupported route.

### Advanced features

Document reusable behavior such as deterministic inference, admission control,
batching, prefill CUDA Graph, MPS/DP, stage offload, colocation, and weight
sharing here. Cookbooks state whether a model supports a feature and explain
only model-specific behavior.

### Deployment

Document stage placement, memory tuning, colocation, and multi-GPU resource
planning here. Keep model-specific validated topologies in the cookbook and
link to the shared deployment guidance.

### Supported models

Maintain the compact model, task, endpoint, pipeline, streaming, validated
hardware, maturity, validation, and cookbook view in
[Supported models](./supported_models.md). Add task-specific capability tables
only when their fields are meaningful for that task.

### Cookbook

A cookbook is a verified operational recipe for one model. It contains the
model's prerequisites, checked-in configuration, pipeline, first request,
capabilities, deviations from shared defaults, known limitations, canonical
benchmark command, and links to shared documentation. Use the
[model cookbook template](./cookbook/template.md).

Do not use a cookbook as a complete API reference, generic feature guide,
benchmark methodology document, performance investigation report, or
implementation design document.

### Benchmarks

Separate benchmark content into three layers:

1. A cookbook gives the canonical command for that model.
2. Benchmark documentation defines datasets, metrics, warmup, concurrency,
   streaming methodology, and reproducibility requirements.
3. A qualification report records the exact commit, model revision, hardware,
   dependencies, launch configuration, parameters, results, and analysis.

Do not keep large current-main result tables or tuning histories in permanent
cookbook prose.

### Developer guide

Keep architecture, pipeline lifecycle, stage interfaces, communication,
profiling, and model integration details here. Operational instructions belong
in a cookbook or deployment guide.

### References

Keep factual configuration, CLI, and error definitions concise. Tutorials and
model recommendations belong elsewhere.

## Sources of truth

Documentation explains how to use the system; it does not redefine facts owned
by code, configuration, CI, or benchmark artifacts.

| Information | Source of truth |
|---|---|
| Supported model registration | Model and pipeline registry |
| API request fields | Request schema and API implementation |
| CLI flags | CLI and configuration implementation |
| Runtime defaults | Runtime configuration |
| Model defaults | Model adapter and pipeline configuration |
| Recommended launch configuration | Checked-in example config |
| Benchmark commands | Benchmark implementation |
| CI qualification | Model CI definition |
| Performance numbers | Benchmark artifact or qualification report |
| Cookbook | Operational explanation and model-specific guidance |

Link to the stronger source when practical. If a cookbook recommends an
override, explain why that model needs it instead of copying the entire shared
configuration reference.

## Supported-model schema

The primary supported-model table uses these fields:

| Field | Meaning |
|---|---|
| Model | Public model or checkpoint family |
| Task | TTS, ASR, Omni, Music, Generation, or another concrete task |
| Endpoint | Public serving endpoint |
| Pipeline | High-level operational stage topology |
| Streaming | Yes, No, or Partial, with a short qualification when needed |
| Validated hardware | A measured or CI-tested configuration, not a theoretical minimum |
| Maturity | Experimental or Supported, as defined below |
| Validation | The recorded CI or performance evidence, as defined below |
| Cookbook | The operational recipe |

Maturity describes the maintenance expectation:

- **Experimental**: an implementation exists but is not regularly qualified.
- **Supported**: the configuration is maintained and expected to work.

Validation describes the evidence recorded for that configuration:

- **Not recorded**: no recurring CI or performance qualification is documented.
- **CI tested**: recurring model CI covers the documented configuration.
- **Performance qualified**: correctness and performance were measured under a
  defined, reproducible benchmark configuration.

Maturity and validation are independent. For example, an experimental model
can still have recurring CI coverage. Record each dimension explicitly instead
of treating CI or benchmark evidence as a stronger maturity level. If multiple
validation types apply and are documented, list each one.

## Hardware claims

State what was tested, not what might fit. Prefer "Validated on 1× RTX 4090 24
GB" or "CI tested on 1× H100" over "Minimum hardware: 24 GB GPU." If no
configuration has been recorded, write "Not yet recorded" rather than inferring
a minimum from model size or free memory.

## Writing style

- Use active voice and second person for procedures.
- Use sentence-case headings.
- Explain what a feature is before explaining how to configure it.
- Put prerequisites before commands.
- Keep examples copy/pasteable and use realistic values.
- Prefer one canonical example over several nearly identical variants.
- Reuse checked-in examples and configuration files when practical.
- Verify flags, defaults, API fields, and runtime behavior against their source
  of truth.
- Link to shared documentation instead of duplicating it.
- Use comments in examples only when they explain a non-obvious constraint.

Avoid marketing language, filler introductions, speculative support claims,
duplicated API tables, large historical benchmark tables, and implementation
details that do not help a user operate the model.

## New-model checklist

A new model should include:

- [ ] An entry in the supported-model matrix with evidence-based maturity and
      validation.
- [ ] A cookbook based on the standard template.
- [ ] A first-class pipeline topology.
- [ ] Validated hardware, or an explicit statement that it is not yet recorded.
- [ ] A checked-in server configuration or a copy/pasteable launch command.
- [ ] A runnable client or curl request.
- [ ] A streaming example when streaming is supported.
- [ ] Model-specific capabilities and known limitations.
- [ ] A canonical benchmark command.
- [ ] Identified CI coverage.
- [ ] Links to shared API, runtime, deployment, and benchmark documentation.
