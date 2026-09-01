# Developer Guide

Learn how SGLang-Omni's multi-stage runtime works, how to extend it, and how
to benchmark and maintain model integrations.

## Architecture and runtime

Start with the runtime architecture, then follow a request through the
pipeline, API server, and inter-stage communication layers.

```{toctree}
:maxdepth: 1
:caption: Architecture and runtime

Runtime Architecture <main.md>
Pipeline Lifecycle <pipeline.md>
API Server <apiserver_design.md>
Communication <communication.md>
Reference Encode Service <reference_encode_service.md>
```

## Development

Use these guides when adding configuration, integrating a TTS model, or
contributing documentation.

```{toctree}
:maxdepth: 1
:caption: Development

Adding a Parameter <adding_parameters.md>
TTS Model Integration <tts_model_integration.md>
Documentation Contribution Guide <../STYLE_GUIDE.md>
```

## Benchmarking and profiling

Profile request execution and record exact checkpoint, configuration,
hardware, and validation evidence without mixing CI details into user-facing
cookbooks.

```{toctree}
:maxdepth: 1
:caption: Benchmarking and profiling

Profiling <profiler.md>
Model Qualification <model_qualification.md>
```

## Maintenance

Operate administrative control paths and update the pinned SGLang stack.

```{toctree}
:maxdepth: 1
:caption: Maintenance

RL Admin Control <rl_admin_control.md>
Bumping the SGLang Version <bump_version.md>
```
