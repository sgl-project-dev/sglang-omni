# SPDX-License-Identifier: Apache-2.0
"""Selective, bounded-memory HF/MLX checkpoint loading for native Torch MPS.

Only safetensor headers are indexed. Owned tensors are read, quantized and
installed immediately; routed stacks are sliced *before* reading their data.
The supported source formats are dense HF and MLX affine uint32, not arbitrary
Hugging Face quantization formats such as compressed-tensors, AWQ or GPTQ.
"""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .torch_mps_quantization import (
    MpsQuantizedExperts,
    MpsQuantizedLinear,
    dequantize_affine_rows,
)

logger = logging.getLogger(__name__)
_ROW_CHUNK = 1024
_COMPONENT_PREFIXES = {
    "thinker.": "thinker",
    "talker.": "talker",
    "code2wav.": "code2wav",
    "thinker.visual.": "vision",
    "thinker.vision_tower.": "vision",
    "visual.": "vision",
    "thinker.audio_tower.": "audio",
    "audio_tower.": "audio",
}


def _json_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint metadata key {key!r}")
        result[key] = value
    return result


def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_json_unique)


def _source_format(directory):
    config_path = directory / "config.json"
    if not config_path.exists() and directory.name in set(_COMPONENT_PREFIXES.values()):
        config_path = directory.parent / "config.json"
    config = _read_json(config_path) if config_path.exists() else {}
    configurations = [
        config[key]
        for key in ("quantization", "quantization_config")
        if config.get(key) is not None
    ]
    if not configurations:
        return "hf_dense", None
    quant = configurations[0]
    if any(item != quant for item in configurations):
        raise ValueError("Conflicting checkpoint quantization metadata")
    if not isinstance(quant, dict):
        # Invalid serialized metadata is a checkpoint value error, not an API type error.
        raise ValueError("Unsupported checkpoint quantization metadata")  # noqa: TRY004
    if (
        set(quant) - {"bits", "group_size", "mode"}
        or quant.get("bits") not in (4, 8)
        or quant.get("group_size") not in (32, 64, 128, 256)
        or quant.get("mode", "affine") != "affine"
    ):
        raise ValueError(
            "Unsupported checkpoint quantization format: native MPS accepts dense HF "
            "or MLX affine uint32 (bits 4/8, groups 32/64/128/256), not "
            "compressed-tensors, AWQ or GPTQ"
        )
    return "mlx_affine", quant


def _canonical_key(key):
    key = key.replace("thinker.language_model.model.", "thinker.model.", 1)
    key = key.replace("thinker.language_model.lm_head.", "thinker.lm_head.", 1)
    return key.replace(".switch_mlp.", ".experts.")


@dataclass(frozen=True)
class _Ref:
    key: str
    index: tuple | None = None


class _Source:
    def __init__(self, directory, prefixes, stack):
        self.format, self.quantization = _source_format(directory)
        self.entries = {}
        self.used = set()
        self.handles = {}
        self.shard_keys = {}
        index = directory / "model.safetensors.index.json"
        weight_map = _read_json(index).get("weight_map") if index.exists() else None
        if index.exists() and (not isinstance(weight_map, dict) or not weight_map):
            raise ValueError("Checkpoint index must contain a nonempty weight_map")

        def local_name(raw):
            name = _canonical_key(raw)
            for prefix in prefixes:
                if name.startswith(prefix):
                    local = name[len(prefix) :]
                    # A text-only holder does not own the thinker's encoder towers.
                    if prefix == "thinker." and local.startswith(
                        ("visual.", "vision_tower.", "audio_tower.")
                    ):
                        return None
                    if self.format == "mlx_affine":
                        if local.startswith("deepstack_merger_list."):
                            local = (
                                "merger_list." + local[len("deepstack_merger_list.") :]
                            )
                        if local.startswith(("merger.", "merger_list.")):
                            local = local.replace(".norm.", ".ln_q.")
                            local = local.replace(".linear_fc1.", ".mlp.0.")
                            local = local.replace(".linear_fc2.", ".mlp.2.")
                    return local
            return None

        def open_shard(filename):
            path = (directory / filename).resolve()
            if path.parent != directory.resolve():
                raise ValueError(
                    f"Checkpoint shard must be in its selected directory: {filename}"
                )
            if filename not in self.handles:
                if not path.is_file():
                    raise ValueError(f"Missing checkpoint shard {filename}")
                self.handles[filename] = stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )
                self.shard_keys[filename] = set(self.handles[filename].keys())
            return self.handles[filename]

        def add(raw, filename):
            name = local_name(raw)
            if name is None:
                return
            if name in self.entries:
                raise ValueError(f"Duplicate checkpoint tensors map to {name!r}")
            handle = open_shard(filename)
            if raw not in self.shard_keys[filename]:
                raise ValueError(f"Missing indexed checkpoint tensor {raw!r}")
            self.entries[name] = (handle, raw)

        if weight_map is not None:
            for raw, filename in weight_map.items():
                if local_name(raw) is not None:
                    add(raw, filename)
        else:
            for path in sorted(directory.glob("*.safetensors")):
                open_shard(path.name)
                for raw in sorted(self.shard_keys[path.name]):
                    add(raw, path.name)
        if not self.entries:
            raise ValueError(f"Missing checkpoint weights for prefixes {prefixes}")

    def shape(self, ref):
        if ref.key not in self.entries:
            raise ValueError(f"Missing checkpoint weight {ref.key!r}")
        handle, raw = self.entries[ref.key]
        shape = tuple(handle.get_slice(raw).get_shape())
        if ref.index is None:
            return shape
        result = []
        for dim, selector in zip(shape, ref.index):
            if isinstance(selector, int):
                if not 0 <= selector < dim:
                    raise ValueError(
                        f"Expert index outside checkpoint shape for {ref.key}"
                    )
            else:
                result.append(len(range(*selector.indices(dim))))
        return tuple(result) + shape[len(ref.index) :]

    def read(self, ref):
        self.shape(ref)
        self.used.add(ref.key)
        handle, raw = self.entries[ref.key]
        view = handle.get_slice(raw)
        if not view.get_shape():
            return handle.get_tensor(raw)
        return view[:] if ref.index is None else view[ref.index]

    def affine_refs(self, ref):
        if ref.key != "weight" and not ref.key.endswith(".weight"):
            return None
        base = ref.key[: -len("weight")]
        keys = (base + "scales", base + "biases")
        found = [key in self.entries for key in keys]
        if any(found):
            if not all(found):
                raise ValueError(f"Missing affine scales/biases for {ref.key}")
            if self.format != "mlx_affine":
                raise ValueError(
                    "Unsupported packed checkpoint without MLX affine metadata"
                )
            return tuple(_Ref(key, ref.index) for key in keys)
        return None

    def dense_shape(self, ref):
        shape = self.shape(ref)
        if self.affine_refs(ref):
            return (*shape[:-1], shape[-1] * (32 // self.quantization["bits"]))
        return shape

    def check_shape(self, ref, expected):
        actual = self.dense_shape(ref)
        if tuple(actual) != tuple(expected):
            raise ValueError(
                f"Shape mismatch for {ref.key}: {actual}, expected {tuple(expected)}"
            )

    def linear(self, ref, shape, *, bits, dtype, device, bias=None):
        self.check_shape(ref, shape)
        affine = self.affine_refs(ref)
        if affine:
            scales, biases = (self.read(item) for item in affine)
            return MpsQuantizedLinear.from_affine(
                self.read(ref),
                scales,
                biases,
                bits=self.quantization["bits"],
                group_size=self.quantization["group_size"],
                target_bits=bits,
                dtype=dtype,
                device=device,
                bias=bias,
            )
        return MpsQuantizedLinear.from_float(
            self.read(ref), bits=bits, dtype=dtype, device=device, bias=bias
        )

    def dense(self, ref, *, dtype, device):
        affine = self.affine_refs(ref)
        if not affine:
            tensor = self.read(ref)
            if not tensor.is_floating_point() and tensor.dtype == torch.uint32:
                raise ValueError(
                    f"Unsupported packed tensor without affine metadata: {ref.key}"
                )
            return tensor.to(device=device, dtype=dtype)
        shape = self.dense_shape(ref)
        if ref.index is not None or len(shape) != 2:
            raise ValueError(f"Dense affine decoding expects a matrix: {ref.key}")
        # Allocate only the final embedding/projection, plus bounded row scratch.
        output = torch.empty(shape, device=device, dtype=dtype)
        for start in range(0, shape[0], _ROW_CHUNK):
            rows = (slice(start, min(start + _ROW_CHUNK, shape[0])),)
            values = [self.read(_Ref(item.key, rows)) for item in (ref, *affine)]
            output[start : start + _ROW_CHUNK].copy_(
                dequantize_affine_rows(
                    *values,
                    bits=self.quantization["bits"],
                    group_size=self.quantization["group_size"],
                ).to(device=device, dtype=dtype)
            )
        return output

    def finish(self):
        unexpected = self.entries.keys() - self.used
        if unexpected:
            raise ValueError(
                f"Unexpected checkpoint weights: {sorted(unexpected)[:12]}"
            )


def _join(path, name):
    return f"{path}.{name}" if path else name


def _expert_ref(source, path, projection, expert, count, intermediate, hidden):
    expected = (
        (hidden, intermediate) if projection == "down_proj" else (intermediate, hidden)
    )
    candidates = []
    numbered = _join(path, f"{expert}.{projection}.weight")
    if numbered in source.entries:
        candidates.append(_Ref(numbered))
    for suffix in (projection, projection + ".weight"):
        stacked = _join(path, suffix)
        if stacked in source.entries:
            source.check_shape(_Ref(stacked), (count, *expected))
            candidates.append(_Ref(stacked, (expert,)))
    if projection != "down_proj":
        for suffix in ("gate_up_proj", "gate_up_proj.weight"):
            fused = _join(path, suffix)
            if fused in source.entries:
                source.check_shape(_Ref(fused), (count, 2 * intermediate, hidden))
                offset = 0 if projection == "gate_proj" else intermediate
                candidates.append(
                    _Ref(fused, (expert, slice(offset, offset + intermediate)))
                )
    if not candidates:
        raise ValueError(f"Missing checkpoint expert {path}.{expert}.{projection}")
    if len(candidates) != 1:
        raise ValueError(
            f"Duplicate checkpoint expert representations for {path}.{expert}.{projection}"
        )
    return candidates[0], expected


def _conv_permutation(module):
    if isinstance(module, nn.ConvTranspose1d):
        return (2, 0, 1)
    if isinstance(module, nn.Conv1d):
        return (0, 2, 1)
    if isinstance(module, nn.Conv2d):
        return (0, 3, 1, 2)
    if isinstance(module, nn.Conv3d):
        return (0, 4, 1, 2, 3)
    return None


def load_quantized_mps_module(
    module: nn.Module,
    model_path: str,
    *,
    prefix: str | tuple[str, ...],
    bits: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> nn.Module:
    """Load an owned meta component shell into native quantized MPS modules.

    Prefix aliases are exclusive: two source tensors mapping to the same local
    name are errors. Real initialization buffers survive when not serialized.
    Tied parameters may omit one serialized alias; other missing weights fail.
    ``text_projection`` and ``hidden_projection`` remain dense for CPU prefill.
    """
    MpsQuantizedLinear._validate_target(bits, dtype, device)
    directory = Path(model_path).expanduser()
    if not directory.is_dir():
        from huggingface_hub import snapshot_download

        directory = Path(snapshot_download(model_path))
    prefixes = (prefix,) if isinstance(prefix, str) else prefix
    if not prefixes or any(not isinstance(value, str) for value in prefixes):
        raise ValueError(
            "Checkpoint prefix must be a string or nonempty tuple of strings"
        )
    if not (directory / "model.safetensors.index.json").exists() and not any(
        directory.glob("*.safetensors")
    ):
        components = {_COMPONENT_PREFIXES.get(value) for value in prefixes}
        if len(components) == 1 and None not in components:
            local = directory / next(iter(components))
            if local.is_dir():
                directory, prefixes = local, ("",)
    aliases = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    assigned = {}
    counts = {"linears": 0, "experts": 0}

    with ExitStack() as stack:
        source = _Source(directory, prefixes, stack)

        def parameter_ref(name, original):
            if name in source.entries:
                return _Ref(name)
            for alias in aliases.get(id(original), ()):
                if alias in source.entries:
                    return _Ref(alias)
            raise ValueError(f"Missing checkpoint weight {name!r}")

        def visit(current, path, keep_dense=False):
            keep_dense = keep_dense or path.split(".")[-1] in (
                "text_projection",
                "hidden_projection",
                "gate",
                "router",
            )
            gate_up = current._parameters.get("gate_up_proj")
            down = current._parameters.get("down_proj")
            if gate_up is not None and gate_up.ndim == 3 and down is not None:
                count, doubled, hidden = gate_up.shape
                intermediate = doubled // 2
                if doubled % 2 or tuple(down.shape) != (count, hidden, intermediate):
                    raise ValueError(f"Invalid expert shell shapes at {path}")
                projections = ([], [], [])
                for expert in range(count):
                    for target, name in zip(
                        projections, ("gate_proj", "up_proj", "down_proj")
                    ):
                        ref, shape = _expert_ref(
                            source, path, name, expert, count, intermediate, hidden
                        )
                        target.append(
                            source.linear(
                                ref, shape, bits=bits, dtype=dtype, device=device
                            )
                        )
                        counts["linears"] += 1
                counts["experts"] += count
                return MpsQuantizedExperts(*projections, current.act_fn)
            if isinstance(current, nn.Linear) and not keep_dense:
                weight_name = _join(path, "weight")
                # An omitted tied output head keeps its dense embedding alias.
                if weight_name in source.entries:
                    bias = None
                    if current.bias is not None:
                        bias_ref = parameter_ref(_join(path, "bias"), current.bias)
                        source.check_shape(bias_ref, current.bias.shape)
                        bias = source.dense(bias_ref, dtype=dtype, device=device)
                    replacement = source.linear(
                        _Ref(weight_name),
                        current.weight.shape,
                        bits=bits,
                        dtype=dtype,
                        device=device,
                        bias=bias,
                    )
                    counts["linears"] += 1
                    return replacement
            for name, original in list(current._parameters.items()):
                if original is None:
                    continue
                full = _join(path, name)
                if id(original) in assigned and full not in source.entries:
                    setattr(current, name, assigned[id(original)])
                    continue
                ref = parameter_ref(full, original)
                permutation = (
                    _conv_permutation(current)
                    if (name == "weight" and source.format == "mlx_affine")
                    else None
                )
                if permutation is None:
                    source.check_shape(ref, original.shape)
                else:
                    source_shape = source.dense_shape(ref)
                    if len(source_shape) != len(permutation) or tuple(
                        source_shape[i] for i in permutation
                    ) != tuple(original.shape):
                        raise ValueError(f"Shape mismatch for MLX convolution {full}")
                value = source.dense(
                    ref,
                    dtype=dtype if original.is_floating_point() else original.dtype,
                    device=device,
                )
                if permutation is not None:
                    value = value.permute(permutation).contiguous()
                parameter = nn.Parameter(value, requires_grad=False)
                setattr(current, name, parameter)
                assigned[id(original)] = parameter
            for name, original in list(current._buffers.items()):
                if original is None:
                    continue
                full = _join(path, name)
                if full in source.entries:
                    ref = _Ref(full)
                    source.check_shape(ref, original.shape)
                    value = source.dense(
                        ref,
                        dtype=dtype if original.is_floating_point() else original.dtype,
                        device=device,
                    )
                elif original.is_meta:
                    raise ValueError(f"Missing initialization/checkpoint buffer {full}")
                else:
                    value = original.to(
                        device=device,
                        dtype=dtype if original.is_floating_point() else original.dtype,
                    )
                setattr(current, name, value)
            for name, child in list(current.named_children()):
                setattr(current, name, visit(child, _join(path, name), keep_dense))
            return current

        module = visit(module, "")
        source.finish()
    module.eval()
    logger.info(
        "Qwen3-Omni Torch MPS quantization bits=%d prefix=%s "
        "quantized_linears=%d quantized_experts=%d "
        "linear_class=%s expert_class=%s source=%s",
        bits,
        prefix,
        counts["linears"],
        counts["experts"],
        MpsQuantizedLinear.__name__,
        MpsQuantizedExperts.__name__,
        source.format,
    )
    return module
