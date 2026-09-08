# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni quantization glue for SGLang-owned quantization.

SGLang owns quantization end-to-end: it parses `quantization_config`,
constructs quantized layers, and executes post-load hooks. This module only
provides the multi-stage compatibility SGLang cannot infer by itself:
stage-local AutoRound config normalization and scale preprocessing for custom
weight loaders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch

# A weight preprocessor maps `(target_name, loaded_weight) -> loaded_weight`.
WeightPreprocessor = Callable[[str, "torch.Tensor"], "torch.Tensor"]


_QUANT_METADATA_KEYS: tuple[str, ...] = ("quantization_config", "compression_config")
_NESTED_QUANT_CONFIG_ATTRS: tuple[str, ...] = (
    "text_config",
    "thinker_config",
    "talker_config",
)
_STAGE_PREFIX_BY_ARCH: dict[str, str] = {
    "Qwen3OmniThinkerForCausalLM": "thinker.",
    "Qwen3ASRForConditionalGeneration": "thinker.",
    "Qwen3OmniTalker": "talker.",
}

__all__ = [
    "resolve_quant_config",
    "quant_method_name",
    "is_fp8_block_quant",
    "convert_fp8_weight_scale_inv",
    "get_weight_preprocessor",
    "needs_quant_config_normalization",
    "get_stage_local_mxfp_policy",
    "resolve_stage_local_mxfp_profile",
    "requires_fp8_gemm_initialization",
    "validate_stage_local_mxfp_config",
    "normalize_quant_config",
    "disable_quantization_for_full_precision_stage",
]

_MXFP_NAMES = {"mx", "mx_fp", "mxfp"}


@dataclass(frozen=True)
class StageLocalMxfpPolicy:
    """Model-family rules that cannot be inferred from root quant metadata."""

    architectures: frozenset[str]
    quantized_stage_prefix: str
    full_precision_architectures: frozenset[str]
    full_precision_stage_prefixes: tuple[str, ...]
    mxfp4_expert_pattern: re.Pattern[str]
    model_label: str
    mxfp8_cuda_moe_backend: str
    mixed_cuda_moe_backend: str


@dataclass(frozen=True)
class StageLocalMxfpProfile:
    """Runtime properties derived from a registered stage-local checkpoint."""

    policy: StageLocalMxfpPolicy
    has_mxfp4_experts: bool

    @property
    def preferred_cuda_moe_backend(self) -> str:
        return (
            self.policy.mixed_cuda_moe_backend
            if self.has_mxfp4_experts
            else self.policy.mxfp8_cuda_moe_backend
        )


_STAGE_LOCAL_MXFP_POLICIES = (
    StageLocalMxfpPolicy(
        architectures=frozenset(
            {
                "Qwen3OmniMoeForConditionalGeneration",
                "Qwen3OmniThinkerForCausalLM",
                "Qwen3OmniTalker",
            }
        ),
        quantized_stage_prefix="thinker.",
        full_precision_architectures=frozenset({"Qwen3OmniTalker"}),
        full_precision_stage_prefixes=("talker.", "code2wav."),
        mxfp4_expert_pattern=re.compile(
            r"^thinker\\?\.model\\?\.layers\\?\..+\\?\.mlp\\?\.experts\\?\."
            r".+\\?\.\(gate\|up\|down\)_proj$"
        ),
        model_label="Qwen3-Omni",
        mxfp8_cuda_moe_backend="flashinfer_trtllm",
        mixed_cuda_moe_backend="flashinfer_mxfp4",
    ),
)


def get_stage_local_mxfp_policy(
    architecture: str | None,
) -> StageLocalMxfpPolicy | None:
    """Return declarative stage-local MXFP rules for a model architecture."""
    return next(
        (
            policy
            for policy in _STAGE_LOCAL_MXFP_POLICIES
            if architecture in policy.architectures
        ),
        None,
    )


def _to_mutable_dict(quant_config: Any, metadata_key: str) -> dict[str, Any]:
    """Normalize a quantization metadata value to a mutable dict."""
    if isinstance(quant_config, dict):
        return quant_config
    if hasattr(quant_config, "to_dict"):
        quant_dict = quant_config.to_dict()
        if isinstance(quant_dict, dict):
            return quant_dict
    if hasattr(quant_config, "__dict__"):
        return vars(quant_config)
    raise TypeError(
        f"{metadata_key} has unsupported type {type(quant_config).__name__!r}. "
        f"Expected dict or object with to_dict()/__dict__."
    )


def _read_metadata(node: Any, key: str) -> Any:
    """Read `key` off an object- or dict-shaped config node, or `None`."""
    if isinstance(node, dict):
        return node.get(key)
    return getattr(node, key, None)


def resolve_quant_config(config: Any) -> dict[str, Any] | None:
    """Extract a `quantization_config` dict from a root or sub-model config."""
    visited: set[int] = set()

    def _search(node: Any) -> dict[str, Any] | None:
        if node is None or id(node) in visited:
            return None
        visited.add(id(node))

        for key in _QUANT_METADATA_KEYS:
            raw_config = _read_metadata(node, key)
            if raw_config is not None:
                return _to_mutable_dict(raw_config, key)

        for attr in _NESTED_QUANT_CONFIG_ATTRS:
            found = _search(_read_metadata(node, attr))
            if found is not None:
                return found
        return None

    return _search(config)


def quant_method_name(quant_dict: dict[str, Any] | None) -> str | None:
    """Return the checkpoint's normalized quantization method name, or `None`."""
    if not quant_dict:
        return None
    method = quant_dict.get("quant_method")
    if method is None:
        return None
    return str(method).lower().replace("_", "-")


def is_fp8_block_quant(quant_dict: dict[str, Any] | None) -> bool:
    """True when the checkpoint is native block-FP8."""
    if not quant_dict:
        return False
    if quant_method_name(quant_dict) != "fp8":
        return False
    return quant_dict.get("weight_block_size") is not None


def convert_fp8_weight_scale_inv(
    target_name: str,
    loaded_weight: "torch.Tensor",
) -> "torch.Tensor":
    """Reciprocate a `weight_scale_inv` tensor into the SGLang runtime scale."""
    if not target_name.endswith("weight_scale_inv"):
        return loaded_weight

    import torch

    if not torch.is_floating_point(loaded_weight):
        raise TypeError(f"FP8 scale tensor for {target_name} must be floating point")
    if loaded_weight.numel() == 0:
        raise ValueError(f"Invalid empty FP8 scale tensor for {target_name}")
    if not bool(torch.isfinite(loaded_weight).all().item()):
        raise ValueError(f"Invalid non-finite FP8 scale tensor for {target_name}")
    if bool(torch.any(loaded_weight == 0).item()):
        raise ValueError(f"Invalid zero FP8 scale tensor for {target_name}")

    return torch.reciprocal(loaded_weight)


def _identity_preprocessor(
    target_name: str, loaded_weight: "torch.Tensor"
) -> "torch.Tensor":
    return loaded_weight


def get_weight_preprocessor(
    config: Any = None,
    *,
    fp8_scale_inverted: bool = False,
) -> WeightPreprocessor:
    """Return the per-tensor weight transform for a checkpoint's quantization."""
    quant_dict = resolve_quant_config(config)

    if fp8_scale_inverted and is_fp8_block_quant(quant_dict):
        return convert_fp8_weight_scale_inv
    return _identity_preprocessor


def needs_quant_config_normalization(quant_dict: dict[str, Any] | None) -> bool:
    """True when the checkpoint's method uses stage-local per-block quant names."""
    method = quant_method_name(quant_dict)
    return method == "auto-round"


def _is_auto_round_mxfp(quant_config: dict[str, Any]) -> bool:
    data_type = str(quant_config.get("data_type", "")).lower().replace("-", "_")
    return quant_method_name(quant_config) == "auto-round" and data_type in _MXFP_NAMES


def _mxfp_bits(config: dict[str, Any], defaults: dict[str, Any]) -> int | None:
    try:
        return int(config.get("bits", defaults.get("bits", 0)))
    except (TypeError, ValueError):
        return None


def _has_mxfp4_overrides(quant_config: dict[str, Any]) -> bool:
    extra_config = quant_config.get("extra_config")
    if not isinstance(extra_config, dict):
        return False
    for entry in extra_config.values():
        if not isinstance(entry, dict):
            continue
        data_type = str(
            entry.get("data_type", quant_config.get("data_type", ""))
        ).lower().replace("-", "_")
        if _mxfp_bits(entry, quant_config) == 4 and data_type in _MXFP_NAMES:
            return True
    return False


def resolve_stage_local_mxfp_profile(
    config: Any,
    architecture: str | None = None,
) -> StageLocalMxfpProfile | None:
    """Resolve registered model-family policy and checkpoint MXFP properties.

    CUDA and other platform layers consume this result without embedding model
    names or checkpoint pattern details. Supporting another Omni family only
    requires registering its stage policy in this module.
    """
    hf_config = getattr(config, "hf_config", config)
    if hf_config is None:
        return None
    if architecture is None:
        architecture = (getattr(hf_config, "architectures", None) or [None])[0]
    policy = get_stage_local_mxfp_policy(architecture)
    quant_config = resolve_quant_config(hf_config)
    if policy is None or not quant_config or not _is_auto_round_mxfp(quant_config):
        return None
    if _mxfp_bits(quant_config, {}) != 8:
        return None
    return StageLocalMxfpProfile(
        policy=policy,
        has_mxfp4_experts=_has_mxfp4_overrides(quant_config),
    )


def requires_fp8_gemm_initialization(
    effective_quantization: str | None,
    config: Any,
) -> bool:
    """Whether model construction can instantiate an FP8-backed dense method."""
    method = (
        effective_quantization.lower().replace("_", "-")
        if effective_quantization
        else None
    )
    if method == "fp8":
        return True
    quant_config = resolve_quant_config(config)
    return bool(
        method == "auto-round"
        and quant_config
        and _is_auto_round_mxfp(quant_config)
        and _mxfp_bits(quant_config, {}) == 8
    )


def _compile_layer_pattern(pattern: Any) -> re.Pattern[str]:
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(
            "AutoRound quantization layer patterns must be non-empty strings"
        )
    try:
        return re.compile(pattern)
    except re.error as err:
        raise ValueError(
            f"Invalid quantization layer pattern {pattern!r}: {err}"
        ) from err


def _block_targets_stage(block: str, stage_prefix: str) -> bool:
    """Match AutoRound's literal ``layer_name.startswith(block)`` semantics."""
    return f"{stage_prefix}model.layers.0.self_attn.q_proj".startswith(block)


def _pattern_targets_routed_expert(
    pattern: str, policy: StageLocalMxfpPolicy
) -> bool:
    """Accept only the exported thinker routed-expert pattern shape.

    Arbitrary regular expressions cannot be proven to exclude dense or other
    stages. Constraining the shape keeps the MXFP4 exception fail-closed.
    """
    _compile_layer_pattern(pattern)
    return policy.mxfp4_expert_pattern.fullmatch(pattern) is not None


def _validate_mxfp_scheme(
    config: dict[str, Any], *, defaults: dict[str, Any], context: str
) -> tuple[int, str]:
    try:
        bits = int(config.get("bits", defaults.get("bits", 0)))
    except (TypeError, ValueError) as err:
        raise ValueError(f"{context} bits must be an integer") from err
    data_type = str(
        config.get("data_type", defaults.get("data_type", ""))
    ).lower().replace("-", "_")
    if data_type in _MXFP_NAMES:
        if bits not in {4, 8}:
            raise ValueError(f"{context} MXFP bits must be 4 or 8, got {bits}")
        try:
            group_size = int(
                config.get("group_size", defaults.get("group_size", 32))
            )
        except (TypeError, ValueError) as err:
            raise ValueError(f"{context} MXFP group_size must be 32") from err
        if group_size != 32:
            raise ValueError(f"{context} MXFP group_size must be 32, got {group_size}")
        if config.get("sym", defaults.get("sym", True)) is not True:
            raise ValueError(
                f"{context} MXFP must use symmetric quantization (sym=true)"
            )
    elif bits != 16 or data_type not in {"float", "fp16", "bf16", "bfloat16"}:
        raise ValueError(
            f"{context} must be MXFP4/MXFP8 or an explicit 16-bit "
            "floating-point exclusion"
        )
    return bits, data_type


def validate_stage_local_mxfp_config(model_config: Any) -> None:
    """Validate registered stage-local AutoRound MXFP checkpoint metadata."""
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return
    architecture = (getattr(hf_config, "architectures", None) or [None])[0]
    policy = get_stage_local_mxfp_policy(architecture)
    if policy is None:
        return
    quant_config = resolve_quant_config(hf_config)
    if not quant_config or not _is_auto_round_mxfp(quant_config):
        return

    default_bits, _default_data_type = _validate_mxfp_scheme(
        quant_config, defaults={}, context="AutoRound default"
    )
    if quant_config.get("packing_format", "auto_round:sglang") != "auto_round:sglang":
        raise ValueError("AutoRound MXFP requires packing_format='auto_round:sglang'")
    extra_config = quant_config.get("extra_config") or {}
    if not isinstance(extra_config, dict):
        raise ValueError("AutoRound MXFP extra_config must be a mapping")

    # MXFP4 has no supported dense LinearBase path in SGLang-Omni.
    # A global 4-bit default necessarily selects dense projections, so mixed
    # checkpoints must use MXFP8/BF16 globally and override routed experts only.
    if default_bits == 4:
        raise ValueError(
            f"{policy.model_label} AutoRound MXFP4 is supported only for routed MoE "
            "experts; a global MXFP4 default would quantize dense/attention "
            "layers. Export a mixed checkpoint with MXFP8 or BF16 as the default."
        )
    for pattern, entry in extra_config.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"AutoRound MXFP extra_config entry {pattern!r} must be a mapping"
            )
        _compile_layer_pattern(pattern)
        entry_bits, entry_dtype = _validate_mxfp_scheme(
            entry,
            defaults=quant_config,
            context=f"AutoRound layer pattern {pattern!r}",
        )
        if (
            entry_bits == 4
            and entry_dtype in _MXFP_NAMES
            and not _pattern_targets_routed_expert(str(pattern), policy)
        ):
            raise ValueError(
                f"{policy.model_label} AutoRound MXFP4 is supported only for registered "
                f"routed MoE expert patterns, but {pattern!r} can select another "
                "layer."
            )

    blocks = quant_config.get("block_name_to_quantize") or []
    if isinstance(blocks, str):
        blocks = [part.strip() for part in blocks.split(",") if part.strip()]

    if not isinstance(blocks, list) or not all(
        isinstance(block, str) and block for block in blocks
    ):
        raise ValueError(
            "AutoRound MXFP block_name_to_quantize must contain non-empty strings"
        )

    if not any(
        _block_targets_stage(block, policy.quantized_stage_prefix) for block in blocks
    ):
        raise ValueError(
            f"{policy.model_label} AutoRound MXFP must target the registered "
            "quantized stage via "
            "block_name_to_quantize"
        )

    violations = []
    for stage_prefix in policy.full_precision_stage_prefixes:
        stage_selected = any(
            _block_targets_stage(block, stage_prefix) for block in blocks
        )
        if not stage_selected:
            continue
        # A stage listed here inherits the quantized default. Exact BF16
        # exceptions cannot prove that every speech-stage linear is excluded;
        # a compliant SGLang export omits these stages from this field entirely.
        violations.append(stage_prefix[:-1])

    if violations:
        raise ValueError(
            f"{policy.model_label} AutoRound MXFP metadata selects "
            "full-precision stages: "
            f"{', '.join(violations)}. Re-export so only the intended stage is "
            "quantized."
        )


def disable_quantization_for_full_precision_stage(
    model_config: Any, architecture: str
) -> bool:
    """Prevent stage-local MXFP metadata from leaking into unquantized stages.

    Omni checkpoints can store several stages under one root quantization
    config. SGLang detects that config before Omni selects a sub-model, so a
    full-precision stage must clear both the detected method and root metadata.
    """
    policy = get_stage_local_mxfp_policy(architecture)
    if policy is None or architecture not in policy.full_precision_architectures:
        return False
    hf_config = getattr(model_config, "hf_config", None)
    quant_config = resolve_quant_config(hf_config)
    if not quant_config or not _is_auto_round_mxfp(quant_config):
        return False

    validate_stage_local_mxfp_config(model_config)
    for key in _QUANT_METADATA_KEYS:
        if isinstance(hf_config, dict):
            hf_config.pop(key, None)
        elif hasattr(hf_config, key):
            setattr(hf_config, key, None)
    model_config.quantization = None
    if hasattr(model_config, "is_fp4_experts"):
        model_config.is_fp4_experts = False
    return True


def _strip_stage_prefix(pattern: str, plain_prefix: str, escaped_prefix: str) -> str:
    """Strip the stage prefix from the start of a regex pattern."""
    if pattern.startswith(escaped_prefix):
        return pattern[len(escaped_prefix) :]
    if pattern.startswith(plain_prefix):
        return pattern[len(plain_prefix) :]
    leading_wildcard_escaped = r".*" + escaped_prefix
    if pattern.startswith(leading_wildcard_escaped):
        # Drop only the prefix part; keep the leading ".*" wildcard so the
        # normalized regex still matches stage-local module names.
        return r".*" + pattern[len(leading_wildcard_escaped) :]
    return pattern


def _normalize_extra_config_keys(
    quant_config: dict[str, Any], stage_prefix: str
) -> bool:
    """Strip `stage_prefix` from the leading edge of every regex key."""
    extra_config = quant_config.get("extra_config")
    if not (isinstance(extra_config, dict) and extra_config):
        return False

    escaped_prefix = stage_prefix.replace(".", r"\.")
    normalized_extra: dict[str, Any] = {}
    changed = False
    for key, value in extra_config.items():
        normalized_key = _strip_stage_prefix(key, stage_prefix, escaped_prefix)
        changed = changed or normalized_key != key
        if normalized_key in normalized_extra:
            raise ValueError(
                "AutoRound quantization patterns collide after stage-prefix "
                f"normalization: {key!r} and another entry both map to "
                f"{normalized_key!r}"
            )
        normalized_extra[normalized_key] = value

    if not changed:
        return False

    quant_config["extra_config"] = normalized_extra
    return True


def _normalize_block_name_to_quantize(
    quant_config: dict[str, Any], stage_prefix: str
) -> bool:
    """Strip `stage_prefix` from every entry of `block_name_to_quantize`."""
    blocks = quant_config.get("block_name_to_quantize")
    if isinstance(blocks, str):
        block_list = [b.strip() for b in blocks.split(",") if b.strip()]
        was_list = False
    elif isinstance(blocks, list):
        block_list = [str(b) for b in blocks]
        was_list = True
    else:
        return False
    if not block_list:
        return False

    normalized_blocks = [
        entry[len(stage_prefix) :] if entry.startswith(stage_prefix) else entry
        for entry in block_list
    ]
    if normalized_blocks == block_list:
        return False

    quant_config["block_name_to_quantize"] = (
        normalized_blocks if was_list else ",".join(normalized_blocks)
    )
    return True


def _load_writable_quant_config(
    hf_config: Any,
) -> tuple[Any, str, dict[str, Any], bool] | None:
    """Return `(owner, metadata_key, quant_config, needs_writeback)` for the
    quant metadata discovered on `hf_config` or a nested stage sub-config,
    or `None` if none is found."""
    visited: set[int] = set()

    def _search(node: Any) -> tuple[Any, str, dict[str, Any], bool] | None:
        if node is None or id(node) in visited:
            return None
        visited.add(id(node))

        for metadata_key in _QUANT_METADATA_KEYS:
            quant_config_raw = _read_metadata(node, metadata_key)
            if quant_config_raw is None:
                continue

            quant_config = _to_mutable_dict(quant_config_raw, metadata_key)
            # If we created a new dict from a non-dict object, we must write it
            # back after mutation so downstream consumers see the normalized names.
            needs_writeback = quant_config is not quant_config_raw
            return node, metadata_key, quant_config, needs_writeback

        for attr in _NESTED_QUANT_CONFIG_ATTRS:
            found = _search(_read_metadata(node, attr))
            if found is not None:
                return found
        return None

    return _search(hf_config)


def _resolve_stage_prefix(hf_config: Any) -> str | None:
    """Return the checkpoint prefix for the active stage architecture."""
    architectures = getattr(hf_config, "architectures", None) or []
    if not architectures:
        return None
    return _STAGE_PREFIX_BY_ARCH.get(architectures[0])


def normalize_quant_config(model_config: Any) -> None:
    """Strip the active stage's checkpoint prefix from the quant config"""
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return

    loaded = _load_writable_quant_config(hf_config)
    if loaded is None:
        return
    owner, metadata_key, quant_config, needs_writeback = loaded

    stage_prefix = _resolve_stage_prefix(hf_config)
    if not stage_prefix:
        return

    blocks_changed = _normalize_block_name_to_quantize(quant_config, stage_prefix)
    extra_changed = _normalize_extra_config_keys(quant_config, stage_prefix)
    if not (blocks_changed or extra_changed):
        return

    if needs_writeback:
        setattr(owner, metadata_key, quant_config)
