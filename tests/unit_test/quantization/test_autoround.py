# SPDX-License-Identifier: Apache-2.0
"""Tests for stage-local checkpoint name normalization (e.g. AutoRound)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.quantization import (
    disable_quantization_for_full_precision_stage,
    get_stage_local_mxfp_policy,
    needs_quant_config_normalization,
    normalize_quant_config,
    resolve_stage_local_mxfp_profile,
    validate_stage_local_mxfp_config,
)


def _make_model_config(
    architecture: str | None,
    quantization_config: object,
) -> SimpleNamespace:
    """Build a minimal ``model_config`` stub for normalization tests."""
    hf_config = SimpleNamespace(
        architectures=[architecture] if architecture is not None else [],
        quantization_config=quantization_config,
    )
    return SimpleNamespace(hf_config=hf_config)


class TestNeedsStageLocalNormalization:
    """Tests for the ``needs_quant_config_normalization`` dispatch."""

    def test_true_for_auto_round(self) -> None:
        assert needs_quant_config_normalization({"quant_method": "auto-round"})

    def test_true_for_underscore_variant(self) -> None:
        assert needs_quant_config_normalization({"quant_method": "auto_round"})

    def test_false_for_fp8(self) -> None:
        assert not needs_quant_config_normalization({"quant_method": "fp8"})

    def test_false_for_none(self) -> None:
        assert not needs_quant_config_normalization(None)


class TestQwen3OmniMxfpConfig:
    def test_policy_resolves_for_each_model_stage(self) -> None:
        assert get_stage_local_mxfp_policy(
            "Qwen3OmniMoeForConditionalGeneration"
        ) is get_stage_local_mxfp_policy("Qwen3OmniThinkerForCausalLM")
        assert get_stage_local_mxfp_policy("OtherOmniModel") is None

    def test_runtime_profile_derives_backend_from_registered_policy(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "bits": 8,
            "data_type": "mx_fp",
            "extra_config": {
                r"thinker\.model\.layers\..*\.mlp\.experts\..*\.(gate|up|down)_proj": {
                    "bits": 4,
                    "data_type": "mx_fp",
                }
            },
        }
        model_config = _make_model_config(
            "Qwen3OmniThinkerForCausalLM", quant_config
        )

        profile = resolve_stage_local_mxfp_profile(model_config)

        assert profile is not None
        assert profile.has_mxfp4_experts
        assert profile.preferred_cuda_moe_backend == "flashinfer_mxfp4"

    def test_accepts_mxfp4_overrides_for_thinker_experts(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "packing_format": "auto_round:sglang",
            "bits": 8,
            "data_type": "mx_fp",
            "group_size": 32,
            "block_name_to_quantize": "thinker.model.layers",
            "extra_config": {
                r"thinker\.model\.layers\..*\.mlp\.experts\..*\.(gate|up|down)_proj": {
                    "bits": 4,
                    "data_type": "mx_fp",
                }
            },
        }

        validate_stage_local_mxfp_config(
            _make_model_config("Qwen3OmniMoeForConditionalGeneration", quant_config)
        )

    @pytest.mark.parametrize(
        "quant_config,error",
        [
            (
                {
                    "quant_method": "auto-round",
                    "packing_format": "auto_round:sglang",
                    "bits": 4,
                    "data_type": "mx_fp",
                    "group_size": 32,
                    "block_name_to_quantize": "thinker.model.layers",
                },
                "only for routed MoE experts",
            ),
            (
                {
                    "quant_method": "auto-round",
                    "packing_format": "auto_round:sglang",
                    "bits": 8,
                    "data_type": "mx_fp",
                    "group_size": 32,
                },
                "must target the registered quantized stage",
            ),
            (
                {
                    "quant_method": "auto-round",
                    "packing_format": "auto_round:sglang",
                    "bits": 8,
                    "data_type": "mx_fp",
                    "group_size": 64,
                    "block_name_to_quantize": "thinker.model.layers",
                },
                "group_size must be 32",
            ),
            (
                {
                    "quant_method": "auto-round",
                    "packing_format": "auto_round:sglang",
                    "bits": 8,
                    "data_type": "mx_fp",
                    "group_size": 32,
                    "block_name_to_quantize": "thinker.model.layers,talker.model.layers",
                },
                "selects full-precision stages",
            ),
            (
                {
                    "quant_method": "auto-round",
                    "packing_format": "auto_round:sglang",
                    "bits": 8,
                    "data_type": "mx_fp",
                    "group_size": 32,
                    "block_name_to_quantize": "thinker.model.layers",
                    "extra_config": {
                        r"thinker\.model\.layers\..*\.self_attn\.q_proj": {
                            "bits": 4,
                            "data_type": "mx_fp",
                        }
                    },
                },
                "only for registered routed MoE expert patterns",
            ),
        ],
    )
    def test_rejects_unsupported_mxfp_configs(
        self, quant_config: dict[str, object], error: str
    ) -> None:
        with pytest.raises(ValueError, match=error):
            validate_stage_local_mxfp_config(
                _make_model_config(
                    "Qwen3OmniMoeForConditionalGeneration", quant_config
                )
            )

    def test_talker_drops_thinker_mxfp_metadata(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "packing_format": "auto_round:sglang",
            "bits": 8,
            "data_type": "mx_fp",
            "group_size": 32,
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config(
            "Qwen3OmniMoeForConditionalGeneration", quant_config
        )
        model_config.quantization = "auto-round"
        model_config.is_fp4_experts = True

        assert disable_quantization_for_full_precision_stage(
            model_config, "Qwen3OmniTalker"
        )
        assert model_config.quantization is None
        assert model_config.hf_config.quantization_config is None
        assert model_config.is_fp4_experts is False


class TestNormalizeStageLocalCheckpointConfig:
    """Tests for stripping the stage prefix from block names / extra_config."""

    def test_strips_thinker_prefix_from_block_names(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_strips_talker_prefix_from_block_names(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "talker.model.layers",
        }
        model_config = _make_model_config("Qwen3OmniTalker", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_asr_architecture_strips_thinker_prefix(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config(
            "Qwen3ASRForConditionalGeneration", quant_config
        )

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_strips_prefix_from_comma_separated_list(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers,thinker.model.experts",
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers,model.experts"

    def test_normalizes_block_name_list_input(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": ["thinker.model.layers", "model.shared"],
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == [
            "model.layers",
            "model.shared",
        ]

    def test_strips_prefix_from_extra_config_keys(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
            "extra_config": {
                r"thinker\.model\.layers\.0": {"bits": 8},
                "thinker.model.layers.1": {"bits": 4},
            },
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8},
            "model.layers.1": {"bits": 4},
        }

    def test_extra_config_normalized_when_block_names_already_stripped(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "model.layers",
            "extra_config": {
                r"thinker\.model\.layers\.0": {"bits": 8},
                "thinker.model.layers.1": {"bits": 4},
            },
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"
        assert quant_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8},
            "model.layers.1": {"bits": 4},
        }

    def test_extra_config_prefix_anchored_at_pattern_start(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
            "extra_config": {
                # leading prefix -> should be stripped
                r"thinker\.model\.layers\.0": {"bits": 8},
                # prefix inside an alternation -> must be preserved
                r"(?:thinker|decoder)\.model\.layers\.1": {"bits": 4},
                # prefix as a substring of another name -> must be preserved
                r"thinker_audio\.model\.layers\.2": {"bits": 4},
            },
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8},
            r"(?:thinker|decoder)\.model\.layers\.1": {"bits": 4},
            r"thinker_audio\.model\.layers\.2": {"bits": 4},
        }

    def test_strips_prefix_from_leading_wildcard_pattern(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
            "extra_config": {
                r".*thinker\.model\.layers\.\d+\.mlp\.gate.*": {"bits": 8},
            },
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["extra_config"] == {
            r".*model\.layers\.\d+\.mlp\.gate.*": {"bits": 8},
        }

    def test_rejects_pattern_collision_after_normalization(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
            "extra_config": {
                r"thinker\.model\.layers\.0": {"bits": 4},
                r"model\.layers\.0": {"bits": 8},
            },
        }
        model_config = _make_model_config(
            "Qwen3OmniThinkerForCausalLM", quant_config
        )

        with pytest.raises(ValueError, match="patterns collide"):
            normalize_quant_config(model_config)

    def test_no_change_when_block_names_lack_prefix(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "model.layers",
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_unknown_architecture_leaves_block_names_unchanged(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config("SomeOtherForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "thinker.model.layers"

    def test_normalizes_config_that_lives_only_on_nested_stage_attr(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        thinker_config = SimpleNamespace(quantization_config=quant_config)
        hf_config = SimpleNamespace(
            architectures=["Qwen3OmniThinkerForCausalLM"],
            thinker_config=thinker_config,
        )
        model_config = SimpleNamespace(hf_config=hf_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"
        assert thinker_config.quantization_config["block_name_to_quantize"] == (
            "model.layers"
        )

    def test_missing_hf_config_is_noop(self) -> None:
        model_config = SimpleNamespace(hf_config=None)
        normalize_quant_config(model_config)

    def test_missing_block_name_to_quantize_is_noop(self) -> None:
        quant_config = {"quant_method": "auto-round"}
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config == {"quant_method": "auto-round"}

    def test_non_dict_quantization_config_raises(self) -> None:
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", "not-a-dict")
        with pytest.raises(TypeError, match="unsupported type"):
            normalize_quant_config(model_config)


class TestObjectShapedConfig:
    """Object-shaped quant configs are converted and written back."""

    def test_object_quant_config_converted_and_written_back(self) -> None:
        quant_config = SimpleNamespace(
            quant_method="auto-round",
            block_name_to_quantize="thinker.model.layers",
            bits=4,
        )
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert isinstance(model_config.hf_config.quantization_config, dict)
        assert model_config.hf_config.quantization_config["block_name_to_quantize"] == (
            "model.layers"
        )

    def test_object_with_to_dict_converted_and_written_back(self) -> None:
        class HasToDict:
            def __init__(self):
                self.quant_method = "auto-round"
                self.block_name_to_quantize = "thinker.model.layers"
                self.bits = 4

            def to_dict(self):
                return {
                    "quant_method": self.quant_method,
                    "block_name_to_quantize": self.block_name_to_quantize,
                    "bits": self.bits,
                }

        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", HasToDict())

        normalize_quant_config(model_config)

        assert isinstance(model_config.hf_config.quantization_config, dict)
        assert model_config.hf_config.quantization_config["block_name_to_quantize"] == (
            "model.layers"
        )

    def test_object_extra_config_normalized_when_blocks_already_stripped(self) -> None:
        quant_config = SimpleNamespace(
            quant_method="auto-round",
            block_name_to_quantize="model.layers",
            extra_config={r"thinker\.model\.layers\.0": {"bits": 8}},
        )
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert isinstance(model_config.hf_config.quantization_config, dict)
        assert model_config.hf_config.quantization_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8}
        }

    def test_unsupported_quant_config_type_raises(self) -> None:
        model_config = _make_model_config(
            "Qwen3OmniThinkerForCausalLM", ["not", "a", "dict"]
        )
        with pytest.raises(TypeError, match="unsupported type"):
            normalize_quant_config(model_config)
