# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from transformers import PretrainedConfig, PreTrainedModel

from sglang_omni.models.auk.reference_encode import AuKConditionEncoder


def test_loading_ignores_only_unused_omni_branches(tmp_path, monkeypatch):
    import transformers

    loading_reports = []

    class TinyThinker(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.proj = torch.nn.Linear(2, 2)
            self.post_init()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            model, report = super().from_pretrained(
                *args, output_loading_info=True, **kwargs
            )
            loading_reports.append(report)
            return model

    original = TinyThinker(PretrainedConfig())
    original.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    weights = load_file(path)
    del weights["proj.bias"]
    for key in (
        "talker.model.weight",
        "token2wav.decoder.weight",
        "unknown.weight",
        "proj.talker.weight",
    ):
        weights[key] = torch.ones(1)
    save_file(weights, path)
    monkeypatch.setattr(
        transformers, "Qwen2_5OmniThinkerForConditionalGeneration", TinyThinker
    )
    monkeypatch.setattr(
        transformers,
        "Qwen2_5OmniProcessor",
        SimpleNamespace(from_pretrained=lambda _: None),
    )

    encoder = AuKConditionEncoder(str(tmp_path), dtype=torch.float32)

    report = loading_reports[0]
    assert set(report["unexpected_keys"]) == {"unknown.weight", "proj.talker.weight"}
    assert set(report["missing_keys"]) == {"proj.bias"}
    torch.testing.assert_close(encoder.model.proj.weight, original.proj.weight)
    assert not TinyThinker._keys_to_ignore_on_load_unexpected
