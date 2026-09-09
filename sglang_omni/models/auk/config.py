# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for AuK (skeleton).

The model registry walks every subpackage of ``sglang_omni.models`` and requires
each ``config`` module to export ``EntryClass``, so this file cannot stay empty
without breaking registry import. Only the architecture hook is declared here;
the stage list, per-stage factory args and validation land with the real
implementation.

TODO(adapter): confirm the released ``config.json::architectures[0]`` value and
replace the placeholder below.
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig

_PKG = "sglang_omni.models.auk"


class AuKPipelineConfig(PipelineConfig):
    """Placeholder pipeline config for AuK."""

    architecture: ClassVar[str | None] = "AuKForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = ()


EntryClass = AuKPipelineConfig
