# SPDX-License-Identifier: Apache-2.0
"""Runtime accelerator probes for ``accelerator``-marked tests.

Probed in the test body, not at collection time, so the marker still assigns
the test to the accelerator CI job (see tests/README.md).
"""

from __future__ import annotations

import pytest
import torch


def require_cuda(min_devices: int = 1) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.device_count() < min_devices:
        pytest.skip(f"requires {min_devices} visible CUDA devices")


def require_accelerator(min_devices: int = 1) -> str:
    """Skip unless the host's own accelerator is usable; return its device type.

    For behavior that must hold on every backend rather than on CUDA
    specifically, so the one test covers the CUDA and XPU CI hosts both.
    """
    from sglang_omni.platforms import current_platform

    device_type = current_platform.device_type
    if device_type == "cpu":
        pytest.skip("no accelerator platform resolved")
    module = torch.get_device_module(device_type)
    if not module.is_available():
        pytest.skip(f"{device_type} is unavailable")
    if module.device_count() < min_devices:
        pytest.skip(f"requires {min_devices} visible {device_type} devices")
    return device_type
