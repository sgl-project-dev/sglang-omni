# SPDX-License-Identifier: Apache-2.0
"""Keep profiler imports safe after SGLang installs its Metal wrapper."""

import subprocess
import sys


def test_torch_profiler_import_after_function_wrapper():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch.profiler; "
            "torch.profiler.profile = lambda *args, **kwargs: None; "
            "from sglang_omni.profiler.torch_profiler import TorchProfiler; "
            "assert TorchProfiler._profiler is None",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
