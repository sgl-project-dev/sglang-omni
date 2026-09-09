# SPDX-License-Identifier: Apache-2.0
"""Import dots.tts on the pinned torch stack."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import threading
from types import ModuleType

_IMPORT_LOCK = threading.Lock()


def _install_tn_shim() -> None:
    """Back tn.chinese / tn.english with wetext when Pynini is absent.

    dots_tts.utils.text imports WeTextProcessing's Normalizers at module
    scope, and WeTextProcessing depends on Pynini, which ships no macOS
    wheels — without a substitute the whole dots_tts package fails to import
    on Apple Silicon. wetext is the pure-Python runtime for the same FSTs,
    so when the real tn is missing and wetext is installed, register
    drop-in normalizer modules under the names dots.tts imports.
    """
    try:
        importlib.import_module("tn.chinese.normalizer")
        return
    except ImportError:
        pass
    try:
        from wetext import Normalizer as _WetextNormalizer
    except ImportError:
        return

    def _make_normalizer_module(package_lang: str, wetext_lang: str) -> ModuleType:
        module = ModuleType(f"tn.{package_lang}.normalizer")

        class Normalizer:
            def __init__(self) -> None:
                self._impl = _WetextNormalizer(lang=wetext_lang, operator="tn")

            def normalize(self, text: str) -> str:
                return self._impl.normalize(text)

        module.Normalizer = Normalizer
        return module

    tn = ModuleType("tn")
    for package_lang, wetext_lang in (("chinese", "zh"), ("english", "en")):
        package = ModuleType(f"tn.{package_lang}")
        normalizer = _make_normalizer_module(package_lang, wetext_lang)
        package.normalizer = normalizer
        tn.__dict__[package_lang] = package
        sys.modules[f"tn.{package_lang}"] = package
        sys.modules[f"tn.{package_lang}.normalizer"] = normalizer
    sys.modules["tn"] = tn


# note (guozhihao-224): installed at module scope because dots_tts submodules
# can be imported without going through import_dots_tts; a no-op where the
# real tn exists.
_install_tn_shim()


def import_dots_tts() -> ModuleType:
    """Import the dots_tts package past its torch/torchaudio version check.

    dots.tts refuses to import unless the torch and torchaudio distributions
    share a minor version (dots_tts/__init__.py, _check_torch_install, which
    reads both through importlib.metadata.version). sglang 0.5.18 pins torch
    2.13.0 with torchaudio 2.11.0, and omni pins the same pair: torchaudio
    2.11.0 is its last release, and its release note states it is compatible
    with torch 2.11 and with future torch versions. The check is stricter than
    that supported pair, and the torchaudio surface dots.tts uses
    (functional.resample, compliance.kaldi fbank, transforms.Resample) is
    torch ops. For the single package import, the torchaudio distribution is
    reported at torch's version and the reader is restored before returning.
    The replacement is process-global for that moment, so every omni import
    of dots_tts goes through this function and nothing else may read
    distribution versions concurrently with the first one. Remove this once
    torchaudio ships a release matching torch's minor or dots.tts drops the
    equality check.
    """
    with _IMPORT_LOCK:
        if "dots_tts" in sys.modules:
            return importlib.import_module("dots_tts")
        reader = importlib.metadata.version

        def bridged(distribution_name: str) -> str:
            if distribution_name == "torchaudio":
                return reader("torch")
            return reader(distribution_name)

        importlib.metadata.version = bridged
        try:
            return importlib.import_module("dots_tts")
        finally:
            importlib.metadata.version = reader


def import_dots_solver_deps() -> ModuleType:
    """Import the DiT solver chain on the host default device.

    torchdiffeq materializes its float64 tableau tensors at import time
    (torchdiffeq/_impl/dopri5.py). SGLang constructs models inside
    ``torch.device("mps")``, where float64 does not exist, so the chain must
    be imported before that context opens. Subsequent imports are no-ops.
    """
    import_dots_tts()
    return importlib.import_module("dots_tts.models.dots_tts.core")
