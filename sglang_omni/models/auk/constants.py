# SPDX-License-Identifier: Apache-2.0
"""AuK architecture constants (mirroring the released checkpoints)."""

from __future__ import annotations

# --- audio / VAE ---
SAMPLE_RATE: int = 24000
VAE_DOWNSAMPLE_RATE: int = 480
LATENT_FRAME_RATE: int = SAMPLE_RATE // VAE_DOWNSAMPLE_RATE  # 50 Hz
LATENT_DIM: int = 64

# --- conditioning ---
TEXT_HIDDEN_DIM: int = 2048
DEFAULT_TEXT_ENCODER: str = "Qwen/Qwen2.5-Omni-3B"
DEFAULT_VOICE_DESCRIPTION: str = "A clear, natural voice."
QWEN_AUDIO_SAMPLE_RATE: int = 16000
NO_PROMPT_AUDIO_MARKER: str = "|<no_prompt_audio>|"

# --- sampling defaults (AuK base) ---
DEFAULT_NFE: int = 32
DEFAULT_CFG_STRENGTH: float = 2.0
DEFAULT_SWAY_SAMPLING_COEF: float = -1.0
DEFAULT_ODE_METHOD: str = "euler"

# --- AuK-Flash (DMD distilled) recipe ---
FLASH_NFE: int = 4
FLASH_CFG_STRENGTH: float = 0.0
FLASH_T_GRID: tuple[float, ...] = (
    0.0,
    0.07612049579620361,
    0.2928932309150696,
    0.6173166036605835,
    1.0,
)

# --- request limits ---
MAX_SECONDS: float = 30.0
DEFAULT_SECONDS: float = 5.0
