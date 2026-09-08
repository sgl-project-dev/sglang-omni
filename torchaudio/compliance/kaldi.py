# SPDX-License-Identifier: Apache-2.0

"""Small Kaldi fbank compatibility subset used by SGLang-Omni audio paths."""

from __future__ import annotations

import torch


_SGLANG_MUSA_SHIM = True


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    freqs = torch.linspace(
        0, sample_rate / 2, n_fft // 2 + 1, device=device, dtype=dtype
    )
    mel_min = _hz_to_mel(freqs.new_tensor(20.0))
    mel_max = _hz_to_mel(freqs.new_tensor(sample_rate / 2))
    mels = torch.linspace(
        mel_min, mel_max, n_mels + 2, device=device, dtype=dtype
    )
    hz = _mel_to_hz(mels)

    filters = []
    for index in range(n_mels):
        left, center, right = hz[index], hz[index + 1], hz[index + 2]
        up = (freqs - left) / (center - left).clamp_min(1e-6)
        down = (right - freqs) / (right - center).clamp_min(1e-6)
        filters.append(torch.maximum(torch.minimum(up, down), freqs.new_zeros(())))
    return torch.stack(filters, dim=0)


def get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
    vtln_low: float,
    vtln_high: float,
    vtln_warp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the subset of Kaldi mel banks used by ``cached_fbank``.

    VTLN arguments are accepted for API compatibility but are intentionally
    not implemented by this MUSA fallback.
    """
    del low_freq, high_freq, vtln_low, vtln_high, vtln_warp
    n_fft = int(window_length_padded)
    banks = _mel_filterbank(
        sample_rate=int(sample_freq),
        n_fft=n_fft,
        n_mels=int(num_bins),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return banks[:, :-1], torch.empty(0)


def _get_waveform_and_window_properties(
    waveform: torch.Tensor,
    channel: int,
    sample_frequency: float,
    frame_shift: float,
    frame_length: float,
    snip_edges: bool,
    preemphasis_coefficient: float,
) -> tuple[torch.Tensor, int, int, int]:
    del snip_edges, preemphasis_coefficient
    if waveform.ndim == 2:
        waveform = waveform[channel]
    window_shift = max(int(round(sample_frequency * frame_shift / 1000.0)), 1)
    window_size = max(int(round(sample_frequency * frame_length / 1000.0)), 1)
    padded_window_size = 1 << (window_size - 1).bit_length()
    return waveform, window_shift, window_size, padded_window_size


def _get_window(
    waveform: torch.Tensor,
    padded_window_size: int,
    window_size: int,
    window_shift: int,
    window_type: str,
    blackman_coeff: float,
    snip_edges: bool,
    raw_energy: bool,
    energy_floor: float,
    dither: float,
    remove_dc_offset: bool,
    preemphasis_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del blackman_coeff, snip_edges, raw_energy, energy_floor, dither
    if remove_dc_offset:
        waveform = waveform - waveform.mean()
    if preemphasis_coefficient:
        waveform = torch.cat(
            [waveform[:1], waveform[1:] - preemphasis_coefficient * waveform[:-1]]
        )
    if waveform.numel() < window_size:
        waveform = torch.nn.functional.pad(
            waveform, (0, window_size - waveform.numel())
        )
    frames = waveform.unfold(0, window_size, window_shift)
    if window_type == "hamming":
        window = torch.hamming_window(
            window_size, periodic=False, device=waveform.device
        )
    elif window_type == "povey":
        window = torch.hann_window(
            window_size, periodic=False, device=waveform.device
        ).pow(0.85)
    else:
        raise ValueError(f"unsupported window type: {window_type!r}")
    frames = frames * window
    if padded_window_size > window_size:
        frames = torch.nn.functional.pad(frames, (0, padded_window_size - window_size))
    return frames, torch.empty(0, device=waveform.device)


def _get_epsilon(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(torch.finfo(dtype).eps, device=device, dtype=dtype)


def fbank(
    waveform: torch.Tensor,
    *,
    num_mel_bins: int = 80,
    sample_frequency: int = 16000,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    dither: float = 0.0,
    window_type: str = "povey",
    **__,
) -> torch.Tensor:
    """Compute the supported Kaldi fbank subset.

    The fallback supports mono waveforms, snip-edges framing, Povey/Hamming
    windows, and log power mel features. Other keyword arguments are accepted
    for compatibility with the existing SGLang-Omni callers.
    """
    if waveform.ndim == 2:
        waveform = waveform[0]
    if waveform.ndim != 1:
        raise ValueError(f"fbank expects 1D or 2D waveform, got {tuple(waveform.shape)}")

    waveform = waveform.to(dtype=torch.float32)
    if dither:
        waveform = waveform + torch.randn_like(waveform) * float(dither)
    _, frame_shift_samples, frame_length_samples, padded_window_size = (
        _get_waveform_and_window_properties(
            waveform,
            0,
            sample_frequency,
            frame_shift,
            frame_length,
            True,
            0.97,
        )
    )
    strided, _ = _get_window(
        waveform[0] if waveform.ndim == 2 else waveform,
        padded_window_size,
        frame_length_samples,
        frame_shift_samples,
        window_type,
        0.42,
        True,
        True,
        0.0,
        dither,
        True,
        0.97,
    )
    spectrum = torch.fft.rfft(strided).abs().pow(2.0)
    banks, _ = get_mel_banks(
        num_mel_bins,
        padded_window_size,
        sample_frequency,
        20.0,
        0.0,
        100.0,
        -500.0,
        1.0,
    )
    banks = torch.nn.functional.pad(
        banks.to(device=spectrum.device, dtype=spectrum.dtype), (0, 1)
    )
    mel = spectrum @ banks.transpose(0, 1)
    return torch.maximum(
        mel, _get_epsilon(spectrum.device, spectrum.dtype)
    ).log()


__all__ = ["fbank", "get_mel_banks"]
