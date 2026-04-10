"""Spectrogram feature extractor for RF IQ data."""

import logging
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("rfml.features")


class SpectrogramExtractor:
    """Compute 3-channel spectrograms from raw IQ data.

    Channels:
        0 - magnitude (log-scaled)
        1 - phase (unwrapped)
        2 - instantaneous frequency (phase derivative)

    The STFT is computed separately on the real (I) and imaginary (Q)
    channels so that complex-valued structure is preserved before the
    three derived feature channels are assembled.

    Args:
        config: dict with keys:
            fft_size      (int)   - FFT size, default 512
            hop_length    (int)   - hop between frames, default 256
            window        (str)   - window type, default "hann"
            output_size   (list)  - [H, W] to resize output, default [512, 512]
            channels      (int)   - number of output channels (must be 3)
            use_complex   (bool)  - separate FFT of I and Q, default True
            device        (str)   - "cuda" or "cpu", default "cpu"
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.fft_size = int(cfg.get("fft_size", 512))
        self.hop_length = int(cfg.get("hop_length", 256))
        self.window_type = str(cfg.get("window", "hann"))
        out = cfg.get("output_size", [512, 512])
        self.output_h = int(out[0])
        self.output_w = int(out[1])
        self.use_complex = bool(cfg.get("use_complex", True))
        device_str = str(cfg.get("device", "cpu"))
        self.device = torch.device(
            device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu"
        )

        self._window: Optional[torch.Tensor] = None
        logger.debug(
            "SpectrogramExtractor: fft_size=%d hop=%d output=%dx%d device=%s",
            self.fft_size,
            self.hop_length,
            self.output_h,
            self.output_w,
            self.device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_window(self) -> torch.Tensor:
        if self._window is None or self._window.device != self.device:
            if self.window_type == "hann":
                self._window = torch.hann_window(self.fft_size, device=self.device)
            elif self.window_type == "hamming":
                self._window = torch.hamming_window(self.fft_size, device=self.device)
            else:
                self._window = torch.ones(self.fft_size, device=self.device)
        return self._window

    def _stft_channel(self, x: torch.Tensor) -> torch.Tensor:
        """Run torch.stft on a real 1-D signal, return complex tensor [F, T]."""
        window = self._get_window()
        # torch.stft expects [N] or [B, N]
        spec = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.fft_size,
            window=window,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )
        return spec  # [F, T]

    def _complex_stft(self, iq: torch.Tensor) -> torch.Tensor:
        """Compute combined complex STFT from I+jQ signal.

        Returns complex tensor [F, T] where F = fft_size // 2 + 1.
        """
        i_chan = iq[0]  # [N]
        q_chan = iq[1]  # [N]
        stft_i = self._stft_channel(i_chan)  # [F, T] complex
        stft_q = self._stft_channel(q_chan)  # [F, T] complex
        # Reconstruct as complex IQ STFT: S_I + j*S_Q
        return stft_i + 1j * stft_q

    def _build_channels(self, spec_complex: torch.Tensor) -> torch.Tensor:
        """Build 3-channel feature image from complex spectrogram.

        Returns float tensor [3, F, T].
        """
        magnitude = torch.abs(spec_complex)
        # Log-compressed magnitude (add small epsilon for numerical stability)
        log_mag = torch.log1p(magnitude)

        phase = torch.angle(spec_complex)  # [-pi, pi]

        # Instantaneous frequency: phase derivative along time axis
        inst_freq = torch.diff(phase, dim=-1, prepend=phase[..., :1])
        # Wrap to [-pi, pi]
        inst_freq = (inst_freq + math.pi) % (2 * math.pi) - math.pi

        return torch.stack([log_mag, phase, inst_freq], dim=0)  # [3, F, T]

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize [3, F, T] -> [3, H, W] using bilinear interpolation."""
        # F.interpolate expects [B, C, H, W]
        x = x.unsqueeze(0)  # [1, 3, F, T]
        x = F.interpolate(
            x,
            size=(self.output_h, self.output_w),
            mode="bilinear",
            align_corners=False,
        )
        return x.squeeze(0)  # [3, H, W]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel mean/std normalization."""
        # x: [3, H, W]
        mean = x.view(3, -1).mean(dim=1, keepdim=True).unsqueeze(-1)  # [3,1,1]
        std = x.view(3, -1).std(dim=1, keepdim=True).unsqueeze(-1) + 1e-8
        return (x - mean) / std

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, iq: torch.Tensor) -> torch.Tensor:
        """Extract spectrogram from a single IQ segment.

        Args:
            iq: float tensor of shape [2, N]  (I and Q channels)

        Returns:
            float tensor of shape [3, output_h, output_w]
        """
        iq = iq.to(self.device)
        spec = self._complex_stft(iq)          # [F, T] complex
        channels = self._build_channels(spec)  # [3, F, T]
        channels = self._resize(channels)      # [3, H, W]
        channels = self._normalize(channels)
        return channels

    def extract_batch(self, iq_batch: torch.Tensor) -> torch.Tensor:
        """Batch spectrogram extraction.

        Args:
            iq_batch: float tensor of shape [B, 2, N]

        Returns:
            float tensor of shape [B, 3, output_h, output_w]
        """
        iq_batch = iq_batch.to(self.device)
        results = []
        for b in range(iq_batch.shape[0]):
            results.append(self.extract(iq_batch[b]))
        return torch.stack(results, dim=0)

    def __call__(
        self, iq: torch.Tensor, batch: bool = False
    ) -> torch.Tensor:
        if batch:
            return self.extract_batch(iq)
        return self.extract(iq)
