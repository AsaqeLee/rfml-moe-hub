"""Wavelet scattering transform feature extractor using Kymatio."""

import logging
from typing import Optional

import torch

logger = logging.getLogger("rfml.features")


class WaveletExtractor:
    """Second-order scattering transform for RF IQ signals.

    Uses the Kymatio library (https://www.kymat.io/) with PyTorch backend.
    Falls back to a simple DWT-based approximation if Kymatio is unavailable.

    The scattering transform is translation-invariant up to the scale 2^J
    and provides a stable, information-preserving feature representation
    suitable as an additional input channel to the spectrogram expert.

    Args:
        config: dict with keys:
            J      (int)  - number of octaves (log2 of max scale), default 8
            Q      (int)  - number of wavelets per octave, default 8
            order  (int)  - scattering order (1 or 2), default 2
            device (str)  - computation device
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.J = int(cfg.get("J", 8))
        self.Q = int(cfg.get("Q", 8))
        self.order = int(cfg.get("order", 2))
        self.library = str(cfg.get("library", "kymatio"))

        device_str = str(cfg.get("device", "cpu"))
        self.device = torch.device(
            device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu"
        )

        self._scattering = None
        self._kymatio_available = False
        self._output_length: Optional[int] = None

        self._init_scattering()

        logger.debug(
            "WaveletExtractor: J=%d Q=%d order=%d kymatio=%s device=%s",
            self.J,
            self.Q,
            self.order,
            self._kymatio_available,
            self.device,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_scattering(self):
        """Try to initialise Kymatio scattering; fall back gracefully."""
        try:
            from kymatio.torch import Scattering1D  # type: ignore

            # Use a placeholder length; will be re-created on first call if
            # the actual signal length differs.
            self._ScatteringClass = Scattering1D
            self._kymatio_available = True
            logger.debug("Kymatio Scattering1D loaded successfully.")
        except ImportError:
            logger.warning(
                "Kymatio not installed. WaveletExtractor will use a "
                "simple multi-scale energy fallback. "
                "Install with: pip install kymatio"
            )
            self._kymatio_available = False

    def _get_scattering(self, N: int):
        """Return (or rebuild) the scattering object for signal length N."""
        if self._scattering is not None and self._last_N == N:
            return self._scattering

        if self._kymatio_available:
            # Clamp J so 2^J <= N
            import math

            J_eff = min(self.J, int(math.log2(N)))
            self._scattering = self._ScatteringClass(
                J=J_eff, Q=self.Q, shape=N
            ).to(self.device)
            self._last_N = N
        return self._scattering

    # ------------------------------------------------------------------
    # Fallback: multi-scale energy via haar DWT approximation
    # ------------------------------------------------------------------

    def _fallback_extract(self, z: torch.Tensor) -> torch.Tensor:
        """Simple multi-scale energy extractor (no Kymatio dependency).

        Iteratively downsamples (Haar averaging) to build J scale levels.
        Returns a 1-D feature vector.

        Args:
            z: complex or real 1-D tensor [N]

        Returns:
            float tensor [J * 2]  (energy + variance per scale)
        """
        if z.is_complex():
            signal = torch.abs(z)
        else:
            signal = z.float()

        features = []
        current = signal.float()
        for _ in range(self.J):
            if current.shape[0] < 2:
                break
            energy = (current ** 2).mean()
            variance = current.var()
            features.extend([energy, variance])
            # Haar low-pass: average adjacent samples
            n = (current.shape[0] // 2) * 2
            current = (current[:n:2] + current[1:n:2]) / 2.0

        feat = torch.stack(features)
        return feat.float()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, iq: torch.Tensor) -> torch.Tensor:
        """Extract scattering coefficients from a single IQ segment.

        Args:
            iq: [2, N] float or [N] complex tensor

        Returns:
            float tensor [D] where D depends on J, Q, order, and N
        """
        iq = iq.to(self.device)

        if iq.is_complex():
            z = iq
            x_real = z.real.float()
        else:
            x_real = iq[0].float()  # use I channel for 1-D scattering

        N = x_real.shape[0]

        if not self._kymatio_available:
            z_for_fallback = (
                iq if iq.is_complex()
                else torch.complex(iq[0].float(), iq[1].float())
            )
            return self._fallback_extract(z_for_fallback)

        scat = self._get_scattering(N)
        # Kymatio expects [B, 1, N] for 1-D scattering
        x_in = x_real.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        out = scat(x_in)                          # [1, C, T] or [1, order+1, C, T]

        # Flatten to 1-D feature vector
        return out.squeeze(0).reshape(-1).float()

    def extract_batch(self, iq_batch: torch.Tensor) -> torch.Tensor:
        """Batch scattering extraction.

        Args:
            iq_batch: [B, 2, N] float or [B, N] complex

        Returns:
            [B, D] float tensor
        """
        iq_batch = iq_batch.to(self.device)
        results = [self.extract(iq_batch[b]) for b in range(iq_batch.shape[0])]
        return torch.stack(results, dim=0)

    def __call__(
        self, iq: torch.Tensor, batch: bool = False
    ) -> torch.Tensor:
        if batch:
            return self.extract_batch(iq)
        return self.extract(iq)
