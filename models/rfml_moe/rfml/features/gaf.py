"""Gramian Angular Field (GAF) image extractor for RF signal representation.

Converts a 1-D RF signal into a 2-D image via Piecewise Aggregation
Approximation (PAA) followed by polar encoding and outer-product construction.
Produces GASF (summation) and GADF (difference) images as described in:

    Wang & Oates, "Encoding Time Series as Images for Visual Inspection and
    Classification Using Tiled Convolutional Neural Networks", AAAI 2015.

    Fu et al., "RFML: Radio Frequency Machine Learning", 2022.
"""

import logging
from typing import Literal

import torch

logger = logging.getLogger("rfml.features")


class GAFExtractor:
    """Gramian Angular Field feature extractor for RF signals.

    Generates a 3-channel image [3, n, n] from a pair of (denoised, raw)
    1-D signals:
        - Channel 0: GASF of denoised signal
        - Channel 1: GADF of denoised signal
        - Channel 2: GASF of raw signal (SNR proxy for router)

    Args:
        n:      PAA target length; output images are n×n. Default 256.
        method: Which fields to compute for extract_single().
                "summation" → GASF only, "difference" → GADF only,
                "both" → [GASF, GADF]. Default "both".
    """

    def __init__(
        self,
        n: int = 256,
        method: Literal["summation", "difference", "both"] = "both",
    ) -> None:
        if method not in ("summation", "difference", "both"):
            raise ValueError(
                f"method must be 'summation', 'difference', or 'both'; got {method!r}"
            )
        self.n = n
        self.method = method
        logger.debug("GAFExtractor: n=%d method=%s", n, method)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, x_denoised: torch.Tensor, x_raw: torch.Tensor
    ) -> torch.Tensor:
        """Generate a 3-channel GAF image from denoised and raw signals.

        Args:
            x_denoised: 1-D real float tensor [L]
            x_raw:      1-D real float tensor [L]

        Returns:
            Float tensor [3, n, n]:
                ch0 = GASF(denoised), ch1 = GADF(denoised), ch2 = GASF(raw)
        """
        phi_d = self._encode(x_denoised)
        phi_r = self._encode(x_raw)

        gasf_d = self._gasf(phi_d)  # [n, n]
        gadf_d = self._gadf(phi_d)  # [n, n]
        gasf_r = self._gasf(phi_r)  # [n, n]

        out = torch.stack([gasf_d, gadf_d, gasf_r], dim=0)  # [3, n, n]
        logger.debug("GAFExtractor.extract: output shape %s", list(out.shape))
        return out

    def extract_single(self, x: torch.Tensor) -> torch.Tensor:
        """Extract GAF image(s) from a single 1-D signal.

        Args:
            x: 1-D real float tensor [L]

        Returns:
            Float tensor [C, n, n] where C depends on self.method:
                "summation"  → [1, n, n]
                "difference" → [1, n, n]
                "both"       → [2, n, n]  (GASF first, GADF second)
        """
        phi = self._encode(x)
        channels = []
        if self.method in ("summation", "both"):
            channels.append(self._gasf(phi))
        if self.method in ("difference", "both"):
            channels.append(self._gadf(phi))
        return torch.stack(channels, dim=0)

    # ------------------------------------------------------------------
    # Core transforms
    # ------------------------------------------------------------------

    def _paa(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Piecewise Aggregation Approximation: reduce [L] → [n].

        Truncates x to the largest multiple of n, then averages each
        segment of length L//n.

        Args:
            x: 1-D tensor [L]
            n: target length

        Returns:
            1-D tensor [n]
        """
        L = x.shape[0]
        if L < n:
            # Pad with last value to reach a multiple of n that >= n
            pad = n - L
            x = torch.cat([x, x[-1:].expand(pad)], dim=0)
            L = x.shape[0]

        seg = L // n
        # Truncate to n * seg then reshape and average
        x_trunc = x[: n * seg]           # [n * seg]
        x_2d = x_trunc.reshape(n, seg)   # [n, seg]
        return x_2d.mean(dim=1)           # [n]

    def _rescale(self, x: torch.Tensor) -> torch.Tensor:
        """Min-max rescale x to [-1, 1].

        Args:
            x: 1-D tensor [L]

        Returns:
            1-D tensor [L] with values in [-1, 1]
        """
        x_min = x.min()
        x_max = x.max()
        denom = x_max - x_min
        if denom == 0:
            return torch.zeros_like(x)
        return 2.0 * (x - x_min) / denom - 1.0

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """PAA + rescale + arccos polar encoding.

        Args:
            x: 1-D real tensor [L]

        Returns:
            phi: 1-D tensor [n] of angles in [0, π]
        """
        x = x.float()
        x_paa = self._paa(x, self.n)               # [n]
        x_scaled = self._rescale(x_paa)             # [n] in [-1, 1]
        phi = torch.arccos(x_scaled.clamp(-1 + 1e-6, 1 - 1e-6))  # [n]
        return phi

    @staticmethod
    def _gasf(phi: torch.Tensor) -> torch.Tensor:
        """GASF[i,j] = cos(φ_i + φ_j).

        Args:
            phi: 1-D tensor [n]

        Returns:
            2-D tensor [n, n]
        """
        outer_sum = phi.unsqueeze(0) + phi.unsqueeze(1)  # [n, n]
        return torch.cos(outer_sum)

    @staticmethod
    def _gadf(phi: torch.Tensor) -> torch.Tensor:
        """GADF[i,j] = sin(φ_i - φ_j).

        Args:
            phi: 1-D tensor [n]

        Returns:
            2-D tensor [n, n]
        """
        outer_diff = phi.unsqueeze(0) - phi.unsqueeze(1)  # [n, n]
        return torch.sin(outer_diff)
