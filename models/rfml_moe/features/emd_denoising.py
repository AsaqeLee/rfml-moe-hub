"""Empirical Mode Decomposition (EMD) denoiser for IQ RF signals.

Decomposes a signal into Intrinsic Mode Functions (IMFs) via the sifting
algorithm, discards the high-frequency noise IMFs, and reconstructs a
denoised version of the original IQ signal.

Not an nn.Module — this is an algorithmic preprocessor that operates
directly on torch.Tensor inputs.
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("rfml.features")

_SCIPY_AVAILABLE = False
try:
    from scipy.interpolate import CubicSpline as _CubicSpline  # type: ignore

    _SCIPY_AVAILABLE = True
    logger.debug("EMDDenoiser: scipy CubicSpline available.")
except Exception:
    logger.warning(
        "scipy not available. EMDDenoiser will use linear envelope interpolation. "
        "Install with: pip install scipy"
    )


class EMDDenoiser:
    """Empirical Mode Decomposition denoiser for IQ RF signals.

    Computes the analytic magnitude of the IQ pair, decomposes it into
    Intrinsic Mode Functions (IMFs) using the sifting algorithm, discards
    the first `noise_imfs` high-frequency IMFs (assumed dominated by noise),
    and scales the original IQ samples proportionally to the reconstructed
    denoised magnitude.

    Args:
        num_imfs:    Total number of IMFs to extract before treating the
                     remainder as a residual.  Default: 4.
        noise_imfs:  Number of leading (highest-frequency) IMFs to discard.
                     Must be < num_imfs.  Default: 2.
        max_sifting: Maximum sifting iterations per IMF.  Default: 20.
    """

    def __init__(
        self,
        num_imfs: int = 4,
        noise_imfs: int = 2,
        max_sifting: int = 20,
    ) -> None:
        if noise_imfs >= num_imfs:
            raise ValueError(
                f"noise_imfs ({noise_imfs}) must be less than num_imfs ({num_imfs})"
            )
        self.num_imfs = num_imfs
        self.noise_imfs = noise_imfs
        self.max_sifting = max_sifting

        logger.debug(
            "EMDDenoiser: num_imfs=%d noise_imfs=%d max_sifting=%d scipy=%s",
            self.num_imfs,
            self.noise_imfs,
            self.max_sifting,
            _SCIPY_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interpolate_envelope(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Interpolate an envelope curve at all sample positions.

        Args:
            x: 1-D array of knot indices (extrema locations).
            y: 1-D array of knot values (extrema amplitudes).
            t: 1-D array of query indices (all sample positions).

        Returns:
            Interpolated envelope as a 1-D numpy array of length len(t).
        """
        if _SCIPY_AVAILABLE:
            cs = _CubicSpline(x, y, bc_type="not-a-knot", extrapolate=True)
            return cs(t)
        # Linear fallback
        return np.interp(t, x, y)

    def _sift(self, signal: torch.Tensor) -> torch.Tensor:
        """Perform a single sifting pass on a 1-D signal.

        Finds local maxima and minima via sign changes of the first difference,
        interpolates upper and lower envelopes, subtracts the mean envelope.

        Args:
            signal: 1-D float tensor [N].

        Returns:
            Sifted signal as a 1-D float tensor [N].
        """
        s = signal.numpy().astype(np.float64)
        N = len(s)
        t = np.arange(N, dtype=np.float64)

        diff = np.diff(s)
        sign = np.sign(diff)
        sign_diff = np.diff(sign)

        # Local maxima: sign change from + to - (sign_diff < 0)
        max_idx = np.where(sign_diff < 0)[0] + 1
        # Local minima: sign change from - to + (sign_diff > 0)
        min_idx = np.where(sign_diff > 0)[0] + 1

        # Need at least 2 extrema of each type to interpolate an envelope
        if len(max_idx) < 2 or len(min_idx) < 2:
            return signal.clone()

        # Pad with boundary values so the envelope spans the full signal
        max_x = np.concatenate(([0], max_idx, [N - 1]))
        max_y = np.concatenate(([s[0]], s[max_idx], [s[-1]]))
        min_x = np.concatenate(([0], min_idx, [N - 1]))
        min_y = np.concatenate(([s[0]], s[min_idx], [s[-1]]))

        upper = self._interpolate_envelope(max_x, max_y, t)
        lower = self._interpolate_envelope(min_x, min_y, t)

        mean_env = (upper + lower) / 2.0
        sifted = s - mean_env

        return torch.from_numpy(sifted.astype(np.float32))

    def _is_imf(self, signal: torch.Tensor, prev: Optional[torch.Tensor]) -> bool:
        """Simple convergence check: SD between consecutive sifts is small."""
        if prev is None:
            return False
        diff = signal - prev
        sd = (diff ** 2).sum() / ((prev ** 2).sum() + 1e-12)
        return sd.item() < 0.2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, signal: torch.Tensor) -> list:
        """Extract IMFs from a 1-D signal via the sifting algorithm.

        Each iteration:
          1. Sift `max_sifting` times (or until convergence).
          2. Store the result as an IMF.
          3. Subtract the IMF from the residual.
          4. Repeat until `num_imfs` IMFs are extracted or the residual
             has fewer than 2 extrema.

        Args:
            signal: 1-D float tensor [N].

        Returns:
            List of IMF tensors followed by the residual tensor.
            Length is at most num_imfs + 1.
        """
        signal = signal.float()
        residual = signal.clone()
        imfs: list = []

        for _ in range(self.num_imfs):
            candidate = residual.clone()
            prev: Optional[torch.Tensor] = None

            for _ in range(self.max_sifting):
                sifted = self._sift(candidate)
                if self._is_imf(sifted, prev):
                    candidate = sifted
                    break
                prev = sifted
                candidate = sifted

            imfs.append(candidate)
            residual = residual - candidate

        imfs.append(residual)
        return imfs

    def denoise(self, iq: torch.Tensor) -> torch.Tensor:
        """Denoise a [2, N] IQ tensor using EMD.

        Processing steps:
          1. Compute magnitude |z| = sqrt(I^2 + Q^2).
          2. Decompose |z| into IMFs.
          3. Discard the first `noise_imfs` IMFs (high-frequency / noise).
          4. Reconstruct denoised magnitude from remaining IMFs + residual.
          5. Scale original IQ by ratio of denoised to original magnitude
             (samples where |z| < eps are left unchanged).

        Args:
            iq: [2, N] float tensor (row 0 = I, row 1 = Q).

        Returns:
            Denoised [2, N] float tensor.
        """
        if iq.ndim != 2 or iq.shape[0] != 2:
            raise ValueError(f"Expected iq shape [2, N], got {tuple(iq.shape)}")

        iq = iq.float()
        magnitude = torch.sqrt(iq[0] ** 2 + iq[1] ** 2)  # [N]

        imfs = self.decompose(magnitude)

        # Discard high-frequency noise IMFs; keep the rest
        kept = imfs[self.noise_imfs:]  # includes residual at end
        if not kept:
            logger.warning(
                "EMDDenoiser: all IMFs discarded (noise_imfs=%d >= total=%d). "
                "Returning original signal.",
                self.noise_imfs,
                len(imfs),
            )
            return iq.clone()

        denoised_mag = sum(kept)  # element-wise sum of tensors

        # Clamp to non-negative (magnitude must be >= 0)
        denoised_mag = torch.clamp(denoised_mag, min=0.0)

        # Scale IQ proportionally
        eps = 1e-9
        ratio = denoised_mag / (magnitude + eps)
        denoised_iq = iq * ratio.unsqueeze(0)  # broadcast over I/Q rows

        return denoised_iq
