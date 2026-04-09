"""Neyman-Pearson energy detector for drone RF signal presence detection."""

import logging
import math
from collections import deque
from typing import Deque, List, Optional

import torch

logger = logging.getLogger("rfml.features")


class EnergyDetector:
    """Neyman-Pearson energy detector for RF signal presence.

    Computes per-sample power (|z|²) over an IQ segment and compares the
    mean energy against a threshold derived from the calibrated noise floor.
    The threshold is set such that the probability of false alarm does not
    exceed ``p_fa`` under the null hypothesis (noise only).

    A rolling buffer of recent energy observations enables self-calibrating
    noise-floor tracking: whenever the buffer fills, the lower
    ``calibration_percentile`` of stored values is used to re-estimate the
    noise statistics, preventing signal leakage from biasing the floor upward.

    Args:
        p_fa: Target false-alarm probability (Neyman-Pearson criterion).
            Must be in (0, 1). Default ``0.01``.
        buffer_size: Number of energy observations retained in the rolling
            buffer before a noise-floor recalibration is triggered.
            Default ``1024``.
        calibration_percentile: Percentile (0–100) of the rolling buffer
            used to estimate the noise floor on recalibration.  Lower values
            track quieter segments more aggressively.  Default ``10.0``.
    """

    def __init__(
        self,
        p_fa: float = 0.01,
        buffer_size: int = 1024,
        calibration_percentile: float = 10.0,
    ) -> None:
        if not (0.0 < p_fa < 1.0):
            raise ValueError(f"p_fa must be in (0, 1), got {p_fa}")
        if buffer_size < 2:
            raise ValueError(f"buffer_size must be >= 2, got {buffer_size}")
        if not (0.0 < calibration_percentile < 100.0):
            raise ValueError(
                f"calibration_percentile must be in (0, 100), got {calibration_percentile}"
            )

        self.p_fa = p_fa
        self.buffer_size = buffer_size
        self.calibration_percentile = calibration_percentile

        # Calibrated noise statistics
        self._mu_noise: Optional[float] = None
        self._sigma_noise: Optional[float] = None

        # Rolling buffer of recent energy observations
        self._buffer: Deque[float] = deque(maxlen=buffer_size)

        logger.debug(
            "EnergyDetector: p_fa=%.4f buffer_size=%d calibration_percentile=%.1f",
            p_fa,
            buffer_size,
            calibration_percentile,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, noise_iq: torch.Tensor) -> None:
        """Calibrate the noise floor from known noise-only IQ samples.

        Computes the per-sample power spectral density (|z|²) across the
        input, then records the mean and standard deviation as the baseline
        noise statistics used by :meth:`detect`.

        Args:
            noise_iq: Float tensor of shape ``[2, N]`` where row 0 is the
                I channel and row 1 is the Q channel.  The tensor may reside
                on any device; it is moved to CPU internally.
        """
        energy_vals = self._compute_energy_vector(noise_iq)
        self._mu_noise = float(energy_vals.mean().item())
        self._sigma_noise = float(energy_vals.std(correction=1).item())
        # Clamp to avoid division-by-zero if input has no variance
        self._sigma_noise = max(self._sigma_noise, 1e-12)
        logger.debug(
            "EnergyDetector calibrated: mu_noise=%.6g sigma_noise=%.6g",
            self._mu_noise,
            self._sigma_noise,
        )

    def detect(self, iq: torch.Tensor) -> bool:
        """Detect whether a drone signal is present in the IQ segment.

        Implements the Neyman-Pearson likelihood-ratio test with threshold::

            T = mu_noise + Q_inv(p_fa) * sigma_noise

        The mean energy of the segment is compared against ``T``.  The
        rolling buffer is updated with each call and triggers automatic
        noise-floor recalibration when it becomes full.

        Args:
            iq: Float tensor of shape ``[2, N]`` (I and Q channels).

        Returns:
            ``True`` if a signal is detected, ``False`` otherwise.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called yet.
        """
        if self._mu_noise is None or self._sigma_noise is None:
            raise RuntimeError(
                "EnergyDetector must be calibrated before calling detect(). "
                "Call calibrate() with noise-only samples first."
            )

        energy_vals = self._compute_energy_vector(iq)
        segment_energy = float(energy_vals.mean().item())

        self._update_buffer(segment_energy)

        threshold = self._mu_noise + self._q_inv(self.p_fa) * self._sigma_noise
        detected = segment_energy > threshold

        logger.debug(
            "detect: energy=%.6g threshold=%.6g detected=%s",
            segment_energy,
            threshold,
            detected,
        )
        return detected

    def snr_estimate_db(self, iq: torch.Tensor) -> float:
        """Estimate the SNR of an IQ segment relative to the noise floor.

        Computes::

            SNR_dB = 10 * log10(max(energy / mu_noise - 1, 1e-10))

        A result of ``0 dB`` indicates that the segment energy equals twice
        the noise floor; negative values indicate sub-noise-floor signals.

        Args:
            iq: Float tensor of shape ``[2, N]`` (I and Q channels).

        Returns:
            Estimated SNR in dB as a Python float.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called yet.
        """
        if self._mu_noise is None:
            raise RuntimeError(
                "EnergyDetector must be calibrated before calling snr_estimate_db(). "
                "Call calibrate() with noise-only samples first."
            )

        energy_vals = self._compute_energy_vector(iq)
        segment_energy = float(energy_vals.mean().item())

        snr_linear = max(segment_energy / max(self._mu_noise, 1e-30) - 1.0, 1e-10)
        return 10.0 * math.log10(snr_linear)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _q_inv(self, p: float) -> float:
        """Inverse Q-function: returns x such that Q(x) = p.

        Uses the Abramowitz & Stegun rational approximation (formula 26.2.17)
        for the inverse complementary CDF of the standard normal, accurate
        to approximately ±4.5e-4 over the full range.

        Args:
            p: Probability in (0, 1).

        Returns:
            The value ``x`` satisfying ``Q(x) = p`` (i.e. the (1-p) quantile
            of the standard normal distribution).
        """
        # A&S 26.2.17 coefficients
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        # Work in terms of the upper tail: Q(x) = p -> Phi(x) = 1-p
        # Use symmetry: if p <= 0.5 work with p, else with 1-p and negate
        if p <= 0.5:
            t = math.sqrt(-2.0 * math.log(p))
            sign = 1.0
        else:
            t = math.sqrt(-2.0 * math.log(1.0 - p))
            sign = -1.0

        numerator = c0 + c1 * t + c2 * t * t
        denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
        x = t - numerator / denominator
        return sign * x

    def _update_buffer(self, energy: float) -> None:
        """Append an energy observation to the rolling buffer.

        When the buffer reaches capacity, the noise floor is re-estimated
        from the low-energy tail (below ``calibration_percentile``) of the
        buffered observations, updating ``_mu_noise`` and ``_sigma_noise``
        in-place.

        Args:
            energy: Scalar energy observation to append.
        """
        self._buffer.append(energy)

        if len(self._buffer) >= self.buffer_size:
            sorted_vals: List[float] = sorted(self._buffer)
            cutoff_idx = max(
                1,
                int(math.ceil(self.calibration_percentile / 100.0 * len(sorted_vals))),
            )
            low_energy = sorted_vals[:cutoff_idx]

            new_mu = sum(low_energy) / len(low_energy)
            if len(low_energy) >= 2:
                variance = sum((v - new_mu) ** 2 for v in low_energy) / (
                    len(low_energy) - 1
                )
                new_sigma = max(math.sqrt(variance), 1e-12)
            else:
                new_sigma = self._sigma_noise if self._sigma_noise is not None else 1e-12

            logger.debug(
                "_update_buffer: recalibrated mu_noise=%.6g -> %.6g, "
                "sigma_noise=%.6g -> %.6g",
                self._mu_noise,
                new_mu,
                self._sigma_noise,
                new_sigma,
            )
            self._mu_noise = new_mu
            self._sigma_noise = new_sigma
            self._buffer.clear()

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_energy_vector(iq: torch.Tensor) -> torch.Tensor:
        """Compute per-sample power |z|² from a ``[2, N]`` IQ tensor.

        Args:
            iq: Float tensor of shape ``[2, N]``.

        Returns:
            1-D float tensor of length ``N`` containing per-sample power.
        """
        # iq[0] = I, iq[1] = Q; power = I^2 + Q^2
        return iq[0].float() ** 2 + iq[1].float() ** 2
