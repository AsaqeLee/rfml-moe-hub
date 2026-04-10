"""Variational Mode Decomposition (VMD) for RF signal denoising."""

import logging
from typing import Tuple

import torch

logger = logging.getLogger("rfml.features")


class VMDExtractor:
    """Variational Mode Decomposition for denoising RF IQ signals.

    Decomposes a signal into K band-limited Intrinsic Mode Functions (IMFs)
    via the ADMM algorithm in the spectral domain.  Effective IMFs are selected
    by Pearson correlation with the original signal and summed to produce a
    denoised output.

    Args:
        K       (int)   - number of modes to decompose into, default 5
        alpha   (float) - bandwidth penalty (larger = narrower bands), default 2000.0
        tau     (float) - Lagrangian multiplier step size (0 = no dual update), default 0.0
        max_iter(int)   - maximum ADMM iterations, default 500
        tol     (float) - convergence threshold (relative change in u_hat), default 1e-7
    """

    def __init__(
        self,
        K: int = 5,
        alpha: float = 2000.0,
        tau: float = 0.0,
        max_iter: int = 500,
        tol: float = 1e-7,
    ) -> None:
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol

        logger.debug(
            "VMDExtractor: K=%d alpha=%.1f tau=%.3f max_iter=%d tol=%.2e",
            K, alpha, tau, max_iter, tol,
        )

    # ------------------------------------------------------------------
    # Core ADMM
    # ------------------------------------------------------------------

    def decompose(
        self, signal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose a 1-D real signal into K IMFs via VMD (ADMM).

        Args:
            signal: real 1-D tensor [N]

        Returns:
            u_hat : real tensor [K, N] - IMFs in time domain
            omega : real tensor [K]    - centre frequencies (normalised, in [0, 0.5])
        """
        signal = signal.float()
        N = signal.shape[0]
        device = signal.device

        # Frequency axis (only positive half for analytic signal)
        freqs = torch.arange(N, dtype=torch.float32, device=device) / N  # [0, 1)

        # FFT of input
        f_hat = torch.fft.fft(signal)  # [N] complex

        # Initialise K centre frequencies linearly in (0, 0.5)
        omega = torch.linspace(0.0, 0.5, steps=self.K + 2, device=device)[1:-1]  # [K]

        # Mode spectra — complex [K, N], initialised to zero
        u_hat = torch.zeros(self.K, N, dtype=torch.complex64, device=device)

        # Dual variable — complex [N]
        lambda_hat = torch.zeros(N, dtype=torch.complex64, device=device)

        # Broadcast frequency axis for batch operations: [1, N]
        freqs_b = freqs.unsqueeze(0)

        for iteration in range(self.max_iter):
            u_hat_prev = u_hat.clone()

            # Accumulate sum of all modes: [N] complex
            u_sum = u_hat.sum(dim=0)

            for k in range(self.K):
                # Remove mode k from the sum
                u_sum_minus_k = u_sum - u_hat[k]

                # Denominator: 1 + 2*alpha*(freq - omega_k)^2  [N]
                denom = 1.0 + 2.0 * self.alpha * (freqs - omega[k]) ** 2

                # Numerator: f_hat - sum_{i!=k} u_hat_i + lambda_hat/2
                numerator = f_hat - u_sum_minus_k + lambda_hat / 2.0

                # Update mode spectrum
                new_uk = numerator / denom
                u_sum = u_sum - u_hat[k] + new_uk
                u_hat[k] = new_uk

                # Update centre frequency: weighted centroid of |u_hat_k|^2
                power = (u_hat[k].abs() ** 2)  # [N]
                # Use only positive frequencies (0..N//2) to stay in [0, 0.5]
                half = N // 2
                power_pos = power[:half]
                freqs_pos = freqs[:half]
                omega[k] = (freqs_pos * power_pos).sum() / (power_pos.sum() + 1e-12)

            # Dual variable update
            if self.tau > 0.0:
                lambda_hat = lambda_hat + self.tau * (u_hat.sum(dim=0) - f_hat)

            # Convergence check: relative Frobenius change in u_hat spectra
            delta = (u_hat - u_hat_prev).abs().pow(2).sum()
            norm = u_hat_prev.abs().pow(2).sum() + 1e-12
            if (delta / norm).item() < self.tol:
                logger.debug("VMD converged at iteration %d", iteration + 1)
                break

        # Convert spectra to time domain — take real part [K, N]
        u_time = torch.fft.ifft(u_hat, dim=-1).real

        return u_time, omega

    # ------------------------------------------------------------------
    # IMF selection
    # ------------------------------------------------------------------

    def select_effective_imfs(
        self,
        signal: torch.Tensor,
        original: torch.Tensor,
        threshold: float = 0.3,
    ) -> torch.Tensor:
        """Select IMFs correlated with the original signal and sum them.

        Args:
            signal   : IMFs in time domain [K, N]
            original : original 1-D signal [N]
            threshold: minimum |Pearson CC| to accept an IMF, default 0.3

        Returns:
            denoised signal [N] (sum of effective IMFs)
        """
        K = signal.shape[0]
        original = original.float()
        orig_zero = original - original.mean()
        orig_std = orig_zero.pow(2).sum().sqrt() + 1e-12

        pccs = torch.zeros(K, device=signal.device)
        for k in range(K):
            imf = signal[k].float()
            imf_zero = imf - imf.mean()
            imf_std = imf_zero.pow(2).sum().sqrt() + 1e-12
            pccs[k] = (imf_zero * orig_zero).sum() / (imf_std * orig_std)

        pcc_abs = pccs.abs()
        mask = pcc_abs > threshold

        if not mask.any():
            # Fall back to the single IMF with highest |PCC|
            best = pcc_abs.argmax()
            mask = torch.zeros(K, dtype=torch.bool, device=signal.device)
            mask[best] = True
            logger.debug(
                "VMD: no IMF exceeded threshold %.2f; using IMF %d (pcc=%.4f)",
                threshold, best.item(), pcc_abs[best].item(),
            )

        denoised = signal[mask].sum(dim=0)
        logger.debug(
            "VMD: selected %d/%d IMFs (threshold=%.2f)",
            int(mask.sum().item()), K, threshold,
        )
        return denoised

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------

    def extract(self, iq: torch.Tensor) -> torch.Tensor:
        """Denoise an IQ signal using VMD on its magnitude envelope.

        Args:
            iq: [2, N] float tensor (I channel first, Q channel second)

        Returns:
            denoised magnitude [N] float tensor
        """
        i_ch = iq[0].float()
        q_ch = iq[1].float()
        magnitude = torch.sqrt(i_ch ** 2 + q_ch ** 2)

        u_time, omega = self.decompose(magnitude)
        denoised = self.select_effective_imfs(u_time, magnitude)

        logger.debug(
            "VMD extract: input N=%d, centre_freqs=%s",
            magnitude.shape[0],
            [f"{w:.4f}" for w in omega.tolist()],
        )
        return denoised
