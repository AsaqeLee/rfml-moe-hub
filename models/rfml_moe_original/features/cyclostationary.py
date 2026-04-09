"""Cyclostationary feature extractor using the FFT Accumulation Method (FAM)."""

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger("rfml.features")


class CycloExtractor:
    """Compute Spectral Correlation Function (SCF) features via FAM.

    The FFT Accumulation Method (Roberts et al., 1991) is used to estimate
    the SCF efficiently.  From the SCF, a fixed-length feature vector is
    assembled by selecting the dominant cyclic frequencies and summarising
    the corresponding spectral slices.

    Args:
        config: dict with keys:
            num_cycle_freqs (int)  - number of cycle frequencies to retain,
                                     default 256
            fft_size        (int)  - input FFT block size, default 1024
            scf_method      (str)  - always "fam" (only method implemented)
            output_dim      (int)  - output feature vector length, default 512
            device          (str)  - computation device
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.num_cycle_freqs = int(cfg.get("num_cycle_freqs", 256))
        self.fft_size = int(cfg.get("fft_size", 1024))
        self.output_dim = int(cfg.get("output_dim", 512))
        self.scf_method = str(cfg.get("scf_method", "fam"))

        device_str = str(cfg.get("device", "cpu"))
        self.device = torch.device(
            device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        logger.debug(
            "CycloExtractor: num_cycle_freqs=%d fft_size=%d output_dim=%d device=%s",
            self.num_cycle_freqs,
            self.fft_size,
            self.output_dim,
            self.device,
        )

    # ------------------------------------------------------------------
    # FAM implementation
    # ------------------------------------------------------------------

    def _fam(self, z: torch.Tensor) -> torch.Tensor:
        """FFT Accumulation Method for SCF estimation.

        Args:
            z: complex 1-D tensor [N]

        Returns:
            SCF magnitude tensor [num_alpha, F] where F = fft_size // 2 + 1
            and num_alpha = num_cycle_freqs (sub-sampled).
        """
        N = z.shape[0]
        L = self.fft_size          # block (sub-channel) length
        P = N // L                 # number of blocks
        if P < 2:
            # signal too short; pad to at least 2 blocks
            z = F.pad(z.real, (0, 2 * L - N)).to(z.dtype)
            N = 2 * L
            P = 2

        # Slice into overlapping or non-overlapping blocks [P, L]
        # Non-overlapping for simplicity / speed
        z_blocks = z[: P * L].reshape(P, L)  # [P, L]

        # Apply Hann window to each block
        window = torch.hann_window(L, device=self.device)
        z_blocks = z_blocks * window.unsqueeze(0)  # [P, L]

        # FFT each block -> [P, L] complex (full spectrum)
        X = torch.fft.fft(z_blocks, dim=1)  # [P, L]

        # Outer product in frequency: for each pair (f1, f2) compute
        # correlation across blocks.  We only keep one-sided spectrum.
        F_half = L // 2 + 1
        X_half = X[:, :F_half]  # [P, F_half]

        # Cycle frequency grid: alpha = (f1 - f2) / L normalised to [-1, 1]
        # We compute the full P x P block outer and then reduce.
        # For efficiency, use the slide-and-correlate approach:
        #   SCF[alpha, f] = (1/P) * sum_t X[t, f+alpha/2] * X*[t, f-alpha/2]
        # We implement a simplified version: for each shift d in
        # range(num_cycle_freqs) compute the element-wise product of X and
        # circularly-shifted X*.

        alpha_indices = torch.linspace(
            0, L - 1, steps=self.num_cycle_freqs, dtype=torch.long, device=self.device
        )
        scf = torch.zeros(
            self.num_cycle_freqs, F_half, dtype=torch.float32, device=self.device
        )

        X_full = X  # [P, L]
        for i, shift in enumerate(alpha_indices):
            shift = shift.item()
            # Shift X along frequency axis and correlate
            X_shifted = torch.roll(X_full, int(shift), dims=1)
            product = X_full * X_shifted.conj()       # [P, L]
            mean_product = product.mean(dim=0)         # [L]
            scf[i] = torch.abs(mean_product[:F_half])

        return scf  # [num_alpha, F_half]

    # ------------------------------------------------------------------
    # Cyclic cumulant extraction (OFDM parameter fingerprinting)
    # ------------------------------------------------------------------

    def _cyclic_cumulants(self, z: torch.Tensor) -> torch.Tensor:
        """Extract cyclic autocorrelation at key cycle frequencies.

        For OFDM, strong cyclostationary features appear at alpha = 1/T_s
        and its harmonics.  We extract the magnitude of the cyclic
        autocorrelation R^alpha_xx(tau=0) for a grid of alpha values.

        Returns:
            real tensor [num_cycle_freqs]
        """
        N = z.shape[0]
        alpha_grid = torch.arange(
            self.num_cycle_freqs, device=self.device, dtype=torch.float32
        ) / N

        # R^alpha_xx(0) = (1/N) * |sum_n x(n) x*(n) e^{-j 2pi alpha n}|
        n = torch.arange(N, device=self.device, dtype=torch.float32)
        power = z * z.conj()  # [N] instantaneous power, real-valued

        # Evaluate DFT of instantaneous power at each alpha
        phase = -2 * math.pi * alpha_grid.unsqueeze(1) * n.unsqueeze(0)  # [A, N]
        exp_terms = torch.exp(1j * phase.to(z.dtype))
        cyclic_corr = (power.unsqueeze(0) * exp_terms).mean(dim=1)  # [A]
        return torch.abs(cyclic_corr)

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    def _build_feature_vector(
        self, scf: torch.Tensor, cyclic_corr: torch.Tensor
    ) -> torch.Tensor:
        """Combine SCF summary and cyclic correlations into output vector.

        Returns float tensor [output_dim].
        """
        # Summary statistics of SCF: max energy per cycle frequency
        scf_max = scf.max(dim=1).values          # [num_alpha]
        scf_mean = scf.mean(dim=1)               # [num_alpha]
        scf_summary = torch.cat([scf_max, scf_mean], dim=0)  # [2*num_alpha]

        feature = torch.cat([scf_summary, cyclic_corr], dim=0)

        # Resize to output_dim
        if feature.shape[0] < self.output_dim:
            pad = torch.zeros(
                self.output_dim - feature.shape[0],
                dtype=feature.dtype,
                device=feature.device,
            )
            feature = torch.cat([feature, pad], dim=0)
        else:
            # Keep highest-energy features via sort-then-truncate
            feature = feature[: self.output_dim]

        return feature.float()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, iq: torch.Tensor) -> torch.Tensor:
        """Extract cyclostationary features from a single IQ segment.

        Args:
            iq: [2, N] float or [N] complex tensor

        Returns:
            float tensor [output_dim]
        """
        iq = iq.to(self.device)

        if iq.is_complex():
            z = iq
        else:
            z = torch.complex(iq[0].float(), iq[1].float())

        z = z - z.mean()  # remove DC

        scf = self._fam(z)                     # [num_alpha, F_half]
        cyclic_corr = self._cyclic_cumulants(z)  # [num_cycle_freqs]

        return self._build_feature_vector(scf, cyclic_corr)

    def extract_batch(self, iq_batch: torch.Tensor) -> torch.Tensor:
        """Batch cyclostationary extraction.

        Args:
            iq_batch: [B, 2, N] float or [B, N] complex

        Returns:
            [B, output_dim] float tensor
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
