"""RF-aware data augmentation pipeline for IQ signal tensors."""

import logging
import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger("rfml.features")


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------


def awgn(
    iq: torch.Tensor,
    snr_db: Optional[float] = None,
    snr_range: Tuple[float, float] = (-20.0, 20.0),
) -> torch.Tensor:
    """Add AWGN noise at a specified or randomly-drawn SNR.

    Args:
        iq:        [2, N] float tensor (I, Q rows)
        snr_db:    target SNR in dB; if None, sampled uniformly from snr_range
        snr_range: (min_snr_db, max_snr_db) for random sampling

    Returns:
        [2, N] noisy tensor
    """
    if snr_db is None:
        snr_db = float(
            torch.empty(1).uniform_(snr_range[0], snr_range[1]).item()
        )

    signal_power = (iq ** 2).mean()
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = (signal_power / (snr_linear + 1e-12)).clamp(min=1e-20)
    noise = torch.randn_like(iq) * noise_power.sqrt()
    return iq + noise


def cyclic_time_shift(iq: torch.Tensor, max_shift_frac: float = 0.5) -> torch.Tensor:
    """Apply a random cyclic time shift.

    Args:
        iq:              [2, N] float tensor
        max_shift_frac:  maximum shift as a fraction of signal length

    Returns:
        [2, N] shifted tensor
    """
    N = iq.shape[1]
    max_shift = max(1, int(N * max_shift_frac))
    shift = torch.randint(0, max_shift + 1, (1,)).item()
    return torch.roll(iq, int(shift), dims=1)


def carrier_frequency_offset(
    iq: torch.Tensor,
    max_cfo_hz: float = 1000.0,
    sampling_rate: float = 100e6,
) -> torch.Tensor:
    """Apply a random carrier frequency offset via phase rotation e^{j2pi*Δf*t}.

    Args:
        iq:            [2, N] float tensor
        max_cfo_hz:    maximum CFO magnitude in Hz
        sampling_rate: sampling rate in Hz

    Returns:
        [2, N] rotated tensor
    """
    N = iq.shape[1]
    delta_f = float(
        torch.empty(1).uniform_(-max_cfo_hz, max_cfo_hz).item()
    )
    t = torch.arange(N, dtype=torch.float32, device=iq.device)
    phase = 2.0 * math.pi * delta_f / sampling_rate * t  # [N]

    z = torch.complex(iq[0], iq[1])          # [N] complex
    rotation = torch.exp(1j * phase.to(z.dtype))
    z_rot = z * rotation
    return torch.stack([z_rot.real, z_rot.imag], dim=0)


def amplitude_scale(
    iq: torch.Tensor,
    scale_range: Tuple[float, float] = (0.5, 2.0),
) -> torch.Tensor:
    """Multiply signal amplitude by a random scale factor.

    Args:
        iq:          [2, N] float tensor
        scale_range: (min_scale, max_scale)

    Returns:
        [2, N] scaled tensor
    """
    scale = float(
        torch.empty(1).uniform_(scale_range[0], scale_range[1]).item()
    )
    return iq * scale


def multipath_fading(
    iq: torch.Tensor,
    num_paths: int = 3,
    max_delay_samples: int = 10,
    fading_type: str = "rayleigh",
    rician_k: float = 1.0,
) -> torch.Tensor:
    """Simulate multipath Rayleigh or Rician fading.

    Generates random complex path gains and integer delays, then sums
    the delayed copies.

    Args:
        iq:                 [2, N] float tensor
        num_paths:          number of multipath components (including LOS)
        max_delay_samples:  maximum delay in samples
        fading_type:        "rayleigh" or "rician"
        rician_k:           K-factor for Rician fading (K=0 -> Rayleigh)

    Returns:
        [2, N] faded tensor (same shape, normalised to original power)
    """
    z = torch.complex(iq[0], iq[1])  # [N]
    original_power = (z.abs() ** 2).mean().clamp(min=1e-12)

    result = torch.zeros_like(z)
    for p in range(num_paths):
        delay = torch.randint(0, max_delay_samples + 1, (1,)).item()

        if fading_type == "rician" and p == 0:
            # LOS component
            los_amp = math.sqrt(rician_k / (rician_k + 1))
            phase = float(torch.empty(1).uniform_(0, 2 * math.pi).item())
            gain = torch.tensor(
                los_amp * math.cos(phase) + 1j * los_amp * math.sin(phase),
                dtype=z.dtype,
                device=z.device,
            )
            scatter_amp = math.sqrt(0.5 / (rician_k + 1))
            scatter_re = torch.randn(1, device=z.device).item() * scatter_amp
            scatter_im = torch.randn(1, device=z.device).item() * scatter_amp
            gain = gain + torch.tensor(
                scatter_re + 1j * scatter_im, dtype=z.dtype, device=z.device
            )
        else:
            # Rayleigh path
            re = torch.randn(1, device=z.device).item() * (1.0 / math.sqrt(2 * num_paths))
            im = torch.randn(1, device=z.device).item() * (1.0 / math.sqrt(2 * num_paths))
            gain = torch.tensor(re + 1j * im, dtype=z.dtype, device=z.device)

        z_delayed = torch.roll(z, int(delay), dims=0)
        result = result + gain * z_delayed

    # Normalise to original power
    out_power = (result.abs() ** 2).mean().clamp(min=1e-12)
    result = result * (original_power / out_power).sqrt()

    return torch.stack([result.real, result.imag], dim=0)


def interference_injection(
    iq: torch.Tensor,
    sir_db: float = 10.0,
    interference_type: str = "tone",
) -> torch.Tensor:
    """Inject an interference signal (tone or narrowband noise).

    Args:
        iq:                [2, N] float tensor
        sir_db:            signal-to-interference ratio in dB
        interference_type: "tone" or "noise"

    Returns:
        [2, N] tensor with interference added
    """
    N = iq.shape[1]
    signal_power = (iq ** 2).mean()
    sir_linear = 10 ** (sir_db / 10.0)
    interf_power = signal_power / (sir_linear + 1e-12)

    if interference_type == "tone":
        freq = float(torch.empty(1).uniform_(0.05, 0.45).item())  # normalised
        phase = float(torch.empty(1).uniform_(0, 2 * math.pi).item())
        t = torch.arange(N, dtype=torch.float32, device=iq.device)
        tone = torch.stack(
            [
                torch.cos(2 * math.pi * freq * t + phase),
                torch.sin(2 * math.pi * freq * t + phase),
            ],
            dim=0,
        )
        amplitude = interf_power.sqrt()
        interf = tone * amplitude
    else:
        interf = torch.randn_like(iq) * interf_power.sqrt()

    return iq + interf


# ---------------------------------------------------------------------------
# Augmentor class
# ---------------------------------------------------------------------------


class RFAugmentor:
    """Composable RF augmentation pipeline.

    Each augmentation is applied independently with its configured probability.
    All operations work on [2, N] float tensors (I, Q rows) or [B, 2, N] batches.

    Args:
        config: dict with keys corresponding to augmentation names.
            Each sub-dict has a "prob" key (float 0-1) plus augmentation-
            specific parameters.  Example::

                awgn:
                    prob: 0.8
                    snr_range: [-10, 20]
                cyclic_time_shift:
                    prob: 0.5
                carrier_frequency_offset:
                    prob: 0.5
                    max_cfo_hz: 1000
                amplitude_scale:
                    prob: 0.5
                    scale_range: [0.5, 2.0]
                multipath_fading:
                    prob: 0.3
                    fading_type: "rayleigh"
                interference_injection:
                    prob: 0.2
                    sir_db: 10
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}

        # AWGN
        awgn_cfg = cfg.get("awgn", {})
        self._awgn_prob = float(awgn_cfg.get("prob", 0.8))
        self._awgn_snr_range = tuple(awgn_cfg.get("snr_range", [-20, 20]))

        # Cyclic time shift
        cts_cfg = cfg.get("cyclic_time_shift", {})
        self._cts_prob = float(cts_cfg.get("prob", 0.5))
        self._cts_max_frac = float(cts_cfg.get("max_shift_frac", 0.5))

        # CFO
        cfo_cfg = cfg.get("carrier_frequency_offset", {})
        self._cfo_prob = float(cfo_cfg.get("prob", 0.5))
        self._cfo_max_hz = float(cfo_cfg.get("max_cfo_hz", 1000.0))
        self._cfo_fs = float(cfo_cfg.get("sampling_rate", 100e6))

        # Amplitude scale
        amp_cfg = cfg.get("amplitude_scale", {})
        self._amp_prob = float(amp_cfg.get("prob", 0.5))
        self._amp_range = tuple(amp_cfg.get("scale_range", [0.5, 2.0]))

        # Multipath fading
        mp_cfg = cfg.get("multipath_fading", {})
        self._mp_prob = float(mp_cfg.get("prob", 0.3))
        self._mp_num_paths = int(mp_cfg.get("num_paths", 3))
        self._mp_max_delay = int(mp_cfg.get("max_delay_samples", 10))
        self._mp_fading_type = str(mp_cfg.get("fading_type", "rayleigh"))
        self._mp_rician_k = float(mp_cfg.get("rician_k", 1.0))

        # Interference injection
        inj_cfg = cfg.get("interference_injection", {})
        self._inj_prob = float(inj_cfg.get("prob", 0.2))
        self._inj_sir_db = float(inj_cfg.get("sir_db", 10.0))
        self._inj_type = str(inj_cfg.get("interference_type", "tone"))

        logger.debug(
            "RFAugmentor: probs=[awgn=%.2f, cts=%.2f, cfo=%.2f, amp=%.2f, mp=%.2f, inj=%.2f]",
            self._awgn_prob, self._cts_prob, self._cfo_prob,
            self._amp_prob, self._mp_prob, self._inj_prob,
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_with_prob(
        fn: Callable, iq: torch.Tensor, prob: float
    ) -> torch.Tensor:
        if torch.rand(1).item() < prob:
            return fn(iq)
        return iq

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def augment(self, iq: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation chain to a single IQ segment.

        Args:
            iq: [2, N] float tensor

        Returns:
            [2, N] augmented float tensor
        """
        iq = iq.float()

        iq = self._apply_with_prob(
            lambda x: awgn(x, snr_range=self._awgn_snr_range),
            iq,
            self._awgn_prob,
        )
        iq = self._apply_with_prob(
            lambda x: cyclic_time_shift(x, self._cts_max_frac),
            iq,
            self._cts_prob,
        )
        iq = self._apply_with_prob(
            lambda x: carrier_frequency_offset(x, self._cfo_max_hz, self._cfo_fs),
            iq,
            self._cfo_prob,
        )
        iq = self._apply_with_prob(
            lambda x: amplitude_scale(x, self._amp_range),
            iq,
            self._amp_prob,
        )
        iq = self._apply_with_prob(
            lambda x: multipath_fading(
                x,
                self._mp_num_paths,
                self._mp_max_delay,
                self._mp_fading_type,
                self._mp_rician_k,
            ),
            iq,
            self._mp_prob,
        )
        iq = self._apply_with_prob(
            lambda x: interference_injection(x, self._inj_sir_db, self._inj_type),
            iq,
            self._inj_prob,
        )
        return iq

    def augment_batch(self, iq_batch: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a batch, independently per sample.

        Args:
            iq_batch: [B, 2, N] float tensor

        Returns:
            [B, 2, N] augmented float tensor
        """
        results = [self.augment(iq_batch[b]) for b in range(iq_batch.shape[0])]
        return torch.stack(results, dim=0)

    def __call__(
        self, iq: torch.Tensor, batch: bool = False
    ) -> torch.Tensor:
        if batch:
            return self.augment_batch(iq)
        return self.augment(iq)
