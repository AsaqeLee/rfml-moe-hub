"""ITU/3GPP channel model implementations for RF signal simulation.

Provides predefined channel profiles based on ITU-R M.1225, IEEE 802.11,
and Watterson HF models. Each profile defines tap delays, tap powers,
Doppler spectrum characteristics, and K-factors for Rician fading.

All operations use PyTorch tensors and are device-agnostic (CPU/CUDA).
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger("rfml.data.synthetic")


# ---------------------------------------------------------------------------
# Predefined channel profiles
# ---------------------------------------------------------------------------

CHANNEL_PROFILES: Dict[str, Dict] = {
    # -----------------------------------------------------------------------
    # ITU-R M.1225 indoor / outdoor models
    # -----------------------------------------------------------------------
    "itu_indoor_a": {
        "delays_ns": [0, 50, 110, 170, 290, 310],
        "powers_db": [0.0, -3.0, -10.0, -18.0, -26.0, -32.0],
        "doppler_hz": 5.0,
        "k_factor": None,
        "description": "ITU Indoor Office A — low delay spread",
    },
    "itu_indoor_b": {
        "delays_ns": [0, 100, 200, 300, 500, 700],
        "powers_db": [0.0, -3.6, -7.2, -10.8, -18.0, -25.2],
        "doppler_hz": 5.0,
        "k_factor": None,
        "description": "ITU Indoor Office B — moderate delay spread",
    },
    "itu_pedestrian_a": {
        "delays_ns": [0, 110, 190, 410],
        "powers_db": [0.0, -9.7, -19.2, -22.8],
        "doppler_hz": 5.0,
        "k_factor": None,
        "description": "ITU Pedestrian A — low delay spread outdoor",
    },
    "itu_pedestrian_b": {
        "delays_ns": [0, 200, 800, 1200, 2300, 3700],
        "powers_db": [0.0, -0.9, -4.9, -8.0, -7.8, -23.9],
        "doppler_hz": 5.0,
        "k_factor": None,
        "description": "ITU Pedestrian B — moderate delay spread outdoor",
    },
    "itu_vehicular_a": {
        "delays_ns": [0, 310, 710, 1090, 1730, 2510],
        "powers_db": [0.0, -1.0, -9.0, -10.0, -15.0, -20.0],
        "doppler_hz": 185.0,
        "k_factor": None,
        "description": "ITU Vehicular A — high Doppler, moderate delay",
    },
    "itu_vehicular_b": {
        "delays_ns": [0, 300, 8900, 12900, 17100, 20000],
        "powers_db": [-2.5, 0.0, -12.8, -10.0, -25.2, -16.0],
        "doppler_hz": 185.0,
        "k_factor": None,
        "description": "ITU Vehicular B — high Doppler, large delay spread",
    },
    # -----------------------------------------------------------------------
    # IEEE 802.11 indoor channel models (TGn)
    # -----------------------------------------------------------------------
    "ieee_802_11_b": {
        "delays_ns": [0, 10, 20, 30, 40, 50, 60, 70, 80],
        "powers_db": [0.0, -5.4, -2.1, -12.5, -7.3, -15.2, -20.1, -22.5, -25.0],
        "doppler_hz": 3.0,
        "k_factor": None,
        "description": "IEEE 802.11 Model B — residential NLOS",
    },
    "ieee_802_11_c": {
        "delays_ns": [0, 10, 20, 30, 50, 80, 110, 140],
        "powers_db": [0.0, -2.1, -4.3, -6.5, -11.4, -15.7, -19.2, -23.1],
        "doppler_hz": 3.0,
        "k_factor": None,
        "description": "IEEE 802.11 Model C — small office NLOS",
    },
    "ieee_802_11_d": {
        "delays_ns": [0, 10, 20, 30, 50, 80, 110, 140, 180, 230, 280,
                       330, 380, 430, 490, 560, 640, 730],
        "powers_db": [0.0, -0.9, -1.7, -2.6, -4.4, -7.3, -9.3, -12.0,
                       -13.6, -18.1, -22.0, -24.1, -26.3, -28.5, -30.7,
                       -33.0, -35.2, -37.4],
        "doppler_hz": 3.0,
        "k_factor": None,
        "description": "IEEE 802.11 Model D — typical office LOS/NLOS",
    },
    # -----------------------------------------------------------------------
    # Watterson HF channel models (ITU-R F.1487)
    # -----------------------------------------------------------------------
    "watterson_good": {
        "delays_ns": [0, 500_000],
        "powers_db": [0.0, 0.0],
        "spreads_hz": [0.5, 0.5],
        "doppler_hz": 0.0,
        "k_factor": None,
        "description": "Watterson good conditions — minimal spread",
    },
    "watterson_moderate": {
        "delays_ns": [0, 1_000_000],
        "powers_db": [0.0, 0.0],
        "spreads_hz": [1.5, 1.5],
        "doppler_hz": 0.0,
        "k_factor": None,
        "description": "Watterson moderate — typical HF channel",
    },
    "watterson_poor": {
        "delays_ns": [0, 2_000_000],
        "powers_db": [0.0, 0.0],
        "spreads_hz": [10.0, 10.0],
        "doppler_hz": 0.0,
        "k_factor": None,
        "description": "Watterson poor — severe multipath HF",
    },
    # -----------------------------------------------------------------------
    # Drone-specific scenario profiles
    # -----------------------------------------------------------------------
    "drone_indoor": {
        "delays_ns": [0, 30, 70, 120, 200],
        "powers_db": [0.0, -3.0, -8.0, -14.0, -22.0],
        "doppler_hz": 10.0,
        "k_factor": 6.0,
        "description": "Indoor drone — short range, LOS dominant, low Doppler",
    },
    "drone_urban": {
        "delays_ns": [0, 150, 400, 900, 1600, 2300],
        "powers_db": [0.0, -2.0, -6.0, -10.0, -16.0, -22.0],
        "doppler_hz": 80.0,
        "k_factor": 3.0,
        "description": "Urban drone — moderate multipath, moderate Doppler",
    },
    "drone_suburban": {
        "delays_ns": [0, 200, 600, 1400, 2800, 5000],
        "powers_db": [0.0, -1.5, -5.0, -9.0, -14.0, -20.0],
        "doppler_hz": 150.0,
        "k_factor": 1.0,
        "description": "Suburban drone — wide delay spread, higher Doppler",
    },
    "drone_vehicular": {
        "delays_ns": [0, 300, 800, 1500, 3000, 6000, 10000],
        "powers_db": [0.0, -1.0, -4.0, -8.0, -12.0, -17.0, -24.0],
        "doppler_hz": 500.0,
        "k_factor": 0.5,
        "description": "Vehicular tracking — extreme Doppler, heavy multipath",
    },
}


class ITUChannelModel:
    """ITU/3GPP-based channel model for generating realistic fading responses.

    Generates time-varying complex channel impulse responses based on
    predefined or custom channel profiles. Supports both Rayleigh and
    Rician fading with configurable Doppler spectra.

    Args:
        profile: Name of a predefined profile from CHANNEL_PROFILES, or a
            custom dict with keys 'delays_ns', 'powers_db', 'doppler_hz',
            and optionally 'k_factor' and 'spreads_hz'.
        sample_rate: Signal sample rate in Hz.
        device: Torch device for all computations.
    """

    def __init__(
        self,
        profile: Union[str, Dict],
        sample_rate: float = 20e6,
        device: Optional[torch.device] = None,
    ):
        self.sample_rate = sample_rate
        self.device = device or torch.device("cpu")

        if isinstance(profile, str):
            if profile not in CHANNEL_PROFILES:
                raise ValueError(
                    f"Unknown channel profile '{profile}'. "
                    f"Available: {list(CHANNEL_PROFILES.keys())}"
                )
            self._profile = CHANNEL_PROFILES[profile].copy()
        else:
            self._profile = profile.copy()

        # Parse profile parameters
        delays_ns = self._profile["delays_ns"]
        powers_db = self._profile["powers_db"]

        if len(delays_ns) != len(powers_db):
            raise ValueError(
                f"delays_ns ({len(delays_ns)}) and powers_db ({len(powers_db)}) "
                f"must have the same length."
            )

        self.num_taps = len(delays_ns)
        self.doppler_hz = float(self._profile.get("doppler_hz", 0.0))
        self.k_factor = self._profile.get("k_factor", None)
        self.spreads_hz = self._profile.get("spreads_hz", None)

        # Convert delays from nanoseconds to sample indices
        delays_sec = torch.tensor(delays_ns, dtype=torch.float64) * 1e-9
        self.tap_delays = (delays_sec * sample_rate).round().long().to(self.device)

        # Convert power from dB to linear and normalise
        powers_linear = 10.0 ** (torch.tensor(powers_db, dtype=torch.float32) / 10.0)
        powers_linear = powers_linear / powers_linear.sum()
        self.tap_powers = powers_linear.to(self.device)

        logger.debug(
            "ITUChannelModel: %d taps, max_delay=%d samples, doppler=%.1f Hz, k=%s",
            self.num_taps,
            int(self.tap_delays.max().item()),
            self.doppler_hz,
            self.k_factor,
        )

    def generate_tap_gains(
        self,
        num_samples: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate time-varying complex tap gains.

        Args:
            num_samples: Number of time-domain samples to generate gains for.
            seed: Optional random seed for reproducibility.

        Returns:
            Complex tensor of shape (num_taps, num_samples) containing the
            time-varying gains for each channel tap.
        """
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)
        else:
            gen = None

        gains = torch.zeros(
            self.num_taps, num_samples,
            dtype=torch.cfloat, device=self.device,
        )

        for tap_idx in range(self.num_taps):
            tap_power = self.tap_powers[tap_idx].item()

            # Generate Rayleigh-distributed fading
            if self.doppler_hz > 0:
                # Shape Doppler spectrum via filtering white noise
                gain_complex = self._generate_doppler_faded(
                    num_samples, self.doppler_hz, gen,
                )
            else:
                # Static fading — single complex gain for entire block
                re = torch.randn(num_samples, device=self.device, generator=gen)
                im = torch.randn(num_samples, device=self.device, generator=gen)
                gain_complex = torch.complex(re, im) / math.sqrt(2.0)

            # Apply Watterson Gaussian spread if defined
            if self.spreads_hz is not None and tap_idx < len(self.spreads_hz):
                spread = self.spreads_hz[tap_idx]
                if spread > 0:
                    gain_complex = self._apply_gaussian_spread(
                        gain_complex, spread,
                    )

            # Scale to target tap power
            gain_complex = gain_complex * math.sqrt(tap_power)

            # Add Rician LOS component for first tap
            if self.k_factor is not None and self.k_factor > 0 and tap_idx == 0:
                k = self.k_factor
                los_power = k / (k + 1.0)
                scatter_power = 1.0 / (k + 1.0)
                los_phase = torch.empty(1, device=self.device).uniform_(0, 2 * math.pi)
                los_component = math.sqrt(los_power * tap_power) * torch.exp(
                    1j * los_phase
                )
                gain_complex = (
                    gain_complex * math.sqrt(scatter_power / tap_power)
                    + los_component
                )

            gains[tap_idx] = gain_complex

        return gains

    def _generate_doppler_faded(
        self,
        num_samples: int,
        doppler_hz: float,
        gen: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Generate complex fading with classical Jakes Doppler spectrum.

        Uses frequency-domain filtering: generate white noise in freq domain,
        multiply by Jakes PSD shape, then IFFT.

        Args:
            num_samples: Length of output sequence.
            doppler_hz: Maximum Doppler frequency in Hz.
            gen: Optional torch generator.

        Returns:
            Complex tensor of shape (num_samples,).
        """
        nfft = max(num_samples, 256)
        # Pad to next power of 2 for FFT efficiency
        nfft = 1 << (nfft - 1).bit_length()

        # White noise in frequency domain
        re = torch.randn(nfft, device=self.device, generator=gen)
        im = torch.randn(nfft, device=self.device, generator=gen)
        white = torch.complex(re, im)

        # Jakes Doppler spectrum: S(f) = 1 / (pi * fd * sqrt(1 - (f/fd)^2))
        freqs = torch.fft.fftfreq(nfft, d=1.0 / self.sample_rate, device=self.device)
        f_norm = freqs / (doppler_hz + 1e-12)
        f_norm_sq = f_norm ** 2

        # Classical Jakes spectrum (clamp to avoid sqrt of negative)
        jakes_psd = torch.zeros(nfft, device=self.device)
        valid = f_norm_sq < 1.0
        jakes_psd[valid] = 1.0 / (
            math.pi * doppler_hz * torch.sqrt(1.0 - f_norm_sq[valid]).clamp(min=1e-12)
        )

        # Shape and transform back
        shaped = white * jakes_psd.sqrt()
        faded = torch.fft.ifft(shaped)[:num_samples]

        # Normalise to unit power
        power = (faded.abs() ** 2).mean().clamp(min=1e-12)
        faded = faded / power.sqrt() / math.sqrt(2.0)

        return faded

    def _apply_gaussian_spread(
        self,
        signal: torch.Tensor,
        spread_hz: float,
    ) -> torch.Tensor:
        """Apply Gaussian spectral spreading (Watterson model).

        Args:
            signal: Complex tensor of shape (num_samples,).
            spread_hz: Gaussian spread bandwidth in Hz.

        Returns:
            Spread complex tensor of same shape.
        """
        num_samples = signal.shape[0]
        nfft = 1 << (num_samples - 1).bit_length()

        S = torch.fft.fft(signal, n=nfft)

        freqs = torch.fft.fftfreq(nfft, d=1.0 / self.sample_rate, device=self.device)
        gaussian_filter = torch.exp(
            -0.5 * (freqs / (spread_hz + 1e-12)) ** 2
        )

        S_spread = S * gaussian_filter
        result = torch.fft.ifft(S_spread)[:num_samples]

        return result

    def apply_to_signal(
        self,
        iq: torch.Tensor,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply channel model to an IQ signal.

        Args:
            iq: Signal tensor of shape (2, N) or (batch, 2, N).
            seed: Optional random seed.

        Returns:
            Channel-impaired signal with same shape as input.
        """
        if iq.dim() == 3:
            # Batch mode — apply independently to each sample
            return torch.stack(
                [self.apply_to_signal(iq[b], seed=None) for b in range(iq.shape[0])],
                dim=0,
            )

        N = iq.shape[1]
        z = torch.complex(iq[0], iq[1])

        gains = self.generate_tap_gains(N, seed=seed)
        max_delay = int(self.tap_delays.max().item())

        result = torch.zeros(N + max_delay, dtype=torch.cfloat, device=iq.device)
        for tap_idx in range(self.num_taps):
            delay = int(self.tap_delays[tap_idx].item())
            result[delay: delay + N] += gains[tap_idx] * z

        # Truncate to original length
        result = result[:N]

        # Normalise to preserve original power
        orig_power = (z.abs() ** 2).mean().clamp(min=1e-12)
        out_power = (result.abs() ** 2).mean().clamp(min=1e-12)
        result = result * (orig_power / out_power).sqrt()

        return torch.stack([result.real, result.imag], dim=0)


def generate_channel_response(
    profile: Union[str, Dict],
    num_samples: int,
    sample_rate: float = 20e6,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate a complex FIR channel impulse response from a channel profile.

    Convenience function that creates an ITUChannelModel and generates
    time-varying tap gains, then constructs a single FIR filter snapshot
    at the midpoint of the fading process.

    Args:
        profile: Name of a predefined profile or custom profile dict.
        num_samples: Number of output FIR taps (filter length).
        sample_rate: Signal sample rate in Hz.
        seed: Optional random seed.
        device: Torch device.

    Returns:
        Complex tensor of shape (num_samples,) representing the channel
        impulse response as FIR filter coefficients.
    """
    dev = device or torch.device("cpu")
    model = ITUChannelModel(profile, sample_rate=sample_rate, device=dev)

    # Generate tap gains for a short block and take the midpoint snapshot
    block_len = max(64, num_samples)
    gains = model.generate_tap_gains(block_len, seed=seed)
    mid = block_len // 2

    max_delay = int(model.tap_delays.max().item())
    fir_len = max(num_samples, max_delay + 1)
    h = torch.zeros(fir_len, dtype=torch.cfloat, device=dev)

    for tap_idx in range(model.num_taps):
        delay = int(model.tap_delays[tap_idx].item())
        if delay < fir_len:
            h[delay] = gains[tap_idx, mid]

    # Truncate or pad to requested length
    if fir_len > num_samples:
        h = h[:num_samples]
    elif fir_len < num_samples:
        h = torch.nn.functional.pad(h, (0, num_samples - fir_len))

    return h


def list_profiles() -> List[str]:
    """Return a sorted list of all available channel profile names."""
    return sorted(CHANNEL_PROFILES.keys())


def get_profile(name: str) -> Dict:
    """Return a copy of a channel profile by name.

    Args:
        name: Profile name (must be in CHANNEL_PROFILES).

    Returns:
        Dict with profile parameters.

    Raises:
        KeyError: If name is not found.
    """
    if name not in CHANNEL_PROFILES:
        raise KeyError(
            f"Channel profile '{name}' not found. "
            f"Available: {list_profiles()}"
        )
    return CHANNEL_PROFILES[name].copy()
