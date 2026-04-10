"""Comprehensive synthetic RF data generator for drone signal detection.

Creates realistic, interference-rich training data by combining synthetic
modulated signals with channel impairments and multi-source interference.
All operations use PyTorch tensors and are device-agnostic (CPU/CUDA).

Typical usage::

    gen = SyntheticDatasetGenerator(sample_rate=20e6, num_samples=8192)
    sample = gen.generate_sample(difficulty="hard")
    gen.generate_dataset(num_samples=10000, output_dir="data/synthetic")
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .channel_models import CHANNEL_PROFILES, ITUChannelModel

logger = logging.getLogger("rfml.data.synthetic")


# ---------------------------------------------------------------------------
# Channel impairments
# ---------------------------------------------------------------------------


class ChannelImpairments:
    """Composable channel impairment transforms for IQ tensors.

    All methods operate on tensors of shape (2, N) or (batch, 2, N)
    where row 0 is I and row 1 is Q. When a batch dimension is present,
    impairments are applied identically across the batch (same parameters).

    Args:
        sample_rate: Nominal sample rate in Hz.
        device: Torch device for computations.
    """

    def __init__(
        self,
        sample_rate: float = 20e6,
        device: Optional[torch.device] = None,
    ):
        self.sample_rate = sample_rate
        self.device = device or torch.device("cpu")

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _to_complex(iq: torch.Tensor) -> torch.Tensor:
        """Convert (2, N) or (B, 2, N) real IQ to complex."""
        if iq.dim() == 2:
            return torch.complex(iq[0], iq[1])
        return torch.complex(iq[:, 0], iq[:, 1])

    @staticmethod
    def _to_iq(z: torch.Tensor) -> torch.Tensor:
        """Convert complex tensor back to real IQ layout."""
        if z.dim() == 1:
            return torch.stack([z.real, z.imag], dim=0)
        return torch.stack([z.real, z.imag], dim=1)

    # -- 1. AWGN -------------------------------------------------------------

    def awgn(
        self,
        iq: torch.Tensor,
        snr_db: float = 10.0,
    ) -> torch.Tensor:
        """Add white Gaussian noise calibrated to a target SNR.

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            snr_db: Target signal-to-noise ratio in dB. Range: -20 to +30.

        Returns:
            Noisy signal with the same shape.
        """
        signal_power = (iq ** 2).mean()
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = (signal_power / (snr_linear + 1e-12)).clamp(min=1e-20)
        noise = torch.randn_like(iq) * noise_power.sqrt()
        return iq + noise

    # -- 2. Carrier frequency offset -----------------------------------------

    def cfo(
        self,
        iq: torch.Tensor,
        delta_f_hz: Optional[float] = None,
        max_cfo_frac: float = 0.01,
        symbol_rate: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply carrier frequency offset: s'(t) = s(t) * exp(j*2pi*df*t).

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            delta_f_hz: Explicit frequency offset in Hz. If None, randomly
                sampled as +/- max_cfo_frac of the symbol rate.
            max_cfo_frac: Maximum CFO as a fraction of symbol rate (default 1%).
            symbol_rate: Symbol rate in Hz. Defaults to sample_rate / 4.

        Returns:
            Frequency-offset signal.
        """
        batched = iq.dim() == 3
        N = iq.shape[-1]

        if delta_f_hz is None:
            sr = symbol_rate if symbol_rate is not None else self.sample_rate / 4.0
            delta_f_hz = float(
                torch.empty(1).uniform_(-max_cfo_frac * sr, max_cfo_frac * sr).item()
            )

        t = torch.arange(N, dtype=torch.float32, device=iq.device) / self.sample_rate
        phase = 2.0 * math.pi * delta_f_hz * t

        z = self._to_complex(iq)
        rotation = torch.exp(1j * phase.to(z.dtype))
        if batched:
            rotation = rotation.unsqueeze(0)
        z_rot = z * rotation
        return self._to_iq(z_rot)

    # -- 3. Sample rate offset ------------------------------------------------

    def sro(
        self,
        iq: torch.Tensor,
        delta: Optional[float] = None,
        max_delta: float = 0.01,
    ) -> torch.Tensor:
        """Apply sample rate offset by resampling with factor (1 + delta).

        Uses linear interpolation to resample, then truncates or pads to
        maintain the original length.

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            delta: Resampling offset. If None, uniformly sampled in
                [-max_delta, max_delta].
            max_delta: Maximum absolute offset (default 0.01 = 1%).

        Returns:
            Resampled signal with original length.
        """
        if delta is None:
            delta = float(torch.empty(1).uniform_(-max_delta, max_delta).item())

        factor = 1.0 + delta
        N = iq.shape[-1]
        N_new = int(round(N * factor))
        if N_new < 2:
            return iq

        batched = iq.dim() == 3
        if not batched:
            iq = iq.unsqueeze(0)  # (1, 2, N)

        # Use grid_sample for differentiable resampling
        # Reshape to (B, 2, 1, N) for 2D interpolation along width
        iq_4d = iq.unsqueeze(2)  # (B, 2, 1, N)
        resampled = F.interpolate(
            iq_4d, size=(1, N_new), mode="bilinear", align_corners=True,
        ).squeeze(2)  # (B, 2, N_new)

        # Truncate or pad to original length
        if N_new >= N:
            result = resampled[..., :N]
        else:
            result = F.pad(resampled, (0, N - N_new))

        if not batched:
            result = result.squeeze(0)
        return result

    # -- 4. Phase offset ------------------------------------------------------

    def phase_offset(
        self,
        iq: torch.Tensor,
        phi: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply a constant random phase rotation.

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            phi: Phase in radians. If None, uniformly sampled in [0, 2*pi).

        Returns:
            Phase-rotated signal.
        """
        if phi is None:
            phi = float(torch.empty(1).uniform_(0, 2 * math.pi).item())

        z = self._to_complex(iq)
        z_rot = z * torch.tensor(
            math.cos(phi) + 1j * math.sin(phi),
            dtype=z.dtype, device=z.device,
        )
        return self._to_iq(z_rot)

    # -- 5. IQ imbalance ------------------------------------------------------

    def iq_imbalance(
        self,
        iq: torch.Tensor,
        amplitude_db: Optional[float] = None,
        phase_rad: Optional[float] = None,
        dc_offset_i: Optional[float] = None,
        dc_offset_q: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply IQ amplitude/phase imbalance and DC offset.

        Models analog front-end imperfections in the receiver.

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            amplitude_db: Amplitude imbalance in dB (default: random in +/-3).
            phase_rad: Phase imbalance in radians (default: random in +/-pi/180).
            dc_offset_i: DC offset on I channel (default: random in +/-0.1).
            dc_offset_q: DC offset on Q channel (default: random in +/-0.1).

        Returns:
            Imbalanced signal.
        """
        if amplitude_db is None:
            amplitude_db = float(torch.empty(1).uniform_(-3.0, 3.0).item())
        if phase_rad is None:
            phase_rad = float(
                torch.empty(1).uniform_(-math.pi / 180, math.pi / 180).item()
            )
        if dc_offset_i is None:
            dc_offset_i = float(torch.empty(1).uniform_(-0.1, 0.1).item())
        if dc_offset_q is None:
            dc_offset_q = float(torch.empty(1).uniform_(-0.1, 0.1).item())

        amp_linear = 10.0 ** (amplitude_db / 20.0)

        batched = iq.dim() == 3
        if batched:
            i_ch = iq[:, 0] * amp_linear * math.cos(phase_rad) + dc_offset_i
            q_ch = iq[:, 1] / amp_linear * math.cos(phase_rad) + (
                iq[:, 0] * amp_linear * math.sin(phase_rad)
            ) + dc_offset_q
            return torch.stack([i_ch, q_ch], dim=1)
        else:
            i_ch = iq[0] * amp_linear * math.cos(phase_rad) + dc_offset_i
            q_ch = iq[1] / amp_linear * math.cos(phase_rad) + (
                iq[0] * amp_linear * math.sin(phase_rad)
            ) + dc_offset_q
            return torch.stack([i_ch, q_ch], dim=0)

    # -- 6. Rayleigh fading ---------------------------------------------------

    def rayleigh_fading(
        self,
        iq: torch.Tensor,
        num_taps: int = 6,
        power_decay_db: float = 3.0,
    ) -> torch.Tensor:
        """Apply Rayleigh fading via FIR filter with tapered power delay profile.

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            num_taps: Number of FIR taps (2-20).
            power_decay_db: Power decay per tap in dB.

        Returns:
            Faded signal normalised to original power.
        """
        num_taps = max(2, min(num_taps, 20))
        z = self._to_complex(iq)
        orig_power = (z.abs() ** 2).mean().clamp(min=1e-12)

        # Tapered power delay profile
        tap_powers = 10.0 ** (
            -power_decay_db * torch.arange(num_taps, device=iq.device) / 10.0
        )
        tap_powers = tap_powers / tap_powers.sum()

        # Complex Gaussian taps
        re = torch.randn(num_taps, device=iq.device)
        im = torch.randn(num_taps, device=iq.device)
        taps = torch.complex(re, im) * tap_powers.sqrt() / math.sqrt(2.0)

        # Convolve
        if z.dim() == 1:
            z_padded = F.pad(z.unsqueeze(0).unsqueeze(0), (num_taps - 1, 0))
            h = taps.flip(0).unsqueeze(0).unsqueeze(0)
            # Split into real/imag for conv1d (torch conv1d doesn't support complex)
            z_re = z_padded.real
            z_im = z_padded.imag
            h_re = h.real
            h_im = h.imag
            out_re = F.conv1d(z_re, h_re) - F.conv1d(z_im, h_im)
            out_im = F.conv1d(z_re, h_im) + F.conv1d(z_im, h_re)
            result = torch.complex(out_re.squeeze(), out_im.squeeze())
        else:
            B = z.shape[0]
            results = []
            for b in range(B):
                zb = z[b]
                zb_padded = F.pad(zb.unsqueeze(0).unsqueeze(0), (num_taps - 1, 0))
                h = taps.flip(0).unsqueeze(0).unsqueeze(0)
                z_re = zb_padded.real
                z_im = zb_padded.imag
                h_re = h.real
                h_im = h.imag
                out_re = F.conv1d(z_re, h_re) - F.conv1d(z_im, h_im)
                out_im = F.conv1d(z_re, h_im) + F.conv1d(z_im, h_re)
                results.append(torch.complex(out_re.squeeze(), out_im.squeeze()))
            result = torch.stack(results, dim=0)

        # Normalise power
        out_power = (result.abs() ** 2).mean().clamp(min=1e-12)
        result = result * (orig_power / out_power).sqrt()
        return self._to_iq(result)

    # -- 7. Rician fading -----------------------------------------------------

    def rician_fading(
        self,
        iq: torch.Tensor,
        k_factor: float = 3.0,
        num_taps: int = 6,
        power_decay_db: float = 3.0,
    ) -> torch.Tensor:
        """Apply Rician fading: Rayleigh scatter + LOS component.

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            k_factor: Rician K-factor (0-10). K=0 reduces to Rayleigh.
            num_taps: Number of scatter path taps.
            power_decay_db: Power decay per tap in dB.

        Returns:
            Faded signal normalised to original power.
        """
        z = self._to_complex(iq)
        orig_power = (z.abs() ** 2).mean().clamp(min=1e-12)

        los_power = k_factor / (k_factor + 1.0)
        scatter_power = 1.0 / (k_factor + 1.0)

        # LOS component with random phase
        los_phase = float(torch.empty(1).uniform_(0, 2 * math.pi).item())
        los = z * math.sqrt(los_power) * (
            math.cos(los_phase) + 1j * math.sin(los_phase)
        )

        # Scatter via Rayleigh
        scatter_iq = self.rayleigh_fading(iq, num_taps=num_taps, power_decay_db=power_decay_db)
        scatter_z = self._to_complex(scatter_iq)
        scatter_z_power = (scatter_z.abs() ** 2).mean().clamp(min=1e-12)
        scatter_z = scatter_z * math.sqrt(scatter_power * orig_power.item() / scatter_z_power.item())

        result = los + scatter_z

        # Normalise
        out_power = (result.abs() ** 2).mean().clamp(min=1e-12)
        result = result * (orig_power / out_power).sqrt()
        return self._to_iq(result)

    # -- 8. Doppler shift -----------------------------------------------------

    def doppler_shift(
        self,
        iq: torch.Tensor,
        max_doppler_hz: float = 100.0,
    ) -> torch.Tensor:
        """Apply time-varying phase shift from Doppler due to relative motion.

        Models a linearly changing Doppler frequency over the observation
        window, simulating acceleration or curved flight paths.

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            max_doppler_hz: Maximum Doppler frequency in Hz (0-500).

        Returns:
            Doppler-shifted signal.
        """
        N = iq.shape[-1]
        t = torch.arange(N, dtype=torch.float32, device=iq.device) / self.sample_rate

        # Random initial and final Doppler to simulate acceleration
        f0 = float(torch.empty(1).uniform_(-max_doppler_hz, max_doppler_hz).item())
        f1 = float(torch.empty(1).uniform_(-max_doppler_hz, max_doppler_hz).item())

        # Linear chirp phase: integral of linearly varying frequency
        freq_t = f0 + (f1 - f0) * t / (t[-1] + 1e-12)
        phase = 2.0 * math.pi * torch.cumsum(freq_t / self.sample_rate, dim=0)

        z = self._to_complex(iq)
        rotation = torch.exp(1j * phase.to(z.dtype))
        if z.dim() == 2:
            rotation = rotation.unsqueeze(0)
        return self._to_iq(z * rotation)

    # -- 9. Power amplifier nonlinearity (Rapp model) -------------------------

    def pa_nonlinearity(
        self,
        iq: torch.Tensor,
        a_sat: float = 1.0,
        p: float = 3.0,
    ) -> torch.Tensor:
        """Apply Rapp power amplifier model: F(r) = r / (1 + (r/A_sat)^(2p))^(1/(2p)).

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            a_sat: Saturation amplitude.
            p: Smoothness parameter (higher = sharper clipping).

        Returns:
            Nonlinearly distorted signal.
        """
        z = self._to_complex(iq)
        r = z.abs().clamp(min=1e-12)
        theta = z.angle()

        r_norm = r / a_sat
        gain = 1.0 / (1.0 + r_norm ** (2.0 * p)) ** (1.0 / (2.0 * p))
        r_out = r * gain

        z_out = r_out * torch.exp(1j * theta)
        return self._to_iq(z_out)

    # -- 10. Impulsive noise --------------------------------------------------

    def impulsive_noise(
        self,
        iq: torch.Tensor,
        snr_db: float = 10.0,
        exponent: float = 2.0,
    ) -> torch.Tensor:
        """Add impulsive noise: n_imp(t) = sign(n) * |n|^x.

        Produces heavy-tailed noise bursts that model ignition noise,
        switching transients, and other non-Gaussian interference.

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            snr_db: Target SNR for the impulsive noise.
            exponent: Nonlinearity exponent x in {1.5, 2, 3}.

        Returns:
            Signal with impulsive noise added.
        """
        signal_power = (iq ** 2).mean()
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = (signal_power / (snr_linear + 1e-12)).clamp(min=1e-20)

        n_gauss = torch.randn_like(iq)
        n_imp = torch.sign(n_gauss) * torch.abs(n_gauss) ** exponent

        # Scale to target power
        imp_power = (n_imp ** 2).mean().clamp(min=1e-12)
        n_imp = n_imp * (noise_power / imp_power).sqrt()

        return iq + n_imp

    # -- 11. Phase noise (Wiener process) -------------------------------------

    def phase_noise(
        self,
        iq: torch.Tensor,
        variance: float = 1e-4,
    ) -> torch.Tensor:
        """Apply phase noise modelled as a Wiener process (random walk).

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            variance: Variance of the phase increment per sample.
                Higher values produce more severe phase noise.

        Returns:
            Phase-noise-corrupted signal.
        """
        N = iq.shape[-1]
        batched = iq.dim() == 3

        if batched:
            B = iq.shape[0]
            increments = torch.randn(B, N, device=iq.device) * math.sqrt(variance)
            phase_walk = torch.cumsum(increments, dim=1)
            z = self._to_complex(iq)
            rotation = torch.exp(1j * phase_walk.to(z.dtype))
        else:
            increments = torch.randn(N, device=iq.device) * math.sqrt(variance)
            phase_walk = torch.cumsum(increments, dim=0)
            z = self._to_complex(iq)
            rotation = torch.exp(1j * phase_walk.to(z.dtype))

        return self._to_iq(z * rotation)

    # -- 12. Multipath with delay spread (ITU models) -------------------------

    def multipath_itu(
        self,
        iq: torch.Tensor,
        profile: str = "itu_pedestrian_a",
    ) -> torch.Tensor:
        """Apply multipath fading using ITU channel model profiles.

        Args:
            iq: Signal tensor (2, N) or (B, 2, N).
            profile: Name of a channel profile from channel_models.CHANNEL_PROFILES.

        Returns:
            Multipath-faded signal normalised to original power.
        """
        model = ITUChannelModel(
            profile, sample_rate=self.sample_rate, device=iq.device,
        )
        return model.apply_to_signal(iq)


# ---------------------------------------------------------------------------
# Interference mixer
# ---------------------------------------------------------------------------


class InterferenceMixer:
    """Mix multiple signal sources to create interference-rich scenarios.

    Combines a primary signal with one or more interferers at configurable
    signal-to-interference ratios, with optional frequency offsets to model
    adjacent-channel and co-channel interference.

    Args:
        sample_rate: Nominal sample rate in Hz.
        device: Torch device for computations.
    """

    def __init__(
        self,
        sample_rate: float = 20e6,
        device: Optional[torch.device] = None,
    ):
        self.sample_rate = sample_rate
        self.device = device or torch.device("cpu")
        self._impairments = ChannelImpairments(sample_rate=sample_rate, device=self.device)

    def mix_signals(
        self,
        primary: torch.Tensor,
        interferers: List[torch.Tensor],
        sir_db: Union[float, List[float]] = 10.0,
        freq_offsets_hz: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Mix a primary signal with multiple interferers.

        Args:
            primary: Primary signal (2, N).
            interferers: List of interferer signals, each (2, M). Will be
                truncated or zero-padded to match primary length.
            sir_db: Signal-to-interference ratio in dB. Can be a single
                value (applied to all) or a list per interferer.
                Range: -10 to +30.
            freq_offsets_hz: Optional frequency offset for each interferer.
                If None, random offsets are generated.

        Returns:
            Mixed signal (2, N).
        """
        N = primary.shape[-1]
        result = primary.clone()
        primary_power = (primary ** 2).mean().clamp(min=1e-12)

        if isinstance(sir_db, (int, float)):
            sir_list = [float(sir_db)] * len(interferers)
        else:
            sir_list = [float(s) for s in sir_db]

        if freq_offsets_hz is None:
            freq_offsets_hz = [
                float(torch.empty(1).uniform_(
                    -self.sample_rate * 0.1, self.sample_rate * 0.1,
                ).item())
                for _ in interferers
            ]

        for idx, interf in enumerate(interferers):
            # Match length
            if interf.shape[-1] > N:
                interf = interf[..., :N]
            elif interf.shape[-1] < N:
                pad_len = N - interf.shape[-1]
                interf = F.pad(interf, (0, pad_len))

            # Apply frequency offset
            if idx < len(freq_offsets_hz) and freq_offsets_hz[idx] != 0.0:
                interf = self._impairments.cfo(
                    interf, delta_f_hz=freq_offsets_hz[idx],
                )

            # Scale to target SIR
            sir_linear = 10.0 ** (sir_list[min(idx, len(sir_list) - 1)] / 10.0)
            interf_target_power = primary_power / (sir_linear + 1e-12)
            interf_power = (interf ** 2).mean().clamp(min=1e-12)
            scale = (interf_target_power / interf_power).sqrt()
            interf = interf * scale

            result = result + interf

        return result

    def generate_wifi_interference(
        self, num_samples: int,
    ) -> torch.Tensor:
        """Generate synthetic WiFi-like OFDM interference.

        Args:
            num_samples: Number of IQ samples.

        Returns:
            Interference signal (2, num_samples).
        """
        sim = SignalSimulator(sample_rate=self.sample_rate, device=self.device)
        return sim.ofdm(
            num_samples=num_samples,
            num_subcarriers=64,
            cp_length=16,
            modulation="qpsk",
        )

    def generate_bluetooth_interference(
        self, num_samples: int,
    ) -> torch.Tensor:
        """Generate synthetic Bluetooth-like FHSS interference.

        Args:
            num_samples: Number of IQ samples.

        Returns:
            Interference signal (2, num_samples).
        """
        sim = SignalSimulator(sample_rate=self.sample_rate, device=self.device)
        return sim.fhss(
            num_samples=num_samples,
            num_channels=79,
            hop_rate=1600,
            modulation="qpsk",
        )

    def generate_zigbee_interference(
        self, num_samples: int,
    ) -> torch.Tensor:
        """Generate synthetic ZigBee-like DSSS interference.

        Args:
            num_samples: Number of IQ samples.

        Returns:
            Interference signal (2, num_samples).
        """
        sim = SignalSimulator(sample_rate=self.sample_rate, device=self.device)
        return sim.dsss(
            num_samples=num_samples,
            spreading_factor=8,
            modulation="qpsk",
        )

    def generate_narrowband_interference(
        self, num_samples: int,
    ) -> torch.Tensor:
        """Generate narrowband (tone) interference.

        Args:
            num_samples: Number of IQ samples.

        Returns:
            Interference signal (2, num_samples).
        """
        t = torch.arange(num_samples, dtype=torch.float32, device=self.device)
        freq = float(torch.empty(1).uniform_(0.05, 0.45).item()) * self.sample_rate
        phase = float(torch.empty(1).uniform_(0, 2 * math.pi).item())
        angle = 2.0 * math.pi * freq / self.sample_rate * t + phase
        return torch.stack([torch.cos(angle), torch.sin(angle)], dim=0)

    def create_interference_scenario(
        self,
        primary: torch.Tensor,
        num_interferers: int = 1,
        sir_range_db: Tuple[float, float] = (-10.0, 30.0),
        interference_types: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Create a complete interference scenario with random parameters.

        Args:
            primary: Primary signal (2, N).
            num_interferers: Number of interferers to add.
            sir_range_db: Range for random SIR selection per interferer.
            interference_types: List of types to choose from. Options:
                "wifi", "bluetooth", "zigbee", "narrowband", "drone".
                If None, randomly selected.

        Returns:
            Mixed signal (2, N).
        """
        N = primary.shape[-1]
        available_types = interference_types or [
            "wifi", "bluetooth", "zigbee", "narrowband",
        ]

        generators = {
            "wifi": self.generate_wifi_interference,
            "bluetooth": self.generate_bluetooth_interference,
            "zigbee": self.generate_zigbee_interference,
            "narrowband": self.generate_narrowband_interference,
        }

        interferers = []
        sir_values = []

        for _ in range(num_interferers):
            itype = available_types[
                torch.randint(0, len(available_types), (1,)).item()
            ]
            gen_fn = generators.get(itype, self.generate_narrowband_interference)
            interf = gen_fn(N)
            interferers.append(interf)
            sir = float(
                torch.empty(1).uniform_(sir_range_db[0], sir_range_db[1]).item()
            )
            sir_values.append(sir)

        return self.mix_signals(primary, interferers, sir_db=sir_values)


# ---------------------------------------------------------------------------
# Signal simulator
# ---------------------------------------------------------------------------


class SignalSimulator:
    """Generate basic synthetic modulated RF signals.

    All generated signals are returned as real-valued (2, N) tensors
    with I and Q components, properly pulse-shaped.

    Args:
        sample_rate: Sample rate in Hz.
        device: Torch device.
    """

    def __init__(
        self,
        sample_rate: float = 20e6,
        device: Optional[torch.device] = None,
    ):
        self.sample_rate = sample_rate
        self.device = device or torch.device("cpu")

    # -- Pulse shaping -------------------------------------------------------

    def _rrc_filter(
        self,
        num_taps: int = 65,
        samples_per_symbol: int = 4,
        rolloff: float = 0.35,
    ) -> torch.Tensor:
        """Generate a root-raised-cosine (RRC) pulse shaping filter.

        Args:
            num_taps: Filter length (should be odd).
            samples_per_symbol: Oversampling factor.
            rolloff: Roll-off factor (0.15-0.60).

        Returns:
            Real tensor of shape (num_taps,) with unit energy.
        """
        T = samples_per_symbol
        beta = rolloff
        t = torch.arange(num_taps, dtype=torch.float32, device=self.device) - (num_taps - 1) / 2.0

        h = torch.zeros(num_taps, dtype=torch.float32, device=self.device)
        for i in range(num_taps):
            ti = t[i].item()
            if abs(ti) < 1e-8:
                h[i] = (1.0 - beta + 4.0 * beta / math.pi) / T
            elif abs(abs(ti) - T / (4.0 * beta + 1e-12)) < 1e-8:
                h[i] = (beta / (T * math.sqrt(2.0))) * (
                    (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * beta))
                    + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * beta))
                )
            else:
                num = math.sin(math.pi * ti / T * (1 - beta)) + (
                    4.0 * beta * ti / T * math.cos(math.pi * ti / T * (1 + beta))
                )
                den = math.pi * ti / T * (1 - (4.0 * beta * ti / T) ** 2 + 1e-12)
                h[i] = num / (den + 1e-12) / T

        # Normalise to unit energy
        h = h / (h ** 2).sum().sqrt().clamp(min=1e-12)
        return h

    def _apply_pulse_shaping(
        self,
        symbols: torch.Tensor,
        samples_per_symbol: int = 4,
        rolloff: float = 0.35,
        filter_taps: int = 65,
    ) -> torch.Tensor:
        """Upsample and pulse-shape a complex symbol sequence.

        Args:
            symbols: Complex tensor of shape (num_symbols,).
            samples_per_symbol: Oversampling ratio.
            rolloff: RRC roll-off factor.
            filter_taps: RRC filter length.

        Returns:
            Complex tensor of pulse-shaped samples.
        """
        num_symbols = symbols.shape[0]
        # Upsample by inserting zeros
        upsampled = torch.zeros(
            num_symbols * samples_per_symbol,
            dtype=symbols.dtype, device=symbols.device,
        )
        upsampled[::samples_per_symbol] = symbols

        # Generate RRC filter
        rrc = self._rrc_filter(filter_taps, samples_per_symbol, rolloff).to(symbols.device)

        # Convolve (real filter applied to real and imag parts separately)
        rrc_k = rrc.unsqueeze(0).unsqueeze(0)  # (1, 1, taps)
        pad_len = filter_taps // 2

        re = F.conv1d(
            upsampled.real.unsqueeze(0).unsqueeze(0),
            rrc_k, padding=pad_len,
        ).squeeze()
        im = F.conv1d(
            upsampled.imag.unsqueeze(0).unsqueeze(0),
            rrc_k, padding=pad_len,
        ).squeeze()

        return torch.complex(re, im)

    # -- Constellation mappers -----------------------------------------------

    def _modulate_symbols(
        self, bits: torch.Tensor, modulation: str,
    ) -> torch.Tensor:
        """Map bits to complex constellation points.

        Args:
            bits: Tensor of 0/1 values.
            modulation: One of "bpsk", "qpsk", "8psk", "16qam", "64qam".

        Returns:
            Complex symbol tensor.
        """
        mod = modulation.lower()

        if mod == "bpsk":
            symbols = 2.0 * bits.float() - 1.0
            return torch.complex(symbols, torch.zeros_like(symbols))

        elif mod == "qpsk":
            # Group into pairs
            n = (bits.shape[0] // 2) * 2
            b = bits[:n].reshape(-1, 2).float()
            re = 2.0 * b[:, 0] - 1.0
            im = 2.0 * b[:, 1] - 1.0
            symbols = torch.complex(re, im) / math.sqrt(2.0)
            return symbols

        elif mod == "8psk":
            n = (bits.shape[0] // 3) * 3
            b = bits[:n].reshape(-1, 3).float()
            idx = (b[:, 0] * 4 + b[:, 1] * 2 + b[:, 2]).long()
            angles = 2.0 * math.pi * idx.float() / 8.0
            return torch.complex(torch.cos(angles), torch.sin(angles))

        elif mod == "16qam":
            n = (bits.shape[0] // 4) * 4
            b = bits[:n].reshape(-1, 4).float()
            re = (2.0 * (b[:, 0] * 2 + b[:, 1]) - 3.0)
            im = (2.0 * (b[:, 2] * 2 + b[:, 3]) - 3.0)
            norm = math.sqrt(10.0)
            return torch.complex(re / norm, im / norm)

        elif mod == "64qam":
            n = (bits.shape[0] // 6) * 6
            b = bits[:n].reshape(-1, 6).float()
            re = 2.0 * (b[:, 0] * 4 + b[:, 1] * 2 + b[:, 2]) - 7.0
            im = 2.0 * (b[:, 3] * 4 + b[:, 4] * 2 + b[:, 5]) - 7.0
            norm = math.sqrt(42.0)
            return torch.complex(re / norm, im / norm)

        else:
            raise ValueError(f"Unsupported modulation: {modulation}")

    # -- Signal generators ---------------------------------------------------

    def simple_modulation(
        self,
        num_samples: int,
        modulation: str = "qpsk",
        samples_per_symbol: int = 4,
        rolloff: float = 0.35,
    ) -> torch.Tensor:
        """Generate a simple digitally modulated signal.

        Args:
            num_samples: Total number of IQ samples to generate.
            modulation: "bpsk", "qpsk", "8psk", "16qam", or "64qam".
            samples_per_symbol: Oversampling factor.
            rolloff: RRC roll-off (0.15-0.60).

        Returns:
            Signal tensor (2, num_samples).
        """
        bits_per_symbol = {
            "bpsk": 1, "qpsk": 2, "8psk": 3, "16qam": 4, "64qam": 6,
        }
        bps = bits_per_symbol.get(modulation.lower(), 2)
        num_symbols = (num_samples // samples_per_symbol) + 64  # extra for filter
        num_bits = num_symbols * bps

        bits = torch.randint(0, 2, (num_bits,), device=self.device)
        symbols = self._modulate_symbols(bits, modulation)

        shaped = self._apply_pulse_shaping(symbols, samples_per_symbol, rolloff)

        # Truncate to requested length
        if shaped.shape[0] > num_samples:
            shaped = shaped[:num_samples]
        elif shaped.shape[0] < num_samples:
            shaped = F.pad(shaped, (0, num_samples - shaped.shape[0]))

        # Normalise to unit power
        power = (shaped.abs() ** 2).mean().clamp(min=1e-12)
        shaped = shaped / power.sqrt()

        return torch.stack([shaped.real, shaped.imag], dim=0)

    def ofdm(
        self,
        num_samples: int,
        num_subcarriers: int = 64,
        cp_length: int = 16,
        modulation: str = "qpsk",
    ) -> torch.Tensor:
        """Generate an OFDM signal.

        Args:
            num_samples: Total number of IQ samples.
            num_subcarriers: Number of OFDM subcarriers (64-2048).
            cp_length: Cyclic prefix length in samples.
            modulation: Per-subcarrier modulation type.

        Returns:
            Signal tensor (2, num_samples).
        """
        symbol_len = num_subcarriers + cp_length
        num_ofdm_symbols = (num_samples // symbol_len) + 2

        output_parts: List[torch.Tensor] = []

        for _ in range(num_ofdm_symbols):
            # Random data on each subcarrier
            bits_per_sym = {"bpsk": 1, "qpsk": 2, "8psk": 3, "16qam": 4, "64qam": 6}
            bps = bits_per_sym.get(modulation.lower(), 2)
            bits = torch.randint(0, 2, (num_subcarriers * bps,), device=self.device)
            freq_domain = self._modulate_symbols(bits, modulation)

            # Pad/truncate to match subcarrier count
            if freq_domain.shape[0] > num_subcarriers:
                freq_domain = freq_domain[:num_subcarriers]
            elif freq_domain.shape[0] < num_subcarriers:
                freq_domain = F.pad(
                    freq_domain, (0, num_subcarriers - freq_domain.shape[0]),
                )

            # IFFT
            time_domain = torch.fft.ifft(freq_domain, n=num_subcarriers)

            # Add cyclic prefix
            cp = time_domain[-cp_length:]
            ofdm_symbol = torch.cat([cp, time_domain])
            output_parts.append(ofdm_symbol)

        signal = torch.cat(output_parts)[:num_samples]

        # Normalise
        power = (signal.abs() ** 2).mean().clamp(min=1e-12)
        signal = signal / power.sqrt()

        return torch.stack([signal.real, signal.imag], dim=0)

    def fhss(
        self,
        num_samples: int,
        num_channels: int = 79,
        hop_rate: float = 1600.0,
        modulation: str = "qpsk",
        dwell_time: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate a frequency-hopping spread spectrum (FHSS) signal.

        Args:
            num_samples: Total number of IQ samples.
            num_channels: Number of hopping channels.
            hop_rate: Hops per second.
            modulation: Modulation used on each hop.
            dwell_time: Time per hop in seconds. If None, computed from hop_rate.

        Returns:
            Signal tensor (2, num_samples).
        """
        if dwell_time is None:
            dwell_time = 1.0 / hop_rate
        samples_per_hop = max(1, int(dwell_time * self.sample_rate))
        num_hops = (num_samples // samples_per_hop) + 2

        # Generate hopping sequence (pseudo-random)
        hop_sequence = torch.randint(0, num_channels, (num_hops,), device=self.device)

        # Channel spacing
        channel_bw = self.sample_rate / (num_channels + 1)

        output_parts: List[torch.Tensor] = []

        for hop_idx in range(num_hops):
            channel = hop_sequence[hop_idx].item()
            freq_offset = (channel - num_channels / 2.0) * channel_bw

            # Generate modulated burst for this hop
            hop_signal = self.simple_modulation(
                num_samples=samples_per_hop,
                modulation=modulation,
                samples_per_symbol=4,
                rolloff=0.35,
            )

            # Apply frequency offset for this channel
            t = torch.arange(
                samples_per_hop, dtype=torch.float32, device=self.device,
            ) / self.sample_rate
            phase = 2.0 * math.pi * freq_offset * t
            z = torch.complex(hop_signal[0], hop_signal[1])
            z = z * torch.exp(1j * phase.to(z.dtype))
            hop_signal = torch.stack([z.real, z.imag], dim=0)

            output_parts.append(hop_signal)

        signal = torch.cat(output_parts, dim=-1)[..., :num_samples]

        # Normalise
        power = (signal ** 2).mean().clamp(min=1e-12)
        signal = signal / power.sqrt()

        return signal

    def dsss(
        self,
        num_samples: int,
        spreading_factor: int = 8,
        modulation: str = "bpsk",
        chip_rate: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate a direct-sequence spread spectrum (DSSS) signal.

        Args:
            num_samples: Total number of IQ samples.
            spreading_factor: PN sequence spreading factor.
            modulation: Data modulation type.
            chip_rate: Chip rate in Hz. If None, uses sample_rate / 4.

        Returns:
            Signal tensor (2, num_samples).
        """
        if chip_rate is None:
            chip_rate = self.sample_rate / 4.0
        samples_per_chip = max(1, int(self.sample_rate / chip_rate))

        num_chips = (num_samples // samples_per_chip) + spreading_factor
        num_data_symbols = (num_chips // spreading_factor) + 1

        # Generate random data bits
        bits_per_sym = {"bpsk": 1, "qpsk": 2, "8psk": 3, "16qam": 4, "64qam": 6}
        bps = bits_per_sym.get(modulation.lower(), 1)
        data_bits = torch.randint(
            0, 2, (num_data_symbols * bps,), device=self.device,
        )
        data_symbols = self._modulate_symbols(data_bits, modulation)

        # Generate PN spreading sequence (Gold-like)
        pn = 2.0 * torch.randint(
            0, 2, (spreading_factor,), dtype=torch.float32, device=self.device,
        ) - 1.0

        # Spread each symbol
        chips: List[torch.Tensor] = []
        for sym_idx in range(min(num_data_symbols, data_symbols.shape[0])):
            spread = data_symbols[sym_idx] * pn
            chips.append(spread)

        chip_seq = torch.cat(chips)[:num_chips]

        # Upsample chips to sample rate
        upsampled = chip_seq.repeat_interleave(samples_per_chip)[:num_samples]

        # Pad if needed
        if upsampled.shape[0] < num_samples:
            upsampled = F.pad(upsampled, (0, num_samples - upsampled.shape[0]))

        # Normalise
        power = (upsampled.abs() ** 2).mean().clamp(min=1e-12)
        upsampled = upsampled / power.sqrt()

        return torch.stack([upsampled.real, upsampled.imag], dim=0)


# ---------------------------------------------------------------------------
# Channel profiles for difficulty levels
# ---------------------------------------------------------------------------

_SCENARIO_PROFILES = {
    "indoor": {
        "channel_profile": "drone_indoor",
        "max_doppler_hz": 10.0,
        "fading_type": "rician",
        "k_factor": 6.0,
    },
    "urban": {
        "channel_profile": "drone_urban",
        "max_doppler_hz": 80.0,
        "fading_type": "rician",
        "k_factor": 3.0,
    },
    "suburban": {
        "channel_profile": "drone_suburban",
        "max_doppler_hz": 150.0,
        "fading_type": "rician",
        "k_factor": 1.0,
    },
    "vehicular": {
        "channel_profile": "drone_vehicular",
        "max_doppler_hz": 500.0,
        "fading_type": "rayleigh",
        "k_factor": 0.5,
    },
}


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------


class SyntheticDatasetGenerator:
    """Main orchestrator for synthetic RF dataset generation.

    Combines signal simulation, channel impairments, and interference
    mixing to produce labelled training data at configurable difficulty
    levels. Output is saved as HDF5 for efficient loading.

    Args:
        sample_rate: Sample rate in Hz.
        num_samples: Number of IQ samples per generated segment.
        device: Torch device.
        signal_types: List of signal types to generate. If None, all
            available types are used.
    """

    # Signal type -> label mappings
    SIGNAL_TYPES = {
        "noise": {"binary": 0, "type_id": 0, "full_id": 0},
        "drone_ofdm": {"binary": 1, "type_id": 1, "full_id": 1},
        "drone_fhss": {"binary": 1, "type_id": 1, "full_id": 2},
        "drone_dsss": {"binary": 1, "type_id": 1, "full_id": 3},
        "wifi": {"binary": 0, "type_id": 2, "full_id": 4},
        "bluetooth": {"binary": 0, "type_id": 3, "full_id": 5},
        "zigbee": {"binary": 0, "type_id": 4, "full_id": 6},
        "bpsk_generic": {"binary": 0, "type_id": 5, "full_id": 7},
        "qpsk_generic": {"binary": 0, "type_id": 5, "full_id": 8},
        "qam16_generic": {"binary": 0, "type_id": 5, "full_id": 9},
    }

    def __init__(
        self,
        sample_rate: float = 20e6,
        num_samples: int = 8192,
        device: Optional[torch.device] = None,
        signal_types: Optional[List[str]] = None,
    ):
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.device = device or torch.device("cpu")
        self.signal_types = signal_types or list(self.SIGNAL_TYPES.keys())

        self.simulator = SignalSimulator(
            sample_rate=sample_rate, device=self.device,
        )
        self.impairments = ChannelImpairments(
            sample_rate=sample_rate, device=self.device,
        )
        self.mixer = InterferenceMixer(
            sample_rate=sample_rate, device=self.device,
        )

        logger.info(
            "SyntheticDatasetGenerator: fs=%.1e, N=%d, types=%s",
            sample_rate, num_samples, self.signal_types,
        )

    def generate_clean_sample(
        self,
        signal_type: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Generate a clean (no impairments) IQ signal.

        Args:
            signal_type: Type of signal to generate (see SIGNAL_TYPES).
            params: Optional override parameters for the signal generator.

        Returns:
            Clean signal tensor (2, num_samples).
        """
        p = params or {}
        N = p.get("num_samples", self.num_samples)

        if signal_type == "noise":
            # Pure noise floor
            return torch.randn(2, N, device=self.device) * 0.01

        elif signal_type == "drone_ofdm":
            return self.simulator.ofdm(
                num_samples=N,
                num_subcarriers=p.get("num_subcarriers", 128),
                cp_length=p.get("cp_length", 32),
                modulation=p.get("modulation", "qpsk"),
            )

        elif signal_type == "drone_fhss":
            return self.simulator.fhss(
                num_samples=N,
                num_channels=p.get("num_channels", 40),
                hop_rate=p.get("hop_rate", 200),
                modulation=p.get("modulation", "qpsk"),
            )

        elif signal_type == "drone_dsss":
            return self.simulator.dsss(
                num_samples=N,
                spreading_factor=p.get("spreading_factor", 16),
                modulation=p.get("modulation", "bpsk"),
            )

        elif signal_type == "wifi":
            return self.simulator.ofdm(
                num_samples=N,
                num_subcarriers=p.get("num_subcarriers", 64),
                cp_length=p.get("cp_length", 16),
                modulation=p.get("modulation", "64qam"),
            )

        elif signal_type == "bluetooth":
            return self.simulator.fhss(
                num_samples=N,
                num_channels=79,
                hop_rate=1600,
                modulation=p.get("modulation", "qpsk"),
            )

        elif signal_type == "zigbee":
            return self.simulator.dsss(
                num_samples=N,
                spreading_factor=8,
                modulation=p.get("modulation", "qpsk"),
            )

        elif signal_type == "bpsk_generic":
            return self.simulator.simple_modulation(N, "bpsk")

        elif signal_type == "qpsk_generic":
            return self.simulator.simple_modulation(N, "qpsk")

        elif signal_type == "qam16_generic":
            return self.simulator.simple_modulation(N, "16qam")

        else:
            logger.warning("Unknown signal type '%s', generating QPSK", signal_type)
            return self.simulator.simple_modulation(N, "qpsk")

    def apply_channel(
        self,
        iq: torch.Tensor,
        channel_profile: Optional[str] = None,
        snr_db: Optional[float] = None,
        add_fading: bool = True,
        add_cfo: bool = False,
        add_phase_noise: bool = False,
        add_iq_imbalance: bool = False,
        add_pa_nonlinearity: bool = False,
        add_doppler: bool = False,
        doppler_hz: float = 50.0,
        phase_noise_var: float = 1e-4,
    ) -> torch.Tensor:
        """Apply channel impairments to a signal.

        Args:
            iq: Clean signal (2, N).
            channel_profile: Name of an ITU channel profile for multipath.
            snr_db: Target SNR in dB. If None, no AWGN is added.
            add_fading: Whether to apply multipath fading.
            add_cfo: Whether to add carrier frequency offset.
            add_phase_noise: Whether to add oscillator phase noise.
            add_iq_imbalance: Whether to add IQ imbalance.
            add_pa_nonlinearity: Whether to apply PA nonlinearity.
            add_doppler: Whether to apply Doppler shift.
            doppler_hz: Maximum Doppler frequency.
            phase_noise_var: Phase noise variance.

        Returns:
            Impaired signal (2, N).
        """
        result = iq.clone()

        # Multipath fading
        if add_fading and channel_profile is not None:
            result = self.impairments.multipath_itu(result, profile=channel_profile)

        # Doppler
        if add_doppler:
            result = self.impairments.doppler_shift(result, max_doppler_hz=doppler_hz)

        # CFO
        if add_cfo:
            result = self.impairments.cfo(result)

        # Phase noise
        if add_phase_noise:
            result = self.impairments.phase_noise(result, variance=phase_noise_var)

        # IQ imbalance
        if add_iq_imbalance:
            result = self.impairments.iq_imbalance(result)

        # PA nonlinearity
        if add_pa_nonlinearity:
            result = self.impairments.pa_nonlinearity(result)

        # AWGN (applied last)
        if snr_db is not None:
            result = self.impairments.awgn(result, snr_db=snr_db)

        return result

    def mix_with_interference(
        self,
        iq: torch.Tensor,
        interferers: List[torch.Tensor],
        sir_db: Union[float, List[float]] = 10.0,
    ) -> torch.Tensor:
        """Mix a primary signal with interference sources.

        Args:
            iq: Primary signal (2, N).
            interferers: List of interferer signals.
            sir_db: Signal-to-interference ratio(s) in dB.

        Returns:
            Mixed signal (2, N).
        """
        return self.mixer.mix_signals(iq, interferers, sir_db=sir_db)

    def generate_sample(
        self,
        difficulty: str = "medium",
        signal_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a complete labelled sample at the specified difficulty.

        Difficulty levels:
            - "easy": High SNR (15-30 dB), no fading, no interference.
            - "medium": Moderate SNR (0-15 dB), light fading, 1 interferer.
            - "hard": Low SNR (-10 to 5 dB), heavy fading, 2-3 interferers,
              impulsive noise, PA nonlinearity.

        Args:
            difficulty: One of "easy", "medium", "hard".
            signal_type: Specific signal type. If None, randomly selected.

        Returns:
            Dict with keys:
                - "iq": Signal tensor (2, num_samples)
                - "label_binary": 0 or 1 (drone vs non-drone)
                - "label_type": Type class index
                - "label_full": Full class index
                - "snr_db": Applied SNR
                - "signal_type": Name of signal type
                - "difficulty": Difficulty level
        """
        if signal_type is None:
            signal_type = self.signal_types[
                torch.randint(0, len(self.signal_types), (1,)).item()
            ]

        labels = self.SIGNAL_TYPES.get(signal_type, {"binary": 0, "type_id": 0, "full_id": 0})

        # Generate clean signal
        iq = self.generate_clean_sample(signal_type)

        # Select scenario based on difficulty
        scenario_names = list(_SCENARIO_PROFILES.keys())

        if difficulty == "easy":
            snr_db = float(torch.empty(1).uniform_(15.0, 30.0).item())
            iq = self.apply_channel(
                iq, snr_db=snr_db,
                add_fading=False, add_cfo=False,
                add_phase_noise=False, add_iq_imbalance=False,
            )

        elif difficulty == "medium":
            snr_db = float(torch.empty(1).uniform_(0.0, 15.0).item())
            scenario = _SCENARIO_PROFILES[
                scenario_names[torch.randint(0, 2, (1,)).item()]  # indoor or urban
            ]
            iq = self.apply_channel(
                iq,
                channel_profile=scenario["channel_profile"],
                snr_db=snr_db,
                add_fading=True,
                add_cfo=True,
                add_phase_noise=torch.rand(1).item() > 0.5,
                add_iq_imbalance=torch.rand(1).item() > 0.5,
                add_doppler=torch.rand(1).item() > 0.7,
                doppler_hz=scenario["max_doppler_hz"],
            )

            # Add 1 interferer
            if torch.rand(1).item() > 0.3:
                iq = self.mixer.create_interference_scenario(
                    iq, num_interferers=1,
                    sir_range_db=(5.0, 20.0),
                )

        elif difficulty == "hard":
            snr_db = float(torch.empty(1).uniform_(-10.0, 5.0).item())
            scenario = _SCENARIO_PROFILES[
                scenario_names[torch.randint(2, 4, (1,)).item()]  # suburban or vehicular
            ]
            iq = self.apply_channel(
                iq,
                channel_profile=scenario["channel_profile"],
                snr_db=snr_db,
                add_fading=True,
                add_cfo=True,
                add_phase_noise=True,
                add_iq_imbalance=True,
                add_pa_nonlinearity=torch.rand(1).item() > 0.5,
                add_doppler=True,
                doppler_hz=scenario["max_doppler_hz"],
                phase_noise_var=float(torch.empty(1).uniform_(1e-4, 1e-3).item()),
            )

            # Add impulsive noise
            if torch.rand(1).item() > 0.4:
                exponent = [1.5, 2.0, 3.0][torch.randint(0, 3, (1,)).item()]
                imp_snr = float(torch.empty(1).uniform_(5.0, 15.0).item())
                iq = self.impairments.impulsive_noise(
                    iq, snr_db=imp_snr, exponent=exponent,
                )

            # Add 2-3 interferers
            num_interf = torch.randint(2, 4, (1,)).item()
            iq = self.mixer.create_interference_scenario(
                iq, num_interferers=num_interf,
                sir_range_db=(-10.0, 10.0),
            )

        else:
            raise ValueError(
                f"Unknown difficulty '{difficulty}'. Use 'easy', 'medium', or 'hard'."
            )

        return {
            "iq": iq,
            "label_binary": labels["binary"],
            "label_type": labels["type_id"],
            "label_full": labels["full_id"],
            "snr_db": snr_db,
            "signal_type": signal_type,
            "difficulty": difficulty,
        }

    def generate_dataset(
        self,
        num_samples: int,
        output_dir: str,
        difficulty_mix: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ) -> Path:
        """Generate a full synthetic dataset and save as HDF5.

        Args:
            num_samples: Total number of samples to generate.
            output_dir: Directory to save the HDF5 file.
            difficulty_mix: Dict mapping difficulty -> fraction (must sum to 1).
                Default: {"easy": 0.3, "medium": 0.4, "hard": 0.3}.
            seed: Random seed for reproducibility.

        Returns:
            Path to the generated HDF5 file.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for dataset generation. "
                "Install with: pip install h5py"
            )

        if seed is not None:
            torch.manual_seed(seed)

        if difficulty_mix is None:
            difficulty_mix = {"easy": 0.3, "medium": 0.4, "hard": 0.3}

        # Validate mix sums to ~1
        total = sum(difficulty_mix.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "difficulty_mix sums to %.3f, not 1.0. Normalising.", total,
            )
            difficulty_mix = {k: v / total for k, v in difficulty_mix.items()}

        # Compute sample counts per difficulty
        counts: Dict[str, int] = {}
        assigned = 0
        for diff, frac in difficulty_mix.items():
            n = int(round(num_samples * frac))
            counts[diff] = n
            assigned += n
        # Assign remainder to the first difficulty
        first_diff = next(iter(counts))
        counts[first_diff] += num_samples - assigned

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        h5_path = output_path / "synthetic_dataset.h5"

        logger.info(
            "Generating %d samples -> %s (mix: %s)",
            num_samples, h5_path, counts,
        )

        with h5py.File(h5_path, "w") as f:
            # Create datasets
            iq_ds = f.create_dataset(
                "iq", shape=(num_samples, 2, self.num_samples),
                dtype="float32", chunks=(min(64, num_samples), 2, self.num_samples),
            )
            labels_ds = f.create_dataset(
                "labels", shape=(num_samples, 4), dtype="float32",
            )
            # Store metadata as string dataset
            signal_types_ds = f.create_dataset(
                "signal_types", shape=(num_samples,),
                dtype=h5py.special_dtype(vlen=str),
            )
            difficulties_ds = f.create_dataset(
                "difficulties", shape=(num_samples,),
                dtype=h5py.special_dtype(vlen=str),
            )

            idx = 0
            for diff, count in counts.items():
                for i in range(count):
                    sample = self.generate_sample(difficulty=diff)

                    iq_np = sample["iq"].cpu().numpy()
                    iq_ds[idx] = iq_np
                    labels_ds[idx] = [
                        sample["label_binary"],
                        sample["label_type"],
                        sample["label_full"],
                        sample["snr_db"],
                    ]
                    signal_types_ds[idx] = sample["signal_type"]
                    difficulties_ds[idx] = sample["difficulty"]

                    idx += 1

                    if (idx % 500) == 0:
                        logger.info("  Generated %d / %d samples", idx, num_samples)

            # Store generation metadata
            f.attrs["sample_rate"] = self.sample_rate
            f.attrs["num_samples_per_segment"] = self.num_samples
            f.attrs["total_samples"] = num_samples
            f.attrs["difficulty_mix"] = str(difficulty_mix)

        logger.info("Dataset saved to %s (%d samples)", h5_path, num_samples)
        return h5_path
