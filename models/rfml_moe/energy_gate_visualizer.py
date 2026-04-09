"""Energy Gate Spectrogram Visualizer and Signal Chunk Labeler.

Provides tools to:
1. Visualize energy gate detection decisions overlaid on spectrograms
2. Auto-label signal chunks in long recordings using energy-based detection
3. Generate annotation tracks compatible with dataset labeling pipelines
4. Export labeled segments for training data curation

The energy gate runs on sliding windows across a long IQ recording,
producing a temporal binary mask (signal present / noise only) that
maps directly onto the spectrogram's time axis.

Usage:
    python -m evaluation.energy_gate_visualizer \\
        --input data/raw/rfuav/sample.iq \\
        --output results/energy_gate_viz/ \\
        --segment-length 32768 \\
        --hop-length 8192
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("rfml.evaluation")


@dataclass
class DetectionEvent:
    """A detected signal region in a recording."""

    start_sample: int
    end_sample: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    mean_snr_db: float
    peak_snr_db: float
    energy_profile: np.ndarray  # energy per window within this event
    label: str = "signal"  # "signal", "drone", "noise", or custom


@dataclass
class AnnotationTrack:
    """Full annotation of a recording with detection events."""

    recording_path: str
    sample_rate: float
    total_samples: int
    total_duration_s: float
    segment_length: int
    hop_length: int
    num_windows: int
    detection_mask: np.ndarray  # [num_windows] bool
    energy_db: np.ndarray       # [num_windows] float, energy in dB
    snr_db: np.ndarray          # [num_windows] float, estimated SNR
    threshold_db: float         # detection threshold in dB
    events: list = field(default_factory=list)
    p_fa: float = 0.01
    noise_floor_db: float = 0.0


class EnergyGateVisualizer:
    """Visualize and label signal chunks using the energy gate algorithm.

    Runs the Neyman-Pearson energy detector on sliding windows across
    a long IQ recording, producing:
    - A spectrogram with energy gate overlay
    - A temporal annotation track (binary mask)
    - Labeled signal/noise segments for dataset curation

    Args:
        segment_length: Window size in IQ samples (default 32768).
        hop_length: Step between windows (default 8192 = 75% overlap).
        p_fa: False alarm probability for detection threshold.
        calibration_seconds: Duration of initial recording to use for
            noise floor calibration (default 0.5s).
        sample_rate: Sample rate of the recording (default 100e6 = 100 MSps).
    """

    def __init__(
        self,
        segment_length: int = 32768,
        hop_length: int = 8192,
        p_fa: float = 0.01,
        calibration_seconds: float = 0.5,
        sample_rate: float = 100e6,
    ):
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.p_fa = p_fa
        self.calibration_seconds = calibration_seconds
        self.sample_rate = sample_rate

    @staticmethod
    def _q_inv(p: float) -> float:
        """Inverse Q-function via Abramowitz & Stegun 26.2.17."""
        t = math.sqrt(-2.0 * math.log(p))
        num = 2.515517 + 0.802853 * t + 0.010328 * t * t
        den = 1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
        return t - num / den

    def _compute_window_energy(self, iq: np.ndarray) -> float:
        """Compute mean energy (power) of an IQ window.

        Args:
            iq: Complex IQ array of shape (N,) or real interleaved (2, N).

        Returns:
            Mean power as float.
        """
        if iq.ndim == 2 and iq.shape[0] == 2:
            return float(np.mean(iq[0] ** 2 + iq[1] ** 2))
        elif np.iscomplexobj(iq):
            return float(np.mean(np.abs(iq) ** 2))
        else:
            return float(np.mean(iq ** 2))

    def analyze_recording(
        self,
        iq_data: np.ndarray,
        sample_rate: Optional[float] = None,
    ) -> AnnotationTrack:
        """Analyze a full IQ recording with sliding-window energy detection.

        Args:
            iq_data: IQ recording as numpy array. Supported shapes:
                - (N,) complex64/128
                - (2, N) float32/64 (I and Q rows)
                - (N, 2) float32/64 (I and Q columns, auto-transposed)
            sample_rate: Override sample rate (Hz). Uses constructor default if None.

        Returns:
            AnnotationTrack with detection mask, energy profile, SNR estimates,
            and labeled signal events.
        """
        sr = sample_rate or self.sample_rate

        # Normalize input shape to (2, N)
        if np.iscomplexobj(iq_data):
            iq_2d = np.stack([iq_data.real, iq_data.imag], axis=0).astype(np.float32)
        elif iq_data.ndim == 2 and iq_data.shape[1] == 2:
            iq_2d = iq_data.T.astype(np.float32)
        elif iq_data.ndim == 2 and iq_data.shape[0] == 2:
            iq_2d = iq_data.astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported IQ shape {iq_data.shape}. "
                "Expected (N,) complex, (2, N), or (N, 2)."
            )

        total_samples = iq_2d.shape[1]
        total_duration = total_samples / sr

        # Compute number of windows
        num_windows = max(1, (total_samples - self.segment_length) // self.hop_length + 1)

        # Phase 1: Compute energy for all windows
        energies = np.zeros(num_windows, dtype=np.float64)
        for i in range(num_windows):
            start = i * self.hop_length
            end = start + self.segment_length
            if end > total_samples:
                break
            window = iq_2d[:, start:end]
            energies[i] = self._compute_window_energy(window)

        # Phase 2: Calibrate noise floor from initial portion
        cal_samples = int(self.calibration_seconds * sr)
        cal_windows = max(1, min(
            cal_samples // self.hop_length,
            num_windows // 4,  # use at most 25% for calibration
        ))

        # Use the lowest-energy windows for calibration (noise floor)
        sorted_energies = np.sort(energies)
        noise_count = max(1, int(num_windows * 0.10))  # bottom 10%
        noise_energies = sorted_energies[:noise_count]

        mu_noise = float(np.mean(noise_energies))
        sigma_noise = float(np.std(noise_energies)) + 1e-20

        # Phase 3: Compute threshold
        q_inv = self._q_inv(self.p_fa)
        threshold = mu_noise + q_inv * sigma_noise

        # Phase 4: Binary detection mask
        detection_mask = energies > threshold

        # Phase 5: Energy in dB and SNR estimates
        energy_db = 10.0 * np.log10(np.maximum(energies, 1e-20))
        threshold_db = 10.0 * np.log10(max(threshold, 1e-20))
        noise_floor_db = 10.0 * np.log10(max(mu_noise, 1e-20))

        snr_db = np.zeros(num_windows, dtype=np.float64)
        for i in range(num_windows):
            snr_linear = max(energies[i] / mu_noise - 1.0, 1e-10)
            snr_db[i] = 10.0 * math.log10(snr_linear)

        # Phase 6: Group detections into events (merge adjacent windows)
        events = self._extract_events(
            detection_mask, snr_db, energies, sr
        )

        logger.info(
            "Analyzed %d windows: %d detections (%.1f%%), %d events, "
            "noise_floor=%.1f dB, threshold=%.1f dB",
            num_windows,
            int(detection_mask.sum()),
            100.0 * detection_mask.mean(),
            len(events),
            noise_floor_db,
            threshold_db,
        )

        return AnnotationTrack(
            recording_path="",
            sample_rate=sr,
            total_samples=total_samples,
            total_duration_s=total_duration,
            segment_length=self.segment_length,
            hop_length=self.hop_length,
            num_windows=num_windows,
            detection_mask=detection_mask,
            energy_db=energy_db,
            snr_db=snr_db,
            threshold_db=threshold_db,
            events=events,
            p_fa=self.p_fa,
            noise_floor_db=noise_floor_db,
        )

    def _extract_events(
        self,
        mask: np.ndarray,
        snr_db: np.ndarray,
        energies: np.ndarray,
        sample_rate: float,
    ) -> list[DetectionEvent]:
        """Group consecutive detections into signal events.

        Args:
            mask: Binary detection mask [num_windows].
            snr_db: SNR estimates per window [num_windows].
            energies: Raw energy values per window [num_windows].
            sample_rate: Sample rate for time conversion.

        Returns:
            List of DetectionEvent objects.
        """
        events = []
        in_event = False
        event_start = 0

        for i in range(len(mask)):
            if mask[i] and not in_event:
                # Start of new event
                event_start = i
                in_event = True
            elif not mask[i] and in_event:
                # End of event
                events.append(self._make_event(
                    event_start, i, snr_db, energies, sample_rate
                ))
                in_event = False

        # Handle event that extends to end of recording
        if in_event:
            events.append(self._make_event(
                event_start, len(mask), snr_db, energies, sample_rate
            ))

        return events

    def _make_event(
        self,
        win_start: int,
        win_end: int,
        snr_db: np.ndarray,
        energies: np.ndarray,
        sample_rate: float,
    ) -> DetectionEvent:
        """Create a DetectionEvent from window indices."""
        start_sample = win_start * self.hop_length
        end_sample = min(
            win_end * self.hop_length + self.segment_length,
            int(sample_rate * 1e9),  # reasonable upper bound
        )
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate

        event_snr = snr_db[win_start:win_end]
        event_energy = energies[win_start:win_end]

        return DetectionEvent(
            start_sample=start_sample,
            end_sample=end_sample,
            start_time_s=start_time,
            end_time_s=end_time,
            duration_s=end_time - start_time,
            mean_snr_db=float(np.mean(event_snr)),
            peak_snr_db=float(np.max(event_snr)),
            energy_profile=event_energy,
        )

    def plot_spectrogram_with_gate(
        self,
        iq_data: np.ndarray,
        annotation: AnnotationTrack,
        output_path: Optional[str] = None,
        title: str = "Energy Gate Detection on Spectrogram",
        figsize: tuple = (16, 10),
        fft_size: int = 1024,
        cmap: str = "viridis",
    ):
        """Generate a spectrogram with energy gate overlay visualization.

        Creates a 3-panel figure:
        - Top: Spectrogram with detection regions highlighted
        - Middle: Energy profile with threshold line
        - Bottom: SNR profile with detection mask

        Args:
            iq_data: IQ recording (complex or [2, N]).
            annotation: AnnotationTrack from analyze_recording().
            output_path: Save path (PNG/PDF). If None, calls plt.show().
            title: Figure title.
            figsize: Figure size in inches.
            fft_size: FFT size for spectrogram computation.
            cmap: Colormap for spectrogram.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        sr = annotation.sample_rate

        # Prepare complex signal for spectrogram
        if np.iscomplexobj(iq_data):
            z = iq_data
        elif iq_data.ndim == 2 and iq_data.shape[0] == 2:
            z = iq_data[0] + 1j * iq_data[1]
        elif iq_data.ndim == 2 and iq_data.shape[1] == 2:
            z = iq_data[:, 0] + 1j * iq_data[:, 1]
        else:
            z = iq_data.astype(np.complex64)

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1, 1]})

        # === Panel 1: Spectrogram with detection overlay ===
        ax_spec = axes[0]

        # Compute spectrogram
        from scipy.signal import spectrogram as scipy_spectrogram
        hop_spec = fft_size // 4
        freqs, times, Sxx = scipy_spectrogram(
            z, fs=sr, nperseg=fft_size, noverlap=fft_size - hop_spec,
            return_onesided=False, mode="psd",
        )

        # Shift zero-frequency to center
        Sxx = np.fft.fftshift(Sxx, axes=0)
        freqs = np.fft.fftshift(freqs)

        # Plot spectrogram in dB
        Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-20))
        im = ax_spec.pcolormesh(
            times, freqs / 1e6, Sxx_db,
            shading="gouraud", cmap=cmap,
        )
        fig.colorbar(im, ax=ax_spec, label="PSD (dB)")

        # Overlay detection regions as semi-transparent red bands
        for event in annotation.events:
            ax_spec.axvspan(
                event.start_time_s, event.end_time_s,
                alpha=0.15, color="red", zorder=2,
            )
            # Add SNR label at top
            mid_time = (event.start_time_s + event.end_time_s) / 2
            ax_spec.text(
                mid_time, freqs[-1] / 1e6 * 0.95,
                f"{event.mean_snr_db:.0f}dB",
                ha="center", va="top", fontsize=7,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="red", alpha=0.7),
            )

        ax_spec.set_ylabel("Frequency (MHz)")
        ax_spec.set_title(title)

        # Legend
        signal_patch = mpatches.Patch(color="red", alpha=0.3, label="Detected signal")
        ax_spec.legend(handles=[signal_patch], loc="upper right", fontsize=8)

        # === Panel 2: Energy profile with threshold ===
        ax_energy = axes[1]

        # Time axis for windows
        win_times = np.arange(annotation.num_windows) * (self.hop_length / sr)

        ax_energy.plot(win_times, annotation.energy_db, color="steelblue",
                       linewidth=0.5, label="Energy (dB)")
        ax_energy.axhline(annotation.threshold_db, color="red", linestyle="--",
                          linewidth=1.5, label=f"Threshold (P_fa={annotation.p_fa})")
        ax_energy.axhline(annotation.noise_floor_db, color="gray", linestyle=":",
                          linewidth=1, label=f"Noise floor ({annotation.noise_floor_db:.1f} dB)")

        # Shade detected regions
        for event in annotation.events:
            ax_energy.axvspan(event.start_time_s, event.end_time_s,
                              alpha=0.15, color="red")

        ax_energy.set_ylabel("Energy (dB)")
        ax_energy.legend(loc="upper right", fontsize=7)
        ax_energy.grid(True, alpha=0.3)

        # === Panel 3: SNR profile with detection mask ===
        ax_snr = axes[2]

        # Color SNR by detection state
        colors = np.where(annotation.detection_mask, "red", "steelblue")
        ax_snr.bar(win_times, annotation.snr_db, width=self.hop_length / sr,
                   color=colors, alpha=0.7, linewidth=0)
        ax_snr.axhline(0, color="black", linewidth=0.5)
        ax_snr.set_ylabel("Est. SNR (dB)")
        ax_snr.set_xlabel("Time (seconds)")
        ax_snr.grid(True, alpha=0.3)

        # Summary stats
        n_detected = int(annotation.detection_mask.sum())
        pct_detected = 100.0 * n_detected / annotation.num_windows
        fig.text(
            0.02, 0.02,
            f"Windows: {annotation.num_windows} | "
            f"Detected: {n_detected} ({pct_detected:.1f}%) | "
            f"Events: {len(annotation.events)} | "
            f"P_fa: {annotation.p_fa} | "
            f"Noise floor: {annotation.noise_floor_db:.1f} dB",
            fontsize=8, family="monospace",
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Saved visualization to %s", output_path)
            plt.close(fig)
        else:
            plt.show()

    def export_labeled_segments(
        self,
        iq_data: np.ndarray,
        annotation: AnnotationTrack,
        output_dir: str,
        format: str = "npy",
        min_event_duration_s: float = 0.001,
    ) -> list[dict]:
        """Export detected signal and noise segments as labeled files.

        Creates two directories:
        - output_dir/signal/ — IQ segments where drone signal was detected
        - output_dir/noise/ — IQ segments classified as noise-only

        Each file is a single segment_length IQ array with metadata.

        Args:
            iq_data: Full IQ recording.
            annotation: AnnotationTrack from analyze_recording().
            output_dir: Base output directory.
            format: Output format ("npy" or "pt" for PyTorch tensor).
            min_event_duration_s: Minimum event duration to export (filters glitches).

        Returns:
            List of dicts with segment metadata (path, label, snr_db, etc.)
        """
        out = Path(output_dir)
        (out / "signal").mkdir(parents=True, exist_ok=True)
        (out / "noise").mkdir(parents=True, exist_ok=True)

        # Normalize to (2, N)
        if np.iscomplexobj(iq_data):
            iq_2d = np.stack([iq_data.real, iq_data.imag], axis=0).astype(np.float32)
        elif iq_data.ndim == 2 and iq_data.shape[1] == 2:
            iq_2d = iq_data.T.astype(np.float32)
        else:
            iq_2d = iq_data.astype(np.float32)

        manifest = []
        seg_len = self.segment_length

        for i in range(annotation.num_windows):
            start = i * self.hop_length
            end = start + seg_len
            if end > iq_2d.shape[1]:
                break

            segment = iq_2d[:, start:end]
            is_signal = annotation.detection_mask[i]
            label = "signal" if is_signal else "noise"
            snr = float(annotation.snr_db[i])

            filename = f"{label}_{i:06d}_snr{snr:+.1f}dB"

            if format == "pt":
                filepath = out / label / f"{filename}.pt"
                torch.save(torch.from_numpy(segment), filepath)
            else:
                filepath = out / label / f"{filename}.npy"
                np.save(filepath, segment)

            manifest.append({
                "path": str(filepath),
                "label": label,
                "label_binary": 1 if is_signal else 0,
                "window_index": i,
                "start_sample": start,
                "end_sample": end,
                "snr_db": snr,
                "energy_db": float(annotation.energy_db[i]),
            })

        # Save manifest
        import json
        manifest_path = out / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        n_signal = sum(1 for m in manifest if m["label"] == "signal")
        n_noise = len(manifest) - n_signal
        logger.info(
            "Exported %d segments (%d signal, %d noise) to %s",
            len(manifest), n_signal, n_noise, output_dir,
        )

        return manifest

    def export_annotation_csv(
        self,
        annotation: AnnotationTrack,
        output_path: str,
    ):
        """Export annotation track as CSV for use in labeling tools.

        Columns: window_index, start_time_s, end_time_s, detected, energy_db, snr_db

        Args:
            annotation: AnnotationTrack from analyze_recording().
            output_path: CSV output path.
        """
        import csv

        sr = annotation.sample_rate

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "window_index", "start_sample", "end_sample",
                "start_time_s", "end_time_s",
                "detected", "energy_db", "snr_db",
            ])

            for i in range(annotation.num_windows):
                start = i * self.hop_length
                end = start + self.segment_length
                writer.writerow([
                    i,
                    start,
                    end,
                    f"{start / sr:.6f}",
                    f"{end / sr:.6f}",
                    int(annotation.detection_mask[i]),
                    f"{annotation.energy_db[i]:.2f}",
                    f"{annotation.snr_db[i]:.2f}",
                ])

        logger.info("Exported annotation CSV to %s", output_path)
