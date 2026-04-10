#!/usr/bin/env python3
"""
Preprocess raw RF datasets into unified format for training.

Converts all downloaded datasets into a common format:
- IQ data: numpy arrays of shape (2, N) - I and Q channels
- Labels: hierarchical (binary, type, full taxonomy)
- Metadata: SNR, dataset source, frequency band

Output structure:
  data/processed/{dataset_name}/{split}/
    samples.h5   (HDF5 with IQ data, labels, metadata)
    manifest.json (sample index with metadata)
"""

import json
import logging
import struct
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm

log = logging.getLogger("rfml.preprocess")

# Label mappings for hierarchical classification
BINARY_LABELS = {"no_drone": 0, "drone": 1}

TYPE_LABELS = {
    "no_signal": 0, "wifi": 1, "bluetooth": 2, "zigbee": 3,
    "dji_fpv": 4, "dji_lightbridge": 5, "dji_ocusync": 6, "dji_wifi": 7,
    "parrot_wifi": 8, "frsky_accst": 9, "frsky_access": 10,
    "crossfire": 11, "elrs": 12, "flysky_afhds": 13, "futaba_fasstest": 14,
}


class DatasetPreprocessor:
    """Base class for dataset-specific preprocessing."""

    def __init__(self, raw_dir: Path, output_dir: Path, sample_length: int = 32768):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.sample_length = sample_length
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self):
        raise NotImplementedError

    def _segment_iq(self, iq_data: np.ndarray, overlap: float = 0.5) -> list[np.ndarray]:
        """Segment long IQ recording into fixed-length segments with overlap."""
        if iq_data.ndim == 1:
            # Interleaved I/Q
            iq_data = iq_data.reshape(-1, 2).T
        elif iq_data.ndim == 2 and iq_data.shape[0] != 2:
            iq_data = iq_data.T

        total_samples = iq_data.shape[1]
        step = int(self.sample_length * (1 - overlap))
        segments = []
        for start in range(0, total_samples - self.sample_length + 1, step):
            seg = iq_data[:, start : start + self.sample_length].copy()
            # RMS power normalization per channel
            for ch in range(2):
                rms = np.sqrt(np.mean(seg[ch] ** 2) + 1e-10)
                seg[ch] = (seg[ch] - np.mean(seg[ch])) / rms
            segments.append(seg.astype(np.float32))
        return segments

    def _save_hdf5(self, samples: list[dict], output_path: Path):
        """Save processed samples to HDF5."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as f:
            n = len(samples)
            iq_ds = f.create_dataset(
                "iq", shape=(n, 2, self.sample_length), dtype="float32",
                chunks=(min(64, n), 2, self.sample_length), compression="lzf",
            )
            label_binary = f.create_dataset("label_binary", shape=(n,), dtype="int64")
            label_type = f.create_dataset("label_type", shape=(n,), dtype="int64")
            label_full = f.create_dataset("label_full", shape=(n,), dtype="int64")
            snr_db = f.create_dataset("snr_db", shape=(n,), dtype="float32")

            for i, s in enumerate(tqdm(samples, desc="Writing HDF5")):
                iq_ds[i] = s["iq"]
                label_binary[i] = s["label_binary"]
                label_type[i] = s["label_type"]
                label_full[i] = s["label_full"]
                snr_db[i] = s.get("snr_db", 30.0)

            f.attrs["num_samples"] = n
            f.attrs["sample_length"] = self.sample_length

        log.info(f"Saved {n} samples to {output_path}")


class RFUAVPreprocessor(DatasetPreprocessor):
    """Preprocess RFUAV dataset (binary IQ fp32, 37 drone types)."""

    def process(self):
        log.info("Processing RFUAV dataset...")
        raw_path = self.raw_dir / "rfuav"
        if not raw_path.exists():
            log.warning(f"RFUAV raw data not found at {raw_path}, skipping")
            return

        samples = []
        for drone_dir in sorted(raw_path.iterdir()):
            if not drone_dir.is_dir():
                continue
            drone_name = drone_dir.name
            label_full = hash(drone_name) % 50  # placeholder mapping
            label_type = hash(drone_name) % 15
            label_binary = 1  # all are drones

            for iq_file in sorted(drone_dir.glob("*.bin")):
                try:
                    data = np.fromfile(iq_file, dtype=np.float32)
                    segments = self._segment_iq(data)
                    for seg in segments:
                        samples.append({
                            "iq": seg,
                            "label_binary": label_binary,
                            "label_type": label_type,
                            "label_full": label_full,
                            "snr_db": 30.0,
                        })
                except Exception as e:
                    log.error(f"Error processing {iq_file}: {e}")

        if samples:
            self._split_and_save(samples, "rfuav")

    def _split_and_save(self, samples: list[dict], name: str):
        np.random.shuffle(samples)
        n = len(samples)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        splits = {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:],
        }
        for split_name, split_samples in splits.items():
            if split_samples:
                self._save_hdf5(
                    split_samples,
                    self.output_dir / name / split_name / "samples.h5",
                )


class DroneDetectPreprocessor(DatasetPreprocessor):
    """Preprocess DroneDetect v2 dataset (complex IQ .dat files)."""

    def process(self):
        log.info("Processing DroneDetect v2 dataset...")
        raw_path = self.raw_dir / "dronedetect_v2"
        if not raw_path.exists():
            log.warning(f"DroneDetect v2 not found at {raw_path}, skipping")
            return

        samples = []
        for dat_file in sorted(raw_path.rglob("*.dat")):
            try:
                data = np.fromfile(dat_file, dtype=np.complex64)
                iq = np.stack([data.real, data.imag], axis=0)
                segments = self._segment_iq(iq)
                # Infer label from directory/filename
                label_binary = 1 if "drone" in dat_file.stem.lower() else 0
                for seg in segments:
                    samples.append({
                        "iq": seg,
                        "label_binary": label_binary,
                        "label_type": 0,
                        "label_full": 0,
                        "snr_db": 20.0,
                    })
            except Exception as e:
                log.error(f"Error processing {dat_file}: {e}")

        if samples:
            self._split_and_save(samples, "dronedetect_v2")

    def _split_and_save(self, samples, name):
        np.random.shuffle(samples)
        n = len(samples)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        for split_name, sl in [
            ("train", slice(0, train_end)),
            ("val", slice(train_end, val_end)),
            ("test", slice(val_end, n)),
        ]:
            if samples[sl]:
                self._save_hdf5(samples[sl], self.output_dir / name / split_name / "samples.h5")


class CardRFPreprocessor(DatasetPreprocessor):
    """Preprocess CardRF dataset (.mat files)."""

    def process(self):
        log.info("Processing CardRF dataset...")
        raw_path = self.raw_dir / "cardrf"
        if not raw_path.exists():
            log.warning(f"CardRF not found at {raw_path}, skipping")
            return

        try:
            import scipy.io as sio
        except ImportError:
            log.error("scipy required for .mat files. Install with: pip install scipy")
            return

        samples = []
        for mat_file in sorted(raw_path.rglob("*.mat")):
            try:
                mat = sio.loadmat(mat_file)
                for key in mat:
                    if key.startswith("_"):
                        continue
                    data = mat[key]
                    if not isinstance(data, np.ndarray):
                        continue
                    if np.iscomplexobj(data):
                        data = data.flatten()
                        iq = np.stack([data.real, data.imag], axis=0)
                    elif data.ndim >= 2 and data.shape[-1] == 2:
                        iq = data.reshape(-1, 2).T
                    else:
                        continue

                    segments = self._segment_iq(iq)
                    is_drone = "uav" in mat_file.stem.lower() or "drone" in mat_file.stem.lower()
                    for seg in segments:
                        samples.append({
                            "iq": seg,
                            "label_binary": 1 if is_drone else 0,
                            "label_type": 0,
                            "label_full": 0,
                            "snr_db": 25.0,
                        })
            except Exception as e:
                log.error(f"Error processing {mat_file}: {e}")

        if samples:
            self._split_and_save(samples, "cardrf")

    def _split_and_save(self, samples, name):
        np.random.shuffle(samples)
        n = len(samples)
        t, v = int(0.7 * n), int(0.85 * n)
        for sn, sl in [("train", slice(0, t)), ("val", slice(t, v)), ("test", slice(v, n))]:
            if samples[sl]:
                self._save_hdf5(samples[sl], self.output_dir / name / sn / "samples.h5")


class DroneRFPreprocessor(DatasetPreprocessor):
    """Preprocess DroneRF dataset (CSV amplitude data)."""

    def process(self):
        log.info("Processing DroneRF dataset...")
        raw_path = self.raw_dir / "dronerf"
        if not raw_path.exists():
            log.warning(f"DroneRF not found at {raw_path}, skipping")
            return

        samples = []
        for csv_file in sorted(raw_path.rglob("*.csv")):
            try:
                import pandas as pd
                df = pd.read_csv(csv_file, header=None)
                data = df.values.flatten().astype(np.float32)
                # DroneRF is amplitude only — create synthetic IQ (I=amplitude, Q=0)
                iq = np.stack([data, np.zeros_like(data)], axis=0)
                segments = self._segment_iq(iq)

                is_drone = any(x in csv_file.stem.lower() for x in ["bebop", "phantom", "ar"])
                for seg in segments:
                    samples.append({
                        "iq": seg,
                        "label_binary": 1 if is_drone else 0,
                        "label_type": 0,
                        "label_full": 0,
                        "snr_db": 20.0,
                    })
            except Exception as e:
                log.error(f"Error processing {csv_file}: {e}")

        if samples:
            self._split_and_save(samples, "dronerf")

    def _split_and_save(self, samples, name):
        np.random.shuffle(samples)
        n = len(samples)
        t, v = int(0.7 * n), int(0.85 * n)
        for sn, sl in [("train", slice(0, t)), ("val", slice(t, v)), ("test", slice(v, n))]:
            if samples[sl]:
                self._save_hdf5(samples[sl], self.output_dir / name / sn / "samples.h5")


class TamperePreprocessor(DatasetPreprocessor):
    """Preprocess Tampere/Zenodo dataset (IQ int16)."""

    def process(self):
        log.info("Processing Tampere/Zenodo dataset...")
        raw_path = self.raw_dir / "tampere_zenodo"
        if not raw_path.exists():
            log.warning(f"Tampere dataset not found at {raw_path}, skipping")
            return

        samples = []
        for bin_file in sorted(raw_path.rglob("*.bin")) + sorted(raw_path.rglob("*.raw")):
            try:
                data = np.fromfile(bin_file, dtype=np.int16).astype(np.float32)
                data /= 32768.0  # normalize int16 to [-1, 1]
                if len(data) % 2 != 0:
                    data = data[:-1]
                iq = data.reshape(-1, 2).T
                segments = self._segment_iq(iq)

                for seg in segments:
                    samples.append({
                        "iq": seg,
                        "label_binary": 1,
                        "label_type": 0,
                        "label_full": 0,
                        "snr_db": 25.0,
                    })
            except Exception as e:
                log.error(f"Error processing {bin_file}: {e}")

        if samples:
            self._split_and_save(samples, "tampere_zenodo")

    def _split_and_save(self, samples, name):
        np.random.shuffle(samples)
        n = len(samples)
        t, v = int(0.7 * n), int(0.85 * n)
        for sn, sl in [("train", slice(0, t)), ("val", slice(t, v)), ("test", slice(v, n))]:
            if samples[sl]:
                self._save_hdf5(samples[sl], self.output_dir / name / sn / "samples.h5")


def preprocess_all(config: dict, raw_dir: str = "data/raw", output_dir: str = "data/processed"):
    """Run all dataset preprocessors."""
    log.info("Starting preprocessing pipeline...")
    raw = Path(raw_dir)
    out = Path(output_dir)
    sample_length = config.get("data", {}).get("sample_length", 32768)

    preprocessors = [
        RFUAVPreprocessor(raw, out, sample_length),
        DroneDetectPreprocessor(raw, out, sample_length),
        CardRFPreprocessor(raw, out, sample_length),
        DroneRFPreprocessor(raw, out, sample_length),
        TamperePreprocessor(raw, out, sample_length),
    ]

    for proc in preprocessors:
        try:
            proc.process()
        except Exception as e:
            log.error(f"Error in {proc.__class__.__name__}: {e}")

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import load_config
    from utils.logging import setup_logging

    setup_logging()
    cfg = load_config("configs/default.yaml")
    preprocess_all(cfg)
