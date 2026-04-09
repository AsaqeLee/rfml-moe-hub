#!/usr/bin/env python3
"""
DRFF-R2 Spectrogram Generator
==============================
Converts DRFF-R2 MATLAB v7.3 (HDF5) .mat files to 640x640 PNG spectrograms
using STFT (FFT=256, Hamming window, Hot colormap) without matplotlib.

Dataset layout:
  /home/rax/mtp/drffr2/DRFF-R2/
    dataset1/   *.mat
    dataset2/   (drone_mixed) — recurse into subfolders
    dataset3/   *.mat
    ...
    dataset7/   *.mat

Each .mat file contains:
  RF0_I        (140000000, 1) float32  — IQ in-phase
  RF0_Q        (140000000, 1) float32  — IQ quadrature
  Fs           float64 = 100e6
  CenterFrequence float64 = 5.745e9
  TD           uint16[]  → ASCII  e.g. "mavicAir2_1"
  State        uint16[]  → ASCII  e.g. "Cruise"
  Distance     uint16[]  → ASCII
  Height       uint16[]  → ASCII

Output:
  /home/rax/mtp/drffr2_spectrograms/{dataset}/{train,val}/{drone_model}/*.png

Metadata CSV per dataset:
  /home/rax/mtp/drffr2_spectrograms/{dataset}/metadata.csv

Usage:
  python3 drffr2_specgen.py [--datasets dataset1 dataset3 ...] [--workers N]
"""

import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.signal import stft

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("/home/rax/mtp/drffr2/DRFF-R2")
OUTPUT_ROOT  = Path("/home/rax/mtp/drffr2_spectrograms")

SAMPLE_RATE   = 100e6          # 100 MSps
CHUNK_SAMPLES = 10_000_000     # 10M samples per spectrogram segment
MAX_CHUNKS    = 10             # max spectrograms per .mat file
TOTAL_SAMPLES = 140_000_000    # expected total IQ samples per file

# STFT parameters
NFFT     = 256
NOVERLAP = 128                 # 50% overlap
WINDOW   = "hamming"

# Output image size
IMG_SIZE = (640, 640)          # (width, height)

# Train/val split ratio (temporal — first 80% -> train, last 20% -> val)
TRAIN_RATIO = 0.80

# Workers (files are huge, 8 is safe)
NUM_WORKERS = 8

# All 7 dataset subfolder names
ALL_DATASETS = [f"dataset{i}" for i in range(1, 8)]

# Known drone models (8 classes)
DRONE_MODELS = [
    "mavicAir2", "mavic3", "mavic3C", "mavic3S",
    "mavicAir2s", "mini3pro", "mini4PRO", "mini5PRO",
]

# ---------------------------------------------------------------------------
# Hot colormap — 256-entry RGB LUT matching matplotlib's "hot"
# Ramp: black -> red -> orange -> yellow -> white
# ---------------------------------------------------------------------------
def _build_hot_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    n0 = 96   # black  -> red    R ramps 0->255
    n1 = 96   # red    -> yellow G ramps 0->255
    n2 = 64   # yellow -> white  B ramps 0->255
    lut[:n0, 0] = np.linspace(0, 255, n0, dtype=np.uint8)
    lut[n0:n0+n1, 0] = 255
    lut[n0:n0+n1, 1] = np.linspace(0, 255, n1, dtype=np.uint8)
    lut[n0+n1:, 0] = 255
    lut[n0+n1:, 1] = 255
    lut[n0+n1:, 2] = np.linspace(0, 255, n2, dtype=np.uint8)
    return lut

HOT_LUT = _build_hot_lut()


def apply_hot_colormap(gray: np.ndarray) -> np.ndarray:
    """Map (H, W) uint8 array through Hot LUT -> (H, W, 3) uint8."""
    return HOT_LUT[gray]


# ---------------------------------------------------------------------------
# Spectrogram computation
# ---------------------------------------------------------------------------

def iq_to_spectrogram_image(iq_chunk: np.ndarray) -> Image.Image:
    """
    Convert a 1-D complex IQ chunk to a 640x640 PIL Image (Hot colormap).

    Steps:
      1. STFT FFT=256, Hamming, 50% overlap -> complex (F, T)
      2. Power in dB: 10*log10(|X|^2 + eps)
      3. Frequency-shift so DC is centred (fftshift on freq axis)
      4. Normalise to [0, 255]
      5. Resize to 640x640
      6. Apply Hot LUT -> RGB
    """
    _, _, Zxx = stft(
        iq_chunk,
        fs=SAMPLE_RATE,
        window=WINDOW,
        nperseg=NFFT,
        noverlap=NOVERLAP,
        nfft=NFFT,
        return_onesided=False,
    )

    power = np.abs(Zxx) ** 2
    power_db = 10.0 * np.log10(power + 1e-12)
    power_db = np.fft.fftshift(power_db, axes=0)

    vmin, vmax = power_db.min(), power_db.max()
    if vmax - vmin < 1e-6:
        gray = np.zeros_like(power_db, dtype=np.uint8)
    else:
        gray = ((power_db - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    pil_gray = Image.fromarray(gray, mode="L")
    pil_gray = pil_gray.resize(IMG_SIZE, Image.LANCZOS)

    gray_arr = np.array(pil_gray)
    rgb_arr  = apply_hot_colormap(gray_arr)
    return Image.fromarray(rgb_arr, mode="RGB")


# ---------------------------------------------------------------------------
# Metadata decoding
# ---------------------------------------------------------------------------

def decode_uint16_str(arr) -> str:
    """Decode a uint16 HDF5 dataset to ASCII string via chr()."""
    try:
        vals = np.array(arr).flatten().astype(np.uint16)
        return "".join(chr(v) for v in vals if v != 0).strip()
    except Exception:
        return ""


def parse_drone_model(td_str: str) -> str:
    """
    Extract drone model from TD field.
    Examples:
      'mavicAir2_1'  -> 'mavicAir2'
      'mini4PRO_3'   -> 'mini4PRO'
      'mavic3C_2'    -> 'mavic3C'
    Strategy: strip trailing underscore+digits.
    """
    # Find the last underscore followed only by digits
    idx = td_str.rfind("_")
    if idx != -1 and td_str[idx+1:].isdigit():
        return td_str[:idx]
    return td_str


def read_scalar_str(val) -> str:
    """Read a scalar uint16 field that may encode a number (Distance, Height)."""
    try:
        arr = np.array(val).flatten()
        if arr.dtype in (np.uint16, np.int16, np.uint8, np.int8):
            s = "".join(chr(v) for v in arr if v != 0).strip()
            return s if s else str(arr[0])
        return str(arr[0])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Worker: process one .mat file
# ---------------------------------------------------------------------------

def process_mat_file(args: tuple) -> tuple:
    """
    Worker function: load one .mat file, generate up to MAX_CHUNKS spectrograms.

    args: (mat_path_str, train_out_dir, val_out_dir, stem, meta_dict)

    Returns:
      (mat_path_str, n_train_saved, n_val_saved, error_msg_or_None, meta_dict)
    """
    mat_path_str, train_out_dir, val_out_dir, stem, meta_dict = args
    try:
        with h5py.File(mat_path_str, "r") as f:
            I = f["RF0_I"][:, 0].astype(np.float32)  # (140000000,)
            Q = f["RF0_Q"][:, 0].astype(np.float32)

        iq = I + 1j * Q
        total = len(iq)

        # Determine available chunks
        n_chunks = min(MAX_CHUNKS, total // CHUNK_SAMPLES)
        if n_chunks == 0:
            return (mat_path_str, 0, 0, f"too few samples: {total}", meta_dict)

        # Temporal 80/20 split at chunk level
        n_train = max(1, int(round(n_chunks * TRAIN_RATIO)))
        n_val   = n_chunks - n_train

        os.makedirs(train_out_dir, exist_ok=True)
        if n_val > 0:
            os.makedirs(val_out_dir, exist_ok=True)

        train_saved = 0
        for idx in range(n_train):
            start = idx * CHUNK_SAMPLES
            chunk = iq[start:start + CHUNK_SAMPLES]
            img   = iq_to_spectrogram_image(chunk)
            out_path = os.path.join(train_out_dir, f"{stem}_chunk{idx:02d}.png")
            img.save(out_path)
            train_saved += 1

        val_saved = 0
        for idx in range(n_val):
            global_idx = n_train + idx
            start = global_idx * CHUNK_SAMPLES
            chunk = iq[start:start + CHUNK_SAMPLES]
            img   = iq_to_spectrogram_image(chunk)
            out_path = os.path.join(val_out_dir, f"{stem}_chunk{global_idx:02d}.png")
            img.save(out_path)
            val_saved += 1

        return (mat_path_str, train_saved, val_saved, None, meta_dict)

    except Exception:
        return (mat_path_str, 0, 0, traceback.format_exc(limit=4), meta_dict)


# ---------------------------------------------------------------------------
# Build work list for one dataset folder
# ---------------------------------------------------------------------------

def find_mat_files(dataset_dir: Path, dataset_name: str) -> list[Path]:
    """
    Find all .mat files under dataset_dir.
    dataset2 (drone_mixed) recurses into subfolders.
    """
    if dataset_name == "dataset2":
        # Recurse
        return sorted(dataset_dir.rglob("*.mat"))
    else:
        return sorted(dataset_dir.glob("*.mat"))


def build_work_list(dataset_name: str) -> tuple[list[tuple], list[dict]]:
    """
    Build work items for a dataset.

    Returns:
      (work_items, meta_rows)
      work_items: list of (mat_path_str, train_out_dir, val_out_dir, stem, meta_dict)
      meta_rows:  list of metadata dicts (populated after processing when possible)
    """
    dataset_dir = DATASET_ROOT / dataset_name
    if not dataset_dir.exists():
        print(f"[SKIP] {dataset_dir} does not exist", flush=True)
        return [], []

    mat_files = find_mat_files(dataset_dir, dataset_name)
    if not mat_files:
        print(f"[WARN] No .mat files in {dataset_dir}", flush=True)
        return [], []

    work = []
    meta_rows = []

    for mat_path in mat_files:
        stem = mat_path.stem

        # Read metadata inline (cheap, just opens file for small fields)
        meta = {
            "filename": mat_path.name,
            "dataset":  dataset_name,
            "drone_model": "",
            "individual": "",
            "state":    "",
            "distance": "",
            "height":   "",
        }
        try:
            with h5py.File(str(mat_path), "r") as f:
                td_str   = decode_uint16_str(f["TD"])
                state    = decode_uint16_str(f["State"])
                distance = read_scalar_str(f["Distance"])
                height   = read_scalar_str(f["Height"])

            model_str = parse_drone_model(td_str)
            meta["individual"]  = td_str
            meta["drone_model"] = model_str
            meta["state"]       = state
            meta["distance"]    = distance
            meta["height"]      = height
        except Exception as e:
            print(f"[WARN] metadata read failed for {mat_path.name}: {e}", flush=True)
            model_str = "unknown"
            meta["drone_model"] = model_str

        # Output directories for train and val
        out_base      = OUTPUT_ROOT / dataset_name
        train_out_dir = str(out_base / "train" / model_str)
        val_out_dir   = str(out_base / "val"   / model_str)

        work.append((str(mat_path), train_out_dir, val_out_dir, stem, meta))
        meta_rows.append(meta)

    return work, meta_rows


# ---------------------------------------------------------------------------
# Save metadata CSV
# ---------------------------------------------------------------------------

def save_metadata_csv(dataset_name: str, meta_rows: list[dict]) -> None:
    csv_path = OUTPUT_ROOT / dataset_name / "metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "dataset", "drone_model", "individual",
                  "state", "distance", "height"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"  Metadata CSV -> {csv_path}", flush=True)


# ---------------------------------------------------------------------------
# Process one dataset
# ---------------------------------------------------------------------------

def process_dataset(dataset_name: str) -> None:
    print(f"\n{'='*70}", flush=True)
    print(f"  Dataset: {dataset_name}", flush=True)
    print(f"{'='*70}", flush=True)

    work, meta_rows = build_work_list(dataset_name)
    if not work:
        return

    total   = len(work)
    done    = 0
    errors  = 0
    t_start = time.time()

    print(f"  {total} .mat files queued, {NUM_WORKERS} workers", flush=True)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(process_mat_file, item): item for item in work}
        for fut in as_completed(futures):
            mat_path, n_tr, n_val, err, _meta = fut.result()
            done += 1
            if err:
                errors += 1
                print(f"  [ERR] {Path(mat_path).name}: {err[:200]}", flush=True)
            else:
                if done % 50 == 0 or done == total:
                    elapsed = time.time() - t_start
                    rate    = done / elapsed if elapsed > 0 else 0
                    eta     = (total - done) / rate if rate > 0 else 0
                    print(
                        f"  [{done}/{total}] train+{n_tr} val+{n_val} | "
                        f"{rate:.1f} files/s | ETA {eta:.0f}s",
                        flush=True,
                    )

    elapsed = time.time() - t_start
    print(
        f"  Done: {done} files in {elapsed:.1f}s, {errors} errors",
        flush=True,
    )

    # Save metadata CSV for this dataset
    save_metadata_csv(dataset_name, meta_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global NUM_WORKERS

    parser = argparse.ArgumentParser(
        description="DRFF-R2 Spectrogram Generator — MI300X ROCm"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help=f"Dataset subfolders to process (default: all). Choices: {ALL_DATASETS}"
    )
    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS,
        help=f"ProcessPool workers (default: {NUM_WORKERS})"
    )
    args = parser.parse_args()

    NUM_WORKERS = args.workers

    print("DRFF-R2 Spectrogram Generator", flush=True)
    print(f"  Dataset root : {DATASET_ROOT}", flush=True)
    print(f"  Output root  : {OUTPUT_ROOT}", flush=True)
    print(f"  STFT         : FFT={NFFT}, window={WINDOW}, overlap={NOVERLAP}", flush=True)
    print(f"  Chunks       : up to {MAX_CHUNKS} x {CHUNK_SAMPLES:,} samples/file", flush=True)
    print(f"  Split        : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)} train/val (temporal)", flush=True)
    print(f"  Image        : {IMG_SIZE[0]}x{IMG_SIZE[1]} px, Hot colormap", flush=True)
    print(f"  Workers      : {NUM_WORKERS}", flush=True)
    print(f"  Datasets     : {args.datasets}", flush=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        process_dataset(dataset_name)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
