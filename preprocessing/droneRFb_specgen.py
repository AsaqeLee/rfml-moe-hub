#!/usr/bin/env python3
"""
DroneRFb spectrogram generator.

Converts DroneRFb-DIR .mat (HDF5/MATLAB v7.3) files to 640x640 PNG spectrograms
using STFT (FFT=256, Hamming window, Hot colormap) without matplotlib.

Usage on remote server:
    python3 /home/rax/mtp/scripts/droneRFb_specgen.py

Output: /home/rax/mtp/droneRFb_spectrograms/{train,test}/{class}/*.png
"""

import os
import re
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
DATASET_ROOT = Path("/home/rax/mtp/droneRFb/extracted/twin_droneRF")
OUTPUT_ROOT  = Path("/home/rax/mtp/droneRFb_spectrograms")

SAMPLE_RATE   = 80e6          # 80 MSps
CHUNK_SAMPLES = 500_000       # samples per spectrogram segment
MAX_CHUNKS    = 8             # max spectrograms per .mat file
TOTAL_SAMPLES = 4_000_000     # expected total IQ samples per file

# STFT parameters (match RFUAV pipeline)
NFFT      = 256
NOVERLAP  = 128               # 50 % overlap
WINDOW    = "hamming"

# Output image size
IMG_SIZE  = (640, 640)        # (width, height)

# Workers
NUM_WORKERS = 16

# ---------------------------------------------------------------------------
# Hot colormap — 256-entry RGB LUT matching matplotlib's "hot"
# Ramp: black→red→orange→yellow→white
# ---------------------------------------------------------------------------
def _build_hot_lut() -> np.ndarray:
    """Return (256, 3) uint8 array for the 'hot' colormap."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    # Phase 0: black → red  (0..95)   R ramps 0→255
    n0 = 96
    lut[:n0, 0] = np.linspace(0, 255, n0, dtype=np.uint8)
    # Phase 1: red → yellow (96..191) G ramps 0→255
    n1 = 96
    lut[n0:n0+n1, 0] = 255
    lut[n0:n0+n1, 1] = np.linspace(0, 255, n1, dtype=np.uint8)
    # Phase 2: yellow → white (192..255) B ramps 0→255
    n2 = 64
    lut[n0+n1:, 0] = 255
    lut[n0+n1:, 1] = 255
    lut[n0+n1:, 2] = np.linspace(0, 255, n2, dtype=np.uint8)
    return lut

HOT_LUT = _build_hot_lut()


def apply_hot_colormap(gray: np.ndarray) -> np.ndarray:
    """
    Map a (H, W) uint8 array through the Hot LUT → (H, W, 3) uint8.
    gray must already be in [0, 255].
    """
    return HOT_LUT[gray]


# ---------------------------------------------------------------------------
# Spectrogram computation
# ---------------------------------------------------------------------------

def iq_to_spectrogram_image(iq_chunk: np.ndarray) -> Image.Image:
    """
    Convert a 1-D complex IQ chunk to a 640x640 PIL Image (Hot colormap).

    Steps:
      1. STFT with FFT=256, Hamming, 50% overlap → complex (F, T)
      2. Power in dB: 10*log10(|X|^2 + eps)
      3. Frequency-shift so DC is centred (fftshift on freq axis)
      4. Normalise to [0, 255]
      5. Resize to 640x640
      6. Apply Hot LUT → RGB
    """
    # STFT — returns (freqs, times, complex_spectrum)
    _, _, Zxx = stft(
        iq_chunk,
        fs=SAMPLE_RATE,
        window=WINDOW,
        nperseg=NFFT,
        noverlap=NOVERLAP,
        nfft=NFFT,
        return_onesided=False,   # two-sided for complex IQ
    )

    # Power spectrum (dB)
    power = np.abs(Zxx) ** 2
    power_db = 10.0 * np.log10(power + 1e-12)

    # Centre-frequency shift on the freq axis
    power_db = np.fft.fftshift(power_db, axes=0)

    # Normalise to uint8
    vmin, vmax = power_db.min(), power_db.max()
    if vmax - vmin < 1e-6:
        gray = np.zeros_like(power_db, dtype=np.uint8)
    else:
        gray = ((power_db - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # PIL image from grayscale array, resize, apply colormap
    # gray shape: (freq_bins, time_bins) — PIL expects (width, height) for resize
    # We treat freq as height (rows) and time as width (cols)
    pil_gray = Image.fromarray(gray, mode="L")
    pil_gray = pil_gray.resize(IMG_SIZE, Image.LANCZOS)

    # Apply Hot colormap
    gray_arr = np.array(pil_gray)          # (640, 640) uint8
    rgb_arr  = apply_hot_colormap(gray_arr) # (640, 640, 3) uint8
    return Image.fromarray(rgb_arr, mode="RGB")


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_mat_file(args: tuple) -> tuple[str, int, str | None]:
    """
    Worker function: load one .mat file, generate up to MAX_CHUNKS spectrograms.

    Returns (mat_path, n_saved, error_msg_or_None).
    """
    mat_path, out_class_dir, stem = args
    try:
        with h5py.File(mat_path, "r") as f:
            I = f["I"][0, :].astype(np.float32)  # (4000000,)
            Q = f["Q"][0, :].astype(np.float32)

        iq = I + 1j * Q
        n_chunks = min(MAX_CHUNKS, len(iq) // CHUNK_SAMPLES)
        if n_chunks == 0:
            return (str(mat_path), 0, "too few samples")

        os.makedirs(out_class_dir, exist_ok=True)
        saved = 0
        for idx in range(n_chunks):
            start = idx * CHUNK_SAMPLES
            chunk = iq[start : start + CHUNK_SAMPLES]
            img   = iq_to_spectrogram_image(chunk)
            out_path = os.path.join(out_class_dir, f"{stem}_chunk{idx:02d}.png")
            img.save(out_path)
            saved += 1

        return (str(mat_path), saved, None)

    except Exception as e:
        return (str(mat_path), 0, traceback.format_exc(limit=3))


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_train_labels(label_file: Path) -> dict[str, str]:
    """
    Parse train_labels.txt: "filename classname" per line.
    Returns {stem: classname}.
    """
    mapping = {}
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                fname, cls = parts[0], parts[1]
                stem = Path(fname).stem
                mapping[stem] = cls
    return mapping


def extract_class_from_stem(stem: str) -> str | None:
    """
    Extract drone class from a test filename stem.
    Examples:
      D1_IN_S2_slice_47  → D1
      A3_OUT_S1_slice_0  → A3
      F3_slice_10        → F3
    Strategy: first token split by '_' that looks like a class label
    (letter(s) followed by digit(s)).
    """
    # known class pattern: one or two uppercase letters optionally followed by one digit,
    # then a non-alpha boundary (underscore, digit end, or end-of-string).
    # Covers: D1, A3, B, G2, etc.
    match = re.match(r"^([A-Z]{1,2}\d?)(?:_|$)", stem)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Build work list
# ---------------------------------------------------------------------------

def build_work_list(split: str) -> list[tuple]:
    """
    Build list of (mat_path, out_class_dir, stem) tuples for a given split.
    """
    mat_dir    = DATASET_ROOT / split
    label_file = DATASET_ROOT / f"{split}_labels.txt"
    out_split  = OUTPUT_ROOT / split

    mat_files  = sorted(mat_dir.glob("*.mat"))
    if not mat_files:
        print(f"[WARN] No .mat files found in {mat_dir}", flush=True)
        return []

    work = []

    if split == "train":
        label_map = parse_train_labels(label_file)
        missing_label = 0
        for mat_path in mat_files:
            stem = mat_path.stem
            cls  = label_map.get(stem)
            if cls is None:
                # Try matching without extension in label file key
                cls = label_map.get(mat_path.name)
            if cls is None:
                missing_label += 1
                continue
            out_class_dir = str(out_split / cls)
            work.append((str(mat_path), out_class_dir, stem))
        if missing_label:
            print(f"[WARN] {missing_label} train files had no label entry", flush=True)

    else:  # test — derive class from filename
        no_class = 0
        for mat_path in mat_files:
            stem = mat_path.stem
            cls  = extract_class_from_stem(stem)
            if cls is None:
                no_class += 1
                continue
            out_class_dir = str(out_split / cls)
            work.append((str(mat_path), out_class_dir, stem))
        if no_class:
            print(f"[WARN] {no_class} test files had no parseable class", flush=True)

    return work


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_split(split: str) -> None:
    print(f"\n=== Processing split: {split} ===", flush=True)
    work = build_work_list(split)
    if not work:
        print(f"  No work items for {split}.", flush=True)
        return

    total   = len(work)
    done    = 0
    errors  = 0
    t_start = time.time()

    print(f"  {total} .mat files queued, {NUM_WORKERS} workers", flush=True)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(process_mat_file, item): item for item in work}
        for fut in as_completed(futures):
            mat_path, n_saved, err = fut.result()
            done += 1
            if err:
                errors += 1
                print(f"  [ERR] {Path(mat_path).name}: {err}", flush=True)
            else:
                # Progress every 100 files
                if done % 100 == 0 or done == total:
                    elapsed = time.time() - t_start
                    rate    = done / elapsed if elapsed > 0 else 0
                    eta     = (total - done) / rate if rate > 0 else 0
                    print(
                        f"  [{done}/{total}] +{n_saved} imgs | "
                        f"{rate:.1f} files/s | ETA {eta:.0f}s",
                        flush=True,
                    )

    elapsed = time.time() - t_start
    print(
        f"  Done: {done} files in {elapsed:.1f}s, {errors} errors",
        flush=True,
    )


def main() -> None:
    print("DroneRFb Spectrogram Generator", flush=True)
    print(f"  Dataset : {DATASET_ROOT}", flush=True)
    print(f"  Output  : {OUTPUT_ROOT}", flush=True)
    print(f"  STFT    : FFT={NFFT}, window={WINDOW}, overlap={NOVERLAP}", flush=True)
    print(f"  Chunks  : {MAX_CHUNKS} x {CHUNK_SAMPLES:,} samples per file", flush=True)
    print(f"  Image   : {IMG_SIZE[0]}x{IMG_SIZE[1]} px, Hot colormap", flush=True)
    print(f"  Workers : {NUM_WORKERS}", flush=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            print(f"[SKIP] {split_dir} does not exist", flush=True)
            continue
        run_split(split)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
