#!/usr/bin/env python3
"""Fast spectrogram generation — no matplotlib, direct numpy→PIL.
Uses 1M samples per spectrogram (0.01s at 100MSps) for speed."""
import numpy as np
import os, sys, glob, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import signal as scipy_signal
from PIL import Image

warnings.filterwarnings('ignore')

RAW_DIR = '/home/rax/mtp/raw'
OUT_DIR = '/home/rax/mtp/spectrograms'
FFT_SIZE = 256
FS = 100e6
SAMPLES_PER_SPEC = 1_000_000   # 0.01s — fast and sufficient
IMG_SIZE = 640
MAX_SPECS_PER_FILE = 10
TRAIN_RATIO = 0.8


def hot_colormap(data_norm):
    """Apply 'hot' colormap: black→red→yellow→white. Input: 0-1 float array."""
    r = np.clip(data_norm * 3, 0, 1)
    g = np.clip(data_norm * 3 - 1, 0, 1)
    b = np.clip(data_norm * 3 - 2, 0, 1)
    return np.stack([r, g, b], axis=-1)


def iq_to_spectrogram_image(iq_segment):
    """Convert IQ → spectrogram → RGB image array (no matplotlib)."""
    f, t, Zxx = scipy_signal.stft(iq_segment, fs=FS, nperseg=FFT_SIZE,
                                   noverlap=FFT_SIZE*3//4, window='hamming')
    mag = np.abs(Zxx)
    log_mag = 10 * np.log10(mag + 1e-10)

    # Normalize to 0-1
    vmin, vmax = np.percentile(log_mag, [2, 98])
    norm = np.clip((log_mag - vmin) / (vmax - vmin + 1e-10), 0, 1)

    # Apply hot colormap
    rgb = hot_colormap(norm)
    rgb = (rgb * 255).astype(np.uint8)

    # Resize to target
    img = Image.fromarray(rgb)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return img


def process_iq_file(args):
    """Process single .iq file → spectrogram PNGs."""
    iq_path, drone_name, split = args
    try:
        file_size = os.path.getsize(iq_path)
        total_complex = file_size // 8  # 2 x float32 per complex sample
        n_segments = min(total_complex // SAMPLES_PER_SPEC, MAX_SPECS_PER_FILE)
        if n_segments == 0:
            return 0

        spec_dir = os.path.join(OUT_DIR, split, drone_name)
        os.makedirs(spec_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(iq_path))[0]

        count = 0
        for i in range(n_segments):
            outpath = os.path.join(spec_dir, f'{basename}_s{i}.png')
            if os.path.exists(outpath):
                count += 1
                continue

            offset = i * SAMPLES_PER_SPEC * 2  # float32 offset (I,Q interleaved)
            raw = np.fromfile(iq_path, dtype=np.float32, offset=offset*4,
                              count=SAMPLES_PER_SPEC*2)
            if len(raw) < SAMPLES_PER_SPEC * 2:
                continue
            iq = raw[0::2] + 1j * raw[1::2]
            iq = iq - np.mean(iq)  # DC removal

            img = iq_to_spectrogram_image(iq)
            img.save(outpath)
            count += 1
        return count
    except Exception as e:
        print(f"  ERR {iq_path}: {e}", flush=True)
        return 0


def main():
    print("=" * 60, flush=True)
    print("RFUAV FAST SPECTROGRAM GENERATION", flush=True)
    print(f"FFT={FFT_SIZE}, samples/spec={SAMPLES_PER_SPEC/1e6:.2f}M, img={IMG_SIZE}", flush=True)
    print("=" * 60, flush=True)

    drones = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
    print(f"Found {len(drones)} drones", flush=True)

    tasks = []
    for drone in drones:
        iq_files = sorted(glob.glob(os.path.join(RAW_DIR, drone, '**', '*.iq'), recursive=True))
        n_train = max(1, int(len(iq_files) * TRAIN_RATIO))
        safe_name = drone.replace(' ', '_').replace('/', '_')
        for i, f in enumerate(iq_files):
            split = 'train' if i < n_train else 'val'
            tasks.append((f, safe_name, split))
        print(f"  {drone}: {len(iq_files)} files ({n_train}T/{len(iq_files)-n_train}V)", flush=True)

    print(f"\nTotal: {len(tasks)} .iq files, max {MAX_SPECS_PER_FILE} specs each", flush=True)
    print(f"Processing with 8 workers...\n", flush=True)

    total = 0
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(process_iq_file, t): t for t in tasks}
        for fut in as_completed(futures):
            n = fut.result()
            total += n
            done += 1
            if done % 10 == 0:
                print(f"  [{done}/{len(tasks)}] {total} spectrograms", flush=True)

    for split in ['train', 'val']:
        d = os.path.join(OUT_DIR, split)
        if os.path.exists(d):
            n = sum(len(os.listdir(os.path.join(d, c))) for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))
            print(f"{split}: {n} images", flush=True)

    print(f"\nTotal: {total} spectrograms → {OUT_DIR}", flush=True)
    print("=== COMPLETE ===", flush=True)

if __name__ == '__main__':
    main()
