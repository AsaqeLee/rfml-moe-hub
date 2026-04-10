#!/usr/bin/env python3
"""Generate test spectrograms for DroneRFb-DIR using label file mapping."""
import numpy as np, h5py, os, re, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import signal as scipy_signal
from PIL import Image

warnings.filterwarnings('ignore')

DATA_DIR = '/home/rax/mtp/droneRFb/extracted/twin_droneRF'
OUT_DIR = '/home/rax/mtp/droneRFb_spectrograms/test'
FFT_SIZE = 256
FS = 80e6
CHUNK = 500_000
MAX_CHUNKS = 8
IMG_SIZE = 640

# Build hot colormap LUT
lut = np.zeros((256, 3), dtype=np.uint8)
for i in range(256):
    t = i / 255.0
    lut[i] = [min(255, int(t * 3 * 255)), min(255, int(max(0, t * 3 - 1) * 255)), min(255, int(max(0, t * 3 - 2) * 255))]

def process_file(args):
    mat_path, cls, out_dir = args
    try:
        f = h5py.File(mat_path, 'r')
        I = np.array(f['I']).flatten()
        Q = np.array(f['Q']).flatten()
        f.close()
        iq = I + 1j * Q
        iq = iq - np.mean(iq)

        n_chunks = min(len(iq) // CHUNK, MAX_CHUNKS)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(mat_path))[0]
        count = 0
        for i in range(n_chunks):
            outpath = os.path.join(out_dir, f'{base}_s{i}.png')
            if os.path.exists(outpath):
                count += 1
                continue
            seg = iq[i*CHUNK:(i+1)*CHUNK]
            _, _, Zxx = scipy_signal.stft(seg, fs=FS, nperseg=FFT_SIZE, noverlap=FFT_SIZE//2, window='hamming', return_onesided=False)
            Zxx = np.fft.fftshift(Zxx, axes=0)
            mag = 10 * np.log10(np.abs(Zxx)**2 + 1e-10)
            vmin, vmax = np.percentile(mag, [2, 98])
            norm = np.clip((mag - vmin) / (vmax - vmin + 1e-10), 0, 1)
            gray = (norm * 255).astype(np.uint8)
            rgb = lut[gray]
            img = Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(outpath)
            count += 1
        return count
    except Exception as e:
        return 0

# Parse test_labels.txt
label_map = {}
with open(os.path.join(DATA_DIR, 'test_labels.txt')) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            orig, idx = parts
            m = re.match(r'^([A-G]\d?)', orig)
            cls = m.group(1) if m else 'B'
            label_map[idx + '.mat'] = cls

print(f"Parsed {len(label_map)} test labels", flush=True)

tasks = []
test_dir = os.path.join(DATA_DIR, 'test')
for fname in sorted(os.listdir(test_dir)):
    if not fname.endswith('.mat'):
        continue
    cls = label_map.get(fname, 'UNKNOWN')
    mat_path = os.path.join(test_dir, fname)
    out = os.path.join(OUT_DIR, cls)
    tasks.append((mat_path, cls, out))

print(f"{len(tasks)} test files, 16 workers", flush=True)
total = 0
done = 0
with ProcessPoolExecutor(max_workers=16) as ex:
    futures = {ex.submit(process_file, t): t for t in tasks}
    for fut in as_completed(futures):
        n = fut.result()
        total += n
        done += 1
        if done % 200 == 0:
            print(f"  [{done}/{len(tasks)}] {total} spectrograms", flush=True)

print(f"\nTotal: {total} test spectrograms → {OUT_DIR}", flush=True)
for cls in sorted(os.listdir(OUT_DIR)):
    d = os.path.join(OUT_DIR, cls)
    if os.path.isdir(d):
        print(f"  {cls}: {len(os.listdir(d))}", flush=True)
print("=== TEST SPECGEN COMPLETE ===", flush=True)
