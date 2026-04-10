#!/usr/bin/env python3
"""Generate spectrograms from DroneRFa .mat files (dual-receiver)."""
import h5py, numpy as np, os, re, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import signal as scipy_signal
from PIL import Image
warnings.filterwarnings('ignore')

DATA = '/home/rax/mtp/droneRFa/extracted/DroneRFa'
OUT = '/home/rax/mtp/droneRFa_spectrograms'
FFT, FS, CHUNK, MAX_C, IMG = 256, 100e6, 1_000_000, 8, 640

lut = np.zeros((256,3), dtype=np.uint8)
for i in range(256):
    t = i/255.0
    lut[i] = [min(255,int(t*3*255)), min(255,int(max(0,t*3-1)*255)), min(255,int(max(0,t*3-2)*255))]

def process_file(args):
    fpath, tcode, split = args
    try:
        f = h5py.File(fpath, 'r')
        I = np.array(f['RF0_I']).flatten()[:80_000_000].astype(np.float32)
        Q = np.array(f['RF0_Q']).flatten()[:80_000_000].astype(np.float32)
        f.close()
        iq = I + 1j*Q; iq -= np.mean(iq)
        n_c = min(len(iq)//CHUNK, MAX_C)
        out_dir = os.path.join(OUT, split, tcode)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(fpath))[0]
        count = 0
        for i in range(n_c):
            op = os.path.join(out_dir, f'{base}_s{i}.png')
            if os.path.exists(op): count += 1; continue
            seg = iq[i*CHUNK:(i+1)*CHUNK]
            _,_,Zxx = scipy_signal.stft(seg, fs=FS, nperseg=FFT, noverlap=FFT//2, window='hamming', return_onesided=False)
            Zxx = np.fft.fftshift(Zxx, axes=0)
            mag = 10*np.log10(np.abs(Zxx)**2+1e-10)
            vmin,vmax = np.percentile(mag,[2,98])
            gray = np.clip((mag-vmin)/(vmax-vmin+1e-10)*255, 0, 255).astype(np.uint8)
            Image.fromarray(lut[gray]).resize((IMG,IMG), Image.LANCZOS).save(op)
            count += 1
        return count
    except Exception as e:
        return 0

files = sorted([f for f in os.listdir(DATA) if f.endswith('.mat')])
# Parse T-code as class, 80/20 split
tcodes = {}
for fn in files:
    m = re.match(r'(T\d+)', fn)
    tc = m.group(1) if m else 'UNK'
    tcodes.setdefault(tc, []).append(fn)

tasks = []
for tc, flist in sorted(tcodes.items()):
    n_train = max(1, int(len(flist)*0.8))
    for i, fn in enumerate(flist):
        split = 'train' if i < n_train else 'val'
        tasks.append((os.path.join(DATA, fn), tc, split))

print(f"DroneRFa: {len(files)} files, {len(tcodes)} T-codes, {len(tasks)} tasks", flush=True)
total, done = 0, 0
with ProcessPoolExecutor(max_workers=4) as ex:  # 4 workers (files are huge ~5GB each)
    for fut in [ex.submit(process_file, t) for t in tasks]:
        n = fut.result(); total += n; done += 1
        if done % 20 == 0: print(f"  [{done}/{len(tasks)}] {total} specs", flush=True)

for s in ['train','val']:
    d = os.path.join(OUT, s)
    if os.path.exists(d):
        n = sum(len(os.listdir(os.path.join(d,c))) for c in os.listdir(d) if os.path.isdir(os.path.join(d,c)))
        print(f"{s}: {n} images", flush=True)
print(f"Total: {total} spectrograms", flush=True)
print("=== DRONERF-A SPECGEN COMPLETE ===", flush=True)
