#!/usr/bin/env python3
"""Extract statistical features from DroneRFb .mat files."""
import h5py, numpy as np, os, json, re
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA = '/home/rax/mtp/droneRFb/extracted/twin_droneRF'
OUT = '/home/rax/mtp/results'

def extract_features(iq):
    """17 baseline + 20 extended = 37 features."""
    feat = []
    power = np.abs(iq)**2
    feat.extend([np.mean(power), np.std(power), np.max(power), np.min(power)])
    fft_vals = np.fft.fft(iq[:32768])
    fft_power = np.abs(fft_vals)**2
    feat.extend([np.mean(fft_power), np.std(fft_power), np.max(fft_power)])
    feat.append(np.argmax(fft_power) / len(fft_power))
    I, Q = np.real(iq), np.imag(iq)
    feat.extend([np.mean(I), np.std(I), np.mean(Q), np.std(Q)])
    phase = np.angle(iq[:32768])
    feat.extend([np.mean(phase), np.std(phase)])
    pd = np.diff(phase)
    feat.extend([np.mean(pd), np.std(pd)])
    bw = np.sum(fft_power > np.max(fft_power)*0.1) / len(fft_power)
    feat.append(bw)
    amp = np.abs(iq)
    feat.extend([kurtosis(amp[:32768]), skew(amp[:32768]), np.percentile(amp,10), np.percentile(amp,90)])
    feat.append(np.max(amp)/(np.mean(amp)+1e-10))
    f, t, Zxx = scipy_signal.stft(iq[:32768], fs=80e6, nperseg=256, noverlap=128)
    mag = np.abs(Zxx)
    sc = np.sum(np.abs(f[:,None])*mag, axis=0)/(np.sum(mag,axis=0)+1e-10)
    feat.extend([np.mean(sc), np.std(sc)])
    sf = np.exp(np.mean(np.log(mag+1e-10),axis=0)) / (np.mean(mag,axis=0)+1e-10)
    feat.extend([np.mean(sf), np.std(sf)])
    return np.nan_to_num(np.array(feat, dtype=np.float32))

def process_file(args):
    fpath, cls = args
    try:
        f = h5py.File(fpath, 'r')
        I = np.array(f['I']).flatten().astype(np.float32)
        Q = np.array(f['Q']).flatten().astype(np.float32)
        f.close()
        iq = I + 1j*Q
        iq -= np.mean(iq)
        feats = extract_features(iq)
        return feats, cls
    except:
        return None, cls

# Parse labels
train_labels = {}
with open(os.path.join(DATA, 'train_labels.txt')) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            fn, cls = parts
            # Collapse to type: A1→A, B→B
            dtype = re.match(r'^([A-G])', cls).group(1)
            train_labels[fn] = dtype

test_labels = {}
with open(os.path.join(DATA, 'test_labels.txt')) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            orig, idx = parts
            m = re.match(r'^([A-G])', orig)
            dtype = m.group(1) if m else 'B'
            test_labels[idx + '.mat'] = dtype

print(f"Train: {len(train_labels)}, Test: {len(test_labels)}", flush=True)

# Extract features
for split, labels, dirname in [('train', train_labels, 'train'), ('test', test_labels, 'test')]:
    print(f"\nExtracting {split}...", flush=True)
    tasks = []
    mat_dir = os.path.join(DATA, dirname)
    for fn in sorted(os.listdir(mat_dir)):
        if not fn.endswith('.mat'): continue
        cls = labels.get(fn, 'UNKNOWN')
        tasks.append((os.path.join(mat_dir, fn), cls))

    X, y = [], []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for feat, cls in ex.map(process_file, tasks):
            if feat is not None:
                X.append(feat)
                y.append(cls)
    
    if split == 'train':
        X_train, y_train = np.array(X), np.array(y)
    else:
        X_test, y_test = np.array(X), np.array(y)
    print(f"  {split}: {len(X)} samples, {len(set(y))} classes", flush=True)

# Train classifiers
print("\nTraining classifiers...", flush=True)
results = {}
for name, clf in [('RF', RandomForestClassifier(200, n_jobs=-1, random_state=42)),
                   ('GBM', GradientBoostingClassifier(n_estimators=100, random_state=42))]:
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    print(f"  {name}: acc={acc:.3f} f1={f1:.3f}", flush=True)
    print(classification_report(y_test, preds), flush=True)
    results[name] = {'accuracy': acc, 'f1_macro': f1}

with open(os.path.join(OUT, 'droneRFb_stat_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("=== DONE ===", flush=True)
