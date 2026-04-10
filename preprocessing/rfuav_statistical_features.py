#!/usr/bin/env python3
"""
RFUAV Statistical Feature Extraction and Classification
========================================================
Extracts statistical features from raw RFUAV IQ data (.iq files, binary float32
interleaved I/Q, 100 MSps) and trains Random Forest / Gradient Boosting / MLP
classifiers across multiple feature modalities.

Data layout:
  /home/rax/mtp/raw/{drone_name}/{drone_name}/VTSBW={bw}/*.iq
  - 37 drone types, 356 .iq files total
  - Each .iq file ~763 MB = 100M complex samples (1 second at 100 MSps)

Processing:
  - Segment each file into 10 chunks of 10M samples (0.1 s each)
  - Extract features per chunk; label = drone type
  - 80/20 temporal split per drone, then classify

Feature modalities (matching rfml_comparison.py extractors):
  baseline      17 features
  iq_stat       37 features
  spectrogram   37 features
  hos           20 features
  combined      all concatenated

Classifiers: RandomForest(200), GradientBoosting(100), MLP(256,128)

Outputs:
  /home/rax/mtp/rfuav_features.npz
  /home/rax/mtp/results/rfuav_statistical_results.json
"""

import os
import sys
import json
import time
import warnings
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_ROOT = "/home/rax/mtp/raw"
FEATURES_OUT   = "/home/rax/mtp/rfuav_features.npz"
RESULTS_OUT    = "/home/rax/mtp/results/rfuav_statistical_results.json"

SAMPLES_PER_FILE  = 100_000_000   # 100 MSps × 1 s
CHUNKS_PER_FILE   = 10
CHUNK_SIZE        = SAMPLES_PER_FILE // CHUNKS_PER_FILE   # 10 M samples
HOS_SUBSAMPLE     = 32_768        # for HOS computation speed
N_WORKERS         = 16
TRAIN_RATIO       = 0.80


# ===========================================================================
# FEATURE EXTRACTORS
# ===========================================================================

def _baseline_features(samples: np.ndarray) -> np.ndarray:
    """Baseline RTL-ML 17 features."""
    features = []

    power = np.abs(samples) ** 2
    features.extend([np.mean(power), np.std(power), np.max(power), np.min(power)])

    fft_vals  = np.fft.fft(samples)
    fft_power = np.abs(fft_vals) ** 2
    features.extend([np.mean(fft_power), np.std(fft_power), np.max(fft_power)])
    features.append(float(np.argmax(fft_power)) / len(fft_power))

    i_s = np.real(samples)
    q_s = np.imag(samples)
    features.extend([np.mean(i_s), np.std(i_s), np.mean(q_s), np.std(q_s)])

    phase = np.angle(samples)
    features.extend([np.mean(phase), np.std(phase)])

    phase_diff = np.diff(phase)
    features.extend([np.mean(phase_diff), np.std(phase_diff)])

    bandwidth = np.sum(fft_power > np.max(fft_power) * 0.1)
    features.append(bandwidth / len(fft_power))

    return np.array(features, dtype=np.float64)  # 17


def _iq_stat_features(samples: np.ndarray) -> np.ndarray:
    """IQ Statistical 37 features (matching RFML IQ expert pathway)."""
    features = []

    i_s       = np.real(samples)
    q_s       = np.imag(samples)
    amplitude = np.abs(samples)
    phase     = np.angle(samples)
    inst_freq = np.diff(np.unwrap(phase))

    # Amplitude statistics (8)
    features.extend([
        np.mean(amplitude),
        np.std(amplitude),
        np.median(amplitude),
        float(kurtosis(amplitude)),
        float(skew(amplitude)),
        float(np.percentile(amplitude, 10)),
        float(np.percentile(amplitude, 90)),
        float(np.max(amplitude) / (np.mean(amplitude) + 1e-10)),  # crest factor
    ])

    # I channel statistics (4)
    features.extend([
        np.mean(i_s), np.std(i_s),
        float(kurtosis(i_s)), float(skew(i_s)),
    ])

    # Q channel statistics (4)
    features.extend([
        np.mean(q_s), np.std(q_s),
        float(kurtosis(q_s)), float(skew(q_s)),
    ])

    # Phase statistics (4)
    features.extend([
        np.mean(phase), np.std(phase),
        float(kurtosis(phase)), float(skew(phase)),
    ])

    # Instantaneous frequency statistics (5)
    features.extend([
        np.mean(inst_freq), np.std(inst_freq),
        float(kurtosis(inst_freq)), float(skew(inst_freq)),
        float(np.median(inst_freq)),
    ])

    # Zero-crossing rates (2)
    i_zc = np.sum(np.diff(np.sign(i_s)) != 0) / len(i_s)
    q_zc = np.sum(np.diff(np.sign(q_s)) != 0) / len(q_s)
    features.extend([float(i_zc), float(q_zc)])

    # Autocorrelation at lags 1, 10, 50, 100 (4)
    amp_slice = amplitude[:1024]
    autocorr  = np.correlate(amp_slice, amp_slice, mode='full')
    autocorr  = autocorr[len(autocorr) // 2:]
    autocorr  = autocorr / (autocorr[0] + 1e-10)
    features.extend([autocorr[1], autocorr[10], autocorr[50], autocorr[100]])

    # Envelope statistics via Hilbert (2)
    analytic = np.abs(scipy_signal.hilbert(i_s[:4096]))
    features.extend([np.mean(analytic), np.std(analytic)])

    # Percentiles of amplitude (4 → makes 37 total above + 4 here ... recount)
    # Running total so far: 8+4+4+4+5+2+4+2 = 33; need 4 more to reach 37
    features.extend([
        float(np.percentile(amplitude, 25)),
        float(np.percentile(amplitude, 50)),
        float(np.percentile(amplitude, 75)),
        float(np.percentile(amplitude, 99)),
    ])

    return np.array(features, dtype=np.float64)  # 37


def _spectrogram_stat_features(samples: np.ndarray) -> np.ndarray:
    """Spectrogram Statistical 37 features (matching RFML spectrogram expert)."""
    features = []

    # STFT — use a modest nperseg that works on 10 M samples
    nperseg  = 512
    noverlap = 256
    # For 10 M samples full STFT is large; use first 200 K for speed
    seg = samples[:200_000]
    f, _t, Zxx = scipy_signal.stft(seg, fs=100e6, nperseg=nperseg, noverlap=noverlap)

    mag   = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Log-magnitude global stats (5)
    log_mag = np.log1p(mag)
    features.extend([
        float(np.mean(log_mag)),
        float(np.std(log_mag)),
        float(np.max(log_mag)),
        float(kurtosis(log_mag.ravel())),
        float(skew(log_mag.ravel())),
    ])

    # Spectral centroid mean/std (2)
    spectral_centroid = (np.sum(f[:, np.newaxis] * mag, axis=0)
                         / (np.sum(mag, axis=0) + 1e-10))
    features.extend([float(np.mean(spectral_centroid)), float(np.std(spectral_centroid))])

    # Spectral bandwidth mean/std (2)
    spectral_bw = np.sqrt(
        np.sum(((f[:, np.newaxis] - spectral_centroid[np.newaxis, :]) ** 2) * mag, axis=0)
        / (np.sum(mag, axis=0) + 1e-10)
    )
    features.extend([float(np.mean(spectral_bw)), float(np.std(spectral_bw))])

    # Spectral rolloff 85% mean/std (2)
    cumsum    = np.cumsum(mag, axis=0)
    total     = cumsum[-1:, :]
    roll_idx  = np.argmax(cumsum >= 0.85 * total, axis=0)
    roll_idx  = np.clip(roll_idx, 0, len(f) - 1)
    spectral_rolloff = f[roll_idx]
    features.extend([float(np.mean(spectral_rolloff)), float(np.std(spectral_rolloff))])

    # Spectral flatness mean/std (2)
    geo_mean  = np.exp(np.mean(np.log(mag + 1e-10), axis=0))
    arith_mean = np.mean(mag, axis=0)
    flatness  = geo_mean / (arith_mean + 1e-10)
    features.extend([float(np.mean(flatness)), float(np.std(flatness))])

    # Phase stats (4)
    features.extend([
        float(np.mean(phase)),
        float(np.std(phase)),
        float(kurtosis(phase.ravel())),
        float(skew(phase.ravel())),
    ])

    # Instantaneous frequency stats from phase derivative (4)
    inst_freq = np.diff(np.unwrap(phase, axis=1), axis=1)
    features.extend([
        float(np.mean(inst_freq)),
        float(np.std(inst_freq)),
        float(kurtosis(inst_freq.ravel())),
        float(skew(inst_freq.ravel())),
    ])

    # Temporal envelope stats (4)
    temporal_envelope = np.mean(mag, axis=0)
    features.extend([
        float(np.mean(temporal_envelope)),
        float(np.std(temporal_envelope)),
        float(kurtosis(temporal_envelope)),
        float(skew(temporal_envelope)),
    ])

    # Band energies — 8 bands (8)
    n_bands   = 8
    band_size = mag.shape[0] // n_bands
    for i in range(n_bands):
        band = mag[i * band_size:(i + 1) * band_size, :]
        features.append(float(np.mean(band)))

    # Spectral contrast — 4 bands (4)
    for i in range(4):
        band  = mag[i * band_size:(i + 1) * band_size, :]
        peak  = np.max(band, axis=0)
        valley = np.min(band, axis=0)
        features.append(float(np.mean(peak - valley)))

    # Running total: 5+2+2+2+2+4+4+4+8+4 = 37
    return np.array(features, dtype=np.float64)  # 37


def _hos_features(samples: np.ndarray) -> np.ndarray:
    """Higher-Order Statistics / Cumulants 20 features (matching RFML HOS expert)."""
    if len(samples) > HOS_SUBSAMPLE:
        samples = samples[:HOS_SUBSAMPLE]

    samples = samples / (np.sqrt(np.mean(np.abs(samples) ** 2)) + 1e-10)

    # Second-order moments / cumulants
    C20 = np.mean(samples ** 2)
    C21 = np.mean(np.abs(samples) ** 2)

    # Fourth-order
    M40 = np.mean(samples ** 4)
    M41 = np.mean(samples ** 3 * np.conj(samples))
    M42 = np.mean((np.abs(samples) ** 2) ** 2)
    M20 = np.mean(samples ** 2)
    M21 = np.mean(np.abs(samples) ** 2)

    C40 = M40 - 3 * M20 ** 2
    C41 = M41 - 3 * M21 * M20
    C42 = M42 - np.abs(M20) ** 2 - 2 * M21 ** 2

    # Sixth-order
    M60 = np.mean(samples ** 6)
    M61 = np.mean(samples ** 5 * np.conj(samples))
    M62 = np.mean(samples ** 4 * np.conj(samples) ** 2)
    M63 = np.mean((np.abs(samples) ** 2) ** 3)

    C60 = M60 - 15 * M20 * M40 + 30 * M20 ** 3
    C61 = M61 - 5 * M21 * M40 - 10 * M20 * M41 + 30 * M20 ** 2 * M21
    C62 = (M62 - np.abs(M20) ** 2 * M42 - 8 * M21 * M41
           - M20 * np.conj(M40) + 6 * M21 ** 2 * M20 + 6 * M20 ** 2 * np.conj(M20))
    C63 = M63 - 9 * M21 * M42 + 12 * M21 ** 3

    cumulants  = [C20, C21, C40, C41, C42, C60, C61, C62, C63]
    orders     = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
    norm_factor = np.abs(C21) ** (orders / 2) + 1e-10
    normalized  = np.array([np.abs(c) for c in cumulants]) / norm_factor  # 9

    features = normalized.tolist()

    # Derived ratios / kurtosis-like (4)
    features.append(float(np.abs(C42) / (np.abs(C21) ** 2 + 1e-10)))
    features.append(float(np.abs(C40) / (np.abs(C20) ** 2 + 1e-10)))
    features.append(float(np.abs(C63) / (np.abs(C21) ** 3 + 1e-10)))
    features.append(float(np.abs(C60) / (np.abs(C20) ** 3 + 1e-10)))

    # Phases of key cumulants (3)
    features.append(float(np.angle(C40)))
    features.append(float(np.angle(C42)))
    features.append(float(np.angle(C60)))

    # Ratios between cumulant pairs (4)
    features.append(float(np.abs(C40) / (np.abs(C42) + 1e-10)))
    features.append(float(np.abs(C60) / (np.abs(C63) + 1e-10)))
    features.append(float(np.abs(C41) / (np.abs(C42) + 1e-10)))
    features.append(float(np.abs(C61) / (np.abs(C62) + 1e-10)))

    # Total: 9 + 4 + 3 + 4 = 20
    return np.array(features, dtype=np.float64)  # 20


def extract_all_features(samples: np.ndarray) -> dict:
    """Extract all modalities from one chunk of complex IQ samples."""
    # DC removal
    samples = samples - np.mean(samples)

    baseline    = _baseline_features(samples)
    iq_stat     = _iq_stat_features(samples)
    spectrogram = _spectrogram_stat_features(samples)
    hos         = _hos_features(samples)
    combined    = np.concatenate([baseline, iq_stat, spectrogram, hos])

    return {
        'baseline':    baseline,
        'iq_stat':     iq_stat,
        'spectrogram': spectrogram,
        'hos':         hos,
        'combined':    combined,
    }


# ===========================================================================
# IQ FILE DISCOVERY
# ===========================================================================

def discover_iq_files(raw_root: str) -> list[tuple[str, str]]:
    """
    Walk raw_root and return list of (drone_name, filepath) pairs.
    Directory structure: raw_root/{drone_name}/{drone_name}/VTSBW=*/*.iq
    """
    entries = []
    raw_path = Path(raw_root)
    for drone_dir in sorted(raw_path.iterdir()):
        if not drone_dir.is_dir():
            continue
        drone_name = drone_dir.name
        # Recurse into nested structure
        for iq_file in sorted(drone_dir.rglob("*.iq")):
            entries.append((drone_name, str(iq_file)))
    print(f"[discover] Found {len(entries)} .iq files across "
          f"{len(set(e[0] for e in entries))} drone types")
    return entries


# ===========================================================================
# WORKER: process one .iq file → list of feature dicts
# ===========================================================================

def _process_one_file(args: tuple) -> list[dict]:
    """
    Worker function (runs in subprocess via ProcessPoolExecutor).
    Returns a list of dicts: {drone, modality_name: features_array, ...}
    """
    drone_name, filepath, chunk_size, chunks_per_file = args

    results = []
    try:
        # Read binary float32 interleaved I/Q
        raw = np.fromfile(filepath, dtype=np.float32)
        # Pair up I and Q
        n_complex = len(raw) // 2
        iq = raw[:n_complex * 2].view(np.complex64).astype(np.complex128)

        # Segment into chunks
        n_chunks = min(chunks_per_file, n_complex // chunk_size)
        for c in range(n_chunks):
            chunk = iq[c * chunk_size:(c + 1) * chunk_size]
            try:
                feats = extract_all_features(chunk)
                entry = {'drone': drone_name}
                for k, v in feats.items():
                    entry[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                results.append(entry)
            except Exception as e:
                print(f"[worker] chunk {c} of {filepath}: {e}", flush=True)

    except Exception as e:
        print(f"[worker] failed to read {filepath}: {e}", flush=True)

    return results


# ===========================================================================
# FEATURE EXTRACTION PIPELINE
# ===========================================================================

def extract_features_parallel(entries: list[tuple[str, str]]) -> dict:
    """
    Run feature extraction in parallel over all .iq files.
    Returns dict: {modality: np.ndarray shape (N, F)}, plus 'labels' array.
    """
    modalities = ['baseline', 'iq_stat', 'spectrogram', 'hos', 'combined']

    all_entries: list[dict] = []
    args_list = [
        (drone, path, CHUNK_SIZE, CHUNKS_PER_FILE)
        for drone, path in entries
    ]

    print(f"\n[extract] Processing {len(args_list)} files with {N_WORKERS} workers ...")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_process_one_file, a): a for a in args_list}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                rows = fut.result()
                all_entries.extend(rows)
            except Exception as e:
                print(f"[extract] future error: {e}", flush=True)
            if done % 10 == 0 or done == len(args_list):
                elapsed = time.time() - t0
                print(f"  [{done}/{len(args_list)}] {len(all_entries)} chunks  "
                      f"({elapsed:.0f}s elapsed)", flush=True)

    print(f"[extract] Done. Total chunks: {len(all_entries)}  "
          f"({time.time()-t0:.0f}s total)")

    if not all_entries:
        raise RuntimeError("No features extracted — check data paths.")

    # Assemble arrays
    labels = np.array([e['drone'] for e in all_entries])
    data   = {mod: np.stack([e[mod] for e in all_entries]) for mod in modalities}
    data['labels'] = labels
    return data


# ===========================================================================
# TEMPORAL TRAIN / VAL SPLIT
# ===========================================================================

def temporal_split(data: dict, modalities: list[str], train_ratio: float = TRAIN_RATIO):
    """
    Per-drone temporal split: first train_ratio chunks → train, rest → val.
    Returns dict: {modality: (X_train, X_val, y_train, y_val)}
    """
    labels    = data['labels']
    drones    = sorted(np.unique(labels))
    splits    = {}

    for mod in modalities:
        X = data[mod]
        X_tr, X_va = [], []
        y_tr, y_va = [], []
        for drone in drones:
            mask    = labels == drone
            X_cls   = X[mask]
            y_cls   = labels[mask]
            split   = max(1, int(len(X_cls) * train_ratio))
            X_tr.extend(X_cls[:split])
            X_va.extend(X_cls[split:])
            y_tr.extend(y_cls[:split])
            y_va.extend(y_cls[split:])
        splits[mod] = (
            np.array(X_tr), np.array(X_va),
            np.array(y_tr), np.array(y_va),
        )

    return splits


# ===========================================================================
# CLASSIFIERS
# ===========================================================================

def build_classifiers():
    return {
        'RandomForest':       RandomForestClassifier(
                                  n_estimators=200, n_jobs=-1,
                                  random_state=42, class_weight='balanced'),
        'GradientBoosting':   GradientBoostingClassifier(
                                  n_estimators=100, max_depth=5,
                                  random_state=42),
        'MLP':                MLPClassifier(
                                  hidden_layer_sizes=(256, 128),
                                  max_iter=300, random_state=42,
                                  early_stopping=True, validation_fraction=0.1),
    }


# ===========================================================================
# TRAIN + EVALUATE
# ===========================================================================

def train_and_evaluate(splits: dict, modalities: list[str]) -> dict:
    """
    Train every classifier × modality combination.
    Returns nested dict: results[modality][clf_name] = {accuracy, report, ...}
    """
    results  = defaultdict(dict)
    le       = LabelEncoder()

    # Fit label encoder on union of all training labels
    all_train_labels = np.concatenate([splits[m][2] for m in modalities])
    le.fit(all_train_labels)

    print(f"\n[train] {len(le.classes_)} classes, {len(modalities)} modalities × "
          f"{len(build_classifiers())} classifiers")

    for mod in modalities:
        X_tr, X_va, y_tr, y_va = splits[mod]
        y_tr_enc = le.transform(y_tr)
        y_va_enc = le.transform(y_va)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        print(f"\n  Modality: {mod}  "
              f"(train={len(X_tr)}, val={len(X_va)}, features={X_tr.shape[1]})")

        classifiers = build_classifiers()
        for clf_name, clf in classifiers.items():
            t0 = time.time()
            try:
                clf.fit(X_tr_s, y_tr_enc)
                y_pred    = clf.predict(X_va_s)
                acc       = float(accuracy_score(y_va_enc, y_pred))
                report    = classification_report(
                    y_va_enc, y_pred,
                    target_names=le.classes_, output_dict=True, zero_division=0)

                feat_imp = None
                if hasattr(clf, 'feature_importances_'):
                    feat_imp = clf.feature_importances_.tolist()

                results[mod][clf_name] = {
                    'accuracy':           acc,
                    'train_samples':      int(len(X_tr)),
                    'val_samples':        int(len(X_va)),
                    'n_features':         int(X_tr.shape[1]),
                    'classification_report': report,
                    'feature_importances':   feat_imp,
                    'elapsed_s':          round(time.time() - t0, 1),
                }
                print(f"    {clf_name:<20} acc={acc:.4f}  ({time.time()-t0:.1f}s)")

            except Exception as e:
                print(f"    {clf_name:<20} FAILED: {e}")
                traceback.print_exc()
                results[mod][clf_name] = {'error': str(e)}

    return dict(results)


# ===========================================================================
# PRINT COMPARISON TABLE
# ===========================================================================

def print_comparison_table(results: dict):
    """Print modality × classifier × accuracy table."""
    clf_names = sorted({cn for mod in results for cn in results[mod]})
    col_w     = 22
    hdr_w     = 16

    print("\n" + "=" * 80)
    print("COMPARISON TABLE: Modality × Classifier × Accuracy")
    print("=" * 80)
    header = f"{'Modality':<{hdr_w}}" + "".join(f"{c:<{col_w}}" for c in clf_names)
    print(header)
    print("-" * len(header))

    for mod, clf_results in results.items():
        row = f"{mod:<{hdr_w}}"
        for cn in clf_names:
            if cn in clf_results and 'accuracy' in clf_results[cn]:
                row += f"{clf_results[cn]['accuracy']:.4f}".ljust(col_w)
            else:
                row += "FAILED".ljust(col_w)
        print(row)

    print("=" * 80)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 80)
    print("RFUAV Statistical Feature Extraction + Classification")
    print("=" * 80)

    modalities = ['baseline', 'iq_stat', 'spectrogram', 'hos', 'combined']

    # ------------------------------------------------------------------
    # Step 1: Load or extract features
    # ------------------------------------------------------------------
    if os.path.exists(FEATURES_OUT):
        print(f"\n[main] Loading cached features from {FEATURES_OUT}")
        npz   = np.load(FEATURES_OUT, allow_pickle=True)
        data  = {k: npz[k] for k in npz.files}
    else:
        entries = discover_iq_files(RAW_DATA_ROOT)
        if not entries:
            print(f"ERROR: No .iq files found under {RAW_DATA_ROOT}")
            sys.exit(1)

        data = extract_features_parallel(entries)

        # Save
        os.makedirs(os.path.dirname(FEATURES_OUT), exist_ok=True)
        save_dict = {k: v for k, v in data.items()}
        np.savez_compressed(FEATURES_OUT, **save_dict)
        print(f"\n[main] Features saved to {FEATURES_OUT}")

    n_samples = len(data['labels'])
    drones    = np.unique(data['labels'])
    print(f"\n[main] Dataset: {n_samples} chunks, {len(drones)} drone types")
    for d in drones:
        print(f"  {d}: {np.sum(data['labels'] == d)} chunks")

    # ------------------------------------------------------------------
    # Step 2: Train / val split
    # ------------------------------------------------------------------
    splits = temporal_split(data, modalities)

    # ------------------------------------------------------------------
    # Step 3: Train and evaluate
    # ------------------------------------------------------------------
    results = train_and_evaluate(splits, modalities)

    # ------------------------------------------------------------------
    # Step 4: Print table
    # ------------------------------------------------------------------
    print_comparison_table(results)

    # ------------------------------------------------------------------
    # Step 5: Save results JSON
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)

    # Add metadata
    output = {
        'metadata': {
            'n_samples':       n_samples,
            'n_drone_types':   int(len(drones)),
            'drone_types':     drones.tolist(),
            'chunks_per_file': CHUNKS_PER_FILE,
            'chunk_size':      CHUNK_SIZE,
            'train_ratio':     TRAIN_RATIO,
            'modalities':      modalities,
        },
        'results': results,
    }

    # Convert numpy types for JSON serialisation
    def _json_clean(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(RESULTS_OUT, 'w') as f:
        json.dump(output, f, indent=2, default=_json_clean)

    print(f"\n[main] Results saved to {RESULTS_OUT}")
    print("[main] Done.")


if __name__ == '__main__':
    main()
