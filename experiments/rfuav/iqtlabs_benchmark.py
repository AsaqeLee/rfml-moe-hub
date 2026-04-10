#!/usr/bin/env python3
"""
IQTLabs Model Benchmark — MI300X ROCm
======================================
Benchmarks IQTLabs-inspired models against our existing results on:
  - RFUAV (37 drone classes, 356 .iq files, binary fp32 interleaved, 100 MSps)
  - DroneRFb (7 type-level classes A-G, HDF5 .mat, 80 MSps)

Models implemented from scratch (no IQTLabs repo imports):
  1. rfuav_net    — 1D CNN (RFClassification repo architecture)
  2. efficientnet — EfficientNet-B0 on 2-channel IQ (rfml train_iq.py style)
  3. psd_svm      — PSD (Welch) features + SVM
  4. vgg16_spec   — VGG16 + spectrogram transfer learning

Our published results for comparison:
  RFUAV 37-class:  LWMExpert 94.1%, MaxViT 97.8%, ConvNeXt 97.5%, Statistical RF 95.7%
  DroneRFb 7-type: ConvNeXt 92.0%, MaxViT 89.5%, SpectrogramExpert 90.4%, Statistical RF 42.3%

Run (on remote MI300X via ssh/sg):
  python iqtlabs_benchmark.py --dataset both --models all
  python iqtlabs_benchmark.py --dataset rfuav --models rfuav_net efficientnet --epochs 30
"""

import os
import sys
import json
import time
import glob
import math
import argparse
import warnings
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RAW_DIR       = '/home/rax/mtp/raw'
DRFB_IQ_DIR   = '/home/rax/mtp/droneRFb/extracted/twin_droneRF'
MODEL_DIR     = '/home/rax/mtp/models'
RESULT_DIR    = '/home/rax/mtp/results'
RESULT_FILE   = os.path.join(RESULT_DIR, 'iqtlabs_benchmark_results.json')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEGMENT_LEN   = 32768   # samples per segment (matches our expert benchmark)
VAL_SPLIT     = 0.2
MAX_SEGMENTS  = 100000  # cap total segments for RFUAV
SPEC_SIZE     = 224     # pixels for spectrogram images

# DroneRFb individual → type mapping (A1/A2 → A, etc.)
INDIVIDUAL_TO_TYPE = {
    'A1': 'A', 'A2': 'A',
    'B':  'B',
    'C1': 'C', 'C2': 'C',
    'D1': 'D', 'D2': 'D',
    'E1': 'E', 'E2': 'E',
    'F1': 'F', 'F2': 'F',
    'G1': 'G', 'G2': 'G',
}

ALL_MODELS = ['rfuav_net', 'efficientnet', 'psd_svm', 'vgg16_spec', 'yolov8n_cls', 'effnet_spec', 'logreg_pca']

# Our published results for comparison
OUR_RESULTS = {
    'rfuav': {
        'LWMExpert':       0.941,
        'MaxViT':          0.978,
        'ConvNeXt':        0.975,
        'Statistical RF':  0.957,
    },
    'droneRFb': {
        'ConvNeXt':            0.920,
        'MaxViT':              0.895,
        'SpectrogramExpert':   0.904,
        'Statistical RF':      0.423,
    },
}

DEFAULT_CFG = {
    'batch_size':      64,
    'epochs':          30,
    'lr':              1e-3,
    'weight_decay':    0.01,
    'label_smoothing': 0.1,
    'patience':        15,
    'num_workers':     4,
    'amp':             True,
}


# ============================================================================
# RFUAV DATA LOADING
# ============================================================================

def discover_rfuav_files():
    """Return list of (file_path, drone_name) for all RFUAV .iq files."""
    entries = []
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")
    for drone_name in sorted(os.listdir(RAW_DIR)):
        drone_top = os.path.join(RAW_DIR, drone_name)
        if not os.path.isdir(drone_top):
            continue
        pattern1 = os.path.join(drone_top, drone_name, 'VTSBW=*', '*.iq')
        pattern2 = os.path.join(drone_top, 'VTSBW=*', '*.iq')
        pattern3 = os.path.join(drone_top, drone_name, '*.iq')
        iq_files = sorted(glob.glob(pattern1))
        if not iq_files:
            iq_files = sorted(glob.glob(pattern2))
        if not iq_files:
            iq_files = sorted(glob.glob(pattern3))
        for f in iq_files:
            entries.append((f, drone_name))
    return entries


def build_rfuav_segments(segment_length=SEGMENT_LEN, val_split=VAL_SPLIT,
                         max_segments=MAX_SEGMENTS):
    """
    Build train/val segment lists for RFUAV. Temporal split 80/20 per drone.
    Caps total segments to max_segments proportionally across classes.
    Returns (train_segs, train_labels, val_segs, val_labels, class_names).
    """
    file_entries = discover_rfuav_files()
    if not file_entries:
        raise RuntimeError(f"No .iq files found under {RAW_DIR}")

    drone_names  = sorted(set(dn for _, dn in file_entries))
    class_to_idx = {name: i for i, name in enumerate(drone_names)}
    print(f"  RFUAV: {len(file_entries)} .iq files, {len(drone_names)} classes")

    files_by_drone = {dn: [] for dn in drone_names}
    for fpath, dn in file_entries:
        files_by_drone[dn].append(fpath)

    train_segs, train_labels = [], []
    val_segs,   val_labels   = [], []

    for drone_name, flist in files_by_drone.items():
        label      = class_to_idx[drone_name]
        drone_segs = []
        for fpath in flist:
            file_size_bytes = os.path.getsize(fpath)
            n_samples       = file_size_bytes // (2 * 4)
            n_segments      = n_samples // segment_length
            for seg_idx in range(n_segments):
                drone_segs.append((fpath, seg_idx * segment_length))

        if not drone_segs:
            print(f"  WARNING: no segments for {drone_name}, skipping")
            continue

        n_total = len(drone_segs)
        n_train = max(1, int(n_total * (1.0 - val_split)))
        for seg in drone_segs[:n_train]:
            train_segs.append(seg)
            train_labels.append(label)
        for seg in drone_segs[n_train:]:
            val_segs.append(seg)
            val_labels.append(label)

    # Cap total segments proportionally
    total = len(train_segs) + len(val_segs)
    if total > max_segments:
        ratio = max_segments / total
        cap_train = max(1, int(len(train_segs) * ratio))
        cap_val   = max(1, int(len(val_segs)   * ratio))
        # Shuffle consistently before capping
        rng = np.random.default_rng(42)
        idx_tr = rng.permutation(len(train_segs))[:cap_train]
        idx_va = rng.permutation(len(val_segs))[:cap_val]
        train_segs   = [train_segs[i]   for i in idx_tr]
        train_labels = [train_labels[i] for i in idx_tr]
        val_segs     = [val_segs[i]     for i in idx_va]
        val_labels   = [val_labels[i]   for i in idx_va]

    print(f"  RFUAV segments: {len(train_segs):,} train, {len(val_segs):,} val")
    return train_segs, train_labels, val_segs, val_labels, drone_names


# ============================================================================
# DRONERF-B DATA LOADING (type-level, train on ind 1&2, test on ind 3)
# ============================================================================

def _read_label_file(label_file):
    """Read train_labels.txt / test_labels.txt → list of label strings."""
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels


def build_droneRFb_datasets(segment_length=SEGMENT_LEN):
    """
    Build DroneRFb IQ datasets using type-level labels (A-G).
    train/ contains individuals 1&2, test/ contains individual 3.
    Returns (train_ds, test_ds, class_names).
    """
    import h5py

    train_dir = os.path.join(DRFB_IQ_DIR, 'train')
    test_dir  = os.path.join(DRFB_IQ_DIR, 'test')

    def load_split(split_dir, is_train):
        label_file = os.path.join(split_dir, 'train_labels.txt' if is_train
                                  else 'test_labels.txt')
        mat_files  = sorted(glob.glob(os.path.join(split_dir, '*.mat')))

        # Determine type-level labels
        if os.path.exists(label_file):
            raw_labels = _read_label_file(label_file)
            # Map individual labels to type labels
            type_labels = [INDIVIDUAL_TO_TYPE.get(l, l) for l in raw_labels]
            assert len(type_labels) == len(mat_files), (
                f"Label count {len(type_labels)} != mat count {len(mat_files)}"
            )
            file_type_labels = list(zip(mat_files, type_labels))
        else:
            # Infer type from filename
            file_type_labels = []
            for mf in mat_files:
                base = os.path.splitext(os.path.basename(mf))[0]
                ind  = base.split('_')[0] if '_' in base else base
                typ  = INDIVIDUAL_TO_TYPE.get(ind, ind)
                file_type_labels.append((mf, typ))

        return file_type_labels

    train_ftl = load_split(train_dir, is_train=True)
    test_ftl  = load_split(test_dir,  is_train=False)

    all_types    = sorted(set(t for _, t in train_ftl + test_ftl))
    class_to_idx = {t: i for i, t in enumerate(all_types)}
    print(f"  DroneRFb: {len(all_types)} type classes: {all_types}")

    train_ds = DroneRFbIQDataset(train_ftl, class_to_idx, segment_length, is_train=True)
    test_ds  = DroneRFbIQDataset(test_ftl,  class_to_idx, segment_length, is_train=False)
    print(f"  DroneRFb: {len(train_ds):,} train segments, {len(test_ds):,} test segments")
    return train_ds, test_ds, all_types


class RFUAVIQDataset(Dataset):
    """RFUAV raw IQ segments. Returns (2, segment_length) tensor."""

    def __init__(self, segments, labels, segment_length, is_train=True):
        self.segments       = segments
        self.labels         = labels
        self.segment_length = segment_length
        self.is_train       = is_train

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        fpath, start = self.segments[idx]
        label        = self.labels[idx]
        offset_bytes = start * 2 * 4
        n_floats     = self.segment_length * 2

        raw = np.fromfile(fpath, dtype=np.float32, count=n_floats, offset=offset_bytes)
        if len(raw) < n_floats:
            raw = np.pad(raw, (0, n_floats - len(raw)))

        I = raw[0::2]
        Q = raw[1::2]
        I = I - I.mean()
        Q = Q - Q.mean()

        if self.is_train:
            I, Q = _iq_augment(I, Q, fs=100e6)

        x = np.stack([I, Q], axis=0).astype(np.float32)
        return torch.from_numpy(x), label


class DroneRFbIQDataset(Dataset):
    """DroneRFb IQ dataset from HDF5 .mat files. Returns (2, segment_length) tensor."""

    def __init__(self, file_type_labels, class_to_idx, segment_length, is_train=True):
        import h5py
        self.segment_length = segment_length
        self.class_to_idx   = class_to_idx
        self.is_train       = is_train
        self.segments       = []  # (mat_path, start, label_idx)

        for mf, type_label in file_type_labels:
            label = class_to_idx[type_label]
            try:
                with h5py.File(mf, 'r') as f:
                    for key in ['I', 'i', 'I_data']:
                        if key in f:
                            ds = f[key]
                            n_samples = ds.shape[-1] if ds.ndim >= 1 else 0
                            break
                    else:
                        first_key = list(f.keys())[0]
                        n_samples = f[first_key].shape[-1]
                n_segments = n_samples // segment_length
                for seg_idx in range(n_segments):
                    self.segments.append((mf, seg_idx * segment_length, label))
            except Exception as e:
                print(f"  WARNING: Failed to index {mf}: {e}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        import h5py
        mat_path, start, label = self.segments[idx]
        end = start + self.segment_length

        with h5py.File(mat_path, 'r') as f:
            I_data = Q_data = None
            for ik, qk in [('I', 'Q'), ('i', 'q'), ('I_data', 'Q_data')]:
                if ik in f and qk in f:
                    ds_i, ds_q = f[ik], f[qk]
                    I_data = ds_i[0, start:end] if ds_i.ndim == 2 else ds_i[start:end]
                    Q_data = ds_q[0, start:end] if ds_q.ndim == 2 else ds_q[start:end]
                    break
            if I_data is None:
                keys   = list(f.keys())
                ds0    = f[keys[0]]
                ds1    = f[keys[1]]
                I_data = ds0[0, start:end] if ds0.ndim == 2 else ds0[start:end]
                Q_data = ds1[0, start:end] if ds1.ndim == 2 else ds1[start:end]

        I_data = np.array(I_data, dtype=np.float32)
        Q_data = np.array(Q_data, dtype=np.float32)

        if len(I_data) < self.segment_length:
            I_data = np.pad(I_data, (0, self.segment_length - len(I_data)))
            Q_data = np.pad(Q_data, (0, self.segment_length - len(Q_data)))

        I_data = I_data - I_data.mean()
        Q_data = Q_data - Q_data.mean()

        if self.is_train:
            I_data, Q_data = _iq_augment(I_data, Q_data, fs=80e6)

        x = np.stack([I_data, Q_data], axis=0).astype(np.float32)
        return torch.from_numpy(x), label


class RFUAVSpectrogramDataset(Dataset):
    """RFUAV IQ → on-the-fly spectrogram. Returns (3, SPEC_SIZE, SPEC_SIZE)."""

    def __init__(self, segments, labels, segment_length, spec_size=SPEC_SIZE, is_train=True):
        self.segments       = segments
        self.labels         = labels
        self.segment_length = segment_length
        self.spec_size      = spec_size
        self.is_train       = is_train

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        fpath, start = self.segments[idx]
        label        = self.labels[idx]
        offset_bytes = start * 2 * 4
        n_floats     = self.segment_length * 2

        raw = np.fromfile(fpath, dtype=np.float32, count=n_floats, offset=offset_bytes)
        if len(raw) < n_floats:
            raw = np.pad(raw, (0, n_floats - len(raw)))

        I = raw[0::2] - raw[0::2].mean()
        Q = raw[1::2] - raw[1::2].mean()

        if self.is_train:
            I, Q = _iq_augment(I, Q, fs=100e6)

        spec = _iq_to_spectrogram(I + 1j * Q, self.spec_size)
        return torch.from_numpy(spec), label


class DroneRFbSpectrogramDataset(Dataset):
    """DroneRFb IQ → on-the-fly spectrogram. Returns (3, SPEC_SIZE, SPEC_SIZE)."""

    def __init__(self, file_type_labels, class_to_idx, segment_length,
                 spec_size=SPEC_SIZE, is_train=True):
        self._iq_ds = DroneRFbIQDataset(
            file_type_labels, class_to_idx, segment_length, is_train=is_train
        )
        self.spec_size = spec_size

    def __len__(self):
        return len(self._iq_ds)

    def __getitem__(self, idx):
        x, label = self._iq_ds[idx]
        I = x[0].numpy()
        Q = x[1].numpy()
        spec = _iq_to_spectrogram(I + 1j * Q, self.spec_size)
        return torch.from_numpy(spec), label


def _iq_augment(I, Q, fs=100e6):
    """RF augmentation: AWGN, CFO, time shift, amplitude scale."""
    if np.random.rand() < 0.8:
        snr_db    = np.random.uniform(0, 30)
        sig_power = np.mean(I**2 + Q**2)
        noise_pow = sig_power / (10 ** (snr_db / 10) + 1e-12)
        noise_std = math.sqrt(max(noise_pow, 0))
        I = I + noise_std * np.random.randn(len(I)).astype(np.float32)
        Q = Q + noise_std * np.random.randn(len(Q)).astype(np.float32)
    if np.random.rand() < 0.5:
        cfo_hz   = np.random.uniform(-500, 500)
        t        = np.arange(len(I), dtype=np.float32) / fs
        phase    = 2 * math.pi * cfo_hz * t
        cos_p    = np.cos(phase).astype(np.float32)
        sin_p    = np.sin(phase).astype(np.float32)
        I, Q     = I * cos_p - Q * sin_p, I * sin_p + Q * cos_p
    if np.random.rand() < 0.5:
        shift = np.random.randint(0, len(I))
        I, Q  = np.roll(I, shift), np.roll(Q, shift)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.5, 2.0)
        I, Q  = (I * scale).astype(np.float32), (Q * scale).astype(np.float32)
    return I.astype(np.float32), Q.astype(np.float32)


def _iq_to_spectrogram(iq_complex, spec_size):
    """Convert complex IQ to 3-channel spectrogram (mag, phase, inst_freq)."""
    from scipy.signal import stft as scipy_stft
    nperseg  = min(256, len(iq_complex) // 4)
    noverlap = nperseg // 2
    _, _, Zxx = scipy_stft(iq_complex, fs=100e6, nperseg=nperseg,
                           noverlap=noverlap, return_onesided=False)

    mag       = np.abs(Zxx).astype(np.float32)
    phase     = np.angle(Zxx).astype(np.float32)
    unwrapped = np.unwrap(np.angle(Zxx), axis=1)
    inst_freq = np.diff(unwrapped, axis=1, prepend=unwrapped[:, :1]).astype(np.float32)

    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        return np.zeros_like(arr) if (mx - mn) < 1e-8 else (arr - mn) / (mx - mn)

    channels = np.stack([_norm(mag), _norm(phase), _norm(inst_freq)], axis=0)
    t = torch.from_numpy(channels).unsqueeze(0)
    t = F.interpolate(t, size=(spec_size, spec_size), mode='bilinear', align_corners=False)
    return t.squeeze(0).numpy().astype(np.float32)


# ============================================================================
# MODEL 1: RFUAV-NET (1D CNN from RFClassification repo)
# ============================================================================

class RFUAVNet(nn.Module):
    """
    Simple 1D CNN from IQTLabs/RFClassification.
    Input: (B, 2, N)
    Architecture:
      Conv1d(2,128,k=7) -> BN -> ReLU -> MaxPool(2)
      Conv1d(128,128,k=5) -> BN -> ReLU -> MaxPool(2)
      Conv1d(128,128,k=3) -> BN -> ReLU -> MaxPool(2)
      Flatten -> Linear(128) -> Linear(num_classes)
    Reference: achieved 99.8% binary detection on DroneRF, 85.4% on DroneDetect.
    """

    def __init__(self, num_classes, segment_length=SEGMENT_LEN):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        # Compute flattened size after 3x MaxPool(2) on segment_length
        pool_out = segment_length // 8
        flat_dim = 128 * pool_out

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# MODEL 2: EFFICIENTNET-B0 ON IQ (from rfml train_iq.py)
# ============================================================================

class EfficientNetIQ(nn.Module):
    """
    EfficientNet-B0 adapted for 2-channel IQ input.
    Mimics IQTLabs/rfml train_iq.py:
      - Input: (B, 2, N) raw IQ
      - Reshape/project to (B, 3, H, W) for EfficientNet
      - Pretrained EfficientNet-B0 with drop_path=0.4, drop_rate=0.4
    Uses timm for EfficientNet-B0.
    """

    def __init__(self, num_classes, segment_length=SEGMENT_LEN):
        super().__init__()
        import timm

        # 1D projection: project (B,2,N) -> (B,3,sqrt(N),sqrt(N)) via 1D conv + reshape
        # We fold the 1D signal into a 2D grid for image-based backbone.
        # Using adaptive approach: N samples -> H x W where H*W = N
        side = int(math.isqrt(segment_length))
        # Adjust to nearest square that fits
        while side * side > segment_length:
            side -= 1
        self.side = side
        self.crop = side * side  # actual samples used

        # 1D conv projection to 3 channels before reshape
        self.proj = nn.Sequential(
            nn.Conv1d(2, 3, kernel_size=1, bias=False),
            nn.BatchNorm1d(3),
        )

        # EfficientNet-B0 with drop_path and drop_rate matching rfml train_iq.py
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=num_classes,
            drop_path_rate=0.4,
            drop_rate=0.4,
            in_chans=3,
        )

        # Patch first conv to accept our projected input (already 3-ch)
        # The backbone expects (B,3,H,W) which we provide directly

    def forward(self, x):
        # x: (B, 2, N)
        B, C, N = x.shape
        # Project 2-ch -> 3-ch
        x = self.proj(x)          # (B, 3, N)
        # Crop and reshape to 2D
        x = x[:, :, :self.crop]   # (B, 3, crop)
        x = x.view(B, 3, self.side, self.side)  # (B, 3, H, W)
        # Normalize to roughly ImageNet range
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-6)
        return self.backbone(x)


# ============================================================================
# MODEL 3: PSD + SVM (from RFClassification repo)
# ============================================================================

def compute_psd_features(I, Q, nfft=256):
    """Compute PSD via Welch method from I/Q arrays. Returns feature vector."""
    from scipy.signal import welch
    iq_complex = I + 1j * Q
    _, psd = welch(iq_complex, fs=1.0, nperseg=nfft, return_onesided=False)
    psd = np.abs(psd).astype(np.float32)
    # Log-scale (avoid log(0))
    psd = np.log1p(psd)
    return psd


def extract_psd_features_from_loader(dataset, nfft=256, max_samples=5000):
    """
    Extract PSD features from a dataset (IQ dataset expected).
    Returns (X, y) numpy arrays.
    """
    X, y = [], []
    indices = list(range(len(dataset)))
    if len(indices) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(indices, max_samples, replace=False).tolist()

    for i, idx in enumerate(indices):
        if i % 500 == 0:
            print(f"    Extracting PSD features: {i}/{len(indices)}", flush=True)
        x, label = dataset[idx]
        I = x[0].numpy()
        Q = x[1].numpy()
        feat = compute_psd_features(I, Q, nfft=nfft)
        X.append(feat)
        y.append(int(label))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ============================================================================
# MODEL 4: VGG16 + SPECTROGRAM TRANSFER LEARNING
# ============================================================================

class VGG16Spectrogram(nn.Module):
    """
    VGG16 pretrained on ImageNet, fine-tuned last layers for RF spectrogram classification.
    Input: (B, 3, SPEC_SIZE, SPEC_SIZE) spectrogram images.
    Only classifier layers are trained initially; feature layers are frozen.
    """

    def __init__(self, num_classes):
        super().__init__()
        import torchvision.models as tv_models

        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)

        # Freeze all feature layers
        for param in vgg.features.parameters():
            param.requires_grad = False

        # Replace classifier head
        vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

        self.model = vgg

    def unfreeze_features(self):
        """Unfreeze last 3 feature blocks for fine-tuning."""
        # VGG16 features: 0-30; unfreeze blocks 4 and 5 (indices 17-30)
        for i, layer in enumerate(self.model.features):
            if i >= 17:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)


# ============================================================================
# TRAINING LOOP (shared for all NN models)
# ============================================================================

def build_weighted_sampler(labels):
    """Build WeightedRandomSampler for imbalanced datasets."""
    labels_arr  = np.array(labels)
    class_counts = np.bincount(labels_arr)
    weights      = 1.0 / class_counts[labels_arr]
    sampler      = WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(labels),
        replacement=True,
    )
    return sampler


def train_nn_model(model, train_loader, val_loader, cfg, model_name, ckpt_suffix):
    """
    Generic training loop for all NN models.
    Returns dict with accuracy, f1, training metadata.
    """
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    scaler    = torch.amp.GradScaler(enabled=cfg['amp'])

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0
    train_start  = time.time()

    for epoch in range(cfg['epochs']):
        # ---- Train ----
        model.train()
        train_loss = train_correct = train_total = 0

        for x, labels in train_loader:
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=cfg['amp']):
                logits = model(x)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                        enabled=cfg['amp']):
                    logits = model(x)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc   = val_correct   / max(val_total,   1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
            marker = '*'
        else:
            no_improve += 1
            marker = ''

        if (epoch + 1) % 5 == 0 or marker == '*':
            elapsed = time.time() - train_start
            print(
                f"  [{model_name}] Epoch {epoch+1:3d}: "
                f"train={train_acc:.4f} val={val_acc:.4f} "
                f"loss={train_loss/max(train_total,1):.4f} "
                f"[{elapsed:.0f}s] {marker}",
                flush=True,
            )

        if no_improve >= cfg['patience']:
            print(f"  [{model_name}] Early stopping at epoch {epoch+1}", flush=True)
            break

    # Load best checkpoint
    if best_state:
        model.load_state_dict(best_state)
        ckpt_path = os.path.join(MODEL_DIR, f'iqtlabs_{ckpt_suffix}_best.pt')
        torch.save(best_state, ckpt_path)
        print(f"  [{model_name}] Saved checkpoint -> {ckpt_path}")

    return model, best_val_acc, time.time() - train_start, epoch + 1


def evaluate_nn(model, val_loader, class_names):
    """Full evaluation of NN model. Returns dict with metrics."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, labels in val_loader:
            x = x.to(DEVICE)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                logits = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc    = accuracy_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, output_dict=True)
    cm     = confusion_matrix(all_labels, all_preds).tolist()
    print(classification_report(all_labels, all_preds, target_names=class_names))
    return acc, f1, report, cm


# ============================================================================
# DRONERF-B TYPE ACCURACY HELPER
# ============================================================================

def compute_type_accuracy(all_labels, all_preds, class_names):
    """Collapse individual predictions to drone-type level."""
    type_labels = [INDIVIDUAL_TO_TYPE.get(class_names[l], class_names[l]) for l in all_labels]
    type_preds  = [INDIVIDUAL_TO_TYPE.get(class_names[p], class_names[p]) for p in all_preds]
    return accuracy_score(type_labels, type_preds)


# ============================================================================
# BENCHMARK RUNNERS
# ============================================================================

def run_rfuav_net(cfg, dataset_name, train_loader, val_loader, num_classes, class_names,
                  segment_length=SEGMENT_LEN):
    print(f"\n{'='*70}")
    print(f"  Model: RFUAV-Net (1D CNN) | Dataset: {dataset_name}")
    print(f"{'='*70}", flush=True)

    n_params = None
    try:
        model    = RFUAVNet(num_classes=num_classes, segment_length=segment_length)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params/1e6:.2f}M", flush=True)

        suffix = f"rfuav_net_{dataset_name}"
        model, best_val, train_time, epochs_done = train_nn_model(
            model, train_loader, val_loader, cfg, 'rfuav_net', suffix
        )
        acc, f1, report, cm = evaluate_nn(model, val_loader, class_names)

        result = {
            'model':          'rfuav_net',
            'dataset':        dataset_name,
            'accuracy':       float(acc),
            'f1_macro':       float(f1),
            'best_val_acc':   float(best_val),
            'params':         int(n_params),
            'train_time_sec': float(train_time),
            'epochs_trained': int(epochs_done),
            'per_class':      {cn: report[cn] for cn in class_names if cn in report},
            'confusion_matrix': cm,
        }
        print(f"\n  RFUAV-Net | {dataset_name}: Acc={acc:.4f} F1={f1:.4f}")
        return result

    except Exception as e:
        traceback.print_exc()
        return {'model': 'rfuav_net', 'dataset': dataset_name, 'error': str(e)}


def run_efficientnet_iq(cfg, dataset_name, train_loader, val_loader, num_classes, class_names,
                        segment_length=SEGMENT_LEN):
    print(f"\n{'='*70}")
    print(f"  Model: EfficientNet-B0 IQ | Dataset: {dataset_name}")
    print(f"{'='*70}", flush=True)

    try:
        model    = EfficientNetIQ(num_classes=num_classes, segment_length=segment_length)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params/1e6:.2f}M  (side={model.side})", flush=True)

        suffix = f"efficientnet_{dataset_name}"
        model, best_val, train_time, epochs_done = train_nn_model(
            model, train_loader, val_loader, cfg, 'efficientnet', suffix
        )
        acc, f1, report, cm = evaluate_nn(model, val_loader, class_names)

        result = {
            'model':          'efficientnet',
            'dataset':        dataset_name,
            'accuracy':       float(acc),
            'f1_macro':       float(f1),
            'best_val_acc':   float(best_val),
            'params':         int(n_params),
            'train_time_sec': float(train_time),
            'epochs_trained': int(epochs_done),
            'per_class':      {cn: report[cn] for cn in class_names if cn in report},
            'confusion_matrix': cm,
        }
        print(f"\n  EfficientNet-IQ | {dataset_name}: Acc={acc:.4f} F1={f1:.4f}")
        return result

    except Exception as e:
        traceback.print_exc()
        return {'model': 'efficientnet', 'dataset': dataset_name, 'error': str(e)}


def run_psd_svm(dataset_name, train_ds, val_ds, class_names, nfft=256, max_samples=5000):
    """Train PSD+SVM classifier. Works on CPU."""
    print(f"\n{'='*70}")
    print(f"  Model: PSD + SVM (RBF) | Dataset: {dataset_name}")
    print(f"{'='*70}", flush=True)

    try:
        t0 = time.time()
        print(f"  Extracting train PSD features (nfft={nfft}, max={max_samples})...",
              flush=True)
        X_train, y_train = extract_psd_features_from_loader(train_ds, nfft, max_samples)
        print(f"  Extracting val PSD features...", flush=True)
        X_val, y_val = extract_psd_features_from_loader(
            val_ds, nfft, max(1000, max_samples // 5)
        )
        print(f"  Train: {X_train.shape}  Val: {X_val.shape}", flush=True)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm',    SVC(kernel='rbf', C=10.0, gamma='scale',
                           decision_function_shape='ovr', verbose=False)),
        ])
        print(f"  Fitting SVM...", flush=True)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        acc    = accuracy_score(y_val, y_pred)
        f1     = f1_score(y_val, y_pred, average='macro')
        train_time = time.time() - t0

        print(classification_report(y_val, y_pred, target_names=class_names,
                                    labels=list(range(len(class_names)))))
        print(f"\n  PSD+SVM | {dataset_name}: Acc={acc:.4f} F1={f1:.4f}")

        result = {
            'model':          'psd_svm',
            'dataset':        dataset_name,
            'accuracy':       float(acc),
            'f1_macro':       float(f1),
            'train_time_sec': float(train_time),
            'nfft':           nfft,
            'n_train':        int(len(X_train)),
        }
        return result

    except Exception as e:
        traceback.print_exc()
        return {'model': 'psd_svm', 'dataset': dataset_name, 'error': str(e)}


def run_vgg16_spec(cfg, dataset_name, train_loader, val_loader, num_classes, class_names):
    """Train VGG16 + spectrogram. Frozen features initially, then fine-tune."""
    print(f"\n{'='*70}")
    print(f"  Model: VGG16 Spectrogram | Dataset: {dataset_name}")
    print(f"{'='*70}", flush=True)

    try:
        model    = VGG16Spectrogram(num_classes=num_classes)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {n_params/1e6:.2f}M total, {n_trainable/1e6:.2f}M trainable",
              flush=True)

        # Phase 1: train only classifier (frozen features)
        phase1_cfg = cfg.copy()
        phase1_cfg['epochs']  = min(10, cfg['epochs'] // 3)
        phase1_cfg['patience'] = max(5, cfg['patience'] // 3)
        suffix = f"vgg16_spec_{dataset_name}"

        model, best_val, train_time1, ep1 = train_nn_model(
            model, train_loader, val_loader, phase1_cfg, 'vgg16_spec_phase1', suffix + '_p1'
        )

        # Phase 2: unfreeze last feature blocks and fine-tune
        model.unfreeze_features()
        phase2_cfg = cfg.copy()
        phase2_cfg['lr']      = cfg['lr'] * 0.1
        phase2_cfg['epochs']  = cfg['epochs'] - phase1_cfg['epochs']
        phase2_cfg['patience'] = cfg['patience']

        model, best_val2, train_time2, ep2 = train_nn_model(
            model, train_loader, val_loader, phase2_cfg, 'vgg16_spec_phase2', suffix + '_p2'
        )

        best_val   = max(best_val, best_val2)
        train_time = train_time1 + train_time2
        acc, f1, report, cm = evaluate_nn(model, val_loader, class_names)

        result = {
            'model':          'vgg16_spec',
            'dataset':        dataset_name,
            'accuracy':       float(acc),
            'f1_macro':       float(f1),
            'best_val_acc':   float(best_val),
            'params':         int(n_params),
            'train_time_sec': float(train_time),
            'epochs_trained': int(ep1 + ep2),
            'per_class':      {cn: report[cn] for cn in class_names if cn in report},
            'confusion_matrix': cm,
        }
        print(f"\n  VGG16-Spec | {dataset_name}: Acc={acc:.4f} F1={f1:.4f}")
        return result

    except Exception as e:
        traceback.print_exc()
        return {'model': 'vgg16_spec', 'dataset': dataset_name, 'error': str(e)}


# ============================================================================
# MODEL 5: YOLOv8n-cls on spectrograms (IQTLabs rfml style)
# ============================================================================

def run_yolov8n_cls(cfg, dataset_name, spec_dir, num_classes, class_names):
    """Train YOLOv8n in classification mode on IQTLabs-style spectrograms."""
    print(f"\n{'='*70}")
    print(f"  Model: YOLOv8n-cls (IQTLabs style) | Dataset: {dataset_name}")
    print(f"{'='*70}", flush=True)
    try:
        from ultralytics import YOLO
        t0 = time.time()
        model = YOLO('yolov8n-cls.pt')
        results = model.train(
            data=spec_dir, epochs=cfg['epochs'], imgsz=224,
            batch=cfg.get('batch_size', 64), patience=cfg['patience'],
            project='/home/rax/mtp/models/iqtlabs_bench',
            name=f'yolov8n_{dataset_name}', device=0,
            optimizer='AdamW', lr0=1e-3, label_smoothing=0.1, verbose=False,
        )
        # Evaluate
        val_dir = os.path.join(spec_dir, 'val' if os.path.isdir(os.path.join(spec_dir, 'val')) else 'test')
        preds, labels = [], []
        classes_found = sorted(os.listdir(val_dir))
        for cls in classes_found:
            cls_dir = os.path.join(val_dir, cls)
            if not os.path.isdir(cls_dir): continue
            for img in sorted(os.listdir(cls_dir)):
                if not img.endswith('.png'): continue
                res = model.predict(os.path.join(cls_dir, img), verbose=False)
                preds.append(res[0].names[res[0].probs.top1])
                labels.append(cls)
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        elapsed = time.time() - t0
        print(f"  YOLOv8n-cls | {dataset_name}: Acc={acc:.4f} F1={f1:.4f} Time={elapsed:.0f}s")
        return {'model': 'yolov8n_cls', 'dataset': dataset_name, 'accuracy': float(acc),
                'f1_macro': float(f1), 'train_time_sec': float(elapsed), 'params': 3500000}
    except Exception as e:
        traceback.print_exc()
        return {'model': 'yolov8n_cls', 'dataset': dataset_name, 'error': str(e)}


# ============================================================================
# MODEL 6: EfficientNet-B0 on spectrograms (IQTLabs rfml spectrogram style)
# ============================================================================

class EfficientNetSpec(nn.Module):
    """EfficientNet-B0 on spectrogram images (IQTLabs rfml train_spec.py style)."""
    def __init__(self, num_classes=37):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True,
                                           num_classes=num_classes, drop_rate=0.4, drop_path_rate=0.4)

    def forward(self, x):
        return self.backbone(x)


def run_effnet_spec(cfg, dataset_name, train_loader, val_loader, num_classes, class_names):
    """Train EfficientNet-B0 on spectrograms (IQTLabs style)."""
    print(f"\n{'='*70}")
    print(f"  Model: EfficientNet-B0 Spectrogram (IQTLabs) | Dataset: {dataset_name}")
    print(f"{'='*70}", flush=True)
    try:
        model = EfficientNetSpec(num_classes=num_classes)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params/1e6:.2f}M", flush=True)
        suffix = f"effnet_spec_{dataset_name}"
        model, best_val, train_time, epochs_done = train_nn_model(
            model, train_loader, val_loader, cfg, 'effnet_spec', suffix
        )
        acc, f1, report, cm = evaluate_nn(model, val_loader, class_names)
        print(f"  EffNet-B0-Spec | {dataset_name}: Acc={acc:.4f} F1={f1:.4f}")
        return {'model': 'effnet_spec', 'dataset': dataset_name, 'accuracy': float(acc),
                'f1_macro': float(f1), 'best_val_acc': float(best_val), 'params': int(n_params),
                'train_time_sec': float(train_time), 'epochs_trained': int(epochs_done),
                'per_class': {cn: report[cn] for cn in class_names if cn in report}}
    except Exception as e:
        traceback.print_exc()
        return {'model': 'effnet_spec', 'dataset': dataset_name, 'error': str(e)}


# ============================================================================
# MODEL 7: Logistic Regression + PCA on PSD features
# ============================================================================

def run_logreg_pca(dataset_name, train_ds, val_ds, class_names, nfft=1024, max_samples=5000):
    """PSD features → PCA → Logistic Regression (IQTLabs RFClassification baseline)."""
    print(f"\n{'='*70}")
    print(f"  Model: LogReg+PCA on PSD | Dataset: {dataset_name}")
    print(f"{'='*70}", flush=True)
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from scipy.signal import welch

        def extract_psd(dataset, max_n):
            X, y = [], []
            indices = list(range(min(len(dataset), max_n)))
            for idx in indices:
                iq_tensor, label = dataset[idx]
                iq = iq_tensor.numpy()
                cplx = iq[0] + 1j * iq[1]
                _, psd = welch(cplx, nperseg=min(nfft, len(cplx)), return_onesided=False)
                X.append(np.log1p(np.abs(psd)).astype(np.float32))
                y.append(label if isinstance(label, int) else label.item())
            return np.array(X), np.array(y)

        print("  Extracting PSD features...", flush=True)
        X_train, y_train = extract_psd(train_ds, max_samples)
        X_val, y_val = extract_psd(val_ds, max_samples // 2)
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}", flush=True)

        n_components = min(50, X_train.shape[1], X_train.shape[0])
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('logreg', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ])
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        from sklearn.metrics import accuracy_score, f1_score, classification_report
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='macro')
        print(f"  LogReg+PCA | {dataset_name}: Acc={acc:.4f} F1={f1:.4f}")
        return {'model': 'logreg_pca', 'dataset': dataset_name, 'accuracy': float(acc),
                'f1_macro': float(f1), 'params': 0, 'n_components': n_components}
    except Exception as e:
        traceback.print_exc()
        return {'model': 'logreg_pca', 'dataset': dataset_name, 'error': str(e)}


# ============================================================================
# COMPARISON TABLE
# ============================================================================

def print_comparison_table(all_results):
    """Print combined comparison table against our published results."""
    print('\n' + '=' * 85)
    print('COMPARISON TABLE — IQTLabs Models vs Our Results')
    print('=' * 85)

    for ds_key, ds_label in [('rfuav', 'RFUAV (37-class)'), ('droneRFb', 'DroneRFb (7-type)')]:
        print(f"\n  {ds_label}")
        print(f"  {'Model':<28} {'Accuracy':>10} {'F1-macro':>10} {'Source':>12}")
        print(f"  {'-'*62}")

        # Our published results first
        for model_name, acc in sorted(OUR_RESULTS[ds_key].items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"  {'[Ours] ' + model_name:<28} {acc:>10.4f} {'N/A':>10} {'published':>12}")

        # IQTLabs results from this run
        ds_results = [r for r in all_results if r.get('dataset') == ds_key]
        ds_results.sort(key=lambda r: r.get('accuracy', 0.0), reverse=True)
        for r in ds_results:
            if 'error' in r:
                print(f"  {'[IQTLabs] ' + r['model']:<28} {'ERROR':>10}")
            else:
                acc = r.get('accuracy', float('nan'))
                f1  = r.get('f1_macro', float('nan'))
                print(f"  {'[IQTLabs] ' + r['model']:<28} {acc:>10.4f} {f1:>10.4f} {'this run':>12}")

    print('\n' + '=' * 85, flush=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='IQTLabs Model Benchmark — MI300X ROCm'
    )
    parser.add_argument('--dataset', choices=['rfuav', 'droneRFb', 'both'],
                        default='both', help='Dataset(s) to benchmark')
    parser.add_argument('--models', nargs='+', default=['all'],
                        choices=ALL_MODELS + ['all'],
                        help='Models to benchmark (default: all)')
    parser.add_argument('--batch-size',    type=int,   default=64)
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--max-segments',  type=int,   default=MAX_SEGMENTS,
                        help='Max total IQ segments for RFUAV (default: 100000)')
    parser.add_argument('--segment-length', type=int,  default=SEGMENT_LEN,
                        help='IQ samples per segment (default: 32768)')
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--num-workers',   type=int,   default=4)
    args = parser.parse_args()

    # Expand 'all' shorthand
    models_to_run = ALL_MODELS if 'all' in args.models else args.models

    cfg = DEFAULT_CFG.copy()
    cfg.update({
        'batch_size': args.batch_size,
        'epochs':     args.epochs,
        'lr':         args.lr,
        'num_workers': args.num_workers,
    })

    seg_len = args.segment_length

    print('=' * 70, flush=True)
    print('IQTLabs Model Benchmark — MI300X ROCm', flush=True)
    print(f'Device: {DEVICE}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU:  {torch.cuda.get_device_name(0)}', flush=True)
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB', flush=True)
    print(f'Models:      {models_to_run}', flush=True)
    print(f'Dataset(s):  {args.dataset}', flush=True)
    print(f'Batch size:  {cfg["batch_size"]}', flush=True)
    print(f'Epochs:      {cfg["epochs"]}', flush=True)
    print(f'Segment len: {seg_len}', flush=True)
    print(f'Max segs:    {args.max_segments}', flush=True)
    print('=' * 70, flush=True)

    all_results = []
    datasets_to_run = (
        ['rfuav', 'droneRFb'] if args.dataset == 'both' else [args.dataset]
    )

    # -----------------------------------------------------------------------
    # RFUAV
    # -----------------------------------------------------------------------
    if 'rfuav' in datasets_to_run:
        print('\n' + '=' * 70)
        print('  Building RFUAV datasets...')
        print('=' * 70, flush=True)

        train_segs, train_labels, val_segs, val_labels, class_names = \
            build_rfuav_segments(seg_len, VAL_SPLIT, args.max_segments)
        num_classes = len(class_names)

        # IQ datasets (for rfuav_net, efficientnet, psd_svm)
        train_iq_ds = RFUAVIQDataset(train_segs, train_labels, seg_len, is_train=True)
        val_iq_ds   = RFUAVIQDataset(val_segs,   val_labels,   seg_len, is_train=False)

        sampler = build_weighted_sampler(train_labels)
        train_iq_loader = DataLoader(
            train_iq_ds, batch_size=cfg['batch_size'], sampler=sampler,
            num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
        )
        val_iq_loader = DataLoader(
            val_iq_ds, batch_size=cfg['batch_size'], shuffle=False,
            num_workers=cfg['num_workers'], pin_memory=True,
        )

        # Spectrogram datasets (for vgg16_spec)
        if any(m in models_to_run for m in ['vgg16_spec', 'effnet_spec', 'yolov8n_cls']):
            print('  Building RFUAV spectrogram datasets...', flush=True)
            train_spec_ds = RFUAVSpectrogramDataset(
                train_segs, train_labels, seg_len, SPEC_SIZE, is_train=True
            )
            val_spec_ds = RFUAVSpectrogramDataset(
                val_segs, val_labels, seg_len, SPEC_SIZE, is_train=False
            )
            spec_sampler = build_weighted_sampler(train_labels)
            train_spec_loader = DataLoader(
                train_spec_ds, batch_size=cfg['batch_size'], sampler=spec_sampler,
                num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
            )
            val_spec_loader = DataLoader(
                val_spec_ds, batch_size=cfg['batch_size'], shuffle=False,
                num_workers=cfg['num_workers'], pin_memory=True,
            )

        # Run RFUAV models
        for model_key in models_to_run:
            if model_key == 'rfuav_net':
                result = run_rfuav_net(
                    cfg, 'rfuav', train_iq_loader, val_iq_loader,
                    num_classes, class_names, seg_len,
                )
                all_results.append(result)

            elif model_key == 'efficientnet':
                result = run_efficientnet_iq(
                    cfg, 'rfuav', train_iq_loader, val_iq_loader,
                    num_classes, class_names, seg_len,
                )
                all_results.append(result)

            elif model_key == 'psd_svm':
                result = run_psd_svm(
                    'rfuav', train_iq_ds, val_iq_ds, class_names,
                    nfft=256, max_samples=5000,
                )
                all_results.append(result)

            elif model_key == 'vgg16_spec':
                result = run_vgg16_spec(
                    cfg, 'rfuav', train_spec_loader, val_spec_loader,
                    num_classes, class_names,
                )
                all_results.append(result)

            elif model_key == 'yolov8n_cls':
                spec_dir = '/home/rax/mtp/spectrograms'
                result = run_yolov8n_cls(cfg, 'rfuav', spec_dir, num_classes, class_names)
                all_results.append(result)

            elif model_key == 'effnet_spec':
                result = run_effnet_spec(
                    cfg, 'rfuav', train_spec_loader, val_spec_loader,
                    num_classes, class_names,
                )
                all_results.append(result)

            elif model_key == 'logreg_pca':
                result = run_logreg_pca(
                    'rfuav', train_iq_ds, val_iq_ds, class_names,
                    nfft=1024, max_samples=5000,
                )
                all_results.append(result)

    # -----------------------------------------------------------------------
    # DroneRFb
    # -----------------------------------------------------------------------
    if 'droneRFb' in datasets_to_run:
        print('\n' + '=' * 70)
        print('  Building DroneRFb datasets...')
        print('=' * 70, flush=True)

        train_drfb_ds, val_drfb_ds, class_names_drfb = build_droneRFb_datasets(seg_len)
        num_classes_drfb = len(class_names_drfb)

        drfb_train_labels = [s[2] for s in train_drfb_ds.segments]
        drfb_sampler = build_weighted_sampler(drfb_train_labels)

        train_drfb_loader = DataLoader(
            train_drfb_ds, batch_size=cfg['batch_size'], sampler=drfb_sampler,
            num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
        )
        val_drfb_loader = DataLoader(
            val_drfb_ds, batch_size=cfg['batch_size'], shuffle=False,
            num_workers=cfg['num_workers'], pin_memory=True,
        )

        # Spectrogram loaders for vgg16_spec
        if any(m in models_to_run for m in ['vgg16_spec', 'effnet_spec', 'yolov8n_cls']):
            print('  Building DroneRFb spectrogram datasets...', flush=True)
            train_dir = os.path.join(DRFB_IQ_DIR, 'train')
            test_dir  = os.path.join(DRFB_IQ_DIR, 'test')

            def _load_ftl(split_dir, is_train):
                lf   = os.path.join(split_dir, 'train_labels.txt' if is_train else 'test_labels.txt')
                mats = sorted(glob.glob(os.path.join(split_dir, '*.mat')))
                if os.path.exists(lf):
                    raw_lbls  = _read_label_file(lf)
                    type_lbls = [INDIVIDUAL_TO_TYPE.get(l, l) for l in raw_lbls]
                    return list(zip(mats, type_lbls))
                return [(m, INDIVIDUAL_TO_TYPE.get(
                    os.path.splitext(os.path.basename(m))[0].split('_')[0], 'A'
                )) for m in mats]

            train_ftl_drfb = _load_ftl(train_dir, is_train=True)
            test_ftl_drfb  = _load_ftl(test_dir,  is_train=False)
            c2i_drfb = {t: i for i, t in enumerate(class_names_drfb)}

            train_drfb_spec_ds = DroneRFbSpectrogramDataset(
                train_ftl_drfb, c2i_drfb, seg_len, SPEC_SIZE, is_train=True
            )
            val_drfb_spec_ds = DroneRFbSpectrogramDataset(
                test_ftl_drfb, c2i_drfb, seg_len, SPEC_SIZE, is_train=False
            )
            drfb_spec_sampler = build_weighted_sampler(
                [s[2] for s in train_drfb_spec_ds._iq_ds.segments]
            )
            train_drfb_spec_loader = DataLoader(
                train_drfb_spec_ds, batch_size=cfg['batch_size'], sampler=drfb_spec_sampler,
                num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
            )
            val_drfb_spec_loader = DataLoader(
                val_drfb_spec_ds, batch_size=cfg['batch_size'], shuffle=False,
                num_workers=cfg['num_workers'], pin_memory=True,
            )

        # Run DroneRFb models
        for model_key in models_to_run:
            if model_key == 'rfuav_net':
                result = run_rfuav_net(
                    cfg, 'droneRFb', train_drfb_loader, val_drfb_loader,
                    num_classes_drfb, class_names_drfb, seg_len,
                )
                all_results.append(result)

            elif model_key == 'efficientnet':
                result = run_efficientnet_iq(
                    cfg, 'droneRFb', train_drfb_loader, val_drfb_loader,
                    num_classes_drfb, class_names_drfb, seg_len,
                )
                all_results.append(result)

            elif model_key == 'psd_svm':
                result = run_psd_svm(
                    'droneRFb', train_drfb_ds, val_drfb_ds, class_names_drfb,
                    nfft=256, max_samples=3000,
                )
                all_results.append(result)

            elif model_key == 'vgg16_spec':
                result = run_vgg16_spec(
                    cfg, 'droneRFb', train_drfb_spec_loader, val_drfb_spec_loader,
                    num_classes_drfb, class_names_drfb,
                )
                all_results.append(result)

            elif model_key == 'yolov8n_cls':
                spec_dir = '/home/rax/mtp/droneRFb_type_spectrograms'
                result = run_yolov8n_cls(cfg, 'droneRFb', spec_dir, num_classes_drfb, class_names_drfb)
                all_results.append(result)

            elif model_key == 'effnet_spec':
                result = run_effnet_spec(
                    cfg, 'droneRFb', train_drfb_spec_loader, val_drfb_spec_loader,
                    num_classes_drfb, class_names_drfb,
                )
                all_results.append(result)

            elif model_key == 'logreg_pca':
                result = run_logreg_pca(
                    'droneRFb', train_drfb_ds, val_drfb_ds, class_names_drfb,
                    nfft=1024, max_samples=3000,
                )
                all_results.append(result)

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    print_comparison_table(all_results)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        'timestamp':    time.strftime('%Y-%m-%dT%H:%M:%S'),
        'our_results':  OUR_RESULTS,
        'iqtlabs_results': all_results,
    }
    with open(RESULT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\nResults saved -> {RESULT_FILE}', flush=True)


if __name__ == '__main__':
    main()
