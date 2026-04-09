#!/usr/bin/env python3
"""
RFML-MoE Expert Benchmark — Individual Expert Evaluation on MI300X
====================================================================
Benchmarks each of the 11 RFML-MoE experts independently on RFUAV (37 classes)
and DroneRFb (7 drone-type classes) datasets.

Each expert exposes:
    forward(x) -> logits
    get_embedding(x) -> (B, 512)
    freeze() / unfreeze()
    num_params property

Experts are classified by input modality:
    Raw IQ experts     (input: B,2,32768):  IQExpert, TFMSExpert, HiWaveTSTExpert,
                                             LWMExpert, NeuroSymbolicRFFExpert,
                                             IQFormerExpert, MambaIQExpert
    Spectrogram experts (input: B,3,H,W):   SpectrogramExpert, SignalFormerRFExpert,
                                             VisualRFDetector
    Feature experts     (input: B,3,256,256): VMDGAFExpert

Usage (on MI300X remote via sg):
    sg render -c "python rfml_expert_benchmark.py --dataset both --experts all"
    sg render -c "python rfml_expert_benchmark.py --dataset rfuav --experts IQExpert TFMSExpert"
    sg render -c "python rfml_expert_benchmark.py --dataset droneRFb --experts SpectrogramExpert --epochs 30"

Output:
    /home/rax/mtp/results/rfml_benchmark/
        rfuav_{expert_name}_result.json
        droneRFb_{expert_name}_result.json
        benchmark_summary.json
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
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, confusion_matrix
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Add RFML expert codebase to path
# ---------------------------------------------------------------------------
RFML_ROOT = '/home/rax/mtp/rfml/new'
if RFML_ROOT not in sys.path:
    sys.path.insert(0, RFML_ROOT)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
MTP_DIR         = '/home/rax/mtp'
RAW_DIR         = os.path.join(MTP_DIR, 'raw')
SPEC_DIR        = os.path.join(MTP_DIR, 'spectrograms')
DRFB_IQ_DIR     = os.path.join(MTP_DIR, 'droneRFb', 'extracted', 'twin_droneRF')
DRFB_SPEC_DIR   = os.path.join(MTP_DIR, 'droneRFb_type_spectrograms')
RESULT_DIR      = os.path.join(MTP_DIR, 'results', 'rfml_benchmark')
MODEL_DIR       = os.path.join(MTP_DIR, 'models', 'rfml_benchmark')

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEGMENT_LEN = 32768  # N for raw IQ experts
SPEC_SIZE   = 224     # H=W for spectrogram experts
GAF_SIZE    = 256     # H=W for VMDGAFExpert

# Expert modality groups
IQ_EXPERTS = [
    'IQExpert', 'TFMSExpert', 'HiWaveTSTExpert', 'LWMExpert',
    'NeuroSymbolicRFFExpert', 'IQFormerExpert', 'MambaIQExpert',
]
SPEC_EXPERTS = ['SpectrogramExpert', 'SignalFormerRFExpert', 'VisualRFDetector']
GAF_EXPERTS  = ['VMDGAFExpert']
ALL_EXPERTS  = IQ_EXPERTS + SPEC_EXPERTS + GAF_EXPERTS

# Module name -> import path (relative to rfml package)
EXPERT_MODULES = {
    'IQExpert':                ('rfml.experts.iq_expert',              'IQExpert'),
    'TFMSExpert':              ('rfml.experts.tfms_expert',            'TFMSExpert'),
    'HiWaveTSTExpert':         ('rfml.experts.hiwavetst_expert',       'HiWaveTSTExpert'),
    'LWMExpert':               ('rfml.experts.lwm_expert',             'LWMExpert'),
    'NeuroSymbolicRFFExpert':  ('rfml.experts.neurosymbolic_rff_expert', 'NeuroSymbolicRFFExpert'),
    'IQFormerExpert':          ('rfml.experts.iqformer_expert',        'IQFormerExpert'),
    'MambaIQExpert':           ('rfml.experts.mamba_iq_expert',        'MambaIQExpert'),
    'SpectrogramExpert':       ('rfml.experts.spectrogram_expert',     'SpectrogramExpert'),
    'SignalFormerRFExpert':    ('rfml.experts.signalformer_expert',    'SignalFormerRFExpert'),
    'VisualRFDetector':        ('rfml.experts.visual_rf_detector',     'VisualRFDetector'),
    'VMDGAFExpert':            ('rfml.experts.vmd_gaf_expert',         'VMDGAFExpert'),
}


# ============================================================================
# DATASETS
# ============================================================================

class RFUAVIQDataset(Dataset):
    """
    RFUAV raw IQ dataset.
    Discovers .iq files under RAW_DIR/{drone_name}/{drone_name}/VTSBW=*/*.iq
    Binary fp32 interleaved I/Q at 100 MSps.
    Returns (2, SEGMENT_LEN) tensors.
    """

    def __init__(self, segments, labels, segment_length=SEGMENT_LEN, is_train=True):
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
            I, Q = self._augment(I, Q)

        x = np.stack([I, Q], axis=0).astype(np.float32)
        return torch.from_numpy(x), label

    def _augment(self, I, Q):
        if np.random.rand() < 0.8:
            snr_db    = np.random.uniform(0, 30)
            sig_power = np.mean(I**2 + Q**2)
            noise_pow = sig_power / (10 ** (snr_db / 10) + 1e-12)
            noise_std = math.sqrt(max(noise_pow, 0))
            I = I + noise_std * np.random.randn(len(I)).astype(np.float32)
            Q = Q + noise_std * np.random.randn(len(Q)).astype(np.float32)
        if np.random.rand() < 0.5:
            cfo_hz = np.random.uniform(-500, 500)
            t      = np.arange(len(I), dtype=np.float32) / 100e6
            phase  = 2 * math.pi * cfo_hz * t
            cos_p, sin_p = np.cos(phase).astype(np.float32), np.sin(phase).astype(np.float32)
            I, Q = I * cos_p - Q * sin_p, I * sin_p + Q * cos_p
        if np.random.rand() < 0.5:
            shift = np.random.randint(0, len(I))
            I, Q = np.roll(I, shift), np.roll(Q, shift)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.5, 2.0)
            I, Q = (I * scale).astype(np.float32), (Q * scale).astype(np.float32)
        return I.astype(np.float32), Q.astype(np.float32)


class RFUAVSpecDataset(Dataset):
    """
    RFUAV spectrogram dataset (on-the-fly generation from IQ data).
    Takes the same segment list as RFUAVIQDataset but generates 3-channel
    spectrograms (magnitude, phase, instantaneous frequency) via scipy STFT.
    Returns (3, SPEC_SIZE, SPEC_SIZE) tensors.
    """

    def __init__(self, segments, labels, segment_length=SEGMENT_LEN,
                 spec_size=SPEC_SIZE, is_train=True):
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

        I = raw[0::2] - np.mean(raw[0::2])
        Q = raw[1::2] - np.mean(raw[1::2])
        iq_complex = I + 1j * Q

        spec = self._iq_to_spectrogram(iq_complex)
        return torch.from_numpy(spec), label

    def _iq_to_spectrogram(self, iq_complex):
        from scipy.signal import stft as scipy_stft
        nperseg = min(256, len(iq_complex) // 4)
        noverlap = nperseg // 2
        _, _, Zxx = scipy_stft(iq_complex, fs=100e6, nperseg=nperseg,
                               noverlap=noverlap, return_onesided=False)

        mag   = np.abs(Zxx).astype(np.float32)
        phase = np.angle(Zxx).astype(np.float32)

        # Instantaneous frequency: diff of unwrapped phase along time axis
        unwrapped = np.unwrap(np.angle(Zxx), axis=1)
        inst_freq = np.diff(unwrapped, axis=1, prepend=unwrapped[:, :1]).astype(np.float32)

        # Normalize each channel to [0, 1]
        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-8:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        mag   = _norm(mag)
        phase = _norm(phase)
        inst_freq = _norm(inst_freq)

        # Resize to spec_size x spec_size via bilinear interpolation
        channels = np.stack([mag, phase, inst_freq], axis=0)  # (3, F, T)
        t = torch.from_numpy(channels).unsqueeze(0)  # (1, 3, F, T)
        t = F.interpolate(t, size=(self.spec_size, self.spec_size),
                          mode='bilinear', align_corners=False)
        return t.squeeze(0).numpy().astype(np.float32)


class RFUAVGAFDataset(Dataset):
    """
    RFUAV GAF dataset (on-the-fly VMD + GAF from IQ data).
    Uses rfml.features.vmd and rfml.features.gaf to generate (3, 256, 256) GAF images.
    Falls back to simple GAF if VMD is unavailable.
    """

    def __init__(self, segments, labels, segment_length=SEGMENT_LEN,
                 gaf_size=GAF_SIZE, is_train=True):
        self.segments       = segments
        self.labels         = labels
        self.segment_length = segment_length
        self.gaf_size       = gaf_size
        self.is_train       = is_train
        self._vmd_available = None
        self._gaf_available = None

    def _check_features(self):
        if self._vmd_available is None:
            try:
                from rfml.features.vmd import vmd_decompose
                self._vmd_available = True
            except ImportError:
                self._vmd_available = False
            try:
                from rfml.features.gaf import compute_gaf
                self._gaf_available = True
            except ImportError:
                self._gaf_available = False

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        self._check_features()
        fpath, start = self.segments[idx]
        label        = self.labels[idx]
        offset_bytes = start * 2 * 4
        n_floats     = self.segment_length * 2

        raw = np.fromfile(fpath, dtype=np.float32, count=n_floats, offset=offset_bytes)
        if len(raw) < n_floats:
            raw = np.pad(raw, (0, n_floats - len(raw)))

        I = raw[0::2] - np.mean(raw[0::2])
        Q = raw[1::2] - np.mean(raw[1::2])
        amplitude = np.sqrt(I**2 + Q**2).astype(np.float32)

        gaf = self._compute_gaf_image(amplitude, I, Q)
        return torch.from_numpy(gaf), label

    def _compute_gaf_image(self, amplitude, I, Q):
        """Generate 3-channel GAF image: GASF(amplitude), GADF(amplitude), GASF(phase)."""
        # Subsample to gaf_size points for tractable GAF computation
        n = len(amplitude)
        step = max(1, n // self.gaf_size)
        amp_sub   = amplitude[::step][:self.gaf_size]
        phase_sub = np.arctan2(Q[::step][:self.gaf_size], I[::step][:self.gaf_size])

        if self._vmd_available and self._gaf_available:
            try:
                from rfml.features.vmd import vmd_decompose
                from rfml.features.gaf import compute_gaf
                # VMD decompose into 3 modes, then GAF each
                modes = vmd_decompose(amplitude, n_modes=3, max_iter=200)
                channels = []
                for mode in modes[:3]:
                    mode_sub = mode[::step][:self.gaf_size]
                    gaf_img  = compute_gaf(mode_sub, image_size=self.gaf_size)
                    channels.append(gaf_img)
                while len(channels) < 3:
                    channels.append(channels[-1])
                out = np.stack(channels, axis=0).astype(np.float32)
                return out
            except Exception:
                pass

        # Fallback: simple Gramian Angular Field
        channels = []
        for sig in [amp_sub, amp_sub, phase_sub]:
            gaf_img = self._simple_gaf(sig)
            channels.append(gaf_img)
        out = np.stack(channels, axis=0).astype(np.float32)
        return out

    def _simple_gaf(self, signal):
        """Compute Gramian Angular Summation Field for a 1D signal."""
        n = len(signal)
        mn, mx = signal.min(), signal.max()
        if mx - mn < 1e-8:
            return np.zeros((self.gaf_size, self.gaf_size), dtype=np.float32)
        scaled = (signal - mn) / (mx - mn) * 2 - 1
        scaled = np.clip(scaled, -1, 1)
        phi = np.arccos(scaled)
        gaf = np.outer(np.cos(phi), np.cos(phi)) + np.outer(np.sin(phi), np.sin(phi))
        # Resize to gaf_size x gaf_size
        t = torch.from_numpy(gaf).unsqueeze(0).unsqueeze(0).float()
        t = F.interpolate(t, size=(self.gaf_size, self.gaf_size),
                          mode='bilinear', align_corners=False)
        return t.squeeze().numpy().astype(np.float32)


class DroneRFbIQDataset(Dataset):
    """
    DroneRFb raw IQ dataset.
    Loads .mat (HDF5) files from DRFB_IQ_DIR/{train,test}/*.mat
    Each .mat has I(1,4M) and Q(1,4M) arrays at 80 MSps.
    Returns (2, SEGMENT_LEN) tensors.
    """

    def __init__(self, mat_dir, segment_length=SEGMENT_LEN, is_train=True,
                 class_names=None, class_to_idx=None):
        import h5py
        self.segment_length = segment_length
        self.is_train       = is_train
        self.segments       = []  # (mat_path, start_idx, label)

        if class_names is not None:
            self.class_names  = class_names
            self.class_to_idx = class_to_idx
        else:
            self.class_names  = sorted(
                d for d in os.listdir(mat_dir)
                if os.path.isdir(os.path.join(mat_dir, d))
            )
            self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}

        # DroneRFb can be organized as flat .mat files with class in filename,
        # or as subdirectories per class
        if any(os.path.isdir(os.path.join(mat_dir, d)) for d in os.listdir(mat_dir)
               if not d.startswith('.')):
            # Subdirectory layout
            for cls_name in self.class_names:
                cls_dir = os.path.join(mat_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                label = self.class_to_idx[cls_name]
                mat_files = sorted(glob.glob(os.path.join(cls_dir, '*.mat')))
                for mf in mat_files:
                    self._index_mat(mf, label)
        else:
            # Flat layout: filename encodes class
            mat_files = sorted(glob.glob(os.path.join(mat_dir, '*.mat')))
            for mf in mat_files:
                basename = os.path.splitext(os.path.basename(mf))[0]
                cls_name = basename.split('_')[0] if '_' in basename else basename
                if cls_name in self.class_to_idx:
                    self._index_mat(mf, self.class_to_idx[cls_name])

    def _index_mat(self, mat_path, label):
        try:
            import h5py
            with h5py.File(mat_path, 'r') as f:
                # Try common key patterns
                for key in ['I', 'i', 'I_data']:
                    if key in f:
                        n_samples = f[key].shape[-1]
                        break
                else:
                    # Use first available dataset
                    first_key = list(f.keys())[0]
                    n_samples = f[first_key].shape[-1]

            n_segments = n_samples // self.segment_length
            for seg_idx in range(n_segments):
                start = seg_idx * self.segment_length
                self.segments.append((mat_path, start, label))
        except Exception as e:
            print(f"  WARNING: Failed to index {mat_path}: {e}")

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
                    I_data = f[ik][0, start:end] if f[ik].ndim == 2 else f[ik][start:end]
                    Q_data = f[qk][0, start:end] if f[qk].ndim == 2 else f[qk][start:end]
                    break
            if I_data is None:
                keys = list(f.keys())
                I_data = f[keys[0]][0, start:end] if f[keys[0]].ndim == 2 else f[keys[0]][start:end]
                Q_data = f[keys[1]][0, start:end] if f[keys[1]].ndim == 2 else f[keys[1]][start:end]

        I_data = np.array(I_data, dtype=np.float32)
        Q_data = np.array(Q_data, dtype=np.float32)

        if len(I_data) < self.segment_length:
            I_data = np.pad(I_data, (0, self.segment_length - len(I_data)))
            Q_data = np.pad(Q_data, (0, self.segment_length - len(Q_data)))

        I_data = I_data - I_data.mean()
        Q_data = Q_data - Q_data.mean()

        if self.is_train:
            I_data, Q_data = _iq_augment(I_data, Q_data)

        x = np.stack([I_data, Q_data], axis=0).astype(np.float32)
        return torch.from_numpy(x), label


class DroneRFbSpecDataset(Dataset):
    """
    DroneRFb type-level spectrogram dataset.
    Loads PNG images from DRFB_SPEC_DIR/{train,test}/{class_name}/*.png
    Returns (3, SPEC_SIZE, SPEC_SIZE) tensors.
    """

    def __init__(self, img_dir, spec_size=SPEC_SIZE, is_train=True):
        from torchvision import transforms, datasets
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((spec_size, spec_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((spec_size, spec_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        self.dataset = datasets.ImageFolder(img_dir, transform=self.transform)
        self.class_names  = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def _iq_augment(I, Q):
    """RF augmentation shared by IQ datasets."""
    if np.random.rand() < 0.8:
        snr_db    = np.random.uniform(0, 30)
        sig_power = np.mean(I**2 + Q**2)
        noise_pow = sig_power / (10 ** (snr_db / 10) + 1e-12)
        noise_std = math.sqrt(max(noise_pow, 0))
        I = I + noise_std * np.random.randn(len(I)).astype(np.float32)
        Q = Q + noise_std * np.random.randn(len(Q)).astype(np.float32)
    if np.random.rand() < 0.5:
        cfo_hz = np.random.uniform(-500, 500)
        t      = np.arange(len(I), dtype=np.float32) / 80e6  # DroneRFb is 80 MSps
        phase  = 2 * math.pi * cfo_hz * t
        cos_p, sin_p = np.cos(phase).astype(np.float32), np.sin(phase).astype(np.float32)
        I, Q = I * cos_p - Q * sin_p, I * sin_p + Q * cos_p
    if np.random.rand() < 0.5:
        shift = np.random.randint(0, len(I))
        I, Q = np.roll(I, shift), np.roll(Q, shift)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.5, 2.0)
        I, Q = (I * scale).astype(np.float32), (Q * scale).astype(np.float32)
    return I.astype(np.float32), Q.astype(np.float32)


# ============================================================================
# DATASET BUILDERS
# ============================================================================

def _discover_rfuav_files():
    """Discover all RFUAV .iq files and return (file_path, drone_name) list."""
    entries = []
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")
    for drone_name in sorted(os.listdir(RAW_DIR)):
        drone_top = os.path.join(RAW_DIR, drone_name)
        if not os.path.isdir(drone_top):
            continue
        pattern = os.path.join(drone_top, drone_name, 'VTSBW=*', '*.iq')
        iq_files = sorted(glob.glob(pattern))
        if not iq_files:
            pattern2 = os.path.join(drone_top, 'VTSBW=*', '*.iq')
            iq_files = sorted(glob.glob(pattern2))
        for f in iq_files:
            entries.append((f, drone_name))
    return entries


def _build_rfuav_segments(segment_length=SEGMENT_LEN, val_split=0.2):
    """Build train/val segment lists for RFUAV IQ data."""
    file_entries = _discover_rfuav_files()
    if not file_entries:
        raise RuntimeError(f"No .iq files found under {RAW_DIR}")

    drone_names  = sorted(set(dn for _, dn in file_entries))
    class_to_idx = {name: i for i, name in enumerate(drone_names)}
    num_classes  = len(drone_names)
    print(f"  RFUAV: {len(file_entries)} .iq files, {num_classes} drone types")

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
                start = seg_idx * segment_length
                drone_segs.append((fpath, start))

        if not drone_segs:
            continue

        n_total = len(drone_segs)
        n_train = max(1, int(n_total * (1.0 - val_split)))
        for seg in drone_segs[:n_train]:
            train_segs.append(seg)
            train_labels.append(label)
        for seg in drone_segs[n_train:]:
            val_segs.append(seg)
            val_labels.append(label)

    print(f"  Train segments: {len(train_segs):,}  Val segments: {len(val_segs):,}")
    return train_segs, train_labels, val_segs, val_labels, drone_names


def build_rfuav_iq_loaders(batch_size, num_workers=4):
    """Build RFUAV IQ DataLoaders for raw IQ experts."""
    train_segs, train_labels, val_segs, val_labels, class_names = \
        _build_rfuav_segments()
    train_ds = RFUAVIQDataset(train_segs, train_labels, SEGMENT_LEN, is_train=True)
    val_ds   = RFUAVIQDataset(val_segs,   val_labels,   SEGMENT_LEN, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(class_names), class_names


def build_rfuav_spec_loaders(batch_size, num_workers=4):
    """Build RFUAV spectrogram DataLoaders (on-the-fly from IQ)."""
    train_segs, train_labels, val_segs, val_labels, class_names = \
        _build_rfuav_segments()
    train_ds = RFUAVSpecDataset(train_segs, train_labels, SEGMENT_LEN,
                                spec_size=SPEC_SIZE, is_train=True)
    val_ds   = RFUAVSpecDataset(val_segs,   val_labels,   SEGMENT_LEN,
                                spec_size=SPEC_SIZE, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(class_names), class_names


def build_rfuav_gaf_loaders(batch_size, num_workers=4):
    """Build RFUAV GAF DataLoaders (on-the-fly VMD+GAF from IQ)."""
    train_segs, train_labels, val_segs, val_labels, class_names = \
        _build_rfuav_segments()
    train_ds = RFUAVGAFDataset(train_segs, train_labels, SEGMENT_LEN,
                               gaf_size=GAF_SIZE, is_train=True)
    val_ds   = RFUAVGAFDataset(val_segs,   val_labels,   SEGMENT_LEN,
                               gaf_size=GAF_SIZE, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(class_names), class_names


def build_droneRFb_iq_loaders(batch_size, num_workers=4):
    """Build DroneRFb IQ DataLoaders."""
    train_dir = os.path.join(DRFB_IQ_DIR, 'train')
    test_dir  = os.path.join(DRFB_IQ_DIR, 'test')

    train_ds = DroneRFbIQDataset(train_dir, SEGMENT_LEN, is_train=True)
    test_ds  = DroneRFbIQDataset(test_dir,  SEGMENT_LEN, is_train=False,
                                  class_names=train_ds.class_names,
                                  class_to_idx=train_ds.class_to_idx)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"  DroneRFb IQ: {len(train_ds):,} train, {len(test_ds):,} test, "
          f"{num_classes} classes: {class_names}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes, class_names


def build_droneRFb_spec_loaders(batch_size, num_workers=4):
    """Build DroneRFb spectrogram DataLoaders from PNG images."""
    train_dir = os.path.join(DRFB_SPEC_DIR, 'train')
    test_dir  = os.path.join(DRFB_SPEC_DIR, 'test')

    train_ds = DroneRFbSpecDataset(train_dir, spec_size=SPEC_SIZE, is_train=True)
    test_ds  = DroneRFbSpecDataset(test_dir,  spec_size=SPEC_SIZE, is_train=False)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"  DroneRFb Spec: {len(train_ds):,} train, {len(test_ds):,} test, "
          f"{num_classes} classes: {class_names}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes, class_names


def build_droneRFb_gaf_loaders(batch_size, num_workers=4):
    """Build DroneRFb GAF DataLoaders (on-the-fly from IQ)."""
    train_dir = os.path.join(DRFB_IQ_DIR, 'train')
    test_dir  = os.path.join(DRFB_IQ_DIR, 'test')

    # Reuse DroneRFbIQDataset to get segments, then wrap with GAF
    train_iq = DroneRFbIQDataset(train_dir, SEGMENT_LEN, is_train=True)
    test_iq  = DroneRFbIQDataset(test_dir,  SEGMENT_LEN, is_train=False,
                                  class_names=train_iq.class_names,
                                  class_to_idx=train_iq.class_to_idx)

    class_names = train_iq.class_names
    num_classes = len(class_names)

    # Build GAF datasets from same segments
    train_segs   = [(s[0], s[1]) for s in train_iq.segments]
    train_labels = [s[2] for s in train_iq.segments]
    test_segs    = [(s[0], s[1]) for s in test_iq.segments]
    test_labels  = [s[2] for s in test_iq.segments]

    # Use RFUAV GAF dataset class (works for any IQ source)
    # but override __getitem__ to use h5py. Instead, create a DroneRFb-specific GAF wrapper.
    train_ds = _DroneRFbGAFDataset(train_iq.segments, SEGMENT_LEN, GAF_SIZE, is_train=True)
    test_ds  = _DroneRFbGAFDataset(test_iq.segments,  SEGMENT_LEN, GAF_SIZE, is_train=False)

    print(f"  DroneRFb GAF: {len(train_ds):,} train, {len(test_ds):,} test, "
          f"{num_classes} classes: {class_names}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes, class_names


class _DroneRFbGAFDataset(Dataset):
    """GAF wrapper around DroneRFb IQ segments."""

    def __init__(self, segments, segment_length, gaf_size, is_train):
        self.segments       = segments  # list of (mat_path, start, label)
        self.segment_length = segment_length
        self.gaf_size       = gaf_size
        self.is_train       = is_train

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
                    I_data = f[ik][0, start:end] if f[ik].ndim == 2 else f[ik][start:end]
                    Q_data = f[qk][0, start:end] if f[qk].ndim == 2 else f[qk][start:end]
                    break
            if I_data is None:
                keys = list(f.keys())
                I_data = f[keys[0]][0, start:end] if f[keys[0]].ndim == 2 else f[keys[0]][start:end]
                Q_data = f[keys[1]][0, start:end] if f[keys[1]].ndim == 2 else f[keys[1]][start:end]

        I_data = np.array(I_data, dtype=np.float32)
        Q_data = np.array(Q_data, dtype=np.float32)
        if len(I_data) < self.segment_length:
            I_data = np.pad(I_data, (0, self.segment_length - len(I_data)))
            Q_data = np.pad(Q_data, (0, self.segment_length - len(Q_data)))

        I_data -= I_data.mean()
        Q_data -= Q_data.mean()
        amplitude = np.sqrt(I_data**2 + Q_data**2).astype(np.float32)

        gaf = self._simple_gaf_3ch(amplitude, I_data, Q_data)
        return torch.from_numpy(gaf), label

    def _simple_gaf_3ch(self, amp, I, Q):
        step = max(1, len(amp) // self.gaf_size)
        amp_sub   = amp[::step][:self.gaf_size]
        phase_sub = np.arctan2(Q[::step][:self.gaf_size], I[::step][:self.gaf_size])
        channels  = []
        for sig in [amp_sub, amp_sub, phase_sub]:
            mn, mx = sig.min(), sig.max()
            if mx - mn < 1e-8:
                channels.append(np.zeros((self.gaf_size, self.gaf_size), dtype=np.float32))
                continue
            scaled = (sig - mn) / (mx - mn) * 2 - 1
            scaled = np.clip(scaled, -1, 1)
            phi = np.arccos(scaled)
            gaf = np.outer(np.cos(phi), np.cos(phi)) + np.outer(np.sin(phi), np.sin(phi))
            t = torch.from_numpy(gaf).unsqueeze(0).unsqueeze(0).float()
            t = F.interpolate(t, size=(self.gaf_size, self.gaf_size),
                              mode='bilinear', align_corners=False)
            channels.append(t.squeeze().numpy().astype(np.float32))
        return np.stack(channels, axis=0)


# ============================================================================
# EXPERT LOADING
# ============================================================================

def load_expert(expert_name, num_classes):
    """
    Import and instantiate an expert, replace classification head with
    nn.Linear(512, num_classes), return (model, n_params).
    """
    mod_path, cls_name = EXPERT_MODULES[expert_name]
    module = __import__(mod_path, fromlist=[cls_name])
    ExpertClass = getattr(module, cls_name)

    # Most experts accept num_classes in constructor; try both patterns
    try:
        expert = ExpertClass(num_classes=num_classes)
    except TypeError:
        try:
            expert = ExpertClass()
        except TypeError:
            # Some experts need additional config
            expert = ExpertClass(num_classes=37)

    # Replace classification head with linear from 512-dim embedding
    # The expert's get_embedding returns (B, 512), so we attach a new head
    expert.head = nn.Linear(512, num_classes)

    n_params = sum(p.numel() for p in expert.parameters())
    print(f"  Loaded {expert_name}: {n_params/1e6:.2f}M params, "
          f"output classes={num_classes}")
    return expert, n_params


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_and_evaluate(expert_name, train_loader, val_loader, num_classes,
                       class_names, cfg, dataset_name):
    """
    Train an expert on the given data and return a result dict.
    Replaces expert head with nn.Linear(512, num_classes).
    Uses AdamW, BF16 AMP, cosine LR, label smoothing, early stopping.
    """
    print(f"\n{'='*70}")
    print(f"  {expert_name} on {dataset_name}")
    print(f"{'='*70}", flush=True)

    try:
        model, n_params = load_expert(expert_name, num_classes)
    except Exception as e:
        print(f"  FAILED to load {expert_name}: {e}")
        traceback.print_exc()
        return {'expert': expert_name, 'dataset': dataset_name,
                'error': str(e), 'status': 'load_failed'}

    model = model.to(DEVICE)
    model.unfreeze()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
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

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=cfg['amp']):
                logits = model(batch_x)
                loss   = criterion(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * batch_y.size(0)
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total   += batch_y.size(0)

        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                        enabled=cfg['amp']):
                    logits = model(batch_x)
                val_correct += (logits.argmax(1) == batch_y).sum().item()
                val_total   += batch_y.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc   = val_correct   / max(val_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
            marker = ' *'
        else:
            no_improve += 1
            marker = ''

        if (epoch + 1) % 5 == 0 or marker:
            elapsed = time.time() - train_start
            print(
                f"  Epoch {epoch+1:3d}/{cfg['epochs']}: "
                f"train={train_acc:.4f} val={val_acc:.4f} "
                f"loss={train_loss/max(train_total,1):.4f} "
                f"[{elapsed:.0f}s]{marker}",
                flush=True
            )

        if no_improve >= cfg['patience']:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break

    # ---- Load best checkpoint ----
    if best_state:
        model.load_state_dict(best_state)
        ckpt_path = os.path.join(MODEL_DIR,
                                 f'{dataset_name}_{expert_name}_best.pt')
        torch.save(best_state, ckpt_path)
        print(f"  Saved checkpoint -> {ckpt_path}")

    train_time = time.time() - train_start

    # ---- Full evaluation ----
    model.eval()
    all_preds, all_labels = [], []
    inference_times = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            t0 = time.time()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=cfg['amp']):
                logits = model(batch_x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.time() - t0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc    = accuracy_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(
        all_labels, all_preds, target_names=class_names,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()

    total_inference_time = sum(inference_times)
    n_val_samples = len(all_labels)
    avg_inference_ms = (total_inference_time / max(n_val_samples, 1)) * 1000

    result = {
        'expert':               expert_name,
        'dataset':              dataset_name,
        'status':               'success',
        'accuracy':             float(acc),
        'f1_macro':             float(f1),
        'best_val_acc':         float(best_val_acc),
        'params':               int(n_params),
        'params_M':             round(n_params / 1e6, 2),
        'train_time_sec':       round(train_time, 1),
        'epochs_trained':       epoch + 1,
        'avg_inference_ms':     round(avg_inference_ms, 3),
        'total_inference_sec':  round(total_inference_time, 2),
        'num_val_samples':      int(n_val_samples),
        'per_class':            {cn: report[cn] for cn in class_names if cn in report},
        'confusion_matrix':     cm,
        'class_names':          class_names,
    }

    print(f"\n  RESULT: {expert_name} on {dataset_name}")
    print(f"  Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
    print(f"  Params: {n_params/1e6:.2f}M | Train: {train_time:.0f}s | "
          f"Inference: {avg_inference_ms:.3f} ms/sample")
    print(classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    ))

    return result


# ============================================================================
# DATA LOADER DISPATCH
# ============================================================================

def get_loaders_for_expert(expert_name, dataset_name, batch_size, num_workers=4):
    """
    Return (train_loader, val_loader, num_classes, class_names) appropriate
    for the expert's input modality and the target dataset.
    """
    modality = _get_modality(expert_name)

    if dataset_name == 'rfuav':
        if modality == 'iq':
            return build_rfuav_iq_loaders(batch_size, num_workers)
        elif modality == 'spec':
            return build_rfuav_spec_loaders(batch_size, num_workers)
        elif modality == 'gaf':
            return build_rfuav_gaf_loaders(batch_size, num_workers)
    elif dataset_name == 'droneRFb':
        if modality == 'iq':
            return build_droneRFb_iq_loaders(batch_size, num_workers)
        elif modality == 'spec':
            return build_droneRFb_spec_loaders(batch_size, num_workers)
        elif modality == 'gaf':
            return build_droneRFb_gaf_loaders(batch_size, num_workers)

    raise ValueError(f"Unknown expert/dataset combo: {expert_name}/{dataset_name}")


def _get_modality(expert_name):
    if expert_name in IQ_EXPERTS:
        return 'iq'
    elif expert_name in SPEC_EXPERTS:
        return 'spec'
    elif expert_name in GAF_EXPERTS:
        return 'gaf'
    raise ValueError(f"Unknown expert modality: {expert_name}")


def _default_batch_size(expert_name):
    if expert_name in IQ_EXPERTS:
        return 64
    return 128


# ============================================================================
# MAIN
# ============================================================================

def print_summary_table(all_results):
    """Print a formatted comparison table sorted by accuracy."""
    valid = [r for r in all_results if r.get('status') == 'success']
    failed = [r for r in all_results if r.get('status') != 'success']

    if not valid:
        print("\n  No successful results to summarize.")
        return

    valid.sort(key=lambda r: r['accuracy'], reverse=True)

    print(f"\n{'='*100}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Dataset':<12} {'Expert':<26} {'Acc':>7} {'F1':>7} "
          f"{'Params':>8} {'Train(s)':>9} {'Inf(ms)':>9}")
    print(f"  {'-'*12} {'-'*26} {'-'*7} {'-'*7} {'-'*8} {'-'*9} {'-'*9}")

    for r in valid:
        print(f"  {r['dataset']:<12} {r['expert']:<26} "
              f"{r['accuracy']:>7.4f} {r['f1_macro']:>7.4f} "
              f"{r['params_M']:>7.2f}M "
              f"{r['train_time_sec']:>9.1f} "
              f"{r['avg_inference_ms']:>9.3f}")

    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for r in failed:
            print(f"    {r.get('dataset','?')}/{r.get('expert','?')}: "
                  f"{r.get('error','unknown')[:80]}")

    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(
        description='RFML-MoE Expert Benchmark — MI300X ROCm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rfml_expert_benchmark.py --dataset both --experts all
  python rfml_expert_benchmark.py --dataset rfuav --experts IQExpert TFMSExpert
  python rfml_expert_benchmark.py --dataset droneRFb --experts SpectrogramExpert --epochs 30
        """
    )
    parser.add_argument(
        '--dataset', type=str, default='both',
        choices=['rfuav', 'droneRFb', 'both'],
        help='Dataset to benchmark on (default: both)'
    )
    parser.add_argument(
        '--experts', nargs='+', default=['all'],
        help='Expert names to benchmark, or "all" (default: all)'
    )
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Override batch size (0=auto: 64 for IQ, 128 for spec)')
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--patience',   type=int, default=15)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-amp',     action='store_true',
                        help='Disable BF16 mixed precision')
    args = parser.parse_args()

    # Resolve expert list
    if 'all' in args.experts:
        experts = ALL_EXPERTS[:]
    else:
        experts = []
        for e in args.experts:
            if e not in ALL_EXPERTS:
                print(f"WARNING: Unknown expert '{e}', skipping. "
                      f"Available: {ALL_EXPERTS}")
            else:
                experts.append(e)
    if not experts:
        print("ERROR: No valid experts specified.")
        sys.exit(1)

    datasets = []
    if args.dataset in ('rfuav', 'both'):
        datasets.append('rfuav')
    if args.dataset in ('droneRFb', 'both'):
        datasets.append('droneRFb')

    # Print banner
    print('=' * 70, flush=True)
    print('RFML-MoE EXPERT BENCHMARK — MI300X ROCm', flush=True)
    print(f'Device:   {DEVICE}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU:      {torch.cuda.get_device_name(0)}', flush=True)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'VRAM:     {mem_gb:.0f} GB', flush=True)
    print(f'Experts:  {experts}', flush=True)
    print(f'Datasets: {datasets}', flush=True)
    print(f'Epochs:   {args.epochs} | LR: {args.lr} | '
          f'Patience: {args.patience}', flush=True)
    print(f'AMP:      {"BF16" if not args.no_amp else "disabled"}', flush=True)
    print(f'Output:   {RESULT_DIR}', flush=True)
    print('=' * 70, flush=True)

    all_results = []

    for dataset_name in datasets:
        for expert_name in experts:
            bs = args.batch_size if args.batch_size > 0 else _default_batch_size(expert_name)

            cfg = {
                'lr':              args.lr,
                'weight_decay':    0.01,
                'label_smoothing': 0.1,
                'patience':        args.patience,
                'epochs':          args.epochs,
                'amp':             not args.no_amp,
            }

            try:
                train_loader, val_loader, num_classes, class_names = \
                    get_loaders_for_expert(expert_name, dataset_name, bs,
                                          args.num_workers)

                result = train_and_evaluate(
                    expert_name, train_loader, val_loader,
                    num_classes, class_names, cfg, dataset_name
                )
            except Exception as e:
                print(f"\n  ERROR: {expert_name} on {dataset_name}: {e}")
                traceback.print_exc()
                result = {
                    'expert':  expert_name,
                    'dataset': dataset_name,
                    'status':  'error',
                    'error':   str(e),
                }

            all_results.append(result)

            # Save individual result
            out_file = os.path.join(
                RESULT_DIR, f'{dataset_name}_{expert_name}_result.json'
            )
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved -> {out_file}", flush=True)

            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save combined summary
    summary_file = os.path.join(RESULT_DIR, 'benchmark_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp':   time.strftime('%Y-%m-%d %H:%M:%S'),
            'device':      str(DEVICE),
            'gpu':         torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'experts':     experts,
            'datasets':    datasets,
            'config': {
                'epochs':          args.epochs,
                'lr':              args.lr,
                'patience':        args.patience,
                'segment_length':  SEGMENT_LEN,
                'spec_size':       SPEC_SIZE,
                'gaf_size':        GAF_SIZE,
            },
            'results':     all_results,
        }, f, indent=2)
    print(f"\n  Summary -> {summary_file}", flush=True)

    print_summary_table(all_results)


if __name__ == '__main__':
    main()
