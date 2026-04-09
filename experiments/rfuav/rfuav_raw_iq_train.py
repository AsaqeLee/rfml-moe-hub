#!/usr/bin/env python3
"""
RFUAV Raw IQ Training Pipeline — PyTorch + ROCm on MI300X
==========================================================
Trains raw IQ classifiers directly on binary float32 IQ data.

37 drone types, 356 .iq files total
Data: /home/rax/mtp/raw/{drone_name}/{drone_name}/VTSBW={bw}/*.iq
Format: binary float32 interleaved I/Q, 100 MSps
Segment length: 100K samples default (0.001s each)

Models: resnet1d, se_resnet1d, cldnn, mcldnn

Run: sg render -c "python rfuav_raw_iq_train.py [--models ...] [--batch-size N] [--epochs N] [--segment-length N]"
"""
import os
import sys
import json
import time
import glob
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, confusion_matrix
)

# ============================================================================
# CONFIG
# ============================================================================

RAW_DIR    = '/home/rax/mtp/raw'
MODEL_DIR  = '/home/rax/mtp/models'
RESULT_DIR = '/home/rax/mtp/results'
RESULT_FILE = os.path.join(RESULT_DIR, 'rfuav_raw_iq_results.json')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CFG = {
    'batch_size':      32,
    'epochs':          50,
    'lr':              1e-3,
    'weight_decay':    0.01,
    'label_smoothing': 0.1,
    'patience':        15,
    'num_workers':     4,
    'amp':             True,   # BF16 mixed precision
    'segment_length':  100000, # 100K samples = 0.001s at 100 MSps
    'val_split':       0.2,
}

ALL_MODELS = ['resnet1d', 'se_resnet1d', 'cldnn', 'mcldnn']


# ============================================================================
# DATASET
# ============================================================================

class RFUAVDataset(Dataset):
    """
    Raw IQ dataset for RFUAV.

    Discovers all .iq files under RAW_DIR/{drone_name}/{drone_name}/VTSBW=*/*.iq
    Builds segments of fixed length from each file (temporal split for val).
    Returns tensor (2, segment_length) with I and Q channels.
    """

    def __init__(self, segments, labels, segment_length, is_train=True):
        """
        segments: list of (file_path, start_sample_idx)
        labels:   list of int class indices
        """
        self.segments       = segments
        self.labels         = labels
        self.segment_length = segment_length
        self.is_train       = is_train

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        fpath, start = self.segments[idx]
        label        = self.labels[idx]

        # Load segment: each complex sample = 2 float32 values
        offset_bytes = start * 2 * 4  # 2 floats * 4 bytes each
        n_floats     = self.segment_length * 2

        raw = np.fromfile(fpath, dtype=np.float32,
                          count=n_floats, offset=offset_bytes)

        if len(raw) < n_floats:
            # Pad with zeros if file is shorter than expected
            raw = np.pad(raw, (0, n_floats - len(raw)))

        I = raw[0::2]  # shape (segment_length,)
        Q = raw[1::2]  # shape (segment_length,)

        # DC removal
        I = I - I.mean()
        Q = Q - Q.mean()

        if self.is_train:
            I, Q = self._augment(I, Q)

        x = np.stack([I, Q], axis=0).astype(np.float32)  # (2, N)
        return torch.from_numpy(x), label

    def _augment(self, I, Q):
        """RF augmentation: AWGN, CFO, time shift, amplitude scale."""
        # AWGN (0-30 dB SNR, 80% probability)
        if np.random.rand() < 0.8:
            snr_db    = np.random.uniform(0, 30)
            sig_power = np.mean(I**2 + Q**2)
            noise_pow = sig_power / (10 ** (snr_db / 10) + 1e-12)
            noise_std = math.sqrt(max(noise_pow, 0))
            I = I + noise_std * np.random.randn(len(I)).astype(np.float32)
            Q = Q + noise_std * np.random.randn(len(Q)).astype(np.float32)

        # CFO (±500 Hz at 100 MSps, 50% probability)
        if np.random.rand() < 0.5:
            cfo_hz = np.random.uniform(-500, 500)
            fs     = 100e6
            t      = np.arange(len(I), dtype=np.float32) / fs
            phase  = 2 * math.pi * cfo_hz * t
            cos_p  = np.cos(phase).astype(np.float32)
            sin_p  = np.sin(phase).astype(np.float32)
            I_new  = I * cos_p - Q * sin_p
            Q_new  = I * sin_p + Q * cos_p
            I, Q   = I_new, Q_new

        # Time shift (circular roll, 50% probability)
        if np.random.rand() < 0.5:
            shift = np.random.randint(0, len(I))
            I = np.roll(I, shift)
            Q = np.roll(Q, shift)

        # Amplitude scale (0.5x–2.0x, 50% probability)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.5, 2.0)
            I = (I * scale).astype(np.float32)
            Q = (Q * scale).astype(np.float32)

        return I.astype(np.float32), Q.astype(np.float32)


def discover_files(raw_dir):
    """
    Returns list of (file_path, drone_name) for all .iq files.
    Pattern: raw_dir/{drone_name}/{drone_name}/VTSBW=*/*.iq
    """
    entries = []
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"RAW_DIR not found: {raw_dir}")

    for drone_name in sorted(os.listdir(raw_dir)):
        drone_top = os.path.join(raw_dir, drone_name)
        if not os.path.isdir(drone_top):
            continue
        pattern = os.path.join(drone_top, drone_name, 'VTSBW=*', '*.iq')
        iq_files = sorted(glob.glob(pattern))
        if not iq_files:
            # Also try without subdirectory nesting
            pattern2 = os.path.join(drone_top, 'VTSBW=*', '*.iq')
            iq_files = sorted(glob.glob(pattern2))
        for f in iq_files:
            entries.append((f, drone_name))

    return entries


def build_datasets(raw_dir, segment_length, val_split=0.2):
    """
    Discovers files, segments them, and splits 80/20 temporally per drone type.
    Returns (train_dataset, val_dataset, class_names).
    """
    file_entries = discover_files(raw_dir)
    if not file_entries:
        raise RuntimeError(f"No .iq files found under {raw_dir}")

    # Build class list
    drone_names = sorted(set(dn for _, dn in file_entries))
    class_to_idx = {name: i for i, name in enumerate(drone_names)}
    num_classes  = len(drone_names)
    print(f"  Found {len(file_entries)} .iq files, {num_classes} drone types")

    # Group files by drone type
    files_by_drone = {dn: [] for dn in drone_names}
    for fpath, dn in file_entries:
        files_by_drone[dn].append(fpath)

    # Build segments with temporal split per drone
    train_segs, train_labels = [], []
    val_segs,   val_labels   = [], []

    for drone_name, flist in files_by_drone.items():
        label      = class_to_idx[drone_name]
        drone_segs = []

        for fpath in flist:
            file_size_bytes = os.path.getsize(fpath)
            n_samples       = file_size_bytes // (2 * 4)  # 2 float32 per complex sample
            n_segments      = n_samples // segment_length
            for seg_idx in range(n_segments):
                start = seg_idx * segment_length
                drone_segs.append((fpath, start))

        if not drone_segs:
            print(f"  WARNING: no segments for {drone_name}, skipping")
            continue

        # Temporal split: first 80% train, last 20% val
        n_total = len(drone_segs)
        n_train = max(1, int(n_total * (1.0 - val_split)))
        for seg in drone_segs[:n_train]:
            train_segs.append(seg)
            train_labels.append(label)
        for seg in drone_segs[n_train:]:
            val_segs.append(seg)
            val_labels.append(label)

    print(f"  Train segments: {len(train_segs):,}  Val segments: {len(val_segs):,}")
    print(f"  Segment length: {segment_length:,} samples = {segment_length/100e6*1000:.2f} ms")

    train_ds = RFUAVDataset(train_segs, train_labels, segment_length, is_train=True)
    val_ds   = RFUAVDataset(val_segs,   val_labels,   segment_length, is_train=False)

    return train_ds, val_ds, drone_names


# ============================================================================
# MODELS
# ============================================================================

# ---- Squeeze-Excitation block ----

class SEBlock1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1)
        return x * s


# ---- ResNet1D basic block ----

class ResBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.se    = SEBlock1d(out_ch) if use_se else None
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


def make_layer1d(in_ch, out_ch, n_blocks, stride, use_se=False):
    layers = [ResBlock1d(in_ch, out_ch, stride=stride, use_se=use_se)]
    for _ in range(1, n_blocks):
        layers.append(ResBlock1d(out_ch, out_ch, stride=1, use_se=use_se))
    return nn.Sequential(*layers)


# ---- ResNet1D ----

class ResNet1D(nn.Module):
    """
    Stem: Conv1d(2,64,k=7,s=4) + BN + ReLU
    3 ResNet layers with stride 4,4,4 for aggressive downsampling
    AdaptiveAvgPool1d(1) -> Linear(256, num_classes)
    Input: (B, 2, N)
    """

    def __init__(self, num_classes, use_se=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = make_layer1d(64,  128, n_blocks=2, stride=4, use_se=use_se)
        self.layer2 = make_layer1d(128, 256, n_blocks=2, stride=4, use_se=use_se)
        self.layer3 = make_layer1d(256, 512, n_blocks=2, stride=4, use_se=use_se)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ---- SE-ResNet1D ----

class SEResNet1D(ResNet1D):
    def __init__(self, num_classes):
        super().__init__(num_classes, use_se=True)


# ---- CLDNN ----

class CLDNN(nn.Module):
    """
    Conv1d feature extraction -> BiLSTM -> DNN head
    Input: (B, 2, N)
    """

    def __init__(self, num_classes, segment_length=100000):
        super().__init__()
        # CNN front-end: heavy downsampling
        self.cnn = nn.Sequential(
            nn.Conv1d(2,  64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        # Estimate sequence length after CNN
        with torch.no_grad():
            dummy     = torch.zeros(1, 2, segment_length)
            cnn_out   = self.cnn(dummy)
            seq_len   = cnn_out.shape[-1]
            cnn_feats = cnn_out.shape[1]

        print(f"  CLDNN: CNN output seq_len={seq_len}, feats={cnn_feats}")

        self.lstm = nn.LSTM(
            input_size=cnn_feats,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)              # (B, C, T)
        x = x.permute(0, 2, 1)      # (B, T, C)
        _, (h, _) = self.lstm(x)
        # Concat last hidden from both directions
        h = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 256)
        return self.head(h)


# ---- MCLDNN ----

class MCLDNN(nn.Module):
    """
    Separate I / Q / IQ CNN branches -> merge -> BiLSTM -> DNN head
    Input: (B, 2, N)
    """

    def __init__(self, num_classes, segment_length=100000):
        super().__init__()

        def make_branch(in_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, 64, kernel_size=7, stride=4, padding=3, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=5, stride=4, padding=2, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=3, stride=4, padding=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=3, stride=4, padding=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            )

        self.branch_I  = make_branch(1)
        self.branch_Q  = make_branch(1)
        self.branch_IQ = make_branch(2)

        # Estimate merged feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 2, segment_length)
            I_out  = self.branch_I(dummy[:, :1, :])
            Q_out  = self.branch_Q(dummy[:, 1:, :])
            IQ_out = self.branch_IQ(dummy)
            seq_len   = I_out.shape[-1]
            merged_ch = I_out.shape[1] + Q_out.shape[1] + IQ_out.shape[1]

        print(f"  MCLDNN: merged seq_len={seq_len}, feats={merged_ch}")

        self.lstm = nn.LSTM(
            input_size=merged_ch,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        I  = x[:, :1, :]  # (B, 1, N)
        Q  = x[:, 1:, :]  # (B, 1, N)
        fi = self.branch_I(I)           # (B, 64, T)
        fq = self.branch_Q(Q)           # (B, 64, T)
        fiq = self.branch_IQ(x)         # (B, 64, T)
        feat = torch.cat([fi, fq, fiq], dim=1)  # (B, 192, T)
        feat = feat.permute(0, 2, 1)             # (B, T, 192)
        _, (h, _) = self.lstm(feat)
        h = torch.cat([h[-2], h[-1]], dim=-1)    # (B, 256)
        return self.head(h)


def create_model(model_key, num_classes, segment_length):
    if model_key == 'resnet1d':
        return ResNet1D(num_classes, use_se=False)
    elif model_key == 'se_resnet1d':
        return SEResNet1D(num_classes)
    elif model_key == 'cldnn':
        return CLDNN(num_classes, segment_length=segment_length)
    elif model_key == 'mcldnn':
        return MCLDNN(num_classes, segment_length=segment_length)
    else:
        raise ValueError(f"Unknown model: {model_key}")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_one_model(model_key, cfg, train_ds, val_ds, class_names):
    print(f"\n{'='*70}")
    print(f"  Training: {model_key}")
    print(f"{'='*70}", flush=True)

    num_classes = len(class_names)
    seg_len     = cfg['segment_length']

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg['num_workers'] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=cfg['num_workers'] > 0,
    )

    model = create_model(model_key, num_classes, seg_len)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params/1e6:.2f}M")
    model = model.to(DEVICE)

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

        for x, labels in train_loader:
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
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
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
                    logits = model(x)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc   = val_correct   / max(val_total, 1)

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
                f"  Epoch {epoch+1:3d}: train={train_acc:.4f} val={val_acc:.4f} "
                f"loss={train_loss/max(train_total,1):.4f} [{elapsed:.0f}s] {marker}",
                flush=True
            )

        if no_improve >= cfg['patience']:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break

    # ---- Save best checkpoint ----
    if best_state:
        model.load_state_dict(best_state)
        ckpt_path = os.path.join(MODEL_DIR, f'rfuav_iq_{model_key}_best.pt')
        torch.save(best_state, ckpt_path)
        print(f"  Saved best checkpoint -> {ckpt_path}")

    # ---- Full evaluation on val set ----
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, labels in val_loader:
            x = x.to(DEVICE)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
                logits = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc    = accuracy_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(
        all_labels, all_preds, target_names=class_names,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()

    total_time  = time.time() - train_start
    epochs_done = epoch + 1

    result = {
        'model':            model_key,
        'accuracy':         float(acc),
        'f1_macro':         float(f1),
        'best_val_acc':     float(best_val_acc),
        'params':           int(n_params),
        'train_time_sec':   float(total_time),
        'epochs_trained':   int(epochs_done),
        'segment_length':   cfg['segment_length'],
        'per_class':        {cn: report[cn] for cn in class_names if cn in report},
        'confusion_matrix': cm,
        'class_names':      class_names,
    }

    print(f"\n  RESULT: {model_key}")
    print(f"  Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
    print(f"  Params: {n_params/1e6:.2f}M | Time: {total_time:.0f}s")
    print(classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    ))

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RFUAV Raw IQ Training Pipeline — MI300X ROCm'
    )
    parser.add_argument(
        '--models', nargs='+',
        default=ALL_MODELS,
        choices=ALL_MODELS,
        help=f'Models to train (default: all). Choices: {ALL_MODELS}',
    )
    parser.add_argument('--batch-size',     type=int,   default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs',         type=int,   default=50,
                        help='Max epochs (default: 50)')
    parser.add_argument('--segment-length', type=int,   default=100000,
                        help='Samples per segment (default: 100000 = 0.001s @ 100MSps)')
    parser.add_argument('--lr',             type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--num-workers',    type=int,   default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--raw-dir',        type=str,   default=RAW_DIR,
                        help=f'Root directory of raw IQ files (default: {RAW_DIR})')
    args = parser.parse_args()

    print('=' * 70, flush=True)
    print('RFUAV RAW IQ TRAINING PIPELINE — MI300X', flush=True)
    print(f'Device: {DEVICE}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU:  {torch.cuda.get_device_name(0)}', flush=True)
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB', flush=True)
    print(f'Data: {args.raw_dir}', flush=True)
    print(f'Segment length: {args.segment_length:,} samples', flush=True)
    print('=' * 70, flush=True)

    cfg = DEFAULT_CFG.copy()
    cfg['batch_size']     = args.batch_size
    cfg['epochs']         = args.epochs
    cfg['lr']             = args.lr
    cfg['segment_length'] = args.segment_length
    cfg['num_workers']    = args.num_workers

    # Build datasets once (shared across all models)
    print("\nBuilding datasets...", flush=True)
    train_ds, val_ds, class_names = build_datasets(
        args.raw_dir, cfg['segment_length'], cfg['val_split']
    )
    print(f"  Classes ({len(class_names)}): {class_names}\n", flush=True)

    all_results = {}

    for model_key in args.models:
        try:
            result = train_one_model(model_key, cfg, train_ds, val_ds, class_names)
            all_results[model_key] = result
        except Exception as e:
            print(f'  ERROR training {model_key}: {e}', flush=True)
            import traceback
            traceback.print_exc()
            all_results[model_key] = {'error': str(e), 'model': model_key}

    # ---- Summary table ----
    print('\n' + '=' * 90, flush=True)
    print('FINAL RESULTS — RFUAV Raw IQ (sorted by accuracy)', flush=True)
    print('=' * 90, flush=True)
    print(
        f"{'Model':<20} {'Accuracy':>10} {'F1-macro':>10} {'Params':>10} {'Time':>10}",
        flush=True
    )
    print('-' * 90, flush=True)

    sorted_names = sorted(
        all_results.keys(),
        key=lambda k: all_results[k].get('accuracy', 0.0),
        reverse=True
    )
    for name in sorted_names:
        r = all_results[name]
        if 'error' in r:
            print(f"{name:<20} {'ERROR':>10}", flush=True)
        else:
            params = r.get('params', 0)
            pstr   = f"{params/1e6:.2f}M" if params else '?'
            tstr   = f"{r.get('train_time_sec', 0):.0f}s"
            print(
                f"{name:<20} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
                f"{pstr:>10} {tstr:>10}",
                flush=True
            )

    # ---- Save results ----
    with open(RESULT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults saved -> {RESULT_FILE}', flush=True)


if __name__ == '__main__':
    main()
