#!/usr/bin/env python3
"""
Neural Network Architecture Comparison for RF Signal Classification
====================================================================
Implements actual DL experts from RFML-MoE + SOTA alternatives,
trains on RTL-ML 800-sample dataset, compares with statistical baselines.
"""
import numpy as np
import os
import json
import time
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy import signal as scipy_signal

warnings.filterwarnings('ignore')

SCRATCH = '/opt1/ml/rtl-ml-exp'
os.makedirs(f'{SCRATCH}/checkpoints', exist_ok=True)
os.makedirs(f'{SCRATCH}/results', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    print(f"Device: {DEVICE}")
    print("WARNING: Running on CPU - training will be slow")

# ============================================================================
# DATA PIPELINE
# ============================================================================

class RFDataset(Dataset):
    """PyTorch dataset for RTL-ML .npy files with RF-aware augmentation."""

    def __init__(self, data_dir='datasets_validated', iq_len=32768, spec_size=128,
                 augment=False, hos_dim=20, cyclo_dim=512):
        self.iq_len = iq_len
        self.spec_size = spec_size
        self.augment = augment
        self.hos_dim = hos_dim
        self.cyclo_dim = cyclo_dim

        self.samples = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
            for f in files:
                self.samples.append(os.path.join(cls_dir, f))
                self.labels.append(cls)

        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(classes)
        self.classes = classes
        print(f"Loaded {len(self.samples)} samples, {self.num_classes} classes: {classes}")

    def __len__(self):
        return len(self.samples)

    def _extract_hos(self, samples):
        """Extract HOS cumulants (20-dim)."""
        if len(samples) > 32768:
            samples = samples[:32768]
        samples = samples / (np.sqrt(np.mean(np.abs(samples)**2)) + 1e-10)

        C20 = np.mean(samples**2)
        C21 = np.mean(np.abs(samples)**2)
        M40 = np.mean(samples**4)
        M41 = np.mean(samples**3 * np.conj(samples))
        M42 = np.mean((np.abs(samples)**2)**2)
        M20 = np.mean(samples**2)
        M21 = np.mean(np.abs(samples)**2)
        C40 = M40 - 3 * M20**2
        C41 = M41 - 3 * M21 * M20
        C42 = M42 - np.abs(M20)**2 - 2 * M21**2

        M60 = np.mean(samples**6)
        M61 = np.mean(samples**5 * np.conj(samples))
        M62 = np.mean(samples**4 * np.conj(samples)**2)
        M63 = np.mean((np.abs(samples)**2)**3)
        C60 = M60 - 15*M20*M40 + 30*M20**3
        C61 = M61 - 5*M21*M40 - 10*M20*M41 + 30*M20**2*M21
        C62 = M62 - np.abs(M20)**2*M42 - 8*M21*M41 - M20*np.conj(M40) + 6*M21**2*M20 + 6*M20**2*np.conj(M20)
        C63 = M63 - 9*M21*M42 + 12*M21**3

        cumulants = [C20, C21, C40, C41, C42, C60, C61, C62, C63]
        norm_factor = np.abs(C21)**(np.array([1,1,2,2,2,3,3,3,3])/2) + 1e-10
        normalized = np.array([np.abs(c) for c in cumulants]) / norm_factor

        features = list(normalized)
        features.extend([
            np.abs(C42)/(np.abs(C21)**2+1e-10), np.abs(C40)/(np.abs(C20)**2+1e-10),
            np.abs(C63)/(np.abs(C21)**3+1e-10), np.abs(C60)/(np.abs(C20)**3+1e-10),
            np.angle(C40), np.angle(C42), np.angle(C60),
            np.abs(C40)/(np.abs(C42)+1e-10), np.abs(C60)/(np.abs(C63)+1e-10),
            np.abs(C41)/(np.abs(C42)+1e-10), np.abs(C61)/(np.abs(C62)+1e-10),
        ])
        return np.array(features[:self.hos_dim], dtype=np.float32)

    def _extract_cyclo(self, samples):
        """Extract cyclostationary SCF features."""
        if len(samples) > 16384:
            samples = samples[:16384]
        N = len(samples)
        Nfft = 256
        num_blocks = N // Nfft
        if num_blocks < 2:
            return np.zeros(self.cyclo_dim, dtype=np.float32)

        blocks = samples[:num_blocks*Nfft].reshape(num_blocks, Nfft)
        window = np.hanning(Nfft)
        X = np.fft.fft(blocks * window, axis=1)

        n_alpha = self.cyclo_dim // 2
        alpha_indices = np.linspace(1, Nfft//2-1, n_alpha, dtype=int)
        scf = []
        for alpha_idx in alpha_indices:
            Sxa = np.mean(X * np.conj(np.roll(X, alpha_idx, axis=1)), axis=0)
            scf.extend([np.max(np.abs(Sxa)), np.mean(np.abs(Sxa))])

        result = np.array(scf[:self.cyclo_dim], dtype=np.float32)
        if len(result) < self.cyclo_dim:
            result = np.pad(result, (0, self.cyclo_dim - len(result)))
        return result

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True).item()
        iq = data['samples']
        iq = iq - np.mean(iq)  # DC removal
        label = self.labels_encoded[idx]

        # RF augmentation
        if self.augment:
            iq = self._augment(iq)

        # 1. Raw IQ tensor (2, iq_len)
        if len(iq) > self.iq_len:
            start = np.random.randint(0, len(iq) - self.iq_len) if self.augment else 0
            iq_crop = iq[start:start+self.iq_len]
        else:
            iq_crop = np.pad(iq, (0, self.iq_len - len(iq)))
        iq_tensor = np.stack([np.real(iq_crop), np.imag(iq_crop)]).astype(np.float32)

        # 2. Spectrogram (3, spec_size, spec_size)
        f, t, Zxx = scipy_signal.stft(iq_crop, fs=1.024e6, nperseg=256, noverlap=192)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        inst_freq = np.diff(np.unwrap(phase, axis=1), axis=1)
        inst_freq = np.pad(inst_freq, ((0,0),(0,1)), mode='edge')

        from PIL import Image
        def resize_channel(ch, size):
            ch_normalized = (ch - ch.min()) / (ch.max() - ch.min() + 1e-10)
            img = Image.fromarray((ch_normalized * 255).astype(np.uint8))
            img = img.resize((size, size), Image.BILINEAR)
            return np.array(img, dtype=np.float32) / 255.0

        spec_tensor = np.stack([
            resize_channel(np.log1p(mag), self.spec_size),
            resize_channel(phase, self.spec_size),
            resize_channel(inst_freq, self.spec_size),
        ]).astype(np.float32)

        # 3. HOS features (20,)
        hos_tensor = self._extract_hos(iq_crop)

        # 4. Cyclo features (512,)
        cyclo_tensor = self._extract_cyclo(iq_crop)

        return {
            'iq': torch.from_numpy(iq_tensor),
            'spectrogram': torch.from_numpy(spec_tensor),
            'hos': torch.from_numpy(np.nan_to_num(hos_tensor)),
            'cyclo': torch.from_numpy(np.nan_to_num(cyclo_tensor)),
            'label': torch.tensor(label, dtype=torch.long),
        }

    def _augment(self, iq):
        """RF-aware augmentation."""
        # AWGN (80% prob, SNR 0-30 dB)
        if np.random.random() < 0.8:
            snr_db = np.random.uniform(0, 30)
            sig_power = np.mean(np.abs(iq)**2)
            noise_power = sig_power / (10**(snr_db/10))
            noise = np.sqrt(noise_power/2) * (np.random.randn(len(iq)) + 1j*np.random.randn(len(iq)))
            iq = iq + noise

        # CFO (50% prob, max 500 Hz)
        if np.random.random() < 0.5:
            cfo = np.random.uniform(-500, 500)
            t = np.arange(len(iq)) / 1.024e6
            iq = iq * np.exp(1j * 2 * np.pi * cfo * t)

        # Time shift (50% prob)
        if np.random.random() < 0.5:
            shift = np.random.randint(0, len(iq)//4)
            iq = np.roll(iq, shift)

        # Amplitude scaling (50% prob)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.5, 2.0)
            iq = iq * scale

        return iq


def create_splits(dataset, train_ratio=0.64, val_ratio=0.16):
    """Temporal split per class."""
    train_idx, val_idx, test_idx = [], [], []
    labels = np.array(dataset.labels_encoded)

    for cls in range(dataset.num_classes):
        cls_indices = np.where(labels == cls)[0]
        n = len(cls_indices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_idx.extend(cls_indices[:train_end].tolist())
        val_idx.extend(cls_indices[train_end:val_end].tolist())
        test_idx.extend(cls_indices[val_end:].tolist())

    return train_idx, val_idx, test_idx


# ============================================================================
# RFML-MoE EXPERT ARCHITECTURES
# ============================================================================

class ComplexConv1d(nn.Module):
    """1D convolution that processes I and Q channels with shared weights."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_ch*2, out_ch*2, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(min(8, out_ch*2), out_ch*2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return F.gelu(x)


class IQCNNTransformer(nn.Module):
    """RFML-MoE IQ Expert: 1D CNN encoder + Transformer.
    Processes raw IQ (2, N) through conv blocks then self-attention."""
    def __init__(self, num_classes=7, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        # CNN encoder: downsample IQ
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, embed_dim, 3, stride=2, padding=1), nn.BatchNorm1d(embed_dim), nn.GELU(),
        )
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
        self.name = "IQ-CNN-Transformer"

    def forward(self, x):
        x = self.cnn(x)  # (B, embed, T')
        x = x.permute(0, 2, 1)  # (B, T', embed)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # (B, embed, T')
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class SpectrogramCNN(nn.Module):
    """RFML-MoE Spectrogram Expert: Lightweight CNN on 3-channel spectrogram.
    Using a compact CNN instead of EfficientNet for 800 samples."""
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(256*4*4, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.name = "Spectrogram-CNN"

    def forward(self, x):
        return self.head(self.features(x))


class FTTransformerHOS(nn.Module):
    """RFML-MoE HOS Expert: Feature Tokenizer Transformer for tabular HOS data."""
    def __init__(self, num_features=20, num_classes=7, embed_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        # Per-feature embedding
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(nn.Linear(1, embed_dim), nn.GELU()) for _ in range(num_features)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_features + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.2), nn.Linear(embed_dim, num_classes))
        self.name = "FT-Transformer-HOS"

    def forward(self, x):
        B = x.size(0)
        tokens = []
        for i, embed in enumerate(self.feature_embeddings):
            tokens.append(embed(x[:, i:i+1]))  # (B, embed_dim)
        tokens = torch.stack(tokens, dim=1)  # (B, N_feat, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]

        tokens = self.transformer(tokens)
        cls_out = tokens[:, 0]
        return self.head(cls_out)


class DilatedTCN(nn.Module):
    """RFML-MoE Cyclo Expert: Temporal Convolutional Network with dilated convolutions."""
    def __init__(self, in_dim=512, num_classes=7, channels=64, n_layers=6):
        super().__init__()
        self.input_proj = nn.Conv1d(1, channels, 1)
        layers = []
        for i in range(n_layers):
            dilation = 2**i
            layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(0.1),
            ))
        self.layers = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.name = "Dilated-TCN-Cyclo"

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, 512)
        x = self.input_proj(x)
        for layer in self.layers:
            residual = x
            x = layer(x) + residual
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ============================================================================
# SOTA ALTERNATIVE ARCHITECTURES
# ============================================================================

class ResNet1D(nn.Module):
    """ResNet-style 1D CNN for raw IQ signals.
    Based on AMR literature achieving >93% on RadioML 2018.01A."""
    def __init__(self, num_classes=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))
        self.name = "ResNet1D"

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + identity)


class CLDNN(nn.Module):
    """CNN-LSTM-DNN: Classic hybrid from AMR literature.
    CNN extracts local features → LSTM captures temporal → DNN classifies."""
    def __init__(self, num_classes=7, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.lstm = nn.LSTM(128, hidden, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden*2, 128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, num_classes)
        )
        self.name = "CLDNN"

    def forward(self, x):
        x = self.cnn(x)  # (B, 128, T')
        x = x.permute(0, 2, 1)  # (B, T', 128)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # last hidden state
        return self.head(x)


class ConvNeXtTinySpec(nn.Module):
    """ConvNeXt-inspired architecture for spectrograms.
    Modern CNN with depthwise convolutions and inverted bottlenecks."""
    def __init__(self, num_classes=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 4, stride=4), nn.BatchNorm2d(48),
        )
        self.stages = nn.Sequential(
            ConvNeXtBlock(48, 96),
            nn.MaxPool2d(2),
            ConvNeXtBlock(96, 192),
            nn.MaxPool2d(2),
            ConvNeXtBlock(192, 384),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(384),
            nn.Dropout(0.3), nn.Linear(384, num_classes)
        )
        self.name = "ConvNeXt-Tiny-Spec"

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_ch, in_ch, 7, padding=3, groups=in_ch)
        self.norm = nn.BatchNorm2d(in_ch)
        self.pw1 = nn.Conv2d(in_ch, in_ch * expansion, 1)
        self.pw2 = nn.Conv2d(in_ch * expansion, out_ch, 1)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.dw_conv(x)
        out = self.norm(out)
        out = F.gelu(self.pw1(out))
        out = self.pw2(out)
        return F.gelu(out + identity)


class LightweightViT(nn.Module):
    """Lightweight Vision Transformer for spectrograms.
    Small patch size, few layers - designed for small datasets."""
    def __init__(self, num_classes=7, img_size=128, patch_size=16, embed_dim=128,
                 n_heads=4, n_layers=3):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
        self.name = "Lightweight-ViT-Spec"

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, embed, H', W')
        x = x.flatten(2).permute(0, 2, 1)  # (B, n_patches, embed)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x + self.pos_embed)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for 1D signals."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excite(w).view(b, c, 1)
        return x * w


class SEResNet1D(nn.Module):
    """SE-ResNet: SOTA on RadioML 2018.01A (63.7% avg, 98.9% peak).
    ResNet + Squeeze-and-Excitation blocks for channel attention."""
    def __init__(self, num_classes=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))
        self.name = "SE-ResNet1D"

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [SEResBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(SEResBlock1D(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class SEResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock1D(out_ch)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.gelu(out + identity)


class MCLDNN(nn.Module):
    """Multi-Channel LSTM-DNN: processes I, Q, and IQ jointly.
    SOTA from 2020, 60.83% avg on RadioML 2016.10a."""
    def __init__(self, num_classes=7, hidden=64):
        super().__init__()
        # Separate CNN branches for I, Q, and combined IQ
        self.cnn_i = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
        )
        self.cnn_q = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
        )
        self.cnn_iq = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
        )
        # Merge + LSTM
        self.merge_conv = nn.Sequential(
            nn.Conv1d(96, 64, 3, stride=2, padding=1), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.lstm = nn.LSTM(64, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden*2, 128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, num_classes)
        )
        self.name = "MCLDNN"

    def forward(self, x):
        i_ch = x[:, 0:1, :]  # (B, 1, N)
        q_ch = x[:, 1:2, :]  # (B, 1, N)
        fi = self.cnn_i(i_ch)
        fq = self.cnn_q(q_ch)
        fiq = self.cnn_iq(x)
        merged = torch.cat([fi, fq, fiq], dim=1)  # (B, 96, N)
        merged = self.merge_conv(merged)  # (B, 64, N/4)
        merged = merged.permute(0, 2, 1)
        lstm_out, _ = self.lstm(merged)
        return self.head(lstm_out[:, -1, :])


class InceptionTime1D(nn.Module):
    """InceptionTime: State-of-the-art for time series classification.
    Uses parallel convolutions at multiple scales."""
    def __init__(self, num_classes=7, n_filters=32):
        super().__init__()
        self.inception1 = InceptionBlock(2, n_filters)
        self.inception2 = InceptionBlock(n_filters*4, n_filters*2)
        self.inception3 = InceptionBlock(n_filters*8, n_filters*4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_filters*16, num_classes))
        self.name = "InceptionTime-1D"

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, n_filters):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, n_filters, 1)
        self.conv_10 = nn.Conv1d(n_filters, n_filters, 11, padding=5)
        self.conv_20 = nn.Conv1d(n_filters, n_filters, 21, padding=10)
        self.conv_40 = nn.Conv1d(n_filters, n_filters, 41, padding=20)
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, n_filters, 1)
        )
        self.bn = nn.BatchNorm1d(n_filters * 4)
        self.downsample = nn.Conv1d(in_ch, n_filters*4, 1) if in_ch != n_filters*4 else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        b = self.bottleneck(x)
        out = torch.cat([self.conv_10(b), self.conv_20(b), self.conv_40(b), self.maxpool(x)], dim=1)
        out = F.gelu(self.bn(out))
        # Align lengths if needed
        min_len = min(out.size(-1), identity.size(-1))
        return F.gelu(out[..., :min_len] + identity[..., :min_len])


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, val_loader, epochs=80, lr=1e-3, patience=15):
    """Train with early stopping and cosine LR scheduling."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch in train_loader:
            label = batch['label'].to(DEVICE)

            # Select input based on model type
            if hasattr(model, 'name'):
                if 'Spec' in model.name or 'ConvNeXt' in model.name or 'ViT' in model.name:
                    x = batch['spectrogram'].to(DEVICE)
                elif 'HOS' in model.name or 'FT-Transformer' in model.name:
                    x = batch['hos'].to(DEVICE)
                elif 'TCN' in model.name or 'Cyclo' in model.name:
                    x = batch['cyclo'].to(DEVICE)
                else:
                    x = batch['iq'].to(DEVICE)
            else:
                x = batch['iq'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * label.size(0)
            train_correct += (logits.argmax(1) == label).sum().item()
            train_total += label.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                label = batch['label'].to(DEVICE)
                if hasattr(model, 'name'):
                    if 'Spec' in model.name or 'ConvNeXt' in model.name or 'ViT' in model.name:
                        x = batch['spectrogram'].to(DEVICE)
                    elif 'HOS' in model.name or 'FT-Transformer' in model.name:
                        x = batch['hos'].to(DEVICE)
                    elif 'TCN' in model.name or 'Cyclo' in model.name:
                        x = batch['cyclo'].to(DEVICE)
                    else:
                        x = batch['iq'].to(DEVICE)
                else:
                    x = batch['iq'].to(DEVICE)

                logits = model(x)
                val_correct += (logits.argmax(1) == label).sum().item()
                val_total += label.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or no_improve == 0:
            print(f"  Epoch {epoch+1:3d}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
                  f"loss={train_loss/train_total:.4f} {'*' if no_improve==0 else ''}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
        # Save checkpoint to scratch
        ckpt_path = f"{SCRATCH}/checkpoints/{model.name.replace(' ', '_')}.pt"
        torch.save(best_state, ckpt_path)
    return model, best_val_acc


def evaluate_model(model, test_loader, classes):
    """Evaluate model on test set."""
    model.eval()
    model = model.to(DEVICE)
    all_preds, all_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            label = batch['label'].to(DEVICE)
            if hasattr(model, 'name'):
                if 'Spec' in model.name or 'ConvNeXt' in model.name or 'ViT' in model.name:
                    x = batch['spectrogram'].to(DEVICE)
                elif 'HOS' in model.name or 'FT-Transformer' in model.name:
                    x = batch['hos'].to(DEVICE)
                elif 'TCN' in model.name or 'Cyclo' in model.name:
                    x = batch['cyclo'].to(DEVICE)
                else:
                    x = batch['iq'].to(DEVICE)
            else:
                x = batch['iq'].to(DEVICE)

            logits = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    inference_time = (time.time() - start_time) / len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'per_class': {cls: report[cls] for cls in classes if cls in report},
        'confusion_matrix': cm.tolist(),
        'inference_time_ms': inference_time * 1000,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("NEURAL NETWORK ARCHITECTURE COMPARISON FOR RF SIGNAL CLASSIFICATION")
    print("="*80)
    print(f"Device: {DEVICE}")

    # Create dataset
    print("\n--- Loading Dataset ---")
    dataset = RFDataset(augment=False, iq_len=32768, spec_size=128, cyclo_dim=512)
    train_idx, val_idx, test_idx = create_splits(dataset)

    # Augmented training set
    aug_dataset = RFDataset(augment=True, iq_len=32768, spec_size=128, cyclo_dim=512)

    train_loader = DataLoader(Subset(aug_dataset, train_idx), batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False, num_workers=0)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    classes = dataset.classes

    # Define all models
    models = {
        # RFML-MoE experts
        'IQ-CNN-Transformer': IQCNNTransformer(num_classes=len(classes)),
        'Spectrogram-CNN': SpectrogramCNN(num_classes=len(classes)),
        'FT-Transformer-HOS': FTTransformerHOS(num_features=20, num_classes=len(classes)),
        'Dilated-TCN-Cyclo': DilatedTCN(in_dim=512, num_classes=len(classes)),
        # SOTA alternatives
        'ResNet1D': ResNet1D(num_classes=len(classes)),
        'SE-ResNet1D': SEResNet1D(num_classes=len(classes)),
        'MCLDNN': MCLDNN(num_classes=len(classes)),
        'CLDNN': CLDNN(num_classes=len(classes)),
        'ConvNeXt-Tiny-Spec': ConvNeXtTinySpec(num_classes=len(classes)),
        'Lightweight-ViT-Spec': LightweightViT(num_classes=len(classes)),
        'InceptionTime-1D': InceptionTime1D(num_classes=len(classes)),
    }

    results = {}

    for name, model in models.items():
        n_params = count_params(model)
        print(f"\n{'='*70}")
        print(f"  {name} ({n_params:,} params)")
        print(f"{'='*70}")

        try:
            model, best_val = train_model(model, train_loader, val_loader,
                                          epochs=80, lr=1e-3, patience=15)
            print(f"  Best validation accuracy: {best_val:.3f}")

            eval_result = evaluate_model(model, test_loader, classes)
            eval_result['params'] = n_params
            eval_result['best_val_acc'] = best_val
            results[name] = eval_result

            print(f"  Test accuracy: {eval_result['accuracy']:.3f}")
            print(f"  F1-macro: {eval_result['f1_macro']:.3f}")
            print(f"  Inference: {eval_result['inference_time_ms']:.1f} ms/sample")
            print(f"  Per-class:")
            for cls in classes:
                if cls in eval_result['per_class']:
                    r = eval_result['per_class'][cls]
                    print(f"    {cls:15s}: P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results[name] = {'accuracy': 0, 'f1_macro': 0, 'error': str(e)}

    # Add statistical baselines for comparison
    print("\n" + "="*70)
    print("STATISTICAL BASELINES (from previous study)")
    print("="*70)
    baselines = {
        'RF-Baseline-17feat': {'accuracy': 0.975, 'f1_macro': 0.971, 'params': 0, 'inference_time_ms': 0.5},
        'RF-IQ-Stat-37feat': {'accuracy': 0.988, 'f1_macro': 0.986, 'params': 0, 'inference_time_ms': 0.5},
        'RF-Spectrogram-37feat': {'accuracy': 1.000, 'f1_macro': 1.000, 'params': 0, 'inference_time_ms': 1.0},
        'RF-Combined-158feat': {'accuracy': 1.000, 'f1_macro': 1.000, 'params': 0, 'inference_time_ms': 2.0},
    }
    results.update(baselines)

    # Final summary
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"\n{'Model':<30} {'Params':>10} {'Test Acc':>10} {'F1-macro':>10} {'ms/sample':>10}")
    print("-" * 72)

    for name in sorted(results.keys(), key=lambda k: results[k].get('accuracy', 0), reverse=True):
        r = results[name]
        params = r.get('params', 0)
        acc = r.get('accuracy', 0)
        f1 = r.get('f1_macro', 0)
        ms = r.get('inference_time_ms', 0)
        params_str = f"{params:,}" if params > 0 else "~200KB"
        print(f"{name:<30} {params_str:>10} {acc:>10.3f} {f1:>10.3f} {ms:>10.1f}")

    # Save results
    with open('nn_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to nn_comparison_results.json")

    return results


if __name__ == '__main__':
    main()
