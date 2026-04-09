#!/usr/bin/env python3
"""
MultiScale-LWM-MaxViT Expert
==============================
Fuses LWM's IQ-to-2D-grid insight with MaxViT's multi-axis attention.

Key innovations:
1. Multi-scale grid reshape: 4 grid widths (64, 128, 256, 512) to capture
   different burst periodicities (0.64μs to 5.12μs at 100MSps)
2. MaxViT-style Block+Grid attention on each scale's 2D grid
3. Cross-scale fusion via attention pooling
4. MBConv for local texture extraction (modulation-specific patterns)

Input:  (B, 2, 32768) raw IQ
Output: (B, num_classes) logits or (B, embed_dim) embedding

Target: Beat LWM's 94.1% and approach MaxViT's 97.8% using raw IQ only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Building Blocks
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.GELU(),
            nn.Linear(mid, channels), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (from MaxViT/EfficientNet).
    Depthwise separable conv + SE attention + residual."""
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        mid = dim * expansion
        self.norm = nn.BatchNorm2d(dim)
        self.expand = nn.Conv2d(dim, mid, 1, bias=False)
        self.depthwise = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.se = SEBlock(mid)
        self.project = nn.Conv2d(mid, dim, 1, bias=False)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.expand(x))
        x = F.gelu(self.bn(self.depthwise(x)))
        x = self.se(x)
        x = self.project(x)
        x = self.dropout(x)
        return x + residual


class WindowAttention(nn.Module):
    """Block (window) self-attention — attend within local windows."""
    def __init__(self, dim, num_heads=4, window_size=7, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Relative position bias
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def _get_rel_pos_index(self, h, w):
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = coords.reshape(2, -1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += h - 1
        rel[:, :, 1] += w - 1
        rel[:, :, 0] *= 2 * w - 1
        return rel.sum(-1)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        # Pad if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape
        nH, nW = Hp // ws, Wp // ws

        # Partition into windows: (B*nH*nW, ws*ws, C)
        x = x.reshape(B, C, nH, ws, nW, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B * nH * nW, ws * ws, C)

        # Self-attention
        x = self.norm(x)
        qkv = self.qkv(x).reshape(-1, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Relative position bias
        idx = self._get_rel_pos_index(ws, ws).to(x.device)
        bias = self.rel_pos_bias[idx.view(-1)].view(ws*ws, ws*ws, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        x = self.proj(x)

        # Reverse partition
        x = x.reshape(B, nH, nW, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x


class GridAttention(nn.Module):
    """Dilated grid self-attention — attend across evenly spaced positions."""
    def __init__(self, dim, num_heads=4, grid_size=7, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.grid_size = grid_size
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        gs = self.grid_size

        pad_h = (gs - H % gs) % gs
        pad_w = (gs - W % gs) % gs
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape
        nH, nW = Hp // gs, Wp // gs

        # Dilated grid partition: (B*nH*nW, gs*gs, C)
        # Take every nH-th row and nW-th column
        x = x.reshape(B, C, gs, nH, gs, nW)
        x = x.permute(0, 3, 5, 2, 4, 1).reshape(B * nH * nW, gs * gs, C)

        # Self-attention (same as window but on dilated grid)
        x = self.norm(x)
        qkv = self.qkv(x).reshape(-1, gs * gs, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(-1, gs * gs, C)
        x = self.proj(x)

        # Reverse
        x = x.reshape(B, nH, nW, gs, gs, C)
        x = x.permute(0, 5, 3, 1, 4, 2).reshape(B, C, Hp, Wp)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x


class MaxViTBlock(nn.Module):
    """Single MaxViT block: MBConv → Block Attention → Grid Attention."""
    def __init__(self, dim, num_heads=4, window_size=7, grid_size=7,
                 expansion=4, dropout=0.1):
        super().__init__()
        self.mbconv = MBConv(dim, expansion, dropout)
        self.block_attn = WindowAttention(dim, num_heads, window_size, dropout)
        self.grid_attn = GridAttention(dim, num_heads, grid_size, dropout)

    def forward(self, x):
        x = self.mbconv(x)
        x = x + self.block_attn(x)  # Local attention + residual
        x = x + self.grid_attn(x)   # Global attention + residual
        return x


# ============================================================================
# Multi-Scale Grid Encoder
# ============================================================================

class ScaleEncoder(nn.Module):
    """Process one grid scale with patch embedding + MaxViT blocks."""
    def __init__(self, grid_h, grid_w, patch_size=8, dim=96,
                 num_blocks=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Patch embedding
        self.patch_embed = nn.Conv2d(2, dim, kernel_size=patch_size,
                                      stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(dim)

        n_patches_h = grid_h // patch_size
        n_patches_w = grid_w // patch_size
        self.n_patches = n_patches_h * n_patches_w

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, n_patches_h, n_patches_w))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Adaptive window/grid sizes based on patch grid dimensions
        ws = min(7, n_patches_h, n_patches_w)
        gs = min(7, n_patches_h, n_patches_w)

        # MaxViT blocks
        self.blocks = nn.ModuleList([
            MaxViTBlock(dim, num_heads, ws, gs, expansion=4, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Pool to fixed-size output
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, iq):
        """
        Args:
            iq: (B, 2, N) raw IQ tensor
        Returns:
            (B, dim) scale embedding
        """
        B, _, N = iq.shape
        needed = self.grid_h * self.grid_w

        # Trim or pad to exact grid size
        if N >= needed:
            iq_grid = iq[:, :, :needed].reshape(B, 2, self.grid_h, self.grid_w)
        else:
            iq_padded = F.pad(iq, (0, needed - N))
            iq_grid = iq_padded.reshape(B, 2, self.grid_h, self.grid_w)

        # Patch embedding
        x = self.patch_embed(iq_grid)  # (B, dim, pH, pW)
        x = x + self.pos_embed

        # MaxViT blocks
        for block in self.blocks:
            x = block(x)

        # Pool
        return self.pool(x).flatten(1)  # (B, dim)


# ============================================================================
# Main Model: MultiScale-LWM-MaxViT
# ============================================================================

class MultiScaleLWMMaxViT(nn.Module):
    """Multi-Scale LWM with MaxViT attention.

    Processes raw IQ at 4 different grid widths simultaneously,
    each capturing different burst periodicities:
      - Scale 64:  0.64μs period (high-rate control channels)
      - Scale 128: 1.28μs period (standard FHSS hop rate)
      - Scale 256: 2.56μs period (longer burst patterns)
      - Scale 512: 5.12μs period (video frame timing)

    Each scale uses MaxViT blocks (MBConv + Block Attn + Grid Attn).
    Cross-scale fusion via attention-weighted aggregation.

    Args:
        embed_dim:   Final embedding dimension for MoE (default 512)
        num_classes: Number of output classes
        dim:         Per-scale channel dimension (default 96)
        num_blocks:  MaxViT blocks per scale (default 2)
        num_heads:   Attention heads (default 4)
        patch_size:  Patch size for embedding (default 8)
        grid_widths: List of grid widths for multi-scale (default [64,128,256,512])
        dropout:     Dropout rate (default 0.1)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 37,
        dim: int = 96,
        num_blocks: int = 2,
        num_heads: int = 4,
        patch_size: int = 8,
        grid_widths: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if grid_widths is None:
            grid_widths = [64, 128, 256, 512]

        self.grid_widths = grid_widths
        self.num_scales = len(grid_widths)
        self.name = "MultiScale-LWM-MaxViT"

        # Per-scale encoders
        self.scale_encoders = nn.ModuleList()
        for gw in grid_widths:
            gh = max(16, 32768 // gw)  # Adjust height to use all 32768 samples
            self.scale_encoders.append(
                ScaleEncoder(gh, gw, patch_size, dim, num_blocks, num_heads, dropout)
            )

        # Cross-scale attention fusion
        self.scale_query = nn.Parameter(torch.randn(1, 1, dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, dropout=dropout,
                                                  batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)

        # Embedding projection
        self.embed_proj = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _extract_features(self, iq):
        """Extract multi-scale features from raw IQ.

        Args:
            iq: (B, 2, N) raw IQ tensor

        Returns:
            (B, embed_dim) embedding vector
        """
        B = iq.size(0)

        # Encode each scale independently
        scale_features = []
        for encoder in self.scale_encoders:
            feat = encoder(iq)  # (B, dim)
            scale_features.append(feat)

        # Stack: (B, num_scales, dim)
        scale_stack = torch.stack(scale_features, dim=1)

        # Cross-scale attention: learnable query attends to all scales
        query = self.scale_query.expand(B, -1, -1)  # (B, 1, dim)
        fused, _ = self.cross_attn(query, scale_stack, scale_stack)  # (B, 1, dim)
        fused = self.cross_norm(fused.squeeze(1))  # (B, dim)

        # Project to embedding
        embedding = self.embed_proj(fused)  # (B, embed_dim)
        return embedding

    def forward(self, x):
        """Classification forward pass."""
        embedding = self._extract_features(x)
        return self.classifier(embedding)

    def get_embedding(self, x):
        """Get embedding for MoE routing."""
        return self._extract_features(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        # Keep classifier trainable
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Quick Benchmark Script
# ============================================================================

if __name__ == '__main__':
    import argparse, os, sys, json, time, glob
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='rfuav', choices=['rfuav', 'droneRFb'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-segments', type=int, default=100000)
    parser.add_argument('--segment-length', type=int, default=32768)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--dim', type=int, default=96)
    parser.add_argument('--num-blocks', type=int, default=2)
    parser.add_argument('--raw-dir', default='/home/rax/mtp/raw')
    parser.add_argument('--rfb-dir', default='/home/rax/mtp/droneRFb/extracted/twin_droneRF')
    parser.add_argument('--result-dir', default='/home/rax/mtp/results')
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ---- RFUAV Dataset ----
    class RFUAVIQDataset(Dataset):
        def __init__(self, segments, labels, seg_len, augment=False):
            self.segments = segments
            self.labels = labels
            self.seg_len = seg_len
            self.augment = augment

        def __len__(self):
            return len(self.segments)

        def __getitem__(self, idx):
            fpath, start = self.segments[idx]
            raw = np.fromfile(fpath, dtype=np.float32, offset=start * 2 * 4,
                              count=self.seg_len * 2)
            if len(raw) < self.seg_len * 2:
                raw = np.pad(raw, (0, self.seg_len * 2 - len(raw)))
            I = raw[0::2].astype(np.float32)
            Q = raw[1::2].astype(np.float32)
            iq = np.stack([I, Q])
            iq = iq - iq.mean(axis=1, keepdims=True)

            if self.augment:
                # AWGN
                if np.random.random() < 0.8:
                    snr = np.random.uniform(5, 30)
                    sig_p = np.mean(iq**2)
                    noise_p = sig_p / (10**(snr/10))
                    iq += np.random.randn(*iq.shape).astype(np.float32) * np.sqrt(noise_p)
                # Time shift
                if np.random.random() < 0.5:
                    shift = np.random.randint(0, self.seg_len // 4)
                    iq = np.roll(iq, shift, axis=1)

            return torch.from_numpy(iq), self.labels[idx]

    # ---- DroneRFb Dataset ----
    class DroneRFbIQDataset(Dataset):
        def __init__(self, files, labels, seg_len):
            self.files = files
            self.labels = labels
            self.seg_len = seg_len

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            import h5py
            f = h5py.File(self.files[idx], 'r')
            I = np.array(f['I']).flatten()[:self.seg_len].astype(np.float32)
            Q = np.array(f['Q']).flatten()[:self.seg_len].astype(np.float32)
            f.close()
            if len(I) < self.seg_len:
                I = np.pad(I, (0, self.seg_len - len(I)))
                Q = np.pad(Q, (0, self.seg_len - len(Q)))
            iq = np.stack([I, Q])
            iq = iq - iq.mean(axis=1, keepdims=True)
            return torch.from_numpy(iq), self.labels[idx]

    # ---- Data Loading ----
    def load_rfuav_data():
        import re
        entries = []
        for drone in sorted(os.listdir(args.raw_dir)):
            dp = os.path.join(args.raw_dir, drone)
            if not os.path.isdir(dp): continue
            iq_files = sorted(
                glob.glob(os.path.join(dp, drone, 'VTSBW=*', '*.iq')) or
                glob.glob(os.path.join(dp, 'VTSBW=*', '*.iq')) or
                glob.glob(os.path.join(dp, drone, '*.iq'))
            )
            for f in iq_files:
                entries.append((f, drone))

        drone_names = sorted(set(d for _, d in entries))
        c2i = {n: i for i, n in enumerate(drone_names)}
        num_classes = len(drone_names)
        print(f"  RFUAV: {len(entries)} files, {num_classes} drones", flush=True)

        train_segs, train_labels, val_segs, val_labels = [], [], [], []
        files_by_drone = {}
        for fp, dn in entries:
            files_by_drone.setdefault(dn, []).append(fp)

        cap = max(10, args.max_segments // len(files_by_drone)) if args.max_segments > 0 else 999999

        for dn, flist in files_by_drone.items():
            label = c2i[dn]
            segs = []
            for fp in flist:
                n_samples = os.path.getsize(fp) // 8
                for si in range(n_samples // args.segment_length):
                    segs.append((fp, si * args.segment_length))
            segs = segs[:cap]
            split = int(len(segs) * 0.8)
            train_segs.extend(segs[:split])
            train_labels.extend([label] * split)
            val_segs.extend(segs[split:])
            val_labels.extend([label] * (len(segs) - split))

        print(f"  Train: {len(train_segs)}, Val: {len(val_segs)}", flush=True)
        train_ds = RFUAVIQDataset(train_segs, train_labels, args.segment_length, augment=True)
        val_ds = RFUAVIQDataset(val_segs, val_labels, args.segment_length, augment=False)
        return train_ds, val_ds, num_classes, drone_names

    def load_droneRFb_data():
        import re, h5py
        train_labels_map = {}
        with open(os.path.join(args.rfb_dir, 'train_labels.txt')) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    m = re.match(r'^([A-G])', parts[1])
                    train_labels_map[parts[0]] = m.group(1) if m else 'B'

        test_labels_map = {}
        with open(os.path.join(args.rfb_dir, 'test_labels.txt')) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    m = re.match(r'^([A-G])', parts[0])
                    test_labels_map[parts[1] + '.mat'] = m.group(1) if m else 'B'

        classes = sorted(set(list(train_labels_map.values()) + list(test_labels_map.values())))
        c2i = {c: i for i, c in enumerate(classes)}
        num_classes = len(classes)

        train_files, train_labels = [], []
        for fn in sorted(os.listdir(os.path.join(args.rfb_dir, 'train'))):
            if fn.endswith('.mat'):
                cls = train_labels_map.get(fn, 'B')
                train_files.append(os.path.join(args.rfb_dir, 'train', fn))
                train_labels.append(c2i[cls])

        test_files, test_labels = [], []
        for fn in sorted(os.listdir(os.path.join(args.rfb_dir, 'test'))):
            if fn.endswith('.mat'):
                cls = test_labels_map.get(fn, 'B')
                test_files.append(os.path.join(args.rfb_dir, 'test', fn))
                test_labels.append(c2i[cls])

        print(f"  DroneRFb: {len(train_files)} train, {len(test_files)} test, {num_classes} classes", flush=True)
        train_ds = DroneRFbIQDataset(train_files, train_labels, args.segment_length)
        val_ds = DroneRFbIQDataset(test_files, test_labels, args.segment_length)
        return train_ds, val_ds, num_classes, classes

    # ---- Training ----
    print("=" * 70, flush=True)
    print("MultiScale-LWM-MaxViT Benchmark", flush=True)
    print("=" * 70, flush=True)

    if args.dataset == 'rfuav':
        train_ds, val_ds, num_classes, class_names = load_rfuav_data()
    else:
        train_ds, val_ds, num_classes, class_names = load_droneRFb_data()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = MultiScaleLWMMaxViT(
        num_classes=num_classes, dim=args.dim, num_blocks=args.num_blocks,
    ).to(DEVICE)
    n_params = model.num_params
    print(f"  Model: MultiScale-LWM-MaxViT ({n_params/1e6:.2f}M params)", flush=True)
    print(f"  Grid widths: {model.grid_widths}", flush=True)
    print(f"  Scales: {model.num_scales}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=True)

    best_acc, best_state, no_improve = 0, None, 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        correct, total, epoch_loss = 0, 0, 0
        for iq, labels in train_loader:
            iq, labels = iq.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(iq)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item() * labels.size(0)
        scheduler.step()

        # Validate
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for iq, labels in val_loader:
                iq = iq.to(DEVICE)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(iq)
                vc += (logits.argmax(1) == labels.to(DEVICE)).sum().item()
                vt += labels.size(0)

        train_acc = correct / max(total, 1)
        val_acc = vc / max(vt, 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = '*'
        else:
            no_improve += 1
            marker = ''

        if (epoch + 1) % 3 == 0 or marker == '*':
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: train={train_acc:.4f} "
                  f"val={val_acc:.4f} loss={epoch_loss/max(total,1):.4f} "
                  f"[{time.time()-t0:.0f}s] {marker}", flush=True)

        if no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break

    # Load best and final eval
    if best_state:
        model.load_state_dict(best_state)
        ckpt_path = os.path.join(args.result_dir, f'ms_lwm_maxvit_{args.dataset}_best.pt')
        torch.save(best_state, ckpt_path)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for iq, labels in val_loader:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(iq.to(DEVICE))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    elapsed = time.time() - t0

    print(f"\n{'='*70}", flush=True)
    print(f"  RESULT: MultiScale-LWM-MaxViT on {args.dataset}", flush=True)
    print(f"  Accuracy: {acc:.4f} | F1-macro: {f1:.4f}", flush=True)
    print(f"  Params: {n_params/1e6:.2f}M | Time: {elapsed:.0f}s", flush=True)
    print(f"{'='*70}", flush=True)
    if len(class_names) <= 40:
        print(classification_report(all_labels, all_preds,
              target_names=class_names if isinstance(class_names[0], str) else [str(c) for c in class_names]),
              flush=True)

    result = {
        'model': 'MultiScale-LWM-MaxViT',
        'dataset': args.dataset,
        'accuracy': acc, 'f1_macro': f1,
        'params': n_params, 'params_M': n_params / 1e6,
        'train_time_sec': elapsed,
        'grid_widths': model.grid_widths,
        'dim': args.dim, 'num_blocks': args.num_blocks,
        'max_segments': args.max_segments,
        'best_val_acc': best_acc,
    }
    result_path = os.path.join(args.result_dir, f'ms_lwm_maxvit_{args.dataset}_result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {result_path}", flush=True)
