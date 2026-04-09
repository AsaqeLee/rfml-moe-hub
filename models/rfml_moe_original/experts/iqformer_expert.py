"""IQFormer Expert: Multi-modal IQ + Time-Frequency transformer with Dynamic Fusion Embedding.

Implements the IQFormer architecture from:
    "IQFormer: A Dual-Branch Transformer for Automatic Modulation Classification
     via Dynamic Fusion of IQ and Time-Frequency Representations"
    IEEE Transactions on Cognitive Communications and Networking (TCCN), 2025.

Key contributions reproduced here:
- Dynamic Fusion Embedding (DFE): learnable gating between IQ and TF branches
- StagedTransformerBlock: local depthwise conv + global self-attention hybrid
- On-the-fly STFT inside the model (no preprocessing required)

Drop-in replacement for IQExpert in the MoE pipeline.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dynamic Fusion Embedding (DFE)
# ---------------------------------------------------------------------------

class _IQBranch(nn.Module):
    """1D convolutional feature extractor for raw IQ samples.

    Progressively downsamples the time axis via stride-2 convolutions and
    widens the channel dimension, mirroring the IQFormer IQ-branch design.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        # Input: (batch, 2, N)  ->  output: (batch, embed_dim, T)
        # 5 stride-2 blocks give 32768 / 32 = 1024 time steps (T=1024)
        self.layers = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, embed_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, N) raw IQ tensor.

        Returns:
            (batch, embed_dim, T) feature map.
        """
        return self.layers(x)


class _TFBranch(nn.Module):
    """Time-frequency branch: on-the-fly STFT + 2D conv feature extractor.

    Computes a short-time Fourier transform inside the forward pass so the
    model is fully end-to-end without requiring offline preprocessing.

    STFT parameters (from IQFormer):
        fft_size  = 256  ->  129 frequency bins (one-sided)
        hop_length = 64
        For N=32768: number of frames = (32768 - 256) / 64 + 1 = 511 ≈ 512
    """

    # Public STFT config so callers can inspect / override.
    FFT_SIZE: int = 256
    HOP_LENGTH: int = 64

    def __init__(self, embed_dim: int = 256, target_seq_len: int = 1024):
        super().__init__()
        self.fft_size = self.FFT_SIZE
        self.hop_length = self.HOP_LENGTH
        self.target_seq_len = target_seq_len

        # Number of one-sided frequency bins after STFT: fft_size // 2 + 1 = 129
        freq_bins = self.fft_size // 2 + 1  # 129

        # 2D conv backbone operating on the spectrogram magnitude.
        # Input: (batch, 2, freq_bins, frames) — two channels (I mag + Q mag)
        self.layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # After 2D convs the spatial dims are reduced; we need to collapse freq
        # and project to embed_dim over the time axis.
        # Frequency collapse: adaptive pool to height=1, then project channels.
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # (batch, C, 1, W)
        self.channel_proj = nn.Conv1d(256, embed_dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(embed_dim)

        # Temporal alignment: adaptive pool / interpolate to target_seq_len
        self.time_pool = nn.AdaptiveAvgPool1d(target_seq_len)

    def _compute_stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample STFT and return magnitude spectrogram.

        Args:
            x: (batch, 2, N) IQ tensor (I on ch-0, Q on ch-1).

        Returns:
            (batch, 2, freq_bins, frames) magnitude spectrogram.
        """
        batch = x.shape[0]
        window = torch.hann_window(self.fft_size, device=x.device, dtype=x.dtype)

        specs = []
        for ch in range(2):  # I channel, then Q channel
            sig = x[:, ch, :]  # (batch, N)
            # torch.stft expects (batch, N) or (N,); use batched form
            stft = torch.stft(
                sig,
                n_fft=self.fft_size,
                hop_length=self.hop_length,
                win_length=self.fft_size,
                window=window,
                center=False,
                return_complex=True,
            )  # (batch, freq_bins, frames)
            specs.append(stft.abs())  # magnitude

        return torch.stack(specs, dim=1)  # (batch, 2, freq_bins, frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, N) raw IQ tensor.

        Returns:
            (batch, embed_dim, target_seq_len) feature map aligned with IQ branch.
        """
        spec = self._compute_stft(x)           # (batch, 2, freq_bins, frames)
        feat = self.layers(spec)               # (batch, 256, h, w)
        feat = self.freq_pool(feat)            # (batch, 256, 1, w)
        feat = feat.squeeze(2)                 # (batch, 256, w)
        feat = self.channel_proj(feat)         # (batch, embed_dim, w)
        feat = self.norm(feat)
        feat = self.time_pool(feat)            # (batch, embed_dim, target_seq_len)
        return feat


class DynamicFusionEmbedding(nn.Module):
    """Dynamic Fusion Embedding (DFE) from IQFormer (TCCN 2025).

    Jointly processes raw IQ samples through two parallel branches:
      - IQ branch  : 1D conv stack on raw I and Q channels.
      - TF branch  : on-the-fly STFT followed by 2D conv stack on the
                     magnitude spectrogram.

    A position-wise sigmoid gate learns to adaptively combine both branches:
        gate  = sigmoid(Linear(concat(iq_feat, tf_feat)))
        fused = gate * iq_feat + (1 - gate) * tf_feat

    The output is a sequence of fused token embeddings ready for the
    transformer stack.

    Args:
        embed_dim:       Embedding dimension for transformer tokens (default 256).
        seq_len:         Output sequence length in tokens (default 1024).
        dropout:         Dropout rate applied after fusion (default 0.1).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.iq_branch = _IQBranch(embed_dim=embed_dim)
        self.tf_branch = _TFBranch(embed_dim=embed_dim, target_seq_len=seq_len)

        # IQ branch may not produce exactly seq_len tokens; align with pool.
        self.iq_pool = nn.AdaptiveAvgPool1d(seq_len)

        # Gate network: operates on concatenated channel dim per token position.
        # Input: (batch, seq_len, 2*embed_dim)  ->  gate: (batch, seq_len, embed_dim)
        self.gate = nn.Linear(2 * embed_dim, embed_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fused token embeddings.

        Args:
            x: Raw IQ tensor of shape (batch, 2, N).

        Returns:
            Token embeddings of shape (batch, seq_len, embed_dim).
        """
        iq_feat = self.iq_branch(x)       # (batch, embed_dim, T_iq)
        iq_feat = self.iq_pool(iq_feat)   # (batch, embed_dim, seq_len)

        tf_feat = self.tf_branch(x)       # (batch, embed_dim, seq_len)

        # Transpose to (batch, seq_len, embed_dim) for token-wise ops
        iq_tokens = iq_feat.permute(0, 2, 1)  # (batch, seq_len, embed_dim)
        tf_tokens = tf_feat.permute(0, 2, 1)  # (batch, seq_len, embed_dim)

        # Dynamic gate: sigmoid(Linear(concat)) in channel dim
        combined = torch.cat([iq_tokens, tf_tokens], dim=-1)  # (batch, seq_len, 2*embed_dim)
        gate = torch.sigmoid(self.gate(combined))              # (batch, seq_len, embed_dim)

        fused = gate * iq_tokens + (1.0 - gate) * tf_tokens   # (batch, seq_len, embed_dim)
        return self.dropout(fused)


# ---------------------------------------------------------------------------
# Staged Transformer Block
# ---------------------------------------------------------------------------

class StagedTransformerBlock(nn.Module):
    """Staged transformer block from IQFormer (TCCN 2025).

    Captures local and global dependencies in two successive stages:
      Stage 1 — Local:  Depthwise Conv1d (kernel_size=7) + BatchNorm + GELU.
      Stage 2 — Global: Multi-head Self-Attention.
    Both stages include residual connections and LayerNorm.  An FFN
    (Linear -> GELU -> Dropout -> Linear) follows the attention stage.

    Args:
        embed_dim:   Token embedding dimension.
        num_heads:   Number of attention heads.
        ffn_ratio:   Hidden-dim multiplier for the FFN (default 4).
        dropout:     Dropout rate for attention and FFN (default 0.1).
        kernel_size: Depthwise conv kernel for Stage 1 (default 7).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # --- Stage 1: Local depthwise conv ---
        padding = kernel_size // 2
        self.local_norm = nn.LayerNorm(embed_dim)
        # Depthwise: groups=embed_dim so each channel is filtered independently
        self.dw_conv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=embed_dim,
            bias=False,
        )
        self.local_bn = nn.BatchNorm1d(embed_dim)
        self.local_act = nn.GELU()

        # --- Stage 2: Global multi-head self-attention ---
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        # --- FFN ---
        ffn_dim = embed_dim * ffn_ratio
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token sequence of shape (batch, seq_len, embed_dim).

        Returns:
            Updated token sequence of shape (batch, seq_len, embed_dim).
        """
        # Stage 1: local depthwise conv with pre-norm and residual
        residual = x
        h = self.local_norm(x)
        # Conv1d expects (batch, channels, seq_len)
        h = h.permute(0, 2, 1)
        h = self.dw_conv(h)
        h = self.local_bn(h)
        h = self.local_act(h)
        h = h.permute(0, 2, 1)   # back to (batch, seq_len, embed_dim)
        x = residual + h

        # Stage 2: global self-attention with pre-norm and residual
        residual = x
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = self.attn_drop(h)
        x = residual + h

        # FFN with pre-norm and residual
        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = residual + h

        return x


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding (shared utility, matches iq_expert.py style)
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, non-learnable)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# IQFormerExpert
# ---------------------------------------------------------------------------

class IQFormerExpert(nn.Module):
    """IQFormer: Dual-branch transformer expert for automatic modulation classification.

    Implements the full IQFormer architecture from:
        IQFormer (IEEE TCCN 2025)

    Input:  (batch, 2, N) IQ tensor, channel 0 = I, channel 1 = Q, N=32768.
    Output: (batch, num_classes) via forward(),  (batch, 512) via get_embedding().

    Architecture:
        1. DynamicFusionEmbedding  — IQ branch + on-the-fly STFT TF branch,
                                     learnable gate -> fused tokens (seq_len=1024,
                                     embed_dim=384).
        2. Sinusoidal positional encoding.
        3. 4 x StagedTransformerBlock (embed_dim=384, num_heads=8, ffn_ratio=8).
        4. Global average pooling over the token sequence.
        5. Linear projection -> 512-dim embedding.
        6. Classification head -> num_classes logits.

    Parameter count: ~13.8 M with defaults (verified with num_params property).

    This class is a drop-in replacement for IQExpert in the MoE pipeline:
    same constructor signature, same forward() / get_embedding() / freeze() /
    unfreeze() / num_params interface.

    Args:
        num_classes: Number of output modulation classes (default 10).
        embed_dim:   Internal transformer token dimension (default 384).
        seq_len:     Number of tokens produced by DFE (default 1024).
        num_layers:  Number of StagedTransformerBlocks (default 4).
        num_heads:   Attention heads per block (default 8).
        ffn_ratio:   FFN hidden-dim multiplier per block (default 8).
        dropout:     Dropout rate throughout (default 0.1).
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 384,
        seq_len: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_ratio: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self._output_dim = 512  # matches IQExpert interface

        # 1. Dynamic Fusion Embedding
        self.dfe = DynamicFusionEmbedding(
            embed_dim=embed_dim,
            seq_len=seq_len,
            dropout=dropout,
        )

        # 2. Positional encoding
        self.pos_encoder = _PositionalEncoding(
            d_model=embed_dim,
            max_len=seq_len + 16,  # small buffer
            dropout=dropout,
        )

        # 3. Staged transformer stack
        self.blocks = nn.ModuleList([
            StagedTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Project pooled representation to 512-dim embedding
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.GELU(),
        )

        # 5. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self._output_dim, self._output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self._output_dim, num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Core feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run DFE + positional encoding + transformer -> 512-dim embedding.

        Args:
            x: (batch, 2, N) raw IQ tensor.

        Returns:
            (batch, 512) embedding vector.
        """
        # Dynamic fusion embedding: (batch, seq_len, embed_dim)
        tokens = self.dfe(x)

        # Positional encoding
        tokens = self.pos_encoder(tokens)

        # Staged transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # Global average pooling over token dimension
        pooled = tokens.mean(dim=1)  # (batch, embed_dim)

        # Project to 512-dim
        return self.embed_proj(pooled)  # (batch, 512)

    # ------------------------------------------------------------------
    # Public interface (matches IQExpert exactly)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return classification logits.

        Args:
            x: Raw IQ tensor of shape (batch, 2, N).

        Returns:
            Logits of shape (batch, num_classes).
        """
        emb = self._extract_features(x)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional embedding for the MoE gate.

        Args:
            x: Raw IQ tensor of shape (batch, 2, N).

        Returns:
            Embedding of shape (batch, 512).
        """
        return self._extract_features(x)

    def freeze(self):
        """Freeze all parameters for progressive / staged training."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
