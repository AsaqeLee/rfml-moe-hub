"""Hi-WaveTST Expert: Hybrid High-Frequency Wavelet-Transformer for IQ time-series.

Implements the Hi-WaveTST architecture (arXiv:2511.01254) with dual-stream
patching: raw temporal patches + wavelet packet decomposition (WPD) features
fused into super-tokens processed by a transformer encoder.

Drop-in replacement for IQExpert in the MoE pipeline.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Generalized Mean (GeM) Pooling
# ---------------------------------------------------------------------------

class GeM(nn.Module):
    """Learnable Generalized Mean Pooling.

    Pools a variable-length last dimension into a scalar per channel using
    a learnable exponent p (initialized to 3.0).
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, L) tensor — pools over the last dimension.

        Returns:
            (*, ) tensor with the last dimension collapsed.
        """
        return (x.clamp(min=self.eps).pow(self.p).mean(dim=-1)).pow(1.0 / self.p)


# ---------------------------------------------------------------------------
# Manual Haar Wavelet Packet Decomposition
# ---------------------------------------------------------------------------

def _haar_wpd(x: torch.Tensor, level: int) -> list[torch.Tensor]:
    """Recursive Haar wavelet packet decomposition (no external deps).

    At each level every node is split into approximation (low-pass) and
    detail (high-pass) coefficients using the Haar filters:
        low  = [1/sqrt(2),  1/sqrt(2)]
        high = [1/sqrt(2), -1/sqrt(2)]

    Args:
        x: (..., L) input signal — L must be divisible by 2^level.
        level: number of decomposition levels.

    Returns:
        List of 2^level coefficient tensors, each of shape (..., L // 2^level).
    """
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    nodes = [x]
    for _ in range(level):
        next_nodes: list[torch.Tensor] = []
        for node in nodes:
            # Reshape last dim into pairs: (..., L//2, 2)
            even = node[..., 0::2]
            odd = node[..., 1::2]
            low = (even + odd) * inv_sqrt2   # approximation
            high = (even - odd) * inv_sqrt2  # detail
            next_nodes.append(low)
            next_nodes.append(high)
        nodes = next_nodes
    return nodes  # 2^level tensors, each (..., L // 2^level)


# ---------------------------------------------------------------------------
# Hybrid Patching Layer
# ---------------------------------------------------------------------------

class HybridPatchEmbedding(nn.Module):
    """Dual-stream patch embedding: raw temporal + WPD-GeM features.

    Stream 1 — Raw patch: Linear projection of each non-overlapping patch.
    Stream 2 — WPD-GeM:   Multi-level Haar WPD per patch, GeM pooling over
               each subband, concatenation, then linear projection.

    The two streams are concatenated and projected to d_model to form
    a "super-token" per patch.

    Args:
        patch_size: Length of each non-overlapping patch (default 128).
        d_model:    Transformer model dimension (default 256).
        wpd_level:  Wavelet packet decomposition levels (default 3 -> 8 subbands).
    """

    def __init__(self, patch_size: int = 128, d_model: int = 256, wpd_level: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.wpd_level = wpd_level
        self.num_subbands = 2 ** wpd_level  # e.g. 8 for level 3

        # Stream 1: raw temporal patch -> d_model
        self.raw_proj = nn.Linear(patch_size, d_model)

        # Stream 2: WPD subbands -> GeM pool each -> concat -> d_model
        self.gem = GeM(p=3.0)
        # Each subband is GeM-pooled to a scalar -> num_subbands features total
        self.wpd_proj = nn.Linear(self.num_subbands, d_model)

        # Merge two streams into d_model
        self.merge = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, L) magnitude signal (L must be divisible by patch_size).

        Returns:
            (batch, num_patches, d_model) super-token embeddings.
        """
        B, L = x.shape
        num_patches = L // self.patch_size

        # Create non-overlapping patches: (B, num_patches, patch_size)
        patches = x.reshape(B, num_patches, self.patch_size)

        # --- Stream 1: raw patch embedding ---
        raw_emb = self.raw_proj(patches)  # (B, num_patches, d_model)

        # --- Stream 2: WPD + GeM ---
        # WPD on each patch independently
        # patches: (B, num_patches, patch_size) — treat first two dims as batch
        flat = patches.reshape(B * num_patches, self.patch_size)
        subbands = _haar_wpd(flat, self.wpd_level)  # list of 2^L tensors, each (B*P, subband_len)

        # GeM pool each subband to a scalar, then stack
        pooled = torch.stack([self.gem(sb) for sb in subbands], dim=-1)  # (B*P, num_subbands)
        pooled = pooled.reshape(B, num_patches, self.num_subbands)

        wpd_emb = self.wpd_proj(pooled)  # (B, num_patches, d_model)

        # --- Merge ---
        merged = torch.cat([raw_emb, wpd_emb], dim=-1)  # (B, num_patches, 2*d_model)
        return self.merge(merged)  # (B, num_patches, d_model)


# ---------------------------------------------------------------------------
# Hi-WaveTST Expert
# ---------------------------------------------------------------------------

class HiWaveTSTExpert(nn.Module):
    """Hi-WaveTST: Hybrid High-Frequency Wavelet-Transformer expert.

    Implements the Hi-WaveTST architecture (arXiv:2511.01254) for raw IQ
    modulation classification.

    Input:  (batch, 2, N) IQ tensor, channel 0 = I, channel 1 = Q, N=32768.
    Output: (batch, num_classes) via forward(),  (batch, 512) via get_embedding().

    Architecture:
        1. Compute magnitude |z(t)| = sqrt(I^2 + Q^2).
        2. HybridPatchEmbedding — raw temporal + WPD-GeM dual-stream patching
           into d_model-dim super-tokens.
        3. Learnable positional encoding.
        4. Transformer encoder (num_layers blocks, num_heads heads).
        5. Mean pooling -> Linear projection to embed_dim (512).
        6. Classification head -> num_classes logits.

    This class is a drop-in replacement for IQExpert in the MoE pipeline:
    same forward() / get_embedding() / freeze() / unfreeze() / num_params
    interface.

    Args:
        embed_dim:   Output embedding dimension (default 512, matches MoE gate).
        num_classes: Number of output modulation classes (default 10).
        d_model:     Internal transformer token dimension (default 256).
        patch_size:  Non-overlapping patch length (default 128).
        wpd_level:   WPD decomposition levels (default 3 -> 8 subbands).
        num_layers:  Transformer encoder layers (default 4).
        num_heads:   Attention heads per layer (default 8).
        dropout:     Dropout rate throughout (default 0.1).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 10,
        d_model: int = 256,
        patch_size: int = 128,
        wpd_level: int = 3,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 1. Hybrid patching (raw + WPD-GeM streams)
        self.patch_embed = HybridPatchEmbedding(
            patch_size=patch_size,
            d_model=d_model,
            wpd_level=wpd_level,
        )

        # 2. Learnable positional encoding
        # Max patches: 32768 / patch_size = 256 for default; add buffer
        max_patches = (32768 // patch_size) + 16
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # 4. Projection to embed_dim
        self.embed_proj = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # 5. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Core feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run magnitude -> hybrid patching -> transformer -> embed_dim embedding.

        Args:
            x: (batch, 2, N) raw IQ tensor.

        Returns:
            (batch, embed_dim) embedding vector.
        """
        # Compute magnitude: |z(t)| = sqrt(I^2 + Q^2)
        mag = torch.sqrt(x[:, 0, :] ** 2 + x[:, 1, :] ** 2)  # (B, N)

        # Hybrid patch embedding -> super-tokens
        tokens = self.patch_embed(mag)  # (B, num_patches, d_model)

        # Add learnable positional encoding
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]

        # Transformer encoder
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Mean pooling over patch/token dimension
        pooled = tokens.mean(dim=1)  # (B, d_model)

        # Project to embed_dim
        return self.embed_proj(pooled)  # (B, embed_dim)

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
