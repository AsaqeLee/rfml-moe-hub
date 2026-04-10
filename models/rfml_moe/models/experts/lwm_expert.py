"""LWM Expert: Large Wireless Model 1.1 adapted as MoE expert.

Based on the LWM 1.1 architecture (2024-2026):
    - Compact wireless foundation model (~2.5M parameters)
    - 2D patch segmentation on IQ resource grids
    - Masked Channel Modeling (MCM) pretraining with 40% masking
    - Embedding size: 128, sequence length: up to 512 tokens

Architecture:
    1. IQ-to-Grid: [B, 2, N] -> [B, 2, grid_h, grid_w]
       Reshapes raw time series into 2D resource grid (subcarriers x symbols)
    2. 2D Patch Embedding: Conv2d(2, d_model, patch_h x patch_w, stride=patch)
       Produces (grid_h/patch_h) x (grid_w/patch_w) = 16 x 32 = 512 tokens
       Plus learnable positional embeddings [512, d_model]
    3. Transformer Encoder: 6 layers, 4 heads, d_model=128, d_ff=512, pre-norm
    4. Projection Head: mean pool -> Linear(128, embed_dim=512) -> LayerNorm -> GELU

Drop-in replacement for IQExpert in the MoE pipeline:
    same forward() / get_embedding() / freeze() / unfreeze() / num_params interface.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 2D Patch Embedding
# ---------------------------------------------------------------------------

class _PatchEmbed2D(nn.Module):
    """Convert a 2D resource grid into a sequence of patch tokens.

    Args:
        d_model:   Token embedding dimension (default 128).
        patch_h:   Patch height in subcarrier axis (default 8).
        patch_w:   Patch width in symbol axis (default 8).
        grid_h:    Resource grid height / number of subcarriers (default 128).
        grid_w:    Resource grid width / number of symbols (default 256).
        dropout:   Dropout on positional embeddings (default 0.1).
    """

    def __init__(
        self,
        d_model: int = 128,
        patch_h: int = 8,
        patch_w: int = 8,
        grid_h: int = 128,
        grid_w: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_h = patch_h
        self.patch_w = patch_w

        num_patches_h = grid_h // patch_h  # 16
        num_patches_w = grid_w // patch_w  # 32
        self.num_patches = num_patches_h * num_patches_w  # 512

        # Single Conv2d replaces flattening + linear projection
        self.proj = nn.Conv2d(
            2, d_model,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
            bias=False,
        )
        self.norm = nn.LayerNorm(d_model)

        # Learnable positional embeddings — shape [1, num_patches, d_model]
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Resource grid tensor (batch, 2, grid_h, grid_w).

        Returns:
            Token sequence (batch, num_patches, d_model).
        """
        # (B, 2, H, W) -> (B, d_model, H/ph, W/pw)
        x = self.proj(x)
        B, C, Hp, Wp = x.shape
        # Flatten spatial dims -> (B, num_patches, d_model)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x + self.pos_embed[:, : Hp * Wp]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Encoder (pre-norm)
# ---------------------------------------------------------------------------

class _TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer.

    Args:
        d_model:   Token dimension.
        num_heads: Number of attention heads.
        d_ff:      Feed-forward hidden dimension (default 4 * d_model).
        dropout:   Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.attn_drop(x)
        x = residual + x

        # Pre-norm FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


# ---------------------------------------------------------------------------
# LWMExpert
# ---------------------------------------------------------------------------

class LWMExpert(nn.Module):
    """Large Wireless Model 1.1 adapted as MoE expert.

    Converts raw IQ [B, 2, N] into a 2D resource grid representation,
    applies 2D patch embedding, and processes with transformer encoder.

    The 2D grid is created by reshaping IQ into (num_subcarriers, num_symbols):
    [2, N] -> [2, H, W] where H*W == N (e.g., H=128 subcarriers, W=256 symbols)

    Parameters: ~2.5M

    Input:  (batch, 2, N) IQ tensor, channel 0 = I, channel 1 = Q.
    Output: (batch, num_classes) via forward(), (batch, embed_dim=512) via get_embedding().

    Args:
        embed_dim:   Output embedding dimension for MoE gate (default 512).
        num_classes: Number of output modulation classes (default 10).
        d_model:     Internal transformer token dimension (default 128).
        num_layers:  Number of transformer encoder layers (default 6).
        num_heads:   Number of attention heads (default 4).
        patch_h:     Patch height in subcarrier axis (default 8).
        patch_w:     Patch width in symbol axis (default 8).
        grid_h:      Resource grid subcarrier count (default 128).
        grid_w:      Resource grid symbol count (default 256).
        dropout:     Dropout rate throughout (default 0.1).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 10,
        d_model: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        patch_h: int = 8,
        patch_w: int = 8,
        grid_h: int = 128,
        grid_w: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_h = grid_h
        self.grid_w = grid_w

        # 1. 2D Patch Embedding (includes learnable positional embeddings)
        self.patch_embed = _PatchEmbed2D(
            d_model=d_model,
            patch_h=patch_h,
            patch_w=patch_w,
            grid_h=grid_h,
            grid_w=grid_w,
            dropout=dropout,
        )

        # 2. Transformer Encoder (6 layers, pre-norm)
        d_ff = 4 * d_model  # 512
        self.encoder = nn.ModuleList([
            _TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # 3. Projection Head: mean pool -> Linear(d_model, embed_dim) -> LN -> GELU
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # 4. Classification Head
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Core feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """IQ -> resource grid -> patch tokens -> transformer -> embed_dim vector.

        Args:
            x: (batch, 2, N) raw IQ tensor.

        Returns:
            (batch, embed_dim) embedding vector.
        """
        B, _, N = x.shape

        # Reshape IQ time series into 2D resource grid
        # [B, 2, N] -> [B, 2, grid_h, grid_w]
        x = x.reshape(B, 2, self.grid_h, self.grid_w)

        # 2D patch embedding -> (B, num_patches, d_model)
        tokens = self.patch_embed(x)

        # Transformer encoder
        for layer in self.encoder:
            tokens = layer(tokens)
        tokens = self.encoder_norm(tokens)

        # Mean pool over token sequence -> (B, d_model)
        pooled = tokens.mean(dim=1)

        # Project to embed_dim
        return self.proj_head(pooled)  # (B, embed_dim)

    # ------------------------------------------------------------------
    # Public interface (matches IQExpert exactly)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return classification logits.

        Args:
            x: Raw IQ tensor of shape (batch, 2, N).
                N must equal grid_h * grid_w (default 32768 = 128 * 256).

        Returns:
            Logits of shape (batch, num_classes).
        """
        emb = self._extract_features(x)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embed_dim-dimensional embedding for the MoE gate.

        Args:
            x: Raw IQ tensor of shape (batch, 2, N).

        Returns:
            Embedding of shape (batch, embed_dim).
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
