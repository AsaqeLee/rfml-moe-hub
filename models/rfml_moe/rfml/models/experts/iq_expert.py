"""IQ-domain Expert: Complex-valued CNN + Transformer hybrid (SignalFormer).

Processes raw IQ samples through complex-valued 1D convolutions followed by
a transformer encoder to produce 512-dimensional embeddings for the MoE gate.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution using real arithmetic.

    Implements (a+bi)*(c+di) = (ac-bd) + (ad+bc)i with four real convolutions
    sharing structure but with independent weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv_rr = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias,
        )
        self.conv_ri = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias,
        )
        self.conv_ir = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias,
        )
        self.conv_ii = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias,
        )

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor):
        # (a+bi)(c+di) = (ac - bd) + (ad + bc)i
        out_real = self.conv_rr(x_real) - self.conv_ii(x_imag)
        out_imag = self.conv_ri(x_real) + self.conv_ir(x_imag)
        return out_real, out_imag


class ComplexConvBlock(nn.Module):
    """Complex conv + GroupNorm + ReLU (applied to magnitude)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        num_groups: int = 8,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = ComplexConv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
        )
        self.norm_real = nn.GroupNorm(num_groups, out_channels)
        self.norm_imag = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor):
        x_real, x_imag = self.conv(x_real, x_imag)
        x_real = self.norm_real(x_real)
        x_imag = self.norm_imag(x_imag)
        # CReLU: apply ReLU independently to real and imaginary parts
        x_real = F.relu(x_real)
        x_imag = F.relu(x_imag)
        return x_real, x_imag


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class IQExpert(nn.Module):
    """SignalFormer: Complex-valued CNN + Transformer hybrid for raw IQ data.

    Input:  (batch, 2, N) where channel 0 = I, channel 1 = Q, N=32768
    Output: (batch, num_classes) via forward(), (batch, 512) via get_embedding()

    Architecture:
        7 complex conv blocks (channels 1->64->64->128->128->256->256->256,
        stride-2 downsampling) followed by a 4-layer transformer encoder.
    """

    def __init__(self, num_classes: int = 10, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # --- Complex CNN backbone ---
        # Input: 1 complex channel (I + jQ) => 7 conv blocks with stride-2
        # After 7 stride-2 layers: 32768 / 2^7 = 256 time steps
        self.conv_blocks = nn.ModuleList([
            ComplexConvBlock(1, 64, kernel_size=7, stride=2, num_groups=8),
            ComplexConvBlock(64, 64, kernel_size=7, stride=2, num_groups=8),
            ComplexConvBlock(64, 128, kernel_size=7, stride=2, num_groups=8),
            ComplexConvBlock(128, 128, kernel_size=7, stride=2, num_groups=16),
            ComplexConvBlock(128, 256, kernel_size=7, stride=2, num_groups=16),
            ComplexConvBlock(256, 256, kernel_size=5, stride=2, num_groups=16),
            ComplexConvBlock(256, 256, kernel_size=5, stride=2, num_groups=16),
        ])

        # Project concatenated real+imag (512 channels) to embed_dim
        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # --- Transformer encoder ---
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=512, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(embed_dim)

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

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
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run CNN + transformer, return 512-dim embedding."""
        # Split I/Q into real and imaginary channels
        x_real = x[:, 0:1, :]  # (batch, 1, N)
        x_imag = x[:, 1:2, :]  # (batch, 1, N)

        for block in self.conv_blocks:
            x_real, x_imag = block(x_real, x_imag)

        # Concatenate real and imag: (batch, 512, seq_len)
        x = torch.cat([x_real, x_imag], dim=1)

        # Reshape to (batch, seq_len, 512) for transformer
        x = x.permute(0, 2, 1)
        x = self.projection(x)

        # Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, embed_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

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
        """Freeze all parameters for progressive training."""
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
