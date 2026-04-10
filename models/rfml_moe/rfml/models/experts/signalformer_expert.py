"""SignalFormer Expert: Hybrid CNN-Transformer for RF drone identification.

Based on the SignalFormer architecture (2023, Sensors) which processes
3-channel Time-Frequency representations: real(STFT), imag(STFT), magnitude(STFT).

Key components:
- C-Tokenizer with D-TFCB (Dilation Time-Frequency Convolution Block)
- Gated Self-Attention (GSA) with decoupled Time/Frequency encoders
- Feature Extraction Module (FEM) with global average pooling
"""

import torch
import torch.nn as nn


class DTFCB(nn.Module):
    """Dilation Time-Frequency Convolution Block.

    Pointwise(in, mid) -> Depthwise(mid, mid, 3x3, dilation) -> Pointwise(mid, out)
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation  # same padding for 3x3 with dilation
        self.block = nn.Sequential(
            # Pointwise 1: cross-channel aggregation
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            # Depthwise: spatial context with dilation
            nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3,
                padding=padding, dilation=dilation, groups=mid_channels, bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            # Pointwise 2: cross-channel aggregation
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TFDB(nn.Module):
    """Time-Frequency Downsampling Block.

    Pyramidal downsampling with 5x5 kernel, stride 2.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GatedSelfAttention(nn.Module):
    """Gated Self-Attention block.

    gate = sigmoid(W_g * x)
    attn_out = MultiHeadAttention(x, x, x)
    output = gate * attn_out + (1 - gate) * x
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        residual = x
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        gate = torch.sigmoid(self.gate_proj(x_norm))
        x = gate * attn_out + (1 - gate) * residual
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TFEncoder(nn.Module):
    """Time or Frequency Encoder.

    Reshapes the 2D feature map to process attention along one axis,
    applies GSA, then reshapes back.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, axis: str = "time"):
        super().__init__()
        assert axis in ("time", "freq")
        self.axis = axis
        self.gsa = GatedSelfAttention(d_model, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) where C == d_model
        B, C, H, W = x.shape

        if self.axis == "time":
            # Self-attention along time (W) for each frequency bin
            # Reshape: (B, C, H, W) -> (B*H, W, C)
            x = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
            x = self.gsa(x)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            # Self-attention along frequency (H) for each time step
            # Reshape: (B, C, H, W) -> (B*W, H, C)
            x = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
            x = self.gsa(x)
            x = x.reshape(B, W, H, C).permute(0, 3, 2, 1)

        return x


class SignalFormerRFExpert(nn.Module):
    """SignalFormer hybrid CNN-Transformer for RF drone identification.

    Processes 3-channel TF representations through a convolutional tokenizer
    with dilated convolutions, followed by alternating time/frequency
    self-attention encoders, and a feature extraction module.

    Input:  (batch, 3, H, W) -- real(STFT), imag(STFT), magnitude(STFT)
    Output: (batch, num_classes) via forward(), (batch, 512) via get_embedding()
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 512,
        num_classes: int = 10,
        num_blocks: int = 4,
        num_heads: int = 8,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # --- C-Tokenizer: D-TFCB stages with multi-scale dilation + TFDB downsampling ---
        tokenizer_channels = [64, 128, 256]
        dilations = [1, 2, 4]

        tokenizer_layers = []
        ch_in = in_channels
        for ch_out, dil in zip(tokenizer_channels, dilations):
            mid = ch_out
            tokenizer_layers.append(DTFCB(ch_in, mid, ch_out, dilation=dil))
            tokenizer_layers.append(TFDB(ch_out, ch_out))
            ch_in = ch_out

        self.tokenizer = nn.Sequential(*tokenizer_layers)

        # Project tokenizer output to d_model if needed
        self.tok_proj = (
            nn.Sequential(
                nn.Conv2d(tokenizer_channels[-1], d_model, kernel_size=1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
            )
            if tokenizer_channels[-1] != d_model
            else nn.Identity()
        )

        # --- Alternating T/F Encoder blocks ---
        self.tf_encoders = nn.ModuleList()
        for i in range(num_blocks):
            axis = "time" if i % 2 == 0 else "freq"
            self.tf_encoders.append(TFEncoder(d_model, num_heads, dropout, axis=axis))

        # --- Feature Extraction Module (FEM) ---
        self.fem = nn.Sequential(
            nn.Conv2d(d_model, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for convolutional and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run tokenizer -> T/F encoders -> FEM -> embed_dim vector."""
        # C-Tokenizer with D-TFCB
        x = self.tokenizer(x)
        x = self.tok_proj(x)

        # Alternating Time/Frequency encoder blocks
        for encoder in self.tf_encoders:
            x = encoder(x)

        # Feature Extraction Module
        x = self.fem(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (batch, embed_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

        Args:
            x: TF representation tensor of shape (batch, 3, H, W).

        Returns:
            Logits of shape (batch, num_classes).
        """
        emb = self._extract_features(x)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embed_dim-dimensional embedding for the MoE gate.

        Args:
            x: TF representation tensor of shape (batch, 3, H, W).

        Returns:
            Embedding of shape (batch, embed_dim).
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
