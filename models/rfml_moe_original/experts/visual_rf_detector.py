"""Visual-RF Detection Expert: DETR-style signal detector on spectrogram images.

Applies transformer-based object detection (inspired by RF-DETR and YOLO
architectures) to spectrogram images for signal region detection and
classification. Instead of detecting drones in camera images, this expert
detects drone signal regions within RF spectrograms.

The architecture uses a CNN backbone for local feature extraction followed
by a lightweight transformer decoder with learned object queries for
end-to-end signal detection without NMS post-processing.

Based on concepts from:
- RF-DETR (Roboflow 2025): Transformer detection with CNN backbone
- ATA-YOLOv8 (2025): Efficient small-target detection
- YOLO26 (2026): NMS-free end-to-end detection

Adapted for RF spectrogram signal detection within the RFML-MoE framework.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rfml.experts")


class ConvBlock(nn.Module):
    """Standard convolution block: Conv2d + BatchNorm + GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """Cross Stage Partial block for efficient feature extraction.

    Splits channels, applies bottleneck convolutions to one branch,
    then concatenates — reduces computation while preserving gradient flow.

    Args:
        channels: Number of input/output channels.
        num_bottlenecks: Number of bottleneck layers in the dense branch.
    """

    def __init__(self, channels: int, num_bottlenecks: int = 2):
        super().__init__()
        mid = channels // 2
        self.split_conv = ConvBlock(channels, mid, 1, 1, 0)
        self.main_conv = ConvBlock(channels, mid, 1, 1, 0)

        bottlenecks = []
        for _ in range(num_bottlenecks):
            bottlenecks.append(nn.Sequential(
                ConvBlock(mid, mid, 3, 1, 1),
                ConvBlock(mid, mid, 3, 1, 1),
            ))
        self.bottlenecks = nn.ModuleList(bottlenecks)

        self.merge_conv = ConvBlock(mid * 2, channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = self.split_conv(x)
        main = self.main_conv(x)
        for bottleneck in self.bottlenecks:
            main = main + bottleneck(main)
        return self.merge_conv(torch.cat([split, main], dim=1))


class DetectorBackbone(nn.Module):
    """CNN backbone for spectrogram feature extraction.

    Multi-scale feature pyramid producing features at 3 resolution levels.
    Inspired by the CSPDarknet backbone used in YOLO architectures.

    Args:
        in_channels: Input channels (3 for mag/phase/IF spectrogram).
        channels: Base channel count (doubled at each stage).
    """

    def __init__(self, in_channels: int = 3, channels: int = 32):
        super().__init__()
        c = channels
        # Stem: rapid initial downsampling
        self.stem = nn.Sequential(
            ConvBlock(in_channels, c, 3, 2, 1),     # /2
            ConvBlock(c, c * 2, 3, 2, 1),            # /4
        )
        # Stage 1: /8
        self.stage1 = nn.Sequential(
            ConvBlock(c * 2, c * 4, 3, 2, 1),
            CSPBlock(c * 4, num_bottlenecks=2),
        )
        # Stage 2: /16
        self.stage2 = nn.Sequential(
            ConvBlock(c * 4, c * 8, 3, 2, 1),
            CSPBlock(c * 8, num_bottlenecks=2),
        )
        # Stage 3: /32
        self.stage3 = nn.Sequential(
            ConvBlock(c * 8, c * 16, 3, 2, 1),
            CSPBlock(c * 16, num_bottlenecks=1),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Returns:
            List of feature maps at 3 scales: [/8, /16, /32]
        """
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3]


class TransformerDecoderLayer(nn.Module):
    """Lightweight transformer decoder layer with cross-attention.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward dimension.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8,
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads,
                                                dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads,
                                                 dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor,
                memory: torch.Tensor) -> torch.Tensor:
        """Decode with self-attention on queries, cross-attention to memory.

        Args:
            queries: Object queries (B, num_queries, d_model).
            memory: Encoder features (B, HW, d_model).

        Returns:
            Updated queries (B, num_queries, d_model).
        """
        # Self-attention
        q = self.norm1(queries)
        q = queries + self.dropout(self.self_attn(q, q, q)[0])

        # Cross-attention to backbone features
        q2 = self.norm2(q)
        q = q + self.dropout(self.cross_attn(q2, memory, memory)[0])

        # Feed-forward
        q = q + self.dropout(self.ffn(self.norm3(q)))
        return q


class VisualRFDetector(nn.Module):
    """DETR-style signal detector on RF spectrogram images.

    Detects and classifies drone signal regions within spectrograms
    using a CNN backbone and transformer decoder with learned object queries.
    End-to-end detection without NMS post-processing (YOLO26 principle).

    When used as an MoE expert, the detector's object query embeddings
    are aggregated via attention pooling to produce a single 512-dim
    embedding for the router.

    Input:  (batch, 3, H, W) — spectrogram (magnitude, phase, inst. freq)
    Output: (batch, num_classes) via forward(), (batch, 512) via get_embedding()

    Args:
        in_channels: Spectrogram channels (default 3).
        num_classes: Number of drone classes.
        embed_dim: MoE embedding dimension.
        d_model: Transformer dimension.
        num_queries: Number of learned object queries.
        num_decoder_layers: Transformer decoder layers.
        num_heads: Attention heads.
        backbone_channels: Base channels for CNN backbone.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 512,
        d_model: int = 256,
        num_queries: int = 10,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        backbone_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.num_queries = num_queries

        # CNN backbone
        self.backbone = DetectorBackbone(in_channels, backbone_channels)
        backbone_out_ch = backbone_channels * 16  # stage3 output channels

        # Project backbone features to d_model
        self.input_proj = nn.Sequential(
            nn.Conv2d(backbone_out_ch, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
        )

        # Learned object queries (DETR-style)
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)

        # Learnable positional encoding for backbone features
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)

        # Transformer decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Attention pooling over object queries → single embedding
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),  # attention scores per query
        )

        # MoE embedding projection
        self.embedding = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Classification head (per-query class prediction, then aggregate)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone + transformer decoder + attention pooling.

        Args:
            x: Spectrogram tensor (B, 3, H, W).

        Returns:
            Embedding of shape (B, embed_dim).
        """
        B = x.shape[0]

        # Extract multi-scale backbone features (use deepest scale)
        features = self.backbone(x)
        feat = features[-1]  # (B, C, h, w) at /32 scale

        # Project to d_model
        feat = self.input_proj(feat)  # (B, d_model, h, w)
        h, w = feat.shape[2], feat.shape[3]
        feat_flat = feat.flatten(2).transpose(1, 2)  # (B, h*w, d_model)

        # Add positional encoding
        pos = self.pos_encoding[:, :feat_flat.shape[1], :]
        memory = feat_flat + pos

        # Expand queries for batch
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d_model)

        # Transformer decoder
        for layer in self.decoder_layers:
            queries = layer(queries, memory)

        # Attention pooling: weighted sum of query embeddings
        attn_weights = self.attn_pool(queries)  # (B, Q, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (queries * attn_weights).sum(dim=1)  # (B, d_model)

        # Project to MoE embed_dim
        return self.embedding(pooled)  # (B, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

        Args:
            x: Spectrogram tensor of shape (batch, 3, H, W).

        Returns:
            Logits of shape (batch, num_classes).
        """
        emb = self._extract_features(x)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional embedding for the MoE gate.

        Args:
            x: Spectrogram tensor of shape (batch, 3, H, W).

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
