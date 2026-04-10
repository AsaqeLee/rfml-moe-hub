"""VMD-GAF Expert: custom CNN backbone for 2D temporal correlation images.

Processes Gramian Angular Field (GAF) images derived from VMD-decomposed IQ
signals with 3 channels (GASF_denoised, GADF_denoised, GASF_raw) through a
4-block convolutional backbone to produce 512-dim embeddings.

Architecture follows Fu et al. 2026 (adapted for RFML-MoE):
    Input:   (batch, 3, 256, 256)
    Output:  (batch, num_classes) via forward(), (batch, 512) via get_embedding()

Parameter count: ~2.5M
"""

import torch
import torch.nn as nn


def _make_conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d(bias=False) → BN → ReLU → MaxPool(2×2)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class VMDGAFExpert(nn.Module):
    """Custom CNN adapted for 3-channel VMD-GAF image classification.

    Input:  (batch, 3, 256, 256) — GASF_denoised, GADF_denoised, GASF_raw
    Output: (batch, num_classes) via forward(), (batch, 512) via get_embedding()

    Parameter count: ~2.5M
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 512,
        dropout: float = 0.5,
        image_size: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Four conv blocks: spatial dims halved at each step
        # [B, 3, 256, 256] → [B, 32, 128, 128] → [B, 64, 64, 64]
        #                   → [B, 128, 32, 32]  → [B, 256, 16, 16]
        self.backbone = nn.Sequential(
            _make_conv_block(3, 32),
            _make_conv_block(32, 64),
            _make_conv_block(64, 128),
            _make_conv_block(128, 256),
        )

        # Adaptive pool → [B, 256, 4, 4] → flatten → 4096
        self.avgpool = nn.AdaptiveAvgPool2d(4)

        # FC embedding: 4096 → 512 with GELU → LayerNorm → Dropout
        self.embedding = nn.Sequential(
            nn.Linear(256 * 4 * 4, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_head_weights()

    def _init_head_weights(self):
        """Xavier-uniform init on all Linear layers in head."""
        for m in [self.embedding, self.classifier]:
            # classifier is a bare Linear, not Sequential — iterate uniformly
            modules = m.modules() if isinstance(m, nn.Module) else [m]
            for layer in modules:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone + embedding, return embed_dim-dimensional vector."""
        x = self.backbone(x)          # (batch, 256, 16, 16)
        x = self.avgpool(x)           # (batch, 256, 4, 4)
        x = torch.flatten(x, 1)      # (batch, 4096)
        x = self.embedding(x)         # (batch, embed_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

        Args:
            x: GAF tensor of shape (batch, 3, 256, 256).

        Returns:
            Logits of shape (batch, num_classes).
        """
        emb = self._extract_features(x)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional embedding for the MoE gate.

        Args:
            x: GAF tensor of shape (batch, 3, 256, 256).

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
