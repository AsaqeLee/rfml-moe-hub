"""Spectrogram-domain Expert: EfficientNet-B2 backbone.

Processes spectrogram images with 3 channels (magnitude, phase, instantaneous
frequency) through a pretrained EfficientNet-B2 to produce 512-dim embeddings.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


class SpectrogramExpert(nn.Module):
    """EfficientNet-B2 adapted for 3-channel spectrogram classification.

    Input:  (batch, 3, 512, 512) — magnitude, phase, inst. frequency channels
    Output: (batch, num_classes) via forward(), (batch, 512) via get_embedding()

    Parameter count: ~9M
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Load EfficientNet-B2 backbone
        if pretrained:
            backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        else:
            backbone = efficientnet_b2(weights=None)

        # Replace first conv to accept 3 custom channels instead of RGB.
        # EfficientNet-B2 first conv: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Since input is also 3 channels we could keep it, but we reinitialize
        # to avoid pretrained RGB-specific filters biasing non-RGB input.
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            3,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        backbone.features[0][0] = new_conv

        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # EfficientNet-B2 final feature channels = 1408
        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1408, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        """Initialize the custom head layers (backbone uses pretrained or default init)."""
        for m in [self.embedding, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone + embedding, return 512-dim vector."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (batch, 1408)
        x = self.embedding(x)    # (batch, embed_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

        Args:
            x: Spectrogram tensor of shape (batch, 3, 512, 512).

        Returns:
            Logits of shape (batch, num_classes).
        """
        emb = self._extract_features(x)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional embedding for the MoE gate.

        Args:
            x: Spectrogram tensor of shape (batch, 3, 512, 512).

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
