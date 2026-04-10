"""Cyclostationary Expert: Temporal Convolutional Network (TCN).

Processes spectral correlation function (SCF) features through a stack of
dilated causal convolutions to produce 512-dimensional embeddings.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """Single TCN block: two dilated causal convolutions with residual connection.

    Structure:
        dilated conv1d -> weight norm -> ReLU -> dropout ->
        dilated conv1d -> weight norm -> ReLU -> dropout ->
        + residual (with optional 1x1 conv for channel matching)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Causal padding: ensures output length == input length
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Amount to chop off the right side for causal alignment
        self.chomp = padding

        # Residual projection if channels differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, channels, seq_len)

        Returns:
            (batch, out_channels, seq_len)
        """
        out = self.conv1(x)
        # Causal chomp: remove future-looking padding from the right
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = self.downsample(x)
        return self.relu_out(out + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with increasing dilation.

    Receptive field = nb_stacks * kernel_size * sum(dilations).
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 64,
        kernel_size: int = 8,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for i, d in enumerate(dilations):
            c_in = in_channels if i == 0 else nb_filters
            layers.append(TemporalBlock(c_in, nb_filters, kernel_size, d, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CycloExpert(nn.Module):
    """Temporal Convolutional Network for cyclostationary SCF features.

    Input:  (batch, channels, seq_len) — e.g., (batch, 1, 512)
    Output: (batch, num_classes) via forward(), (batch, 512) via get_embedding()

    Architecture:
        TCN (6 dilated causal blocks, 64 filters) -> global avg pool ->
        MLP projection to 512-dim embedding

    Parameter count: ~5-10M
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        nb_filters: int = 64,
        kernel_size: int = 8,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # TCN backbone
        self.tcn = TemporalConvNet(
            in_channels=in_channels,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )

        # Expand channels after TCN for richer representation
        self.channel_expand = nn.Sequential(
            weight_norm(nn.Conv1d(nb_filters, 256, kernel_size=1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(256, 512, kernel_size=1)),
            nn.ReLU(),
        )

        # Global average pooling is applied in forward

        # Embedding projection
        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # Classification head
        self.classifier = nn.Sequential(
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
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run TCN + pooling + projection, return 512-dim embedding."""
        # TCN: (batch, in_channels, seq_len) -> (batch, nb_filters, seq_len)
        x = self.tcn(x)

        # Channel expansion: (batch, nb_filters, seq_len) -> (batch, 512, seq_len)
        x = self.channel_expand(x)

        # Global average pooling: (batch, 512)
        x = x.mean(dim=2)

        # Projection to embed_dim
        return self.projection(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

        Args:
            x: SCF features of shape (batch, channels, seq_len).

        Returns:
            Logits of shape (batch, num_classes).
        """
        emb = self._extract_features(x)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional embedding for the MoE gate.

        Args:
            x: SCF features of shape (batch, channels, seq_len).

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
