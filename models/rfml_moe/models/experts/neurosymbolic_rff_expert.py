"""Neuro-Symbolic RF Fingerprinting Expert: 2D Shapelet-based IQ classification.

Extracts variable-length 2D shapelets from IQ signals that map to specific
hardware impairments (power amplifier non-linearities, mixer imbalances),
then embeds them for classification.

Based on: arXiv:2602.03035 — shapelet-based time-series classification adapted
to 2D (I/Q) RF fingerprinting.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeletExtractor(nn.Module):
    """Extract K learnable 2D shapelets from IQ signal.

    Shapelets are small, discriminative subsequences that characterize
    specific RF hardware fingerprints. Unlike fixed feature extraction,
    shapelets are learned end-to-end via gradient descent.

    Each shapelet is a learnable parameter of shape [2, L_k] where L_k
    is the shapelet length. The shapelet distance is computed as the
    minimum sliding-window distance between the shapelet and the input.

    Efficient computation uses F.conv1d to evaluate all window positions
    simultaneously:
        dist(t) = ||x[:,t:t+L] - s||² = ||x[:,t:t+L]||² - 2<x[:,t:t+L], s> + ||s||²

    The cross-term -2<x, s> is computed via conv1d; the squared-norm of
    x windows is computed via a cumulative sum over a precomputed squared
    signal. The minimum over all time positions yields the shapelet distance.
    """

    def __init__(
        self,
        num_shapelets: int = 32,
        shapelet_lengths: List[int] = None,
        in_channels: int = 2,
    ):
        super().__init__()
        if shapelet_lengths is None:
            shapelet_lengths = [16, 32, 64, 128]

        self.in_channels = in_channels
        self.shapelet_lengths = shapelet_lengths
        self.num_shapelets = num_shapelets

        # Distribute shapelets evenly across lengths
        n_per_length = num_shapelets // len(shapelet_lengths)
        remainder = num_shapelets - n_per_length * len(shapelet_lengths)

        # Build one ParameterList per length; lengths with index < remainder
        # get one extra shapelet so the total sums exactly to num_shapelets
        self.shapelet_groups: nn.ParameterList = nn.ParameterList()
        self._group_sizes: List[int] = []
        for i, L in enumerate(shapelet_lengths):
            count = n_per_length + (1 if i < remainder else 0)
            self._group_sizes.append(count)
            for _ in range(count):
                # [in_channels, L]
                param = nn.Parameter(torch.randn(in_channels, L) * 0.01)
                self.shapelet_groups.append(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute minimum sliding-window distances for all shapelets.

        Args:
            x: IQ signal of shape [B, 2, N].

        Returns:
            Distance vector of shape [B, num_shapelets].
        """
        B, C, N = x.shape
        # Precompute per-sample squared signal: [B, N]
        x_sq = (x ** 2).sum(dim=1)  # [B, N]

        distances = []
        for shapelet in self.shapelet_groups:
            L = shapelet.shape[-1]  # shapelet length

            # Cross-term: F.conv1d treats each input channel independently.
            # shapelet: [C, L] -> conv filter [1, C, L]
            # x: [B, C, N] -> conv1d with groups=1 sums over C
            # conv_out: [B, 1, N-L+1]
            conv_out = F.conv1d(x, shapelet.unsqueeze(0))  # [B, 1, N-L+1]
            conv_out = conv_out.squeeze(1)  # [B, N-L+1]

            # Sliding squared norm of x windows via cumulative sum.
            # x_sq_cs[b, t] = sum_{i=0}^{t-1} x_sq[b, i]  (exclusive prefix)
            x_sq_cs = torch.zeros(B, N + 1, device=x.device, dtype=x.dtype)
            x_sq_cs[:, 1:] = x_sq.cumsum(dim=1)
            # window sum for start position t: cs[t+L] - cs[t]
            # valid positions: t in [0, N-L]
            x_sq_windows = x_sq_cs[:, L:] - x_sq_cs[:, :N - L + 1]  # [B, N-L+1]

            # Shapelet squared norm (scalar)
            s_sq = (shapelet ** 2).sum()

            # Pointwise distance at each window position
            dist_all = x_sq_windows - 2.0 * conv_out + s_sq  # [B, N-L+1]

            # Clamp to avoid negative values from floating-point errors
            dist_all = dist_all.clamp(min=0.0)

            # Minimum distance over all positions
            dist_min = dist_all.min(dim=1).values  # [B]
            distances.append(dist_min)

        return torch.stack(distances, dim=1)  # [B, num_shapelets]


class ShapeletEmbedding(nn.Module):
    """Map shapelet distance vector to high-dimensional embedding.

    Transforms the K-dimensional shapelet distance vector through a
    multi-layer network to produce embeddings suitable for the MoE
    router and classification heads.
    """

    def __init__(
        self,
        num_shapelets: int = 32,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_shapelets, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Embed shapelet distances.

        Args:
            distances: [B, num_shapelets]

        Returns:
            [B, embed_dim]
        """
        return self.net(distances)


class NeuroSymbolicRFFExpert(nn.Module):
    """Neuro-Symbolic RF Fingerprinting expert.

    Extracts learnable 2D shapelets from IQ signals, computes
    sliding-window distances, and maps to embeddings for
    hardware fingerprint-based drone identification.

    Each shapelet represents a discriminative temporal pattern associated
    with a specific hardware impairment (e.g. PA non-linearity, IQ
    imbalance). The minimum sliding-window distance quantifies how well
    the impairment pattern appears anywhere in the captured frame.

    Input:  [B, 2, N] raw IQ
    Output: [B, num_classes] via forward()
            [B, embed_dim]   via get_embedding()
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 512,
        num_shapelets: int = 32,
        shapelet_lengths: List[int] = None,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        if shapelet_lengths is None:
            shapelet_lengths = [16, 32, 64, 128]

        self.embed_dim = embed_dim

        self.shapelet_extractor = ShapeletExtractor(
            num_shapelets=num_shapelets,
            shapelet_lengths=shapelet_lengths,
            in_channels=2,
        )
        self.shapelet_embedding = ShapeletEmbedding(
            num_shapelets=num_shapelets,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run shapelet extraction + embedding, return embed_dim vector.

        Args:
            x: [B, 2, N]

        Returns:
            [B, embed_dim]
        """
        distances = self.shapelet_extractor(x)   # [B, num_shapelets]
        embedding = self.shapelet_embedding(distances)  # [B, embed_dim]
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

        Args:
            x: Raw IQ tensor of shape [B, 2, N].

        Returns:
            Logits of shape [B, num_classes].
        """
        emb = self._extract_features(x)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional embedding for the MoE gate.

        Args:
            x: Raw IQ tensor of shape [B, 2, N].

        Returns:
            Embedding of shape [B, embed_dim].
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
