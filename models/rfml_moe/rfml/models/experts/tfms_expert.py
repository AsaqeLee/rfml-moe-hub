"""Time-Frequency Multiscale CNN Expert (TFMS).

Dual parallel branches over time-domain magnitude envelope and one-sided FFT
magnitude spectrum, merged and projected to a fixed-dimension embedding.

Based on Mandal & Satija 2023, adapted for RFML-MoE.

Input:  raw IQ [B, 2, N]  (I on channel 0, Q on channel 1)
Output: [B, num_classes] via forward(), [B, embed_dim] via get_embedding()

Parameter count: ~4-5M
"""

import torch
import torch.nn as nn


def _make_conv_branch(in_channels: int, L: int, kernel_size: int, dropout: float) -> nn.Sequential:
    """Build a single conv branch with L blocks.

    Each block: Conv1d → ReLU → MaxPool1d(2) → Dropout
    Channels follow 2^(i+4): 16, 32, 64, 128, 256
    """
    layers: list[nn.Module] = []
    ch_in = in_channels
    for i in range(L):
        ch_out = 2 ** (i + 4)  # 16, 32, 64, 128, 256
        layers.append(nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=2, padding=kernel_size // 2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool1d(2))
        layers.append(nn.Dropout(p=dropout))
        ch_in = ch_out
    layers.append(nn.AdaptiveAvgPool1d(1))
    return nn.Sequential(*layers)


class TFMSExpert(nn.Module):
    """Time-Frequency Multiscale CNN Expert.

    Input:  raw IQ [B, 2, N]
    Output: [B, num_classes] via forward(), [B, embed_dim] via get_embedding()
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 512,
        L: int = 5,
        kernel_size: int = 5,
        D: int = 3,
        n_dense: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Dual parallel conv branches
        self.time_branch = _make_conv_branch(1, L, kernel_size, dropout)
        self.freq_branch = _make_conv_branch(1, L, kernel_size, dropout)

        # Channels after L=5 blocks: 2^(4+4) = 256 each, concatenated → 512
        branch_out = 2 ** (L - 1 + 4)  # 256
        merge_dim = branch_out * 2      # 512

        # Dense layers
        dense_blocks: list[nn.Module] = []
        d_in = merge_dim
        for _ in range(D):
            dense_blocks.append(nn.Linear(d_in, n_dense))
            dense_blocks.append(nn.ReLU(inplace=True))
            dense_blocks.append(nn.Dropout(p=dropout))
            d_in = n_dense
        self.dense = nn.Sequential(*dense_blocks)

        # Projection to embed_dim
        self.embedding = nn.Sequential(
            nn.Linear(n_dense, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for Conv1d, Xavier for Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _extract_features(self, iq: torch.Tensor) -> torch.Tensor:
        """Construct time/freq inputs from raw IQ and run both branches.

        Args:
            iq: Raw IQ tensor [B, 2, N].

        Returns:
            Embedding [B, embed_dim].
        """
        # Time branch: magnitude envelope |z(t)|
        z_mag = (iq[:, 0:1, :] ** 2 + iq[:, 1:2, :] ** 2).sqrt()  # [B, 1, N]

        # Freq branch: one-sided FFT magnitude spectrum
        # fft() accepts complex input; slice first N//2+1 bins for one-sided spectrum
        z_complex = torch.complex(iq[:, 0, :], iq[:, 1, :])           # [B, N]
        N = iq.shape[-1]
        z_fft = torch.fft.fft(z_complex).abs()[:, : N // 2 + 1].unsqueeze(1)  # [B, 1, N//2+1]

        t_out = self.time_branch(z_mag).squeeze(-1)   # [B, 256]
        f_out = self.freq_branch(z_fft).squeeze(-1)   # [B, 256]

        merged = torch.cat([t_out, f_out], dim=1)     # [B, 512]
        x = self.dense(merged)                        # [B, n_dense]
        x = self.embedding(x)                         # [B, embed_dim]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

        Args:
            x: Raw IQ tensor [B, 2, N].

        Returns:
            Logits [B, num_classes].
        """
        emb = self._extract_features(x)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embed_dim-dimensional embedding for the MoE gate.

        Args:
            x: Raw IQ tensor [B, 2, N].

        Returns:
            Embedding [B, embed_dim].
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
