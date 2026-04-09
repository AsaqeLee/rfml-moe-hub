"""Self-supervised pretraining objectives for individual experts.

Provides masked autoencoder (MAE) and MoCo-v3 style contrastive learning
tailored to RF signal data.
"""

import copy
import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rfml.training")


class MaskedAutoencoder(nn.Module):
    """Masked autoencoder for self-supervised pretraining of RF experts.

    Masks a fraction of input patches and trains a lightweight transformer
    decoder to reconstruct the original signal from unmasked patches.

    For IQ data: 1D patches of size patch_size along the time axis.
    For spectrograms: 2D patches of size (patch_h, patch_w).

    Args:
        input_dim: Dimensionality of each patch when flattened.
        embed_dim: Embedding dimension for encoder/decoder tokens.
        decoder_dim: Hidden dimension of the decoder transformer.
        decoder_layers: Number of transformer layers in the decoder.
        decoder_heads: Number of attention heads in the decoder.
        mask_ratio: Fraction of patches to mask (default 0.75).
        patch_size: Patch size for 1D signals (IQ).
        patch_h: Patch height for 2D signals (spectrogram).
        patch_w: Patch width for 2D signals (spectrogram).
        mode: "iq" for 1D patching, "spectrogram" for 2D patching.
    """

    def __init__(
        self,
        input_dim: int = 512,
        embed_dim: int = 512,
        decoder_dim: int = 256,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        mask_ratio: float = 0.75,
        patch_size: int = 256,
        patch_h: int = 32,
        patch_w: int = 32,
        mode: str = "iq",
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.patch_size = patch_size
        self.patch_h = patch_h
        self.patch_w = patch_w

        # Encoder projection (patches -> embed_dim)
        if mode == "iq":
            # IQ: each patch is (2, patch_size) flattened to 2*patch_size
            self.input_dim = 2 * patch_size
        else:
            # Spectrogram: each patch is (3, patch_h, patch_w)
            self.input_dim = 3 * patch_h * patch_w

        self.encoder_proj = nn.Linear(self.input_dim, embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Decoder
        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_proj = nn.Linear(decoder_dim, self.input_dim)

        # Positional embeddings (set dynamically based on max patches)
        max_patches = 512
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, max_patches, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed_enc, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_dec, std=0.02)

    def _patchify_iq(self, x: torch.Tensor) -> torch.Tensor:
        """Convert IQ signal (B, 2, N) into patches (B, num_patches, patch_dim)."""
        B, C, N = x.shape
        num_patches = N // self.patch_size
        # Reshape to (B, num_patches, 2 * patch_size)
        x = x[:, :, : num_patches * self.patch_size]
        x = x.reshape(B, C, num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, num_patches, C * self.patch_size)
        return x

    def _patchify_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram (B, 3, H, W) into patches (B, num_patches, patch_dim)."""
        B, C, H, W = x.shape
        nH = H // self.patch_h
        nW = W // self.patch_w
        # (B, C, nH, patch_h, nW, patch_w) -> (B, nH*nW, C*patch_h*patch_w)
        x = x[:, :, : nH * self.patch_h, : nW * self.patch_w]
        x = x.reshape(B, C, nH, self.patch_h, nW, self.patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, nH * nW, C * self.patch_h * self.patch_w)
        return x

    def _random_mask(
        self, B: int, num_patches: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random mask indices.

        Returns:
            ids_keep: indices of unmasked patches (B, num_keep)
            ids_restore: indices to unshuffle back to original order (B, num_patches)
            mask: binary mask, 1 = masked (B, num_patches)
        """
        num_keep = int(num_patches * (1.0 - self.mask_ratio))

        noise = torch.rand(B, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]

        mask = torch.ones(B, num_patches, device=device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: patchify, mask, encode, decode, reconstruct.

        Args:
            x: Input tensor. (B, 2, N) for IQ or (B, 3, H, W) for spectrogram.

        Returns:
            loss: MSE reconstruction loss on masked patches.
            mask: Binary mask indicating which patches were masked.
        """
        if self.mode == "iq":
            patches = self._patchify_iq(x)
        else:
            patches = self._patchify_spectrogram(x)

        B, num_patches, D = patches.shape

        # Encode all patches
        tokens = self.encoder_proj(patches) + self.pos_embed_enc[:, :num_patches]

        # Generate mask
        ids_keep, ids_restore, mask = self._random_mask(B, num_patches, x.device)

        # Keep only unmasked tokens for encoder
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        visible_tokens = torch.gather(tokens, dim=1, index=ids_keep_expanded)

        # Map to decoder dim
        visible_tokens = self.encoder_to_decoder(visible_tokens)

        # Append mask tokens
        num_masked = num_patches - ids_keep.shape[1]
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        full_tokens = torch.cat([visible_tokens, mask_tokens], dim=1)

        # Unshuffle to original order
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, full_tokens.shape[-1])
        full_tokens = torch.gather(full_tokens, dim=1, index=ids_restore_expanded)

        # Add decoder positional embeddings
        full_tokens = full_tokens + self.pos_embed_dec[:, :num_patches]

        # Decode
        decoded = self.decoder(full_tokens)
        decoded = self.decoder_norm(decoded)
        pred = self.decoder_proj(decoded)

        # MSE loss on masked patches only
        loss = (pred - patches) ** 2
        loss = loss.mean(dim=-1)  # per-patch loss
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)

        return loss, mask


class ContrastiveLearning(nn.Module):
    """MoCo-v3 style contrastive learning with RF-specific augmentations.

    Maintains a momentum encoder updated via exponential moving average.
    Uses InfoNCE loss with learnable temperature.

    Args:
        encoder: The expert encoder module (must have get_embedding method or be callable).
        embed_dim: Embedding dimension from the encoder.
        proj_dim: Projection head output dimension.
        temperature: Initial temperature for InfoNCE loss.
        momentum: EMA coefficient for momentum encoder updates.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 512,
        proj_dim: int = 256,
        temperature: float = 0.07,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.momentum = momentum

        # Online encoder + projection head
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

        # Momentum encoder + projection head (no gradients)
        self.momentum_encoder = copy.deepcopy(encoder)
        self.momentum_projector = copy.deepcopy(self.projector)
        for p in self.momentum_encoder.parameters():
            p.requires_grad = False
        for p in self.momentum_projector.parameters():
            p.requires_grad = False

        # Learnable temperature
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature))
        )

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Update momentum encoder parameters via EMA."""
        for param_q, param_k in zip(
            self.encoder.parameters(), self.momentum_encoder.parameters()
        ):
            param_k.data.mul_(self.momentum).add_(
                param_q.data, alpha=1.0 - self.momentum
            )
        for param_q, param_k in zip(
            self.projector.parameters(), self.momentum_projector.parameters()
        ):
            param_k.data.mul_(self.momentum).add_(
                param_q.data, alpha=1.0 - self.momentum
            )

    def _get_embedding(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from encoder, supporting both callable and get_embedding."""
        if hasattr(encoder, "get_embedding"):
            return encoder.get_embedding(x)
        return encoder(x)

    @staticmethod
    def augment_iq(x: torch.Tensor) -> torch.Tensor:
        """Apply RF-specific augmentations to IQ data.

        Augmentations (applied randomly):
            - Cyclic time shift
            - Frequency offset (phase rotation)
            - SNR variation (additive noise)

        Args:
            x: IQ tensor of shape (B, 2, N).

        Returns:
            Augmented IQ tensor of same shape.
        """
        B, C, N = x.shape

        # Cyclic time shift: shift by random amount
        shifts = torch.randint(0, N, (B,), device=x.device)
        augmented = torch.stack(
            [torch.roll(x[i], shifts=shifts[i].item(), dims=-1) for i in range(B)]
        )

        # Frequency offset: apply phase rotation e^{j*2*pi*f0*t}
        f0 = (torch.rand(B, 1, device=x.device) - 0.5) * 0.01  # small freq offset
        t = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(0)  # (1, N)
        phase = 2.0 * math.pi * f0 * t  # (B, N)
        cos_phase = torch.cos(phase).unsqueeze(1)  # (B, 1, N)
        sin_phase = torch.sin(phase).unsqueeze(1)

        i_aug = augmented[:, 0:1] * cos_phase - augmented[:, 1:2] * sin_phase
        q_aug = augmented[:, 0:1] * sin_phase + augmented[:, 1:2] * cos_phase
        augmented = torch.cat([i_aug, q_aug], dim=1)

        # SNR variation: add scaled Gaussian noise
        noise_scale = torch.rand(B, 1, 1, device=x.device) * 0.1
        noise = torch.randn_like(augmented) * noise_scale
        augmented = augmented + noise

        return augmented

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for a batch.

        Creates two augmented views, computes InfoNCE loss between
        online and momentum encoder projections.

        Args:
            x: Input tensor (B, C, ...) - raw signal data.

        Returns:
            InfoNCE contrastive loss scalar.
        """
        # Create two augmented views
        x1 = self.augment_iq(x)
        x2 = self.augment_iq(x)

        # Online encoder: encode + project + predict
        z1 = self.predictor(self.projector(self._get_embedding(self.encoder, x1)))
        z2 = self.predictor(self.projector(self._get_embedding(self.encoder, x2)))

        # Momentum encoder: encode + project (no gradient)
        with torch.no_grad():
            self._update_momentum_encoder()
            k1 = self.momentum_projector(
                self._get_embedding(self.momentum_encoder, x1)
            )
            k2 = self.momentum_projector(
                self._get_embedding(self.momentum_encoder, x2)
            )

        # InfoNCE loss (symmetric)
        loss = self._infonce_loss(z1, k2) + self._infonce_loss(z2, k1)
        return loss * 0.5

    def _infonce_loss(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            q: Query features (B, D), from online encoder.
            k: Key features (B, D), from momentum encoder.

        Returns:
            InfoNCE loss scalar.
        """
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        temperature = self.log_temperature.exp().clamp(min=0.01, max=1.0)

        # Positive logits: (B,)
        logits_pos = (q * k).sum(dim=-1, keepdim=True) / temperature

        # Negative logits: all other keys in batch (B, B)
        logits_neg = q @ k.T / temperature

        # Labels: diagonal is positive
        labels = torch.arange(q.shape[0], device=q.device)
        loss = F.cross_entropy(logits_neg, labels)

        return loss


class MaskedFrequencyPredictor(nn.Module):
    """Masked frequency prediction for self-supervised pretraining of the TFMS expert.

    Masks a fraction of frequency bins in the FFT magnitude spectrum and trains
    the TFMS expert to predict the masked content from remaining time and frequency
    features. This is the frequency-domain equivalent of MAE, natural for the TFMS
    architecture which has an explicit frequency branch.

    Based on Mandal & Satija (2023) — adapted for self-supervised pretraining
    within the RFML-MoE 4-phase training pipeline.

    Args:
        expert: The TFMSExpert module (must have get_embedding method).
        embed_dim: Embedding dimension from the expert.
        mask_ratio: Fraction of frequency bins to mask (default 0.2).
        pred_dim: Output dimension of the predictor head.
    """

    def __init__(
        self,
        expert: nn.Module,
        embed_dim: int = 512,
        mask_ratio: float = 0.2,
        pred_dim: int = 256,
    ):
        super().__init__()
        self.expert = expert
        self.mask_ratio = mask_ratio

        # Predictor head: embedding -> masked frequency content
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, pred_dim),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        """Compute masked frequency prediction loss.

        Args:
            iq: IQ tensor of shape (B, 2, N).

        Returns:
            MSE loss on masked frequency bins (scalar).
        """
        B, C, N = iq.shape

        # Compute full FFT magnitude (target)
        z_complex = torch.complex(iq[:, 0, :], iq[:, 1, :])
        freq_mag = torch.fft.rfft(z_complex).abs()  # (B, N//2+1)
        N_freq = freq_mag.shape[1]

        # Generate random frequency mask
        n_mask = max(1, int(N_freq * self.mask_ratio))
        mask_idx = torch.randperm(N_freq, device=iq.device)[:n_mask]

        # Save target values at masked positions
        target = freq_mag[:, mask_idx].clone()  # (B, n_mask)

        # Create masked IQ: zero out masked frequency bins in the signal
        z_fft = torch.fft.rfft(z_complex)  # (B, N//2+1)
        z_fft_masked = z_fft.clone()
        z_fft_masked[:, mask_idx] = 0.0

        # Reconstruct masked IQ signal
        z_masked = torch.fft.irfft(z_fft_masked, n=N)  # (B, N)
        iq_masked = torch.stack([z_masked.real, z_masked.imag], dim=1)  # (B, 2, N)

        # Get embedding from expert on masked input
        emb = self.expert.get_embedding(iq_masked)  # (B, embed_dim)

        # Predict masked frequency content
        pred = self.predictor(emb)  # (B, pred_dim)

        # Compress target to match pred_dim via adaptive pooling
        target_compressed = F.adaptive_avg_pool1d(
            target.unsqueeze(1), pred.shape[-1]
        ).squeeze(1)  # (B, pred_dim)

        # MSE loss on masked frequencies
        loss = F.mse_loss(pred, target_compressed.detach())
        return loss
