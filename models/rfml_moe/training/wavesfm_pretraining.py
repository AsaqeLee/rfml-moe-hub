"""WavesFM-based Masked Wireless Modeling (MWM) pretraining module.

Implements the ViT-based pretraining strategy from WavesFM (IEEE 2026) for
self-supervised representation learning on RF signals. Key components:

  - WirelessPatchEmbedding: 2D patch tokenizer for OFDM resource grids.
  - MaskedWirelessModeling: MAE-style pretraining with aggressive masking (40-75%).
  - LoRAAdapter: Low-rank adaptation for parameter-efficient fine-tuning.
  - apply_lora_to_model: Utility to inject LoRA into all Linear layers.

The aggressive masking ratio (vs. standard MAE 75%) forces the model to learn
richer spectral-temporal structure from sparse visible patches, which transfers
well to downstream RF classification and anomaly detection tasks.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rfml.training")


class WirelessPatchEmbedding(nn.Module):
    """Convert a 2D wireless signal (spectrogram / OFDM resource grid) into patch tokens.

    Uses a single Conv2d with kernel_size == stride == patch_size so each
    non-overlapping spatial patch maps to one embedding vector.

    Args:
        in_channels: Number of input channels (e.g. 3 for RGB spectrogram).
        patch_size: Height and width of each square patch in pixels.
        embed_dim: Dimensionality of the output patch embedding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # One conv with kernel == stride collapses each patch to a single vector.
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Embed patches from a 2D wireless signal.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            tokens: Patch embeddings of shape (B, num_patches, embed_dim).
            nH: Number of patch rows.
            nW: Number of patch columns.
        """
        B, C, H, W = x.shape
        # Trim to patch-aligned dimensions.
        H_trim = (H // self.patch_size) * self.patch_size
        W_trim = (W // self.patch_size) * self.patch_size
        if H_trim != H or W_trim != W:
            x = x[:, :, :H_trim, :W_trim]

        # (B, embed_dim, nH, nW)
        tokens = self.proj(x)
        nH, nW = tokens.shape[2], tokens.shape[3]
        # (B, nH*nW, embed_dim)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return tokens, nH, nW


class MaskedWirelessModeling(nn.Module):
    """MWM pretraining: mask patches, encode visible, reconstruct masked.

    Implements the WavesFM (IEEE 2026) pretraining recipe for OFDM resource
    grids.  An aggressive mask ratio (0.4-0.75) is applied so the encoder only
    processes a fraction of the patches, providing both a hard self-supervised
    task and significant compute savings during pretraining.

    The decoder is intentionally lightweight (2 transformer layers by default)
    and is discarded after pretraining; only the encoder weights are kept.

    Args:
        encoder: Backbone encoder (must accept patch tokens of shape
                 (B, num_visible, embed_dim) and return the same shape, or
                 a callable that returns (B, num_visible, embed_dim)).
        embed_dim: Embedding dimension expected by the encoder.
        decoder_dim: Hidden dimension of the reconstruction decoder.
        decoder_layers: Number of transformer layers in the decoder.
        decoder_heads: Number of attention heads in the decoder.
        mask_ratio: Fraction of patches to mask (0.4 to 0.75 recommended).
        patch_size: Patch size in pixels (must match WirelessPatchEmbedding).
        in_channels: Number of input image channels.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 256,
        decoder_dim: int = 128,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        mask_ratio: float = 0.5,
        patch_size: int = 16,
        in_channels: int = 3,
    ):
        super().__init__()
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_dim = in_channels * patch_size * patch_size

        # Patch embedding layer.
        self.patch_embed = WirelessPatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # Learnable mask token (decoder side).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Project encoder output to decoder dimension.
        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_dim)

        # Lightweight decoder transformer.
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

        # Project decoder output back to pixel space for each patch.
        self.decoder_proj = nn.Linear(decoder_dim, self.patch_dim)

        # Positional embeddings (encoder and decoder share the same grid).
        max_patches = 1024
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, max_patches, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed_enc, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_dec, std=0.02)

        logger.debug(
            "MaskedWirelessModeling: embed_dim=%d decoder_dim=%d "
            "mask_ratio=%.2f patch_size=%d",
            embed_dim,
            decoder_dim,
            mask_ratio,
            patch_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange (B, C, H, W) into flat patch vectors (B, N, C*p*p)."""
        B, C, H, W = x.shape
        p = self.patch_size
        nH, nW = H // p, W // p
        # Trim to patch-aligned size first.
        x = x[:, :, : nH * p, : nW * p]
        # (B, C, nH, p, nW, p) -> (B, nH*nW, C*p*p)
        x = x.reshape(B, C, nH, p, nW, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, nH * nW, C * p * p)
        return x

    def _random_mask(
        self, B: int, num_patches: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random masking indices.

        Returns:
            ids_keep: Indices of visible (unmasked) patches, shape (B, num_keep).
            ids_restore: Argsort to unshuffle back to original order, shape (B, N).
            mask: Binary tensor, 1 = masked, shape (B, N).
        """
        num_keep = int(num_patches * (1.0 - self.mask_ratio))

        noise = torch.rand(B, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]

        mask = torch.ones(B, num_patches, device=device)
        mask[:, :num_keep] = 0.0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def _get_encoder_output(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run encoder; supports get_embedding interface or direct call."""
        if hasattr(self.encoder, "get_embedding"):
            return self.encoder.get_embedding(tokens)
        return self.encoder(tokens)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Patchify, mask, encode visible patches, decode all, compute loss.

        Args:
            x: Input tensor of shape (B, C, H, W) — spectrogram or resource grid.

        Returns:
            loss: Scalar MSE reconstruction loss computed only on masked patches.
            mask: Binary tensor of shape (B, num_patches), 1 = patch was masked.
        """
        # 1. Embed patches and obtain flat pixel-level patches for the target.
        tokens, nH, nW = self.patch_embed(x)          # (B, N, embed_dim)
        patches = self._patchify(x)                    # (B, N, C*p*p)  — reconstruction target
        B, num_patches, _ = tokens.shape

        # Add encoder positional embeddings.
        tokens = tokens + self.pos_embed_enc[:, :num_patches]

        # 2. Random masking — select visible subset.
        ids_keep, ids_restore, mask = self._random_mask(B, num_patches, x.device)

        # Gather visible tokens only.
        ids_keep_exp = ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        visible_tokens = torch.gather(tokens, dim=1, index=ids_keep_exp)

        # 3. Encode visible patches.
        encoded = self._get_encoder_output(visible_tokens)   # (B, num_keep, embed_dim)

        # 4. Project to decoder dimension.
        encoded = self.encoder_to_decoder(encoded)           # (B, num_keep, decoder_dim)

        # Append mask tokens for masked positions.
        num_masked = num_patches - ids_keep.shape[1]
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        full_tokens = torch.cat([encoded, mask_tokens], dim=1)  # (B, N, decoder_dim)

        # Unshuffle to restore original patch order.
        ids_restore_exp = ids_restore.unsqueeze(-1).expand(-1, -1, full_tokens.shape[-1])
        full_tokens = torch.gather(full_tokens, dim=1, index=ids_restore_exp)

        # Add decoder positional embeddings.
        full_tokens = full_tokens + self.pos_embed_dec[:, :num_patches]

        # 5. Decode all positions.
        decoded = self.decoder(full_tokens)
        decoded = self.decoder_norm(decoded)
        pred = self.decoder_proj(decoded)                    # (B, N, C*p*p)

        # 6. MSE loss on masked patches only.
        loss = (pred - patches) ** 2                         # (B, N, patch_dim)
        loss = loss.mean(dim=-1)                             # (B, N) per-patch scalar
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)

        return loss, mask


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation (LoRA) wrapper for a single nn.Linear layer.

    The adapted forward computes:
        y = W x + (B @ A) x * scale

    where W is the frozen original weight, A has shape (rank, in_features),
    and B has shape (out_features, rank).  Only A and B are trained.

    Args:
        original_layer: The nn.Linear layer to adapt.
        rank: LoRA rank r; should be much smaller than min(in, out).
        scale: Scaling factor applied to the LoRA term (alpha/r in the paper).
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        scale: float = 1.0,
    ):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False
        if original_layer.bias is not None:
            original_layer.bias.requires_grad = False

        self.rank = rank
        self.scale = scale

        # A ~ N(0, 0.01), B = 0 so the adaptation starts at zero.
        self.lora_A = nn.Parameter(
            torch.randn(rank, original_layer.in_features) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(original_layer.out_features, rank)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply original linear + LoRA correction."""
        original_out = self.original(x)
        # (... , in) @ (in, r) @ (r, out) -> (... , out)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return original_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Replace Linear layers in a model with LoRA-adapted versions.

    Freezes all parameters first, then replaces each qualifying nn.Linear
    with a LoRAAdapter so only the LoRA matrices (A, B) are trainable.
    This typically leaves ~80% of parameters frozen.

    Args:
        model: The model to modify in-place.
        rank: LoRA rank applied to every replaced layer.
        target_modules: Optional allowlist of module name substrings.  If
            None, every nn.Linear in the model is replaced.  Example:
            ``["q_proj", "v_proj"]`` replaces only attention Q/V projections.

    Returns:
        stats: Dict with keys "total", "replaced", "trainable_params",
               "frozen_params" for quick logging.
    """
    # Freeze everything first.
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    total_linear = 0

    # Collect (parent_module, attr_name, child_module) triples for all Linear layers.
    replacements: List[Tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        for attr_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{attr_name}" if name else attr_name
                total_linear += 1
                if target_modules is None or any(t in full_name for t in target_modules):
                    replacements.append((module, attr_name, child))

    for parent, attr_name, layer in replacements:
        adapter = LoRAAdapter(layer, rank=rank)
        setattr(parent, attr_name, adapter)
        replaced += 1
        logger.debug("LoRA: replaced %s.%s (in=%d, out=%d, rank=%d)",
                     parent.__class__.__name__, attr_name,
                     layer.in_features, layer.out_features, rank)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    stats = {
        "total": total_linear,
        "replaced": replaced,
        "trainable_params": trainable,
        "frozen_params": frozen,
    }
    logger.info(
        "apply_lora_to_model: %d/%d Linear layers replaced with LoRA "
        "(rank=%d). Trainable params: %d / %d (%.1f%%)",
        replaced,
        total_linear,
        rank,
        trainable,
        trainable + frozen,
        100.0 * trainable / max(1, trainable + frozen),
    )
    return stats
