"""JEPA (Joint Embedding Predictive Architecture) self-supervised pretraining for RF signals.

Implements WirelessJEPA-style pretraining (arXiv:2601.20190) that learns by predicting
latent representations of masked regions rather than reconstructing raw signals.

Key insight: predict in latent space, not pixel/signal space. This forces the model to
learn semantic representations rather than low-level signal statistics.
"""

import copy
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rfml.training.jepa")


class JEPAMasking:
    """Structured masking strategies for IQ signals.

    Provides four masking strategies tailored to the temporal and channel
    structure of IQ (in-phase/quadrature) RF signals. Structured masks
    (contiguous blocks) are more effective than random masking for temporal
    signals because they force the model to reason about signal continuity.

    All methods operate on batched tensors of shape (B, 2, N) where:
        B = batch size
        2 = I and Q channels
        N = number of time samples

    Returns are consistent across methods:
        masked_x: input with masked positions zeroed out
        mask_indices: flat indices of masked time positions (B, num_masked)
        target_indices: flat indices of visible time positions (B, num_visible)
    """

    def time_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask a single contiguous block of time steps.

        Selects a random contiguous span covering approximately mask_ratio
        of the time axis. Effective for temporal RF signals because it
        challenges the predictor to infer large missing regions.

        Args:
            x: IQ tensor of shape (B, 2, N).
            mask_ratio: Fraction of time steps to mask (default 0.5).

        Returns:
            masked_x: Copy of x with masked time steps zeroed (B, 2, N).
            mask_indices: Masked time-step indices (B, num_masked).
            target_indices: Visible time-step indices (B, num_visible).
        """
        B, C, N = x.shape
        num_masked = int(N * mask_ratio)
        num_visible = N - num_masked

        # Sample a random start position for each batch element
        max_start = N - num_masked
        starts = torch.randint(0, max(max_start, 1), (B,), device=x.device)

        # Build index tensors
        arange = torch.arange(N, device=x.device)
        mask_indices = torch.stack(
            [arange[s : s + num_masked] for s in starts.tolist()], dim=0
        )  # (B, num_masked)

        # Visible indices: complement of masked block
        target_indices_list: List[torch.Tensor] = []
        for s in starts.tolist():
            before = arange[:s]
            after = arange[s + num_masked :]
            target_indices_list.append(torch.cat([before, after], dim=0))
        target_indices = torch.stack(target_indices_list, dim=0)  # (B, num_visible)

        masked_x = self._apply_mask(x, mask_indices)
        return masked_x, mask_indices, target_indices

    def channel_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask the I or Q channel entirely for each batch element.

        Randomly selects one channel (I=0 or Q=1) to zero out per sample.
        This challenges the model to reconstruct missing channel information
        from the surviving channel, exploiting I/Q correlation structure.

        When mask_ratio < 0.5, partial time masking is applied within
        the selected channel to honour the requested ratio (the entire
        channel being masked counts as ~0.5 for a 2-channel signal).

        Args:
            x: IQ tensor of shape (B, 2, N).
            mask_ratio: Ignored (full channel is always masked). Kept for
                interface consistency.

        Returns:
            masked_x: Copy of x with one channel zeroed per sample (B, 2, N).
            mask_indices: Masked time-step indices (B, N). Returns all N
                steps to indicate the full channel span.
            target_indices: Visible time-step indices (B, N). Returns full
                range of the surviving channel positions.
        """
        B, C, N = x.shape
        masked_x = x.clone()

        # Random channel selection per batch element
        channels_to_mask = torch.randint(0, C, (B,), device=x.device)
        for b in range(B):
            masked_x[b, channels_to_mask[b], :] = 0.0

        # mask_indices / target_indices defined over time axis (per MAE convention)
        arange = torch.arange(N, device=x.device)
        mask_indices = arange.unsqueeze(0).expand(B, -1)   # all time steps "masked"
        target_indices = arange.unsqueeze(0).expand(B, -1)  # all still "visible" in the other channel
        return masked_x, mask_indices, target_indices

    def multi_block_masking(
        self,
        x: torch.Tensor,
        num_blocks: int = 4,
        mask_ratio: float = 0.6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask multiple non-overlapping contiguous blocks.

        Partitions the time axis into num_blocks equal segments and randomly
        selects a subset to mask according to mask_ratio. Non-overlapping
        blocks ensure full coverage of the masked budget.

        Args:
            x: IQ tensor of shape (B, 2, N).
            num_blocks: Number of segments to divide the time axis into.
            mask_ratio: Fraction of blocks to mask (default 0.6).

        Returns:
            masked_x: Copy of x with selected blocks zeroed (B, 2, N).
            mask_indices: Concatenated masked time-step indices (B, num_masked).
            target_indices: Remaining visible time-step indices (B, num_visible).
        """
        B, C, N = x.shape
        block_size = N // num_blocks
        num_mask_blocks = max(1, int(num_blocks * mask_ratio))

        masked_x = x.clone()
        mask_indices_list: List[torch.Tensor] = []
        target_indices_list: List[torch.Tensor] = []
        arange = torch.arange(N, device=x.device)

        for b in range(B):
            # Randomly choose which blocks to mask
            perm = torch.randperm(num_blocks, device=x.device)
            masked_block_ids = perm[:num_mask_blocks].sort().values
            visible_block_ids = perm[num_mask_blocks:].sort().values

            masked_steps: List[torch.Tensor] = []
            visible_steps: List[torch.Tensor] = []

            for bid in range(num_blocks):
                start = bid * block_size
                end = start + block_size if bid < num_blocks - 1 else N
                block_idx = arange[start:end]
                if bid in masked_block_ids.tolist():
                    masked_steps.append(block_idx)
                    masked_x[b, :, start:end] = 0.0
                else:
                    visible_steps.append(block_idx)

            mask_indices_list.append(
                torch.cat(masked_steps, dim=0) if masked_steps else arange[:0]
            )
            target_indices_list.append(
                torch.cat(visible_steps, dim=0) if visible_steps else arange[:0]
            )

        # Pad to uniform length for batching
        mask_indices = self._pad_to_uniform(mask_indices_list)
        target_indices = self._pad_to_uniform(target_indices_list)
        return masked_x, mask_indices, target_indices

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard random (i.i.d.) masking over time steps (MAE baseline).

        Independently masks each time step with probability mask_ratio.
        Useful as a baseline to compare against structured masking strategies.

        Args:
            x: IQ tensor of shape (B, 2, N).
            mask_ratio: Fraction of time steps to mask (default 0.75).

        Returns:
            masked_x: Copy of x with masked time steps zeroed (B, 2, N).
            mask_indices: Masked time-step indices (B, num_masked).
            target_indices: Visible time-step indices (B, num_visible).
        """
        B, C, N = x.shape
        num_masked = int(N * mask_ratio)
        num_visible = N - num_masked

        # Per-sample random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)

        mask_indices = ids_shuffle[:, :num_masked].contiguous()
        target_indices = ids_shuffle[:, num_masked:].contiguous()

        masked_x = self._apply_mask(x, mask_indices)
        return masked_x, mask_indices, target_indices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_mask(x: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """Zero out time positions specified in mask_indices.

        Args:
            x: (B, C, N) tensor.
            mask_indices: (B, M) integer indices into the N dimension.

        Returns:
            Masked copy of x.
        """
        masked = x.clone()
        B, C, N = x.shape
        # Expand mask_indices to (B, C, M) for scatter
        idx = mask_indices.unsqueeze(1).expand(-1, C, -1)  # (B, C, M)
        masked.scatter_(2, idx, 0.0)
        return masked

    @staticmethod
    def _pad_to_uniform(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Pad a list of 1-D tensors to the same length and stack."""
        max_len = max(t.shape[0] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[0] < max_len:
                pad = t.new_zeros(max_len - t.shape[0])
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0)


# ---------------------------------------------------------------------------
# Predictor network
# ---------------------------------------------------------------------------


class JEPAPredictor(nn.Module):
    """Lightweight predictor that maps context representations to target representations.

    Operates entirely in latent space — the key JEPA design decision. A full
    pixel-space decoder would allow the model to collapse to low-level statistics;
    a lightweight latent-space predictor forces the encoder to capture structure
    that is actually useful for prediction.

    Architecture: positional encoding -> 2-layer depthwise separable convolution stack.
    Depthwise separable convolutions are used because they are parameter-efficient
    and respect the temporal structure of the latent sequence.

    Args:
        embed_dim: Dimensionality of encoder output tokens.
        depth: Number of depthwise separable conv layers (default 2).
        kernel_size: Temporal receptive field of each conv layer (default 7).
        max_seq_len: Maximum number of latent positions (for positional encoding).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 2,
        kernel_size: int = 7,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable positional encoding over latent positions
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, max_seq_len)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Depthwise separable conv layers
        layers: List[nn.Module] = []
        for _ in range(depth):
            layers.extend([
                # Depthwise conv: per-channel temporal filtering
                nn.Conv1d(
                    embed_dim,
                    embed_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=embed_dim,
                    bias=False,
                ),
                # Pointwise conv: channel mixing
                nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False),
                nn.LayerNorm(embed_dim),  # applied after transpose
                nn.GELU(),
            ])
        self.conv_layers = nn.ModuleList(layers)

        # Output projection (latent -> latent, same dim as target encoder)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        context_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Predict representations at all positions from context representations.

        The predictor is applied to the full context representation sequence
        (visible positions only, already embedded). It propagates information
        to the masked positions through its receptive field.

        Args:
            context_repr: Context encoder output at visible positions,
                shape (B, num_visible, embed_dim). Expected to be pre-padded
                to the full sequence length with zeros at masked positions.

        Returns:
            predicted_repr: Predicted latent representations at all positions,
                shape (B, num_visible, embed_dim).
        """
        # context_repr: (B, L, D) -> transpose to (B, D, L) for Conv1d
        x = context_repr.transpose(1, 2)  # (B, D, L)
        L = x.shape[2]

        # Add positional encoding (truncated to actual length)
        x = x + self.pos_embed[:, :, :L]

        # Apply depthwise separable conv layers
        # Layers are stored flat: [dw_conv, pw_conv, norm, act] * depth
        for i in range(0, len(self.conv_layers), 4):
            dw = self.conv_layers[i]
            pw = self.conv_layers[i + 1]
            norm = self.conv_layers[i + 2]
            act = self.conv_layers[i + 3]

            residual = x
            x = dw(x)
            x = pw(x)
            # LayerNorm expects (..., D): transpose, norm, transpose back
            x = norm(x.transpose(1, 2)).transpose(1, 2)
            x = act(x)
            x = x + residual  # residual connection

        # (B, D, L) -> (B, L, D)
        x = x.transpose(1, 2)
        x = self.out_proj(x)
        return x


# ---------------------------------------------------------------------------
# VICReg loss
# ---------------------------------------------------------------------------


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance regularization loss.

    Prevents representation collapse in JEPA training by enforcing three
    criteria on the predicted and target representation distributions:

    - Invariance: predicted and target representations should be similar
      (standard L2 loss).
    - Variance: each dimension of z should have std >= 1 across the batch
      (prevents collapse to a constant).
    - Covariance: off-diagonal elements of the covariance matrix should be
      small (prevents dimensions from encoding the same information).

    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning", ICLR 2022.

    Args:
        lambda_inv: Weight for invariance loss (default 25.0).
        lambda_var: Weight for variance loss (default 25.0).
        lambda_cov: Weight for covariance loss (default 1.0).
        eps: Numerical stability epsilon for std computation (default 1e-4).
    """

    def __init__(
        self,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.eps = eps

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute VICReg loss between predicted and target representations.

        Args:
            z_pred: Predicted representations at masked positions,
                shape (N_total, D) where N_total = B * num_masked.
            z_target: Target representations at masked positions,
                same shape as z_pred.

        Returns:
            total_loss: Weighted sum of the three VICReg terms.
            components: Dict with keys 'invariance', 'variance', 'covariance'
                containing the unweighted loss terms for logging.
        """
        # Invariance: MSE between predicted and target (collapsed across batch)
        inv_loss = F.mse_loss(z_pred, z_target)

        # Variance: encourage std >= 1 along each feature dimension
        var_loss = self._variance_loss(z_pred) + self._variance_loss(z_target)

        # Covariance: penalise off-diagonal covariance elements
        cov_loss = self._covariance_loss(z_pred) + self._covariance_loss(z_target)

        total = (
            self.lambda_inv * inv_loss
            + self.lambda_var * var_loss
            + self.lambda_cov * cov_loss
        )

        components = {
            "invariance": inv_loss.detach(),
            "variance": var_loss.detach(),
            "covariance": cov_loss.detach(),
        }
        return total, components

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Hinge loss encouraging std(z) >= 1 along each dimension.

        Args:
            z: Representations (N, D).

        Returns:
            Scalar variance loss.
        """
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return F.relu(1.0 - std).mean()

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Penalise off-diagonal elements of the feature covariance matrix.

        Args:
            z: Representations (N, D).

        Returns:
            Scalar covariance loss.
        """
        N, D = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / max(N - 1, 1)
        # Zero out diagonal, keep off-diagonal
        diag_mask = torch.eye(D, device=z.device, dtype=torch.bool)
        off_diag = cov[~diag_mask]
        return (off_diag ** 2).sum() / D


# ---------------------------------------------------------------------------
# Main JEPA pretraining module
# ---------------------------------------------------------------------------


class JEPAPretraining(nn.Module):
    """JEPA pretraining module with EMA target encoder and structured masking.

    Implements the WirelessJEPA training loop:
      1. Apply structured masking to input x.
      2. Pass visible patches through context_encoder -> context_repr.
      3. Pass full input through target_encoder (no grad) -> target_repr.
      4. predictor(context_repr) -> predicted_repr at masked positions.
      5. VICReg loss between predicted_repr and target_repr[masked].
      6. Update target_encoder via EMA.

    The target encoder is never directly optimised — it is updated via
    exponential moving average of the context encoder. This prevents
    the trivial solution where context and target encoders collapse together.

    Args:
        encoder: The main context encoder being trained (e.g. IQ expert backbone).
            Must accept (B, 2, N) tensors. Output should be (B, D) or (B, L, D).
        embed_dim: Dimensionality of encoder output (default 512).
        predictor_depth: Number of conv layers in JEPAPredictor (default 2).
        momentum: Initial EMA coefficient for target encoder updates (default 0.996).
        masking_strategy: Which masking strategy to use during training.
            One of 'time', 'channel', 'multi_block', 'random' (default 'time').
        mask_ratio: Fraction of signal to mask (default 0.5).
        use_vicreg: Whether to use VICReg loss instead of smooth_l1 (default True).
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 512,
        predictor_depth: int = 2,
        momentum: float = 0.996,
        masking_strategy: str = "time",
        mask_ratio: float = 0.5,
        use_vicreg: bool = True,
    ):
        super().__init__()

        if masking_strategy not in ("time", "channel", "multi_block", "random"):
            raise ValueError(
                f"masking_strategy must be one of 'time', 'channel', 'multi_block', "
                f"'random'. Got: {masking_strategy!r}"
            )

        self.embed_dim = embed_dim
        self.momentum = momentum
        self.masking_strategy = masking_strategy
        self.mask_ratio = mask_ratio
        self.use_vicreg = use_vicreg

        # Context encoder (trained via backprop)
        self.context_encoder = encoder

        # Target encoder (updated only via EMA — no gradients)
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Lightweight latent-space predictor
        self.predictor = JEPAPredictor(embed_dim=embed_dim, depth=predictor_depth)

        # Loss
        self.vicreg = VICRegLoss() if use_vicreg else None

        # Masking utility
        self.masking = JEPAMasking()

        logger.info(
            "JEPAPretraining initialised | embed_dim=%d | strategy=%s | "
            "mask_ratio=%.2f | momentum=%.4f | vicreg=%s",
            embed_dim,
            masking_strategy,
            mask_ratio,
            momentum,
            use_vicreg,
        )

    def _get_encoder_output(
        self, encoder: nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        """Run encoder and normalise output shape to (B, L, D).

        Handles encoders that return (B, D) flat embeddings by expanding
        them to (B, 1, D), and encoders that return (B, L, D) sequences
        directly.

        Args:
            encoder: Encoder module.
            x: Input tensor (B, 2, N).

        Returns:
            Encoder output (B, L, D).
        """
        out = encoder(x)
        if out.dim() == 2:
            # (B, D) -> (B, 1, D)
            out = out.unsqueeze(1)
        return out

    def _apply_masking(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch to the selected masking strategy.

        Args:
            x: IQ tensor (B, 2, N).

        Returns:
            masked_x, mask_indices, target_indices
        """
        if self.masking_strategy == "time":
            return self.masking.time_masking(x, mask_ratio=self.mask_ratio)
        elif self.masking_strategy == "channel":
            return self.masking.channel_masking(x, mask_ratio=self.mask_ratio)
        elif self.masking_strategy == "multi_block":
            return self.masking.multi_block_masking(x, mask_ratio=self.mask_ratio)
        else:  # random
            return self.masking.random_masking(x, mask_ratio=self.mask_ratio)

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        """Update target encoder parameters via EMA.

        θ_target = momentum * θ_target + (1 - momentum) * θ_context
        """
        for param_ctx, param_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_tgt.data.mul_(self.momentum).add_(
                param_ctx.data, alpha=1.0 - self.momentum
            )

    def update_momentum(self, current_step: int, total_steps: int) -> None:
        """Update the EMA momentum coefficient on a cosine schedule.

        Follows the WirelessJEPA schedule: momentum increases from the initial
        value (e.g. 0.996) to 1.0 over the full training run. Higher momentum
        at the end of training stabilises the target encoder.

        Args:
            current_step: Current global training step (0-indexed).
            total_steps: Total number of training steps.
        """
        # Cosine schedule: m(t) = 1 - (1 - m_base) * (cos(pi*t/T) + 1) / 2
        # At t=0: m = m_base; at t=T: m = 1.0
        m_base = self.momentum
        progress = current_step / max(total_steps - 1, 1)
        cosine_factor = (1.0 + math.cos(math.pi * progress)) / 2.0
        new_momentum = 1.0 - (1.0 - m_base) * cosine_factor
        self.momentum = float(new_momentum)
        logger.debug(
            "Momentum updated | step=%d/%d | momentum=%.6f",
            current_step,
            total_steps,
            self.momentum,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute JEPA pretraining loss for a batch of IQ signals.

        Steps:
          1. Structured masking -> masked_x, mask_indices, target_indices.
          2. context_encoder(masked_x) -> context_repr (B, L, D).
          3. target_encoder(x) (no grad) -> target_repr (B, L, D).
          4. predictor(context_repr) -> predicted_repr (B, L, D).
          5. Gather predicted and target at masked positions.
          6. VICReg or smooth_l1 loss on those positions.
          7. EMA update of target_encoder.

        Args:
            x: IQ input tensor of shape (B, 2, N).

        Returns:
            loss: Scalar pretraining loss.
            info: Dict with diagnostic values for logging:
                  'loss', 'invariance', 'variance', 'covariance' (last three
                  only when use_vicreg=True), 'momentum'.
        """
        B, C, N = x.shape

        # Step 1: masking
        masked_x, mask_indices, target_indices = self._apply_masking(x)

        # Step 2: context encoder on masked input
        context_repr = self._get_encoder_output(self.context_encoder, masked_x)
        # context_repr: (B, L, D)

        # Step 3: target encoder on full input (no gradient flow)
        with torch.no_grad():
            target_repr = self._get_encoder_output(self.target_encoder, x)

        # Step 4: predictor maps context -> full sequence predictions
        predicted_repr = self.predictor(context_repr)
        # predicted_repr: (B, L, D)

        # Step 5: gather representations at masked positions.
        # mask_indices are over the time axis N; L (latent length) may differ
        # from N when the encoder downsamples. We clamp indices into [0, L-1].
        L = predicted_repr.shape[1]
        mask_idx_clamped = mask_indices.clamp(max=L - 1)  # (B, M)

        # Expand to (B, M, D) for gather
        M = mask_idx_clamped.shape[1]
        idx_exp = mask_idx_clamped.unsqueeze(-1).expand(-1, -1, self.embed_dim)

        pred_masked = torch.gather(predicted_repr, dim=1, index=idx_exp)
        tgt_masked = torch.gather(target_repr, dim=1, index=idx_exp)

        # Flatten to (B*M, D) for loss computation
        pred_flat = pred_masked.reshape(-1, self.embed_dim)
        tgt_flat = tgt_masked.reshape(-1, self.embed_dim)

        # Step 6: compute loss
        info: Dict[str, torch.Tensor] = {}
        if self.use_vicreg and self.vicreg is not None:
            loss, components = self.vicreg(pred_flat, tgt_flat)
            info.update(components)
        else:
            loss = F.smooth_l1_loss(pred_flat, tgt_flat)

        info["loss"] = loss.detach()
        info["momentum"] = torch.tensor(self.momentum, device=x.device)

        # Step 7: EMA update (during training; no-op in eval mode)
        if self.training:
            self._update_target_encoder()

        logger.debug(
            "JEPA forward | strategy=%s | loss=%.4f | momentum=%.5f",
            self.masking_strategy,
            loss.item(),
            self.momentum,
        )
        return loss, info


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_jepa_trainer(
    model: nn.Module,
    config: Dict,
) -> "JEPAPretraining":
    """Create a JEPAPretraining module from a model and configuration dict.

    This factory function resolves config keys to JEPAPretraining constructor
    arguments, providing sensible defaults for any unspecified values.

    Supported config keys:
        embed_dim (int): Encoder output dimensionality. Default 512.
        predictor_depth (int): Conv layers in predictor. Default 2.
        momentum (float): Initial EMA momentum. Default 0.996.
        masking_strategy (str): One of 'time', 'channel', 'multi_block', 'random'.
            Default 'time'.
        mask_ratio (float): Fraction of signal to mask. Default 0.5.
        use_vicreg (bool): Use VICReg loss. Default True.

    Args:
        model: The encoder backbone to pretrain (e.g. IQExpert or similar).
        config: Configuration dictionary. Unknown keys are ignored.

    Returns:
        JEPAPretraining module ready for training.

    Example::

        trainer = create_jepa_trainer(
            model=my_encoder,
            config={
                "embed_dim": 512,
                "masking_strategy": "multi_block",
                "mask_ratio": 0.6,
                "momentum": 0.996,
            },
        )
        loss, info = trainer(batch)
        optimizer.step()
        trainer.update_momentum(step, total_steps)
    """
    embed_dim = int(config.get("embed_dim", 512))
    predictor_depth = int(config.get("predictor_depth", 2))
    momentum = float(config.get("momentum", 0.996))
    masking_strategy = str(config.get("masking_strategy", "time"))
    mask_ratio = float(config.get("mask_ratio", 0.5))
    use_vicreg = bool(config.get("use_vicreg", True))

    logger.info(
        "Creating JEPAPretraining | embed_dim=%d | strategy=%s | "
        "mask_ratio=%.2f | momentum=%.4f | vicreg=%s",
        embed_dim,
        masking_strategy,
        mask_ratio,
        momentum,
        use_vicreg,
    )

    return JEPAPretraining(
        encoder=model,
        embed_dim=embed_dim,
        predictor_depth=predictor_depth,
        momentum=momentum,
        masking_strategy=masking_strategy,
        mask_ratio=mask_ratio,
        use_vicreg=use_vicreg,
    )
