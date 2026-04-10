"""SpectrumFM dual-objective self-supervised pretraining module.

Implements the CNN+MHSA hybrid encoder with two simultaneous SSL objectives:
  1. MaskedReconstructionTask: mask 15% of input, reconstruct via decoder.
  2. NextSlotPredictionTask: predict the k-th symbol from observed sequence.

Based on SpectrumFM architecture (ResearchGate 2024).
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rfml.training")


class SpectrumFMEncoder(nn.Module):
    """Hybrid CNN+MHSA encoder from SpectrumFM.

    CNN blocks extract local spectral patterns; transformer layers capture
    long-range dependencies across the symbol sequence.

    Can be used as a feature backbone for any expert: accepts IQ tensors
    [B, 2, N] and returns contextualized sequence embeddings [B, seq_len, d_model].

    Args:
        input_dim: Input feature dimension (after CNN projection). For raw IQ
            pass input_dim=2; the CNN will project to d_model internally.
        d_model: Hidden dimension of transformer layers (default 128).
        d_ff: Feedforward dimension inside each transformer layer (default 256).
        num_layers: Number of transformer encoder layers (default 4).
        num_heads: Number of attention heads (default 4).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_ff: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # CNN feature extractor: local spectral pattern detection
        # Three Conv1d stages with increasing receptive field
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Positional encoding (fixed sinusoidal, max 2048 steps)
        self._max_len = 2048
        pe = self._build_sinusoidal_pe(self._max_len, d_model)
        self.register_buffer("pe", pe)  # (1, max_len, d_model)

        # Transformer layers: MHSA for global dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Build sinusoidal positional encoding table."""
        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return pe

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input sequence.

        Args:
            x: Input tensor. Shape [B, input_dim, N] (channel-first IQ or feature map).
            src_key_padding_mask: Boolean mask [B, N]; True positions are ignored.

        Returns:
            Contextualized embeddings [B, N, d_model].
        """
        # CNN: (B, input_dim, N) -> (B, d_model, N)
        h = self.cnn(x)
        # Transpose to sequence-first for transformer: (B, N, d_model)
        h = h.permute(0, 2, 1)
        seq_len = h.size(1)
        h = h + self.pe[:, :seq_len, :]
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        return h


class MaskedReconstructionTask(nn.Module):
    """Mask 15% of normalized input sequence, reconstruct via lightweight decoder.

    - Generate binary mask m where P(m_i=1) = mask_ratio (default 0.15).
    - Masked sequence: x_masked = x * (1 - m), expanded to channel dim.
    - Attention mask: assign -inf to masked positions so encoder cannot attend
      to them (simulated via src_key_padding_mask passed to SpectrumFMEncoder).
    - Lightweight MLP decoder reconstructs x_hat from encoder output.
    - Loss: MSE on masked positions only.

    Args:
        encoder: SpectrumFMEncoder (or compatible module).
        input_dim: Number of input channels / feature dim per time step.
        decoder_dim: Hidden dimension of the decoder MLP (default 256).
        mask_ratio: Probability of masking each position (default 0.15).
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        decoder_dim: int = 256,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.encoder = encoder
        self.input_dim = input_dim
        self.mask_ratio = mask_ratio

        d_model = getattr(encoder, "d_model", decoder_dim)

        # Lightweight decoder: two-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(d_model, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute masked reconstruction loss.

        Args:
            x: Input tensor [B, input_dim, N] (channel-first).

        Returns:
            loss: MSE reconstruction loss on masked positions (scalar).
            mask: Binary mask [B, N], 1 = masked position.
        """
        B, C, N = x.shape
        device = x.device

        # Generate binary mask: (B, N), 1 = masked
        mask = (torch.rand(B, N, device=device) < self.mask_ratio).float()

        # Zero out masked time positions: broadcast mask over channels
        # x_masked: (B, C, N)
        x_masked = x * (1.0 - mask).unsqueeze(1)

        # src_key_padding_mask: True where masked (encoder ignores these positions)
        padding_mask = mask.bool()  # (B, N)

        # Encode: (B, N, d_model)
        enc_out = self.encoder(x_masked, src_key_padding_mask=padding_mask)

        # Decode: (B, N, input_dim)
        x_hat = self.decoder(enc_out)

        # Target: transpose x to (B, N, C) for comparison
        target = x.permute(0, 2, 1)  # (B, N, C)

        # MSE on masked positions only
        loss = F.mse_loss(x_hat, target, reduction="none")  # (B, N, C)
        loss = loss.mean(dim=-1)  # (B, N) per-position
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)

        return loss, mask


class NextSlotPredictionTask(nn.Module):
    """Predict the k-th symbol from the observed sequence [x_1, ..., x_{k-1}].

    Uses causal (look-ahead) masking to prevent information leakage — position i
    can only attend to positions < i. Trains temporal awareness and spectral
    continuity by autoregressively predicting the next pred_steps symbols.

    Args:
        encoder: SpectrumFMEncoder (or compatible module).
        input_dim: Number of input channels / feature dim per time step.
        pred_steps: Number of future steps to predict (default 1).
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        pred_steps: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.input_dim = input_dim
        self.pred_steps = pred_steps

        d_model = getattr(encoder, "d_model", 128)

        # Prediction head: project encoder output to input space
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, input_dim * pred_steps),
        )

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Build upper-triangular causal mask [seq_len, seq_len].

        Positions where mask[i, j] = True are blocked (j > i).
        Compatible with nn.TransformerEncoderLayer attn_mask convention.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute next-slot prediction loss.

        The encoder processes positions [0 .. N-pred_steps-1] under causal
        masking; the prediction head at position i predicts x[i+1 .. i+pred_steps].

        Args:
            x: Input tensor [B, input_dim, N] (channel-first).

        Returns:
            loss: MSE prediction loss (scalar).
        """
        B, C, N = x.shape
        if N <= self.pred_steps:
            return x.new_zeros(())

        # Context: all but the last pred_steps positions
        ctx_len = N - self.pred_steps
        x_ctx = x[:, :, :ctx_len]  # (B, C, ctx_len)

        # Build causal attention mask for the transformer inside the encoder.
        # We pass it via a monkey-patch approach: temporarily override the
        # transformer's forward to use an attn_mask.
        # Simpler: use src_key_padding_mask=None and rely on the causal mask
        # injected at each layer. Since nn.TransformerEncoder does not expose
        # per-call attn_mask easily across all layers, we pass it as the mask
        # argument which is forwarded as attn_mask to each layer.
        causal = self._causal_mask(ctx_len, x.device)  # (ctx_len, ctx_len)

        # CNN stage (manual, to pass attn_mask to transformer only)
        h = self.encoder.cnn(x_ctx)           # (B, d_model, ctx_len)
        h = h.permute(0, 2, 1)                # (B, ctx_len, d_model)
        h = h + self.encoder.pe[:, :ctx_len, :]
        h = self.encoder.transformer(h, mask=causal)
        h = self.encoder.norm(h)              # (B, ctx_len, d_model)

        # Predict from each context position
        preds = self.pred_head(h)             # (B, ctx_len, C * pred_steps)
        preds = preds.view(B, ctx_len, self.pred_steps, C)

        # Targets: x[i+1 .. i+pred_steps] for each i in [0, ctx_len)
        # Build target tensor (B, ctx_len, pred_steps, C)
        targets = torch.stack(
            [x[:, :, i + 1 : i + 1 + self.pred_steps].permute(0, 2, 1)
             for i in range(ctx_len)],
            dim=1,
        )  # (B, ctx_len, pred_steps, C)

        loss = F.mse_loss(preds, targets)
        return loss


class DualObjectivePretrainer(nn.Module):
    """Combine masked reconstruction and next-slot prediction for SpectrumFM pretraining.

    Total loss = alpha * L_masked + beta * L_nextslot.
    Default: alpha=0.7, beta=0.3 per the SpectrumFM paper.

    SpectrumFM encoder defaults:
        - L=4 transformer layers
        - d_model=128 (hidden dim)
        - d_ff=256 (feedforward dim)
        - num_heads=4
        - sequence_length=50 symbols per step
        - dropout=0.1

    Args:
        encoder: Pre-constructed SpectrumFMEncoder. If None, one is built from
            the remaining kwargs.
        input_dim: Input channels per time step (default 512 for feature tensors;
            use 2 for raw IQ [B, 2, N]).
        d_model: Transformer hidden dim (default 128).
        d_ff: Feedforward dim (default 256).
        num_layers: Number of transformer layers (default 4).
        num_heads: Number of attention heads (default 4).
        mask_ratio: Masking probability for reconstruction task (default 0.15).
        alpha: Weight for masked reconstruction loss (default 0.7).
        beta: Weight for next-slot prediction loss (default 0.3).
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        input_dim: int = 512,
        d_model: int = 128,
        d_ff: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mask_ratio: float = 0.15,
        alpha: float = 0.7,
        beta: float = 0.3,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        if encoder is None:
            encoder = SpectrumFMEncoder(
                input_dim=input_dim,
                d_model=d_model,
                d_ff=d_ff,
                num_layers=num_layers,
                num_heads=num_heads,
            )

        self.encoder = encoder

        self.masked_task = MaskedReconstructionTask(
            encoder=self.encoder,
            input_dim=input_dim,
            decoder_dim=d_ff,
            mask_ratio=mask_ratio,
        )
        self.nextslot_task = NextSlotPredictionTask(
            encoder=self.encoder,
            input_dim=input_dim,
            pred_steps=1,
        )

        logger.info(
            "DualObjectivePretrainer: alpha=%.2f (masked) beta=%.2f (nextslot) "
            "input_dim=%d d_model=%d num_layers=%d",
            alpha,
            beta,
            input_dim,
            d_model,
            num_layers,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute dual-objective pretraining loss.

        Args:
            x: Input tensor [B, input_dim, N] (channel-first). For raw IQ use
               [B, 2, N]; for feature tensors use [B, D, T].

        Returns:
            total_loss: Weighted sum alpha*L_masked + beta*L_nextslot.
            l_masked: Masked reconstruction loss.
            l_nextslot: Next-slot prediction loss.
        """
        l_masked, _mask = self.masked_task(x)
        l_nextslot = self.nextslot_task(x)

        total_loss = self.alpha * l_masked + self.beta * l_nextslot

        logger.debug(
            "DualObjective loss=%.4f (masked=%.4f nextslot=%.4f)",
            total_loss.item(),
            l_masked.item(),
            l_nextslot.item(),
        )

        return total_loss, l_masked, l_nextslot
