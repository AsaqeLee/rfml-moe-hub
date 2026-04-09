"""Mamba/SSM-based IQ Expert: Selective State Space Model for raw IQ data.

Implements a Mamba-based backbone as an alternative to the transformer-based
IQExpert, achieving O(L) complexity instead of O(L^2) for long IQ sequences.

Based on the MAMCA architecture (arXiv:2405.11263) which combines Mamba selective
state space models with attention-based soft thresholding for robust signal
classification under low-SNR conditions.

Key references:
    - Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
    - MAMCA: A Mamba-Attention Architecture for Signal Modulation Classification (2024)
    - HiPPO: Recurrent Memory with Optimal Polynomial Projections (Gu et al., 2020)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hippo_initializer(d_state: int) -> torch.Tensor:
    """Initialize the A matrix using HiPPO-LegS (Legendre State Space).

    The HiPPO framework provides a principled initialization for the state
    matrix A that enables long-range dependency modeling. The diagonal
    approximation A_n = -(n + 1/2) captures the key spectral properties.

    Reference: Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial
    Projections", NeurIPS 2020.

    Args:
        d_state: State dimension N.

    Returns:
        Tensor of shape (d_state,) with HiPPO-LegS diagonal entries.
    """
    return -(torch.arange(1, d_state + 1, dtype=torch.float32) + 0.5)


class SelectiveSSMBlock(nn.Module):
    """Core Selective State Space Model block (Mamba S6).

    Implements the selective scan mechanism where the SSM parameters B, C, and
    the discretization step dt are input-dependent, allowing the model to
    selectively propagate or forget information along the sequence.

    The computation follows:
        1. Input projection: x, z = split(Linear(u))
        2. Conv1d for local feature extraction on x path
        3. Input-dependent SSM parameters: dt, B, C from x
        4. Discretization: A_bar = exp(dt * A), B_bar = dt * B
        5. Sequential scan: h(t) = A_bar * h(t-1) + B_bar * x(t)
                            y(t) = C * h(t) + D * x(t)
        6. Output gating: y * SiLU(z)

    Args:
        d_model: Input/output dimension.
        d_state: SSM state expansion factor N (default: 16).
        d_inner: Inner dimension for input projection (default: 2 * d_model).
        dt_rank: Rank of dt projection (default: ceil(d_model / 16)).
        conv_kernel: Kernel size for local convolution (default: 4).
        conv_bias: Whether to use bias in Conv1d (default: True).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_inner: int | None = None,
        dt_rank: int | None = None,
        conv_kernel: int = 4,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner or 2 * d_model
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # Input projection: produces x and z paths
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Conv1d for local feature extraction (causal, applied to x path)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,  # causal padding (trim later)
            groups=self.d_inner,
            bias=conv_bias,
        )

        # SSM parameter projections (input-dependent / selective)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt bias for stable discretization
        # Inverse softplus of uniform [dt_min, dt_max] for initialization
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt_bias = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        # Inverse softplus so that softplus(bias) recovers the desired range
        self.dt_proj.bias.data = dt_bias + torch.log(-torch.expm1(-dt_bias))

        # State matrix A: diagonal, initialized via HiPPO
        # Stored in log space for numerical stability (A is negative)
        A = _hippo_initializer(d_state).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(A.log())  # (d_inner, d_state)

        # Skip connection parameter D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential selective scan over the sequence.

        Implements the recurrence:
            h(t) = A_bar(t) * h(t-1) + B_bar(t) * x(t)
            y(t) = C(t) * h(t)

        where A_bar and B_bar are discretized using the Zero-Order Hold (ZOH):
            A_bar = exp(dt * A)
            B_bar = dt * B

        Note: This is a sequential implementation. A work-efficient parallel
        scan (Blelloch, 1990) can replace this for GPU-optimized training.

        Args:
            x: Input of shape (B, L, D) where D = d_inner.
            dt: Discretization steps of shape (B, L, D).
            B: Input-dependent B matrix of shape (B, L, N).
            C: Input-dependent C matrix of shape (B, L, N).

        Returns:
            Output of shape (B, L, D).
        """
        batch, seq_len, d_inner = x.shape
        d_state = self.d_state

        # Recover A from log space (always negative)
        A = -self.A_log.exp()  # (d_inner, d_state)

        # Discretize: ZOH
        # dt: (B, L, D) -> A_bar: (B, L, D, N)
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, L, D, N)
        A_bar = torch.exp(dt_A)
        # B_bar: (B, L, D, N)
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, 1, N) * broadcast

        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            # h = A_bar * h + B_bar * x
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            # y = C * h, summed over state dim
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, D)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Selective SSM block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Project input to x and z paths
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_path, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Conv1d on x path (for local feature extraction)
        # Reshape for Conv1d: (B, D, L)
        x_path = x_path.transpose(1, 2)
        x_path = self.conv1d(x_path)[:, :, :seq_len]  # trim causal padding
        x_path = x_path.transpose(1, 2)  # back to (B, L, D)
        x_path = F.silu(x_path)

        # Compute input-dependent SSM parameters
        x_proj = self.x_proj(x_path)  # (B, L, dt_rank + 2*d_state)
        dt_x, B, C = x_proj.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # dt: project from dt_rank to d_inner and apply softplus
        dt = F.softplus(self.dt_proj(dt_x))  # (B, L, d_inner)

        # Selective scan
        y = self._selective_scan(x_path, dt, B, C)

        # Add skip connection (D * x)
        y = y + x_path * self.D.unsqueeze(0).unsqueeze(0)

        # Gate with z path
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Mamba block with pre-norm residual connections.

    Structure:
        x -> LayerNorm -> SelectiveSSMBlock -> + residual
          -> LayerNorm -> FFN (SiLU)         -> + residual

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_inner: Inner dimension for SSM projection.
        ffn_mult: FFN hidden dimension multiplier (default: 4.0).
        dropout: Dropout rate for FFN.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_inner: int | None = None,
        ffn_mult: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # SSM branch
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSMBlock(d_model=d_model, d_state=d_state, d_inner=d_inner)

        # FFN branch
        self.norm2 = nn.LayerNorm(d_model)
        ffn_hidden = int(d_model * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SoftThresholdDenoiser(nn.Module):
    """Attention-based soft thresholding for low-SNR robustness.

    From the MAMCA architecture (arXiv:2405.11263): learns per-channel
    thresholds to suppress noise while preserving signal features. Uses
    soft thresholding (shrinkage) instead of hard thresholding to avoid
    gradient vanishing.

    The shrinkage function is:
        S(x, tau) = sign(x) * max(|x| - tau, 0)

    where tau >= 0 is a learned threshold per channel. This provides a
    smooth, differentiable denoising operation that acts as a nonlinear
    attention mechanism on the feature magnitudes.

    Args:
        d_model: Feature dimension (number of channels).
        init_threshold: Initial threshold value (default: 0.1).
    """

    def __init__(self, d_model: int, init_threshold: float = 0.1):
        super().__init__()
        # Learnable thresholds, one per channel
        self.threshold_logit = nn.Parameter(
            torch.full((d_model,), math.log(init_threshold))
        )
        # Channel attention for adaptive thresholding
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply soft thresholding denoising.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Denoised output of shape (batch, seq_len, d_model).
        """
        # Compute adaptive thresholds via channel attention
        # Pool over sequence dimension: (B, D, L) -> attention weights
        x_t = x.transpose(1, 2)  # (B, D, L)
        attn = self.channel_attn(x_t)  # (B, D)

        # Base threshold (always positive via exp)
        base_threshold = self.threshold_logit.exp()  # (D,)

        # Scale threshold by channel attention
        tau = (base_threshold.unsqueeze(0) * attn).unsqueeze(1)  # (B, 1, D)

        # Soft thresholding (shrinkage function)
        return torch.sign(x) * F.relu(x.abs() - tau)


class MambaIQExpert(nn.Module):
    """Mamba-based IQ Expert using Selective State Space Models.

    An alternative backbone to the transformer-based IQExpert that achieves
    O(L) complexity for processing long IQ sequences (N=32768), compared to
    the O(L^2) complexity of self-attention.

    Based on the MAMCA architecture (arXiv:2405.11263) combining Mamba SSM
    blocks with soft thresholding denoisers for robust signal classification
    under low-SNR conditions.

    Architecture:
        1. Input embedding: Conv1d(2, d_model, kernel_size=7, stride=4)
           reduces sequence from 32768 to 8192 while projecting IQ channels.
        2. Stack of 8 MambaBlocks (d_model=384, d_state=16, d_inner=768).
        3. SoftThresholdDenoiser inserted after blocks 3 and 6 for
           progressive noise suppression.
        4. Global average pooling over sequence -> 512-dim embedding.

    Input:  (batch, 2, N) where channel 0 = I, channel 1 = Q, N=32768
    Output: (batch, num_classes) via forward(), (batch, 512) via get_embedding()

    Args:
        num_classes: Number of output classes (default: 10).
        embed_dim: Output embedding dimension (default: 512).
        d_model: Internal model dimension (default: 384).
        d_state: SSM state dimension (default: 16).
        d_inner: SSM inner dimension (default: 768).
        n_layers: Number of Mamba blocks (default: 8).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 512,
        d_model: int = 384,
        d_state: int = 16,
        d_inner: int = 768,
        n_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_model = d_model

        # --- Input embedding ---
        # Conv1d to project 2 IQ channels to d_model and reduce sequence length
        # 32768 / stride=4 = 8192 sequence positions
        self.input_embed = nn.Sequential(
            nn.Conv1d(2, d_model, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(24, d_model),
            nn.SiLU(),
        )

        # --- Mamba blocks with interleaved denoisers ---
        self.blocks = nn.ModuleList()
        self.denoisers = nn.ModuleDict()
        for i in range(n_layers):
            self.blocks.append(
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_inner=d_inner,
                    dropout=dropout,
                )
            )
            # Insert denoisers after blocks 3 and 6 (0-indexed: 2 and 5)
            if i in (2, 5):
                self.denoisers[str(i)] = SoftThresholdDenoiser(d_model)

        self.final_norm = nn.LayerNorm(d_model)

        # --- Embedding projection ---
        self.embed_proj = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following Mamba conventions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, "weight"):
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run Mamba backbone and return embedding.

        Args:
            x: Raw IQ tensor of shape (batch, 2, N).

        Returns:
            Embedding of shape (batch, embed_dim).
        """
        # Input embedding: (B, 2, N) -> (B, d_model, L)
        x = self.input_embed(x)

        # Reshape for Mamba blocks: (B, L, d_model)
        x = x.transpose(1, 2)

        # Process through Mamba blocks with interleaved denoisers
        for i, block in enumerate(self.blocks):
            x = block(x)
            if str(i) in self.denoisers:
                x = self.denoisers[str(i)](x)

        x = self.final_norm(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (B, d_model)

        # Project to embedding dimension
        x = self.embed_proj(x)  # (B, embed_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

        Args:
            x: Raw IQ tensor of shape (batch, 2, N).

        Returns:
            Logits of shape (batch, num_classes).
        """
        emb = self._extract_features(x)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional embedding for the MoE gate.

        Args:
            x: Raw IQ tensor of shape (batch, 2, N).

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
