"""Expert Choice and Token Choice routers for Mixture-of-Experts gating."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple


class RouterOutput(NamedTuple):
    """Output from routing decisions."""
    weights: torch.Tensor        # (batch, top_k) routing weights
    expert_indices: torch.Tensor  # (batch, top_k) selected expert indices
    router_logits: torch.Tensor   # (batch, num_experts) raw logits
    load_balance_stats: dict      # per-expert load fractions


class ExpertChoiceRouter(nn.Module):
    """Expert Choice routing where each expert selects its top-c samples.

    Based on "Mixture-of-Experts with Expert Choice Routing" (Zhou et al., 2022).
    Each expert independently picks which samples to process, leading to better
    load balancing than token-choice routing.

    Args:
        input_dim: Dimension of concatenated expert embeddings (default 2048 = 4*512).
        num_experts: Number of experts to route across.
        top_k: Number of experts activated per sample.
        jitter_noise: Noise scale added during training for exploration.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        jitter_noise: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

        self.router = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_experts),
        )

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """Route inputs to experts via expert-choice mechanism.

        Args:
            x: Concatenated expert embeddings (batch, input_dim).

        Returns:
            RouterOutput with weights, indices, logits, and load stats.
        """
        batch_size = x.size(0)

        # Add jitter noise during training for exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(x) * self.jitter_noise
            x = x + noise

        # Compute router logits
        router_logits = self.router(x)  # (batch, num_experts)

        # Keep gating softmax in FP32 (Switch Transformer finding)
        logits_fp32 = router_logits.float()
        router_probs = F.softmax(logits_fp32, dim=-1)  # (batch, num_experts)

        # Expert Choice: each expert selects its top-c samples
        # Capacity per expert: c = (top_k * batch_size) / num_experts
        capacity = max(1, (self.top_k * batch_size) // self.num_experts)

        # Transpose to (num_experts, batch) so each expert can pick samples
        expert_scores = router_probs.t()  # (num_experts, batch)

        # Each expert selects top-c samples
        expert_top_values, expert_top_indices = torch.topk(
            expert_scores, k=min(capacity, batch_size), dim=-1
        )  # both (num_experts, capacity)

        # Build per-sample routing: find which experts selected each sample
        # and with what weight
        weights = torch.zeros(batch_size, self.top_k, device=x.device, dtype=torch.float32)
        expert_indices = torch.full(
            (batch_size, self.top_k), -1, device=x.device, dtype=torch.long
        )

        # Count how many experts have been assigned to each sample
        sample_count = torch.zeros(batch_size, device=x.device, dtype=torch.long)

        for expert_id in range(self.num_experts):
            for idx in range(expert_top_indices.size(1)):
                sample_id = expert_top_indices[expert_id, idx].item()
                count = sample_count[sample_id].item()
                if count < self.top_k:
                    weights[sample_id, count] = expert_top_values[expert_id, idx]
                    expert_indices[sample_id, count] = expert_id
                    sample_count[sample_id] += 1

        # Handle samples not selected by any expert: assign top-k by their router probs
        unassigned = (sample_count == 0).nonzero(as_tuple=True)[0]
        if unassigned.numel() > 0:
            unassigned_probs = router_probs[unassigned]  # (num_unassigned, num_experts)
            top_vals, top_ids = torch.topk(unassigned_probs, k=self.top_k, dim=-1)
            weights[unassigned] = top_vals
            expert_indices[unassigned] = top_ids

        # Normalize weights per sample
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights / weight_sum

        # Compute load balance statistics
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            expert_usage[e] = (expert_indices == e).any(dim=-1).float().sum()
        load_fractions = expert_usage / max(batch_size, 1)

        load_balance_stats = {
            "load_fractions": load_fractions,
            "router_probs": router_probs,
        }

        return RouterOutput(
            weights=weights,
            expert_indices=expert_indices,
            router_logits=router_logits,
            load_balance_stats=load_balance_stats,
        )


class SNRAdaptiveRouter(ExpertChoiceRouter):
    """Expert Choice router conditioned on estimated SNR.

    Extends ExpertChoiceRouter with an SNR embedding that biases routing
    decisions based on signal quality. At low SNR, the router learns to
    prefer noise-robust experts (VMD-GAF, TFMS); at high SNR, it favors
    high-resolution experts (IQ, Spectrogram).

    The SNR bias is learned during Phase 3 gating training and provides
    a principled way to adapt expert selection to operating conditions.

    Args:
        input_dim: Dimension of concatenated expert embeddings (default 3072 = 6*512).
        num_experts: Number of experts to route across (default 6).
        top_k: Number of experts activated per sample.
        snr_dim: Dimension of the SNR embedding (default 32).
        jitter_noise: Noise scale added during training for exploration.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        num_experts: int = 6,
        top_k: int = 2,
        snr_dim: int = 32,
        jitter_noise: float = 0.01,
    ):
        # Parent router operates on input_dim + snr_dim
        super().__init__(
            input_dim=input_dim + snr_dim,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
        )
        self._base_input_dim = input_dim
        self.snr_dim = snr_dim

        # Learned SNR embedding: scalar dB -> snr_dim vector
        self.snr_embedder = nn.Sequential(
            nn.Linear(1, snr_dim),
            nn.Tanh(),  # bounded output for training stability
        )

        # Learnable per-expert SNR bias
        # Initialize to prefer VMD-GAF and TFMS at low SNR
        bias_init = torch.zeros(num_experts)
        if num_experts >= 5:
            bias_init[-2] = 0.5  # VMD-GAF
        if num_experts >= 6:
            bias_init[-1] = 0.3  # TFMS
        self.snr_bias = nn.Parameter(bias_init)

    def forward(
        self,
        x: torch.Tensor,
        snr_db: torch.Tensor | None = None,
    ) -> RouterOutput:
        """Route inputs to experts with optional SNR conditioning.

        Args:
            x: Concatenated expert embeddings (batch, input_dim).
            snr_db: Optional estimated SNR in dB (batch,). When provided,
                the router biases expert selection based on signal quality.

        Returns:
            RouterOutput with weights, indices, logits, and load stats.
        """
        if snr_db is not None:
            # Normalize SNR: map [-30, 30] dB -> [-1, 1]
            snr_norm = (snr_db.float().unsqueeze(-1) / 30.0).clamp(-1, 1)
            snr_emb = self.snr_embedder(snr_norm)  # (batch, snr_dim)
            x_aug = torch.cat([x, snr_emb], dim=-1)  # (batch, input_dim + snr_dim)
        else:
            # Pad with zeros when no SNR info available
            x_aug = torch.cat(
                [x, torch.zeros(x.shape[0], self.snr_dim, device=x.device)],
                dim=-1,
            )

        # Get base routing from parent
        out = super().forward(x_aug)

        # Apply SNR-conditioned bias to router logits
        if snr_db is not None:
            # Low SNR (negative snr_norm) -> positive bias scale
            # -> amplifies preference for VMD-GAF/TFMS experts
            bias_scale = (-snr_norm).clamp(0, 1)  # (batch, 1)
            effective_bias = self.snr_bias.unsqueeze(0) * bias_scale  # (batch, num_experts)

            # Return modified output with biased logits
            biased_logits = out.router_logits + effective_bias
            return RouterOutput(
                weights=out.weights,
                expert_indices=out.expert_indices,
                router_logits=biased_logits,
                load_balance_stats=out.load_balance_stats,
            )

        return out


class TokenChoiceRouter(nn.Module):
    """Standard top-k routing where each token picks its top-k experts.

    Fallback router with the same interface as ExpertChoiceRouter.

    Args:
        input_dim: Dimension of concatenated expert embeddings (default 2048).
        num_experts: Number of experts to route across.
        top_k: Number of experts activated per sample.
        jitter_noise: Noise scale added during training for exploration.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        jitter_noise: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

        self.router = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_experts),
        )

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """Route inputs to experts via token-choice (each sample picks top-k).

        Args:
            x: Concatenated expert embeddings (batch, input_dim).

        Returns:
            RouterOutput with weights, indices, logits, and load stats.
        """
        batch_size = x.size(0)

        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(x) * self.jitter_noise
            x = x + noise

        router_logits = self.router(x)  # (batch, num_experts)

        # FP32 softmax for numerical stability
        logits_fp32 = router_logits.float()
        router_probs = F.softmax(logits_fp32, dim=-1)

        # Each token picks its top-k experts
        weights, expert_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        # Normalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Load balance statistics
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            expert_usage[e] = (expert_indices == e).any(dim=-1).float().sum()
        load_fractions = expert_usage / max(batch_size, 1)

        load_balance_stats = {
            "load_fractions": load_fractions,
            "router_probs": router_probs,
        }

        return RouterOutput(
            weights=weights,
            expert_indices=expert_indices,
            router_logits=router_logits,
            load_balance_stats=load_balance_stats,
        )
