"""Advanced MoE routing algorithms: DeepSeek-V3, Soft MoE, and Adaptive routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .router import RouterOutput, ExpertChoiceRouter, TokenChoiceRouter
from .load_balance import deepseek_bias_update


class DeepSeekRouter(nn.Module):
    """Auxiliary-loss-free load balancing router from DeepSeek-V3 (arXiv:2412.19437).

    Uses dynamic per-expert bias terms to balance load without auxiliary losses.
    Each step, bias terms are adjusted based on whether each expert is over- or
    under-utilized relative to the target load (1/num_experts).

    Args:
        input_dim: Dimension of input embeddings.
        num_experts: Number of experts to route across.
        top_k: Number of experts activated per sample.
        epsilon: Bias update rate per step.
        jitter_noise: Noise scale added during training for exploration.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        epsilon: float = 0.001,
        jitter_noise: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.epsilon = epsilon
        self.jitter_noise = jitter_noise

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_experts),
        )

        # Dynamic bias terms: one learnable scalar per expert
        self.bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)
        self.target_load = 1.0 / num_experts

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """Route inputs using DeepSeek-V3 bias-adjusted gating.

        Args:
            x: Input embeddings (batch, input_dim).

        Returns:
            RouterOutput with weights, indices, logits, and load stats.
        """
        batch_size = x.size(0)

        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(x) * self.jitter_noise
            x = x + noise

        # Compute router logits
        router_logits = self.gate(x)  # (batch, num_experts)

        # Keep softmax in FP32
        logits_fp32 = router_logits.float()
        router_probs = F.softmax(logits_fp32 + self.bias.float(), dim=-1)

        # Top-k selection from bias-adjusted probabilities
        weights, expert_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        # Normalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Compute per-expert load fractions
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            expert_usage[e] = (expert_indices == e).any(dim=-1).float().sum()
        load_fractions = expert_usage / max(batch_size, 1)

        # Update bias terms based on load (no auxiliary loss needed)
        if self.training:
            self.bias.data = deepseek_bias_update(
                self.bias.data, load_fractions, self.target_load, self.epsilon
            )

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


class SoftMoERouter(nn.Module):
    """Fully differentiable Soft MoE routing (ICLR 2024).

    Instead of discrete token-to-expert assignment, each expert slot receives
    a soft weighted combination of ALL inputs. No token dropping, no load
    imbalance, and fully differentiable.

    Args:
        input_dim: Dimension of input embeddings.
        num_experts: Number of experts.
        num_slots: Number of slots per expert. If None, determined at forward time
            as batch_size // num_experts (minimum 1).
        top_k: Number of experts per sample (used for RouterOutput compatibility).
    """

    def __init__(
        self,
        input_dim: int = 2048,
        num_experts: int = 4,
        num_slots: int | None = None,
        top_k: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.top_k = top_k

        # Phi will be lazily initialized if num_slots is None
        if num_slots is not None:
            total_slots = num_experts * num_slots
            self.phi = nn.Parameter(torch.randn(input_dim, total_slots) * 0.02)
        else:
            self.phi = None

        # Small linear to produce router_logits for RouterOutput compatibility
        self.logit_proj = nn.Linear(input_dim, num_experts)

    def _get_phi(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get or lazily create the Phi parameter."""
        if self.phi is not None:
            return self.phi

        # Dynamic: num_slots = batch_size // num_experts, minimum 1
        num_slots = max(1, batch_size // self.num_experts)
        total_slots = self.num_experts * num_slots

        # Create and register parameter on first use
        self.phi = nn.Parameter(
            torch.randn(self.input_dim, total_slots, device=device) * 0.02
        )
        return self.phi

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """Route inputs using soft dispatch and combine weights.

        Args:
            x: Input embeddings (batch, input_dim).

        Returns:
            RouterOutput with soft weights (not discrete indices).
            - weights: (batch, top_k) top-k combine weights per sample
            - expert_indices: (batch, top_k) experts with highest combine weight
            - router_logits: (batch, num_experts) logits for compatibility
            - load_balance_stats includes dispatch_weights and combine_weights
        """
        batch_size = x.size(0)
        phi = self._get_phi(batch_size, x.device)
        total_slots = phi.size(1)

        # Compute logit matrix: (batch, total_slots)
        logit_matrix = x.float() @ phi.float()

        # Dispatch weights: softmax over tokens (dim=0)
        # D[i,s] = how much token i contributes to slot s
        dispatch_weights = F.softmax(logit_matrix, dim=0)  # (batch, total_slots)

        # Combine weights: softmax over slots (dim=1)
        # C[i,s] = how much slot s contributes to reconstructing token i's output
        combine_weights = F.softmax(logit_matrix, dim=1)  # (batch, total_slots)

        # For RouterOutput compatibility, compute per-expert soft weights
        # Reshape combine weights to (batch, num_experts, slots_per_expert)
        slots_per_expert = total_slots // self.num_experts
        combine_per_expert = combine_weights.view(
            batch_size, self.num_experts, slots_per_expert
        )

        # Sum combine weight across slots for each expert
        expert_weights = combine_per_expert.sum(dim=-1)  # (batch, num_experts)

        # Pick top-k experts by combine weight
        top_weights, top_indices = torch.topk(
            expert_weights, k=min(self.top_k, self.num_experts), dim=-1
        )

        # Normalize
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Router logits for compatibility
        router_logits = self.logit_proj(x)

        # Load fractions: uniform by construction in Soft MoE
        load_fractions = expert_weights.mean(dim=0)
        load_fractions = load_fractions / load_fractions.sum().clamp(min=1e-8)

        load_balance_stats = {
            "load_fractions": load_fractions,
            "router_probs": F.softmax(router_logits.float(), dim=-1),
            "dispatch_weights": dispatch_weights,
            "combine_weights": combine_weights,
        }

        return RouterOutput(
            weights=top_weights.to(x.dtype),
            expert_indices=top_indices,
            router_logits=router_logits,
            load_balance_stats=load_balance_stats,
        )


class AdaptiveRouter(nn.Module):
    """Meta-router that delegates to a selected routing strategy.

    Provides a unified interface for switching between routing algorithms.

    Args:
        input_dim: Dimension of input embeddings.
        num_experts: Number of experts.
        top_k: Number of experts activated per sample.
        strategy: One of "expert_choice", "token_choice", "deepseek", "soft_moe".
        **kwargs: Additional keyword arguments passed to the selected router.
    """

    STRATEGIES = ("expert_choice", "token_choice", "deepseek", "soft_moe")

    def __init__(
        self,
        input_dim: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        strategy: str = "expert_choice",
        **kwargs,
    ):
        super().__init__()
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Must be one of {self.STRATEGIES}"
            )
        self.strategy = strategy

        if strategy == "expert_choice":
            self.router = ExpertChoiceRouter(
                input_dim=input_dim, num_experts=num_experts, top_k=top_k, **kwargs
            )
        elif strategy == "token_choice":
            self.router = TokenChoiceRouter(
                input_dim=input_dim, num_experts=num_experts, top_k=top_k, **kwargs
            )
        elif strategy == "deepseek":
            self.router = DeepSeekRouter(
                input_dim=input_dim, num_experts=num_experts, top_k=top_k, **kwargs
            )
        elif strategy == "soft_moe":
            self.router = SoftMoERouter(
                input_dim=input_dim, num_experts=num_experts, top_k=top_k, **kwargs
            )

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """Route inputs using the selected strategy.

        Args:
            x: Input embeddings (batch, input_dim).

        Returns:
            RouterOutput from the selected router.
        """
        return self.router(x)
