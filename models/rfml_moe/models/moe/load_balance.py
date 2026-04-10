"""Load balancing losses and monitoring for MoE routing."""

import warnings
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def router_z_loss(router_logits: torch.Tensor, coefficient: float = 0.001) -> torch.Tensor:
    """Router Z-loss to encourage smaller logits for numerical stability.

    L_z = coefficient * (1/B) * sum(log(sum(exp(h_j)))^2)

    From ST-MoE (Zoph et al., 2022).

    Args:
        router_logits: Raw router logits (batch, num_experts).
        coefficient: Loss scaling coefficient.

    Returns:
        Scalar Z-loss tensor.
    """
    # Compute in FP32 for stability
    logits = router_logits.float()
    log_sum_exp = torch.logsumexp(logits, dim=-1)  # (batch,)
    z_loss = coefficient * (log_sum_exp ** 2).mean()
    return z_loss


def load_balance_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: Optional[int] = None,
) -> torch.Tensor:
    """Auxiliary loss encouraging uniform expert utilization.

    L_balance = num_experts * sum(f_i * P_i)
    where f_i = fraction of tokens routed to expert i
    and P_i = mean routing probability for expert i

    Args:
        router_probs: Routing probabilities (batch, num_experts).
        expert_indices: Selected expert indices (batch, top_k).
        num_experts: Number of experts (inferred from router_probs if None).

    Returns:
        Scalar load balance loss tensor.
    """
    if num_experts is None:
        num_experts = router_probs.size(-1)

    batch_size = router_probs.size(0)

    # f_i: fraction of tokens dispatched to each expert
    # Create one-hot mask of expert assignments
    expert_mask = torch.zeros(
        batch_size, num_experts, device=router_probs.device, dtype=router_probs.dtype
    )
    for k in range(expert_indices.size(1)):
        valid = expert_indices[:, k] >= 0
        if valid.any():
            idx = expert_indices[:, k].clamp(min=0)
            expert_mask.scatter_add_(
                1,
                idx.unsqueeze(1),
                valid.float().unsqueeze(1).to(router_probs.dtype),
            )

    f = expert_mask.mean(dim=0)  # (num_experts,)

    # P_i: mean routing probability per expert
    p = router_probs.mean(dim=0)  # (num_experts,)

    loss = num_experts * (f * p).sum()
    return loss


def compute_expert_utilization(
    expert_indices: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Compute per-expert usage fractions.

    Args:
        expert_indices: Selected expert indices (batch, top_k).
        num_experts: Total number of experts.

    Returns:
        Tensor of shape (num_experts,) with usage fractions in [0, 1].
    """
    batch_size = expert_indices.size(0)
    if batch_size == 0:
        return torch.zeros(num_experts, device=expert_indices.device)

    usage = torch.zeros(num_experts, device=expert_indices.device)
    for e in range(num_experts):
        usage[e] = (expert_indices == e).any(dim=-1).float().sum()

    return usage / batch_size


def expert_similarity(expert_modules: list[nn.Module]) -> torch.Tensor:
    """Pairwise cosine similarity between expert weight matrices.

    Computes similarity using the first Linear layer's weight from each expert.

    Args:
        expert_modules: List of expert nn.Module instances.

    Returns:
        Tensor of shape (num_experts, num_experts) with cosine similarities.
    """
    weight_vectors = []
    for module in expert_modules:
        # Find first Linear layer and flatten its weight
        for m in module.modules():
            if isinstance(m, nn.Linear):
                weight_vectors.append(m.weight.data.flatten())
                break
        else:
            raise ValueError(f"No Linear layer found in expert {module}")

    # Stack and compute pairwise cosine similarity
    stacked = torch.stack(weight_vectors)  # (num_experts, D)
    normed = F.normalize(stacked, p=2, dim=-1)
    similarity = normed @ normed.t()  # (num_experts, num_experts)
    return similarity


class LoadBalanceMonitor:
    """Tracks expert utilization history and warns on underutilized experts.

    Args:
        num_experts: Number of experts.
        window_size: Number of recent steps to track.
        min_utilization: Threshold below which a warning is raised.
    """

    def __init__(
        self,
        num_experts: int,
        window_size: int = 100,
        min_utilization: float = 0.10,
    ):
        self.num_experts = num_experts
        self.window_size = window_size
        self.min_utilization = min_utilization
        self.history: deque[torch.Tensor] = deque(maxlen=window_size)

    def update(self, expert_indices: torch.Tensor) -> dict:
        """Record a batch of routing decisions and check utilization.

        Args:
            expert_indices: Selected expert indices (batch, top_k).

        Returns:
            Dict with 'utilization' tensor and 'warnings' list.
        """
        utilization = compute_expert_utilization(expert_indices, self.num_experts)
        self.history.append(utilization.detach().cpu())

        result: dict = {"utilization": utilization, "warnings": []}

        if len(self.history) >= 10:
            avg_util = torch.stack(list(self.history)).mean(dim=0)
            for e in range(self.num_experts):
                if avg_util[e].item() < self.min_utilization:
                    msg = (
                        f"Expert {e} underutilized: "
                        f"{avg_util[e].item():.1%} average utilization "
                        f"(threshold: {self.min_utilization:.0%})"
                    )
                    warnings.warn(msg, stacklevel=2)
                    result["warnings"].append(msg)

        return result

    def get_average_utilization(self) -> Optional[torch.Tensor]:
        """Return average utilization over the tracked window."""
        if not self.history:
            return None
        return torch.stack(list(self.history)).mean(dim=0)


def deepseek_bias_update(
    bias_terms: torch.Tensor,
    expert_loads: torch.Tensor,
    target_load: float,
    epsilon: float = 0.001,
) -> torch.Tensor:
    """DeepSeek-V3 auxiliary-loss-free bias update rule.

    Adjusts per-expert bias terms to steer load toward the target. Experts
    receiving more than the target load have their bias decreased, and
    under-utilized experts have their bias increased.

    Args:
        bias_terms: Current bias terms, shape (num_experts,).
        expert_loads: Observed per-expert load fractions, shape (num_experts,).
        target_load: Desired load fraction per expert (typically 1/num_experts).
        epsilon: Step size for bias adjustment.

    Returns:
        Updated bias terms tensor (same shape as input).
    """
    # Increase bias for under-loaded experts, decrease for over-loaded
    updated = bias_terms.clone()
    over = expert_loads > target_load
    under = expert_loads < target_load
    updated[over] -= epsilon
    updated[under] += epsilon
    return updated
