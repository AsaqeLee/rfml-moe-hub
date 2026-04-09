"""Hierarchical classification losses for MoE drone RF detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moe.load_balance import load_balance_loss, router_z_loss


class HierarchicalLoss(nn.Module):
    """Combined hierarchical classification loss with MoE auxiliary losses.

    Combines cross-entropy losses at three hierarchy levels with smoothly
    transitioning weights, plus router Z-loss and load balance loss.

    Loss schedule:
        - Early training (epoch < transition_epoch): emphasize coarse levels
          weights = early_weights = [0.5, 0.3, 0.2]
        - Late training (epoch >= transition_epoch): emphasize fine levels
          weights = late_weights = [0.1, 0.2, 0.7]
        - Smooth cosine transition between the two over transition_epochs window.

    Args:
        early_weights: Loss weights for [level1, level2, level3] early in training.
        late_weights: Loss weights for [level1, level2, level3] late in training.
        transition_epoch: Epoch at which transition is centered.
        transition_width: Number of epochs over which the transition occurs.
        z_loss_coefficient: Coefficient for router Z-loss.
        balance_loss_coefficient: Coefficient for load balance loss.
        label_smoothing: Label smoothing for cross-entropy losses.
    """

    def __init__(
        self,
        early_weights: tuple[float, ...] = (0.5, 0.3, 0.2),
        late_weights: tuple[float, ...] = (0.1, 0.2, 0.7),
        transition_epoch: int = 50,
        transition_width: int = 20,
        z_loss_coefficient: float = 0.001,
        balance_loss_coefficient: float = 0.01,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.register_buffer(
            "early_weights", torch.tensor(early_weights, dtype=torch.float32)
        )
        self.register_buffer(
            "late_weights", torch.tensor(late_weights, dtype=torch.float32)
        )
        self.transition_epoch = transition_epoch
        self.transition_width = transition_width
        self.z_loss_coefficient = z_loss_coefficient
        self.balance_loss_coefficient = balance_loss_coefficient
        self.label_smoothing = label_smoothing

    def get_level_weights(self, epoch: int) -> torch.Tensor:
        """Compute smoothly interpolated level weights for given epoch.

        Uses cosine interpolation centered at transition_epoch.

        Args:
            epoch: Current training epoch.

        Returns:
            Tensor of shape (3,) with level weights summing to ~1.
        """
        half_width = self.transition_width / 2
        start = self.transition_epoch - half_width
        end = self.transition_epoch + half_width

        if epoch <= start:
            alpha = 0.0
        elif epoch >= end:
            alpha = 1.0
        else:
            # Cosine interpolation
            progress = (epoch - start) / self.transition_width
            alpha = 0.5 * (1 - torch.cos(torch.tensor(progress * torch.pi)).item())

        return (1 - alpha) * self.early_weights + alpha * self.late_weights

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined hierarchical + MoE auxiliary losses.

        Args:
            predictions: Dict with keys "level1", "level2", "level3" (logits),
                "router_logits", and optionally "router_probs"/"expert_indices".
            targets: Dict with keys "level1", "level2", "level3" (class indices).
            epoch: Current training epoch for weight scheduling.

        Returns:
            Tuple of (total_loss, component_dict) where component_dict has
            individual loss values for logging.
        """
        level_weights = self.get_level_weights(epoch)
        components: dict[str, torch.Tensor] = {}

        # Classification losses at each hierarchy level
        total_cls_loss = torch.tensor(0.0, device=predictions["level1"].device)
        for i, level in enumerate(["level1", "level2", "level3"]):
            if level in targets and targets[level] is not None:
                cls_loss = F.cross_entropy(
                    predictions[level],
                    targets[level],
                    label_smoothing=self.label_smoothing,
                )
                components[f"{level}_loss"] = cls_loss
                total_cls_loss = total_cls_loss + level_weights[i] * cls_loss
            else:
                components[f"{level}_loss"] = torch.tensor(0.0, device=total_cls_loss.device)

        components["classification_loss"] = total_cls_loss

        # Router Z-loss
        z_loss = torch.tensor(0.0, device=total_cls_loss.device)
        if "router_logits" in predictions and predictions["router_logits"] is not None:
            z_loss = router_z_loss(
                predictions["router_logits"],
                coefficient=self.z_loss_coefficient,
            )
        components["z_loss"] = z_loss

        # Load balance loss
        bal_loss = torch.tensor(0.0, device=total_cls_loss.device)
        if (
            "router_probs" in predictions
            and "expert_indices" in predictions
            and predictions["router_probs"] is not None
        ):
            bal_loss = self.balance_loss_coefficient * load_balance_loss(
                predictions["router_probs"],
                predictions["expert_indices"],
            )
        components["balance_loss"] = bal_loss

        total_loss = total_cls_loss + z_loss + bal_loss
        components["total_loss"] = total_loss
        components["level_weights"] = level_weights

        return total_loss, components
