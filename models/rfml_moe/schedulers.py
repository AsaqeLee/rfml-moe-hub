"""Learning rate schedulers for progressive MoE training."""

import logging
import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LRScheduler,
    OneCycleLR,
)

logger = logging.getLogger("rfml.training")


class WarmupCosineScheduler(LRScheduler):
    """Linear warmup followed by cosine decay.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps with linear ramp.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate at end of cosine decay.
        last_epoch: Index of last epoch (for resumption).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


def create_scheduler(
    optimizer: Optimizer,
    config,
    steps_per_epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
) -> LRScheduler:
    """Factory function to create a learning rate scheduler from config.

    Args:
        optimizer: The optimizer to wrap.
        config: Config object with scheduler settings under training.scheduler.
        steps_per_epoch: Steps per epoch (needed for step-based schedulers).
        total_epochs: Total number of epochs (needed for step-based schedulers).

    Returns:
        Configured LRScheduler instance.
    """
    sched_cfg = config.get_nested("training.scheduler", {})
    sched_type = sched_cfg.get("type", "cosine_annealing_warm_restarts")

    if sched_type == "cosine_annealing_warm_restarts":
        T_0 = int(sched_cfg.get("T_0", 10))
        T_mult = int(sched_cfg.get("T_mult", 2))
        eta_min = float(sched_cfg.get("eta_min", 1e-7))
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        logger.info(
            "Scheduler: CosineAnnealingWarmRestarts T_0=%d T_mult=%d eta_min=%e",
            T_0, T_mult, eta_min,
        )

    elif sched_type == "warmup_cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
        min_lr = float(sched_cfg.get("eta_min", 1e-7))

        if steps_per_epoch is None or total_epochs is None:
            raise ValueError(
                "warmup_cosine scheduler requires steps_per_epoch and total_epochs"
            )

        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )
        logger.info(
            "Scheduler: WarmupCosine warmup=%d total=%d min_lr=%e",
            warmup_steps, total_steps, min_lr,
        )

    elif sched_type == "one_cycle":
        max_lr = float(sched_cfg.get("max_lr", 1e-3))
        if steps_per_epoch is None or total_epochs is None:
            raise ValueError(
                "one_cycle scheduler requires steps_per_epoch and total_epochs"
            )
        total_steps = total_epochs * steps_per_epoch
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=float(sched_cfg.get("pct_start", 0.3)),
            anneal_strategy=str(sched_cfg.get("anneal_strategy", "cos")),
        )
        logger.info(
            "Scheduler: OneCycleLR max_lr=%e total_steps=%d", max_lr, total_steps
        )

    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    return scheduler
