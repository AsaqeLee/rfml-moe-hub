"""Common helper functions for the RFML pipeline."""

import os
import random
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from .device import DeviceManager


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda") -> torch.device:
    """Get the torch device, falling back to CPU if CUDA/ROCm unavailable.

    Delegates to DeviceManager for unified CUDA/ROCm detection while
    preserving the original signature for backward compatibility.
    """
    try:
        dm = DeviceManager(device_str=device_str)
        return dm.device
    except (RuntimeError, ValueError):
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str | Path,
    scheduler=None,
    extra: Optional[dict] = None,
):
    """Save a training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        state.update(extra)

    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> dict:
    """Load a training checkpoint."""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return state


def format_size(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def setup_rocm_env(env_config: dict):
    """Set ROCm-specific environment variables for MI300X optimization.

    Applies any caller-supplied overrides, then delegates to DeviceManager
    to apply the standard ROCm/MI300X defaults for any keys not already set.
    Preserves the original signature for backward compatibility.
    """
    # Apply explicit caller overrides first (hard-set, not setdefault)
    for key, value in env_config.items():
        os.environ[key] = str(value)

    # Apply standard ROCm defaults for keys not covered by the caller
    try:
        dm = DeviceManager(device_str="rocm")
        dm.setup_environment()
    except (RuntimeError, ValueError):
        # Not running on ROCm — silently skip the default setup
        pass
