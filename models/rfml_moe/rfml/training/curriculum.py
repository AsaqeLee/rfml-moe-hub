"""Curriculum learning managers for progressive MoE training.

Provides SNR-based sample filtering and hierarchical loss weight scheduling
to gradually increase task difficulty during training.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("rfml.training")


class SNRCurriculum:
    """Controls which samples are included based on SNR and current epoch.

    Training starts with high-SNR (easy) samples and progressively introduces
    lower-SNR (harder) samples via a linearly decreasing threshold.

    Args:
        start_snr: Initial SNR threshold in dB (only samples above this).
        end_snr: Final SNR threshold in dB (all samples above this by end).
        total_epochs: Number of epochs over which the transition occurs.
    """

    def __init__(
        self,
        start_snr: float = 20.0,
        end_snr: float = -10.0,
        total_epochs: int = 75,
    ):
        self.start_snr = start_snr
        self.end_snr = end_snr
        self.total_epochs = total_epochs
        logger.info(
            "SNRCurriculum: %+.0f dB -> %+.0f dB over %d epochs",
            start_snr, end_snr, total_epochs,
        )

    def get_snr_threshold(self, epoch: int, total_epochs: Optional[int] = None) -> float:
        """Return the current minimum SNR threshold for the given epoch.

        Args:
            epoch: Current epoch (0-indexed).
            total_epochs: Override for total epoch count. Uses init value if None.

        Returns:
            SNR threshold in dB. Samples below this should be excluded.
        """
        n = total_epochs if total_epochs is not None else self.total_epochs
        progress = min(epoch / max(1, n - 1), 1.0)
        threshold = self.start_snr + progress * (self.end_snr - self.start_snr)
        return threshold

    def filter_batch(
        self, batch: Dict[str, torch.Tensor], threshold: float
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Remove samples from a batch whose SNR falls below the threshold.

        Args:
            batch: Dict containing at least "snr_db" key with shape (B,).
            threshold: Current SNR threshold in dB.

        Returns:
            Filtered batch dict, or None if all samples were removed.
        """
        snr = batch["snr_db"]
        mask = snr >= threshold

        if not mask.any():
            return None

        if mask.all():
            return batch

        filtered = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                filtered[key] = val[mask]
            else:
                filtered[key] = val
        return filtered


class HierarchyScheduler:
    """Controls loss weights across classification hierarchy levels.

    Smoothly transitions from emphasizing coarse-grained (binary drone/no-drone)
    to fine-grained (make/model/protocol) classification via cosine interpolation.

    Args:
        early_weights: Weight vector for early training [level1, level2, level3].
        late_weights: Weight vector for late training [level1, level2, level3].
        transition_epoch: Epoch at which the transition is centered.
        transition_width: Number of epochs over which the transition occurs.
            The cosine transition spans [transition_epoch - width/2, transition_epoch + width/2].
    """

    def __init__(
        self,
        early_weights: Optional[List[float]] = None,
        late_weights: Optional[List[float]] = None,
        transition_epoch: int = 100,
        transition_width: int = 40,
    ):
        self.early_weights = early_weights or [0.5, 0.3, 0.2]
        self.late_weights = late_weights or [0.1, 0.2, 0.7]
        self.transition_epoch = transition_epoch
        self.transition_width = transition_width

        logger.info(
            "HierarchyScheduler: %s -> %s, transition at epoch %d (width %d)",
            self.early_weights,
            self.late_weights,
            transition_epoch,
            transition_width,
        )

    def get_weights(self, epoch: int) -> List[float]:
        """Return the current loss weight vector for the given epoch.

        Uses a smooth cosine transition centered at transition_epoch.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            List of 3 floats: [weight_binary, weight_type, weight_full].
        """
        half_width = self.transition_width / 2.0
        start = self.transition_epoch - half_width
        end = self.transition_epoch + half_width

        if epoch <= start:
            alpha = 0.0
        elif epoch >= end:
            alpha = 1.0
        else:
            # Smooth cosine transition in [0, 1]
            progress = (epoch - start) / max(1.0, end - start)
            alpha = 0.5 * (1.0 - math.cos(math.pi * progress))

        weights = [
            early * (1.0 - alpha) + late * alpha
            for early, late in zip(self.early_weights, self.late_weights)
        ]
        return weights
