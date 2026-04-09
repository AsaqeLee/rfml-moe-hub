"""Multi-modal fusion modules."""

from .cross_attention import CrossAttentionFusion, ConfidenceWeightedFusion

__all__ = ["CrossAttentionFusion", "ConfidenceWeightedFusion"]
