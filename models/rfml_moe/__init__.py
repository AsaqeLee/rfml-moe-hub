"""Evaluation package for the RFML MoE drone detection system."""

from .metrics import (
    compute_classification_metrics,
    compute_hierarchical_f1,
    compute_snr_stratified_metrics,
)
from .openset import OpenMaxDetector
from .evaluator import MoEEvaluator
from .visualization import (
    plot_confusion_matrix,
    plot_snr_accuracy_curve,
    plot_expert_utilization,
    plot_roc_curves,
    plot_hierarchical_performance,
    plot_training_curves,
)

__all__ = [
    "compute_classification_metrics",
    "compute_hierarchical_f1",
    "compute_snr_stratified_metrics",
    "OpenMaxDetector",
    "MoEEvaluator",
    "plot_confusion_matrix",
    "plot_snr_accuracy_curve",
    "plot_expert_utilization",
    "plot_roc_curves",
    "plot_hierarchical_performance",
    "plot_training_curves",
]
