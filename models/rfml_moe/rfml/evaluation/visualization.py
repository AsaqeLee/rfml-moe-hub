"""Publication-quality plotting utilities for RF-ML evaluation results."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("rfml.evaluation")

# ---------------------------------------------------------------------------
# Shared matplotlib setup helper
# ---------------------------------------------------------------------------

def _get_mpl():
    """Return (matplotlib, pyplot) with Agg backend set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


_FONT_SIZE = 12
_DPI = 300


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot an annotated confusion matrix heatmap.

    Args:
        cm: Confusion matrix of shape (C, C).
        class_names: List of C class name strings.
        title: Figure title.
        save_path: File path to save the figure (.png recommended).
    """
    mpl, plt = _get_mpl()

    cm = np.asarray(cm)
    n = cm.shape[0]
    # Normalised version for colour mapping
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig_size = max(6, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=_FONT_SIZE + 2)
    ax.set_xlabel("Predicted label", fontsize=_FONT_SIZE)
    ax.set_ylabel("True label", fontsize=_FONT_SIZE)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    if n <= 30:
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=max(6, _FONT_SIZE - n // 5))
        ax.set_yticklabels(class_names, fontsize=max(6, _FONT_SIZE - n // 5))
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Annotate cells only when matrix is small enough to be readable
    if n <= 20:
        thresh = cm_norm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=max(6, _FONT_SIZE - 2),
                )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved confusion matrix to %s", save_path)
    plt.close(fig)


def plot_snr_accuracy_curve(
    snr_bins: list,
    accuracies: list,
    save_path: Optional[str] = None,
    thresholds: Optional[list[float]] = None,
) -> None:
    """Plot accuracy vs SNR with optional threshold markers.

    Args:
        snr_bins: List of SNR bin centre values in dB.
        accuracies: Per-bin accuracy values (may contain NaN for empty bins).
        save_path: File path for saving.
        thresholds: Horizontal threshold lines to draw (default: 0.80, 0.90, 0.95).
    """
    mpl, plt = _get_mpl()

    if thresholds is None:
        thresholds = [0.80, 0.90, 0.95]

    snr_arr = np.asarray(snr_bins, dtype=float)
    acc_arr = np.asarray(accuracies, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snr_arr, acc_arr, marker="o", linewidth=1.5, markersize=4, label="Accuracy")

    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    labels = ["80% threshold", "90% threshold", "95% threshold"]
    for thr, color, lbl in zip(thresholds, colors, labels):
        ax.axhline(thr, linestyle="--", linewidth=1.0, color=color, alpha=0.8, label=lbl)

    ax.set_xlabel("SNR (dB)", fontsize=_FONT_SIZE)
    ax.set_ylabel("Accuracy", fontsize=_FONT_SIZE)
    ax.set_title("Accuracy vs SNR", fontsize=_FONT_SIZE + 2)
    ax.set_xlim(snr_arr.min() - 1, snr_arr.max() + 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=_FONT_SIZE - 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=_FONT_SIZE)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved SNR accuracy curve to %s", save_path)
    plt.close(fig)


def plot_expert_utilization(
    utilization_dict: dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot a stacked bar chart of expert utilisation per class.

    Args:
        utilization_dict: dict mapping class name -> list of per-expert utilisation
            fractions summing to ~1.
        save_path: File path for saving.
    """
    mpl, plt = _get_mpl()

    class_names = list(utilization_dict.keys())
    data = np.array([utilization_dict[c] for c in class_names], dtype=float)
    n_classes, n_experts = data.shape

    x = np.arange(n_classes)
    width = 0.6

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_experts)]

    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.4), 5))
    bottom = np.zeros(n_classes)
    for e in range(n_experts):
        ax.bar(x, data[:, e], width, bottom=bottom, color=colors[e], label=f"Expert {e}")
        bottom += data[:, e]

    ax.set_xlabel("Class", fontsize=_FONT_SIZE)
    ax.set_ylabel("Utilisation fraction", fontsize=_FONT_SIZE)
    ax.set_title("Expert Utilisation per Class", fontsize=_FONT_SIZE + 2)
    ax.set_xticks(x)
    if n_classes <= 30:
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=max(6, _FONT_SIZE - n_classes // 5))
    else:
        ax.set_xticklabels([])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=_FONT_SIZE - 1, ncol=min(n_experts, 4))
    ax.tick_params(labelsize=_FONT_SIZE)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved expert utilization chart to %s", save_path)
    plt.close(fig)


def plot_roc_curves(
    fpr_dict: dict,
    tpr_dict: dict,
    auc_dict: dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot multi-class ROC curves on a single axes.

    Args:
        fpr_dict: dict mapping class/split name -> FPR array.
        tpr_dict: dict mapping class/split name -> TPR array.
        auc_dict: dict mapping class/split name -> AUROC scalar.
        save_path: File path for saving.
    """
    mpl, plt = _get_mpl()

    fig, ax = plt.subplots(figsize=(7, 6))

    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(fpr_dict.keys()):
        fpr = np.asarray(fpr_dict[name])
        tpr = np.asarray(tpr_dict[name])
        auc_val = float(auc_dict.get(name, float("nan")))
        label = f"{name} (AUC={auc_val:.3f})" if not np.isnan(auc_val) else name
        ax.plot(fpr, tpr, linewidth=1.5, color=cmap(i % 10), label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=_FONT_SIZE)
    ax.set_ylabel("True Positive Rate", fontsize=_FONT_SIZE)
    ax.set_title("ROC Curves", fontsize=_FONT_SIZE + 2)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.legend(loc="lower right", fontsize=_FONT_SIZE - 2, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=_FONT_SIZE)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved ROC curves to %s", save_path)
    plt.close(fig)


def plot_hierarchical_performance(
    metrics_per_level: dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot grouped bar chart comparing metrics across hierarchy levels.

    Args:
        metrics_per_level: dict mapping level label -> dict of metric name -> value.
            Example: {"Level 1": {"accuracy": 0.97, "f1_macro": 0.96}, ...}
        save_path: File path for saving.
    """
    mpl, plt = _get_mpl()

    levels = list(metrics_per_level.keys())
    # Collect union of metric names
    metric_names: list[str] = []
    for v in metrics_per_level.values():
        for k in v:
            if k not in metric_names:
                metric_names.append(k)

    n_levels = len(levels)
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    total_width = 0.7
    bar_w = total_width / n_levels

    cmap = plt.get_cmap("Set2")
    fig, ax = plt.subplots(figsize=(max(7, n_metrics * 1.5), 5))

    for i, level in enumerate(levels):
        vals = [float(metrics_per_level[level].get(m, 0.0)) for m in metric_names]
        offset = (i - n_levels / 2.0 + 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w, label=level, color=cmap(i % 8))

    ax.set_xlabel("Metric", fontsize=_FONT_SIZE)
    ax.set_ylabel("Score", fontsize=_FONT_SIZE)
    ax.set_title("Performance Across Hierarchy Levels", fontsize=_FONT_SIZE + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20, ha="right", fontsize=_FONT_SIZE)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=_FONT_SIZE - 1)
    ax.tick_params(labelsize=_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved hierarchical performance plot to %s", save_path)
    plt.close(fig)


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_metrics: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves, with optional metric overlay.

    Args:
        train_losses: Per-epoch training loss values.
        val_losses: Per-epoch validation loss values.
        val_metrics: Optional dict mapping metric name -> list of per-epoch values.
            Plotted on a secondary y-axis.
        save_path: File path for saving.
    """
    mpl, plt = _get_mpl()

    epochs = np.arange(1, len(train_losses) + 1)
    has_metrics = bool(val_metrics)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(epochs, train_losses, linewidth=1.5, label="Train loss", color="#2980b9")
    ax1.plot(epochs, val_losses, linewidth=1.5, linestyle="--", label="Val loss", color="#e74c3c")
    ax1.set_xlabel("Epoch", fontsize=_FONT_SIZE)
    ax1.set_ylabel("Loss", fontsize=_FONT_SIZE, color="#2c3e50")
    ax1.tick_params(labelsize=_FONT_SIZE)

    lines, labels = ax1.get_legend_handles_labels()

    if has_metrics:
        ax2 = ax1.twinx()
        cmap = plt.get_cmap("tab10")
        for i, (metric_name, metric_vals) in enumerate(val_metrics.items()):
            ep_m = np.arange(1, len(metric_vals) + 1)
            line, = ax2.plot(
                ep_m, metric_vals,
                linewidth=1.5,
                linestyle=":",
                color=cmap((i + 2) % 10),
                label=metric_name,
            )
            lines.append(line)
            labels.append(metric_name)
        ax2.set_ylabel("Metric score", fontsize=_FONT_SIZE)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(labelsize=_FONT_SIZE)

    ax1.set_title("Training Curves", fontsize=_FONT_SIZE + 2)
    ax1.legend(lines, labels, loc="upper right", fontsize=_FONT_SIZE - 1)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved training curves to %s", save_path)
    plt.close(fig)
