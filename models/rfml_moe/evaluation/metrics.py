"""Comprehensive metrics computation for RF signal classification evaluation."""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger("rfml.evaluation")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
) -> dict:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth integer class labels, shape (N,).
        y_pred: Predicted integer class labels, shape (N,).
        y_prob: Predicted class probabilities, shape (N, C). Optional.
        num_classes: Total number of classes (used for label ordering). Optional.

    Returns:
        dict with accuracy, balanced_accuracy, per-averaging precision/recall/f1,
        per-class precision/recall/f1, confusion_matrix, and optionally auroc and
        average_precision.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = list(range(num_classes)) if num_classes is not None else None

    metrics: dict = {}

    # Scalar accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    # Aggregated precision / recall / f1
    for avg in ("macro", "micro", "weighted"):
        metrics[f"precision_{avg}"] = float(
            precision_score(y_true, y_pred, average=avg, labels=labels, zero_division=0)
        )
        metrics[f"recall_{avg}"] = float(
            recall_score(y_true, y_pred, average=avg, labels=labels, zero_division=0)
        )
        metrics[f"f1_{avg}"] = float(
            f1_score(y_true, y_pred, average=avg, labels=labels, zero_division=0)
        )

    # Per-class metrics
    per_class_precision = precision_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    metrics["per_class_precision"] = per_class_precision.tolist()
    metrics["per_class_recall"] = per_class_recall.tolist()
    metrics["per_class_f1"] = per_class_f1.tolist()

    # Confusion matrix as numpy array (kept as ndarray, not JSON-serialised here)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)

    # Probabilistic metrics
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                metrics["auroc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
            else:
                # Binary: use positive class column
                prob_col = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["auroc"] = float(roc_auc_score(y_true, prob_col))
        except ValueError as exc:
            logger.warning("AUROC computation failed: %s", exc)
            metrics["auroc"] = float("nan")

        try:
            if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                # Macro-average of per-class AP
                ap_scores = []
                classes = labels if labels is not None else sorted(set(y_true))
                for i, cls in enumerate(classes):
                    binary_true = (y_true == cls).astype(int)
                    ap_scores.append(
                        average_precision_score(binary_true, y_prob[:, i])
                    )
                metrics["average_precision"] = float(np.mean(ap_scores))
            else:
                prob_col = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["average_precision"] = float(
                    average_precision_score(y_true, prob_col)
                )
        except ValueError as exc:
            logger.warning("Average precision computation failed: %s", exc)
            metrics["average_precision"] = float("nan")

    logger.debug(
        "Classification metrics: acc=%.4f  bal_acc=%.4f  f1_macro=%.4f",
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["f1_macro"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Hierarchical F1
# ---------------------------------------------------------------------------

def _ancestors(node: str, tree: dict) -> set:
    """Return all ancestors of *node* in the hierarchy tree (including itself).

    Args:
        node: Node identifier.
        tree: dict mapping child -> parent.  Root nodes have no entry (or None).

    Returns:
        Set of node identifiers including *node* and all ancestors up to root.
    """
    result = {node}
    current = node
    while current in tree and tree[current] is not None:
        current = tree[current]
        result.add(current)
    return result


def compute_hierarchical_f1(
    y_true_levels: list,
    y_pred_levels: list,
    hierarchy_tree: dict,
) -> dict:
    """Compute hierarchical precision, recall, and F1.

    The hierarchy tree maps each node to its parent.  Ancestor sets are used to
    compute set-based overlap between predictions and ground truth as defined in
    Kiritchenko et al. (2005).

    Also computes a tree-distance weighted error: the path length to the LCA of
    each (pred, true) pair.

    Args:
        y_true_levels: List of N per-sample ground-truth leaf labels (finest level).
        y_pred_levels: List of N per-sample predicted leaf labels (finest level).
        hierarchy_tree: dict {child: parent}.  Root nodes map to None or are absent.

    Returns:
        dict with hP, hR, hF1, and mean_lca_distance.
    """
    y_true_levels = list(y_true_levels)
    y_pred_levels = list(y_pred_levels)
    assert len(y_true_levels) == len(y_pred_levels), (
        "y_true_levels and y_pred_levels must have the same length"
    )

    # Cache ancestor sets
    _anc_cache: dict = {}

    def ancestors(node):
        if node not in _anc_cache:
            _anc_cache[node] = _ancestors(str(node), {str(k): (str(v) if v is not None else None) for k, v in hierarchy_tree.items()})
        return _anc_cache[node]

    # Build depth map for LCA distance computation
    def depth(node):
        d = 0
        current = str(node)
        str_tree = {str(k): (str(v) if v is not None else None) for k, v in hierarchy_tree.items()}
        while current in str_tree and str_tree[current] is not None:
            current = str_tree[current]
            d += 1
        return d

    total_pred_anc = 0
    total_true_anc = 0
    total_overlap = 0
    lca_distances = []

    for true_label, pred_label in zip(y_true_levels, y_pred_levels):
        true_anc = ancestors(true_label)
        pred_anc = ancestors(pred_label)

        overlap = len(true_anc & pred_anc)
        total_overlap += overlap
        total_true_anc += len(true_anc)
        total_pred_anc += len(pred_anc)

        # LCA distance: depth(true) + depth(pred) - 2*depth(LCA)
        lca_nodes = true_anc & pred_anc
        if lca_nodes:
            lca = max(lca_nodes, key=depth)
            lca_dist = depth(true_label) + depth(pred_label) - 2 * depth(lca)
        else:
            lca_dist = depth(true_label) + depth(pred_label)
        lca_distances.append(lca_dist)

    hP = total_overlap / total_pred_anc if total_pred_anc > 0 else 0.0
    hR = total_overlap / total_true_anc if total_true_anc > 0 else 0.0
    hF1 = (2 * hP * hR / (hP + hR)) if (hP + hR) > 0 else 0.0
    mean_lca_distance = float(np.mean(lca_distances)) if lca_distances else 0.0

    result = {
        "hP": float(hP),
        "hR": float(hR),
        "hF1": float(hF1),
        "mean_lca_distance": mean_lca_distance,
    }
    logger.debug("Hierarchical metrics: hP=%.4f  hR=%.4f  hF1=%.4f", hP, hR, hF1)
    return result


# ---------------------------------------------------------------------------
# SNR-stratified metrics
# ---------------------------------------------------------------------------

def compute_snr_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr_values: np.ndarray,
    snr_bins: Optional[np.ndarray] = None,
) -> dict:
    """Compute per-SNR-bin accuracy and minimum detectable SNR thresholds.

    Args:
        y_true: Ground truth integer labels, shape (N,).
        y_pred: Predicted integer labels, shape (N,).
        snr_values: Per-sample SNR in dB, shape (N,).
        snr_bins: Bin edges (left edges) in dB.  Defaults to -20 to +30 in 2 dB steps.

    Returns:
        dict with per_bin_accuracy, per_bin_balanced_accuracy, bin_centers,
        min_snr_80, min_snr_90, min_snr_95.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    snr_values = np.asarray(snr_values, dtype=float)

    if snr_bins is None:
        snr_bins = np.arange(-20, 32, 2, dtype=float)  # -20, -18, ..., 30

    per_bin_accuracy: list = []
    per_bin_balanced_accuracy: list = []
    bin_centers: list = []

    for i, bin_lo in enumerate(snr_bins):
        bin_hi = snr_bins[i + 1] if i + 1 < len(snr_bins) else bin_lo + 2.0
        mask = (snr_values >= bin_lo) & (snr_values < bin_hi)
        center = float((bin_lo + bin_hi) / 2)
        bin_centers.append(center)

        if mask.sum() == 0:
            per_bin_accuracy.append(float("nan"))
            per_bin_balanced_accuracy.append(float("nan"))
        else:
            per_bin_accuracy.append(
                float(accuracy_score(y_true[mask], y_pred[mask]))
            )
            try:
                per_bin_balanced_accuracy.append(
                    float(balanced_accuracy_score(y_true[mask], y_pred[mask]))
                )
            except ValueError:
                per_bin_balanced_accuracy.append(float("nan"))

    # Minimum detectable SNR at threshold accuracies
    bin_centers_arr = np.array(bin_centers)
    acc_arr = np.array(per_bin_accuracy)

    def min_snr_at_threshold(threshold: float) -> float:
        valid = ~np.isnan(acc_arr)
        hits = valid & (acc_arr >= threshold)
        if not hits.any():
            return float("nan")
        return float(bin_centers_arr[hits].min())

    result = {
        "bin_centers": bin_centers,
        "per_bin_accuracy": per_bin_accuracy,
        "per_bin_balanced_accuracy": per_bin_balanced_accuracy,
        "min_snr_80": min_snr_at_threshold(0.80),
        "min_snr_90": min_snr_at_threshold(0.90),
        "min_snr_95": min_snr_at_threshold(0.95),
    }
    logger.debug(
        "SNR stratified: min_snr_80=%.1f  min_snr_90=%.1f  min_snr_95=%.1f",
        result["min_snr_80"] if not np.isnan(result["min_snr_80"]) else -999,
        result["min_snr_90"] if not np.isnan(result["min_snr_90"]) else -999,
        result["min_snr_95"] if not np.isnan(result["min_snr_95"]) else -999,
    )
    return result
