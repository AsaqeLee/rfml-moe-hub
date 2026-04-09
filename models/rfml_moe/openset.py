"""Open-set recognition using OpenMax for RF signal classification."""

import logging
import math
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger("rfml.evaluation")


class _Weibull:
    """Minimal 2-parameter Weibull fit via MLE (shape, scale).

    Uses scipy.stats.weibull_min when available, otherwise falls back to a
    method-of-moments estimator so the class has no hard scipy dependency.
    """

    def __init__(self) -> None:
        self.shape: float = 1.0
        self.scale: float = 1.0
        self._fitted: bool = False

    def fit(self, distances: np.ndarray) -> None:
        distances = np.asarray(distances, dtype=float)
        distances = distances[distances > 0]
        if len(distances) == 0:
            logger.warning("Empty distance array for Weibull fit – using defaults.")
            return
        try:
            from scipy.stats import weibull_min  # type: ignore

            shape, _, scale = weibull_min.fit(distances, floc=0)
            self.shape = float(shape)
            self.scale = float(scale)
        except ImportError:
            # Method-of-moments fallback
            mean = float(np.mean(distances))
            std = float(np.std(distances)) + 1e-10
            # k ≈ (mean/std)^1.086
            self.shape = float((mean / std) ** 1.086)
            self.scale = float(mean / math.gamma(1.0 + 1.0 / self.shape))
        self._fitted = True

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Weibull CDF: F(x) = 1 - exp(-(x/scale)^shape)."""
        x = np.asarray(x, dtype=float)
        return 1.0 - np.exp(-((x / (self.scale + 1e-10)) ** self.shape))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        s, k = self.scale + 1e-10, self.shape
        return (k / s) * ((x / s) ** (k - 1)) * np.exp(-((x / s) ** k))


class OpenMaxDetector:
    """OpenMax open-set detector following Bendale & Boult (CVPR 2016).

    Fits per-class Weibull distributions on distances from Mean Activation
    Vectors (MAVs).  At inference, probability mass is redistributed to an
    explicit *unknown* class.

    Args:
        num_classes: Number of known classes.
        tail_size: Number of tail samples used to fit each Weibull model.
        alpha: Number of top-scoring classes whose activations are revised.
        threshold: Score threshold below which a sample is labelled *unknown*.
    """

    UNKNOWN_CLASS_ID = -1

    def __init__(
        self,
        num_classes: int,
        tail_size: int = 20,
        alpha: int = 10,
        threshold: float = 0.5,
    ) -> None:
        self.num_classes = num_classes
        self.tail_size = tail_size
        self.alpha = min(alpha, num_classes)
        self.threshold = threshold

        self._mavs: dict[int, np.ndarray] = {}          # class -> MAV
        self._weibulls: dict[int, _Weibull] = {}         # class -> Weibull model
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        activation_vectors: np.ndarray,
        labels: np.ndarray,
        tail_size: Optional[int] = None,
    ) -> None:
        """Fit per-class Weibull distributions on MAV distances.

        Args:
            activation_vectors: Pre-softmax logits / penultimate activations,
                shape (N, D).
            labels: Integer class labels, shape (N,).
            tail_size: Override instance tail_size for this fit call.
        """
        activation_vectors = np.asarray(activation_vectors, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if tail_size is not None:
            self.tail_size = tail_size

        for cls in range(self.num_classes):
            mask = labels == cls
            if mask.sum() == 0:
                logger.warning("Class %d has no training samples – skipping Weibull fit.", cls)
                continue

            class_vecs = activation_vectors[mask]
            mav = class_vecs.mean(axis=0)
            self._mavs[cls] = mav

            distances = np.linalg.norm(class_vecs - mav, axis=1)
            # Use the tail (largest distances) for extreme value fitting
            tail_dists = np.sort(distances)[-self.tail_size:]

            weibull = _Weibull()
            weibull.fit(tail_dists)
            self._weibulls[cls] = weibull

        self._fitted = True
        logger.info("OpenMax: fitted Weibull models for %d classes.", len(self._mavs))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute_openmax_prob(self, activations: np.ndarray) -> np.ndarray:
        """Redistribute softmax probability mass to the unknown class.

        Args:
            activations: Single sample pre-softmax activations, shape (D,) or (C,).
                If shape matches num_classes, treated as logits over classes.

        Returns:
            OpenMax probability vector of length num_classes + 1.  The last
            element is the *unknown* class probability.
        """
        if not self._fitted:
            raise RuntimeError("OpenMaxDetector must be fitted before calling predict.")

        activations = np.asarray(activations, dtype=float)
        # Softmax over known classes
        exp_a = np.exp(activations - activations.max())
        softmax_scores = exp_a / (exp_a.sum() + 1e-10)

        # Rank classes by softmax score (descending) – revise top-alpha
        ranked = np.argsort(softmax_scores)[::-1]
        revised = softmax_scores.copy()
        unknown_mass = 0.0

        for rank, cls in enumerate(ranked[: self.alpha]):
            if cls not in self._mavs:
                continue
            dist = float(np.linalg.norm(activations - self._mavs[cls]))
            w = float(self._weibulls[cls].cdf(np.array([dist]))[0])
            revised[cls] = softmax_scores[cls] * (1.0 - w)
            unknown_mass += softmax_scores[cls] * w

        openmax_probs = np.append(revised, unknown_mass)
        # Re-normalise
        total = openmax_probs.sum() + 1e-10
        return openmax_probs / total

    def predict(
        self, activation_vector: np.ndarray
    ) -> tuple[int, float]:
        """Predict class or unknown for a single sample.

        Args:
            activation_vector: Pre-softmax activations, shape (D,).

        Returns:
            (predicted_class_or_unknown, confidence) where predicted_class_or_unknown
            is UNKNOWN_CLASS_ID (-1) when the sample is rejected.
        """
        probs = self.compute_openmax_prob(activation_vector)
        known_probs = probs[: self.num_classes]
        unknown_prob = probs[self.num_classes]

        best_class = int(np.argmax(known_probs))
        confidence = float(known_probs[best_class])

        if unknown_prob > self.threshold:
            return self.UNKNOWN_CLASS_ID, float(unknown_prob)
        return best_class, confidence

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        known_activations: np.ndarray,
        unknown_activations: np.ndarray,
    ) -> dict:
        """Compute open-set detection metrics.

        Args:
            known_activations: Activations from known-class samples, shape (N_k, C).
            unknown_activations: Activations from unknown-class samples, shape (N_u, C).

        Returns:
            dict with auroc, fpr_at_tpr95, openness.
        """
        known_activations = np.asarray(known_activations, dtype=float)
        unknown_activations = np.asarray(unknown_activations, dtype=float)

        # Unknown probability scores: high = more likely unknown
        known_scores = np.array(
            [self.compute_openmax_prob(a)[self.num_classes] for a in known_activations]
        )
        unknown_scores = np.array(
            [self.compute_openmax_prob(a)[self.num_classes] for a in unknown_activations]
        )

        # Ground truth: 0 = known, 1 = unknown
        y_true = np.concatenate(
            [np.zeros(len(known_scores)), np.ones(len(unknown_scores))]
        )
        y_score = np.concatenate([known_scores, unknown_scores])

        try:
            auroc = float(roc_auc_score(y_true, y_score))
        except ValueError as exc:
            logger.warning("AUROC computation failed: %s", exc)
            auroc = float("nan")

        # FPR at TPR=0.95
        fpr_at_tpr95 = float("nan")
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            idx = np.searchsorted(tpr, 0.95)
            if idx < len(fpr):
                fpr_at_tpr95 = float(fpr[idx])
        except ValueError as exc:
            logger.warning("ROC curve computation failed: %s", exc)

        # Openness: O = 1 - sqrt(C_train / C_test)
        # C_train = num_classes known, C_test = known + inferred unknowns
        c_train = self.num_classes
        n_unknown_classes = max(1, len(unknown_activations) // max(1, len(known_activations) // c_train))
        c_test = c_train + n_unknown_classes
        openness = float(1.0 - math.sqrt(c_train / c_test))

        result = {
            "auroc": auroc,
            "fpr_at_tpr95": fpr_at_tpr95,
            "openness": openness,
            "num_known_samples": len(known_scores),
            "num_unknown_samples": len(unknown_scores),
        }
        logger.info(
            "Open-set evaluation: AUROC=%.4f  FPR@TPR95=%.4f  openness=%.4f",
            auroc,
            fpr_at_tpr95 if not math.isnan(fpr_at_tpr95) else -1,
            openness,
        )
        return result
