"""Main MoEEvaluator class for comprehensive model evaluation."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import (
    compute_classification_metrics,
    compute_hierarchical_f1,
    compute_snr_stratified_metrics,
)
from .openset import OpenMaxDetector
from .visualization import (
    plot_confusion_matrix,
    plot_expert_utilization,
    plot_hierarchical_performance,
    plot_roc_curves,
    plot_snr_accuracy_curve,
)

logger = logging.getLogger("rfml.evaluation")


class MoEEvaluator:
    """End-to-end evaluator for the multi-modal MoE drone RF detection model.

    Args:
        model: Trained PyTorch model.  Expected to return a dict or tuple that
               exposes logits at multiple hierarchy levels and expert routing
               information.
        config: Configuration dict (loaded from default.yaml) or omegaconf DictConfig.
        device: torch device string or torch.device.
    """

    def __init__(self, model: torch.nn.Module, config: Any, device: Any = "cpu") -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)

        # Resolve hierarchy config
        try:
            self.num_classes_per_level: list[int] = [
                config.classification.hierarchy.level1_classes,
                config.classification.hierarchy.level2_classes,
                config.classification.hierarchy.level3_classes,
            ]
        except (AttributeError, KeyError):
            self.num_classes_per_level = [2, 15, 50]

        try:
            snr_cfg = config.evaluation
            self.snr_bins = np.arange(
                snr_cfg.snr_range[0],
                snr_cfg.snr_range[1] + snr_cfg.snr_step,
                snr_cfg.snr_step,
                dtype=float,
            )
        except (AttributeError, KeyError):
            self.snr_bins = np.arange(-20, 32, 2, dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inference(self, dataloader: DataLoader) -> dict:
        """Collect model outputs and targets over a dataloader.

        Expects each batch to be a dict or tuple.  When a dict, keys used:
          - ``labels_l1``, ``labels_l2``, ``labels_l3``: per-level targets
          - ``snr``: per-sample SNR in dB (optional)

        Model output is expected to be a dict with:
          - ``logits_l1``, ``logits_l2``, ``logits_l3``: per-level logits
          - ``router_output``: RouterOutput namedtuple (optional)
          - ``expert_embeddings``: raw expert activations (optional)

        Returns:
            dict with collected numpy arrays.
        """
        all_preds: dict[str, list] = {
            "l1": [], "l2": [], "l3": [],
            "true_l1": [], "true_l2": [], "true_l3": [],
            "snr": [],
            "router_weights": [],
            "router_indices": [],
            "expert_activations": [],
            "probs_l1": [], "probs_l2": [], "probs_l3": [],
        }

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else {}
                elif isinstance(batch, dict):
                    inputs = {k: v for k, v in batch.items() if k not in (
                        "labels_l1", "labels_l2", "labels_l3", "snr"
                    )}
                    targets = batch
                else:
                    inputs = batch
                    targets = {}

                # Move inputs to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                elif isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}

                # Forward pass
                outputs = self.model(inputs) if not isinstance(inputs, dict) else self.model(**inputs)

                # Extract per-level logits
                for lvl, key in enumerate(("logits_l1", "logits_l2", "logits_l3"), start=1):
                    lvl_key = f"l{lvl}"
                    if isinstance(outputs, dict) and key in outputs:
                        logits = outputs[key].float()
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                        preds = logits.argmax(dim=-1).cpu().numpy()
                        all_preds[lvl_key].append(preds)
                        all_preds[f"probs_{lvl_key}"].append(probs)

                # Collect targets
                for lvl, tkey in enumerate(("labels_l1", "labels_l2", "labels_l3"), start=1):
                    true_key = f"true_l{lvl}"
                    src = targets if isinstance(targets, dict) else {}
                    if tkey in src:
                        t = src[tkey]
                        if isinstance(t, torch.Tensor):
                            t = t.cpu().numpy()
                        all_preds[true_key].append(np.asarray(t))

                # SNR
                if isinstance(targets, dict) and "snr" in targets:
                    snr = targets["snr"]
                    if isinstance(snr, torch.Tensor):
                        snr = snr.cpu().numpy()
                    all_preds["snr"].append(np.asarray(snr, dtype=float))

                # Router outputs
                if isinstance(outputs, dict) and "router_output" in outputs:
                    ro = outputs["router_output"]
                    if hasattr(ro, "weights"):
                        all_preds["router_weights"].append(ro.weights.cpu().numpy())
                    if hasattr(ro, "expert_indices"):
                        all_preds["router_indices"].append(ro.expert_indices.cpu().numpy())

                # Expert activations for open-set
                if isinstance(outputs, dict) and "expert_activations" in outputs:
                    all_preds["expert_activations"].append(
                        outputs["expert_activations"].cpu().numpy()
                    )

        # Concatenate lists
        result: dict = {}
        for k, v in all_preds.items():
            if v:
                result[k] = np.concatenate(v, axis=0)
            else:
                result[k] = np.array([])

        return result

    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

    def evaluate(self, dataloader: DataLoader, phase: str = "test") -> dict:
        """Run full evaluation at all hierarchy levels.

        Args:
            dataloader: DataLoader yielding batches.
            phase: Evaluation phase label (used in logging/keys).

        Returns:
            dict with per-level classification metrics and hierarchical F1.
        """
        logger.info("Starting %s evaluation.", phase)
        collected = self._run_inference(dataloader)

        results: dict = {"phase": phase}

        for lvl in (1, 2, 3):
            pred_key = f"l{lvl}"
            true_key = f"true_l{lvl}"
            prob_key = f"probs_l{lvl}"

            if collected[pred_key].size == 0 or collected[true_key].size == 0:
                logger.warning("Level %d: no predictions or targets found – skipping.", lvl)
                continue

            y_true = collected[true_key].astype(int)
            y_pred = collected[pred_key].astype(int)
            y_prob = collected[prob_key] if collected[prob_key].size > 0 else None
            n_cls = self.num_classes_per_level[lvl - 1]

            metrics = compute_classification_metrics(
                y_true, y_pred, y_prob=y_prob, num_classes=n_cls
            )
            # Convert confusion matrix to list for JSON serialisation
            cm = metrics.pop("confusion_matrix")
            metrics["confusion_matrix"] = cm.tolist()
            metrics["confusion_matrix_array"] = cm  # keep ndarray for plotting

            results[f"level_{lvl}"] = metrics
            logger.info(
                "Level %d: acc=%.4f  f1_macro=%.4f",
                lvl, metrics["accuracy"], metrics["f1_macro"],
            )

        # Hierarchical F1 (if finest level available)
        if collected["true_l3"].size > 0 and collected["l3"].size > 0:
            # Derive a simple parent tree from integer labels:
            # l3 -> l2 mapping via modulo (placeholder – real mapping injected externally)
            n_l2 = self.num_classes_per_level[1]
            n_l3 = self.num_classes_per_level[2]
            tree = {f"l3_{i}": f"l2_{i * n_l2 // n_l3}" for i in range(n_l3)}
            tree.update({f"l2_{j}": f"l1_{j // (n_l2 // 2)}" for j in range(n_l2)})

            y_true_leaf = [f"l3_{int(x)}" for x in collected["true_l3"]]
            y_pred_leaf = [f"l3_{int(x)}" for x in collected["l3"]]
            h_metrics = compute_hierarchical_f1(y_true_leaf, y_pred_leaf, tree)
            results["hierarchical"] = h_metrics

        return results

    def evaluate_snr_stratified(self, dataloader: DataLoader) -> dict:
        """Evaluate accuracy as a function of SNR.

        Args:
            dataloader: DataLoader whose batches include ``snr`` field.

        Returns:
            dict with per-level SNR-stratified curves.
        """
        logger.info("Starting SNR-stratified evaluation.")
        collected = self._run_inference(dataloader)

        if collected["snr"].size == 0:
            logger.warning("No SNR values found in dataloader – returning empty result.")
            return {}

        results: dict = {}
        for lvl in (1, 2, 3):
            pred_key = f"l{lvl}"
            true_key = f"true_l{lvl}"
            if collected[pred_key].size == 0 or collected[true_key].size == 0:
                continue
            snr_metrics = compute_snr_stratified_metrics(
                collected[true_key].astype(int),
                collected[pred_key].astype(int),
                collected["snr"],
                snr_bins=self.snr_bins,
            )
            results[f"level_{lvl}"] = snr_metrics
            logger.info(
                "SNR stratified level %d: min_snr_90=%.1f dB",
                lvl, snr_metrics["min_snr_90"] if not np.isnan(snr_metrics["min_snr_90"]) else -999,
            )

        return results

    def evaluate_cross_dataset(self, dataloaders_dict: dict[str, DataLoader]) -> dict:
        """Evaluate per-dataset and aggregate cross-dataset metrics.

        Args:
            dataloaders_dict: dict mapping dataset name -> DataLoader.

        Returns:
            dict with per-dataset results and aggregated summary.
        """
        logger.info("Starting cross-dataset evaluation over %d datasets.", len(dataloaders_dict))
        results: dict = {}
        all_preds_l1: list = []
        all_true_l1: list = []

        for name, loader in dataloaders_dict.items():
            logger.info("Evaluating dataset: %s", name)
            per_ds = self.evaluate(loader, phase=name)
            results[name] = per_ds

            if "level_1" in per_ds:
                # Accumulate for aggregate metric (best-effort via accuracy scalar)
                pass

        # Simple aggregate: mean accuracy across datasets at level 1
        accs = [
            results[n]["level_1"]["accuracy"]
            for n in results
            if "level_1" in results[n]
        ]
        results["aggregate"] = {
            "mean_accuracy_l1": float(np.mean(accs)) if accs else float("nan"),
            "std_accuracy_l1": float(np.std(accs)) if accs else float("nan"),
            "datasets": list(dataloaders_dict.keys()),
        }
        return results

    def evaluate_expert_routing(self, dataloader: DataLoader) -> dict:
        """Analyse expert utilisation per class and per SNR bin.

        Args:
            dataloader: DataLoader.

        Returns:
            dict with utilization arrays and per-class/per-SNR routing statistics.
        """
        logger.info("Starting expert routing evaluation.")
        collected = self._run_inference(dataloader)

        if collected["router_indices"].size == 0:
            logger.warning("No router indices found – returning empty routing stats.")
            return {}

        router_indices = collected["router_indices"].astype(int)  # (N, top_k)
        n_samples = router_indices.shape[0]

        try:
            num_experts: int = self.config.moe.num_experts
        except (AttributeError, KeyError):
            num_experts = int(router_indices.max()) + 1

        # Overall utilisation: fraction of tokens routed to each expert
        expert_counts = np.zeros(num_experts, dtype=int)
        for e in range(num_experts):
            expert_counts[e] = (router_indices == e).any(axis=-1).sum()
        utilization = expert_counts / max(n_samples, 1)

        results: dict = {
            "overall_utilization": utilization.tolist(),
            "expert_counts": expert_counts.tolist(),
            "num_samples": n_samples,
        }

        # Per-class routing
        if collected["true_l1"].size > 0:
            true_l1 = collected["true_l1"].astype(int)
            n_cls = self.num_classes_per_level[0]
            per_class_util = np.zeros((n_cls, num_experts), dtype=float)
            for cls in range(n_cls):
                mask = true_l1 == cls
                if mask.sum() == 0:
                    continue
                cls_indices = router_indices[mask]
                for e in range(num_experts):
                    per_class_util[cls, e] = (cls_indices == e).any(axis=-1).mean()
            results["per_class_utilization"] = per_class_util.tolist()

        # Per-SNR routing
        if collected["snr"].size > 0:
            per_snr_util: dict = {}
            for i, bin_lo in enumerate(self.snr_bins):
                bin_hi = self.snr_bins[i + 1] if i + 1 < len(self.snr_bins) else bin_lo + 2.0
                mask = (collected["snr"] >= bin_lo) & (collected["snr"] < bin_hi)
                if mask.sum() == 0:
                    continue
                bin_indices = router_indices[mask]
                bin_util = np.array([
                    (bin_indices == e).any(axis=-1).mean() for e in range(num_experts)
                ])
                per_snr_util[f"{bin_lo:.0f}_{bin_hi:.0f}"] = bin_util.tolist()
            results["per_snr_utilization"] = per_snr_util

        return results

    def evaluate_open_set(
        self,
        known_loader: DataLoader,
        unknown_loader: DataLoader,
    ) -> dict:
        """Evaluate open-set detection using OpenMax.

        Args:
            known_loader: DataLoader for known-class test samples.
            unknown_loader: DataLoader for unknown / out-of-distribution samples.

        Returns:
            dict with AUROC, FPR@TPR95, openness, and per-threshold stats.
        """
        logger.info("Starting open-set evaluation.")

        try:
            tail_size = self.config.evaluation.open_set.tail_size
        except (AttributeError, KeyError):
            tail_size = 20

        num_classes = self.num_classes_per_level[2]  # finest level
        detector = OpenMaxDetector(num_classes=num_classes, tail_size=tail_size)

        # Fit on known training-like data from known_loader
        known_collected = self._run_inference(known_loader)
        if known_collected["expert_activations"].size == 0:
            logger.warning(
                "No expert_activations in known_loader – falling back to level-3 logits."
            )
            fit_vecs = known_collected["probs_l3"]
        else:
            fit_vecs = known_collected["expert_activations"]

        if fit_vecs.size == 0:
            logger.error("Cannot fit OpenMax: no activation vectors found.")
            return {"error": "no activation vectors"}

        fit_labels = known_collected["true_l3"].astype(int)
        if fit_labels.size != fit_vecs.shape[0]:
            fit_labels = np.zeros(fit_vecs.shape[0], dtype=int)

        detector.fit(fit_vecs, fit_labels, tail_size=tail_size)

        # Collect unknown activations
        unknown_collected = self._run_inference(unknown_loader)
        if unknown_collected["expert_activations"].size == 0:
            unknown_vecs = unknown_collected["probs_l3"]
        else:
            unknown_vecs = unknown_collected["expert_activations"]

        if unknown_vecs.size == 0:
            logger.error("Cannot evaluate open-set: no unknown activation vectors.")
            return {"error": "no unknown activation vectors"}

        return detector.evaluate(fit_vecs, unknown_vecs)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, results: dict, output_dir: str) -> None:
        """Save JSON results and generate evaluation plots.

        Args:
            results: Output from any evaluate_* method.
            output_dir: Directory where outputs are saved.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save JSON (strip numpy arrays)
        def _to_serialisable(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _to_serialisable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serialisable(x) for x in obj]
            return obj

        json_path = out_path / "results.json"
        with open(json_path, "w") as fh:
            json.dump(_to_serialisable(results), fh, indent=2)
        logger.info("Saved JSON results to %s", json_path)

        # ---- Confusion matrices ----
        for lvl in (1, 2, 3):
            key = f"level_{lvl}"
            if key not in results:
                continue
            lvl_data = results[key]
            if "confusion_matrix_array" in lvl_data:
                cm = np.asarray(lvl_data["confusion_matrix_array"])
            elif "confusion_matrix" in lvl_data:
                cm = np.asarray(lvl_data["confusion_matrix"])
            else:
                continue
            n = cm.shape[0]
            class_names = [str(i) for i in range(n)]
            plot_confusion_matrix(
                cm=cm,
                class_names=class_names,
                title=f"Confusion Matrix – Level {lvl}",
                save_path=str(out_path / f"confusion_matrix_l{lvl}.png"),
            )

        # ---- SNR accuracy curves ----
        if "snr_stratified" in results:
            for lvl in (1, 2, 3):
                key = f"level_{lvl}"
                if key in results["snr_stratified"]:
                    snr_data = results["snr_stratified"][key]
                    plot_snr_accuracy_curve(
                        snr_bins=snr_data["bin_centers"],
                        accuracies=snr_data["per_bin_accuracy"],
                        save_path=str(out_path / f"snr_accuracy_l{lvl}.png"),
                    )

        # ---- Expert utilisation ----
        if "routing" in results and "per_class_utilization" in results["routing"]:
            util = np.asarray(results["routing"]["per_class_utilization"])
            util_dict = {f"class_{i}": util[i].tolist() for i in range(util.shape[0])}
            plot_expert_utilization(
                utilization_dict=util_dict,
                save_path=str(out_path / "expert_utilization.png"),
            )

        # ---- Per-class F1 bar charts ----
        for lvl in (1, 2, 3):
            key = f"level_{lvl}"
            if key not in results:
                continue
            lvl_data = results[key]
            if "per_class_f1" not in lvl_data:
                continue
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                f1s = lvl_data["per_class_f1"]
                fig, ax = plt.subplots(figsize=(max(6, len(f1s) * 0.4), 5))
                ax.bar(range(len(f1s)), f1s)
                ax.set_xlabel("Class", fontsize=12)
                ax.set_ylabel("F1 Score", fontsize=12)
                ax.set_title(f"Per-class F1 – Level {lvl}", fontsize=12)
                ax.tick_params(labelsize=12)
                plt.tight_layout()
                fig.savefig(str(out_path / f"per_class_f1_l{lvl}.png"), dpi=300)
                plt.close(fig)
            except Exception as exc:
                logger.warning("Could not generate per-class F1 plot: %s", exc)

        # ---- Hierarchical performance ----
        h_metrics: dict = {}
        for lvl in (1, 2, 3):
            key = f"level_{lvl}"
            if key in results:
                h_metrics[f"Level {lvl}"] = {
                    "accuracy": results[key].get("accuracy", 0.0),
                    "f1_macro": results[key].get("f1_macro", 0.0),
                    "balanced_accuracy": results[key].get("balanced_accuracy", 0.0),
                }
        if h_metrics:
            plot_hierarchical_performance(
                metrics_per_level=h_metrics,
                save_path=str(out_path / "hierarchical_performance.png"),
            )

        # ---- ROC curves for open-set ----
        if "open_set" in results and "auroc" in results["open_set"]:
            # Placeholder: single-class open/closed ROC summary
            logger.info(
                "Open-set AUROC=%.4f – ROC curve requires raw score arrays (not stored in results dict).",
                results["open_set"]["auroc"],
            )

        logger.info("Report generation complete. Outputs in %s", out_path)
