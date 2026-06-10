"""Main MoE Trainer implementing 4-phase progressive training.

Phase 1: Self-supervised pretraining (MAE + contrastive) per expert
Phase 2: Supervised curriculum training per expert (SNR-based)
Phase 3: Gating network training (experts frozen)
Phase 4: End-to-end fine-tuning
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.dataset import RFDataset, create_dataloaders
from training.curriculum import HierarchyScheduler, SNRCurriculum
from training.impairments import SignalAugmentationPipeline
from training.pretraining import ContrastiveLearning, MaskedAutoencoder
from training.schedulers import WarmupCosineScheduler, create_scheduler
from utils.helpers import save_checkpoint

logger = logging.getLogger("rfml.training")


class MoETrainer:
    """Orchestrates 4-phase progressive training for the MoE drone detector.

    Args:
        model: The full MoE model (expects freeze/unfreeze on experts,
               gating network accessible via model.gate / model.shared_expert).
        config: Config object with training, data, and classification settings.
        device: Torch device for training.
    """

    def __init__(self, model: nn.Module, config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        # Training hyperparameters from config
        train_cfg = config.get_nested("training", {})
        opt_cfg = config.get_nested("training.optimizer", {})

        self.grad_clip = float(opt_cfg.get("grad_clip", 1.0))
        self.accumulation_steps = int(train_cfg.get("accumulation_steps", 4))
        self.batch_size = int(train_cfg.get("batch_size", 128))

        # Checkpoint settings
        ckpt_cfg = config.get_nested("training.checkpointing", {})
        self.save_every = int(ckpt_cfg.get("save_every_epochs", 5))
        self.keep_top_k = int(ckpt_cfg.get("keep_top_k", 3))
        self.save_dir = Path(ckpt_cfg.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Tracked checkpoints: list of (val_loss, path) sorted ascending
        self._top_checkpoints: List[tuple] = []

        # Optimizer (AdamW)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("phases", {}).get("pretrain", {}).get("lr", 1e-3)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # Scheduler
        self.scheduler = create_scheduler(self.optimizer, config)

        # GradScaler for mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        # Curriculum managers
        sup_cfg = config.get_nested("training.phases.supervised", {})
        self.snr_curriculum = SNRCurriculum(
            start_snr=float(sup_cfg.get("snr_start_db", 20)),
            end_snr=float(sup_cfg.get("snr_end_db", -10)),
            total_epochs=int(sup_cfg.get("epochs", 75)),
        )

        cls_cfg = config.get_nested("classification.loss_weights", {})
        self.hierarchy_scheduler = HierarchyScheduler(
            early_weights=cls_cfg.get("early", [0.5, 0.3, 0.2]),
            late_weights=cls_cfg.get("late", [0.1, 0.2, 0.7]),
            transition_epoch=int(cls_cfg.get("transition_epoch", 100)),
        )

        # Sim-to-Real Augmentation
        aug_cfg = train_cfg.get("augmentation", {})
        sampling_rate = float(config.get_nested("data.sampling_rate", 100e6))
        cfo_limit_hz = float(aug_cfg.get("cfo_max_hz", 1000))
        
        self.augmentation = SignalAugmentationPipeline(
            cfo_limit=cfo_limit_hz / sampling_rate,
            iq_gain_limit=float(aug_cfg.get("iq_gain_db", 1.0)),
            iq_phase_limit=float(aug_cfg.get("iq_phase_deg", 5.0))
        ).to(device)

        # Loss functions for hierarchical classification
        self.criterion_binary = nn.CrossEntropyLoss()
        self.criterion_type = nn.CrossEntropyLoss()
        self.criterion_full = nn.CrossEntropyLoss()

        # Optional torch.compile
        self._compiled = False
        if config.get_nested("training.compile", False):
            try:
                self.model = torch.compile(model, mode="max-autotune")
                self._compiled = True
                logger.info("Model compiled with torch.compile(mode='max-autotune')")
            except Exception as e:
                logger.warning("torch.compile failed, falling back to eager: %s", e)

        self.model.to(device)
        logger.info(
            "MoETrainer initialized: device=%s, accumulation=%d, grad_clip=%.1f",
            device, self.accumulation_steps, self.grad_clip,
        )

    # ------------------------------------------------------------------
    # Phase 1: Self-supervised pretraining
    # ------------------------------------------------------------------

    def pretrain_experts(self, train_loader: DataLoader):
        """Phase 1: Self-supervised pretraining per expert.

        - IQ expert: Masked autoencoder (75% masking)
        - Spectrogram expert: Contrastive learning (MoCo-v3)
        - HOS/Cyclo experts: Skip (tabular/1D data, less benefit from SSL)

        Uses 30% subset of training data for efficiency.
        """
        phase_cfg = self.config.get_nested("training.phases.pretrain", {})
        epochs = int(phase_cfg.get("epochs", 75))
        lr = float(phase_cfg.get("lr", 1e-3))
        subset_fraction = float(phase_cfg.get("subset_fraction", 0.3))

        logger.info("=" * 60)
        logger.info("PHASE 1: Self-supervised pretraining (%d epochs)", epochs)
        logger.info("=" * 60)

        # Create subset
        dataset = train_loader.dataset
        subset_size = max(1, int(len(dataset) * subset_fraction))
        indices = torch.randperm(len(dataset))[:subset_size].tolist()
        subset = Subset(dataset, indices)
        subset_loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        # --- IQ Expert: Masked Autoencoder ---
        iq_expert = getattr(self.model, "experts", {}).get("iq", None) if hasattr(self.model, "experts") else getattr(self.model, "iq_expert", None)
        if iq_expert is not None:
            logger.info("Pretraining IQ expert with MAE...")
            mae = MaskedAutoencoder(mode="iq", patch_size=256).to(self.device)
            mae_optimizer = torch.optim.AdamW(
                list(iq_expert.parameters()) + list(mae.parameters()),
                lr=lr, weight_decay=0.01,
            )

            for epoch in range(epochs):
                mae.train()
                iq_expert.train()
                epoch_loss = 0.0
                num_batches = 0

                for batch in subset_loader:
                    iq = batch["iq"].to(self.device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss, _ = mae(iq)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(mae_optimizer)
                    self.scaler.update()
                    mae_optimizer.zero_grad(set_to_none=True)

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / max(1, num_batches)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        "  [MAE] Epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss
                    )

            del mae, mae_optimizer

        # --- Spectrogram Expert: Contrastive Learning ---
        spec_expert = getattr(self.model, "experts", {}).get("spectrogram", None) if hasattr(self.model, "experts") else getattr(self.model, "spectrogram_expert", None)
        if spec_expert is not None:
            logger.info("Pretraining spectrogram expert with contrastive learning...")
            contrastive = ContrastiveLearning(
                encoder=spec_expert,
                embed_dim=512,
            ).to(self.device)

            cl_params = [
                p for p in contrastive.parameters() if p.requires_grad
            ]
            cl_optimizer = torch.optim.AdamW(cl_params, lr=lr, weight_decay=0.01)

            for epoch in range(epochs):
                contrastive.train()
                epoch_loss = 0.0
                num_batches = 0

                for batch in subset_loader:
                    iq = batch["iq"].to(self.device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = contrastive(iq)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(cl_optimizer)
                    self.scaler.update()
                    cl_optimizer.zero_grad(set_to_none=True)

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / max(1, num_batches)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        "  [Contrastive] Epoch %d/%d  loss=%.4f",
                        epoch + 1, epochs, avg_loss,
                    )

            del contrastive, cl_optimizer

        logger.info("Phase 1 complete.")

    # ------------------------------------------------------------------
    # Phase 2: Supervised curriculum training
    # ------------------------------------------------------------------

    def supervised_curriculum(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        """Phase 2: Supervised training with SNR-based curriculum.

        Starts with high-SNR (easy) samples and progressively introduces
        lower-SNR (harder) samples via a linearly decreasing threshold.

        Each expert is trained individually.
        """
        phase_cfg = self.config.get_nested("training.phases.supervised", {})
        epochs = int(phase_cfg.get("epochs", 75))
        lr = float(phase_cfg.get("lr", 5e-4))

        logger.info("=" * 60)
        logger.info("PHASE 2: Supervised curriculum training (%d epochs)", epochs)
        logger.info("=" * 60)

        # Reset optimizer for this phase
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=float(
                self.config.get_nested("training.optimizer.weight_decay", 0.01)
            ),
        )
        self.scheduler = create_scheduler(self.optimizer, self.config)

        for epoch in range(epochs):
            snr_threshold = self.snr_curriculum.get_snr_threshold(epoch, epochs)
            metrics = self.train_epoch(
                phase="supervised",
                epoch=epoch,
                dataloader=train_loader,
                snr_threshold=snr_threshold,
            )

            val_metrics = self.validate(val_loader)
            self.scheduler.step(epoch)

            logger.info(
                "  Epoch %d/%d  snr_thresh=%+.1f dB  train_loss=%.4f  "
                "val_loss=%.4f  val_acc=%.4f",
                epoch + 1, epochs, snr_threshold,
                metrics["loss"], val_metrics["loss"], val_metrics["accuracy"],
            )

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(
                    epoch, val_metrics["loss"], phase="supervised"
                )

    # ------------------------------------------------------------------
    # Phase 3: Gating network training
    # ------------------------------------------------------------------

    def train_gating(self, train_loader: DataLoader, val_loader: DataLoader):
        """Phase 3: Train only gating network + shared layers.

        All expert parameters are frozen; only the router/gate and any
        shared expert layers are updated.
        """
        phase_cfg = self.config.get_nested("training.phases.gating", {})
        epochs = int(phase_cfg.get("epochs", 35))
        lr = float(phase_cfg.get("lr", 1e-4))

        logger.info("=" * 60)
        logger.info("PHASE 3: Gating network training (%d epochs)", epochs)
        logger.info("=" * 60)

        # Freeze all experts
        self._freeze_experts()

        # Collect trainable parameters (gating + shared layers)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info("  Trainable parameters: %d", sum(p.numel() for p in trainable_params))

        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=0.01
        )

        for epoch in range(epochs):
            metrics = self.train_epoch(
                phase="gating", epoch=epoch, dataloader=train_loader
            )
            val_metrics = self.validate(val_loader)

            logger.info(
                "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch + 1, epochs,
                metrics["loss"], val_metrics["loss"], val_metrics["accuracy"],
            )

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, val_metrics["loss"], phase="gating")

    # ------------------------------------------------------------------
    # Phase 4: End-to-end fine-tuning
    # ------------------------------------------------------------------

    def finetune_all(self, train_loader: DataLoader, val_loader: DataLoader):
        """Phase 4: Unfreeze everything and fine-tune end-to-end."""
        phase_cfg = self.config.get_nested("training.phases.finetune", {})
        epochs = int(phase_cfg.get("epochs", 15))
        lr = float(phase_cfg.get("lr", 1e-5))

        logger.info("=" * 60)
        logger.info("PHASE 4: End-to-end fine-tuning (%d epochs)", epochs)
        logger.info("=" * 60)

        # Unfreeze everything
        self._unfreeze_all()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )

        for epoch in range(epochs):
            metrics = self.train_epoch(
                phase="finetune", epoch=epoch, dataloader=train_loader
            )
            val_metrics = self.validate(val_loader)

            logger.info(
                "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch + 1, epochs,
                metrics["loss"], val_metrics["loss"], val_metrics["accuracy"],
            )

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, val_metrics["loss"], phase="finetune")

    # ------------------------------------------------------------------
    # Core training / validation loops
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        phase: str,
        epoch: int,
        dataloader: DataLoader,
        snr_threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """Run a single training epoch with BF16 autocast and gradient accumulation.

        Args:
            phase: Current training phase name (for logging).
            epoch: Current epoch index.
            dataloader: Training DataLoader.
            snr_threshold: Optional SNR threshold for curriculum filtering.

        Returns:
            Dict of epoch metrics: loss, accuracy, expert utilization.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_steps = 0

        # Compute global epoch for hierarchy weights (accumulate across phases)
        global_epoch = self._global_epoch(phase, epoch)
        hierarchy_weights = self.hierarchy_scheduler.get_weights(global_epoch)

        for step, batch in enumerate(dataloader):
            # SNR-based curriculum filtering
            if snr_threshold is not None:
                batch = self.snr_curriculum.filter_batch(batch, snr_threshold)
                if batch is None:
                    continue

            # Move to device
            iq = batch["iq"].to(self.device, non_blocking=True)
            spectrogram = batch["spectrogram"].to(self.device, non_blocking=True)
            hos = batch["hos"].to(self.device, non_blocking=True)
            cyclo = batch["cyclo"].to(self.device, non_blocking=True)
            label_binary = batch["label_binary"].to(self.device, non_blocking=True)
            label_type = batch["label_type"].to(self.device, non_blocking=True)
            label_full = batch["label_full"].to(self.device, non_blocking=True)

            # Sim-to-Real Augmentation
            if phase in ["supervised", "finetune"]:
                iq = self.augmentation(iq)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    iq=iq, spectrogram=spectrogram, hos=hos, cyclo=cyclo
                )

                # Hierarchical loss
                loss_binary = self.criterion_binary(outputs["logits_binary"], label_binary)
                loss_type = self.criterion_type(outputs["logits_type"], label_type)
                loss_full = self.criterion_full(outputs["logits_full"], label_full)

                loss = (
                    hierarchy_weights[0] * loss_binary
                    + hierarchy_weights[1] * loss_type
                    + hierarchy_weights[2] * loss_full
                )

                # Add auxiliary losses (load balancing, router z-loss) if present
                if "aux_loss" in outputs:
                    loss = loss + outputs["aux_loss"]

                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.accumulation_steps
            preds = outputs["logits_binary"].argmax(dim=-1)
            total_correct += (preds == label_binary).sum().item()
            total_samples += label_binary.shape[0]
            num_steps += 1

        avg_loss = total_loss / max(1, num_steps)
        accuracy = total_correct / max(1, total_samples)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": total_samples,
            "hierarchy_weights": hierarchy_weights,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation loop and return metrics.

        Args:
            dataloader: Validation DataLoader.

        Returns:
            Dict with loss, accuracy, and per-level accuracies.
        """
        self.model.eval()

        total_loss = 0.0
        correct_binary = 0
        correct_type = 0
        correct_full = 0
        total_samples = 0
        num_steps = 0

        for batch in dataloader:
            iq = batch["iq"].to(self.device, non_blocking=True)
            spectrogram = batch["spectrogram"].to(self.device, non_blocking=True)
            hos = batch["hos"].to(self.device, non_blocking=True)
            cyclo = batch["cyclo"].to(self.device, non_blocking=True)
            label_binary = batch["label_binary"].to(self.device, non_blocking=True)
            label_type = batch["label_type"].to(self.device, non_blocking=True)
            label_full = batch["label_full"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    iq=iq, spectrogram=spectrogram, hos=hos, cyclo=cyclo
                )
                loss_binary = self.criterion_binary(outputs["logits_binary"], label_binary)
                loss_type = self.criterion_type(outputs["logits_type"], label_type)
                loss_full = self.criterion_full(outputs["logits_full"], label_full)
                loss = loss_binary + loss_type + loss_full

            total_loss += loss.item()
            correct_binary += (outputs["logits_binary"].argmax(-1) == label_binary).sum().item()
            correct_type += (outputs["logits_type"].argmax(-1) == label_type).sum().item()
            correct_full += (outputs["logits_full"].argmax(-1) == label_full).sum().item()
            total_samples += label_binary.shape[0]
            num_steps += 1

        n = max(1, total_samples)
        return {
            "loss": total_loss / max(1, num_steps),
            "accuracy": correct_binary / n,
            "accuracy_binary": correct_binary / n,
            "accuracy_type": correct_type / n,
            "accuracy_full": correct_full / n,
            "num_samples": total_samples,
        }

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self):
        """Run all 4 training phases sequentially."""
        start_time = time.time()
        logger.info("Starting 4-phase progressive training pipeline")

        loaders = create_dataloaders(self.config)
        train_loader = loaders["train"]
        val_loader = loaders["val"]

        # Phase 1: Self-supervised pretraining
        self.pretrain_experts(train_loader)

        # Phase 2: Supervised curriculum training
        self.supervised_curriculum(train_loader, val_loader)

        # Phase 3: Gating network training
        self.train_gating(train_loader, val_loader)

        # Phase 4: End-to-end fine-tuning
        self.finetune_all(train_loader, val_loader)

        elapsed = time.time() - start_time
        logger.info(
            "Training complete. Total time: %.1f hours", elapsed / 3600
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_experts(self):
        """Freeze all expert modules, leave gating/shared trainable."""
        if hasattr(self.model, "experts"):
            for name, expert in self.model.experts.items():
                for p in expert.parameters():
                    p.requires_grad = False
                logger.info("  Frozen: %s", name)
        elif hasattr(self.model, "freeze_experts"):
            self.model.freeze_experts()

    def _unfreeze_all(self):
        """Unfreeze all model parameters."""
        for p in self.model.parameters():
            p.requires_grad = True
        logger.info("  All parameters unfrozen")

    def _global_epoch(self, phase: str, epoch: int) -> int:
        """Compute a global epoch index for hierarchy weight scheduling."""
        phases_cfg = self.config.get_nested("training.phases", {})
        offset = 0
        phase_order = ["pretrain", "supervised", "gating", "finetune"]
        for p in phase_order:
            if p == phase:
                return offset + epoch
            offset += int(phases_cfg.get(p, {}).get("epochs", 0))
        return offset + epoch

    def _save_checkpoint(self, epoch: int, val_loss: float, phase: str):
        """Save checkpoint and maintain top-K by validation loss."""
        filename = f"checkpoint_{phase}_epoch{epoch + 1}.pt"
        path = self.save_dir / filename

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=val_loss,
            path=path,
            scheduler=self.scheduler,
            extra={"phase": phase},
        )

        self._top_checkpoints.append((val_loss, path))
        self._top_checkpoints.sort(key=lambda x: x[0])

        # Remove excess checkpoints
        while len(self._top_checkpoints) > self.keep_top_k:
            _, old_path = self._top_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()
                logger.debug("Removed checkpoint: %s", old_path)

        logger.info(
            "  Saved checkpoint: %s (val_loss=%.4f, keeping top %d)",
            filename, val_loss, self.keep_top_k,
        )
