#!/usr/bin/env python3
"""
RFML-MoE: Multi-modal Mixture-of-Experts Drone RF Signal Detection Pipeline

Complete pipeline for:
1. Downloading drone RF datasets (RFUAV, DroneDetect, CardRF, DroneRF, Tampere)
2. Feature extraction (IQ, Spectrogram, HOS, Cyclostationary)
3. Training expert models with progressive 4-phase curriculum
4. Building and training MoE ensemble with Expert Choice routing
5. Evaluation with SNR-stratified metrics, hierarchical F1, open-set detection
"""

import os
import sys
import click
import torch
from pathlib import Path

from utils.config import load_config
from utils.logging import setup_logging, get_logger
from utils.helpers import set_seed, get_device, setup_rocm_env, count_parameters


@click.group()
@click.option("--config", "-c", default="configs/default.yaml", help="Path to config file")
@click.option("--log-level", default="INFO", help="Logging level")
@click.pass_context
def cli(ctx, config, log_level):
    """RFML-MoE: Drone RF Signal Detection Pipeline"""
    ctx.ensure_object(dict)
    cfg = load_config(config)
    ctx.obj["config"] = cfg
    setup_logging(level=log_level)


@cli.command()
@click.option("--datasets", "-d", multiple=True, default=None,
              help="Specific datasets to download (default: all enabled)")
@click.pass_context
def download(ctx, datasets):
    """Download and prepare RF datasets."""
    from data.download import DownloadManager

    log = get_logger("rfml.main")
    cfg = ctx.obj["config"]

    log.info("Starting dataset download...")
    manager = DownloadManager(cfg)

    if datasets:
        for ds in datasets:
            manager.download_dataset(ds)
    else:
        manager.download_all()

    log.info("Dataset download complete.")


@cli.command()
@click.option("--dataset", "-d", default=None, help="Process specific dataset")
@click.option("--workers", "-w", default=12, help="Number of parallel workers")
@click.pass_context
def extract_features(ctx, dataset, workers):
    """Extract features from raw IQ data (spectrograms, HOS, cyclostationary)."""
    from features.pipeline import FeaturePipeline

    log = get_logger("rfml.main")
    cfg = ctx.obj["config"]

    log.info("Starting feature extraction pipeline...")
    pipeline = FeaturePipeline(cfg)
    pipeline.process_all(num_workers=workers, dataset_filter=dataset)
    log.info("Feature extraction complete.")


@cli.command()
@click.option("--phase", "-p", type=click.Choice(["all", "pretrain", "supervised", "gating", "finetune"]),
              default="all", help="Training phase to run")
@click.option("--resume", "-r", default=None, help="Checkpoint path to resume from")
@click.option("--compile/--no-compile", default=False, help="Use torch.compile")
@click.pass_context
def train(ctx, phase, resume, compile):
    """Train the MoE drone RF detection model."""
    from models.moe.moe_model import DroneRFMoE
    from models.moe.losses import HierarchicalLoss
    from training.trainer import MoETrainer
    from data.dataset import create_dataloaders

    log = get_logger("rfml.main")
    cfg = ctx.obj["config"]
    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)

    # Setup ROCm environment if available
    if hasattr(cfg, "rocm") and torch.cuda.is_available():
        setup_rocm_env(cfg.rocm.env)
        log.info("ROCm environment configured for MI300X")

    log.info("Building model...")
    model = DroneRFMoE(cfg).to(device)
    log.info(f"Model parameters: {count_parameters(model):,} trainable")

    if compile and hasattr(torch, "compile"):
        log.info("Compiling model with torch.compile (max-autotune)...")
        model = torch.compile(model, mode="max-autotune")

    log.info("Creating data loaders...")
    loaders = create_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders.get("test")

    log.info("Initializing trainer...")
    trainer = MoETrainer(
        model=model,
        config=cfg,
        device=device,
    )

    if resume:
        log.info(f"Resuming from checkpoint: {resume}")
        from utils.helpers import load_checkpoint as load_ckpt
        load_ckpt(resume, model, trainer.optimizer, device=str(device))

    if phase == "all":
        log.info("Running full 4-phase progressive training...")
        trainer.run()
    elif phase == "pretrain":
        trainer.pretrain_experts(train_loader)
    elif phase == "supervised":
        trainer.supervised_curriculum(train_loader, val_loader)
    elif phase == "gating":
        trainer.train_gating(train_loader, val_loader)
    elif phase == "finetune":
        trainer.finetune_all(train_loader, val_loader)

    log.info("Training complete.")


@cli.command()
@click.option("--checkpoint", "-ckpt", required=True, help="Model checkpoint path")
@click.option("--output", "-o", default="results", help="Output directory for results")
@click.option("--open-set/--no-open-set", default=False, help="Run open-set evaluation")
@click.pass_context
def evaluate(ctx, checkpoint, output, open_set):
    """Evaluate a trained model with comprehensive metrics."""
    from models.moe.moe_model import DroneRFMoE
    from evaluation.evaluator import MoEEvaluator
    from data.dataset import create_dataloaders
    from utils.helpers import load_checkpoint as load_ckpt

    log = get_logger("rfml.main")
    cfg = ctx.obj["config"]
    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)

    log.info("Building model...")
    model = DroneRFMoE(cfg).to(device)
    load_ckpt(checkpoint, model, device=str(device))
    log.info(f"Loaded checkpoint: {checkpoint}")

    log.info("Creating data loaders...")
    loaders = create_dataloaders(cfg)
    test_loader = loaders.get("test", loaders.get("val"))

    evaluator = MoEEvaluator(model=model, config=cfg, device=device)

    log.info("Running evaluation...")
    results = evaluator.evaluate(test_loader)

    log.info("Running SNR-stratified evaluation...")
    snr_results = evaluator.evaluate_snr_stratified(test_loader)
    results["snr_stratified"] = snr_results

    log.info("Analyzing expert routing...")
    routing_results = evaluator.evaluate_expert_routing(test_loader)
    results["routing"] = routing_results

    if open_set:
        log.info("Running open-set evaluation...")
        os_results = evaluator.evaluate_open_set(test_loader, test_loader)
        results["open_set"] = os_results

    output_dir = Path(output)
    evaluator.generate_report(results, output_dir)
    log.info(f"Results saved to {output_dir}")

    # Print summary
    for level in ["level1", "level2", "level3"]:
        if level in results:
            acc = results[level].get("accuracy", 0)
            f1 = results[level].get("f1_macro", 0)
            log.info(f"  {level}: accuracy={acc:.4f}, F1(macro)={f1:.4f}")


@cli.command()
@click.pass_context
def info(ctx):
    """Display pipeline configuration and system info."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    cfg = ctx.obj["config"]

    console.print("\n[bold cyan]RFML-MoE Pipeline Configuration[/bold cyan]\n")

    # System info
    table = Table(title="System Information")
    table.add_column("Property", style="green")
    table.add_column("Value")

    table.add_row("PyTorch", torch.__version__)
    backend = "ROCm" if hasattr(torch.version, "hip") and torch.version.hip else "CUDA" if torch.cuda.is_available() else "CPU"
    table.add_row("Backend", backend)
    table.add_row("GPU Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("GPU", torch.cuda.get_device_name(0))
        mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        table.add_row("GPU Memory", f"{mem_gb:.1f} GB")
        if hasattr(torch.version, "hip") and torch.version.hip:
            table.add_row("ROCm Version", str(torch.version.hip))
        elif hasattr(torch.version, "cuda") and torch.version.cuda:
            table.add_row("CUDA Version", str(torch.version.cuda))
    table.add_row("Precision", cfg.project.precision)
    table.add_row("Seed", str(cfg.project.seed))
    console.print(table)

    # Dataset info
    ds_table = Table(title="Datasets")
    ds_table.add_column("Dataset", style="green")
    ds_table.add_column("Enabled")
    ds_table.add_column("Source")
    for name, ds_cfg in cfg.data.datasets.items():
        ds_table.add_row(name, str(ds_cfg.get("enabled", False)), ds_cfg.get("source", "unknown"))
    console.print(ds_table)

    # Expert info
    exp_table = Table(title="Expert Models")
    exp_table.add_column("Expert", style="green")
    exp_table.add_column("Type")
    exp_table.add_column("Params (approx)")
    for name, exp_cfg in cfg.experts.items():
        exp_table.add_row(exp_cfg["name"], exp_cfg["type"], exp_cfg.get("params_approx", "?"))
    console.print(exp_table)

    # Training phases
    phase_table = Table(title="Training Phases")
    phase_table.add_column("Phase", style="green")
    phase_table.add_column("Epochs")
    phase_table.add_column("LR")
    phase_table.add_column("Description")
    for name, phase_cfg in cfg.training.phases.items():
        phase_table.add_row(
            name, str(phase_cfg["epochs"]), str(phase_cfg["lr"]), phase_cfg["description"]
        )
    console.print(phase_table)


@cli.command()
@click.option("--num-samples", "-n", default=100000, help="Number of synthetic samples to generate")
@click.option("--difficulty", "-d", type=click.Choice(["easy", "medium", "hard", "mixed"]),
              default="mixed", help="Difficulty level for synthetic data")
@click.option("--output", "-o", default="data/processed/synthetic", help="Output directory")
@click.pass_context
def generate_synthetic(ctx, num_samples, difficulty, output):
    """Generate synthetic RF training data with realistic impairments."""
    from data.synthetic import SyntheticDatasetGenerator

    log = get_logger("rfml.main")
    cfg = ctx.obj["config"]
    set_seed(cfg.project.seed)

    log.info("Generating %d synthetic samples (difficulty=%s)...", num_samples, difficulty)

    generator = SyntheticDatasetGenerator(cfg)
    generator.generate_dataset(
        num_samples=num_samples,
        output_dir=output,
        difficulty=difficulty,
    )
    log.info("Synthetic dataset generation complete. Output: %s", output)


@cli.command()
@click.pass_context
def run_all(ctx):
    """Run the complete pipeline: download → extract → train → evaluate."""
    log = get_logger("rfml.main")
    log.info("=" * 60)
    log.info("RFML-MoE: Full Pipeline Execution")
    log.info("=" * 60)

    # Invoke each stage
    ctx.invoke(download)
    ctx.invoke(generate_synthetic)
    ctx.invoke(extract_features)
    ctx.invoke(train)

    # Find best checkpoint
    ckpt_dir = Path(ctx.obj["config"].training.checkpointing.save_dir)
    checkpoints = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoints:
        ctx.invoke(evaluate, checkpoint=str(checkpoints[0]), output="results")
    else:
        log.warning("No checkpoints found after training.")

    log.info("=" * 60)
    log.info("Pipeline complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    cli()
