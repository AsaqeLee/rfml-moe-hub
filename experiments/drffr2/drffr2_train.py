#!/usr/bin/env python3
"""
DRFF-R2 Training Pipeline — PyTorch + ROCm on MI300X
=====================================================
Multi-experiment classifier for drone RF fingerprinting.

Tasks:
  model      — 8-class drone model classification
  individual — 26-class drone individual identification
  state      — flight state classification
  multi      — joint model+individual (multi-head, experimental)

Experiments:
  Single dataset: --dataset dataset1
  All datasets:   --dataset all
  Cross-scenario: train on dataset1, eval on dataset3 and dataset6

Usage:
  python3 drffr2_train.py --dataset dataset1 --task model
  python3 drffr2_train.py --dataset all --task individual --models convnext_base efficientnet_b0
  python3 drffr2_train.py --dataset dataset1 --task model --cross-eval
  python3 drffr2_train.py --dataset dataset1 --task model --yolo
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
import timm
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

# ============================================================================
# PATHS & CONSTANTS
# ============================================================================

SPEC_ROOT  = "/home/rax/mtp/drffr2_spectrograms"
MODEL_DIR  = "/home/rax/mtp/models"
RESULT_DIR = "/home/rax/mtp/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 8 drone models
DRONE_MODELS = [
    "mavicAir2", "mavic3", "mavic3C", "mavic3S",
    "mavicAir2s", "mini3pro", "mini4PRO", "mini5PRO",
]

# 7 dataset folders
ALL_DATASETS = [f"dataset{i}" for i in range(1, 8)]

# Datasets to use for cross-scenario evaluation
CROSS_EVAL_DATASETS = ["dataset3", "dataset6"]

# timm model registry
TIMM_MODELS = {
    "convnext_base":     "convnext_base",
    "maxvit_base":       "maxvit_base_tf_224",
    "efficientnet_b0":   "efficientnet_b0",
    "resnet18":          "resnet18",
    "mobilenetv3_large": "mobilenetv3_large_100",
}

DEFAULT_CFG = {
    "batch_size":      64,
    "img_size":        224,
    "epochs":          80,
    "lr":              1e-4,
    "weight_decay":    0.01,
    "label_smoothing": 0.1,
    "patience":        20,
    "num_workers":     8,
    "amp":             True,   # BF16 mixed precision
}


# ============================================================================
# DATA
# ============================================================================

def get_transforms(img_size: int, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def dataset_split_dir(dataset_name: str, split: str) -> str:
    return os.path.join(SPEC_ROOT, dataset_name, split)


def load_image_folder(split_dir: str, transform) -> datasets.ImageFolder | None:
    """Load ImageFolder if directory exists and has class subdirs."""
    if not os.path.isdir(split_dir):
        return None
    # Verify at least one class subdir with images exists
    subdirs = [
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ]
    if not subdirs:
        return None
    return datasets.ImageFolder(split_dir, transform=transform)


def load_single_dataset(dataset_name: str, img_size: int, batch_size: int,
                        num_workers: int) -> tuple:
    """
    Load train/val splits for a single dataset folder.

    Returns (train_loader, val_loader, num_classes, class_names)
    """
    train_dir = dataset_split_dir(dataset_name, "train")
    val_dir   = dataset_split_dir(dataset_name, "val")

    train_ds = load_image_folder(train_dir, get_transforms(img_size, is_train=True))
    val_ds   = load_image_folder(val_dir,   get_transforms(img_size, is_train=False))

    if train_ds is None:
        raise FileNotFoundError(f"No train data found at {train_dir}")
    if val_ds is None:
        raise FileNotFoundError(f"No val data found at {val_dir}")

    # Align class indices: val must use same class-to-idx as train
    val_ds.class_to_idx = train_ds.class_to_idx
    val_ds.classes      = train_ds.classes
    # Remap val sample indices
    val_ds.samples  = [
        (p, train_ds.class_to_idx[os.path.basename(os.path.dirname(p))])
        for p, _ in val_ds.samples
        if os.path.basename(os.path.dirname(p)) in train_ds.class_to_idx
    ]
    val_ds.targets = [s[1] for s in val_ds.samples]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    num_classes = len(train_ds.classes)
    class_names = train_ds.classes

    print(f"  Train: {len(train_ds):,} images, {num_classes} classes")
    print(f"  Val:   {len(val_ds):,} images")
    print(f"  Classes: {class_names}")

    return train_loader, val_loader, num_classes, class_names


def load_all_datasets(img_size: int, batch_size: int, num_workers: int) -> tuple:
    """
    Concatenate all available datasets.
    Class universe is the union of all classes found.

    Returns (train_loader, val_loader, num_classes, class_names)
    """
    all_train_ds = []
    all_val_ds   = []
    all_classes  = set()

    # First pass: collect all class names
    for ds_name in ALL_DATASETS:
        train_dir = dataset_split_dir(ds_name, "train")
        ds = load_image_folder(train_dir, get_transforms(img_size, is_train=True))
        if ds is not None:
            all_classes.update(ds.classes)

    if not all_classes:
        raise RuntimeError("No datasets found under " + SPEC_ROOT)

    class_names   = sorted(all_classes)
    class_to_idx  = {c: i for i, c in enumerate(class_names)}
    num_classes   = len(class_names)

    # Second pass: load and remap indices
    for ds_name in ALL_DATASETS:
        train_dir = dataset_split_dir(ds_name, "train")
        val_dir   = dataset_split_dir(ds_name, "val")

        for split_dir, is_train, container in [
            (train_dir, True,  all_train_ds),
            (val_dir,   False, all_val_ds),
        ]:
            ds = load_image_folder(
                split_dir, get_transforms(img_size, is_train=is_train)
            )
            if ds is None:
                continue
            # Remap to unified class_to_idx
            ds.class_to_idx = class_to_idx
            ds.classes      = class_names
            ds.samples      = [
                (p, class_to_idx[os.path.basename(os.path.dirname(p))])
                for p, _ in ds.samples
                if os.path.basename(os.path.dirname(p)) in class_to_idx
            ]
            ds.targets = [s[1] for s in ds.samples]
            container.append(ds)

    if not all_train_ds:
        raise RuntimeError("No train data found across all datasets")

    train_combined = ConcatDataset(all_train_ds)
    val_combined   = ConcatDataset(all_val_ds) if all_val_ds else None

    train_loader = DataLoader(
        train_combined, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_combined, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    ) if val_combined else None

    print(f"  Train (all): {len(train_combined):,} images, {num_classes} classes")
    if val_combined:
        print(f"  Val   (all): {len(val_combined):,} images")
    print(f"  Classes: {class_names}")

    return train_loader, val_loader, num_classes, class_names


def load_eval_dataset(dataset_name: str, img_size: int, batch_size: int,
                      num_workers: int, class_to_idx: dict,
                      class_names: list) -> DataLoader | None:
    """
    Load a dataset for cross-scenario evaluation, mapping to a fixed class set.
    """
    val_dir = dataset_split_dir(dataset_name, "val")
    ds = load_image_folder(val_dir, get_transforms(img_size, is_train=False))
    if ds is None:
        # Try train dir as fallback
        train_dir = dataset_split_dir(dataset_name, "train")
        ds = load_image_folder(train_dir, get_transforms(img_size, is_train=False))
    if ds is None:
        print(f"  [WARN] No data found for cross-eval dataset {dataset_name}", flush=True)
        return None

    ds.class_to_idx = class_to_idx
    ds.classes      = class_names
    ds.samples      = [
        (p, class_to_idx[os.path.basename(os.path.dirname(p))])
        for p, _ in ds.samples
        if os.path.basename(os.path.dirname(p)) in class_to_idx
    ]
    ds.targets = [s[1] for s in ds.samples]

    if len(ds.samples) == 0:
        print(f"  [WARN] 0 samples matched for cross-eval {dataset_name}", flush=True)
        return None

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ============================================================================
# MODEL
# ============================================================================

def create_timm_model(model_key: str, num_classes: int,
                      pretrained: bool = True) -> tuple[nn.Module, int]:
    timm_name = TIMM_MODELS[model_key]
    model     = timm.create_model(timm_name, pretrained=pretrained,
                                  num_classes=num_classes)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {timm_name} ({n_params/1e6:.1f}M params)")
    return model, n_params


# ============================================================================
# TRAINING LOOP
# ============================================================================

def evaluate(model: nn.Module, loader: DataLoader, cfg: dict) -> tuple[float, list, list]:
    """
    Run inference on loader.
    Returns (accuracy, all_preds, all_labels).
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=cfg["amp"]):
                outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds.tolist(), all_labels.tolist()


def train_one_model(model_key: str, train_loader: DataLoader,
                    val_loader: DataLoader, num_classes: int,
                    class_names: list, cfg: dict,
                    cross_eval_loaders: dict | None = None) -> dict:
    """
    Full train + eval cycle for one timm model.

    Returns result dict.
    """
    print(f"\n{'='*70}")
    print(f"  Training: {model_key}  ({num_classes} classes)")
    print(f"{'='*70}", flush=True)

    model, n_params = create_timm_model(model_key, num_classes)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler    = torch.amp.GradScaler(enabled=cfg["amp"])

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0
    train_start  = time.time()
    epoch        = 0

    for epoch in range(cfg["epochs"]):
        # ---- Train ----
        model.train()
        train_loss = train_correct = train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=cfg["amp"]):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        # ---- Validate ----
        val_acc, _, _ = evaluate(model, val_loader, cfg) if val_loader else (0.0, [], [])

        train_acc = train_correct / max(train_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
            marker = "*"
        else:
            no_improve += 1
            marker = ""

        if (epoch + 1) % 5 == 0 or marker == "*":
            elapsed = time.time() - train_start
            print(
                f"  Epoch {epoch+1:3d}: train={train_acc:.4f} val={val_acc:.4f} "
                f"loss={train_loss/max(train_total,1):.4f} [{elapsed:.0f}s] {marker}",
                flush=True,
            )

        if no_improve >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break

    # ---- Load best checkpoint ----
    if best_state:
        model.load_state_dict(best_state)
        ckpt_path = os.path.join(MODEL_DIR, f"drffr2_{model_key}_best.pt")
        torch.save(best_state, ckpt_path)
        print(f"  Saved best checkpoint -> {ckpt_path}")

    # ---- Full evaluation on val set ----
    if val_loader:
        val_acc_final, all_preds, all_labels = evaluate(model, val_loader, cfg)
        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        f1     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        report = classification_report(
            all_labels, all_preds, target_names=class_names,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(all_labels, all_preds).tolist()
        print(f"\n  RESULT: {model_key}")
        print(f"  Val Accuracy: {val_acc_final:.4f} | F1-macro: {f1:.4f}")
        print(f"  Params: {n_params/1e6:.1f}M | Epochs: {epoch+1} | "
              f"Time: {time.time()-train_start:.0f}s")
        print(classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0
        ))
    else:
        val_acc_final = best_val_acc
        f1     = 0.0
        report = {}
        cm     = []
        all_preds  = np.array([])
        all_labels = np.array([])

    result = {
        "model":          model_key,
        "accuracy":       float(val_acc_final),
        "f1_macro":       float(f1),
        "best_val_acc":   float(best_val_acc),
        "params":         int(n_params),
        "train_time_sec": float(time.time() - train_start),
        "epochs_trained": int(epoch + 1),
        "per_class":      {cn: report[cn] for cn in class_names if cn in report},
        "confusion_matrix": cm,
        "class_names":    class_names,
    }

    # ---- Cross-scenario evaluation ----
    if cross_eval_loaders:
        result["cross_eval"] = {}
        for ds_name, loader in cross_eval_loaders.items():
            if loader is None:
                continue
            ce_acc, ce_preds, ce_labels = evaluate(model, loader, cfg)
            ce_preds  = np.array(ce_preds)
            ce_labels = np.array(ce_labels)
            ce_f1 = f1_score(ce_labels, ce_preds, average="macro", zero_division=0)
            result["cross_eval"][ds_name] = {
                "accuracy": float(ce_acc),
                "f1_macro": float(ce_f1),
            }
            print(f"  Cross-eval [{ds_name}]: acc={ce_acc:.4f} f1={ce_f1:.4f}",
                  flush=True)

    return result


# ============================================================================
# YOLO CLASSIFIER
# ============================================================================

def train_yolo_classifier(dataset_dir: str, cfg: dict) -> dict:
    """Train yolo11n-cls on a dataset directory."""
    from ultralytics import YOLO

    model_variant = "yolo11n-cls"
    print(f"\n{'='*70}")
    print(f"  Training: {model_variant}")
    print(f"{'='*70}", flush=True)

    train_start = time.time()
    model = YOLO(f"{model_variant}.pt")

    model.train(
        data=dataset_dir,
        epochs=cfg["epochs"],
        imgsz=cfg["img_size"],
        batch=cfg["batch_size"],
        patience=cfg["patience"],
        project=MODEL_DIR,
        name=f"drffr2_{model_variant}",
        device=0,
        workers=cfg["num_workers"],
        optimizer="AdamW",
        lr0=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        label_smoothing=cfg["label_smoothing"],
        pretrained=True,
        verbose=True,
        cos_lr=True,
    )

    # Evaluate on val split
    val_dir = os.path.join(dataset_dir, "val")
    classes = sorted(
        d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))
    ) if os.path.isdir(val_dir) else []

    all_preds_str, all_labels_str = [], []
    for cls in classes:
        cls_dir = os.path.join(val_dir, cls)
        imgs    = sorted(f for f in os.listdir(cls_dir) if f.endswith(".png"))
        for img_name in imgs:
            img_path = os.path.join(cls_dir, img_name)
            res      = model.predict(img_path, verbose=False)
            pred_cls = res[0].names[res[0].probs.top1]
            all_preds_str.append(pred_cls)
            all_labels_str.append(cls)

    acc = accuracy_score(all_labels_str, all_preds_str) if all_labels_str else 0.0
    f1  = f1_score(all_labels_str, all_preds_str,
                   average="macro", zero_division=0) if all_labels_str else 0.0
    total_time = time.time() - train_start

    result = {
        "model":          model_variant,
        "accuracy":       float(acc),
        "f1_macro":       float(f1),
        "train_time_sec": float(total_time),
        "type":           "yolo-cls",
    }

    print(f"\n  {model_variant}: Accuracy={acc:.4f}  F1={f1:.4f}")
    if all_labels_str:
        print(classification_report(all_labels_str, all_preds_str, zero_division=0))

    # Copy best weights
    yolo_best = os.path.join(
        MODEL_DIR, f"drffr2_{model_variant}", "weights", "best.pt"
    )
    dest = os.path.join(MODEL_DIR, f"drffr2_{model_variant}_best.pt")
    if os.path.exists(yolo_best):
        shutil.copy2(yolo_best, dest)
        print(f"  Saved best checkpoint -> {dest}")

    return result


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DRFF-R2 Training Pipeline — MI300X ROCm"
    )
    parser.add_argument(
        "--dataset", default="dataset1",
        help=(
            "Dataset subfolder to train on. "
            "Use 'all' to concatenate all datasets. "
            f"Choices: {ALL_DATASETS + ['all']}. Default: dataset1"
        ),
    )
    parser.add_argument(
        "--task", default="model",
        choices=["model", "individual", "state", "multi"],
        help=(
            "Classification task: "
            "'model' = 8-class drone model, "
            "'individual' = 26-class drone identity, "
            "'state' = flight state, "
            "'multi' = joint model+individual (experimental). "
            "Default: model"
        ),
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["convnext_base", "maxvit_base", "efficientnet_b0"],
        choices=list(TIMM_MODELS.keys()),
        help="timm model keys to train. Default: convnext_base maxvit_base efficientnet_b0",
    )
    parser.add_argument("--batch-size", type=int,  default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--epochs",     type=int,  default=DEFAULT_CFG["epochs"])
    parser.add_argument("--lr",         type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument(
        "--cross-eval", action="store_true",
        help=(
            f"After training on --dataset, evaluate on "
            f"{CROSS_EVAL_DATASETS}. Ignored when --dataset all."
        ),
    )
    parser.add_argument(
        "--yolo", action="store_true",
        help="Also train yolo11n-cls",
    )
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("DRFF-R2 TRAINING PIPELINE — MI300X", flush=True)
    print(f"Device  : {DEVICE}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB",
              flush=True)
    print(f"Dataset : {args.dataset}", flush=True)
    print(f"Task    : {args.task}", flush=True)
    print(f"Models  : {args.models}", flush=True)
    print("=" * 70, flush=True)

    cfg = DEFAULT_CFG.copy()
    cfg["batch_size"] = args.batch_size
    cfg["epochs"]     = args.epochs
    cfg["lr"]         = args.lr

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    print("\nLoading data...", flush=True)

    try:
        if args.dataset == "all":
            train_loader, val_loader, num_classes, class_names = load_all_datasets(
                cfg["img_size"], cfg["batch_size"], cfg["num_workers"]
            )
            cross_eval_loaders = None  # not meaningful for "all"
        else:
            train_loader, val_loader, num_classes, class_names = load_single_dataset(
                args.dataset, cfg["img_size"], cfg["batch_size"], cfg["num_workers"]
            )
            # Prepare cross-eval loaders
            cross_eval_loaders = None
            if args.cross_eval:
                # Build class_to_idx from train data
                train_ds = train_loader.dataset
                c2i      = (
                    train_ds.class_to_idx
                    if hasattr(train_ds, "class_to_idx")
                    else {c: i for i, c in enumerate(class_names)}
                )
                cross_eval_loaders = {}
                for ds_name in CROSS_EVAL_DATASETS:
                    if ds_name == args.dataset:
                        continue
                    loader = load_eval_dataset(
                        ds_name, cfg["img_size"], cfg["batch_size"],
                        cfg["num_workers"], c2i, class_names,
                    )
                    cross_eval_loaders[ds_name] = loader
    except FileNotFoundError as e:
        print(f"ERROR: {e}", flush=True)
        print(
            "Run drffr2_specgen.py first to generate spectrograms.",
            flush=True,
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Train timm models
    # ------------------------------------------------------------------ #
    all_results = {}

    for model_key in args.models:
        try:
            result = train_one_model(
                model_key, train_loader, val_loader,
                num_classes, class_names, cfg,
                cross_eval_loaders=cross_eval_loaders,
            )
            all_results[model_key] = result
        except Exception as e:
            print(f"ERROR training {model_key}: {e}", flush=True)
            traceback.print_exc()
            all_results[model_key] = {"error": str(e), "model": model_key}

    # ------------------------------------------------------------------ #
    # YOLO
    # ------------------------------------------------------------------ #
    if args.yolo:
        # Point YOLO at the per-dataset spectrogram folder (expects train/val subdirs)
        if args.dataset == "all":
            # YOLO can't easily handle ConcatDataset; use dataset1 as proxy
            yolo_data_dir = os.path.join(SPEC_ROOT, "dataset1")
            print(
                "[WARN] --yolo with --dataset all: using dataset1 for YOLO data dir",
                flush=True,
            )
        else:
            yolo_data_dir = os.path.join(SPEC_ROOT, args.dataset)

        try:
            result = train_yolo_classifier(yolo_data_dir, cfg)
            all_results["yolo11n-cls"] = result
        except Exception as e:
            print(f"ERROR training yolo11n-cls: {e}", flush=True)
            traceback.print_exc()
            all_results["yolo11n-cls"] = {"error": str(e), "model": "yolo11n-cls"}

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print(f"\n{'='*90}", flush=True)
    print(
        f"FINAL RESULTS — DRFF-R2  dataset={args.dataset}  task={args.task}",
        flush=True,
    )
    print("=" * 90, flush=True)
    print(
        f"{'Model':<25} {'Acc':>8} {'F1-macro':>10} {'Params':>10} {'Epochs':>7} {'Time':>8}",
        flush=True,
    )
    print("-" * 90, flush=True)

    sorted_names = sorted(
        all_results.keys(),
        key=lambda k: all_results[k].get("accuracy", 0.0),
        reverse=True,
    )
    for name in sorted_names:
        r = all_results[name]
        if "error" in r:
            print(f"  {name:<23} {'ERROR':>8}", flush=True)
        else:
            params = r.get("params", 0)
            pstr   = f"{params/1e6:.1f}M" if params else "?"
            tstr   = f"{r.get('train_time_sec', 0):.0f}s"
            ep     = r.get("epochs_trained", "?")
            print(
                f"  {name:<23} {r['accuracy']:>8.4f} {r['f1_macro']:>10.4f} "
                f"{pstr:>10} {ep:>7} {tstr:>8}",
                flush=True,
            )
            if "cross_eval" in r and r["cross_eval"]:
                for ds_n, ce in r["cross_eval"].items():
                    print(
                        f"    cross-eval [{ds_n}]: "
                        f"acc={ce['accuracy']:.4f}  f1={ce['f1_macro']:.4f}",
                        flush=True,
                    )

    # ------------------------------------------------------------------ #
    # Save results JSON
    # ------------------------------------------------------------------ #
    result_file = os.path.join(
        RESULT_DIR, f"drffr2_{args.dataset}_{args.task}_results.json"
    )
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved -> {result_file}", flush=True)


if __name__ == "__main__":
    main()
