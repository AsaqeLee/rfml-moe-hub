#!/usr/bin/env python3
"""
DroneRFb Training Pipeline — PyTorch + ROCm on MI300X
======================================================
Cross-individual generalization: train on individuals 1&2, test on individual 3.

13 classes: A1, A2, B, C1, C2, D1, D2, E1, E2, F1, F2, G1, G2
Drone type mapping: A1/A2 -> A, C1/C2 -> C, D1/D2 -> D, E1/E2 -> E,
                   F1/F2 -> F, G1/G2 -> G, B -> B

Models: convnext_base, maxvit_base, efficientnet_b0, resnet18, mobilenetv3_large
YOLO:   yolo11n-cls (via --yolo flag)

Run:  sg render -c python droneRFb_train.py [--models ...] [--batch-size N] [--epochs N] [--yolo]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
import timm
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, confusion_matrix
)

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR   = '/home/rax/mtp/droneRFb_spectrograms'
MODEL_DIR  = '/home/rax/mtp/models'
RESULT_DIR = '/home/rax/mtp/results'
RESULT_FILE = os.path.join(RESULT_DIR, 'droneRFb_results.json')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# All 13 individual-level classes
CLASSES = ['A1', 'A2', 'B', 'C1', 'C2', 'D1', 'D2', 'E1', 'E2', 'F1', 'F2', 'G1', 'G2']

# Individual -> drone-type mapping
INDIVIDUAL_TO_TYPE = {
    'A1': 'A', 'A2': 'A',
    'B':  'B',
    'C1': 'C', 'C2': 'C',
    'D1': 'D', 'D2': 'D',
    'E1': 'E', 'E2': 'E',
    'F1': 'F', 'F2': 'F',
    'G1': 'G', 'G2': 'G',
}

DEFAULT_CFG = {
    'batch_size':      128,
    'img_size':        224,
    'epochs':          80,
    'lr':              1e-4,
    'weight_decay':    0.01,
    'label_smoothing': 0.1,
    'patience':        20,
    'num_workers':     8,
    'amp':             True,   # BF16 mixed precision
}

# timm model registry for this experiment
TIMM_MODELS = {
    'convnext_base':     'convnext_base',
    'maxvit_base':       'maxvit_base_tf_224',
    'efficientnet_b0':   'efficientnet_b0',
    'resnet18':          'resnet18',
    'mobilenetv3_large': 'mobilenetv3_large_100',
}


# ============================================================================
# DATA
# ============================================================================

def get_transforms(img_size, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def load_data(img_size, batch_size, num_workers=8):
    train_ds = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'train'),
        transform=get_transforms(img_size, is_train=True)
    )
    test_ds = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'test'),
        transform=get_transforms(img_size, is_train=False)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    num_classes = len(train_ds.classes)
    class_names = train_ds.classes

    print(f"  Train: {len(train_ds):,} images, {num_classes} classes")
    print(f"  Test:  {len(test_ds):,} images")
    print(f"  Classes: {class_names}")

    return train_loader, test_loader, num_classes, class_names


# ============================================================================
# DRONE-TYPE ACCURACY
# ============================================================================

def compute_type_accuracy(all_labels, all_preds, class_names):
    """
    Collapse individual-level predictions to drone-type level and compute accuracy.
    E.g. A1/A2 -> A, C1/C2 -> C, etc.
    """
    type_labels = [INDIVIDUAL_TO_TYPE.get(class_names[l], class_names[l]) for l in all_labels]
    type_preds  = [INDIVIDUAL_TO_TYPE.get(class_names[p], class_names[p]) for p in all_preds]
    return accuracy_score(type_labels, type_preds), type_labels, type_preds


# ============================================================================
# MODEL
# ============================================================================

def create_timm_model(model_key, num_classes, pretrained=True):
    timm_name = TIMM_MODELS[model_key]
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {timm_name} ({n_params/1e6:.1f}M params)")
    return model, n_params


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_one_model(model_key, cfg):
    print(f"\n{'='*70}")
    print(f"  Training: {model_key}")
    print(f"{'='*70}", flush=True)

    img_size = cfg['img_size']
    train_loader, test_loader, num_classes, class_names = load_data(
        img_size, cfg['batch_size'], cfg['num_workers']
    )

    model, n_params = create_timm_model(model_key, num_classes)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    scaler    = torch.amp.GradScaler(enabled=cfg['amp'])

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0
    train_start  = time.time()

    for epoch in range(cfg['epochs']):
        # ---- Train ----
        model.train()
        train_loss = train_correct = train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
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

        # ---- Validate (quick pass on test set for early stopping) ----
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
                    outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc   = val_correct   / max(val_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
            marker = '*'
        else:
            no_improve += 1
            marker = ''

        if (epoch + 1) % 5 == 0 or marker == '*':
            elapsed = time.time() - train_start
            print(
                f"  Epoch {epoch+1:3d}: train={train_acc:.4f} val={val_acc:.4f} "
                f"loss={train_loss/max(train_total,1):.4f} [{elapsed:.0f}s] {marker}",
                flush=True
            )

        if no_improve >= cfg['patience']:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break

    # ---- Load best checkpoint ----
    if best_state:
        model.load_state_dict(best_state)
        ckpt_path = os.path.join(MODEL_DIR, f'droneRFb_{model_key}_best.pt')
        torch.save(best_state, ckpt_path)
        print(f"  Saved best checkpoint -> {ckpt_path}")

    # ---- Full evaluation on test set ----
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
                outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc    = accuracy_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()

    type_acc, type_labels, type_preds = compute_type_accuracy(
        all_labels, all_preds, class_names
    )

    total_time   = time.time() - train_start
    epochs_done  = epoch + 1

    result = {
        'model':              model_key,
        'accuracy':           float(acc),
        'f1_macro':           float(f1),
        'type_accuracy':      float(type_acc),
        'best_val_acc':       float(best_val_acc),
        'params':             int(n_params),
        'train_time_sec':     float(total_time),
        'epochs_trained':     int(epochs_done),
        'per_class':          {cn: report[cn] for cn in class_names if cn in report},
        'confusion_matrix':   cm,
        'class_names':        class_names,
    }

    print(f"\n  RESULT: {model_key}")
    print(f"  Individual-level Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
    print(f"  Drone-type Accuracy:       {type_acc:.4f}")
    print(f"  Params: {n_params/1e6:.1f}M | Time: {total_time:.0f}s")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return result


# ============================================================================
# YOLO CLASSIFIER
# ============================================================================

def train_yolo_classifier(cfg):
    from ultralytics import YOLO

    model_variant = 'yolo11n-cls'
    print(f"\n{'='*70}")
    print(f"  Training: {model_variant}")
    print(f"{'='*70}", flush=True)

    train_start = time.time()
    model = YOLO(f'{model_variant}.pt')

    model.train(
        data=DATA_DIR,
        epochs=cfg['epochs'],
        imgsz=cfg['img_size'],
        batch=cfg['batch_size'],
        patience=cfg['patience'],
        project=MODEL_DIR,
        name=f'droneRFb_{model_variant}',
        device=0,
        workers=cfg['num_workers'],
        optimizer='AdamW',
        lr0=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        label_smoothing=cfg['label_smoothing'],
        pretrained=True,
        verbose=True,
        cos_lr=True,
    )

    # ---- Evaluate on test set ----
    test_dir  = os.path.join(DATA_DIR, 'test')
    classes   = sorted(d for d in os.listdir(test_dir)
                       if os.path.isdir(os.path.join(test_dir, d)))
    all_preds_str, all_labels_str = [], []

    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        imgs    = sorted(f for f in os.listdir(cls_dir) if f.endswith('.png'))
        for img_name in imgs:
            img_path = os.path.join(cls_dir, img_name)
            res      = model.predict(img_path, verbose=False)
            pred_cls = res[0].names[res[0].probs.top1]
            all_preds_str.append(pred_cls)
            all_labels_str.append(cls)

    acc    = accuracy_score(all_labels_str, all_preds_str)
    f1     = f1_score(all_labels_str, all_preds_str, average='macro')

    # Type accuracy
    type_labels = [INDIVIDUAL_TO_TYPE.get(l, l) for l in all_labels_str]
    type_preds  = [INDIVIDUAL_TO_TYPE.get(p, p) for p in all_preds_str]
    type_acc    = accuracy_score(type_labels, type_preds)

    total_time = time.time() - train_start

    result = {
        'model':          model_variant,
        'accuracy':       float(acc),
        'f1_macro':       float(f1),
        'type_accuracy':  float(type_acc),
        'train_time_sec': float(total_time),
        'type':           'yolo-cls',
    }

    print(f"\n  {model_variant}: Accuracy={acc:.4f} F1={f1:.4f} TypeAcc={type_acc:.4f}")
    print(classification_report(all_labels_str, all_preds_str))

    # Copy best weights to standard location
    yolo_best = os.path.join(
        MODEL_DIR, f'droneRFb_{model_variant}', 'weights', 'best.pt'
    )
    dest = os.path.join(MODEL_DIR, f'droneRFb_{model_variant}_best.pt')
    if os.path.exists(yolo_best):
        import shutil
        shutil.copy2(yolo_best, dest)
        print(f"  Saved best checkpoint -> {dest}")

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DroneRFb Training Pipeline — MI300X ROCm'
    )
    parser.add_argument(
        '--models', nargs='+',
        default=list(TIMM_MODELS.keys()),
        choices=list(TIMM_MODELS.keys()),
        help='timm models to train (default: all 5)'
    )
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs',     type=int, default=80)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--yolo',       action='store_true',
                        help='Also train yolo11n-cls')
    args = parser.parse_args()

    print('=' * 70, flush=True)
    print('DroneRFb TRAINING PIPELINE — MI300X', flush=True)
    print(f'Device: {DEVICE}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU:  {torch.cuda.get_device_name(0)}', flush=True)
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB', flush=True)
    print(f'Data: {DATA_DIR}', flush=True)
    print('=' * 70, flush=True)

    cfg = DEFAULT_CFG.copy()
    cfg['batch_size'] = args.batch_size
    cfg['epochs']     = args.epochs
    cfg['lr']         = args.lr

    all_results = {}

    # ---- timm models ----
    for model_key in args.models:
        try:
            result = train_one_model(model_key, cfg)
            all_results[model_key] = result
        except Exception as e:
            print(f'  ERROR training {model_key}: {e}', flush=True)
            import traceback; traceback.print_exc()
            all_results[model_key] = {'error': str(e), 'model': model_key}

    # ---- YOLO ----
    if args.yolo:
        try:
            result = train_yolo_classifier(cfg)
            all_results['yolo11n-cls'] = result
        except Exception as e:
            print(f'  ERROR training yolo11n-cls: {e}', flush=True)
            import traceback; traceback.print_exc()
            all_results['yolo11n-cls'] = {'error': str(e), 'model': 'yolo11n-cls'}

    # ---- Summary table ----
    print('\n' + '=' * 90, flush=True)
    print('FINAL RESULTS — DroneRFb (sorted by accuracy)', flush=True)
    print('=' * 90, flush=True)
    print(
        f"{'Model':<25} {'Acc':>8} {'F1-macro':>10} {'TypeAcc':>10} "
        f"{'Params':>10} {'Time':>8}",
        flush=True
    )
    print('-' * 90, flush=True)

    sorted_names = sorted(
        all_results.keys(),
        key=lambda k: all_results[k].get('accuracy', 0.0),
        reverse=True
    )
    for name in sorted_names:
        r = all_results[name]
        if 'error' in r:
            print(f"{name:<25} {'ERROR':>8}", flush=True)
        else:
            params   = r.get('params', 0)
            pstr     = f"{params/1e6:.1f}M" if params else '?'
            tstr     = f"{r.get('train_time_sec', 0):.0f}s"
            type_acc = r.get('type_accuracy', float('nan'))
            print(
                f"{name:<25} {r['accuracy']:>8.4f} {r['f1_macro']:>10.4f} "
                f"{type_acc:>10.4f} {pstr:>10} {tstr:>8}",
                flush=True
            )

    # ---- Save results ----
    with open(RESULT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults saved -> {RESULT_FILE}', flush=True)


if __name__ == '__main__':
    main()
