#!/usr/bin/env python3
"""
RFUAV Full Training Pipeline — PyTorch + ROCm on MI300X
========================================================
Trains multiple SOTA architectures on RFUAV spectrogram data:
  1. RFUAV baselines: ResNet18, ResNet50, ViT-L-16, ViT-B-32
  2. Our best from RTL-ML: ConvNeXt-Base, YOLOv11-cls
  3. SOTA models: Swin-V2-B, EfficientNet-V2-L, DeiT-III
  4. RFML-MoE inspired: Multi-model confidence ensemble

Designed for MI300X (206GB VRAM) — large batch sizes, BF16, torch.compile.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# ============================================================================
# CONFIG
# ============================================================================

SPEC_DIR = '/home/rax/mtp/spectrograms'
MODEL_DIR = '/home/rax/mtp/models'
RESULT_DIR = '/home/rax/mtp/results'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MI300X-optimized defaults
DEFAULT_CFG = {
    'batch_size': 128,       # MI300X can do 256+ but be conservative
    'img_size': 224,         # Standard for timm models
    'epochs': 100,
    'lr': 1e-4,
    'weight_decay': 0.01,
    'label_smoothing': 0.1,
    'patience': 20,
    'num_workers': 8,
    'amp': True,             # BF16 mixed precision
    'compile': False,        # torch.compile (set True for prod runs)
}

# Models to train — name maps to timm model ID or 'yolo'
MODELS = {
    # RFUAV paper baselines
    'resnet18':          {'timm': 'resnet18',             'img_size': 224},
    'resnet50':          {'timm': 'resnet50',             'img_size': 224},
    'vit_b_32':          {'timm': 'vit_base_patch32_224', 'img_size': 224},
    'vit_l_16':          {'timm': 'vit_large_patch16_224','img_size': 224},
    # Our RTL-ML winners
    'convnext_base':     {'timm': 'convnext_base',        'img_size': 224},
    'convnext_large':    {'timm': 'convnext_large',       'img_size': 224},
    # SOTA
    'swin_v2_base':      {'timm': 'swinv2_base_window12to16_192to256', 'img_size': 256},
    'efficientnetv2_l':  {'timm': 'tf_efficientnetv2_l',  'img_size': 224},
    'deit3_base':        {'timm': 'deit3_base_patch16_224','img_size': 224},
    'maxvit_base':       {'timm': 'maxvit_base_tf_224',   'img_size': 224},
    'eva02_base':        {'timm': 'eva02_base_patch14_224','img_size': 224},
    # Lightweight (edge deployment)
    'mobilenetv3_large': {'timm': 'mobilenetv3_large_100','img_size': 224},
    'efficientnet_b0':   {'timm': 'efficientnet_b0',      'img_size': 224},
}


# ============================================================================
# DATA
# ============================================================================

def get_transforms(img_size, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
        os.path.join(SPEC_DIR, 'train'),
        transform=get_transforms(img_size, is_train=True)
    )
    val_ds = datasets.ImageFolder(
        os.path.join(SPEC_DIR, 'val'),
        transform=get_transforms(img_size, is_train=False)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes
    
    print(f"  Train: {len(train_ds)} images, {num_classes} classes")
    print(f"  Val:   {len(val_ds)} images")
    print(f"  Classes: {class_names}")
    
    return train_loader, val_loader, num_classes, class_names


# ============================================================================
# MODELS
# ============================================================================

def create_model(model_name, num_classes, pretrained=True):
    """Create model from timm registry."""
    cfg = MODELS[model_name]
    timm_name = cfg['timm']
    
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {timm_name} ({n_params/1e6:.1f}M params)")
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_one_model(model_name, cfg=None):
    """Train a single model end-to-end."""
    if cfg is None:
        cfg = DEFAULT_CFG.copy()
    
    model_cfg = MODELS[model_name]
    img_size = model_cfg.get('img_size', cfg['img_size'])
    
    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"{'='*70}")
    
    # Data
    train_loader, val_loader, num_classes, class_names = load_data(
        img_size, cfg['batch_size'], cfg['num_workers']
    )
    
    # Model
    model = create_model(model_name, num_classes)
    model = model.to(DEVICE)
    
    if cfg.get('compile'):
        model = torch.compile(model)
        print("  torch.compile enabled")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                   weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    scaler = torch.amp.GradScaler(enabled=cfg['amp'])
    
    # Training loop
    best_val_acc = 0
    best_state = None
    no_improve = 0
    train_start = time.time()
    
    for epoch in range(cfg['epochs']):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
                    outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = '*'
        else:
            no_improve += 1
            marker = ''
        
        if (epoch + 1) % 5 == 0 or marker == '*':
            elapsed = time.time() - train_start
            print(f"  Epoch {epoch+1:3d}: train={train_acc:.3f} val={val_acc:.3f} "
                  f"loss={train_loss/max(train_total,1):.4f} [{elapsed:.0f}s] {marker}", flush=True)
        
        if no_improve >= cfg['patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best and evaluate
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(MODEL_DIR, f'{model_name}_best.pt'))
    
    # Final evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
                outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    total_time = time.time() - train_start
    n_params = sum(p.numel() for p in model.parameters())
    
    result = {
        'model': model_name,
        'accuracy': acc,
        'f1_macro': f1,
        'best_val_acc': best_val_acc,
        'params': n_params,
        'train_time_sec': total_time,
        'epochs_trained': epoch + 1,
        'per_class': {cn: report[cn] for cn in class_names if cn in report},
    }
    
    print(f"\n  RESULT: {model_name}")
    print(f"  Accuracy: {acc:.3f} | F1-macro: {f1:.3f} | Params: {n_params/1e6:.1f}M | Time: {total_time:.0f}s")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Save result
    with open(os.path.join(RESULT_DIR, f'{model_name}_result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return result


# ============================================================================
# YOLO CLASSIFIER
# ============================================================================

def train_yolo_classifier(model_variant='yolo11n-cls', img_size=224, epochs=100):
    """Train YOLO in classification mode."""
    from ultralytics import YOLO
    
    print(f"\n{'='*70}")
    print(f"  Training: {model_variant}")
    print(f"{'='*70}")
    
    model = YOLO(f'{model_variant}.pt')
    results = model.train(
        data=SPEC_DIR,
        epochs=epochs,
        imgsz=img_size,
        batch=128,
        patience=20,
        project=MODEL_DIR,
        name=model_variant,
        device=0,
        workers=8,
        optimizer='AdamW',
        lr0=1e-3,
        weight_decay=0.01,
        label_smoothing=0.1,
        pretrained=True,
        verbose=True,
    )
    
    # Evaluate
    val_dir = os.path.join(SPEC_DIR, 'val')
    classes = sorted(os.listdir(val_dir))
    all_preds, all_labels = [], []
    
    for cls in classes:
        cls_dir = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for img_name in sorted(os.listdir(cls_dir)):
            if not img_name.endswith('.png'):
                continue
            img_path = os.path.join(cls_dir, img_name)
            res = model.predict(img_path, verbose=False)
            pred_cls = res[0].names[res[0].probs.top1]
            all_preds.append(pred_cls)
            all_labels.append(cls)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    result = {
        'model': model_variant,
        'accuracy': acc,
        'f1_macro': f1,
        'type': 'yolo-cls',
    }
    
    print(f"\n  {model_variant}: Accuracy={acc:.3f} F1={f1:.3f}")
    print(classification_report(all_labels, all_preds))
    
    with open(os.path.join(RESULT_DIR, f'{model_variant}_result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='RFUAV Training Pipeline')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to train (default: all)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--yolo', action='store_true', help='Also train YOLO classifiers')
    args = parser.parse_args()
    
    print("=" * 70, flush=True)
    print("RFUAV TRAINING PIPELINE — MI300X", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB", flush=True)
    print("=" * 70, flush=True)
    
    cfg = DEFAULT_CFG.copy()
    cfg['batch_size'] = args.batch_size
    cfg['epochs'] = args.epochs
    cfg['lr'] = args.lr
    cfg['compile'] = args.compile
    
    models_to_train = args.models or list(MODELS.keys())
    
    all_results = {}
    
    # Train timm models
    for model_name in models_to_train:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}, skipping")
            continue
        try:
            result = train_one_model(model_name, cfg)
            all_results[model_name] = result
        except Exception as e:
            print(f"  ERROR training {model_name}: {e}", flush=True)
            import traceback; traceback.print_exc()
            all_results[model_name] = {'error': str(e)}
    
    # Train YOLO classifiers
    if args.yolo:
        for yolo_model in ['yolo11n-cls', 'yolo11s-cls', 'yolov8n-cls']:
            try:
                result = train_yolo_classifier(yolo_model, epochs=args.epochs)
                all_results[yolo_model] = result
            except Exception as e:
                print(f"  ERROR training {yolo_model}: {e}", flush=True)
                all_results[yolo_model] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 70, flush=True)
    print("FINAL RESULTS SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1-macro':>10} {'Params':>12} {'Time':>10}", flush=True)
    print("-" * 70, flush=True)
    
    for name in sorted(all_results.keys(), key=lambda k: all_results[k].get('accuracy', 0), reverse=True):
        r = all_results[name]
        if 'error' in r:
            print(f"{name:<25} {'ERROR':>10}", flush=True)
        else:
            params = r.get('params', 0)
            pstr = f"{params/1e6:.1f}M" if params else "?"
            tstr = f"{r.get('train_time_sec', 0):.0f}s"
            print(f"{name:<25} {r['accuracy']:>10.3f} {r['f1_macro']:>10.3f} {pstr:>12} {tstr:>10}", flush=True)
    
    # Save all results
    with open(os.path.join(RESULT_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {RESULT_DIR}/all_results.json", flush=True)


if __name__ == '__main__':
    main()
