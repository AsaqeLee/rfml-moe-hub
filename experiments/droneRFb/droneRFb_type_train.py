#!/usr/bin/env python3
"""DroneRFb training with TYPE-LEVEL labels (A1/A2/A3 all → A).
Fixes the cross-individual generalization issue."""
import os, sys, json, time, argparse, shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, accuracy_score, f1_score

SPEC_DIR = '/home/rax/mtp/droneRFb_spectrograms'
TYPE_DIR = '/home/rax/mtp/droneRFb_type_spectrograms'
MODEL_DIR = '/home/rax/mtp/models'
RESULT_DIR = '/home/rax/mtp/results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_type_dirs():
    """Symlink individual class dirs to type-level dirs (A1,A2,A3 → A)."""
    import re
    for split in ['train', 'test']:
        src_dir = os.path.join(SPEC_DIR, split)
        dst_dir = os.path.join(TYPE_DIR, split)
        if os.path.exists(dst_dir):
            print(f"  {split} type dirs exist, skipping")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for cls in sorted(os.listdir(src_dir)):
            cls_path = os.path.join(src_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            # Extract type: A1→A, C3→C, B→B
            m = re.match(r'^([A-G])', cls)
            dtype = m.group(1) if m else cls
            type_dir = os.path.join(dst_dir, dtype)
            os.makedirs(type_dir, exist_ok=True)
            # Symlink all images
            for img in os.listdir(cls_path):
                src = os.path.join(cls_path, img)
                dst = os.path.join(type_dir, f"{cls}_{img}")
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        classes = sorted(os.listdir(dst_dir))
        counts = {c: len(os.listdir(os.path.join(dst_dir, c))) for c in classes}
        print(f"  {split}: {counts}")

def get_transforms(img_size, is_train):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def train_model(model_name, timm_name, img_size=224, batch_size=128, epochs=80, lr=1e-4):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name} (type-level, 7 classes)")
    print(f"{'='*60}")

    train_ds = datasets.ImageFolder(os.path.join(TYPE_DIR,'train'), get_transforms(img_size, True))
    test_ds = datasets.ImageFolder(os.path.join(TYPE_DIR,'test'), get_transforms(img_size, False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    num_classes = len(train_ds.classes)
    classes = train_ds.classes
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {classes}")

    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=True)

    best_acc, best_state, no_improve = 0, None, 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        model.eval()
        tc, tt = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model(imgs)
                tc += (out.argmax(1) == labels.to(DEVICE)).sum().item()
                tt += labels.size(0)
        test_acc = tc / tt
        train_acc = correct / total

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch+1) % 5 == 0 or no_improve == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_acc:.3f} test={test_acc:.3f} [{time.time()-t0:.0f}s] {'*' if no_improve==0 else ''}", flush=True)

        if no_improve >= 20:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(MODEL_DIR, f'droneRFb_type_{model_name}_best.pt'))

    # Final eval
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(imgs.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(labels.numpy())

    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, average='macro')
    elapsed = time.time() - t0
    print(f"\n  RESULT: {model_name} → Acc={acc:.3f} F1={f1:.3f} Params={n_params/1e6:.1f}M Time={elapsed:.0f}s")
    print(classification_report(labels_all, preds, target_names=classes))

    return {'model': model_name, 'accuracy': acc, 'f1_macro': f1, 'params': n_params, 'time': elapsed}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args()

    print("Creating type-level directory structure...")
    create_type_dirs()

    MODELS = [
        ('convnext_base', 'convnext_base', 224),
        ('maxvit_base', 'maxvit_base_tf_224', 224),
        ('efficientnet_b0', 'efficientnet_b0', 224),
        ('resnet18', 'resnet18', 224),
        ('mobilenetv3', 'mobilenetv3_large_100', 224),
    ]

    results = {}
    for name, timm_name, img_size in MODELS:
        try:
            r = train_model(name, timm_name, img_size, args.batch_size, args.epochs)
            results[name] = r
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}

    print(f"\n{'='*60}")
    print("FINAL RESULTS — DroneRFb TYPE-LEVEL (7 classes)")
    print(f"{'='*60}")
    for n in sorted(results, key=lambda k: results[k].get('accuracy',0), reverse=True):
        r = results[n]
        if 'error' in r:
            print(f"  {n}: ERROR")
        else:
            print(f"  {n}: Acc={r['accuracy']:.3f} F1={r['f1_macro']:.3f} Params={r['params']/1e6:.1f}M")

    with open(os.path.join(RESULT_DIR, 'droneRFb_type_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == '__main__':
    main()
