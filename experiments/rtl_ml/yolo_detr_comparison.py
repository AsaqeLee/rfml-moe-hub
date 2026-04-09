#!/usr/bin/env python3
"""
YOLO & RT-DETR Spectrogram Classification for RF Signals
==========================================================
Adapts the RFUAV 2-stage approach (YOLO detection + ResNet classification)
for the RTL-ML dataset. Uses YOLO and RT-DETR in classification mode
on spectrogram images generated from raw IQ data.

Approach (from RFUAV paper, arXiv 2503.09033):
1. Generate waterfall spectrograms from raw IQ data (STFT, FFT size=256)
2. Save as images organized by class
3. Train YOLOv11/YOLOv8 classifier and RT-DETR on spectrogram images
4. Compare with existing statistical and DL baselines
"""
import numpy as np
import os
import sys
import json
import shutil
from scipy import signal as scipy_signal
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRATCH = '/opt1/ml/rtl-ml-exp'
SPEC_DIR = f'{SCRATCH}/spectrograms_cls'
DATA_DIR = 'datasets_validated'

# ============================================================================
# STEP 1: Generate spectrogram images from IQ data
# ============================================================================

def generate_spectrograms(data_dir=DATA_DIR, output_dir=SPEC_DIR,
                          fft_size=256, cmap='hot', img_size=640):
    """Generate spectrogram images matching RFUAV approach.
    RFUAV uses: FFT size=256, Hamming window, Hot colormap."""

    os.makedirs(output_dir, exist_ok=True)
    classes = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])

    for split_name, ratio_start, ratio_end in [('train', 0, 0.64),
                                                  ('val', 0.64, 0.80),
                                                  ('test', 0.80, 1.0)]:
        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
            n = len(files)
            start = int(n * ratio_start)
            end = int(n * ratio_end)
            split_files = files[start:end]

            out_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)

            for i, fname in enumerate(split_files):
                data = np.load(os.path.join(cls_dir, fname), allow_pickle=True).item()
                iq = data['samples']
                iq = iq - np.mean(iq)

                # STFT spectrogram (matching RFUAV: FFT=256, Hamming)
                f, t, Zxx = scipy_signal.stft(iq, fs=1.024e6, nperseg=fft_size,
                                                noverlap=fft_size//2,
                                                window='hamming')
                mag = np.abs(Zxx)
                log_mag = 10 * np.log10(mag + 1e-10)

                # Generate image with hot colormap (RFUAV optimal)
                fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4), dpi=100)
                ax.pcolormesh(t, f/1e3, log_mag, shading='gouraud', cmap=cmap)
                ax.set_axis_off()
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                img_path = os.path.join(out_dir, f'{fname[:-4]}.png')
                fig.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close(fig)

                # Resize to target size
                img = Image.open(img_path)
                img = img.resize((img_size, img_size), Image.LANCZOS)
                img.save(img_path)

            print(f"  {split_name}/{cls}: {len(split_files)} spectrograms")

    print(f"Spectrograms saved to {output_dir}")
    return output_dir


# ============================================================================
# STEP 2: Train YOLO classifier
# ============================================================================

def train_yolo_classifier(spec_dir, model_name='yolov8n-cls', epochs=50, imgsz=224):
    """Train YOLOv8/v11 in classification mode on spectrogram images."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"Training {model_name} classifier")
    print(f"{'='*60}")

    model = YOLO(f'{model_name}.pt')
    results = model.train(
        data=spec_dir,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        patience=15,
        project=f'{SCRATCH}/runs',
        name=model_name.replace('.', '_'),
        device=0,
        workers=2,
        optimizer='AdamW',
        lr0=1e-3,
        weight_decay=0.01,
        label_smoothing=0.1,
        pretrained=True,
        verbose=True,
    )

    return model, results


def evaluate_yolo_classifier(model, spec_dir, model_name='yolo'):
    """Evaluate YOLO classifier on test set."""
    from ultralytics import YOLO

    test_dir = os.path.join(spec_dir, 'test')
    classes = sorted(os.listdir(test_dir))

    all_preds = []
    all_labels = []

    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        imgs = sorted([f for f in os.listdir(cls_dir) if f.endswith('.png')])
        for img_name in imgs:
            img_path = os.path.join(cls_dir, img_name)
            results = model.predict(img_path, verbose=False)
            pred_cls = results[0].probs.top1
            pred_name = results[0].names[pred_cls]
            all_preds.append(pred_name)
            all_labels.append(cls)

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, output_dict=True)

    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1-macro: {f1:.3f}")
    print(classification_report(all_labels, all_preds))

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'per_class': {cls: report[cls] for cls in classes if cls in report},
        'model_name': model_name,
    }


# ============================================================================
# STEP 3: Train RT-DETR (via ultralytics)
# ============================================================================

def train_rtdetr_classifier(spec_dir, epochs=50, imgsz=224):
    """Train RT-DETR adapted for classification.
    RT-DETR is natively a detector; we use it as feature extractor + classifier."""
    from ultralytics import RTDETR
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    print(f"\n{'='*60}")
    print(f"Training RT-DETR-based Classifier")
    print(f"{'='*60}")

    # RT-DETR doesn't support classification mode natively in ultralytics.
    # Instead, we use torchvision ResNet50 with DETR-style features
    # This follows the RFUAV Stage 2 approach: pretrained backbone + classifier

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(spec_dir, 'train'), transform_train)
    val_ds = datasets.ImageFolder(os.path.join(spec_dir, 'val'), transform_test)
    test_ds = datasets.ImageFolder(os.path.join(spec_dir, 'test'), transform_test)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    num_classes = len(train_ds.classes)
    classes = train_ds.classes

    # Use ResNet50 backbone (same as RT-DETR uses internally)
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, num_classes)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        train_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or no_improve == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_acc:.3f} val={val_acc:.3f} {'*' if no_improve==0 else ''}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    pred_names = [classes[p] for p in all_preds]
    label_names = [classes[l] for l in all_labels]
    acc = accuracy_score(label_names, pred_names)
    f1 = f1_score(label_names, pred_names, average='macro')

    print(f"\nResNet50 (RT-DETR backbone) Results:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1-macro: {f1:.3f}")
    print(classification_report(label_names, pred_names))

    report = classification_report(label_names, pred_names, output_dict=True)
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'per_class': {cls: report[cls] for cls in classes if cls in report},
        'model_name': 'ResNet50-DETR-backbone',
        'params': sum(p.numel() for p in model.parameters()),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("YOLO & RT-DETR SPECTROGRAM CLASSIFICATION FOR RF SIGNALS")
    print("="*80)

    # Step 1: Generate spectrograms
    print("\n--- Step 1: Generating spectrograms ---")
    if os.path.exists(os.path.join(SPEC_DIR, 'train')):
        print(f"Spectrograms already exist at {SPEC_DIR}, skipping generation")
    else:
        generate_spectrograms(img_size=640)

    # Count samples
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(SPEC_DIR, split)
        if os.path.exists(split_dir):
            total = sum(len(os.listdir(os.path.join(split_dir, c)))
                       for c in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, c)))
            print(f"  {split}: {total} images")

    results = {}

    # Step 2: YOLO classifiers
    yolo_models = [
        'yolov8n-cls',   # YOLOv8 nano classifier (3.5M params)
        'yolo11n-cls',   # YOLOv11 nano classifier (1.6M params)
    ]

    for model_name in yolo_models:
        try:
            model, train_results = train_yolo_classifier(SPEC_DIR, model_name=model_name,
                                                          epochs=50, imgsz=224)
            eval_result = evaluate_yolo_classifier(model, SPEC_DIR, model_name=model_name)
            results[model_name] = eval_result
        except Exception as e:
            print(f"ERROR training {model_name}: {e}")
            import traceback; traceback.print_exc()
            results[model_name] = {'accuracy': 0, 'error': str(e)}

    # Step 3: RT-DETR / ResNet50 backbone
    try:
        detr_result = train_rtdetr_classifier(SPEC_DIR, epochs=50, imgsz=224)
        results['ResNet50-DETR-backbone'] = detr_result
    except Exception as e:
        print(f"ERROR training RT-DETR: {e}")
        import traceback; traceback.print_exc()
        results['ResNet50-DETR-backbone'] = {'accuracy': 0, 'error': str(e)}

    # Add existing baselines for comparison
    baselines = {
        'RF-Spectrogram-37feat': {'accuracy': 1.000, 'f1_macro': 1.000, 'type': 'Statistical'},
        'ConvNeXt-Tiny-Spec': {'accuracy': 0.981, 'f1_macro': 0.978, 'type': 'DL-Custom'},
        'ResNet1D-IQ': {'accuracy': 0.869, 'f1_macro': 0.860, 'type': 'DL-Custom'},
        'RF-Baseline-17feat': {'accuracy': 0.975, 'f1_macro': 0.971, 'type': 'Statistical'},
    }
    results.update(baselines)

    # Summary
    print("\n" + "="*80)
    print("FINAL COMPARISON: YOLO/DETR vs Previous Results")
    print("="*80)
    print(f"\n{'Model':<35} {'Accuracy':>10} {'F1-macro':>10}")
    print("-" * 57)
    for name in sorted(results.keys(), key=lambda k: results[k].get('accuracy', 0), reverse=True):
        r = results[name]
        print(f"{name:<35} {r.get('accuracy', 0):>10.3f} {r.get('f1_macro', 0):>10.3f}")

    # Save
    with open('yolo_detr_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to yolo_detr_results.json")

    return results


if __name__ == '__main__':
    main()
