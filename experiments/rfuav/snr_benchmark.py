#!/usr/bin/env python3
"""
SNR Benchmark Evaluation — Test models across -20 to +20 dB SNR
================================================================
Adds AWGN to spectrograms at various SNR levels and evaluates
each trained model. Produces SNR-accuracy curves.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import accuracy_score, f1_score

SPEC_DIR = '/home/rax/mtp/spectrograms/val'
MODEL_DIR = '/home/rax/mtp/models'
RESULT_DIR = '/home/rax/mtp/results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SNR_LEVELS = list(range(-20, 22, 2))  # -20 to +20 dB in 2 dB steps


class SNRAugmentedDataset(Dataset):
    """Load spectrogram images and add Gaussian noise at specified SNR."""
    def __init__(self, root_dir, snr_db, img_size=224):
        self.samples = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root_dir) 
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for f in sorted(os.listdir(cls_dir)):
                if f.endswith('.png'):
                    self.samples.append(os.path.join(cls_dir, f))
                    self.labels.append(self.class_to_idx[cls])
        
        self.snr_db = snr_db
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        tensor = self.transform(img)  # 0-1 range
        
        # Add noise at specified SNR
        if self.snr_db is not None:
            sig_power = tensor.pow(2).mean()
            noise_power = sig_power / (10 ** (self.snr_db / 10))
            noise = torch.randn_like(tensor) * noise_power.sqrt()
            tensor = (tensor + noise).clamp(0, 1)
        
        tensor = self.normalize(tensor)
        return tensor, self.labels[idx]


def evaluate_model_at_snr(model, snr_db, img_size=224, batch_size=64):
    """Evaluate a model at a specific SNR level."""
    dataset = SNRAugmentedDataset(SPEC_DIR, snr_db, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1


def benchmark_model(model_name, img_size=224):
    """Run full SNR benchmark for a model."""
    ckpt = os.path.join(MODEL_DIR, f'{model_name}_best.pt')
    if not os.path.exists(ckpt):
        print(f"  No checkpoint for {model_name}, skipping", flush=True)
        return None
    
    # Determine num_classes from dataset
    classes = sorted([d for d in os.listdir(SPEC_DIR) if os.path.isdir(os.path.join(SPEC_DIR, d))])
    num_classes = len(classes)
    
    # Load model
    from train_rfuav import MODELS
    timm_name = MODELS[model_name]['timm']
    model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model = model.to(DEVICE)
    
    print(f"\n  Benchmarking {model_name} across {len(SNR_LEVELS)} SNR levels:", flush=True)
    
    results = {}
    for snr in SNR_LEVELS:
        acc, f1 = evaluate_model_at_snr(model, snr, img_size)
        results[snr] = {'accuracy': acc, 'f1_macro': f1}
        print(f"    SNR {snr:+3d} dB: acc={acc:.3f} f1={f1:.3f}", flush=True)
    
    # Also evaluate clean (no noise)
    acc_clean, f1_clean = evaluate_model_at_snr(model, None, img_size)
    results['clean'] = {'accuracy': acc_clean, 'f1_macro': f1_clean}
    print(f"    Clean:     acc={acc_clean:.3f} f1={f1_clean:.3f}", flush=True)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=None)
    args = parser.parse_args()
    
    print("=" * 60, flush=True)
    print("SNR BENCHMARK EVALUATION", flush=True)
    print(f"SNR range: {SNR_LEVELS[0]} to {SNR_LEVELS[-1]} dB", flush=True)
    print("=" * 60, flush=True)
    
    # Find available models
    available = [f.replace('_best.pt', '') for f in os.listdir(MODEL_DIR) if f.endswith('_best.pt')]
    models = args.models or available
    print(f"Models to benchmark: {models}", flush=True)
    
    all_results = {}
    for model_name in models:
        try:
            result = benchmark_model(model_name)
            if result:
                all_results[model_name] = result
        except Exception as e:
            print(f"  ERROR: {model_name}: {e}", flush=True)
    
    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("SNR BENCHMARK SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    
    header = f"{'Model':<25}"
    for snr in [-20, -10, 0, 10, 20]:
        header += f" {snr:+3d}dB"
    header += " Clean"
    print(header, flush=True)
    print("-" * 80, flush=True)
    
    for name in sorted(all_results.keys(), key=lambda k: all_results[k].get('clean', {}).get('accuracy', 0), reverse=True):
        r = all_results[name]
        row = f"{name:<25}"
        for snr in [-20, -10, 0, 10, 20]:
            acc = r.get(snr, {}).get('accuracy', 0)
            row += f" {acc:.3f}"
        row += f" {r.get('clean', {}).get('accuracy', 0):.3f}"
        print(row, flush=True)
    
    with open(os.path.join(RESULT_DIR, 'snr_benchmark.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {RESULT_DIR}/snr_benchmark.json", flush=True)

if __name__ == '__main__':
    main()
