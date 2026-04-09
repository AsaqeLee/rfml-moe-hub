#!/usr/bin/env python3
"""Re-run only the models that failed due to dataset move."""
import sys, os, json
sys.path.insert(0, '/opt1/ml/pylibs')

import torch
import numpy as np
from nn_comparison import (
    RFDataset, create_splits, train_model, evaluate_model, count_params,
    FTTransformerHOS, DilatedTCN, ResNet1D, SEResNet1D, InceptionTime1D,
    DEVICE
)
from torch.utils.data import DataLoader, Subset

print(f"Device: {DEVICE}")
dataset = RFDataset(augment=False, iq_len=32768, spec_size=128, cyclo_dim=512)
train_idx, val_idx, test_idx = create_splits(dataset)
aug_dataset = RFDataset(augment=True, iq_len=32768, spec_size=128, cyclo_dim=512)

train_loader = DataLoader(Subset(aug_dataset, train_idx), batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False, num_workers=0)

classes = dataset.classes
nc = len(classes)

models = {
    'FT-Transformer-HOS': FTTransformerHOS(num_features=20, num_classes=nc),
    'Dilated-TCN-Cyclo': DilatedTCN(in_dim=512, num_classes=nc),
    'ResNet1D': ResNet1D(num_classes=nc),
    'SE-ResNet1D': SEResNet1D(num_classes=nc),
    'InceptionTime-1D': InceptionTime1D(num_classes=nc),
}

results = {}
for name, model in models.items():
    n_params = count_params(model)
    print(f"\n{'='*60}")
    print(f"  {name} ({n_params:,} params)")
    print(f"{'='*60}")
    try:
        model, best_val = train_model(model, train_loader, val_loader, epochs=80, lr=1e-3, patience=15)
        print(f"  Best val acc: {best_val:.3f}")
        ev = evaluate_model(model, test_loader, classes)
        ev['params'] = n_params
        ev['best_val_acc'] = best_val
        results[name] = ev
        print(f"  Test accuracy: {ev['accuracy']:.3f}")
        print(f"  F1-macro: {ev['f1_macro']:.3f}")
        for cls in classes:
            if cls in ev['per_class']:
                r = ev['per_class'][cls]
                print(f"    {cls:15s}: P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results[name] = {'accuracy': 0, 'error': str(e)}

# Merge with existing results
existing = json.load(open('nn_comparison_results.json'))
existing.update(results)
with open('nn_comparison_results.json', 'w') as f:
    json.dump(existing, f, indent=2, default=str)

print("\n" + "="*60)
print("RE-RUN RESULTS")
print("="*60)
for name, r in results.items():
    print(f"  {name}: {r.get('accuracy', 0):.3f}")
print("\nMerged into nn_comparison_results.json")
