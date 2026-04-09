# RFUAV Full Implementation Plan — PyTorch + ROCm on MI300X

## Hardware

| Component | Spec |
|-----------|------|
| **GPU** | AMD Instinct MI300X (192 GB HBM3, 750W TDP) |
| **ROCm** | Installed, rocm-smi working |
| **Storage** | 625 GB available on /home/rax (697 GB total) |
| **Dataset** | 102 GB compressed (.rar), ~400-500 GB extracted |
| **Server** | rax@129.212.188.94 |

## Dataset: RFUAV

- **37 drone/RC types**, 1.3 TB raw IQ data (100 MSps, fp32 interleaved)
- **Frequencies**: 2.4 GHz and 5.8 GHz bands
- **Capture device**: USRP (Universal Software Radio Peripheral)
- Currently extracting to `/home/rax/mtp/raw/`

---

## Phase 0: Environment Setup

```
Server: rax@129.212.188.94
Working dir: /home/rax/mtp/
├── raw/                    # Extracted IQ data (37 drone folders)
├── RFUAV/                  # Cloned RFUAV codebase
├── spectrograms/           # Generated spectrogram images
│   ├── train/
│   └── val/
├── models/                 # Trained model checkpoints
├── results/                # Evaluation results & plots
├── configs/                # Training configs
└── rfuav_pipeline.py       # Our unified training script
```

**Dependencies** (installing in tmux `setup` session):
- PyTorch 2.x + ROCm 6.3
- torchvision, torchaudio
- numpy, scipy, scikit-learn, matplotlib, albumentations, opencv, pillow, pandas, seaborn, pyyaml, tqdm

---

## Phase 1: Data Preparation Pipeline

### 1.1 Raw IQ → Spectrogram Generation

Following RFUAV paper's optimal parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| FFT size (STFTP) | 256 | Paper's optimum (58.28% accuracy, boundary effect above) |
| Window | Hamming | Paper default |
| Colormap | Hot | Paper's best (58.16% vs 56.44% Parula) |
| Duration | 0.1s per frame | 10M samples per frame at 100 MSps |
| Image size | 640×640 | YOLO/classifier input standard |
| Overlap | 50% | Standard STFT overlap |

**Processing plan:**
- For each of 37 drone folders, read raw IQ binary (fp32 interleaved)
- Extract I/Q: `data_complex = data[::2] + 1j * data[1::2]`
- Segment into 0.1s chunks (10M samples each)
- STFT → log-magnitude spectrogram → Hot colormap → save PNG
- Split train/val: 80/20 random per drone type
- Expected output: ~50,000-200,000 spectrogram images

### 1.2 SNR Augmentation

For robustness evaluation (matching paper methodology):
- Add AWGN at SNR levels: -20, -18, -16, ..., +18, +20 dB (21 levels)
- Generate SNR-specific validation sets for benchmark evaluation
- Clean (high-SNR) data for training, noisy for validation/test

### 1.3 Data Augmentation (Training)

Using Albumentations (matching RFUAV codebase):
- AdvancedBlur, CLAHE, ColorJitter, GaussNoise, ISONoise, Sharpen
- Plus RF-aware: random SNR noise injection during training

---

## Phase 2: Reproduce RFUAV Baselines

### 2.1 Classification Models (Stage 2)

Reproduce the paper's 5-class results first, then scale to full 37-class:

| Model | Paper Accuracy (5-class) | Params | Priority |
|-------|--------------------------|--------|----------|
| ViT-L-16 | 56.44% overall, 98.55% high-SNR | 304M | High |
| ResNet18 | 54.78% overall, 99.93% high-SNR | 11M | High |
| ViT-B-32 | ~56%, 100% high-SNR | 88M | Medium |
| Swin-V2-T | Not reported | 28M | Medium |
| EfficientNet-B0 | Not reported | 5.3M | Medium |
| ConvNeXt-Tiny | Not reported (our best on RTL-ML) | 28M | High |
| MobileNet-V3-L | Not reported | 5.4M | Low (edge) |

**Training config** (matching paper):
- Optimizer: Adam, lr=0.0001
- Loss: CrossEntropyLoss
- Batch size: 64 (MI300X can handle much larger)
- Epochs: 100
- Image size: 640
- Pretrained: ImageNet weights

### 2.2 Detection Model (Stage 1)

- YOLOv5s for signal detection in spectrograms
- Train on labeled spectrogram bounding boxes
- Generate YOLO labels using `tools/label_generate.py`
- Evaluate: mAP@0.5, precision, recall

### 2.3 Two-Stage Pipeline

- Stage 1: YOLOv5s detects signal region in spectrogram
- Stage 2: Classification model identifies drone type from cropped region
- Evaluate end-to-end accuracy per SNR level

---

## Phase 3: Our Enhanced Models

### 3.1 Models from RTL-ML Experiments (Adapted for RFUAV)

Based on our findings (rtl-ml-exp), scale winning architectures:

| Model | RTL-ML Result | RFUAV Adaptation | Expected Benefit |
|-------|---------------|------------------|-----------------|
| **YOLOv11n-cls** | 99.4% | YOLOv11 classifier on RFUAV spectrograms | Transfer learning powerhouse |
| **ConvNeXt-Tiny** | 98.1% | ConvNeXt-Small/Base (more capacity for 37 classes) | Modern CNN, strong on spectrograms |
| **ResNet1D** | 86.9% | Direct IQ input (bypass spectrogram) | Raw signal processing |
| **Confidence-weighted ensemble** | 100% | Multi-model confidence routing | Best ensemble strategy |

### 3.2 RFML-MoE Full Implementation

Implement the complete MoE architecture from `/home/rax/exp/iq/rfml`:

```
                    ┌─────────────┐
                    │Expert Choice│
                    │   Router    │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────────┐ ┌───────────┐ ┌──────────────┐
    │ IQ Expert    │ │Spec Expert│ │ HOS Expert   │
    │ ResNet1D     │ │ConvNeXt-B │ │FT-Transformer│
    │ (updated)    │ │(updated)  │ │              │
    └──────┬───────┘ └─────┬─────┘ └──────┬───────┘
           └───────────────┼───────────────┘
                           ▼
                 ┌────────────────┐
                 │ Cross-Attention│
                 │    Fusion      │
                 └────────┬───────┘
                          ▼
                 ┌────────────────┐
                 │ 3-Level Head   │
                 │ Binary→Type→   │
                 │ Model          │
                 └────────────────┘
```

Changes from original RFML-MoE based on our experiments:
1. Replace SignalFormerIQ → ResNet1D (better at small-to-mid scale)
2. Replace EfficientNet → ConvNeXt-Base (better spectrogram accuracy)
3. Drop cyclostationary expert (worst performer, slow extraction)
4. Add confidence-weighted routing fallback
5. Add YOLOv11 as optional spectrogram expert

### 3.3 Progressive Training (4-Phase)

Adapted from RFML design for MI300X:

| Phase | Description | Epochs | LR | What Trains |
|-------|-------------|--------|-----|-------------|
| 1 | Self-supervised pretraining | 50 | 1e-3 | IQ: MAE, Spec: MoCo-v3 |
| 2 | Supervised curriculum (SNR easy→hard) | 75 | 5e-4 | Individual experts |
| 3 | Gating network training | 35 | 1e-4 | Router + fusion (experts frozen) |
| 4 | End-to-end fine-tuning | 15 | 1e-5 | All parameters |

**MI300X optimizations:**
- Batch size: 512+ (192GB VRAM handles it easily)
- Mixed precision: BF16 (MI300X native)
- `torch.compile()` for 2-3x speedup
- No gradient accumulation needed
- DataLoader: num_workers=16, pin_memory=True

---

## Phase 4: Evaluation & Benchmarking

### 4.1 Metrics

Per the RFUAV benchmark system:
- **Per-SNR accuracy**: -20 to +20 dB in 2 dB steps
- **Overall accuracy**: Weighted across all SNR levels
- **Per-class metrics**: Precision, recall, F1 per drone type
- **Top-k accuracy**: Top-1, Top-3, Top-5
- **Confusion matrix**: Per SNR level
- **mAP**: For detection stage

### 4.2 Experiment Matrix

| Experiment | Models | Classes | Input | Goal |
|------------|--------|---------|-------|------|
| E1: Reproduce paper | ViT-L-16, ResNet18 | 5 | Spectrograms (Hot, 256) | Match 56-58% baseline |
| E2: Our best classifiers | YOLOv11, ConvNeXt-B | 5 | Spectrograms | Beat paper baseline |
| E3: Full 37-class | Top models from E1/E2 | 37 | Spectrograms | First reported 37-class results |
| E4: Raw IQ models | ResNet1D, IQ-CNN-Trans | 5+37 | Raw IQ | Bypass spectrogram |
| E5: MoE ensemble | Full RFML-MoE | 5+37 | Multi-modal | Ultimate accuracy |
| E6: Colormap ablation | ConvNeXt-B | 5 | Hot/Parula/HSV/Autumn | Verify paper findings |
| E7: STFTP ablation | ConvNeXt-B | 5 | 64/128/256/512/1024 | Optimal resolution |
| E8: SNR robustness | All top models | 5+37 | Variable SNR | Low-SNR performance |
| E9: Two-stage pipeline | YOLOv5 + classifier | 37 | Spectrograms | End-to-end detection |

### 4.3 SNR Curriculum Evaluation

```
SNR Level   │  Expected Model Behavior
─────────── │  ────────────────────────────────
≥ +10 dB    │  All models ~99%+ (easy)
0 to +8 dB  │  Spectrogram models 70-90%
-8 to 0 dB  │  MoE advantage emerges (IQ + Spec complementary)
≤ -10 dB    │  Hard — only robust features survive (7-22%)
            │  → This is where MoE multi-modal excels
```

---

## Phase 5: Deliverables

1. **Trained model checkpoints** for all experiments
2. **SNR-accuracy curves** per model and per drone class
3. **Confusion matrices** at key SNR levels
4. **Comparison table**: RFUAV paper baselines vs our enhanced models
5. **Full 37-class benchmark** (not in original paper)
6. **MoE vs single-model** analysis on real drone data
7. **Edge deployment analysis**: Model size, inference time, accuracy trade-offs

---

## Implementation Timeline

| Step | Task | Dependencies | Est. Time |
|------|------|-------------|-----------|
| 0.1 | Extract all .rar files | None | Running (tmux `unrar`) |
| 0.2 | Install PyTorch ROCm + deps | None | Running (tmux `setup`) |
| 1.1 | Generate spectrograms (all 37 drones) | 0.1 | 2-4 hours |
| 1.2 | Create train/val splits + SNR augmentation | 1.1 | 1 hour |
| 2.1 | Train ResNet18 + ViT-L-16 (5-class, reproduce paper) | 0.2, 1.2 | 4-8 hours |
| 2.2 | Train YOLOv5 detection model | 1.2 | 4 hours |
| 3.1 | Train YOLOv11-cls + ConvNeXt-B (5-class) | 1.2 | 4 hours |
| 3.2 | Scale to 37-class for top models | 3.1 | 8 hours |
| 3.3 | Implement + train RFML-MoE | 3.2 | 12-24 hours |
| 4.1 | Full benchmark evaluation | All training | 4 hours |
| 5.1 | Generate reports, plots, comparison tables | 4.1 | 2 hours |

**Total estimated**: ~3-4 days (with MI300X, most training runs complete in hours not days)

---

## File Structure for Implementation

```python
# /home/rax/mtp/rfuav_pipeline.py — Main unified script
#
# Sections:
#   1. IQ Data Loading & Spectrogram Generation
#   2. Dataset Classes (PyTorch)
#   3. Model Definitions (classifiers + MoE)
#   4. Training Loop (ROCm optimized)
#   5. Evaluation & Benchmarking
#   6. Experiment Runner
```

---

## Key Design Decisions

### Why ConvNeXt over EfficientNet?
Our RTL-ML experiments showed ConvNeXt-Tiny (98.1%) significantly outperforms basic CNNs (89.4%) on spectrograms. ConvNeXt-Base with its larger capacity should handle 37 classes better than EfficientNet-B0/B2.

### Why drop the cyclostationary expert?
It scored 55% on RTL-ML (worst of all modalities) and SCF extraction is extremely slow (~1 sec/sample). With RFUAV's 100 MSps data, SCF computation would be prohibitive. Replace with a second spectrogram expert at different STFTP.

### Why keep HOS expert?
Despite 76.9% on RTL-ML (small dataset), HOS cumulants should perform better on RFUAV's larger dataset. The power-invariant normalization makes cumulants distance-independent — important for real-world drone distances.

### Why YOLOv11-cls as primary classifier?
It achieved 99.4% on RTL-ML spectrograms — the best DL model we tested. ImageNet pretrained backbone + modern CSP architecture + efficient training pipeline make it the strongest starting point.

### Batch size strategy on MI300X?
192GB VRAM allows batch size 512+ for most models. Larger batches → better batch normalization statistics → more stable training. We'll start at 256 and increase if GPU utilization is low.
