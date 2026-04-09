# Comprehensive Experiment Documentation

This document details every experiment conducted across all datasets, including methodology, hyperparameters, results, and key findings.

---

## Table of Contents

1. [Phase 1: RTL-ML Statistical Features](#phase-1-rtl-ml-statistical-features)
2. [Phase 2: RTL-ML Ensemble & MoE Methods](#phase-2-rtl-ml-ensemble--moe-methods)
3. [Phase 3: RTL-ML Neural Networks](#phase-3-rtl-ml-neural-networks)
4. [Phase 4: RTL-ML YOLO & RT-DETR](#phase-4-rtl-ml-yolo--rt-detr)
5. [Phase 5: RFUAV 37-Class Spectrogram Training](#phase-5-rfuav-37-class-spectrogram-training)
6. [Phase 6: RFUAV Statistical Features](#phase-6-rfuav-statistical-features)
7. [Phase 7: RFML Expert Benchmark](#phase-7-rfml-expert-benchmark)
8. [Phase 8: DroneRFb Cross-Individual](#phase-8-droneRFb-cross-individual)
9. [Phase 9: MultiScale-LWM-MaxViT](#phase-9-multiscale-lwm-maxvit)
10. [Cross-Phase Analysis](#cross-phase-analysis)

---

## Phase 1: RTL-ML Statistical Features

**Dataset**: RTL-ML -- 800 real-world IQ captures, 7 signal classes, 1.024 MSps
**Hardware**: GTX 1660 Ti (6GB)
**Script**: `experiments/rtl_ml/rfml_comparison.py`
**Results**: `results/rtl_ml/comparison_results.json`

### Methodology

Each of 800 IQ captures (512,000 complex samples at 1.024 MSps) is compressed into a fixed-length feature vector. Five feature modalities are tested, each capturing a different "view" of the RF signal:

```
Raw IQ (512K complex) --> Feature Extractor --> Feature Vector (17-158 dim) --> Classifier
```

Data split: temporal 80/20 per class (640 train, 160 test). No data leakage -- test samples come from later time captures.

### Feature Modalities

#### Modality A: Baseline RTL-ML (17 features)

Original features from the RTL-ML project designed for simplicity and speed:

| # | Feature | Physical Meaning |
|---|---------|-----------------|
| 1-4 | Power stats (mean, std, max, min of abs(x)^2) | Signal amplitude distribution |
| 5-7 | FFT stats (mean, std, max of abs(FFT)^2) | Spectral energy distribution |
| 8 | Peak frequency (argmax(FFT)/N) | Dominant spectral component |
| 9-12 | I/Q stats (mean, std of real and imag) | Modulation depth indicators |
| 13-14 | Phase stats (mean, std of angle(x)) | Phase stability |
| 15-16 | Phase derivative (mean, std of diff(angle(x))) | Instantaneous frequency / modulation rate |
| 17 | Bandwidth ratio (fraction of spectrum above 10% max) | Spectral occupancy width |

**Result: 97.5%** with Random Forest (200 trees). Misclassifies 2 FRS/GMRS as ISM and 2 pager as APRS.

#### Modality B: Extended IQ Statistics (37 features)

Adds higher-order amplitude statistics beyond the baseline:

- Kurtosis, skewness of amplitude envelope
- Percentiles (P5, P10, P90, P95) of I and Q channels
- Crest factor (peak/mean ratio -- separates constant-envelope FM from bursty digital)
- Zero-crossing rate (proxy for instantaneous bandwidth)
- Autocorrelation at lags 1, 10, 50, 100 (captures repetition structure)
- Hilbert envelope statistics (amplitude modulation characteristics)

**Result: 98.8%**. Kurtosis and crest factor resolve the pager/APRS confusion that the baseline misses.

#### Modality C: Spectrogram Statistics (37 features)

Derived from the Short-Time Fourier Transform (STFT), capturing both frequency and time structure:

- Spectral centroid, bandwidth, rolloff, flatness (frequency domain shape)
- Band energy ratios across frequency sub-bands (occupancy patterns)
- Temporal envelope kurtosis (bursty vs continuous in time)
- Time-frequency correlation statistics

**Result: 100.0%**. The combination of spectral and temporal statistics encodes the full signal character. FRS/GMRS (the hardest class) is perfectly separated by its narrowband burst pattern in the time-frequency plane.

#### Modality D: HOS Cumulants (20 features)

Higher-Order Statistics -- 2nd through 6th order cumulants for modulation classification:

- C20, C21, C40, C41, C42, C60, C61, C62, C63 (normalized, power-invariant)
- Cumulant ratios (C40/C42 distinguishes FM from digital modulations)
- Phase of complex cumulants (modulation type indicator)

**Result: 93.8%**. The weakest modality. 6th-order cumulants have high variance with only 512K samples. FRS/GMRS at 75% accuracy is particularly poor -- GFSK variants in FRS, ISM, and pager produce overlapping cumulant profiles.

#### Modality E: Cyclostationary Features (64 features)

Spectral Correlation Function (SCF) via FFT accumulation at 32 cycle frequencies:

- SCF peak values at low cycle frequencies (carrier detection)
- SCF mean at mid-range frequencies (symbol rate detection)
- Statistics across alpha axis

**Result: 95.6%** with Gradient Boosting. Fails on sporadic signals (APRS 85%, FRS/GMRS 80%) because cyclostationary analysis requires periodicity, and human-initiated or irregular transmissions lack it.

### Classifiers Tested Per Modality

Each modality was evaluated with three classifiers:

| Classifier | Configuration |
|-----------|--------------|
| Random Forest | 200 trees, unlimited depth, sqrt(n) features |
| Gradient Boosting | 100 estimators, max_depth=6, lr=0.1 |
| MLP | 256-128 hidden layers, ReLU, Adam, 200 epochs |

Random Forest was best for all modalities except cyclostationary (where GBM won).

### Per-Class Accuracy by Expert

| Signal | Baseline | IQ Stat | Spectrogram | HOS | Cyclo |
|--------|----------|---------|-------------|-----|-------|
| APRS | 100% | 100% | 100% | 95% | 85% |
| FM Broadcast | 100% | 100% | 100% | 100% | 100% |
| FRS/GMRS | 90% | 95% | 100% | 75% | 80% |
| ISM Sensors | 100% | 100% | 100% | 90% | 100% |
| NOAA Weather | 100% | 100% | 100% | 100% | 100% |
| Noise | 100% | 100% | 100% | 100% | 100% |
| Pager | 90% | 95% | 100% | 90% | 100% |

### Feature Importance (Baseline 17 Features)

| Rank | Feature | Importance | What It Captures |
|------|---------|------------|-----------------|
| 1 | power_max | 0.155 | Peak signal strength |
| 2 | phase_diff_std | 0.139 | Modulation rate |
| 3 | q_std | 0.103 | Quadrature spread |
| 4 | power_mean | 0.097 | Average signal level |
| 5 | fft_mean | 0.073 | Spectral energy |

Top 3 features account for 39.7% of total importance.

---

## Phase 2: RTL-ML Ensemble & MoE Methods

**Script**: `experiments/rtl_ml/rfml_comparison.py` (ensemble section)
**Results**: `results/rtl_ml/ensemble_results.json`

### Methods Tested

Six ensemble strategies combining the 5 expert modalities:

| Method | Accuracy | F1-macro | Description |
|--------|----------|----------|-------------|
| **Majority Vote** | **100.0%** | 1.000 | Hard vote across 5 experts |
| **Stacking (LR)** | **100.0%** | 1.000 | Logistic Regression meta-learner on expert predictions |
| **Feature Concat (RF)** | **100.0%** | 1.000 | All 158 features + Random Forest 300 trees |
| **Confidence Routing** | **100.0%** | 1.000 | Weight by max prediction probability |
| Soft Vote | 99.4% | 0.993 | Average probability distributions |
| Learned Gating (MLP) | 98.8% | 0.986 | 2-layer MLP router, trained end-to-end |

### Expert Disagreement Analysis

When experts disagree, which one should be trusted?

```
Trust hierarchy: Spectrogram (always right when disagreeing)
                   > IQ Statistical (right 75-90% vs others)
                     > Baseline (right 67-71% vs HOS/Cyclo)
                       > Cyclostationary (right 57% vs HOS)
                         > HOS (least reliable)
```

### Architecture Improvement Proposals (Tested)

| Proposal | Accuracy | vs Baseline | Verdict |
|----------|----------|-------------|---------|
| **Confidence-Weighted Routing** | **100.0%** | +2.5% | Recommended |
| Two-Stage Hierarchical | 99.4% | +1.9% | Marginal |
| SNR-Aware Feature Selection (top 50) | 96.3% | -1.2% | Harmful |
| Gradient Boosted Expert Fusion | 94.4% | -3.1% | Overfits |

**Key finding**: Confidence-weighted routing (zero additional parameters) achieves perfect accuracy. Learned gating (MLP) overfits on 800 samples. Simple methods win at small scale.

---

## Phase 3: RTL-ML Neural Networks

**Dataset**: RTL-ML 800 samples, temporal split 64/16/20 (512 train, 128 val, 160 test)
**Hardware**: GTX 1660 Ti (6GB VRAM)
**Script**: `experiments/rtl_ml/nn_comparison.py`
**Results**: `results/rtl_ml/nn_comparison_results.json`

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=0.01) |
| Scheduler | Cosine annealing with warm restarts (T_0=20) |
| Loss | CrossEntropy with label smoothing (0.1) |
| Augmentation | AWGN (0-30 dB), CFO (+-500 Hz), time shift, amplitude scaling |
| Early stopping | Patience 15 epochs |
| Max epochs | 80 |
| Gradient clipping | 1.0 |

### Architectures and Results

#### RFML-MoE Expert Variants

| Model | Input | Architecture | Accuracy | Params |
|-------|-------|-------------|----------|--------|
| IQ-CNN-Transformer | Raw IQ (2, 32768) | 5-layer 1D CNN + 2-layer Transformer | 80.6% | 549K |
| Spectrogram-CNN | Spectrogram (3, 128, 128) | 4-layer 2D CNN + MaxPool | 89.4% | 1.4M |
| FT-Transformer-HOS | HOS cumulants (20,) | Per-feature embedding + 2-layer Transformer | 76.9% | 105K |
| Dilated-TCN-Cyclo | SCF features (512,) | 6-layer dilated TCN (dilations 1-32) | 55.0% | 159K |

#### SOTA Alternatives

| Model | Input | Architecture | Accuracy | Params |
|-------|-------|-------------|----------|--------|
| **ConvNeXt-Tiny-Spec** | Spectrogram | Depthwise conv + inverted bottleneck | **98.1%** | 703K |
| **Lightweight-ViT-Spec** | Spectrogram | Patch embedding + 3-layer Transformer | **91.2%** | 703K |
| ResNet1D | Raw IQ | 6-block residual CNN | 86.9% | 960K |
| SE-ResNet1D | Raw IQ | ResNet + Squeeze-and-Excitation | 73.8% | 1.0M |
| InceptionTime-1D | Raw IQ | Multi-scale parallel convolutions | 76.2% | 458K |
| CLDNN | Raw IQ | CNN + BiLSTM + DNN | 65.0% | 785K |
| MCLDNN | Raw IQ | Multi-channel CNN (I/Q/IQ) + BiLSTM | 63.1% | 231K |

### Key Observations

1. **Spectrogram-based DL >> raw IQ DL**: ConvNeXt-Spec (98.1%) vs best raw IQ ResNet1D (86.9%)
2. **SE blocks hurt at small scale**: SE-ResNet1D (73.8%) < plain ResNet1D (86.9%)
3. **LSTM models fail on IQ**: CLDNN (65.0%) and MCLDNN (63.1%) -- 512K timesteps is too long for recurrent processing
4. **Statistical features still win**: Best DL (98.1%) < RF+Spectrogram features (100%)
5. **Gap is closing**: Best DL is within 1.9% of statistical perfection

---

## Phase 4: RTL-ML YOLO & RT-DETR

**Script**: `experiments/rtl_ml/yolo_detr_comparison.py`
**Results**: `results/rtl_ml/yolo_detr_results.json`

### Methodology

Convert IQ samples to spectrogram images (128x128, Hot colormap, FFT=256), then use pretrained image classification models:

| Model | Accuracy | F1-macro | Notes |
|-------|----------|----------|-------|
| **YOLOv11n-cls** | **99.4%** | 0.993 | Best DL model on RTL-ML |
| YOLOv8n-cls | 98.1% | 0.979 | Ties with ConvNeXt |
| ResNet50-DETR-backbone | 96.3% | 0.959 | RT-DETR adapted for classification |

**Key finding**: Pretrained YOLO models achieve near-perfect accuracy with minimal fine-tuning (100 epochs), dramatically outperforming training from scratch.

---

## Phase 5: RFUAV 37-Class Spectrogram Training

**Dataset**: RFUAV -- 37 drone types, 3553 spectrograms (2623 train, 890 val)
**Hardware**: AMD Instinct MI300X (206 GB HBM3)
**Script**: `experiments/rfuav/train_rfuav.py`
**Results**: `results/rfuav/`

### Spectrogram Generation

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| FFT size | 256 | RFUAV paper's optimum |
| Window | Hamming | Paper default |
| Colormap | Hot | Paper's best (58.16% vs 56.44% Parula) |
| Duration | 0.01s per frame (1M samples at 100 MSps) | Balance resolution vs quantity |
| Image size | 640x640 → resized per model | Standard classifier input |
| Overlap | 50% | Standard STFT overlap |

Generated from 356 .iq files across 37 drone folders, yielding 3553 spectrogram images with 80/20 train/val split per drone type.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-4) |
| Precision | BF16 mixed precision |
| Label smoothing | 0.1 |
| Scheduler | CosineAnnealing |
| Early stopping | Patience 20 epochs |
| Max epochs | 50 |
| Pretrained | ImageNet for all timm models |
| Augmentation | Albumentations: AdvancedBlur, CLAHE, ColorJitter, GaussNoise, ISONoise, Sharpen |

### Results (16 timm Models + 3 YOLO)

| Rank | Model | Accuracy | F1 | Params | Train Time |
|------|-------|----------|-----|--------|-----------|
| 1 | **MaxViT-Base** | **97.8%** | 0.972 | 118.7M | 722s |
| 2 | ConvNeXt-Base | 97.5% | 0.970 | 87.6M | 281s |
| 2 | EfficientNetV2-L | 97.5% | 0.969 | 117.3M | 914s |
| 4 | YOLOv11n-cls | 97.4% | 0.967 | ~1.6M | -- |
| 5 | ConvNeXt-Large | 97.1% | 0.964 | 196.3M | 586s |
| 5 | MobileNetV3-Large | 97.1% | 0.964 | 4.2M | 577s |
| 7 | YOLOv11s-cls | 97.2% | 0.965 | ~5M | -- |
| 8 | ViT-L-16 | 96.9% | 0.962 | 303.3M | 437s |
| 9 | DeiT3-Base | 96.6% | 0.959 | 85.8M | 439s |
| 9 | Swin-V2-Base | 96.6% | 0.957 | 86.9M | 609s |
| 11 | ViT-B-32 | 96.5% | 0.958 | 87.5M | 515s |
| 11 | EVA-02-Base | 96.5% | 0.958 | 85.8M | 576s |
| 13 | YOLOv8n-cls | 96.3% | 0.951 | ~3.5M | -- |
| 14 | EfficientNet-B0 | 96.2% | 0.950 | 4.1M | 563s |
| 15 | ResNet50 | 95.3% | 0.937 | 23.6M | 590s |
| 16 | ResNet18 | 91.8% | 0.897 | 11.2M | 381s |

### Efficiency Analysis

| Model | Accuracy | Params | Acc/Param Ratio |
|-------|----------|--------|-----------------|
| MobileNetV3-Large | 97.1% | 4.2M | 23.1 |
| YOLOv11n-cls | 97.4% | ~1.6M | 60.9 |
| EfficientNet-B0 | 96.2% | 4.1M | 23.5 |
| ConvNeXt-Base | 97.5% | 87.6M | 1.1 |
| MaxViT-Base | 97.8% | 118.7M | 0.8 |
| ViT-L-16 | 96.9% | 303.3M | 0.3 |

**Key findings**:
- MaxViT's multi-axis attention (block + grid) is optimal for spectrogram classification
- Bigger models are NOT better: ConvNeXt-Large < ConvNeXt-Base; ViT-L < MaxViT-Base
- MobileNetV3 at 97.1% with 4.2M params is the edge deployment champion
- YOLOv11n-cls is the best parameter-efficient model (97.4% with ~1.6M params)

---

## Phase 6: RFUAV Statistical Features

**Dataset**: RFUAV -- 356 .iq files, 10 chunks/file = 3560 samples, 37 classes
**Hardware**: CPU (feature extraction), MI300X (optional for MLP)
**Script**: `preprocessing/rfuav_statistical_features.py`
**Results**: `results/rfuav/rfuav_statistical_results.json`

### Feature Extraction

Same 5 modalities as Phase 1, adapted for 100 MSps / 10M sample segments:

| Modality | Features | RFUAV Accuracy | RTL-ML Accuracy |
|----------|----------|----------------|-----------------|
| Combined (all) | 158 | **95.7%** | 100.0% |
| IQ Statistical | 37 | 95.3% | 98.8% |
| Spectrogram | 37 | 93.8% | 100.0% |
| Baseline | 17 | 91.2% | 97.5% |
| Cyclostationary | 64 | 78.4% | 95.6% |
| HOS | 20 | 72.1% | 93.8% |

### The Crossover Point

Statistical features degrade from 100% to 95.7% as complexity scales from 7 to 37 classes. Meanwhile, DL (MaxViT 97.8%) surpasses statistical methods. The crossover occurs at approximately 1,000--2,000 training samples / 10--15 classes.

```
                 Accuracy
                 100% ─ ● Statistical (RTL-ML, 7 classes)
                  98% ─   ● DL best (RTL-ML)
                  97% ─                    ● DL best (RFUAV, 37 classes)
                  96% ─                  ● Statistical (RFUAV)
                  95% ─
                       ─────────────────────────────────
                       800 samples        3500 samples
                       7 classes          37 classes
```

---

## Phase 7: RFML Expert Benchmark

**Dataset**: RFUAV (37 classes) + DroneRFb (7 types)
**Hardware**: AMD Instinct MI300X
**Script**: `experiments/rfml_benchmark/rfml_expert_benchmark.py`
**Results**: `results/rfml_benchmark/`

### RFML-MoE Expert Results on RFUAV

All 11 experts from the RFML-MoE codebase, trained from scratch on RFUAV raw IQ data:

| Expert | Input | RFUAV Accuracy | Params |
|--------|-------|----------------|--------|
| **LWMExpert** | Raw IQ | **94.1%** | 1.3M |
| SpectrogramExpert | Spectrogram | ~93.7% | ~9M |
| SignalFormerRFExpert | Raw IQ | 85.5% | ~15M |
| TFMSExpert | Raw IQ | 82.3% | ~8M |
| HiWaveTSTExpert | Raw IQ | 75.9% | ~5M |
| IQExpert | Raw IQ | ~74% | ~20M |
| IQFormerExpert | Raw IQ | ~71% | 13.8M |
| NeuroSymbolicRFFExpert | Raw IQ | ~68% | ~12M |
| VMDGAFExpert | GAF images | 35.1% | ~8M |
| VisualRFDetector | Spectrogram | -- | ~15M |

### LWMExpert: The Efficiency Champion

LWMExpert achieves 94.1% on RFUAV raw IQ with only 1.3M parameters by:

1. **IQ-to-2D reshape**: Treats 1D IQ sequence as a 2D grid, converting the temporal problem into a spatial one
2. **Lightweight Multi-axis attention**: Block (local window) + Grid (dilated global) attention, similar to MaxViT but without the heavy CNN backbone
3. **No spectrogram preprocessing**: Operates directly on raw IQ, eliminating the STFT computation overhead

This makes LWMExpert the most parameter-efficient drone classifier in the entire study, approaching spectrogram-DL performance without spectrogram generation.

### RFML Expert Results on DroneRFb

| Expert | DroneRFb Type Acc | Notes |
|--------|-------------------|-------|
| SpectrogramExpert | 90.4% | Best RFML expert on cross-individual |
| SignalFormerRFExpert | ~85% | |
| VisualRFDetector | ~82% | |

---

## Phase 8: DroneRFb Cross-Individual

**Dataset**: DroneRFb-DIR -- Train: individuals 1&2 (2177 files), Test: individual 3 (2513 files)
**Hardware**: AMD Instinct MI300X
**Scripts**: `experiments/droneRFb/droneRFb_train.py`, `experiments/droneRFb/droneRFb_type_train.py`
**Results**: `results/droneRFb/`

### Spectrogram Generation

| Parameter | Value |
|-----------|-------|
| Input format | MATLAB v7.3 (HDF5), keys I and Q, 1x4000000 float32 |
| Sample rate | 80 MSps |
| FFT size | 256, Hamming window |
| Colormap | Hot |
| Segments per file | 4 (1M samples each) |
| Image size | 640x640 |

### 13-Class Individual Identification Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| ConvNeXt-Base | 92.0% | Best overall |
| MaxViT-Base | 89.5% | |
| EfficientNet-B0 | 87.3% | |
| MobileNetV3-Large | 86.8% | |
| ResNet18 | 83.1% | |

### 7-Class Type-Level Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| ConvNeXt-Base | 94.2% | Best |
| SpectrogramExpert (RFML) | 90.4% | |
| MaxViT-Base | 93.1% | |

### Statistical Features on DroneRFb

| Modality | Accuracy | Notes |
|----------|----------|-------|
| Combined RF | 42.3% | Dramatic failure |
| IQ Statistical | 39.1% | |
| Spectrogram stats | 38.7% | |

**Key finding**: Statistical features completely fail on cross-individual generalization (42.3% vs 92.0% for DL). Individual hardware variations create unique RF fingerprints that are invisible to aggregate statistics but captured by learned spatial features in spectrograms. This is the strongest evidence that DL is necessary for real-world drone detection.

---

## Phase 9: MultiScale-LWM-MaxViT

**Architecture**: Novel fusion of LWM's IQ-to-2D-grid insight with MaxViT's multi-axis attention
**Script**: `models/multiscale_lwm_maxvit.py`
**Status**: Architecture implemented, training in progress

### Design Rationale

LWMExpert (94.1% on raw IQ, 1.3M params) shows that reshaping 1D IQ to 2D grids is powerful. MaxViT-Base (97.8% on spectrograms, 118.7M params) shows that multi-axis attention is optimal for 2D RF representations. MultiScale-LWM-MaxViT combines both insights:

### Architecture Details

| Component | Configuration |
|-----------|--------------|
| Input | Raw IQ (B, 2, 32768) |
| Stem | 1D Conv, 2 channels to 64 |
| Grid scales | 4: {64x512, 128x256, 256x128, 512x64} |
| Per-scale blocks | MBConv (depthwise separable + SE) + Block Attention (window=7) + Grid Attention (grid=7) |
| Fusion | Cross-scale attention pooling with learned importance |
| Classifier | Linear (embed_dim -> num_classes) |
| Total params | 2.44M |
| Target | Beat LWM 94.1%, approach MaxViT 97.8% on raw IQ |

### Multi-Scale Grid Interpretation

Each grid width captures different burst periodicities at 100 MSps:

| Grid Width | Temporal Period | Physical Meaning |
|------------|----------------|-----------------|
| 64 | 0.64 us | Intra-symbol modulation patterns |
| 128 | 1.28 us | Symbol-level structure |
| 256 | 2.56 us | Burst envelope patterns |
| 512 | 5.12 us | Inter-burst timing / protocol-level |

---

## Cross-Phase Analysis

### Statistical Features vs Deep Learning: The Complete Picture

| Scenario | Statistical | Best DL | Winner |
|----------|-----------|---------|--------|
| RTL-ML (800 samples, 7 classes) | 100.0% | 99.4% (YOLO) | Statistical |
| RFUAV (3553 samples, 37 classes) | 95.7% | 97.8% (MaxViT) | **DL** |
| DroneRFb cross-individual (13 classes) | 42.3% | 92.0% (ConvNeXt) | **DL** |

### When to Use What

| Condition | Recommended Approach | Expected Accuracy |
|-----------|---------------------|-------------------|
| Small dataset (<1K samples), few classes (<10) | Statistical features + RF ensemble | >97% |
| Medium dataset (1K-10K), many classes (10-50) | Pretrained spectrogram DL (MaxViT/ConvNeXt) | >95% |
| Cross-individual/cross-device generalization | DL only (statistical features fail) | >90% |
| Edge deployment (limited compute) | MobileNetV3 or YOLOv11n on spectrograms | >97% |
| Raw IQ (no preprocessing budget) | LWMExpert or MultiScale-LWM-MaxViT | >94% |

### Architecture Recommendations for RFML-MoE

Based on all experiments:

1. **Replace EfficientNet-B2 with MaxViT-Base** as spectrogram expert (97.8% vs ~93.7%)
2. **Replace SignalFormerIQ with LWMExpert** as IQ expert (94.1% vs 85.5%, 10x fewer params)
3. **Drop cyclostationary expert** (55% RTL-ML, worst everywhere, slow SCF)
4. **Add confidence-weighted routing** as fallback when learned router entropy is high
5. **Add adaptive expert depth** -- spectrogram-only for confident (>95%) predictions
6. **Add YOLOv11-cls** as lightweight spectrogram expert for edge deployment
7. **Use Hot colormap + FFT=256 + Hamming** for all spectrogram generation
8. **MobileNetV3-Large for edge** (97.1%, 4.2M params, within 0.7% of MaxViT)

### Open Questions

1. Does MultiScale-LWM-MaxViT (2.44M params, raw IQ) close the gap with MaxViT-Base (118.7M, spectrograms)?
2. How do models perform at low SNR (-10 to 0 dB)? SNR benchmark pending.
3. Can models trained on RFUAV (5.8 GHz) transfer to DroneRFb (2.4 GHz)?
4. What is the impact of DRFF-R2's 7 scenarios on cross-scenario generalization?
5. Can VMD + GAF preprocessing rescue the failed VMDGAFExpert (35.1%)?
