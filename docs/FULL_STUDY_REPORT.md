# Comprehensive Study: RF Signal Classification — From Statistical Features to Deep Learning

**Project**: rtl-ml-exp  
**Dataset**: RTL-ML (800 real-world samples, 7 signal classes, 1.024 MSPS IQ captures)  
**Date**: March 2026  
**Hardware**: GTX 1660 Ti (6GB), Manjaro Linux  

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [Experiment Scope & Dataset](#2-experiment-scope--dataset)
3. [Phase 1: Statistical Feature Engineering](#3-phase-1-statistical-feature-engineering)
4. [Phase 2: Ensemble & MoE Methods](#4-phase-2-ensemble--moe-methods)
5. [Phase 3: Neural Network Architectures](#5-phase-3-neural-network-architectures)
6. [Phase 4: YOLO & RT-DETR (RFUAV Approach)](#6-phase-4-yolo--rt-detr-rfuav-approach)
7. [Consolidated Results](#7-consolidated-results)
8. [Deep Feature Analysis](#8-deep-feature-analysis)
9. [The RFUAV Dataset & Its Algorithms](#9-the-rfuav-dataset--its-algorithms)
10. [RFML-MoE Architecture Deep Dive](#10-rfml-moe-architecture-deep-dive)
11. [State of the Art (2024–2026)](#11-state-of-the-art-20242026)
12. [Improvement Roadmap](#12-improvement-roadmap)
13. [Lessons Learned](#13-lessons-learned)

---

## 1. Introduction & Motivation

Radio frequency (RF) signal classification is the task of automatically identifying the type of radio transmission present in a captured segment of electromagnetic spectrum. This has applications in spectrum monitoring, drone detection, interference management, and cognitive radio.

This study systematically compares **18 different approaches** spanning four paradigms:

```
┌──────────────────────────────────────────────────────────┐
│              RF Signal Classification Approaches          │
├──────────────┬──────────────┬──────────┬─────────────────┤
│  Statistical  │  Ensemble/   │  Neural  │  Detection-     │
│  Features     │  MoE         │  Networks│  based (YOLO)   │
├──────────────┼──────────────┼──────────┼─────────────────┤
│ RF+17 feat   │ Majority Vote│ CNN+Tran │ YOLOv8n-cls     │
│ RF+37 IQ     │ Soft Vote    │ Spec-CNN │ YOLOv11n-cls    │
│ RF+37 Spec   │ Stacking     │ FT-Trans │ ResNet50-DETR   │
│ RF+20 HOS    │ Gating       │ TCN      │                 │
│ RF+64 Cyclo  │ Confidence   │ ResNet1D │                 │
│ RF+158 Comb  │ Concat       │ SE-Res1D │                 │
│              │              │ MCLDNN   │                 │
│              │              │ CLDNN    │                 │
│              │              │ ConvNeXt │                 │
│              │              │ ViT      │                 │
│              │              │ Inception│                 │
└──────────────┴──────────────┴──────────┴─────────────────┘
```

### The Core Question

> Can deep learning, with its ability to learn representations automatically, outperform hand-crafted signal processing features on a small real-world RF dataset?

**Answer: No — but the gap is closing to 0.6%.**

---

## 2. Experiment Scope & Dataset

### 2.1 The RTL-ML Dataset

| Property | Value |
|----------|-------|
| **Source** | TrevTron/rtl-ml-dataset (HuggingFace) |
| **Hardware** | RTL-SDR Blog V4 + Indiedroid Nova / Raspberry Pi 5 |
| **Location** | Temecula, CA (real-world captures, not synthetic) |
| **Sample Rate** | 1.024 MSPS (ARM-optimized to prevent USB overflow) |
| **Capture Duration** | 0.5 seconds per sample (512,000 complex IQ samples) |
| **Total Samples** | 800 |
| **Format** | `.npy` dictionaries with keys: `samples`, `center_freq`, `sample_rate`, `timestamp`, `label`, `snr_db`, `version` |
| **Quality Gates** | DC offset removed, 6 dB minimum SNR, per-class validation |

### 2.2 Signal Classes

```
Signal          Freq (MHz)     Samples  Nature           Modulation
─────────────── ────────────── ──────── ──────────────── ──────────────
FM Broadcast    88.5–105.7     200      Continuous       Wideband FM
NOAA Weather    162.4          100      Continuous       Narrowband FM
APRS            144.39         100      Bursty/Sporadic  AFSK 1200 baud
Pager           152.84         100      Bursty/Periodic  POCSAG/FLEX
ISM Sensors     433.92         100      Bursty/Short     OOK/FSK
FRS/GMRS        462.5625       100      Bursty/Voice     Narrowband FM
Noise           145.0          100      Continuous       Thermal noise
```

**Why these classes matter:**

- **FM Broadcast** (200 samples, 5 frequencies): The "easy" class. Wideband (~200 kHz), continuous, high SNR (~17.5 dB). Its wide spectral footprint makes it trivially separable. 200 samples across 5 frequencies test whether models learn "FM-ness" vs memorizing a specific frequency.

- **FRS/GMRS** and **Pager**: The "hard" classes. Both are bursty, narrowband, digital signals. FRS uses voice-activated narrowband FM; pager uses POCSAG/FLEX digital encoding. Their similar burst patterns confuse every model we tested.

- **Noise**: The baseline. A quiet frequency (145 MHz) capturing thermal noise. Every model should classify this perfectly; failure indicates a fundamental problem.

### 2.3 Data Split

All experiments use the same **temporal split** to prevent data leakage:

```
                Per-class temporal ordering
    ┌──────────────────┬─────────┬──────────┐
    │      Train       │   Val   │   Test   │
    │      64%         │  16%    │   20%    │
    └──────────────────┴─────────┴──────────┘
    ← Earlier captures                Later →

    Total: 512 train, 128 val, 160 test
```

The first 64% of each class's temporally-ordered captures go to training, the next 16% to validation, and the final 20% to test. This ensures the model is evaluated on signals captured at different times than training, preventing same-moment correlation from inflating accuracy.

---

## 3. Phase 1: Statistical Feature Engineering

### 3.1 The Intuition

Rather than feeding raw IQ samples (512,000 complex numbers) directly into a classifier, we compress each sample into a small vector of hand-crafted numerical features that encode known signal processing properties. This is essentially **injecting domain expertise** into the feature space.

```
Raw IQ (512K complex)  →  Feature Extractor  →  Feature Vector (17-158 dim)  →  Classifier
     ┌─────────────┐       ┌──────────────┐       ┌─────────────┐
     │ I₁+jQ₁      │       │ Power stats  │       │ [0.23,      │       ┌─────┐
     │ I₂+jQ₂      │  →    │ FFT analysis │  →    │  1.45,      │  →    │ FM  │
     │ ...          │       │ Phase stats  │       │  0.87,      │       └─────┘
     │ I₅₁₂ₖ+jQ₅₁₂ₖ│       │ Bandwidth    │       │  ...]       │
     └─────────────┘       └──────────────┘       └─────────────┘
```

### 3.2 Five Feature Modalities Tested

Each modality captures a different "view" of the signal, inspired by the four expert pathways in the RFML-MoE architecture:

#### Modality A: Baseline RTL-ML (17 features)

The original features from the RTL-ML project. Designed for simplicity and speed.

| # | Feature | Formula | Physical Meaning |
|---|---------|---------|-----------------|
| 1-4 | Power stats | `mean/std/max/min(|x|²)` | Signal amplitude distribution. FM has high stable power; noise has low variable power |
| 5-7 | FFT stats | `mean/std/max(|FFT(x)|²)` | Spectral energy distribution. Wideband signals (FM) have high FFT mean; narrowband signals have high FFT max |
| 8 | Peak frequency | `argmax(|FFT|²) / N` | Normalized location of dominant spectral component. Varies by center frequency offset |
| 9-12 | I/Q stats | `mean/std` of real and imag | In-phase and quadrature balance. Modulated signals have non-zero Q spread |
| 13-14 | Phase stats | `mean/std(∠x)` | Phase distribution. FM has rapidly varying phase; AM has stable phase |
| 15-16 | Phase derivative | `mean/std(Δ∠x)` | Instantaneous frequency. **2nd most important feature** — directly measures modulation rate |
| 17 | Bandwidth ratio | `Σ(|FFT|² > 0.1·max) / N` | Fraction of spectrum occupied. FM ≈ 0.2; narrowband ≈ 0.01-0.05 |

**Why it works (97.5%)**: These 17 features create a 17-dimensional space where each signal class occupies a distinct region. FM has high `power_max` + high `bandwidth_ratio`; noise has low `power_mean` + random `phase_std`; bursty signals have high `power_std`.

**Where it fails**: FRS/GMRS → ISM (2 errors), pager → APRS (2 errors). These bursty signals have similar power statistics in aggregate — the time dimension is lost.

#### Modality B: Extended IQ Statistics (37 features)

Adds higher-order statistics that capture amplitude distribution shape.

| Category | Features Added | Why They Help |
|----------|---------------|---------------|
| Kurtosis, skewness | `kurtosis(|x|)`, `skew(|x|)` | Separates Gaussian (noise) from non-Gaussian (modulated) signals |
| Percentiles | P10, P90, P5, P95 of I, Q | Captures tail behavior without parametric assumptions |
| Crest factor | `max(|x|) / mean(|x|)` | FM has crest factor ≈ 1 (constant envelope); pager has crest factor >> 1 (bursty) |
| Zero-crossing rate | `Σ(sign changes) / N` | Proxy for instantaneous bandwidth. Higher for wideband FM |
| Autocorrelation | `R(1), R(10), R(50), R(100)` | Captures repetition structure. ISM sensors have periodic bursts visible at lag-100 |
| Envelope stats | Hilbert transform envelope | Separates amplitude-modulated from frequency-modulated signals |

**Result: 98.8%** (+1.3% over baseline). Kurtosis and crest factor resolve the pager/APRS confusion.

#### Modality C: Spectrogram Statistics (37 features)

Derived from the Short-Time Fourier Transform (STFT) — a time-frequency representation.

```
How STFT works:

Time domain signal:  ───────────────────────────────────────→ t
                     Window slides across signal
                     ┌───┐
                     │FFT│ → spectrum at t₁
                     └───┘
                         ┌───┐
                         │FFT│ → spectrum at t₂
                         └───┘
                              ┌───┐
                              │FFT│ → spectrum at t₃
                              └───┘

Result: 2D matrix (frequency × time)

    frequency
    ↑ │████████│           │       │  FM: wide band, continuous
      │████████│           │       │
      │        │           │       │
      │        │ ██  ██  ██│       │  APRS: narrow bursts, sporadic
      │        │           │       │
      │████████│████████│████████│  Noise: uniform across all
      └────────┴──────────┴───────→ time
```

The 37 features extracted from this 2D representation include:

| Feature | What it captures |
|---------|-----------------|
| Spectral centroid (mean, std) | Center of mass of spectrum over time — stable for FM, varying for bursty |
| Spectral bandwidth (mean, std) | Width of signal — wide for FM, narrow for pager |
| Spectral rolloff (mean, std) | Frequency below which 85% of energy lies |
| Spectral flatness (mean, std) | How noise-like vs tonal the spectrum is (Wiener entropy) |
| Band energies (8 bands) | Energy distribution across frequency — unique "fingerprint" per signal type |
| Spectral contrast (4 bands) | Peak-to-valley ratio — high for tonal signals, low for noise |
| Phase statistics | Mean/std/kurtosis/skew of STFT phase |
| Instantaneous frequency stats | Phase derivative of STFT — modulation dynamics |
| Temporal envelope | Mean/std/kurtosis/skew of signal amplitude over time |

**Result: 100%** — perfect classification. The key insight is that spectrogram statistics encode **both frequency AND time structure simultaneously**. This is exactly what separates FRS/GMRS from ISM (temporal burst patterns differ) and pager from APRS (packet timing differs).

**Why spectrogram dominates**: Consider the FRS/GMRS vs ISM confusion that plagues other modalities. In the frequency domain alone, both are narrowband signals at UHF frequencies. In the time domain alone, both are bursty. But in the time-frequency plane, FRS/GMRS shows voice-length bursts (~0.5-2 seconds) with constant spectral width, while ISM shows rapid micro-bursts (~1-10 ms) with simpler spectral structure. Spectrogram features capture this distinction; other modalities cannot.

#### Modality D: Higher-Order Statistics / Cumulants (20 features)

Cumulants are statistical measures that characterize the shape of a probability distribution beyond mean and variance.

```
Cumulant Order   What it measures        Example
────────────── ─────────────────────── ──────────────────────────────
2nd (C20, C21)  Variance/power          Basic signal energy
4th (C40-C42)   Kurtosis-like           Distinguishes modulation types:
                                         BPSK: C42 = -2
                                         QPSK: C42 = -1
                                         8PSK: C42 = 0
6th (C60-C63)   Higher shape details    Separates similar modulations
                                         (16QAM vs 64QAM)
```

These cumulants are **power-invariant** when normalized by C21:

```
Ĉ_{p,q} = C_{p,q} / (C21)^{p/2}
```

This normalization removes amplitude dependence, making the features invariant to distance and gain settings — theoretically ideal for RF classification.

**Result: 93.8%** — weakest modality. Higher-order cumulants need large sample counts for stable estimation. With 512K samples per capture, the 6th-order estimates have high variance. Signals using similar modulation schemes (GFSK in FRS, ISM, pager) produce overlapping cumulant profiles.

#### Modality E: Cyclostationary Features (64 features)

Cyclostationary analysis detects **hidden periodicities** in signals — periodicities that don't appear in the signal itself but in its statistical properties (like autocorrelation).

```
Spectral Correlation Function (SCF):

For a signal x(t), the SCF is:
  S_x^α(f) = lim_{T→∞} (1/T) · X_T(f+α/2) · X_T*(f-α/2)

where α = cycle frequency (the hidden periodicity)

    cycle freq α
    ↑
    │  ●              ← carrier frequency peak
    │     ●           ← symbol rate peak
    │        ●        ← sub-carrier peaks
    │                 
    │●●●●●●●●●●●●●●● ← noise (no cyclic features)
    └─────────────────→ spectral freq f
```

For each of 32 cycle frequencies, we extract `max(|SCF|)` and `mean(|SCF|)`, yielding 64 features.

**Result: 95.6%** — mid-range. FM's strong carrier produces clear cyclic peaks. ISM sensor repetition rates create detectable periodicities. But APRS (sporadic, no fixed timing) and FRS (human-initiated, irregular) lack strong cyclostationary properties, leading to 85% and 80% accuracy respectively.

### 3.3 The Random Forest Classifier

All statistical feature modalities use **Random Forest** (200 trees) as the classifier, chosen for its robustness on small datasets:

```
How Random Forest works on our features:

Training (200 trees):
  For each tree:
    1. Bootstrap sample: randomly select 640 samples WITH replacement
    2. At each split node: try √17 ≈ 4 random features
    3. Split on best (most information gain)
    4. Grow tree until pure leaves or min_samples

Prediction:
  New sample → All 200 trees vote → Majority class wins

Why RF beats DL at 800 samples:
  ✓ Each tree sees different data subset → diverse ensemble
  ✓ Random feature selection → decorrelated trees
  ✓ No gradient optimization → no overfitting
  ✓ Inherently handles nonlinear decision boundaries
  ✓ Provably converges to Bayes-optimal with enough trees
```

We also tested Gradient Boosting (94.4%) and MLP (varies by modality), confirming RF's superiority at this scale. GBM's sequential tree-building overfits; RF's parallel independent trees resist it.

---

## 4. Phase 2: Ensemble & MoE Methods

### 4.1 The Intuition

Since each feature modality captures different signal properties, combining them should improve on any individual modality — unless one modality already achieves perfection.

```
Expert Specialization Map:

                   Easy signals              Hard signals
                   (FM, NOAA, Noise)         (FRS, Pager, APRS)
    ───────────────────────────────────────────────────────
    Baseline       ████████████████████       ██████████████░░
    IQ Stat        ████████████████████       ███████████████░
    Spectrogram    ████████████████████       ████████████████  ← perfect
    HOS            ████████████████████       ██████████░░░░░░
    Cyclo          ████████████████████       █████████░░░░░░░
```

### 4.2 Six Ensemble Methods Tested

| Method | Mechanism | Accuracy | Why |
|--------|-----------|----------|-----|
| **Majority Vote** | Each expert predicts → most common wins | **100%** | Experts rarely all err on same sample |
| **Stacking** | Expert probabilities → Logistic Regression | **100%** | Meta-learner exploits expert confidence patterns |
| **Feature Concat** | All 158 features → single RF (300 trees) | **100%** | RF handles high-dim feature selection naturally |
| **Confidence-Weighted** | Weight by max prediction probability | **100%** | Confident experts dominate; uncertain ones suppressed |
| **Soft Vote** | Average prediction probabilities | 99.4% | Equal weighting allows weak experts (HOS) to drag down strong ones |
| **Learned Gating** | MLP on [baseline features + expert probas] | 98.8% | MLP overfits on 640 samples; too many parameters for the routing task |

### 4.3 Expert Agreement Analysis

```
Agreement Matrix (% of test samples where experts agree):

              Base   IQ    Spec   HOS   Cyclo
    Base      100%   97.5  97.5   91.2  94.4
    IQ              100%   98.8   93.8  95.0
    Spec                   100%   93.8  95.6
    HOS                           100%  91.2
    Cyclo                                100%

Key insight: HOS disagrees with everyone ~9% of the time,
             and is WRONG in most disagreements.
```

**When experts disagree, who's right?**

```
    Spectrogram vs ANY other → Spectrogram wins 100% (23/23 disagreements)
    IQ Stat vs HOS          → IQ Stat wins 90%   (9/10 disagreements)
    IQ Stat vs Cyclo         → IQ Stat wins 75%   (6/8 disagreements)
    Baseline vs HOS          → Baseline wins 71%  (10/14 disagreements)
    HOS vs Cyclo             → Cyclo wins 57%     (8/14 disagreements)

Trust hierarchy: Spectrogram >> IQ Stat > Baseline > Cyclo > HOS
```

### 4.4 MoE Architecture Improvements Tested

| Proposal | Idea | Result | Verdict |
|----------|------|--------|---------|
| Feature Selection (top 50) | Reduce dimensionality | 96.3% (-3.7%) | Harmful — discards complementary info |
| Gradient Boosting fusion | Replace RF with GBM | 94.4% (-5.6%) | Overfits at this scale |
| Hierarchical (broad→fine) | 2-stage: continuous/bursty/noise → specific | 99.4% (-0.6%) | Marginal; stage-1 errors propagate |
| **Confidence routing** | Weight experts by prediction confidence | **100%** | Zero training cost, matches oracle |

---

## 5. Phase 3: Neural Network Architectures

### 5.1 The Paradigm Shift

Instead of hand-crafting features, let neural networks learn representations directly from raw data.

```
Statistical approach:           Neural network approach:
                                
IQ → [Human-designed features]  IQ → [Learned features] → [Learned classifier]
     → [Trained classifier]          
                                
Advantages:                     Advantages:
✓ Domain knowledge encoded      ✓ No manual feature engineering
✓ Works with tiny datasets      ✓ Can discover unknown patterns
✓ Fast inference                ✓ Scales to 50+ classes
✗ Limited by human insight      ✗ Needs thousands of samples
✗ Can't discover novel patterns ✗ Black box
```

### 5.2 RFML-MoE Expert Architectures

#### A. IQ-CNN-Transformer (549K params, 80.6%)

```
Raw IQ (2, 32768)
    │
    ├── Conv1d(2→32, k=7, s=2) + BN + GELU        ← Local pattern extraction
    ├── Conv1d(32→64, k=5, s=2) + BN + GELU       
    ├── Conv1d(64→128, k=5, s=2) + BN + GELU      ← Downsample by 32×
    ├── Conv1d(128→128, k=3, s=2) + BN + GELU     
    ├── Conv1d(128→128, k=3, s=2) + BN + GELU     
    │                                               
    ├── Permute to (B, T, C)                        
    ├── TransformerEncoder(2 layers, 4 heads)       ← Global context
    │                                               
    ├── AdaptiveAvgPool1d → (B, 128)               
    └── Linear(128, 7) → predictions                
```

**Intuition**: The CNN blocks act as a learned STFT, extracting local frequency patterns while downsampling. The Transformer then attends to long-range temporal relationships between these patterns. This mirrors how a human would analyze a signal: first identify local spectral features, then look at how they evolve over time.

**Why 80.6%**: With only 512 training samples, the Transformer's self-attention mechanism has insufficient data to learn meaningful long-range dependencies. The attention weights become noisy rather than informative.

#### B. Spectrogram-CNN (1.4M params, 89.4%)

```
Spectrogram (3, 128, 128)
    │                          3 channels:
    │                          Ch 0: log-magnitude (energy distribution)
    │                          Ch 1: wrapped phase (modulation info)
    │                          Ch 2: instantaneous frequency (modulation rate)
    │
    ├── Conv2d(3→32) + BN + GELU + MaxPool2d     ← 128→64
    ├── Conv2d(32→64) + BN + GELU + MaxPool2d    ← 64→32
    ├── Conv2d(64→128) + BN + GELU + MaxPool2d   ← 32→16
    ├── Conv2d(128→256) + BN + GELU              
    ├── AdaptiveAvgPool2d(4)                      ← 16→4
    │
    ├── Flatten → Dropout(0.4)
    ├── Linear(4096, 256) + GELU + Dropout(0.3)
    └── Linear(256, 7)
```

**Intuition**: The 3-channel spectrogram is treated as an image — the CNN learns visual patterns like "wide horizontal band = FM" or "scattered dots = noise". This is exactly the RFUAV paper's approach, but with a custom simple CNN instead of a pretrained backbone.

**Why 89.4%**: Good, but the simple CNN lacks the capacity to learn fine-grained texture differences between similar signal types. It wastes parameters learning basic image features that a pretrained backbone already knows.

#### C. FT-Transformer-HOS (105K params, 76.9%)

```
HOS vector (20 features: C20, C21, C40...C63 + derived ratios)
    │
    ├── Per-feature embedding: each of 20 features → 64-dim token
    │   (Feature Tokenizer: 20 separate Linear(1, 64) layers)
    │
    ├── Prepend [CLS] token + positional embedding
    │
    ├── TransformerEncoder(2 layers, 4 heads, dim=64)
    │   ├── Self-attention between feature tokens
    │   └── Each feature "talks to" all other features
    │
    ├── Extract [CLS] output → LayerNorm
    └── Linear(64, 7)
```

**Intuition**: The Feature Tokenizer Transformer (Gorishniy et al., NeurIPS 2021) treats each numerical feature as a separate "token", allowing the Transformer to learn cross-feature interactions. For HOS, this means the model can learn that "when C42 is high AND C40/C42 ratio is low, this is BPSK-like modulation" — interactions that a flat MLP would need to discover through nonlinear hidden layers.

**Why 76.9%**: The architecture is appropriate for tabular data. The bottleneck is the HOS features themselves: with only 512K samples per capture, 6th-order cumulant estimates are noisy, limiting what any classifier can extract.

#### D. Dilated TCN (159K params, 55.0%)

```
Cyclo features (1, 512)
    │
    ├── Conv1d(1→64, k=1)                    ← Project to channels
    │
    ├── TemporalBlock(d=1):  Conv1d(k=3, d=1)  + BN + GELU + residual
    ├── TemporalBlock(d=2):  Conv1d(k=3, d=2)  + BN + GELU + residual
    ├── TemporalBlock(d=4):  Conv1d(k=3, d=4)  + BN + GELU + residual
    ├── TemporalBlock(d=8):  Conv1d(k=3, d=8)  + BN + GELU + residual
    ├── TemporalBlock(d=16): Conv1d(k=3, d=16) + BN + GELU + residual
    ├── TemporalBlock(d=32): Conv1d(k=3, d=32) + BN + GELU + residual
    │                                         
    │   Receptive field = 2 · Σ(dilation) · (k-1) + 1 = 2·63·2+1 = 253
    │   Covers ~50% of the 512-dim SCF vector
    │
    ├── AdaptiveAvgPool1d → (B, 64)
    ├── Linear(64, 128) + GELU + Dropout(0.3)
    └── Linear(128, 7)
```

**Intuition**: Dilated convolutions grow the receptive field exponentially without increasing parameters. For cyclostationary features, neighboring cycle frequencies are related (harmonics of the same fundamental), so a TCN can capture these multi-scale patterns.

**Why 55.0%**: The underlying SCF features are the weakest modality (95.6% with RF, 55.0% with TCN). The TCN can't compensate for noisy input features. Additionally, the 512-dim SCF has high redundancy — effective dimensionality is ~10-15, but the TCN treats all 512 positions as meaningful.

### 5.3 SOTA Alternative Architectures

#### E. ResNet1D (960K params, 86.9%)

```
Raw IQ (2, 32768)
    │
    ├── Stem: Conv1d(2→64, k=7, s=2) + BN + GELU
    │
    ├── Layer1: 2× ResBlock(64→64, s=1)
    │   ┌─────────────────────────────────┐
    │   │  x → Conv(k=3) → BN → GELU     │
    │   │  → Conv(k=3) → BN              │
    │   │  + identity shortcut            │   ← Skip connections prevent
    │   │  → GELU                         │      gradient vanishing
    │   └─────────────────────────────────┘
    ├── Layer2: 2× ResBlock(64→128, s=2)
    ├── Layer3: 2× ResBlock(128→256, s=2)
    │
    ├── AdaptiveAvgPool1d → (B, 256)
    └── Dropout(0.3) → Linear(256, 7)
```

**Why 86.9%**: ResNet1D is the best raw-IQ model because residual connections solve the vanishing gradient problem across long sequences. The 32768-sample input requires deep networks (many layers), which without skip connections would lose gradient signal. Simple yet effective.

**Why not better**: Raw IQ requires the model to simultaneously learn both spectral analysis (what frequencies are present) and temporal pattern detection (how the signal evolves). With 512 training samples, the model can learn one well but not both.

#### F. SE-ResNet1D (1.0M params, 73.8%)

Adds **Squeeze-and-Excitation** blocks that learn to weight channels adaptively:

```
SE Block:
    feature_maps (B, C, T)
        │
        ├── GlobalAvgPool → (B, C)           ← Squeeze
        ├── Linear(C, C/4) → ReLU            
        ├── Linear(C/4, C) → Sigmoid         ← Excite: learn channel weights
        │
        └── feature_maps × weights            ← Recalibrate channels
```

**Why 73.8% (worse than plain ResNet)**: SE blocks add ~87K trainable parameters to learn attention over channels. With only 512 training samples, these attention weights overfit — the model learns spurious channel correlations from the training set that don't generalize. This is a textbook example of **the regularization-capacity trade-off**: more model capacity ≠ better results when data is scarce.

#### G. ConvNeXt-Tiny-Spec (703K params, 98.1%)

```
Spectrogram (3, 128, 128)
    │
    ├── Stem: Conv2d(3→48, k=4, s=4) + BN     ← Aggressive downsampling (128→32)
    │
    ├── ConvNeXt Block (48→96):
    │   ├── Depthwise Conv2d(k=7, groups=in_ch)  ← Spatial mixing per channel
    │   ├── BatchNorm2d
    │   ├── Pointwise Conv2d(1×1, expand 4×)     ← Channel mixing
    │   ├── GELU
    │   ├── Pointwise Conv2d(1×1, project back)
    │   └── + residual shortcut
    │
    ├── MaxPool2d(2)                              ← 16→8
    ├── ConvNeXt Block (96→192)
    ├── MaxPool2d(2)                              ← 8→4
    ├── ConvNeXt Block (192→384)
    ├── AdaptiveAvgPool2d(1)
    │
    ├── Flatten → LayerNorm(384)
    └── Dropout(0.3) → Linear(384, 7)
```

**Why 98.1% (best custom DL model)**: ConvNeXt modernizes the CNN with insights from Vision Transformers:
- **Depthwise separable convolutions** reduce parameters while maintaining spatial resolution
- **Inverted bottleneck** (expand→contract) is more efficient than traditional bottleneck
- **Larger kernel (7×7)** captures wider spectral/temporal context per layer
- **GELU activation** provides smoother gradients than ReLU
- **Fewer parameters (703K vs 1.4M)** mean less overfitting on 512 training samples

#### H. MCLDNN (231K params, 63.1%)

```
Raw IQ (2, 32768)
    │
    ├── Branch I:  Conv1d(1→32, k=7) → Conv1d(32→32, k=5)  ← In-phase only
    ├── Branch Q:  Conv1d(1→32, k=7) → Conv1d(32→32, k=5)  ← Quadrature only
    ├── Branch IQ: Conv1d(2→32, k=7) → Conv1d(32→32, k=5)  ← Both channels
    │
    ├── Concatenate → (B, 96, T)                             ← Multi-channel fusion
    ├── Conv1d(96→64, k=3, s=2) × 2                         ← Downsample
    │
    ├── BiLSTM(64→128, 2 layers)                             ← Temporal modeling
    │   Forward:  h₁ → h₂ → h₃ → ... → h_T
    │   Backward: h_T → ... → h₃ → h₂ → h₁
    │   Concat: [forward_h_T; backward_h_1] → 256-dim
    │
    └── Linear(256, 128) → GELU → Linear(128, 7)
```

**Why 63.1%**: MCLDNN's multi-channel approach is sound — processing I and Q separately before fusion lets the model learn channel-specific patterns. However, BiLSTM's temporal modeling requires **diverse temporal examples** to learn meaningful dynamics. With 512 training samples of 0.5-second captures from a fixed location, the temporal diversity is insufficient. On RadioML 2016.10a (220K synthetic samples), MCLDNN achieves 60.83% — our 63.1% on 7 real-world classes is actually comparable, suggesting the architecture is data-limited, not fundamentally flawed.

#### I. Lightweight ViT (703K params, 91.2%)

```
Spectrogram (3, 128, 128)
    │
    ├── Patch Embedding: Conv2d(3→128, k=16, s=16)  ← Split into 8×8 = 64 patches
    │
    ├── Prepend [CLS] token
    ├── Add positional embeddings (65 positions)
    ├── Dropout(0.1)
    │
    ├── TransformerEncoder(3 layers, 4 heads, dim=128)
    │   ├── Multi-head self-attention: each patch attends to all others
    │   │   Attention(Q, K, V) = softmax(QK^T / √d) · V
    │   └── FFN: Linear(128, 512) → GELU → Linear(512, 128)
    │
    ├── LayerNorm → Extract [CLS] output
    └── Dropout(0.3) → Linear(128, 7)
```

**Why 91.2%**: ViT treats the spectrogram as a sequence of patches, allowing global attention between distant time-frequency regions. This is powerful for signals where the discriminative pattern spans multiple frequency bands or time intervals. However, ViT needs more data than CNNs to learn good patch representations — pure ViTs underperform CNNs on small datasets, which is why ConvNeXt (98.1%) beats ViT (91.2%) here.

---

## 6. Phase 4: YOLO & RT-DETR (RFUAV Approach)

### 6.1 The RFUAV Paradigm

The RFUAV paper (arXiv 2503.09033) introduces a fundamentally different approach to RF signal classification. Instead of feeding IQ samples or features into a classifier, it:

1. **Generates waterfall spectrograms** from raw IQ using STFT (FFT=256, Hamming window)
2. **Treats classification as image recognition** using pretrained object detection models
3. Uses a **two-stage pipeline**: YOLO for signal detection → ResNet for classification

```
RFUAV Two-Stage Pipeline:

    Raw IQ data (binary IQ fp32, 100 MSps)
         │
         ▼
    ┌─────────────────────────────┐
    │  STFT Spectrogram Generator │    FFT=256, Hamming window
    │  (Dual-buffer + FFT)        │    "Hot" colormap (found optimal)
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Stage 1: YOLO Detector     │    Detect signal regions in spectrogram
    │  (Object detection)         │    Output: bounding boxes around signals
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Stage 2: ResNet/ViT        │    Classify detected signal type
    │  (Image classification)     │    Output: drone model identification
    └─────────────────────────────┘
```

### 6.2 Adapting RFUAV for RTL-ML

For our 7-class classification task (no detection needed), we simplified to **classification mode**:

```
Our adapted pipeline:

    RTL-ML .npy files (512K complex IQ @ 1.024 MSPS)
         │
         ▼
    ┌─────────────────────────────┐
    │  STFT → Hot colormap → PNG  │    FFT=256, Hamming window
    │  Resize to 640×640          │    Matching RFUAV parameters
    └─────────────┬───────────────┘
                  │
         800 spectrogram images
         (train/val/test split)
                  │
         ┌───────┼───────┐
         ▼       ▼       ▼
    ┌─────────┐ ┌──────┐ ┌──────────┐
    │YOLOv8n  │ │YOLO  │ │ResNet50  │
    │-cls     │ │v11n  │ │(RT-DETR  │
    │         │ │-cls  │ │backbone) │
    └─────────┘ └──────┘ └──────────┘
      98.1%     99.4%      96.3%
```

### 6.3 Why YOLO Works So Well for RF Spectrograms

YOLOv11n-cls achieves **99.4%** — the best DL result across all experiments. Three factors explain this:

1. **ImageNet pretraining**: YOLO's backbone is pretrained on millions of natural images. Spectrograms have similar visual patterns to natural images — textures, edges, regions of uniform color. The pretrained features transfer remarkably well.

2. **Modern architecture**: YOLOv11 uses C3k2 blocks (a variant of CSPNet) with efficient channel shuffling and concatenation. This extracts multi-scale features more efficiently than plain CNNs.

3. **Optimized training pipeline**: Ultralytics' training includes augmentation (mosaic, mixup, flipping), cosine LR schedule, and label smoothing — a battle-tested recipe for image classification.

### 6.4 Per-Class Results

```
                YOLOv11n-cls                      YOLOv8n-cls
    Signal      P     R     F1              P     R     F1
    ──────────  ────  ────  ────            ────  ────  ────
    APRS        1.00  1.00  1.00            1.00  0.95  0.97
    FM          1.00  1.00  1.00            1.00  1.00  1.00
    FRS/GMRS    0.95  1.00  0.98            0.91  1.00  0.95   ← Still the hardest
    ISM         1.00  1.00  1.00            1.00  1.00  1.00
    NOAA        1.00  1.00  1.00            1.00  1.00  1.00
    Noise       1.00  0.95  0.97            0.95  0.95  0.95
    Pager       1.00  1.00  1.00            1.00  0.95  0.97
```

YOLOv11 makes only **1 error** (noise classified as FRS/GMRS). YOLOv8 makes **3 errors** — all in the familiar hard classes.

---

## 7. Consolidated Results

### 7.1 Final Rankings (All 18 Approaches)

```
Rank  Model                      Type          Accuracy   F1      Params
────  ─────────────────────────  ────────────  ────────   ─────   ──────
  1   RF-Spectrogram-37feat      Statistical   100.0%     1.000   ~200KB
  2   RF-Combined-158feat        Statistical   100.0%     1.000   ~200KB
  3   YOLOv11n-cls               DL-YOLO       99.4%     0.993   1.6M*
  4   RF-IQ-Stat-37feat          Statistical    98.8%     0.986   ~200KB
  5   ConvNeXt-Tiny-Spec         DL-CNN        98.1%     0.978   703K
  5   YOLOv8n-cls                DL-YOLO       98.1%     0.979   3.5M*
  7   RF-Baseline-17feat         Statistical    97.5%     0.971   ~200KB
  8   ResNet50-DETR-backbone     DL-DETR       96.3%     0.957   23.5M*
  9   Lightweight-ViT-Spec       DL-ViT        91.2%     0.911   703K
 10   Spectrogram-CNN            DL-CNN        89.4%     0.891   1.4M
 11   ResNet1D                   DL-IQ         86.9%     0.889   960K
 12   IQ-CNN-Transformer         DL-IQ         80.6%     0.728   549K
 13   FT-Transformer-HOS        DL-Transformer 76.9%    0.730   105K
 14   InceptionTime-1D           DL-IQ         76.2%     0.715   458K
 15   SE-ResNet1D                DL-IQ         73.8%     0.732   1.0M
 16   CLDNN                      DL-IQ         65.0%     0.649   785K
 17   MCLDNN                     DL-IQ         63.1%     0.604   231K
 18   Dilated-TCN-Cyclo          DL-TCN        55.0%     0.374   159K
```

*Pretrained backbone parameters included

### 7.2 Accuracy by Paradigm

```
    100% ┤██████████████████████████████████████████████████  RF-Spectrogram
         │██████████████████████████████████████████████████  RF-Combined
  99.4%  │█████████████████████████████████████████████████   YOLOv11n-cls
  98.8%  │████████████████████████████████████████████████    RF-IQ-Stat
  98.1%  │████████████████████████████████████████████████    ConvNeXt/YOLOv8
  97.5%  │███████████████████████████████████████████████     RF-Baseline
  96.3%  │██████████████████████████████████████████████      ResNet50-DETR
  91.2%  │████████████████████████████████████████            ViT-Spec
  89.4%  │███████████████████████████████████████             Spec-CNN
  86.9%  │█████████████████████████████████████               ResNet1D
  80.6%  │████████████████████████████████                    IQ-CNN-Transformer
  76.9%  │██████████████████████████████                      FT-Trans-HOS
  76.2%  │█████████████████████████████                       InceptionTime
  73.8%  │████████████████████████████                        SE-ResNet1D
  65.0%  │█████████████████████████                           CLDNN
  63.1%  │████████████████████████                            MCLDNN
  55.0%  │████████████████████                                Dilated-TCN
         └────────────────────────────────────────────────→
```

---

## 8. Deep Feature Analysis

### 8.1 Feature Importance by Modality

#### Baseline (17 features) — Random Forest Feature Importance

```
power_max         ████████████████  0.155   Peak amplitude — FM >> noise
phase_diff_std    ██████████████    0.139   Modulation rate — FM fast, noise random
q_std             ██████████        0.103   Quadrature spread — modulation depth
power_mean        █████████         0.097   Average signal level
fft_mean          ███████           0.073   Overall spectral energy
fft_std           ███████           0.072   Spectral shape variation
power_std         ██████            0.069   Amplitude stability — bursty vs continuous
bandwidth_ratio   █████             0.057   Spectral occupancy width
fft_max           █████             0.050   Peak spectral component
i_std             ████              0.044   In-phase signal spread
```

The top 3 features (power_max, phase_diff_std, q_std) account for **39.7%** of total importance. This means nearly 40% of the classification decision is based on:
- How strong the signal is at its peak
- How fast the phase changes (FM has rapid phase changes; noise has random phase)
- How spread the quadrature component is (correlates with modulation depth)

#### HOS (20 features)

```
C42_norm          ██████████        0.106   Kurtosis-like (Gaussian vs non-Gaussian)
kurtosis_C42      █████████         0.097   4th-order shape measure
C63_norm          █████████         0.095   6th-order signal shape
kurtosis_C63      █████████         0.094   Derived from 6th-order cumulant
C60_norm          ███████           0.074   Raw 6th-order cumulant magnitude
phase_C42         ██████            0.066   Phase of 4th-order cumulant
C40_norm          █████             0.058   4th-order non-Gaussianity
ratio_C40_C42     █████             0.058   Modulation type discriminator
C62_norm          █████             0.055   6th-order cross-term
ratio_C61_C62     █████             0.052   Higher-order ratio
```

C42 (4th-order cumulant) dominates because it directly measures **non-Gaussianity** — the fundamental distinction between structured signals and noise. The BPSK-QPSK-8PSK modulation family produces characteristic C42 values (-2, -1, 0 respectively).

#### Cyclostationary (64 features)

```
scf_max_0         █████             0.058   Peak SCF at lowest cycle frequency
scf_mean_28       ████              0.036   Mean SCF at mid-range cycle freq
scf_mean_26       ███               0.034   (neighboring cycle frequency)
scf_mean_25       ███               0.033   
scf_mean_19       ███               0.027   
...
```

No single SCF feature dominates — importance is distributed across many cycle frequencies. This indicates the cyclostationary features lack a clear discriminative structure at this sample size.

### 8.2 Why Spectrogram Features Are Uniquely Powerful

Spectrogram statistics encode **both spectral structure and temporal dynamics** in a single feature vector. This dual encoding is what gives them perfect classification:

```
Feature                  What it tells us about the signal

Spectral bandwidth       FM: ~200 kHz (wide)
  (mean)                 Pager: ~10 kHz (narrow)
                         Noise: uniform (very wide)

Spectral flatness        Noise: ~1.0 (uniform energy = flat)
  (mean)                 FM: ~0.3 (peaked around carrier)
                         ISM: ~0.1 (very tonal/peaky)

Temporal envelope        FM: low kurtosis (continuous)
  (kurtosis)             APRS: high kurtosis (sparse bursts)
                         FRS: medium kurtosis (voice bursts)
                         ISM: very high kurtosis (micro-bursts)

Band energy ratio        FM: energy spread across bands 1-4
  (8 bands)              Pager: energy concentrated in band 3
                         Noise: equal energy all bands
```

The combination of `spectral_bandwidth` + `temporal_envelope_kurtosis` alone would separate 6 of 7 classes. Adding `spectral_flatness` and `band_energy_ratios` gives the remaining discrimination.

---

## 9. The RFUAV Dataset & Its Algorithms

### 9.1 Dataset Overview

RFUAV is the largest public drone RF dataset (1.3 TB, 37 UAV types), released in 2025:

| Property | RFUAV | RTL-ML (our dataset) |
|----------|-------|---------------------|
| **Size** | 1.3 TB | 6.2 GB |
| **Samples** | ~millions | 800 |
| **Classes** | 37 drone types | 7 signal types |
| **Frequency** | 2.4/5.8 GHz | 88–462 MHz |
| **Sample Rate** | 100 MSps (USRP) | 1.024 MSps (RTL-SDR) |
| **Format** | Binary IQ fp32 | .npy dictionaries |
| **Task** | Drone detection + identification | Signal type classification |

### 9.2 RFUAV's Algorithm Design

#### Spectrogram Generation

RFUAV uses a **dual-buffer queue + FFT** algorithm for efficient real-time spectrogram generation:

```
IQ Stream → [Buffer A] → FFT → Spectrogram frame 1
                         ↑
            [Buffer B] → FFT → Spectrogram frame 2
                         ↑
            [Buffer A] → FFT → Spectrogram frame 3  (reused)
            ...

Parameters found optimal:
  STFTP (FFT points): 256 (balanced time-frequency resolution)
  Window: Hamming
  Colormap: Hot (58.16% accuracy vs 56.44% with Parula)
```

The choice of **Hot colormap** is noteworthy — it outperforms the signal processing standard (Parula) because Hot's high-contrast color mapping makes subtle amplitude differences more visually distinct for CNN feature extraction.

#### Two-Stage Detection Architecture

```
Stage 1: YOLO Signal Detector
├── Input: Full spectrogram image (variable size)
├── Task: Locate drone signal regions (bounding boxes)
├── Output: Cropped signal regions
└── Why YOLO: Real-time, single-pass detection

Stage 2: ResNet/ViT Classifier
├── Input: Cropped signal region from Stage 1
├── Task: Identify drone type (37 classes)
├── Best models tested:
│   ├── ViT-L-16: 56.44% overall (best at high SNR)
│   ├── ResNet18: 54.78% overall
│   └── ViT-B-32: 100% at SNR≥10dB
└── Key challenge: Low SNR degrades all models significantly
```

#### RFUAV's Key Findings

1. **STFT parameter optimization matters**: FFT size 256 with Hamming window provides the best balance. Too small (64) loses frequency resolution; too large (1024) loses temporal resolution.

2. **Colormap affects accuracy**: Hot > Parula > HSV > Autumn. The colormap is not cosmetic — it affects which gradients the CNN backbone can extract.

3. **High SNR vs Low SNR**: ViT-B-32 achieves 100% at SNR≥10dB but collapses at low SNR. This mirrors our findings that spectrogram-based approaches are sensitive to signal quality.

4. **37 classes is genuinely hard**: Even with 1.3 TB of data, per-drone accuracy varies widely. Some drones use identical protocols (DJI OcuSync) and are nearly indistinguishable by RF alone.

### 9.3 RF-YOLO (2025 Paper)

A separate paper (Telecommunication Systems, 2025) introduces **RF-YOLO**, a YOLO modification specifically for RF spectrograms:

| Model | mAP | Precision | Recall |
|-------|-----|-----------|--------|
| **RF-YOLO** | **0.9213** | **0.9800** | **0.9750** |
| YOLOv8 | 0.8433 | 0.9600 | 0.9500 |
| RT-DETR | 0.9053 | 0.9700 | 0.9650 |
| RetinaNet | 0.8743 | 0.9500 | 0.9400 |

RF-YOLO outperforms standard YOLO by +7.8% mAP through RF-specific modifications to the backbone and neck architecture.

---

## 10. RFML-MoE Architecture Deep Dive

### 10.1 The Mixture-of-Experts Concept

```
                    ┌─────────────┐
                    │   Router    │    Learned gating network
                    │  (selects   │    decides which experts
                    │   experts)  │    to use for each input
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌─────────────┐ ┌───────────┐ ┌──────────────┐
    │ IQ Expert   │ │ Spec Expert│ │ HOS Expert   │ ...
    │ (SignalFormer│ │(EfficientNet│ │(FT-Transformer│
    │  15-25M)    │ │  9M)       │ │  2-5M)       │
    └──────┬──────┘ └─────┬─────┘ └──────┬───────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                ┌────────────────┐
                │ Cross-Attention │    2 layers of bidirectional
                │    Fusion      │    attention between experts
                └────────┬───────┘
                         ▼
                ┌────────────────┐
                │  Shared Expert │    DeepSeek-style always-active
                │  (2-layer MLP) │    expert for common features
                └────────┬───────┘
                         ▼
                ┌────────────────┐
                │ Hierarchical   │    Level 1: drone/no-drone
                │ Classification │    Level 2: type (15 classes)
                │                │    Level 3: model (50 classes)
                └────────────────┘
```

### 10.2 Routing Mechanisms

RFML-MoE implements four routing strategies:

1. **Expert Choice** (primary): Each expert selects its preferred inputs, eliminating load imbalance by construction.

2. **Token Choice** (fallback): Standard top-k routing where each sample selects k experts.

3. **DeepSeek Router**: Auxiliary-loss-free routing with dynamic bias terms. No explicit load balancing loss needed.

4. **Soft MoE**: Fully differentiable — all experts process soft-weighted combinations. Perfect load balance but more expensive.

### 10.3 Progressive 4-Phase Training

```
Phase 1: Self-supervised (75 epochs)
├── IQ Expert: Masked Autoencoder (75% masking)
├── Spec Expert: MoCo-v3 contrastive learning
└── HOS/Cyclo: Skipped

Phase 2: Supervised Curriculum (75 epochs)
├── SNR curriculum: easy (+20dB) → medium (+5dB) → hard (-10dB)
├── Loss weighting transitions: [0.5, 0.3, 0.2] → [0.1, 0.2, 0.7]
└── Experts train individually

Phase 3: Gating Training (35 epochs)
├── All expert parameters FROZEN
├── Train only: router, fusion, shared expert, classification heads
└── Prevents disrupting learned expert representations

Phase 4: End-to-end Fine-tuning (15 epochs)
├── All parameters unfrozen
├── Very low learning rate (1e-5)
└── Allows co-adaptation between experts
```

### 10.4 Assessment: Is RFML-MoE SOTA?

Based on our experiments:

| Component | RFML Choice | Our Finding | Better Alternative |
|-----------|-------------|-------------|-------------------|
| IQ Expert | SignalFormerIQ (CNN+Transformer) | 80.6% | ResNet1D (86.9%) or skip Transformer |
| Spec Expert | EfficientNet-B2 | 89.4% (our CNN proxy) | ConvNeXt-Tiny (98.1%) or YOLOv11 (99.4%) |
| HOS Expert | FT-Transformer | 76.9% | Architecture is fine; features are bottleneck |
| Cyclo Expert | Dilated TCN | 55.0% | Drop entirely or replace with wavelet expert |
| Routing | Expert Choice | Learned gating overfits (98.8%) | Confidence weighting (100%) at small scale |
| Fusion | Cross-attention | Not tested (800 samples insufficient) | Simple voting matches oracle |

---

## 11. State of the Art (2024–2026)

### 11.1 Verified SOTA on Standard Benchmarks

| Benchmark | Best Model | Accuracy | Year |
|-----------|-----------|----------|------|
| RadioML 2016.10a | IQFormer | 68.52% avg | 2025 |
| RadioML 2016.10a | ECDAT* | 95.05% avg | 2024 |
| RadioML 2018.01A | SE+Dilated CNN | 63.7% avg, 98.9% peak | 2024 |
| Sig53 | ConvMamba | Competitive (no exact %) | 2025 |
| HisarMod2019.1 | MAMR | 79.01% | 2025 |
| RFUAV (37 drones) | ViT-L-16 | 56.44% | 2025 |

*ECDAT's 95.05% is unverified — 30pp above all others.

### 11.2 Emerging Paradigms

```
2020 ── MCLDNN (60.8%) ── CNN-RNN hybrid era
   │
2022 ── PET-CGDNN (62.0%) ── Phase estimation + gating
   │
2024 ── TLDNN (62.8%) ── Transformer enters AMR
   │    CC-MSNet (62.9%) ── Complex-valued networks
   │    SE+Dilated (63.7%) ── Channel attention + multi-scale CNN
   │    ECDAT (95.0%??) ── Dual-attention Transformer (unverified)
   │
2025 ── IQFormer (68.5%) ── Multi-modal fusion Transformer
   │    ConvMamba ── State-space models enter AMR
   │    IQFM ── Foundation models for RF (99.67% few-shot!)
   │    RF-YOLO (mAP 0.92) ── Object detection for RF
   │
2026 ── GAF-MAE ── Semi-supervised ViT for AMR
        WavesFM ── Cross-modal RF foundation model
```

### 11.3 Foundation Models for RF

The most exciting development is **IQFM** (2025) — a wireless foundation model pretrained on diverse RF data:
- Achieves **99.67% with 1-shot learning** (in-distribution)
- Uses LoRA fine-tuning for adaptation to new tasks
- Dramatically reduces labeled data requirements
- Could potentially solve our 800-sample limitation entirely

---

## 12. Improvement Roadmap

### 12.1 Immediate Wins (No Additional Data Needed)

| Improvement | Expected Gain | Effort |
|-------------|---------------|--------|
| YOLOv11x-cls (larger backbone) | +0.3-0.6% over v11n | Low — just change model size |
| Ensemble: YOLO + RF-Spectrogram | Likely 100% | Low — weighted average of predictions |
| Test-time augmentation (TTA) | +0.3-0.5% for DL models | Low — average predictions over augmented inputs |
| Hyperparameter sweep (lr, augmentation intensity) | +0.5-1% | Medium — grid/Bayesian search |

### 12.2 Architecture Improvements for RFML-MoE

1. **Replace EfficientNet with YOLOv11 backbone** for the spectrogram expert (99.4% > 89.4%)
2. **Replace SignalFormerIQ with ResNet1D** (86.9% > 80.6%, simpler)
3. **Add confidence-gated routing** as fallback when learned router is uncertain
4. **Drop cyclostationary expert** (55.0% adds noise to ensemble)
5. **Implement adaptive expert depth** — spectrogram-only for confident predictions, full MoE for uncertain ones

### 12.3 Methods Worth Trying

| Method | Type | Why Promising | Expected Complexity |
|--------|------|---------------|---------------------|
| **IQFM-style self-supervised pretraining** | Foundation model | 99.67% few-shot on standard benchmarks | High |
| **ConvMamba for long IQ** | State-space model | O(n) complexity, good for 512K samples | Medium |
| **Complex-valued CNNs** | Architecture | Preserves IQ phase natively | Medium |
| **GAF-MAE** (Gramian Angular Field + Masked Autoencoder) | Semi-supervised | Works with 5% labeled data | High |
| **RF-Diffusion** synthetic augmentation | Data augmentation | Generate diverse synthetic training samples | High |
| **Multi-resolution spectrograms** | Feature engineering | Different FFT sizes capture different patterns | Low |
| **Wavelet scattering transform** | Feature engineering | Theoretically invariant to time shifts | Medium |
| **Knowledge distillation** | Model compression | Compress YOLO into tiny model for edge | Medium |

### 12.4 Data-Dependent Improvements

| Data Change | Impact |
|-------------|--------|
| **More samples (2K-10K)** | DL models would close the gap to statistical features |
| **More classes (20+)** | Statistical features would start failing; DL would excel |
| **Variable SNR captures** | Would test model robustness; current data is all >6 dB |
| **Multi-antenna captures** | Spatial features (angle of arrival) add a new dimension |
| **Different geographic locations** | Tests generalization beyond Temecula, CA propagation environment |

---

## 13. Lessons Learned

### 13.1 On Statistical Features vs Deep Learning

> **At 800 samples, hand-crafted features + Random Forest (100%) beat all 14 neural networks tested. The best DL model (YOLOv11, 99.4%) comes within 0.6%.**

This is not surprising if you understand the bias-variance trade-off:
- **Statistical features** have **high bias** (limited by human-designed feature space) but **low variance** (stable with small data).
- **Neural networks** have **low bias** (can learn any function) but **high variance** (unstable with small data, prone to overfitting).

At 800 samples, low variance wins. At 10K+ samples, low bias would win.

### 13.2 On Spectrogram Dominance

> **Every top-performing approach uses spectrogram-based input.**

Whether it's statistical features from STFT (100%), ConvNeXt on spectrograms (98.1%), or YOLO on spectrogram images (99.4%), the spectrogram representation consistently outperforms raw IQ, HOS cumulants, and cyclostationary features. The time-frequency representation captures the complete signal structure in a form that both human-designed features and learned models can exploit.

### 13.3 On Transfer Learning

> **Pretrained models (YOLO, ResNet50) dramatically outperform training from scratch on small datasets.**

YOLOv11n-cls (99.4%) has a backbone pretrained on ImageNet. ConvNeXt-Tiny trained from scratch achieves 98.1%. The 1.3% gap shows that ImageNet features (edges, textures, color gradients) transfer to RF spectrograms — the visual patterns are more similar than you might expect.

### 13.4 On Architecture Complexity

> **Simpler architectures often outperform complex ones at small scale.**

- Plain ResNet1D (86.9%) > SE-ResNet1D (73.8%) — SE blocks add unhelpful complexity
- ConvNeXt-Tiny (98.1%) > EfficientNet-proxy Spectrogram-CNN (89.4%) — fewer parameters, better design
- Random Forest (100%) > Gradient Boosting (94.4%) — bagging > boosting at small scale

### 13.5 On the MoE Paradigm

> **The MoE concept is sound, but the implementation must match the data regime.**

At 800 samples: simple voting (100%) matches the oracle; learned routing overfits (98.8%). At 100K+ samples (RFML's target), the situation reverses — learned routing discovers expert specializations that simple voting cannot.

The right architecture for a given problem depends primarily on **data volume**, not theoretical elegance.

### 13.6 On Experimental Rigor

> **Temporal splits prevent the most common source of evaluation inflation in RF classification.**

Our temporal split (train on earlier captures, test on later) is stricter than random splits. With random splits on temporally correlated data, we would likely see inflated numbers for all models. The fact that statistical features still achieve 100% under temporal split validates that the features capture signal properties, not temporal artifacts.

---

*This report synthesizes results from 18 approaches, 30+ SOTA papers, and the complete RTL-ML and RFUAV datasets. All code and data available at https://github.com/r4d10n/rtl-ml-exp.*
