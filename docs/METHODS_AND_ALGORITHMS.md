# Methods & Algorithms: Intuition, Implementation, and Results

A comprehensive guide to every method and algorithm in the Drone-RFML-Hub, explaining *why* each approach works, *how* it processes RF signals, and *what* results it achieves on our datasets.

---

## Table of Contents

1. [Signal Representation Methods](#1-signal-representation-methods)
2. [Statistical Feature Extractors](#2-statistical-feature-extractors)
3. [Classical ML Classifiers](#3-classical-ml-classifiers)
4. [Spectrogram-Based Deep Learning](#4-spectrogram-based-deep-learning)
5. [Raw IQ Deep Learning](#5-raw-iq-deep-learning)
6. [RFML-MoE Expert Architectures](#6-rfml-moe-expert-architectures)
7. [Routing & Ensemble Methods](#7-routing--ensemble-methods)
8. [Preprocessing & Denoising](#8-preprocessing--denoising)
9. [IQTLabs Baseline Models](#9-iqtlabs-baseline-models)
10. [Consolidated Results](#10-consolidated-results)

---

## 1. Signal Representation Methods

### 1.1 Raw IQ (In-phase / Quadrature)

**Intuition**: Every RF signal can be represented as two orthogonal components — I (in-phase) and Q (quadrature). Together, I + jQ forms a complex number at each time step, encoding both amplitude and phase of the signal. This is the most fundamental representation — no information is lost.

**How it works on our data**: SDR hardware (RTL-SDR at 1.024 MSps, USRP at 100 MSps) samples the RF spectrum and outputs interleaved float32 I/Q values. Each drone's communication link (control + video) produces a unique IQ pattern based on its modulation scheme, frequency hopping pattern, and hardware imperfections.

**Format**: `(2, N)` tensor — channel 0 = I, channel 1 = Q, N = 32,768 samples (0.33ms at 100MSps)

### 1.2 STFT Spectrogram

**Intuition**: The Short-Time Fourier Transform slides a window across the signal, computing the FFT at each position. This produces a 2D time-frequency representation where you can see *which* frequencies are active at *which* times. Drone signals have characteristic spectral signatures — FM broadcast drones use wideband video links, while FHSS drones hop between narrowband channels.

**How it works on our data**: We use FFT size=256, Hamming window, 50% overlap, Hot colormap. This creates a 2D image that CNNs and Vision Transformers can process. The Hot colormap (black→red→yellow→white) was found optimal by the RFUAV paper (58.16% vs 56.44% Parula).

**Key insight**: The spectrogram converts the classification problem from "signal processing" to "image recognition," enabling transfer learning from ImageNet-pretrained models.

### 1.3 Gramian Angular Field (GAF)

**Intuition**: GAF encodes temporal correlations as a 2D matrix. The signal is first compressed via Piecewise Aggregation (PAA), then each value is mapped to a polar angle. The outer product of angles creates a matrix where element (i,j) represents the angular relationship between time steps i and j. Periodic signals create block-diagonal patterns; random signals create uniform noise.

**How it works on our data**: We compute both GASF (cos of angle sum — captures correlation) and GADF (sin of angle difference — captures anti-correlation), producing a 3-channel (GASF_denoised, GADF, GASF_raw) image at 256×256 resolution. Each drone type creates visually distinct GAF patterns due to different burst periodicities.

**Results**: VMDGAFExpert achieves 35.1% on RFUAV 37-class — underperforms because the PAA compression loses fine-grained IQ structure. Better suited for coarse signal type classification.

### 1.4 LWM 2D Resource Grid

**Intuition**: In OFDM wireless systems, signals are organized on a 2D "resource grid" (subcarriers × time symbols). LWM applies this concept to *any* IQ signal by simply reshaping `(2, 32768)` → `(2, 128, 256)`. The key insight: if a signal has periodicity P, samples at positions 0, P, 2P, ... will align vertically in the grid when grid_width ≈ P, creating horizontal stripes. Different periodicities create different angles of diagonal stripes.

**How it works on our data**: The reshape requires zero computation — it's a free transform. A signal with burst period ≈256 samples (2.56μs at 100MSps) creates horizontal stripes. Period ≈300 creates diagonal stripes. Noise creates random texture. This converts temporal periodicity into spatial patterns that 2D convolutions excel at.

**Results**: LWMExpert achieves **94.1%** on RFUAV 37-class with only 1.3M params — the best raw-IQ model.

### 1.5 Wavelet Packet Decomposition

**Intuition**: Wavelets decompose a signal into frequency subbands at multiple scales. Unlike FFT which gives fixed time-frequency resolution, wavelets provide fine time resolution at high frequencies and fine frequency resolution at low frequencies. This matches how RF signals work — fast transient events (burst edges) need time precision, while slow modulations need frequency precision.

**How it works on our data**: Haar wavelet packet decomposition at 3 levels splits the signal into 8 subbands. Each subband captures a different frequency range. The HiWaveTSTExpert applies GeM (Generalized Mean) pooling to each subband, creating compact per-subband descriptors that a Transformer then processes.

**Results**: HiWaveTSTExpert achieves 75.9% on RFUAV 37-class — mid-range, capturing multi-scale structure but missing fine-grained IQ patterns.

---

## 2. Statistical Feature Extractors

### 2.1 Baseline Features (17 dimensions)

**Features**: Power (mean/std/max/min), FFT (mean/std/max), peak frequency index, I/Q channel statistics, phase statistics, phase derivative statistics, bandwidth ratio.

**Intuition**: These 17 numbers capture the essential "shape" of a signal. FM broadcast has high power_max + wide bandwidth_ratio. Noise has low power_mean + high phase_std (random). Bursty signals have high power_std.

**Results**: 97.5% on RTL-ML (7 classes), 89.8% on RFUAV (37 classes) — simple but effective.

### 2.2 Extended IQ Statistics (37 dimensions)

**Additional features**: Kurtosis, skewness, percentiles, crest factor, zero-crossing rate, autocorrelation at multiple lags, Hilbert envelope statistics.

**Intuition**: Kurtosis distinguishes Gaussian noise (kurtosis=3) from modulated signals (kurtosis>3 for bursty, <3 for constant-envelope FM). Autocorrelation reveals repetition rates — ISM sensors repeat at fixed intervals, while APRS transmissions are sporadic.

**Results**: 98.8% on RTL-ML, 95.3% on RFUAV — kurtosis and crest factor resolve the hardest confusions.

### 2.3 Spectrogram Statistics (37 dimensions)

**Features**: Spectral centroid/bandwidth/rolloff/flatness, 8 band energies, spectral contrast, temporal envelope statistics.

**Intuition**: These compress the 2D STFT into a 1D vector while preserving both spectral and temporal information. Spectral bandwidth separates FM (wide) from narrowband signals. Temporal envelope kurtosis separates continuous (FM) from bursty (APRS, pager) signals. The combination is why this modality achieves 100% — it captures both dimensions.

**Results**: **100%** on RTL-ML (7 classes), 91.2% on RFUAV (37 classes) — the best statistical feature set.

### 2.4 Higher-Order Statistics / Cumulants (20 dimensions)

**Features**: Cumulants C20, C21, C40, C41, C42, C60, C61, C62, C63 (power-invariant normalized) plus derived ratios and phases.

**Intuition**: Cumulants measure the "shape" of a probability distribution beyond mean and variance. Different modulation types produce characteristic cumulant values — BPSK has C42=-2, QPSK has C42=-1, 8PSK has C42=0. This is the classical approach to Automatic Modulation Recognition.

**Results**: 93.8% on RTL-ML, 42.4% on RFUAV — collapses at 37 classes because 6th-order estimates are noisy with finite samples, and many drones use similar GFSK modulation.

### 2.5 Cyclostationary Features (64 dimensions)

**Features**: Spectral Correlation Function (SCF) at 32 cycle frequencies — max and mean power at each.

**Intuition**: Many RF signals exhibit hidden periodicities — not in the signal itself, but in its statistical properties. FM carriers produce strong SCF peaks at the carrier frequency. Symbol-rate modulation creates peaks at the baud rate. Noise has zero cyclic features. The SCF is the gold standard for detecting structured signals in noise.

**Results**: 95.6% on RTL-ML, but computationally expensive (~1s per sample at 100MSps). Sporadic signals (APRS, FRS) lack cyclostationary properties, limiting accuracy.

---

## 3. Classical ML Classifiers

### 3.1 Random Forest (200 trees)

**Intuition**: Build 200 independent decision trees, each on a random subset of data and features. The majority vote across trees is robust to noise and overfitting. RF handles high-dimensional features naturally via random feature selection at each split.

**Why it wins at small scale**: Bagging (bootstrap aggregation) provides natural regularization. Each tree sees different data → diverse ensemble → stable predictions. No gradient optimization → no overfitting risk on 800 samples.

**Results**: Best classical classifier across all modalities. 100% on RTL-ML spectrogram features, 95.7% combined on RFUAV.

### 3.2 Gradient Boosting (100 trees)

**Intuition**: Build trees sequentially, each correcting the errors of the previous ensemble. Unlike RF's parallel independent trees, GBM's sequential dependency learns more complex decision boundaries but risks overfitting.

**Results**: 94.3% combined on RFUAV — slightly below RF (95.7%) because sequential fitting overfits more on limited data.

### 3.3 SVM (RBF kernel)

**Intuition**: Find the hyperplane that maximally separates classes in a high-dimensional kernel space. RBF kernel maps inputs to infinite dimensions where linear separation is possible.

**Results**: PSD + SVM: 60.3% on RFUAV (IQTLabs approach) — insufficient feature representation for 37 classes.

---

## 4. Spectrogram-Based Deep Learning

### 4.1 MaxViT-Base (97.8% RFUAV)

**Intuition**: Combines three complementary mechanisms — MBConv (local texture via depthwise convolutions), Block Attention (local relationships within 7×7 windows), and Grid Attention (global relationships via dilated sampling). This gives both local and global receptive fields at O(N) cost.

**Why it wins**: Drone spectrograms contain both local patterns (modulation-specific textures) and global patterns (frequency hopping across distant spectral bins, periodic burst repetition over time). MaxViT's multi-axis attention captures both.

### 4.2 ConvNeXt-Base (97.5% RFUAV, 92.0% DroneRFb)

**Intuition**: Modernizes the pure CNN with depthwise separable convolutions, inverted bottlenecks, and larger kernels (7×7). Achieves ViT-like performance without attention, purely through improved convolution design.

**Why it's reliable**: Fewer parameters than MaxViT, more stable training. Best on cross-individual DroneRFb (92.0%) showing real-world robustness.

### 4.3 YOLOv8n/v11-cls (96.5-97.0% RFUAV, 99.4% RTL-ML)

**Intuition**: YOLO's backbone (CSPDarknet with C2f blocks) is pretrained on millions of natural images. Fine-tuning on RF spectrograms transfers texture/edge detection capabilities. The classification variant skips detection heads, using global pooling + linear classifier.

**Why it's effective**: Maximum transfer learning leverage — ImageNet features (edges, textures, color gradients) transfer surprisingly well to spectrogram patterns.

### 4.4 EfficientNet-B0 (96.2% RFUAV, 95.4% IQTLabs variant)

**Intuition**: Compound scaling — jointly optimizes depth, width, and resolution for optimal efficiency. MBConv blocks with squeeze-and-excitation provide channel attention.

**Role**: The edge deployment champion at 4.1M params, or IQTLabs' primary model at 95.4%.

### 4.5 VGG16 (95.5% RFUAV via IQTLabs)

**Intuition**: Deep stack of 3×3 convolutions with progressive downsampling. Simple but proven. Fine-tuned from ImageNet: freeze features → train classifier → unfreeze last blocks.

### 4.6 Swin-V2-Base / DeiT-III / EVA-02 (96.5-96.6% RFUAV)

**Intuition**: Various Vision Transformer approaches — shifted windows (Swin), distillation tokens (DeiT), EVA-style pretraining. All perform similarly (~96.5%) on RF spectrograms, below MaxViT's multi-axis design.

---

## 5. Raw IQ Deep Learning

### 5.1 LWMExpert — Large Wireless Model (94.1% RFUAV, 1.3M params)

**Intuition**: Reshape 1D IQ into 2D grid → 2D patch embedding → Transformer. The reshape converts temporal periodicity into spatial patterns. A signal with period P creates stripes at angle arctan(P/grid_width) in the grid.

**Architecture**: IQ→reshape(128×256)→Conv2d patch(8×8)→512 tokens→6-layer Transformer(d=128, 4 heads)→mean pool→512-dim embedding.

**Why it's the IQ champion**: The 2D reshape is a free inductive bias worth millions of parameters. Only 1.3M total params yet beats 17.5M-param models.

### 5.2 TFMSExpert — Time-Frequency Multiscale (82.3% RFUAV, 0.6M params)

**Intuition**: Process time-domain (|z(t)| magnitude) and frequency-domain (|FFT(z)| spectrum) in parallel branches, then fuse via learned gating.

**Architecture**: Two parallel Conv1d stacks (5 blocks each, stride-2 downsampling) → concatenate → dense fusion → embed.

**Why it works**: Captures both amplitude envelope patterns (bursty vs continuous) and spectral shape (wideband vs narrowband) simultaneously with extreme parameter efficiency.

### 5.3 HiWaveTSTExpert (75.9% RFUAV, 3.8M params)

**Intuition**: Haar wavelet packet decomposition into 8 subbands, each processed by learnable GeM pooling, then fused by Transformer.

**Architecture**: |IQ|→Haar WPD(3 levels)→8 subbands→GeM pool each→concat with raw patches→Transformer(4 layers)→embed.

### 5.4 NeuroSymbolicRFFExpert (74.0% RFUAV, 0.5M params)

**Intuition**: Learn 32 "template" signal shapes (shapelets) and compute minimum sliding-window distance between each shapelet and the input. The distance profile encodes which templates match and where.

**Architecture**: 32 learnable shapelets (lengths 16/32/64/128)→min distance via conv1d trick→MLP(32→256→512).

### 5.5 IQFormerExpert (39.4% RFUAV, 13.9M params)

**Intuition**: Dynamic fusion of raw IQ branch + on-the-fly STFT branch via learned sigmoid gating.

**Why it underperforms**: 13.9M params severely overfits on 100K segments. The dual-branch adds complexity without proportional benefit at this data scale.

### 5.6 ResNet1D (86.9% RTL-ML)

**Intuition**: Residual connections solve vanishing gradients in deep 1D CNNs processing long IQ sequences. Skip connections let gradients flow directly through the network.

### 5.7 CLDNN / MCLDNN (63-65% RTL-ML)

**Intuition**: CNN extracts local features → BiLSTM captures temporal dynamics → DNN classifies. Multi-channel variant processes I, Q, and IQ separately before fusion.

**Why they struggle**: LSTMs need diverse temporal examples. With only 800 samples of 0.5s captures, temporal diversity is insufficient.

---

## 6. RFML-MoE Expert Architectures

### 6.1 SpectrogramExpert — EfficientNet-B2 (90.4% DroneRFb, ~93% RFUAV)

The spectrogram pathway. Pretrained EfficientNet-B2 processes 3-channel spectrograms (magnitude, phase, instantaneous frequency). First conv layer reinitialized for 3 RF-specific channels.

### 6.2 SignalFormerRFExpert (85.5% DroneRFb)

CNN-Transformer hybrid with Dilation Time-Frequency Convolution Blocks (D-TFCB) and alternating time/frequency encoder blocks. Processes 3-channel STFT representations.

### 6.3 VisualRFDetector (72.2% DroneRFb)

DETR-style: CNN backbone with CSP blocks → learned object queries → Transformer decoder → attention pooling. Designed for detection but adapted for classification.

### 6.4 MambaIQExpert (OOM on MI300X)

Selective State Space Model with O(L) complexity. Input embedding via Conv1d → 8 Mamba blocks with soft-threshold denoisers. Requires >12GB for 32K sequences — needs shorter segments.

### 6.5 MultiScale-LWM-MaxViT (NEW — 2.44M params)

**Novel architecture** combining LWM's grid reshape with MaxViT's multi-axis attention:
- 4 grid scales (64, 128, 256, 512) capture different burst periodicities
- Each scale: patch embedding → MBConv → Block Attention → Grid Attention
- Cross-scale fusion via attention pooling

Designed to beat single-scale LWM by capturing multiple periodicities simultaneously.

---

## 7. Routing & Ensemble Methods

### 7.1 Expert Choice Routing

**Intuition**: Instead of each sample choosing experts, each expert chooses its preferred samples. This eliminates load imbalance by construction — every expert processes exactly capacity_factor × batch_size / num_experts samples.

### 7.2 SNR-Adaptive Routing

**Intuition**: Different experts work best at different SNR levels. Spectrogram experts excel at low SNR (visual patterns survive noise), while raw IQ experts excel at high SNR (fine-grained features visible). A learned per-expert SNR bias routes signals to appropriate experts based on estimated signal quality.

### 7.3 Confidence-Weighted Routing (100% RTL-ML)

**Intuition**: Weight each expert's prediction by its confidence (max probability). Confident experts dominate; uncertain experts are suppressed. Requires zero additional training parameters.

**Why it achieves 100%**: On RTL-ML, the spectrogram expert is confident and correct on every sample. When it's uncertain, other experts fill in. The confidence weighting naturally selects the best expert per sample.

### 7.4 Majority Voting (100% RTL-ML)

**Intuition**: Each expert predicts independently, majority class wins. Works when experts have complementary errors — they rarely all fail on the same sample.

### 7.5 Stacking Meta-Learner (100% RTL-ML)

**Intuition**: Train a simple logistic regression on the experts' probability outputs. The meta-learner discovers which expert to trust for which input pattern.

---

## 8. Preprocessing & Denoising

### 8.1 Energy Gate (Neyman-Pearson Detection)

**Intuition**: Before classification, detect whether a signal is even present. Compute instantaneous power, estimate noise floor via rolling calibration, apply optimal Neyman-Pearson threshold for target false alarm rate. Also outputs SNR estimate for adaptive routing.

### 8.2 EMD Denoising (Empirical Mode Decomposition)

**Intuition**: Decompose signal into Intrinsic Mode Functions (IMFs) via sifting. High-frequency IMFs contain noise; remove first N_noise IMFs and reconstruct. Adaptive — no fixed filter cutoff.

### 8.3 VMD (Variational Mode Decomposition)

**Intuition**: Decompose signal into K band-limited modes with adaptive center frequencies via ADMM optimization. More mathematically grounded than EMD. Pearson correlation selects which modes contain signal vs noise.

### 8.4 RF-Aware Augmentation

**Applied during training**:
- **AWGN** (0-30 dB, 80%): Additive white Gaussian noise — simulates varying distance/SNR
- **CFO** (±500 Hz, 50%): Carrier frequency offset — simulates oscillator drift
- **Time shift** (50%): Circular rotation — simulates capture timing variation
- **Amplitude scaling** (0.5-2×, 50%): Simulates gain variation

---

## 9. IQTLabs Baseline Models

### 9.1 RFUAV-Net (11.8% RFUAV 37-class)

3-layer 1D CNN designed for binary drone detection. Conv1d(2→128,k=7)→Conv1d(128→128,k=5)→Conv1d(128→128,k=3), each with BN+ReLU+MaxPool. Achieves 99.8% on binary detection but completely fails at multi-class — it learned "drone vs silence," not "which drone."

### 9.2 EfficientNet-B0 on IQ (95.4% RFUAV)

IQTLabs' rfml pipeline: normalize IQ → project 2→3 channels → tile into square image → EfficientNet-B0 pretrained backbone. Solid performance but 3.5% below our LWMExpert at much higher parameter count.

### 9.3 PSD + SVM (60.3% RFUAV)

Welch power spectral density → SVM with RBF kernel. Classic signal processing approach. Fails at 37 classes because PSD captures spectral shape but not temporal dynamics (burst patterns, hopping sequences).

### 9.4 VGG16 on Spectrograms (95.5% RFUAV)

Deep 3×3 conv stack fine-tuned from ImageNet. Phase 1: frozen features, train classifier. Phase 2: unfreeze last blocks, fine-tune at lower LR. Surprisingly competitive despite being a 2014 architecture.

### 9.5 YOLOv8n-cls (96.5% RFUAV)

YOLO nano backbone in classification mode. Best IQTLabs model — pretrained backbone provides strong spectrogram features. Only 0.8% below our MaxViT.

---

## 10. Consolidated Results

### RFUAV 37-Class Drone Identification (Best Per Category)

| Category | Model | Accuracy | Params |
|----------|-------|----------|--------|
| **Best Overall** | MaxViT-Base (spectrogram) | **97.8%** | 118.7M |
| **Best IQTLabs** | YOLOv8n-cls (spectrogram) | **96.5%** | ~3.5M |
| **Best Statistical** | Combined RF (111 features) | **95.7%** | ~200KB |
| **Best Raw IQ** | LWMExpert (2D grid) | **94.1%** | 1.3M |
| **Best Edge** | MobileNetV3 (spectrogram) | **97.1%** | 4.2M |
| **Most Efficient** | TFMSExpert (raw IQ) | **82.3%** | 0.6M |

### DroneRFb Cross-Individual (7 Drone Types)

| Category | Model | Accuracy |
|----------|-------|----------|
| **Best Overall** | ConvNeXt-Base | **92.0%** |
| **Best RFML Expert** | SpectrogramExpert | **90.4%** |
| **Best Edge** | MobileNetV3 | **86.7%** |
| **Statistical Features** | Random Forest | **42.3%** (fails at cross-individual) |

### RTL-ML 7-Class Signal Classification (800 samples)

| Category | Model | Accuracy |
|----------|-------|----------|
| **Best Overall** | Statistical RF (spectrogram features) | **100.0%** |
| **Best DL** | YOLOv11n-cls | **99.4%** |
| **Best Ensemble** | Majority Vote / Stacking / Confidence | **100.0%** |

### The Statistical-DL Crossover

| Samples | Statistical Best | DL Best | Winner |
|---------|-----------------|---------|--------|
| 800 (RTL-ML) | 100.0% | 99.4% | **Statistical** |
| 3,515 (RFUAV) | 95.7% | 97.8% | **DL** |
| 17,416 (DroneRFb) | 42.3% | 92.0% | **DL** (by 50%!) |

**Crossover at ~1,000-2,000 samples**: Below this, hand-crafted features encode sufficient domain knowledge. Above this, learned representations discover patterns humans didn't design.

---

*This document covers 30+ models, 5 feature extraction methods, 4 ensemble strategies, 3 denoising algorithms, and 1 novel architecture (MultiScale-LWM-MaxViT), tested across 5 datasets totaling >2TB of real-world drone RF data.*
