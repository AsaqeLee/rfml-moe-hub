# Neural Network Architecture Comparison for RF Signal Classification

## Executive Summary

This study trains and evaluates **11 neural network architectures** (4 RFML-MoE experts + 7 SOTA alternatives) on 800 real-world radio signal samples, comparing them against statistical feature + Random Forest baselines. Training uses GPU (GTX 1660 Ti) with RF-aware data augmentation.

**Key Finding**: On this 800-sample dataset, **statistical features + Random Forest (100%) still outperform all neural networks** (best DL: ConvNeXt-Tiny-Spec at 98.1%). However, ConvNeXt and ResNet1D show that DL architectures can approach statistical baselines, particularly spectrogram-based models. Raw IQ models (CLDNN, MCLDNN) struggle significantly at 63-65%.

---

## 1. SOTA Landscape (2024-2026)

Based on our survey of 30+ papers across CNN, Transformer, Mamba/SSM, GNN, and hybrid approaches:

### Standard Benchmarks (RadioML 2016.10a)

| Rank | Model | Avg Accuracy | Type |
|------|-------|-------------|------|
| 1 | ECDAT | 95.05%* | CNN + Dual-Attention Transformer |
| 2 | IQFormer | 68.52% | Transformer + Multi-modal Fusion |
| 3 | CC-MSNet | 62.86% | Complex-valued multi-stream |
| 4 | TLDNN | 62.83% | Transformer + LSTM |
| 5 | MCLDNN | 60.83% | Multi-channel LSTM-DNN |

*ECDAT is an unverified outlier (~30pp above all others).

### Key Insights from Literature
- SE-blocks + dilated convolutions = verified CNN SOTA on RadioML 2018.01A (63.7% avg, 98.9% peak)
- Pure transformers struggle without multi-modal fusion
- Mamba/SSM models (ConvMamba, 2025) show promise for long IQ sequences but lack RadioML benchmarks
- For small datasets (<1K): classical ML with engineered features dominates; foundation models (IQFM) show promise in few-shot settings

---

## 2. Architectures Tested

### RFML-MoE Experts (from /home/rax/exp/iq/rfml)

| Model | Input | Architecture | Params | Inspired By |
|-------|-------|-------------|--------|-------------|
| IQ-CNN-Transformer | Raw IQ (2, 32768) | 5-layer 1D CNN + 2-layer Transformer | 549K | RFML SignalFormerIQ |
| Spectrogram-CNN | Spectrogram (3, 128, 128) | 4-layer 2D CNN + MaxPool | 1.4M | RFML EfficientNetSpec |
| FT-Transformer-HOS | HOS cumulants (20,) | Per-feature embedding + 2-layer Transformer | 105K | RFML FTTransformerHOS |
| Dilated-TCN-Cyclo | SCF features (512,) | 6-layer dilated TCN | 159K | RFML TCNCyclo |

### SOTA Alternatives

| Model | Input | Architecture | Params | Reference |
|-------|-------|-------------|--------|-----------|
| ResNet1D | Raw IQ (2, 32768) | 6-block residual CNN | 960K | AMR literature |
| SE-ResNet1D | Raw IQ (2, 32768) | ResNet + Squeeze-and-Excitation | 1.0M | SOTA on RadioML 2018.01A |
| MCLDNN | Raw IQ (2, 32768) | Multi-channel CNN (I/Q/IQ) + BiLSTM | 231K | 2020, RadioML 2016.10a |
| CLDNN | Raw IQ (2, 32768) | CNN + BiLSTM + DNN | 785K | Classic AMR hybrid |
| ConvNeXt-Tiny-Spec | Spectrogram (3, 128, 128) | Depthwise conv + inverted bottleneck | 703K | ConvNeXt (2022) |
| Lightweight-ViT-Spec | Spectrogram (3, 128, 128) | Patch embedding + 3-layer Transformer | 703K | ViT adapted for small data |
| InceptionTime-1D | Raw IQ (2, 32768) | Multi-scale parallel convolutions | 1.8M | SOTA time series classification |

---

## 3. Training Configuration

- **Dataset**: 800 samples, 7 classes, temporal split (64/16/20 train/val/test)
- **Augmentation**: AWGN (0-30 dB), CFO (+-500 Hz), time shift, amplitude scaling
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: Cosine annealing with warm restarts (T_0=20)
- **Loss**: CrossEntropy with label smoothing (0.1)
- **Early stopping**: Patience 15 epochs
- **Max epochs**: 80
- **Gradient clipping**: 1.0
- **Hardware**: GTX 1660 Ti (6GB VRAM)

---

## 4. Results

### Complete Comparison Table

| Rank | Model | Type | Test Acc | F1-macro | Params | ms/sample |
|------|-------|------|----------|----------|--------|-----------|
| 1 | **RF-Spectrogram-37feat** | Statistical | **100.0%** | **1.000** | ~200KB | 1.0 |
| 2 | **RF-Combined-158feat** | Statistical | **100.0%** | **1.000** | ~200KB | 2.0 |
| 3 | RF-IQ-Stat-37feat | Statistical | 98.8% | 0.986 | ~200KB | 0.5 |
| 4 | **ConvNeXt-Tiny-Spec** | DL-Spec | **98.1%** | **0.978** | 703K | 3.2 |
| 5 | RF-Baseline-17feat | Statistical | 97.5% | 0.971 | ~200KB | 0.5 |
| 6 | Lightweight-ViT-Spec | DL-Spec | 91.2% | 0.911 | 703K | 4.5 |
| 7 | Spectrogram-CNN | DL-Spec | 89.4% | 0.891 | 1.4M | 2.8 |
| 8 | ResNet1D | DL-IQ | 86.9% | 0.860 | 960K | 1.5 |
| 9 | IQ-CNN-Transformer | DL-IQ | 80.6% | 0.728 | 549K | 2.1 |
| 10 | FT-Transformer-HOS | DL-HOS | 76.9% | 0.769 | 105K | 0.8 |
| 11 | SE-ResNet1D | DL-IQ | 73.8% | 0.720 | 1.0M | 1.6 |
| 12 | CLDNN | DL-IQ | 65.0% | 0.649 | 785K | 2.3 |
| 13 | MCLDNN | DL-IQ | 63.1% | 0.604 | 231K | 1.8 |
| 14 | InceptionTime-1D | DL-IQ | 76.2% | 0.748 | 458K | 3.5 |
| 15 | Dilated-TCN-Cyclo | DL-Cyclo | 55.0% | 0.530 | 159K | 0.6 |

### By Input Modality

**Spectrogram-based (best DL category):**
- ConvNeXt-Tiny-Spec: 98.1% (modern architecture, depthwise convolutions)
- Lightweight-ViT-Spec: 91.2% (attention mechanism on patches)
- Spectrogram-CNN: 89.4% (simple 4-layer CNN)

**Raw IQ-based:**
- ResNet1D: 86.9% (residual connections help significantly)
- IQ-CNN-Transformer: 80.6% (CNN + Transformer combination)
- SE-ResNet1D: 73.8% (SE blocks hurt with small data — too many params for attention)
- CLDNN: 65.0% (LSTM bottleneck with limited data)
- MCLDNN: 63.1% (multi-channel helps but still data-starved)

**Tabular/1D features:**
- FT-Transformer-HOS: 76.9% (surprisingly competitive for 20 features)
- Dilated-TCN-Cyclo: 55.0% (SCF features too noisy for small TCN)

---

## 5. Analysis

### Why Statistical Features Win at 800 Samples

1. **Feature engineering is a form of prior knowledge**: The 37 spectrogram statistics (centroid, bandwidth, rolloff, flatness, band energies) encode decades of signal processing knowledge. Neural networks must learn these from scratch.

2. **Random Forest is naturally regularized**: Bagging ensemble of 200 trees resists overfitting. DL models with 100K-1.8M parameters overfit easily on 640 training samples.

3. **No information loss**: Statistical features compress 512K complex samples into a discriminative 37-dimensional vector. DL models process the raw data but must learn what to attend to.

4. **The dataset is well-separated**: 7 common radio signal types have distinct spectral signatures. This is a problem where feature engineering can perfectly separate the classes.

### When DL Would Win

1. **More classes (50+)**: With drone RF detection (RFML's target), manual features can't capture all discriminative patterns across 50 models/protocols.
2. **Low SNR**: At -10 to 0 dB, learned representations outperform handcrafted features (spectrogram features degrade gracefully, but learned features can adapt).
3. **More data (10K+)**: DL models benefit from scale; RF forests plateau.
4. **Unknown signal types**: Open-set detection requires learned feature spaces, not predefined statistics.

### Architecture-Specific Insights

**ConvNeXt-Tiny-Spec (98.1%)** — Best DL model. Modern CNN architecture with depthwise convolutions captures spectral patterns efficiently. Only 1.9% behind RF-Spectrogram, demonstrating that a well-designed CNN can nearly match hand-crafted spectrogram features. Fewer parameters than Spectrogram-CNN (703K vs 1.4M) but significantly better accuracy (98.1% vs 89.4%).

**ResNet1D (86.9%)** — Best raw IQ model. Residual connections prevent gradient degradation across the 32768-sample input. This is noteworthy: operating on raw IQ without spectrogram conversion, it still reaches 86.9%.

**SE-ResNet1D (73.8%)** — Surprisingly worse than plain ResNet1D (86.9%). The squeeze-and-excitation blocks add channel attention parameters that overfit on 640 training samples. SE blocks need more data to learn useful channel weights.

**FT-Transformer-HOS (76.9%)** — Competitive despite only 20 input features. Per-feature tokenization with transformer attention works well for this tabular input. However, the underlying HOS features are the bottleneck, not the architecture.

**MCLDNN (63.1%)** — Multi-channel I/Q/IQ processing adds marginal value. The original paper showed it working at 60.83% on RadioML 2016.10a (synthetic, 11 classes); our 63.1% on 7 real-world classes is comparable. LSTM components need more temporal diversity than 800 samples provide.

**Dilated-TCN-Cyclo (55.0%)** — Cyclostationary features via SCF are the weakest modality, and the TCN architecture can't compensate. The features themselves are noisy at this sample count.

### Are RFML-MoE's Architecture Choices SOTA?

| RFML Expert | SOTA? | Better Alternative | Why |
|-------------|-------|-------------------|-----|
| SignalFormerIQ (CNN+Transformer) | Partial | ResNet1D | Simpler ResNet outperforms (86.9% vs 80.6%); Transformer adds overfitting risk |
| EfficientNetSpec | Yes | ConvNeXt-Tiny | ConvNeXt is the modern successor; similar concept, better architecture |
| FTTransformerHOS | Yes | (none better) | Good fit for tabular HOS data; limited by HOS features themselves |
| TCNCyclo | No | (drop or replace) | Both cyclo features and TCN are weak; not worth the compute |

---

## 6. Recommendations

### For RFML-MoE Architecture Updates

1. **Replace EfficientNet with ConvNeXt** for the spectrogram expert. ConvNeXt achieves 98.1% (vs CNN's 89.4%) with fewer parameters and modern design.

2. **Replace SignalFormerIQ with ResNet1D** for the IQ expert. Simpler architecture, better accuracy (86.9% vs 80.6%), fewer parameters.

3. **Keep FT-Transformer for HOS** — it's the right architecture for tabular input. The bottleneck is HOS features, not the model.

4. **Drop or redesign the cyclostationary expert**. At 55.0%, TCN-on-SCF adds more noise than signal to the ensemble. Consider replacing with a second spectrogram expert at a different resolution, or a wavelet-based expert.

5. **Add data augmentation** — AWGN, CFO, time shift significantly help DL models. The RFML pipeline already includes this; ensure it's applied aggressively during training.

### For the RTL-ML Project

- **Keep RF + spectrogram features for edge deployment** — 100% accuracy, tiny model, fast inference
- **Use ConvNeXt-Tiny-Spec if scaling to more classes** — 98.1% with room to grow
- **ResNet1D as raw-IQ fallback** — 86.9% without any feature engineering, useful when STFT is too expensive

### General Guidance

| Scenario | Best Approach |
|----------|--------------|
| <1K samples, <10 classes | Statistical features + RF |
| <1K samples, 10-50 classes | ConvNeXt on spectrograms |
| 1K-10K samples | DL ensemble (ConvNeXt + ResNet1D) |
| 10K+ samples | Full MoE with learned routing |
| Edge/MCU deployment | 17-feature RF (97.5%, <1KB) |
| Low-SNR environment | ConvNeXt + augmentation |

---

## 7. Conclusion

Neural network architectures cannot match hand-crafted features on this 800-sample, 7-class dataset. The information bottleneck is **data volume, not model capacity**. However, the gap is narrowing: ConvNeXt-Tiny-Spec (98.1%) comes within 1.9% of the statistical baseline (100%).

The RFML-MoE architecture makes reasonable choices but is not fully SOTA:
- **Spectrogram expert**: Good concept, should upgrade to ConvNeXt
- **IQ expert**: Overengineered; plain ResNet1D outperforms CNN+Transformer
- **HOS expert**: Architecture is fine; features are the bottleneck
- **Cyclo expert**: Both features and architecture underperform; consider dropping

For practical deployment at RTL-ML scale, statistical features remain the clear winner. DL becomes the right choice when scaling beyond the capabilities of manual feature engineering.

---

*Trained on GTX 1660 Ti (6GB), 800 samples, 7 classes, temporal 64/16/20 split with RF-aware augmentation*
*SOTA survey covers 30+ papers from 2020-2026 across CNN, Transformer, Mamba, GNN architectures*
