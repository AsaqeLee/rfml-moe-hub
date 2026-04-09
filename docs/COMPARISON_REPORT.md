# RFML-MoE vs RTL-ML: Comprehensive Comparison Study

## Executive Summary

This study compares the **RTL-ML Random Forest baseline** (17 handcrafted features) against **RFML-MoE style feature modalities** (IQ statistical, spectrogram, higher-order statistics, cyclostationary) on 800 real-world radio signal samples across 7 classes. We evaluate individual experts, ensemble methods, and propose architecture improvements.

**Key Finding**: Spectrogram-derived statistical features alone achieve **100% accuracy**, matching the full RFML combined feature set. The original RTL-ML baseline (97.5%) can be improved to 100% by adding spectrogram features, without requiring the full MoE complexity.

---

## 1. Methodology

### Dataset
- **Source**: TrevTron/rtl-ml-dataset (HuggingFace)
- **Samples**: 800 total (512K complex IQ samples each at 1.024 MSPS)
- **Classes**: 7 (APRS, FM Broadcast, FRS/GMRS, ISM Sensors, NOAA Weather, Noise, Pager)
- **Split**: Temporal 80/20 per class (160 test samples)

### Feature Modalities Tested

| Modality | Features | Inspired By | Description |
|----------|----------|-------------|-------------|
| **Baseline** | 17 | RTL-ML original | Power, FFT, I/Q, phase, bandwidth stats |
| **IQ Statistical** | 37 | RFML SignalFormerIQ | Extended amplitude, phase, inst. freq, autocorrelation, envelope |
| **Spectrogram** | 37 | RFML EfficientNetSpec | STFT-derived: spectral centroid, bandwidth, rolloff, flatness, band energies |
| **HOS** | 20 | RFML FTTransformerHOS | Cumulants C20-C63, power-invariant normalization, ratios |
| **Cyclostationary** | 64 | RFML TCNCyclo | Spectral Correlation Function via FFT accumulation |
| **Combined** | 158 | Full RFML pipeline | All modalities concatenated |

### Classifiers
Each modality tested with Random Forest (200 trees), Gradient Boosting (100 trees), and MLP (256-128 hidden). Best model selected per modality.

---

## 2. Individual Expert Results

### Overall Accuracy

| Modality | Features | Accuracy | F1-macro | Best Classifier |
|----------|----------|----------|----------|-----------------|
| **Spectrogram** | 37 | **100.0%** | **1.000** | Random Forest |
| **Combined RFML** | 158 | **100.0%** | **1.000** | Random Forest |
| **IQ Statistical** | 37 | 98.8% | 0.986 | Random Forest |
| **Baseline RTL-ML** | 17 | 97.5% | 0.971 | Random Forest |
| **Cyclostationary** | 64 | 95.6% | 0.950 | Gradient Boosting |
| **HOS Cumulants** | 20 | 93.8% | 0.928 | Random Forest |

### Per-Class Accuracy by Expert

| Signal | Baseline | IQ Stat | Spectrogram | HOS | Cyclo | Best Expert |
|--------|----------|---------|-------------|-----|-------|-------------|
| APRS | 100% | 100% | **100%** | 95% | 85% | baseline/iq/spec |
| FM Broadcast | 100% | 100% | **100%** | 100% | 100% | all |
| FRS/GMRS | 90% | 95% | **100%** | 75% | 80% | **spectrogram** |
| ISM Sensors | 100% | 100% | **100%** | 90% | 100% | baseline/iq/spec/cyclo |
| NOAA Weather | 100% | 100% | **100%** | 100% | 100% | all |
| Noise | 100% | 100% | **100%** | 100% | 100% | all |
| Pager | 90% | 95% | **100%** | 90% | 100% | **spec/cyclo** |

### Key Observations

1. **Spectrogram features are the dominant expert** - 100% on every class. Time-frequency representations capture all discriminative information for these 7 signal types.

2. **HOS is the weakest modality** (93.8%). Cumulants struggle with FRS/GMRS (75%) and signals lacking strong modulation signatures. The 800-sample dataset may be insufficient for reliable higher-order statistics estimation.

3. **Cyclostationary features** (95.6%) perform unevenly: perfect on FM/ISM/NOAA/Noise/Pager, but only 85% on APRS and 80% on FRS/GMRS. Bursty, sporadic signals (APRS, FRS) don't exhibit strong cyclostationary properties.

4. **IQ Statistical** (98.8%) is the strongest non-spectrogram expert, with only 2 misclassifications. Adding kurtosis, skewness, and autocorrelation to baseline features provides meaningful lift.

5. **Baseline RTL-ML** (97.5%) is remarkably effective for its simplicity. The 17 features capture most discriminative power; the 4 misclassifications are all in the bursty signal pairs (FRS/GMRS, pager).

---

## 3. Expert Agreement Analysis

### Agreement Matrix

| | Baseline | IQ Stat | Spectrogram | HOS | Cyclo |
|-----------|----------|---------|-------------|------|-------|
| Baseline | 100% | 97.5% | 97.5% | 91.2% | 94.4% |
| IQ Stat | 97.5% | 100% | 98.8% | 93.8% | 95.0% |
| Spectrogram | 97.5% | 98.8% | 100% | 93.8% | 95.6% |
| HOS | 91.2% | 93.8% | 93.8% | 100% | 91.2% |
| Cyclo | 94.4% | 95.0% | 95.6% | 91.2% | 100% |

### When Experts Disagree, Who's Right?

| Pair | Disagreements | Winner | Win Rate |
|------|--------------|--------|----------|
| Baseline vs Spectrogram | 4 | **Spectrogram** | 100% (4/4) |
| IQ Stat vs Spectrogram | 2 | **Spectrogram** | 100% (2/2) |
| Spectrogram vs HOS | 10 | **Spectrogram** | 100% (10/10) |
| Spectrogram vs Cyclo | 7 | **Spectrogram** | 100% (7/7) |
| Baseline vs IQ Stat | 4 | **IQ Stat** | 75% (3/4) |
| IQ Stat vs HOS | 10 | **IQ Stat** | 90% (9/10) |
| IQ Stat vs Cyclo | 8 | **IQ Stat** | 75% (6/8) |
| Baseline vs HOS | 14 | **Baseline** | 71% (10/14) |
| Baseline vs Cyclo | 9 | **Baseline** | 67% (6/9) |
| HOS vs Cyclo | 14 | **Cyclo** | 57% (8/14) |

**Spectrogram is the oracle expert** - it wins 100% of disagreements with every other expert. When in doubt, trust the spectrogram.

### Most Common Misclassifications

| Expert | Error | Count | Why |
|--------|-------|-------|-----|
| Baseline | FRS/GMRS → ISM | 2 | Both bursty UHF signals |
| Baseline | Pager → APRS | 2 | Both sparse packet structure |
| HOS | FRS/GMRS → ISM | 3 | Similar cumulant profiles |
| Cyclo | FRS/GMRS → ISM | 4 | Similar cyclic patterns |
| Cyclo | APRS → Noise | 3 | Sporadic APRS lacks strong cyclostationarity |

---

## 4. Ensemble Method Results

| Method | Accuracy | F1-macro | Notes |
|--------|----------|----------|-------|
| **Oracle (upper bound)** | 100.0% | 1.000 | Best expert per sample |
| **Majority Vote** | 100.0% | 1.000 | Simple, effective |
| **Stacking Meta-Learner** | 100.0% | 1.000 | LR on expert probabilities |
| **Feature Concatenation** | 100.0% | 1.000 | RF on all 158 features |
| **Confidence-Weighted** | 100.0% | 1.000 | Max-prob weighted averaging |
| **Soft Vote** | 99.4% | 0.993 | Probability averaging |
| **Learned Gating (MLP)** | 98.8% | 0.988 | MLP on baseline + expert probs |

### Analysis

1. **Majority vote achieves 100%** - The simplest ensemble matches the oracle. This means the 5 experts rarely all make the same mistake simultaneously.

2. **Stacking works perfectly** - Even a simple logistic regression meta-learner on expert probabilities achieves 100%.

3. **Soft voting slightly underperforms** (99.4%) - Equal-weight probability averaging allows weak experts (HOS, cyclo) to drag down strong ones (spectrogram).

4. **Learned gating underperforms voting** (98.8%) - The MLP gating network overfits on the small training set. With 800 samples, the gating network has insufficient data to learn optimal routing.

5. **Key insight**: For this dataset, the MoE gating mechanism adds no value over simple voting because the spectrogram expert alone is sufficient. MoE gains its advantage when no single expert dominates all classes.

---

## 5. Architecture Improvement Proposals

### Proposal 1: SNR-Aware Feature Selection
- **Idea**: Select top-K features using mutual information instead of using all 158
- **Result**: 50 selected features = 96.3% (vs 100% full) — **worse**
- **Verdict**: Feature selection hurts because it discards complementary information. With RF, more features don't cause overfitting.

### Proposal 2: Gradient Boosted Expert Fusion
- **Idea**: Replace Random Forest with Gradient Boosting for better handling of heterogeneous features
- **Result**: GBM = 94.4% (vs RF = 100%) — **worse**
- **Verdict**: GBM overfits more readily on 800 samples with 158 features. RF's bagging is more robust at this scale.

### Proposal 3: Two-Stage Hierarchical Classification
- **Idea**: First classify broad category (continuous/bursty/noise), then specialize
- **Result**: Stage 1 = 99.4%, Final = 99.4% — **marginal loss**
- **Verdict**: The hierarchy achieves 99.4%, nearly matching flat classification. Useful for interpretability but no accuracy gain. Errors in stage 1 propagate irreversibly.

### Proposal 4: Confidence-Weighted Expert Routing
- **Idea**: Weight each expert's contribution by its prediction confidence (max probability)
- **Result**: 100% — **matches best**
- **Verdict**: This is the most practical RFML-MoE adaptation. It naturally suppresses uncertain experts and amplifies confident ones. Implementable with zero additional training.

---

## 6. Recommendations for RFML-MoE Architecture

### What Works Well in RFML-MoE (Keep)

1. **Multi-modal feature extraction**: The 4-expert design captures genuinely complementary information. Spectrogram, IQ, HOS, and cyclo features each encode different signal properties.

2. **Expert Choice routing**: Letting experts select samples rather than samples selecting experts is elegant. Our confidence-weighted routing achieves similar effect with less complexity.

3. **Cross-attention fusion**: For larger datasets (10K+ samples), cross-attention between expert embeddings would capture inter-modal relationships that simple voting misses.

4. **Progressive training**: The 4-phase training schedule (pretrain → supervised → gating → finetune) is well-motivated. Freezing experts while training the router prevents expert collapse.

### Proposed Changes for RFML-MoE

#### Change 1: Add Spectrogram Expert Bias
The spectrogram expert is the dominant modality. The router should have an inductive bias favoring it:
- Initialize spectrogram expert's routing weight higher than others
- Use a "default to spectrogram, override when others are more confident" strategy
- **Rationale**: Spectrogram wins 100% of disagreements; it should be the fallback

#### Change 2: Replace HOS Expert with Extended IQ Expert
HOS (93.8%) is consistently the weakest expert. The IQFormer or extended IQ statistics (98.8%) provide better discriminative power:
- Swap `FTTransformerHOS` for `IQFormerExpert` (dynamic fusion of IQ + on-the-fly STFT)
- Or: Merge HOS features into the IQ expert as additional input channels
- **Rationale**: HOS cumulants need much larger sample counts for reliable estimation than the practical dataset sizes allow

#### Change 3: Confidence-Gated Routing Instead of Learned Router
Replace the learned routing network with confidence-based gating:
```python
# Instead of: router_logits = Linear(concatenated_embeddings)
# Use: weight each expert by its prediction confidence
for expert in experts:
    confidence = max(expert.predict_proba(x))
    weighted_output += confidence * expert.embedding(x)
```
- **Rationale**: With small-to-medium datasets (<10K samples), learned routers overfit. Confidence gating requires no additional training and achieves 100% on our benchmark.

#### Change 4: Adaptive Expert Depth by Signal Difficulty
Easy signals (FM, noise) don't need all 4 experts. Hard signals (FRS/GMRS, pager) benefit from the full ensemble:
- Use spectrogram-only for high-confidence predictions (>95%)
- Activate additional experts only when spectrogram confidence is low
- **Rationale**: Reduces inference time by ~4x for easy signals while maintaining accuracy on hard ones. Matches the "early exit" paradigm in efficient inference.

#### Change 5: Cyclostationary Feature Redesign
The current SCF extraction is computationally expensive (~1 sec/sample) and produces mixed results (85% APRS, 80% FRS):
- Replace full SCF with targeted cyclic frequency probing at known symbol rates
- For unknown signals, use a lightweight spectral symmetry detector instead
- **Rationale**: The cyclo expert adds value only for signals with known periodicities. A targeted approach is faster and more reliable.

---

## 7. When to Use Which Approach

| Scenario | Recommended Approach | Why |
|----------|---------------------|-----|
| **Edge device, <1K samples** | RTL-ML baseline + spectrogram features | 100% accuracy, 186KB model, <200ms inference |
| **Edge device, real-time** | Baseline RF only (17 features) | 97.5% accuracy, 14ms inference, minimal computation |
| **Server, large dataset (10K+)** | Full RFML-MoE with changes 1-5 | Deep learning experts can leverage data volume |
| **Unknown signal types** | RFML-MoE (all experts) | Different modalities cover different unknowns |
| **Known signal types, high accuracy** | Spectrogram + confidence routing | 100% accuracy, simpler than full MoE |
| **IoT/MCU deployment** | Baseline 17 features only | Fits in <1KB, runs on Cortex-M4 |

---

## 8. Conclusion

The RFML-MoE architecture is well-designed for its target domain (drone RF detection with 50+ classes across diverse SNR conditions). However, for simpler signal classification tasks (7 common radio signal types), the full MoE complexity is unnecessary:

1. **Spectrogram statistical features alone match the full MoE** (100% accuracy)
2. **Simple ensembles (majority vote, stacking) match learned gating** — no need for complex routing
3. **The baseline RTL-ML approach is 97.5% accurate with 17 features** — remarkably effective
4. **HOS features are the weakest link** — consider replacing with extended IQ features
5. **Confidence-weighted routing is the most practical MoE adaptation** — zero additional training, matches oracle performance

The RFML-MoE architecture's value proposition emerges at scale: more classes, noisier conditions, and larger datasets where individual experts genuinely specialize. At the RTL-ML scale, the overhead of training 36-54M parameters across 4 experts is not justified when a 200-tree Random Forest on 37 spectrogram features achieves perfection.

---

*Generated by rfml_comparison.py on RTL-ML dataset (800 samples, 7 classes)*
*All results use temporal 80/20 train/test split with no data leakage*
