# Architecture Improvement Proposals for RFML-MoE

## Experimental Results Summary

All proposals below were tested experimentally on the RTL-ML 800-sample dataset.

| # | Proposal | Accuracy | vs Baseline (97.5%) | vs Combined RF (100%) | Tested? |
|---|----------|----------|--------------------|-----------------------|---------|
| 1 | SNR-Aware Feature Selection (top 50) | 96.3% | -1.2% | -3.7% | Yes |
| 2 | Gradient Boosted Expert Fusion | 94.4% | -3.1% | -5.6% | Yes |
| 3 | Two-Stage Hierarchical Classification | 99.4% | +1.9% | -0.6% | Yes |
| 4 | **Confidence-Weighted Expert Routing** | **100%** | **+2.5%** | **0%** | **Yes** |

## Proposal 1: SNR-Aware Feature Selection (Tested - Not Recommended)

**Problem**: With 158 combined features, some may be redundant or noisy.

**Experiment**: Used `SelectKBest` with mutual information to select top 50 features from all 158, then trained RF.

**Result**: 96.3% accuracy (-3.7% vs combined RF at 100%)

**Verdict**: Feature selection *hurts* because it discards complementary cross-modality information. Random Forest handles high-dimensional data well via bagging; dimensionality reduction is counterproductive here.

**RFML-MoE implication**: Do NOT add feature selection between experts and the fusion layer. Let the router/fusion learn which features matter.

## Proposal 2: Gradient Boosted Expert Fusion (Tested - Not Recommended)

**Problem**: Random Forest may not optimally handle the heterogeneous feature scales across modalities.

**Experiment**: Replaced RF (300 trees) with Gradient Boosting (300 estimators, max_depth=6, lr=0.1) on all 158 combined features.

**Result**: 94.4% accuracy (-5.6% vs combined RF at 100%)

**Verdict**: GBM overfits more readily on 800 samples with 158 features. RF's bagging ensemble is more robust at this data scale. GBM would likely outperform RF with 10K+ samples.

**RFML-MoE implication**: For the full RFML pipeline with large datasets (100K+ samples), gradient boosting or learned fusion (cross-attention) is appropriate. But for small-scale deployment, stick with RF or voting ensembles.

## Proposal 3: Two-Stage Hierarchical Classification (Tested - Marginal)

**Problem**: Flat 7-class classification treats all confusions equally, but some signals are structurally similar (bursty vs continuous).

**Experiment**: Stage 1 classifies broad category (continuous/bursty/noise) at 99.4% accuracy. Stage 2 specializes within each category. Combined result: 99.4%.

**Result**: 99.4% accuracy (-0.6% vs flat RF at 100%)

**Verdict**: Hierarchy adds interpretability but no accuracy gain. Stage 1 errors propagate irreversibly to stage 2. The RFML-MoE's hierarchical classification (binary → type → model) is better suited to its 50-class problem with natural taxonomy.

**RFML-MoE implication**: The 3-level hierarchy (2/15/50 classes) in RFML is well-motivated for drone classification. The cosine loss weight schedule (early coarse → late fine) is the right approach. For simpler problems (<10 classes), flat classification suffices.

## Proposal 4: Confidence-Weighted Expert Routing (Tested - Recommended)

**Problem**: Learned routing (MLP gating) overfits on small datasets: 98.8% vs 100% for simple voting.

**Experiment**: Weight each expert's probability output by its prediction confidence (max probability), then aggregate:
```python
for expert_proba in aligned_probas:
    confidence = np.max(expert_proba, axis=1, keepdims=True)
    weighted_sum += expert_proba * confidence
```

**Result**: 100% accuracy (matches oracle, +2.5% vs baseline)

**Verdict**: This is the most practical MoE adaptation. It naturally suppresses uncertain experts and amplifies confident ones. Requires zero additional training parameters.

**RFML-MoE implication**: Add confidence-gated routing as a fallback to the Expert Choice Router:
```python
def hybrid_route(self, expert_outputs, router_logits):
    router_entropy = -(F.softmax(router_logits) * F.log_softmax(router_logits)).sum(-1)
    confidence_weights = torch.stack([
        expert_out.softmax(-1).max(-1).values for expert_out in expert_outputs
    ], dim=-1)
    # High entropy → use confidence; Low entropy → use learned routing
    alpha = torch.sigmoid(self.entropy_threshold - router_entropy)
    weights = alpha * F.softmax(router_logits) + (1 - alpha) * F.softmax(confidence_weights)
    return weights
```

## Additional Recommendations (from experimental observations, not separately tested)

### A. Spectrogram Expert Bias
The spectrogram expert wins 100% of disagreements with all other experts. Initialize the router with a bias favoring spectrogram as the default expert.

### B. Replace HOS with IQFormer
HOS (93.8%) is the weakest expert. The `IQFormerExpert` (already in RFML codebase) with Dynamic Fusion Embedding would better capture modulation characteristics.

### C. Adaptive Expert Depth
Use spectrogram-only for high-confidence (>95%) predictions; activate other experts only when uncertain. Expected 3-4x inference speedup.

### D. Cyclostationary Feature Redesign
SCF is slow (~1 sec/sample) and fails on sporadic signals (APRS 85%, FRS 80%). Replace with targeted cyclic probing at known symbol rates.

## Implementation Priority

1. **Confidence-Weighted Routing** — Experimentally validated at 100%, zero training cost
2. **Spectrogram Expert Bias** — Easy change, well-supported by disagreement analysis
3. **Adaptive Expert Depth** — Major inference speedup for production deployment
4. **Replace HOS with IQFormer** — Strongest expected accuracy gain on weak classes
5. **Cyclo Redesign** — Optional, reduces extraction latency
