# Model Architectures

## Overview

This directory contains model implementations used in the Drone-RFML-Hub research.

## Directory Structure

```
models/
├── rfml_moe/                      # Full RFML-MoE framework
│   └── rfml/                      # Copied from the RFML-MoE codebase
│       ├── models/
│       │   ├── experts/           # 11 expert architectures
│       │   │   ├── iq_expert.py           # SignalFormerIQ (Complex CNN + Transformer)
│       │   │   ├── spectrogram_expert.py  # EfficientNet-B2 pretrained
│       │   │   ├── hos_expert.py          # FT-Transformer for cumulants
│       │   │   ├── cyclo_expert.py        # Dilated TCN for SCF
│       │   │   ├── lwm_expert.py          # Lightweight Multi-axis (IQ->2D grid)
│       │   │   ├── iqformer_expert.py     # Dynamic Fusion Embedding
│       │   │   ├── hiwavtst_expert.py     # Hierarchical Wavelet TST
│       │   │   ├── tfms_expert.py         # Time-Frequency Multi-Scale
│       │   │   ├── vmdgaf_expert.py       # VMD + GAF images
│       │   │   ├── signalformerrf_expert.py  # SignalFormer RF variant
│       │   │   └── neurosymbolic_expert.py   # Neuro-symbolic RF fingerprinting
│       │   ├── moe/               # Mixture of Experts routing
│       │   │   └── moe_model.py   # DroneRFMoE with Expert/Token Choice routing
│       │   └── fusion/            # Cross-attention fusion layers
│       ├── training/              # Training pipelines
│       ├── evaluation/            # Evaluation utilities
│       ├── features/              # Feature extraction
│       ├── utils/                 # Shared utilities
│       ├── configs/               # YAML training configs
│       ├── scripts/               # Helper scripts
│       └── tests/                 # Unit tests
├── multiscale_lwm_maxvit.py       # Novel architecture (see below)
└── README.md                      # This file
```

## RFML-MoE Architecture

The RFML-MoE (RF Machine Learning - Mixture of Experts) framework implements a multi-expert architecture for drone RF signal classification.

### Expert Architectures

| Expert | Input | Architecture | Params | Best RFUAV Acc |
|--------|-------|-------------|--------|----------------|
| LWMExpert | Raw IQ (2, 32768) | IQ-to-2D grid + multi-axis attention | 1.3M | 94.1% |
| SpectrogramExpert | Spectrogram (3, 512, 512) | EfficientNet-B2 pretrained | ~9M | ~93.7% |
| SignalFormerRFExpert | Raw IQ (2, 32768) | Complex CNN 7 blocks + 4-layer Transformer | ~15M | 85.5% |
| TFMSExpert | Raw IQ (2, 32768) | Time-Frequency Multi-Scale CNN | ~8M | 82.3% |
| HiWaveTSTExpert | Raw IQ (2, 32768) | Hierarchical Wavelet TST | ~5M | 75.9% |
| IQExpert | Raw IQ (2, 32768) | SignalFormerIQ variant | ~20M | ~74% |
| IQFormerExpert | Raw IQ (2, 32768) | Dynamic Fusion Embedding | 13.8M | ~71% |
| NeuroSymbolicRFFExpert | Raw IQ (2, 32768) | Neuro-symbolic RF fingerprinting | ~12M | ~68% |
| HOSExpert | HOS cumulants (20,) | FT-Transformer | 2-5M | -- |
| CycloExpert | SCF features (1, 512) | Dilated TCN 6 layers | 5-10M | -- |
| VMDGAFExpert | GAF images | VMD decomposition + GAF + CNN | ~8M | 35.1% |

### Routing Mechanisms

| Router | Description |
|--------|------------|
| Expert Choice Router | Each expert selects top-c preferred samples (primary) |
| Token Choice Router | Standard top-k per sample (fallback) |
| DeepSeek Router | Auxiliary-loss-free with dynamic bias |
| Soft MoE | Fully differentiable, all experts process soft-weighted inputs |

### Fusion & Classification

- **Cross-Attention Fusion**: 2 layers, 8 heads, 512-dim bidirectional attention
- **Shared Expert**: DeepSeek-style always-active 2-layer MLP (2048 -> 512 -> 512)
- **Hierarchical Classification**: Level 1 (binary: drone/no-drone), Level 2 (type: 15 classes), Level 3 (model: 50 classes)
- **Loss Schedule**: Cosine transition from coarse [0.5, 0.3, 0.2] to fine [0.1, 0.2, 0.7]

### Progressive 4-Phase Training

| Phase | Epochs | Description |
|-------|--------|-------------|
| 1 | 75 | Self-supervised: IQ MAE 75% masking, Spec MoCo-v3 |
| 2 | 75 | Supervised curriculum: SNR curriculum +20 to -10 dB |
| 3 | 35 | Gating network: Experts frozen, train router + fusion |
| 4 | 15 | End-to-end fine-tuning: All params, lr=1e-5 |

## MultiScale-LWM-MaxViT (Novel Architecture)

A novel fusion of LWM's IQ-to-2D-grid insight with MaxViT's multi-axis attention.

### Key Innovations

1. **Multi-scale grid reshape**: 4 grid widths (64, 128, 256, 512) capture different burst periodicities (0.64us to 5.12us at 100 MSps)
2. **MaxViT-style Block+Grid attention** on each scale's 2D grid
3. **Cross-scale fusion** via attention pooling with learned scale importance
4. **MBConv** for local texture extraction (modulation-specific patterns)

### Architecture Summary

| Component | Details |
|-----------|---------|
| Input | Raw IQ (B, 2, 32768) |
| Stem | 1D Conv, 2 -> 64 channels |
| Grid scales | 64x512, 128x256, 256x128, 512x64 |
| Per-scale | MBConv + WindowAttention(7) + GridAttention(7) |
| Fusion | Cross-scale attention pooling |
| Output | (B, num_classes) logits |
| Parameters | 2.44M |
| Target | Beat LWM 94.1%, approach MaxViT 97.8% using raw IQ only |

### Design Rationale

LWMExpert shows that reshaping 1D IQ to 2D grids is powerful (94.1%, 1.3M params). MaxViT-Base shows that multi-axis attention is optimal for 2D RF representations (97.8%, 118.7M params). This architecture combines both insights at a fraction of the parameter cost.

The multi-scale grid captures different temporal periodicities:

| Grid Width | Period at 100 MSps | Physical Meaning |
|------------|-------------------|-----------------|
| 64 | 0.64 us | Intra-symbol modulation |
| 128 | 1.28 us | Symbol-level structure |
| 256 | 2.56 us | Burst envelope patterns |
| 512 | 5.12 us | Protocol-level timing |
