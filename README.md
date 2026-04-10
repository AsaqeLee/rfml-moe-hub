# Drone-RFML-Hub: Comprehensive RF-Based Drone Detection & Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![ROCm 6.3](https://img.shields.io/badge/ROCm-6.3-green.svg)](https://rocm.docs.amd.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A centralized research hub for RF-based drone detection and classification using machine learning. This repository consolidates experiments across **5 datasets**, **30+ model architectures**, and **6 experimental phases** spanning statistical features, deep learning, mixture-of-experts, and novel architectures.

---

## Overview

Radio frequency (RF) signal classification enables passive, non-line-of-sight drone detection by analyzing the electromagnetic emissions from drone communication links. This research systematically compares approaches from simple statistical features to state-of-the-art deep learning across multiple real-world datasets.

### The Core Question

> When does deep learning surpass hand-crafted signal processing features for RF drone classification?

**Answer**: At approximately 1,000--2,000 training samples. Below that threshold, statistical features with Random Forest dominate. Above it, pretrained vision models on spectrograms achieve the best results.

### Key Results

| Dataset | Best Model | Accuracy | Type | Params |
|---------|-----------|----------|------|--------|
| **RFUAV** (37 drones) | MaxViT-Base | **97.8%** | Spectrogram DL | 118.7M |
| **RFUAV** (37 drones) | LWMExpert (raw IQ) | **94.1%** | Raw IQ DL | 1.3M |
| **RFUAV** (37 drones) | Combined RF (stats) | **95.7%** | Statistical | ~200KB |
| **RTL-ML** (7 classes) | Spectrogram RF | **100.0%** | Statistical | ~200KB |
| **RTL-ML** (7 classes) | YOLOv11n-cls | **99.4%** | Spectrogram DL | ~1.6M |
| **DroneRFb** (13 classes) | ConvNeXt-Base | **92.0%** | Spectrogram DL | 87.6M |
| **DroneRFb** (7 types) | SpectrogramExpert | **90.4%** | RFML Expert | ~9M |

### Results Highlights

- **MaxViT-Base achieves 97.8%** on RFUAV 37-class drone identification from spectrograms, beating all 15 other architectures including ViT-L-16 (303M params, 96.9%)
- **LWMExpert reaches 94.1%** on RFUAV raw IQ with only 1.3M parameters -- the most parameter-efficient drone classifier
- **Statistical features hit 100%** on RTL-ML (800 samples) but deep learning wins at scale (3,500+ samples)
- **ConvNeXt-Base at 92.0%** on DroneRFb cross-individual generalization, showing real-world robustness
- **MultiScale-LWM-MaxViT** (new architecture, 2.44M params) fuses LWM's IQ-to-2D insight with MaxViT's multi-axis attention

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │         Drone-RFML-Hub Architecture          │
                        └─────────────────────────────────────────────┘

    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │   Raw IQ     │    │ Spectrograms │    │  Statistical  │    │   HOS/Cyclo  │
    │  (binary)    │    │  (STFT+Hot)  │    │  Features     │    │  Features    │
    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
           │                   │                   │                   │
    ┌──────▼───────┐    ┌──────▼───────┐    ┌──────▼───────┐    ┌──────▼───────┐
    │  LWMExpert   │    │  MaxViT-Base │    │ Random Forest│    │ FT-Transform │
    │  ResNet1D    │    │  ConvNeXt    │    │ GBM / MLP    │    │ Dilated TCN  │
    │  IQFormer    │    │  YOLO/DETR   │    │              │    │              │
    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
           │                   │                   │                   │
           └───────────────────┴───────────┬───────┴───────────────────┘
                                           │
                              ┌─────────────▼─────────────┐
                              │    RFML-MoE Framework      │
                              │  Expert Choice Router      │
                              │  Cross-Attention Fusion    │
                              │  Hierarchical Classifier   │
                              │  (binary→type→model)       │
                              └─────────────┬─────────────┘
                                            │
                              ┌─────────────▼─────────────┐
                              │  Drone Detection Output    │
                              │  - Presence (binary)       │
                              │  - Type (15 classes)       │
                              │  - Model (50 classes)      │
                              └───────────────────────────┘
```

### MultiScale-LWM-MaxViT (Novel Architecture)

```
  Raw IQ (B, 2, 32768)
         │
    ┌────▼────┐
    │ 1D Conv │  Stem: project 2→64 channels
    │ Stem    │
    └────┬────┘
         │
    ┌────▼──────────────────────────────────────────┐
    │         Multi-Scale Grid Reshape               │
    │                                                │
    │  Scale 1: 64×512   (0.64μs burst periods)     │
    │  Scale 2: 128×256  (1.28μs burst periods)     │
    │  Scale 3: 256×128  (2.56μs burst periods)     │
    │  Scale 4: 512×64   (5.12μs burst periods)     │
    └───┬────────┬────────┬────────┬────────────────┘
        │        │        │        │
    ┌───▼──┐ ┌──▼───┐ ┌──▼───┐ ┌──▼───┐
    │MBConv│ │MBConv│ │MBConv│ │MBConv│  Local texture
    │Block │ │Block │ │Block │ │Block │  extraction
    │Attn  │ │Attn  │ │Attn  │ │Attn  │  (modulation)
    │Grid  │ │Grid  │ │Grid  │ │Grid  │
    │Attn  │ │Attn  │ │Attn  │ │Attn  │  Global context
    └───┬──┘ └──┬───┘ └──┬───┘ └──┬───┘
        │       │        │        │
    ┌───▼───────▼────────▼────────▼───┐
    │    Cross-Scale Attention Fusion  │
    │    Learned scale importance      │
    └─────────────┬───────────────────┘
                  │
            ┌─────▼─────┐
            │ Classifier │  2.44M params
            └───────────┘
```

---

## Datasets

| Dataset | Classes | Samples | Size | Frequency | Sample Rate | Task |
|---------|---------|---------|------|-----------|-------------|------|
| [RTL-ML](datasets/rtl_ml.md) | 7 signals | 800 | 6.2 GB | 88-462 MHz | 1.024 MSps | Signal classification |
| [RFUAV](datasets/rfuav.md) | 37 drones | 356 files | 1.3 TB | 5.8 GHz | 100 MSps | Drone identification |
| [DroneRFb-DIR](datasets/droneRFb_dir.md) | 13 (6 types) | 4,690 | 65 GB | 2.4 GHz | 80 MSps | Cross-individual ID |
| [DroneRFa](datasets/droneRFa.md) | TBD | TBD | 574 GB | 2.4 GHz | 80 MSps | Dual-receiver |
| [DRFF-R2](datasets/drffr2.md) | 26 drones | 730 files | 400 GB | Multi | Varies | Multi-scenario |

See [datasets/README.md](datasets/README.md) for detailed documentation of each dataset.

---

## Experiment Phases

### Phase 1: Statistical Feature Engineering (RTL-ML)

Five feature modalities extracted from 800 IQ captures, each classified with Random Forest:

| Modality | Features | Accuracy | Key Insight |
|----------|----------|----------|-------------|
| Spectrogram stats | 37 | **100.0%** | Time-frequency features dominate |
| Combined (all) | 158 | 100.0% | No gain over spectrogram alone |
| IQ statistical | 37 | 98.8% | Kurtosis + crest factor key |
| Baseline RTL-ML | 17 | 97.5% | Original features, solid baseline |
| Cyclostationary | 64 | 95.6% | Fails on sporadic signals |
| HOS cumulants | 20 | 93.8% | High variance at small N |

### Phase 2: Ensemble & MoE Methods (RTL-ML)

| Method | Accuracy | Notes |
|--------|----------|-------|
| Majority Vote | 100.0% | Simplest, matches oracle |
| Stacking | 100.0% | LR meta-learner |
| Confidence Routing | 100.0% | Zero training cost, recommended |
| Soft Vote | 99.4% | Equal weighting suboptimal |
| Learned Gating (MLP) | 98.8% | Overfits at 800 samples |

### Phase 3: Neural Networks (RTL-ML, 11 architectures)

| Model | Input | Accuracy | Params |
|-------|-------|----------|--------|
| ConvNeXt-Tiny | Spectrogram | 98.1% | 703K |
| Lightweight-ViT | Spectrogram | 91.2% | 703K |
| ResNet1D | Raw IQ | 86.9% | 960K |
| IQ-CNN-Transformer | Raw IQ | 80.6% | 549K |
| CLDNN | Raw IQ | 65.0% | 785K |

### Phase 4: YOLO & RT-DETR (RTL-ML)

| Model | Accuracy |
|-------|----------|
| YOLOv11n-cls | 99.4% |
| YOLOv8n-cls | 98.1% |
| ResNet50-DETR | 96.3% |

### Phase 5: RFUAV 37-Class (MI300X, 16 models)

| Model | Accuracy | Params |
|-------|----------|--------|
| MaxViT-Base | 97.8% | 118.7M |
| ConvNeXt-Base | 97.5% | 87.6M |
| EfficientNetV2-L | 97.5% | 117.3M |
| YOLOv11n-cls | 97.4% | ~1.6M |
| MobileNetV3-Large | 97.1% | 4.2M |
| ViT-L-16 | 96.9% | 303.3M |

### Phase 6: RFML Expert Benchmark (RFUAV + DroneRFb)

| Expert | RFUAV Acc | DroneRFb Acc | Input |
|--------|-----------|-------------|-------|
| LWMExpert | 94.1% | -- | Raw IQ |
| SpectrogramExpert | ~93.7% | 90.4% | Spectrogram |
| SignalFormerRF | 85.5% | -- | Raw IQ |
| TFMSExpert | 82.3% | -- | Raw IQ |
| HiWaveTSTExpert | 75.9% | -- | Raw IQ |
| VMDGAFExpert | 35.1% | -- | GAF images |

See [EXPERIMENTS.md](EXPERIMENTS.md) for comprehensive documentation of all experiments.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/r4d10n/drone-rfml-hub.git
cd drone-rfml-hub

# Set up environment (see INSTALL.md for full details)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
pip install timm ultralytics albumentations h5py scipy scikit-learn matplotlib tqdm

# Download RTL-ML dataset (smallest, good for testing)
pip install huggingface-hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('TrevTron/rtl-ml-dataset', local_dir='data/rtl_ml')"

# Run RTL-ML statistical features experiment
python experiments/rtl_ml/rfml_comparison.py

# Run RFUAV spectrogram training (requires MI300X or large GPU)
python preprocessing/rfuav_specgen.py
python experiments/rfuav/train_rfuav.py
```

---

## Repository Structure

```
drone-rfml-hub/
├── README.md                    # This file
├── INSTALL.md                   # Installation & dataset download guide
├── EXPERIMENTS.md               # Comprehensive experiment documentation
├── LICENSE                      # MIT License
├── datasets/                    # Dataset documentation & download scripts
├── preprocessing/               # IQ → spectrogram/feature generation
├── models/                      # Model architectures (RFML-MoE + novel)
├── experiments/                 # Training & benchmark scripts
├── results/                     # Experiment results (JSON)
├── checkpoints/                 # Trained model weights
├── visualizations/              # Generated plots and images
├── docs/                        # Extended documentation & reports
├── logs/                        # Training logs
└── intermediate/                # Intermediate processed files
```

---

## Key Findings

### 1. The Crossover Point

Statistical features with Random Forest achieve 100% on 800 samples but only 95.7% on 3,553 samples (37 classes). Deep learning (MaxViT) reaches 97.8% on the same 3,553 samples. The crossover where DL surpasses statistical methods occurs at approximately 1,000--2,000 samples.

### 2. Spectrogram Dominance

Every top-performing approach uses spectrogram-based input. The STFT encodes both frequency structure and temporal dynamics simultaneously. Hot colormap with FFT=256 and Hamming window is optimal (confirmed across RFUAV and RTL-ML).

### 3. Architecture Insights

- **MaxViT** wins on RFUAV: multi-axis attention (block + grid) combines local CNN features with global Transformer context
- **ConvNeXt** is the reliable workhorse: 97.5% RFUAV, 98.1% RTL-ML, 92.0% DroneRFb
- **MobileNetV3-Large** (4.2M params) at 97.1%: edge deployment champion, within 0.7% of MaxViT at 28x fewer parameters
- **Bigger is not always better**: ConvNeXt-Large (196M) < ConvNeXt-Base (88M); ViT-L-16 (303M) < MaxViT-Base (119M)

### 4. Expert Trust Hierarchy

When experts disagree, the spectrogram expert is always correct. The full hierarchy:

```
Spectrogram (always right) > IQ Statistical (75-90%) > Baseline (67-71%)
    > Cyclostationary (57%) > HOS (least reliable)
```

### 5. Cross-Individual Generalization

On DroneRFb (train on individuals 1&2, test on individual 3), ConvNeXt-Base achieves 92.0% -- significantly harder than within-individual classification but still viable for real-world deployment. Statistical features collapse to 42.3%, confirming that learned representations generalize better across hardware instances.

---

## Citation

If you use this research or codebase, please cite:

```bibtex
@misc{drone-rfml-hub-2026,
  title={Drone-RFML-Hub: Comprehensive RF-Based Drone Detection and Classification},
  author={Rax},
  year={2026},
  url={https://github.com/r4d10n/drone-rfml-hub}
}
```

### Dataset Citations

```bibtex
@article{rfuav2025,
  title={RFUAV: A Benchmark Dataset for UAV Detection and Identification},
  author={Kito et al.},
  journal={arXiv:2503.09033},
  year={2025}
}

@article{droneRFb2025,
  title={Cross-Individual Drone Identification via RF Fingerprinting},
  journal={JEIT},
  year={2025},
  doi={10.11999/JEIT240804}
}

@article{drffr2-2026,
  title={A Multi-Scenario UAV RF Dataset with Real-World Acquisition},
  journal={arXiv:2603.00106},
  year={2026}
}

@dataset{rtl-ml-2026,
  title={RTL-ML Dataset: Real-World RF Signal Classification},
  author={Trevor Unland},
  year={2026},
  url={https://huggingface.co/datasets/TrevTron/rtl-ml-dataset}
}
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [EXPERIMENTS.md](EXPERIMENTS.md) | All experiments across all datasets |
| [docs/FULL_STUDY_REPORT.md](docs/FULL_STUDY_REPORT.md) | Comprehensive 1146-line research report |
| [docs/NN_ARCHITECTURE_REPORT.md](docs/NN_ARCHITECTURE_REPORT.md) | Neural network comparison report |
| [docs/COMPARISON_REPORT.md](docs/COMPARISON_REPORT.md) | Statistical vs DL comparison |
| [docs/expert_analysis.md](docs/expert_analysis.md) | Expert specialization analysis |
| [docs/architecture_proposals.md](docs/architecture_proposals.md) | RFML-MoE improvement proposals |
| [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md) | 10-track research plan |
| [docs/RFUAV_IMPLEMENTATION_PLAN.md](docs/RFUAV_IMPLEMENTATION_PLAN.md) | MI300X training plan |
| [models/README.md](models/README.md) | Model architecture documentation |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Individual datasets have their own licenses -- see each dataset's documentation for details.

---

## IQTLabs GamutRF Comparison (NEW)

Head-to-head benchmark of IQTLabs models vs our RFML-MoE on the same datasets:

### RFUAV 37-Class Drone Identification

| Model | Source | Accuracy | F1 | Params |
|-------|--------|----------|-----|--------|
| MaxViT-Base | **Ours** | **97.8%** | 0.972 | 118.7M |
| ConvNeXt-Base | **Ours** | **97.5%** | 0.970 | 87.6M |
| Combined RF (stats) | **Ours** | **95.7%** | - | ~200KB |
| LWMExpert (raw IQ) | **Ours** | **94.1%** | 0.944 | 1.3M |
| EfficientNet-B0 IQ | IQTLabs | 90.6% | 0.882 | ~5M |
| VGG16 Spectrogram | IQTLabs | ~80% | - | 138M |
| PSD + SVM | IQTLabs | 20.6% | 0.119 | - |
| RFUAV-Net (1D CNN) | IQTLabs | 11.8% | 0.006 | 67M |

Our models outperform IQTLabs by **+3.5% to +86%** across all comparisons.

### Integration with GamutRF

Models are **additive** — not replacing IQTLabs models but extending the model registry:
- Users choose models via YAML config (`gamutrf_models.yml`)
- Mixing modes: single model, voting, weighted, full MoE routing
- TorchServe `.mar` export for direct GamutRF deployment
- See [docs/IQTLABS_GAMUTRF_REPORT.md](docs/IQTLABS_GAMUTRF_REPORT.md) for full integration plan

---

## Survey Documents

- [RF Drone Detection SOTA Models](docs/RF_Drone_Detection_SOTA_Models.md) — Comprehensive survey of state-of-the-art models for RF-based drone detection
- [RF ML Practitioner's Survey](docs/RF_ML_Practitioners_Survey.md) — Technical survey on RF machine learning for drone detection and edge deployment

---

## References & Papers

### Architectures Implemented

| Architecture | Paper | Year | Our Implementation |
|-------------|-------|------|-------------------|
| MaxViT | Tu et al., "MaxViT: Multi-Axis Vision Transformer" (ECCV 2022) | 2022 | `experiments/rfuav/train_rfuav.py` (via timm) |
| ConvNeXt | Liu et al., "A ConvNet for the 2020s" (CVPR 2022) | 2022 | `experiments/rfuav/train_rfuav.py` (via timm) |
| EfficientNet | Tan & Le, "EfficientNet: Rethinking Model Scaling" (ICML 2019) | 2019 | `models/rfml_moe/experts/spectrogram_expert.py` |
| Vision Transformer (ViT) | Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021) | 2021 | `experiments/rfuav/train_rfuav.py` (via timm) |
| Swin Transformer | Liu et al., "Swin Transformer V2" (CVPR 2022) | 2022 | `experiments/rfuav/train_rfuav.py` (via timm) |
| DeiT III | Touvron et al., "DeiT III: Revenge of the ViT" (ECCV 2022) | 2022 | `experiments/rfuav/train_rfuav.py` (via timm) |
| EVA-02 | Fang et al., "EVA-02: A Visual Representation for Neon Genesis" (2023) | 2023 | `experiments/rfuav/train_rfuav.py` (via timm) |
| YOLOv8/v11 | Jocher et al., Ultralytics YOLO (2023-2025) | 2023+ | `experiments/rtl_ml/yolo_detr_comparison.py` |
| RT-DETR | Lv et al., "DETRs Beat YOLOs on Real-time Object Detection" (2023) | 2023 | `experiments/rtl_ml/yolo_detr_comparison.py` |
| MobileNetV3 | Howard et al., "Searching for MobileNetV3" (ICCV 2019) | 2019 | `experiments/rfuav/train_rfuav.py` (via timm) |
| ResNet | He et al., "Deep Residual Learning" (CVPR 2016) | 2016 | `experiments/rtl_ml/nn_comparison.py` |
| VGG16 | Simonyan & Zisserman, "Very Deep Convolutional Networks" (ICLR 2015) | 2015 | `experiments/rfuav/iqtlabs_benchmark.py` |
| InceptionTime | Fawaz et al., "InceptionTime: Finding AlexNet for Time Series" (DMKD 2020) | 2020 | `experiments/rtl_ml/nn_comparison.py` |

### RFML-MoE Expert Architectures

| Expert | Based On | Paper/Reference | Our Implementation |
|--------|----------|-----------------|-------------------|
| LWMExpert | Large Wireless Model 1.1 | Shao et al., "A Wireless Foundation Model" (2024) | `models/rfml_moe/experts/lwm_expert.py` |
| TFMSExpert | Time-Freq Multiscale CNN | Mandal & Satija, "Time-Frequency Multiscale CNN for RF" (2023) | `models/rfml_moe/experts/tfms_expert.py` |
| HiWaveTSTExpert | HiWave-TST | arXiv:2511.01254 "Hierarchical Wavelet TST" (2025) | `models/rfml_moe/experts/hiwavetst_expert.py` |
| VMDGAFExpert | VMD + Gramian Angular Field | Fu et al., "VMD and GAF Empowered CNN for Drone RF" (2026) | `models/rfml_moe/experts/vmd_gaf_expert.py` |
| SignalFormerRFExpert | SignalFormer | Inspired by Dong et al., "SignalFormer" (2024) | `models/rfml_moe/experts/signalformer_expert.py` |
| IQFormerExpert | IQFormer | Li et al., "IQFormer: Multi-Modality Fusion for AMR" (IEEE TCCN 2025) | `models/rfml_moe/experts/iqformer_expert.py` |
| MambaIQExpert | Mamba SSM | Gu & Dao, "Mamba: Linear-Time Sequence Modeling" (2024) | `models/rfml_moe/experts/mamba_iq_expert.py` |
| NeuroSymbolicRFFExpert | Shapelet-based RFF | Inspired by Grabocka et al., "Learning Shapelets" (2014) | `models/rfml_moe/experts/neurosymbolic_rff_expert.py` |
| VisualRFDetector | DETR | Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020) | `models/rfml_moe/experts/visual_rf_detector.py` |
| MultiScale-LWM-MaxViT | LWM + MaxViT fusion | **Novel** (this work) | `models/multiscale_lwm_maxvit.py` |

### Feature Extraction Methods

| Method | Reference | Implementation |
|--------|-----------|---------------|
| Higher-Order Statistics (HOS) | Swami & Sadler, "Hierarchical Digital Modulation Classification" (2000) | `models/rfml_moe/hos.py` |
| Cyclostationary (SCF) | Gardner, "Exploitation of Spectral Redundancy in Cyclostationary Signals" (1991) | `models/rfml_moe/cyclostationary.py` |
| VMD (Variational Mode Decomposition) | Dragomiretskiy & Zosso, "VMD" (IEEE TSP 2014) | `models/rfml_moe/vmd.py` |
| GAF (Gramian Angular Field) | Wang & Oates, "Encoding Time Series as Images" (2015) | `models/rfml_moe/gaf.py` |
| EMD Denoising | Huang et al., "Empirical Mode Decomposition" (1998) | `models/rfml_moe/emd_denoising.py` |
| Energy Gate Detection | Neyman-Pearson lemma (1933) | `models/rfml_moe/energy_gate.py` |
| Wavelet Scattering | Mallat, "Group Invariant Scattering" (CPAM 2012) | `models/rfml_moe/wavelet.py` |
| TorchSig Augmentation | IQTLabs/TorchSig, "RF Signal Augmentation Framework" (GRCon 2024) | External dependency |

### Routing & Ensemble Methods

| Method | Reference | Implementation |
|--------|-----------|---------------|
| Expert Choice Routing | Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (NeurIPS 2022) | `models/rfml_moe/moe/router.py` |
| SNR-Adaptive Routing | **Novel** (this work) — learned per-expert SNR bias | `models/rfml_moe/moe/router.py` |
| DeepSeek Router | Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization" (2024) | `models/rfml_moe/moe/advanced_routing.py` |
| Soft MoE | Puigcerver et al., "From Sparse to Soft Mixtures of Experts" (ICLR 2024) | `models/rfml_moe/moe/advanced_routing.py` |
| FT-Transformer | Gorishniy et al., "Revisiting Deep Learning for Tabular Data" (NeurIPS 2021) | `models/rfml_moe/experts/hos_expert.py` |

### Datasets

| Dataset | Paper/Source | Year |
|---------|-------------|------|
| RFUAV | Ren et al., "RFUAV: A Benchmark Dataset for UAV Detection" (arXiv:2503.09033) | 2025 |
| DroneRFb-DIR | Ren et al., "DroneRFb-DIR: RF Signal Dataset for Drone Individual Identification" (JEIT 2025) | 2025 |
| DRFF-R2 | Zheng & Gao, "A Multi-Scenario UAV RF Dataset" (arXiv:2603.00106) | 2026 |
| RTL-ML | Unland, "Building an AI Radio Scanner with RTL-SDR" (2026) | 2026 |
| DroneRF | Al-Sa'd et al., "DroneRF Dataset" (Data in Brief, 2019) | 2019 |

### IQTLabs Ecosystem

| Tool | Reference | Integration |
|------|-----------|-------------|
| GamutRF | IQTLabs, "SDR Orchestrated Scanner and Collector" (GitHub, archived 2025) | [Integration Plan](docs/IQTLABS_GAMUTRF_REPORT.md) |
| rfml | IQTLabs, "RF Machine Learning Pipeline" (GitHub, archived 2025) | Model export compatibility |
| TorchSig | IQTLabs, "RF Signal Processing Framework" (GRCon 2024-2025) | Augmentation pipeline |
| RFClassification | IQTLabs, "RF-Based Drone Detection" (GitHub) | Benchmark comparison |
| BirdsEye | IQTLabs, "RL-Based RF Target Localization" (GitHub) | Tracking integration |

### Key AMR/Signal Classification Papers

| Paper | Authors | Year | Relevance |
|-------|---------|------|-----------|
| MCLDNN | Wu et al., "Multi-Channel LSTM-DNN" | 2020 | Baseline IQ model (60.8% RadioML) |
| ECDAT | Wang et al., "Efficient Conv Dual-Attention Transformer" | 2024 | Unverified SOTA (95.05% RadioML) |
| IQFormer | Li et al., "Multi-Modality Fusion for AMR" | 2025 | Multi-modal IQ+TF fusion (68.5% RadioML) |
| ConvMamba | "Modulation Recognition Based on Mamba" | 2025 | First SSM for AMR |
| IQFM | "A Wireless Foundation Model for IQ Streams" | 2025 | Foundation model (99.67% few-shot) |
| GAF-MAE | "Reconstruction-Driven ViT for AMR Under Limited Labels" | 2025 | Semi-supervised (5% labeled) |
| RF-YOLO | "Modified YOLO for UAV Detection using RF Spectrograms" | 2025 | mAP 0.9213 on spectrograms |

---
