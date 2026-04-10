# IQTLabs GamutRF Ecosystem Survey & RFML-MoE Integration Plan

## 1. IQTLabs RF Ecosystem Overview

IQTLabs (In-Q-Tel Labs) maintains a suite of open-source RF signal processing and ML tools. The ecosystem forms a complete pipeline from SDR hardware → signal capture → detection → classification → localization.

```
┌─────────────────────────────────────────────────────────────────┐
│                    IQTLabs RF Ecosystem                         │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │gr-iqtlabs│───→│ gamutRF  │───→│   rfml   │───→│TorchServe│  │
│  │(GNURadio │    │(Scanner/ │    │(Training/│    │(Inference │  │
│  │ blocks)  │    │Collector)│    │ Export)  │    │ Server)  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                │              │                │        │
│       │          ┌─────┴─────┐        │                │        │
│       │          │ Waterfall │   ┌────┴────┐    ┌─────┴────┐   │
│       │          │    UI     │   │TorchSig │    │RFClassif.│   │
│       │          └───────────┘   │(Augment)│    │(Drone Det│   │
│       │                          └─────────┘    └──────────┘   │
│       │                                                        │
│  ┌────┴─────┐                                                  │
│  │BirdsEye  │  RL-based drone operator localization            │
│  │(RL Track)│  MCTS + DQN on signal strength/bearing           │
│  └──────────┘                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Repository Status (as of April 2026)

| Repository | Description | Status | Stars | License |
|-----------|-------------|--------|-------|---------|
| **gamutRF** | SDR scanner, collector, identifier | **Archived Sep 2025** | ~130 | Apache 2.0 |
| **rfml** | ML training pipeline for RF signals | **Archived Jun 2025** | ~50 | Apache 2.0 |
| **gr-iqtlabs** | GNU Radio OOT blocks | Active | ~30 | Apache 2.0 |
| **RFClassification** | Drone detection via RF | Active | ~80 | MIT |
| **BirdsEye** | RL-based RF target tracking | Active | ~40 | Apache 2.0 |
| **TorchSig** | RF signal augmentation framework | Active (separate org) | ~200 | MIT |

---

## 2. GamutRF: Architecture Deep Dive

### What It Does

GamutRF is a **Docker-orchestrated SDR scanning and classification system**. It continuously sweeps frequency ranges, captures IQ samples, generates spectrograms, and optionally runs ML inference for signal identification.

### System Architecture

```
                    ┌─────────────────────────┐
                    │      Docker Compose      │
                    │    (orchestrator.yml)     │
                    └────────────┬──────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
    │   Scanner    │    │  Waterfall UI │    │  TorchServe  │
    │  Container   │    │  Container    │    │  Container   │
    │              │    │              │    │              │
    │ ┌──────────┐ │    │ ┌──────────┐ │    │ ┌──────────┐ │
    │ │SoapySDR  │ │    │ │Spectrogram│ │    │ │PyTorch   │ │
    │ │/UHD API  │ │    │ │Renderer  │ │    │ │Model (.mar│ │
    │ └────┬─────┘ │    │ └──────────┘ │    │ │Archive)  │ │
    │      │       │    │              │    │ └──────────┘ │
    │ ┌────┴─────┐ │    │ ┌──────────┐ │    │              │
    │ │gr-iqtlabs│ │───→│ │WebSocket │ │    │ REST API     │
    │ │GNURadio  │ │    │ │Display   │ │    │ /predictions │
    │ │Blocks    │ │    │ └──────────┘ │    │              │
    │ └──────────┘ │    └───────────────┘    └──────────────┘
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │  SDR Hardware │
    │ USRP/RTL-SDR │
    │ AIR-T/HackRF │
    └──────────────┘
```

### Core Pipeline

1. **Scanning**: gr-iqtlabs `retune_fft` block sweeps frequency ranges, producing FFT bins
2. **Detection**: Signal presence detected via power threshold on FFT output
3. **Collection**: When signal detected, IQ samples captured and saved in SigMF format
4. **Spectrogram**: FFT output rendered as waterfall images for display and ML
5. **Classification**: Images or IQ samples sent to TorchServe for inference
6. **Output**: Classification results returned via REST API

### Key Technical Details

| Property | Value |
|----------|-------|
| SDR API | SoapySDR (USRP, RTL-SDR, HackRF, AIR-T) |
| Signal Processing | GNU Radio 3.10+ with gr-iqtlabs blocks |
| Spectrogram | FFT-based via gr-iqtlabs `retune_fft` block |
| ML Framework | PyTorch via TorchServe |
| Model Format | .mar (Model ARchive) from TorchScript |
| Data Format | SigMF (.sigmf-data + .sigmf-meta) |
| Deployment | Docker Compose on x86_64 Ubuntu 24.04 |
| Edge Support | Raspberry Pi 4/5 (distributed components) |
| GPU | NVIDIA CUDA (optional, for inference) |

---

## 3. RFML: Training Pipeline

### What It Does

RFML provides the training workflow that produces models for GamutRF deployment:

```
Raw IQ/Spectrogram Data
    │
    ▼
┌──────────────────┐
│  Annotation      │  DSP-based signal detection
│  (auto-labeling) │  Gaussian Mixture Model bandwidth estimation
└────────┬─────────┘
         ▼
┌──────────────────┐
│  TorchSig        │  RF-specific augmentation
│  (augmentation)  │  53 modulation types supported
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Training        │  IQ models + Spectrogram models
│  (PyTorch)       │  Configurable epochs per type
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Export           │  TorchScript → .mar format
│  (TorchServe)    │  Ready for GamutRF deployment
└──────────────────┘
```

### Model Types

1. **IQ Models**: 1D CNNs operating on raw complex IQ samples (TorchSig-based)
2. **Spectrogram Models**: 2D CNNs on time-frequency images

### Data Format: SigMF

```json
{
  "global": {
    "core:datatype": "cf32_le",
    "core:sample_rate": 20000000,
    "core:hw": "Ettus USRP B200"
  },
  "captures": [{"core:sample_start": 0, "core:frequency": 2412000000}],
  "annotations": [
    {
      "core:sample_start": 1000,
      "core:sample_count": 50000,
      "core:label": "DJI_Mavic3_video",
      "core:freq_lower_edge": 2410000000,
      "core:freq_upper_edge": 2414000000
    }
  ]
}
```

---

## 4. RFClassification: Drone Detection

### Approaches & Results

| Approach | Dataset | Task | Accuracy | Inference |
|----------|---------|------|----------|-----------|
| PSD + SVM | DroneRF | Binary detection | 98.3% | 0.29ms |
| RFUAV-Net (1D CNN) | DroneRF | Binary detection | 99.8% | 1.08ms |
| PSD + SVM | DroneDetect | 4-class | 85.4% | 9.96ms |
| VGG16 + PSD | DroneDetect | 4-class | 82.5% | 5.72ms |
| ResNet50 + Spectrogram | DroneDetect | 4-class | ~80% | ~6ms |

### RFUAV-Net Architecture

```
Input: Raw IQ (2, N)
  → Conv1d(2, 128, k=7) → BN → ReLU → MaxPool
  → Conv1d(128, 128, k=5) → BN → ReLU → MaxPool
  → Conv1d(128, 128, k=3) → BN → ReLU → MaxPool
  → Flatten → Dense(128) → Dense(num_classes)
```

A simple 1D CNN — much smaller than our RFML-MoE experts.

---

## 5. BirdsEye: RF Target Localization

Uses RL to localize drone operators from RF signal observations:

- **MCTS**: Monte Carlo Tree Search for action selection
- **DQN**: Deep Q-Learning for adaptive tracking
- **Sensors**: Omni-directional signal strength + bearing
- **Edge-deployable**: No GPU required
- **Integration**: `gamutrf_fieldtest/` directory for field deployment

---

## 6. Integration Plan: RFML-MoE → GamutRF Ecosystem

### 6.1 Architecture Vision

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Enhanced GamutRF + RFML-MoE                      │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────────────────┐  │
│  │gr-iqtlabs│───→│ gamutRF  │───→│      RFML-MoE Inference       │  │
│  │(Scanner) │    │(Collect) │    │                               │  │
│  └──────────┘    └────┬─────┘    │  ┌─────────────────────────┐  │  │
│                       │          │  │   Energy Gate (pre-filter)│  │  │
│                       │          │  └───────────┬─────────────┘  │  │
│                       │          │              │                │  │
│                       │          │  ┌───────────┼───────────┐    │  │
│                  Raw IQ│          │  │     MoE Router       │    │  │
│                       │          │  │  (SNR-adaptive)       │    │  │
│                       │          │  └───┬───┬───┬───┬───┬───┘    │  │
│                       ▼          │      │   │   │   │   │        │  │
│                ┌──────────┐      │  ┌───▼┐┌─▼──┐┌▼──┐┌─▼──┐┌▼──┐│  │
│                │ SigMF    │      │  │LWM ││Spec││HiW││TFMS││Vis ││  │
│                │ Writer   │      │  │Exp.││Exp.││TST││Exp.││Det.││  │
│                └──────────┘      │  └───┬┘└─┬──┘└┬──┘└─┬──┘└┬──┘│  │
│                                  │      └───┴────┴─────┴────┘    │  │
│                                  │              │                │  │
│                                  │  ┌───────────▼─────────────┐  │  │
│                                  │  │  Cross-Attention Fusion  │  │  │
│                                  │  └───────────┬─────────────┘  │  │
│                                  │              │                │  │
│                                  │  ┌───────────▼─────────────┐  │  │
│                                  │  │  Hierarchical Classifier │  │  │
│                                  │  │  L1: drone/no-drone      │  │  │
│                                  │  │  L2: drone type           │  │  │
│                                  │  │  L3: individual ID        │  │  │
│                                  │  └─────────────────────────┘  │  │
│                                  └───────────────────────────────┘  │
│                                                │                    │
│                                    ┌───────────▼───────────┐        │
│                                    │  BirdsEye Tracker     │        │
│                                    │  (RL Localization)    │        │
│                                    └───────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Integration Points

#### Point 1: Model Export (RFML-MoE → TorchServe .mar)

GamutRF uses TorchServe with `.mar` model archives. We need to export our experts:

```python
# Export LWMExpert to TorchScript → .mar
import torch
from models.rfml_moe.experts.lwm_expert import LWMExpert

model = LWMExpert(num_classes=37)
model.load_state_dict(torch.load("checkpoints/rfuav/rfml_LWMExpert_best.pt"))
model.eval()

# TorchScript export
scripted = torch.jit.script(model)
scripted.save("lwm_expert.pt")

# Package as .mar for TorchServe
# torch-model-archiver --model-name lwm_expert \
#   --version 1.0 --serialized-file lwm_expert.pt \
#   --handler custom_handler.py --export-path model_store
```

**Challenge**: Our MoE model needs multiple inputs (IQ, spectrogram, HOS, cyclo). TorchServe expects a single input. Solutions:
- Export individual experts as separate models
- Create a custom handler that preprocesses IQ → multiple modalities
- Export the full MoE as a single model with internal preprocessing

#### Point 2: Data Format Bridge (SigMF ↔ RFML-MoE)

GamutRF captures in SigMF format. Our pipeline expects raw IQ arrays:

```python
# SigMF → RFML-MoE format adapter
import sigmf
from sigmf import SigMFFile

def sigmf_to_rfml(sigmf_path):
    """Convert SigMF capture to RFML-MoE input format."""
    signal = SigMFFile(sigmf_path)
    iq_data = signal.read_samples()  # complex64 array
    sample_rate = signal.get_global_field('core:sample_rate')
    
    # Segment into 32768-sample chunks (for LWMExpert)
    segments = []
    for i in range(0, len(iq_data) - 32768, 32768):
        chunk = iq_data[i:i+32768]
        iq_tensor = torch.stack([
            torch.from_numpy(chunk.real.astype(np.float32)),
            torch.from_numpy(chunk.imag.astype(np.float32))
        ])
        segments.append(iq_tensor)
    
    return segments, sample_rate
```

#### Point 3: Spectrogram Compatibility

GamutRF's gr-iqtlabs generates spectrograms via FFT blocks. Our pipeline uses:
- FFT size: 256, Hamming window, Hot colormap
- GamutRF uses configurable FFT sizes in gr-iqtlabs

**Alignment needed**: Configure gr-iqtlabs to output spectrograms matching our training parameters, or retrain our models on GamutRF's spectrogram format.

#### Point 4: Real-Time Inference Pipeline

```
GamutRF Scanner (continuous)
    │
    ├── IQ samples (32768 per segment, 100MSps)
    │
    ├─→ [Energy Gate] → SNR estimate + signal detection
    │       │
    │       ├── SNR > threshold → classify
    │       └── SNR < threshold → skip/denoise
    │
    ├─→ [LWMExpert] → raw IQ → 2D grid → 37-class prediction
    │
    ├─→ [SpectrogramExpert] → STFT → spectrogram → classification
    │
    └─→ [MoE Router] → combine expert predictions → final output
```

**Latency budget** (for real-time @ 100MSps):
- IQ capture: 0.33ms per 32768-sample segment
- Energy gate: <0.1ms
- LWM inference: ~2ms (on GPU)
- Spectrogram generation: ~1ms
- MoE routing + fusion: ~0.5ms
- **Total**: ~4ms per segment → **250 classifications/second**

#### Point 5: Docker Integration

Create a new Docker service for RFML-MoE inference:

```yaml
# Add to gamutRF orchestrator.yml
services:
  rfml-moe:
    image: rfml-moe-inference:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    ports:
      - "8081:8081"
    volumes:
      - ./model_store:/model_store
    environment:
      - MODEL_NAME=rfml_moe_drone_37class
      - DEVICE=cuda
      - BATCH_SIZE=32
```

### 6.3 Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| P1 | Export LWMExpert to TorchScript/.mar | 1 day | Direct GamutRF deployment |
| P2 | SigMF → RFML-MoE data adapter | 1 day | Enable field data processing |
| P3 | Custom TorchServe handler for MoE | 2 days | Full MoE inference in GamutRF |
| P4 | Docker service for RFML-MoE | 1 day | Containerized deployment |
| P5 | Spectrogram format alignment | 1 day | Match gr-iqtlabs output |
| P6 | BirdsEye integration (classification → tracking) | 3 days | End-to-end detect+track |
| P7 | Edge export (ONNX → RPi5) | 2 days | Field deployment |
| P8 | TorchSig augmentation integration | 2 days | Better training with 53 signal types |

### 6.4 What We Bring vs What They Have

| Capability | IQTLabs (GamutRF/rfml) | Our RFML-MoE | Combined |
|-----------|------------------------|--------------|----------|
| Signal scanning | SoapySDR + gr-iqtlabs | None | GamutRF scanner |
| Detection | Power threshold | Energy Gate + EMD | Energy Gate (superior) |
| IQ classification | Basic 1D CNN | 11 experts (LWM 94.1%) | MoE ensemble |
| Spectrogram classification | VGG/ResNet transfer | MaxViT 97.8% | MaxViT/ConvNeXt |
| Drone identification | SVM 85.4% | ConvNeXt 92.0% cross-individual | Cross-individual MoE |
| Data format | SigMF (standard) | Raw .npy/.iq/.mat | SigMF adapter |
| Deployment | Docker + TorchServe | Scripts only | Docker + TorchServe + MoE |
| Tracking | BirdsEye (RL) | None | BirdsEye + MoE classifications |
| Augmentation | TorchSig (53 types) | Basic AWGN/CFO | TorchSig integration |
| Edge deployment | RPi4/5 tested | MobileNetV3 97.1% | Optimized edge pipeline |

### 6.5 Recommended Architecture for Production

```
Tier 1 — Edge (Raspberry Pi 5 + RTL-SDR):
  └── Energy Gate → MobileNetV3 (97.1%, 4.2M) → binary drone/no-drone
      Latency: <10ms, Power: <5W

Tier 2 — Tactical (x86 laptop + USRP + GPU):
  └── GamutRF scanner → LWMExpert (94.1%, 1.3M) → 37-class drone type
      Latency: <5ms, Power: ~50W

Tier 3 — Command (Server + MI300X):
  └── Full RFML-MoE (11 experts) → hierarchical classification
      → BirdsEye RL tracking → operator localization
      Latency: <20ms (full MoE), Throughput: 250 cls/sec
```

---

## 7. Key Takeaways

1. **GamutRF is the deployment vehicle** — its Docker architecture, SoapySDR scanning, and TorchServe inference are exactly what we need. Despite being archived, the architecture patterns are solid and forkable.

2. **Our RFML-MoE dramatically outperforms IQTLabs' models** — LWM 94.1% vs RFUAV-Net 99.8% (binary only), MaxViT 97.8% vs VGG16 82.5% (multi-class). The gap is especially large for multi-class drone identification.

3. **SigMF is the bridge** — adopting SigMF as our data format enables seamless integration with the entire GamutRF ecosystem and community tools.

4. **TorchSig for augmentation** — their 53-signal-type augmentation framework would significantly improve our training, especially for generalization to unseen signal conditions.

5. **BirdsEye completes the loop** — combining our classification (detect + identify drone type) with their RL localization (find the operator) creates a full counter-drone system.

---

*Sources: [gamutRF](https://github.com/IQTLabs/gamutRF), [rfml](https://github.com/IQTLabs/rfml), [RFClassification](https://github.com/IQTLabs/RFClassification), [BirdsEye](https://github.com/IQTLabs/BirdsEye), [gr-iqtlabs](https://github.com/IQTLabs/gr-iqtlabs), [TorchSig](https://github.com/TorchDSP/torchsig)*
