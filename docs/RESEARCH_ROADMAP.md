# Research Roadmap: RF-Based Drone Detection & Tracking

## Mission
Build a comprehensive drone detection, classification, and tracking system using RF signals, leveraging multiple datasets and SOTA architectures on MI300X.

---

## Datasets Inventory

| Dataset | Size | Classes | Task | Status |
|---------|------|---------|------|--------|
| **RTL-ML** | 6.2 GB | 7 signal types | Signal classification | Complete, all experiments done |
| **RFUAV** | 263 GB | 37 drones | Drone type identification | Complete, 16 models trained |
| **DroneRFb-DIR** | 65 GB | 13 (6 types × 2 ind + BG) | Individual identification | Training in progress |
| **DroneRFa** | 574 GB | TBD | Companion to DroneRFb | Extracting (61%) |
| **DRFF-R2** | 400 GB | 26 drones, 7 scenarios | Cross-scenario generalization | Downloading (90/730) |

## Experiment Tracks

### Track 1: Spectrogram Classification (COMPLETE for RFUAV, IN PROGRESS for DroneRFb)
- Generate spectrograms from raw IQ (FFT=256, Hamming, Hot colormap)
- Train 16 SOTA models (MaxViT, ConvNeXt, ViT, EfficientNet, YOLO, etc.)
- Best: MaxViT-Base 97.8% on RFUAV

### Track 2: Raw IQ Classification (IN PROGRESS)
- ResNet1D, SE-ResNet1D, CLDNN, MCLDNN directly on IQ tensors
- Scripts written, ready to launch after DroneRFb training completes

### Track 3: Statistical Features (IN PROGRESS)
- 5 modalities: baseline 17, IQ stat 37, spectrogram stat 37, HOS 20, cyclo 64
- RF + GBM + MLP classifiers
- Feature extraction running on RFUAV (140/356 files)

### Track 4: MoE Ensemble (PLANNED)
- Combine spectrogram DL + raw IQ DL + statistical features
- Voting, stacking, confidence routing, expert choice
- Script written, awaiting Track 2+3 completion

### Track 5: SNR Robustness (PLANNED)
- Evaluate all models at -20 to +20 dB SNR (AWGN injection)
- SNR benchmark script ready at /home/rax/mtp/scripts/snr_benchmark.py

### Track 6: Cross-Dataset Generalization (PLANNED)
- Train on RFUAV → test on DroneRFb/DRFF-R2
- Train on DRFF-R2 → test on RFUAV
- Tests whether models learn generalizable drone RF fingerprints

### Track 7: Multi-Scenario Evaluation (PLANNED — needs DRFF-R2)
- 7 scenarios: single drone states, mixed, hover, dual-freq, absorbent cotton, WiFi mixed, environment
- Cross-scenario: train on scenario 1 → test on scenarios 2-7

### Track 8: Detection + Tracking (PLANNED)
- YOLO detection on spectrograms (bounding box around drone signal)
- Track signals across time using spectrogram video frames
- Two-stage: detect → classify → track

### Track 9: Paper Integration (PLANNED)
- Energy Gate + EMD denoising (Tanveer 2026) for low-SNR pre-filtering
- VMD + GAF temporal correlation images (Fu 2026) as new MoE expert
- Time-Frequency Multiscale CNN (Mandal & Satija 2023) as new MoE expert

### Track 10: Edge Deployment (PLANNED)
- MobileNetV3-Large: 97.1% at 4.2M params (RFUAV)
- EfficientNet-B0: 96.2% at 4.1M params
- ONNX export → Raspberry Pi 5 / ESP32-P4 inference

---

## Current Active Tasks (Remote Server)

| tmux | Task | Resource | ETA |
|------|------|----------|-----|
| rfb-train | DroneRFb 6-model training | GPU | ~1hr |
| rfuav-stats | RFUAV statistical features (356 files) | CPU | ~30min |
| extractRFa | DroneRFa unrar (574GB) | Disk | ~30min |
| drffr2 | DRFF-R2 download (400GB, 730 files) | Network | ~hours |

## Priority Queue (Next)

1. Fix DroneRFb test evaluation (class mapping for cross-individual)
2. Launch RFUAV raw IQ training (after DroneRFb GPU frees)
3. Run SNR benchmark on RFUAV models
4. Generate DRFF-R2 spectrograms (when download completes)
5. Cross-dataset generalization experiments
6. Implement paper integration (VMD+GAF, TFMS, Energy Gate)
