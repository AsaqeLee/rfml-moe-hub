# Complete Session Context — RF Signal Classification Research

**Date**: March-April 2026
**Repo**: https://github.com/r4d10n/rtl-ml-exp (private)
**Local**: /home/rax/exp/esp32/p4/host/ml/rtl-ml-exp
**Remote Server**: rax@129.212.188.94 (MI300X GPU, 5TB storage at /home/rax/mtp)
**RFML-MoE Codebase**: /home/rax/exp/iq/rfml

---

## 1. Project Overview

End-to-end research comparing RF signal classification approaches — from simple statistical features to SOTA deep learning — across multiple datasets. The goal is to evaluate, improve, and extend the RFML-MoE (Mixture of Experts) architecture for drone RF detection and general signal classification.

---

## 2. Datasets

### 2.1 RTL-ML (Primary experimental dataset)
- **Source**: TrevTron/rtl-ml-dataset on HuggingFace
- **Location**: /home/rax/exp/esp32/p4/host/ml/rtl-ml-exp/datasets_validated (symlink → /opt1/ml/rtl-ml-exp/datasets_validated)
- **Size**: 800 samples, 6.2 GB
- **Format**: .npy dictionaries with keys: samples (512K complex128), center_freq, sample_rate, timestamp, label, snr_db, version
- **Sample Rate**: 1.024 MSps (RTL-SDR Blog V4)
- **Classes**: 7 — APRS (144.39 MHz), FM_broadcast (88.5-105.7 MHz, 200 samples across 5 freqs), FRS_GMRS (462.5625 MHz), ISM_sensors (433.92 MHz), NOAA_weather (162.4 MHz), noise (145.0 MHz), pager (152.84 MHz)
- **Split**: Temporal 80/20 per class (640 train, 160 test) or 64/16/20 for DL (512/128/160)
- **Quality**: DC offset removed, 6 dB min SNR gate, per-class validation

### 2.2 RFUAV (Drone RF classification — large scale)
- **Source**: kitofrank/RFUAV on HuggingFace + github.com/kitoweeknd/RFUAV
- **Location (remote)**: /home/rax/mtp/raw/ (263 GB extracted), /home/rax/mtp/spectrograms/ (3553 images)
- **Size**: 102 GB compressed (37 .rar files), 263 GB extracted raw IQ
- **Format**: Binary IQ fp32 interleaved (.iq files), XML metadata per pack
- **Sample Rate**: 100 MSps (USRP X310)
- **Frequency**: 5.765 GHz (5.8 GHz ISM band)
- **Classes**: 37 drone/RC types (DJI AVATA2, FPV COMBO, MAVIC3 PRO, MINI3/4 PRO, FLYSKY, FRSKY, FUTABA, Herelink, JR PROPO, JUMPER, RadioMaster, Radiolink, SIYI, SKYDROID, WFLY, YUNZHUO)
- **Structure**: Each drone → subfolders by VTS bandwidth (10/20/40/60 MHz) → .iq files segmented to 1-second chunks (~763 MB each)
- **Paper**: arXiv 2503.09033 — RFUAV: A Benchmark Dataset for UAV Detection and Identification
- **Code**: /home/rax/mtp/RFUAV/ (cloned from github)
- **Spectrograms generated**: 3553 total (2623 train, 890 val) using FFT=256, Hamming, Hot colormap, 1M samples/spec

### 2.3 DroneRFb-DIR (Drone individual identification)
- **Source**: scidb.cn, dataSetId=84cf9101e739402784b1396783881202
- **Location (remote)**: /home/rax/mtp/droneRFb/extracted/ (65 GB)
- **Size**: 64 GB compressed (32-part split zip), 65 GB extracted
- **Format**: MATLAB v7.3 (HDF5) .mat files with keys I (1×4000000 float32) and Q (1×4000000 float32)
- **Sample Rate**: 80 MSps (SDR)
- **Frequency**: 2.4-2.48 GHz
- **Classes**: 13 — 6 drone types (A-G) × 2 individuals (1,2) + B (background). Train uses individuals 1&2, test uses individual 3 (cross-identity generalization)
- **Signal types**: FCS (Flight Control) + VTS (Video Transmission)
- **Conditions**: LOS and NLOS, indoor and outdoor
- **Structure**: twin_droneRF/{train,test}/ with {train,test}_labels.txt
- **Train**: 2177 files, **Test**: 2513 files (total 4690 segments)
- **Label format**: Train: `filename class_name`, Test: `filename index`
- **Paper**: JEIT 2025, DOI 10.11999/JEIT240804

### 2.4 DroneRFa (Companion dataset to DroneRFb)
- **Location (remote)**: /home/rax/mtp/droneRFa/ (574 GB, downloading)
- **Download URL**: china.scidb.cn, fileId=c403fc76444e4b9989e4f3ff570f3b3d

### 2.5 DRFF-R2 (Multi-scenario UAV RF dataset)
- **Source**: scidb.cn, dataSetId=b8a16448c1284fd1be1ded9ccc45be20
- **Location (remote)**: /home/rax/mtp/drffr2/ (14 GB downloaded so far, 400 GB total)
- **Format**: MATLAB .mat files
- **Size**: 400.6 GB, 730 files across 7 scenario datasets
- **Classes**: 26 drones across 8 models (mavic3, mavic3C, mavic3S, mavicAir2, mavicAir2s, mini3pro, mini4PRO, mini5PRO)
- **Scenarios**: dataset1 (single drone states: cruise/ascend/descend/takeoff/landing/shading), dataset2 (drone mixed), dataset3 (hover), dataset4 (dual frequency), dataset5 (inside absorbent cotton), dataset6 (WiFi mixed), dataset7 (environment)
- **Code**: /V3/code/ — Technical_Validation (experiment1.m, experiment2.py), signal_collection (signal_collection1.py, signal_collection2.py)
- **Paper**: arXiv 2603.00106 — A Multi-Scenario UAV RF Dataset with Real-World Acquisition
- **SciDB API**: gin-sdb-filetree/public/file/childrenFileListByPath (POST, auth via Bearer token)
- **Download pattern**: https://china.scidb.cn/download?fileId={id}&username=...&traceId=...
- **File list saved**: /tmp/drffr2_files.json (730 entries with IDs)

---

## 3. RFML-MoE Architecture (from /home/rax/exp/iq/rfml)

### 3.1 Four Expert Architecture
| Expert | Model | Input | Output | Params |
|--------|-------|-------|--------|--------|
| IQ Expert | SignalFormerIQ (Complex CNN 7 blocks + 4-layer Transformer) | (B, 2, 32768) | (B, 512) | 15-25M |
| Spectrogram Expert | EfficientNet-B2 (pretrained ImageNet) | (B, 3, 512, 512) | (B, 512) | 9M |
| HOS Expert | FT-Transformer (per-feature tokenization) | (B, 20) | (B, 512) | 2-5M |
| Cyclo Expert | Dilated TCN (6 layers, dilations 1-32) | (B, 1, 512) | (B, 512) | 5-10M |

### 3.2 Routing Mechanisms
- **Expert Choice Router** (primary): Each expert selects top-c preferred samples
- **Token Choice Router** (fallback): Standard top-k per sample
- **DeepSeek Router**: Auxiliary-loss-free with dynamic bias
- **Soft MoE**: Fully differentiable, all experts process soft-weighted inputs

### 3.3 Fusion & Classification
- **Cross-Attention Fusion**: 2 layers, 8 heads, 512-dim bidirectional attention between experts
- **Shared Expert**: DeepSeek-style always-active 2-layer MLP (2048→512→512)
- **Hierarchical Classification**: Level 1 (binary: drone/no-drone), Level 2 (type: 15 classes), Level 3 (model: 50 classes)
- **Loss Schedule**: Cosine transition [0.5, 0.3, 0.2] early → [0.1, 0.2, 0.7] late

### 3.4 Progressive 4-Phase Training
- Phase 1: Self-supervised (75 ep) — IQ: MAE 75% masking, Spec: MoCo-v3
- Phase 2: Supervised curriculum (75 ep) — SNR curriculum +20→-10 dB, individual expert training
- Phase 3: Gating network (35 ep) — Experts frozen, train router + fusion + shared expert
- Phase 4: End-to-end fine-tuning (15 ep) — All params, lr=1e-5

### 3.5 Alternative IQ Experts in RFML Codebase
- **IQFormer**: Dynamic Fusion Embedding (IQ branch + on-the-fly STFT branch), 13.8M params
- **MambaIQExpert**: Selective SSM with HiPPO-LegS init, soft-threshold denoisers, O(L) complexity, 11-14M params

---

## 4. Experiments & Results

### 4.1 Phase 1: Statistical Feature Engineering (RTL-ML, 800 samples)

**5 feature modalities tested** (all with Random Forest 200 trees):

| Modality | Features | Accuracy | Key Features |
|----------|----------|----------|-------------|
| **Spectrogram** | 37 | **100.0%** | Spectral centroid/bandwidth/rolloff/flatness, band energies, temporal envelope |
| Combined | 158 | 100.0% | All modalities concatenated |
| IQ Statistical | 37 | 98.8% | Kurtosis, crest factor, autocorrelation, zero-crossing, envelope |
| Baseline RTL-ML | 17 | 97.5% | Power stats, FFT stats, I/Q stats, phase stats, bandwidth ratio |
| Cyclostationary | 64 | 95.6% | SCF peak/mean at 32 cycle frequencies |
| HOS Cumulants | 20 | 93.8% | C20-C63 normalized, ratios, phases |

**Feature importance** (baseline 17): power_max (0.155) > phase_diff_std (0.139) > q_std (0.103) > power_mean (0.097)

**Expert disagreement hierarchy**: Spectrogram (always right) > IQ Stat (75-90%) > Baseline (67-71%) > Cyclo (57%) > HOS (least reliable)

### 4.2 Phase 2: Ensemble/MoE Methods (RTL-ML)

| Method | Accuracy | Notes |
|--------|----------|-------|
| Majority Vote | 100.0% | Simplest, matches oracle |
| Stacking (LR meta-learner) | 100.0% | |
| Feature Concatenation (RF 300 trees) | 100.0% | |
| Confidence-Weighted Routing | 100.0% | Zero training cost, recommended |
| Soft Vote | 99.4% | Equal weighting hurts |
| Learned Gating (MLP) | 98.8% | Overfits on 800 samples |

**Architecture improvements tested**:
- Feature selection (top 50): 96.3% — harmful, discards complementary info
- Gradient Boosting: 94.4% — overfits at this scale
- Hierarchical (broad→fine): 99.4% — marginal
- **Confidence routing: 100%** — best practical MoE adaptation

### 4.3 Phase 3: Neural Network Architectures (RTL-ML, GPU: GTX 1660 Ti 6GB)

**11 DL models trained** with RF-aware augmentation (AWGN, CFO, time shift, amplitude scaling):

| Model | Type | Accuracy | Params |
|-------|------|----------|--------|
| ConvNeXt-Tiny-Spec | DL-CNN | 98.1% | 703K |
| Lightweight-ViT-Spec | DL-ViT | 91.2% | 703K |
| Spectrogram-CNN | DL-CNN | 89.4% | 1.4M |
| ResNet1D | DL-IQ | 86.9% | 960K |
| IQ-CNN-Transformer | DL-IQ | 80.6% | 549K |
| FT-Transformer-HOS | DL-Transformer | 76.9% | 105K |
| InceptionTime-1D | DL-IQ | 76.2% | 458K |
| SE-ResNet1D | DL-IQ | 73.8% | 1.0M |
| CLDNN | DL-IQ | 65.0% | 785K |
| MCLDNN | DL-IQ | 63.1% | 231K |
| Dilated-TCN-Cyclo | DL-TCN | 55.0% | 159K |

### 4.4 Phase 4: YOLO & RT-DETR (RTL-ML, RFUAV-style spectrograms)

| Model | Accuracy | Notes |
|-------|----------|-------|
| **YOLOv11n-cls** | **99.4%** | Best DL model overall on RTL-ML |
| YOLOv8n-cls | 98.1% | Ties with ConvNeXt |
| ResNet50-DETR-backbone | 96.3% | RT-DETR adapted for classification |

### 4.5 Phase 5: RFUAV 37-Class Training (Remote, GPU: MI300X 206GB VRAM)

**3553 spectrograms** (2623 train, 890 val) generated from 356 .iq files (FFT=256, Hamming, Hot colormap, 1M samples/spec).

**Training config**: AdamW lr=1e-4, BF16 mixed precision, label smoothing 0.1, CosineAnnealing, early stopping patience 20.

| Rank | Model | Accuracy | F1 | Params | Time |
|------|-------|----------|-----|--------|------|
| 1 | **MaxViT-Base** | **97.8%** | 0.972 | 118.7M | 722s |
| 2 | ConvNeXt-Base | 97.5% | 0.970 | 87.6M | 281s |
| 2 | EfficientNetV2-L | 97.5% | 0.969 | 117.3M | 914s |
| 4 | YOLOv11n-cls | 97.4% | 0.967 | ~1.6M | - |
| 5 | ConvNeXt-Large | 97.1% | 0.964 | 196.3M | 586s |
| 5 | MobileNetV3-Large | 97.1% | 0.964 | 4.2M | 577s |
| 7 | YOLOv11s-cls | 97.2% | 0.965 | ~5M | - |
| 8 | ViT-L-16 | 96.9% | 0.962 | 303.3M | 437s |
| 9 | DeiT3-Base | 96.6% | 0.959 | 85.8M | 439s |
| 9 | Swin-V2-Base | 96.6% | 0.957 | 86.9M | 609s |
| 11 | ViT-B-32 | 96.5% | 0.958 | 87.5M | 515s |
| 11 | EVA-02-Base | 96.5% | 0.958 | 85.8M | 576s |
| 13 | YOLOv8n-cls | 96.3% | 0.951 | ~3.5M | - |
| 14 | EfficientNet-B0 | 96.2% | 0.950 | 4.1M | 563s |
| 15 | ResNet50 | 95.3% | 0.937 | 23.6M | 590s |
| 16 | ResNet18 | 91.8% | 0.897 | 11.2M | 381s |

---

## 5. SOTA Survey Findings (30+ papers, 2020-2026)

### 5.1 Standard Benchmarks (RadioML 2016.10a)
| Rank | Model | Avg Accuracy | Type |
|------|-------|-------------|------|
| 1 | ECDAT* | 95.05% | CNN + Dual-Attention Transformer |
| 2 | IQFormer | 68.52% | Transformer + Multi-modal Fusion |
| 3 | CC-MSNet | 62.86% | Complex-valued multi-stream |
| 4 | TLDNN | 62.83% | Transformer + LSTM |
| 5 | MCLDNN | 60.83% | Multi-channel LSTM-DNN |
*ECDAT unverified outlier

### 5.2 Emerging Paradigms
- **ConvMamba** (2025): Mamba SSM for AMR on Sig53, O(n) complexity
- **IQFM** (2025): Foundation model, 99.67% with 1-shot learning
- **GAF-MAE** (2025): Semi-supervised ViT, effective with 5% labeled data
- **RF-YOLO** (2025): Modified YOLO for RF spectrograms, mAP 0.9213

### 5.3 RFUAV Paper Results (5-class, SNR-averaged)
- ViT-L-16: 56.44% overall, 98.55% at SNR≥10dB
- ResNet18: 54.78% overall, 99.93% at SNR≥10dB
- Hot colormap: 58.16% (best), Parula: 56.44%
- Optimal STFTP: 256

---

## 6. Key Learnings

### 6.1 Statistical Features vs DL
- At 800 samples: RF + spectrogram features (100%) > all 14 neural networks
- Best DL (YOLOv11): 99.4% — gap is only 0.6%
- At 2623 samples (RFUAV): MaxViT (97.8%) — DL becomes competitive
- **Crossover point**: ~1K-2K samples is where DL starts matching statistical features

### 6.2 Spectrogram Dominance
- Every top-performing approach uses spectrogram-based input
- Spectrogram encodes BOTH frequency AND time structure simultaneously
- Hot colormap > Parula for CNN feature extraction (RFUAV finding confirmed)

### 6.3 Architecture Insights
- **MaxViT wins on RFUAV** (97.8%): Multi-axis attention combines local CNN + global Transformer
- **ConvNeXt is the reliable workhorse**: 97.5% RFUAV, 98.1% RTL-ML — modern depthwise CNNs
- **MobileNetV3-Large (4.2M params) at 97.1%**: Edge deployment champion, within 0.7% of MaxViT with 28x fewer params
- **Bigger ≠ better**: ConvNeXt-Large (196M) < ConvNeXt-Base (88M); ViT-L-16 (303M) < MaxViT-Base (119M)
- **SE blocks hurt at small scale**: SE-ResNet1D (73.8%) < plain ResNet1D (86.9%)
- **Transfer learning is king**: Pretrained YOLO/ConvNeXt >> training from scratch

### 6.4 RFML-MoE Assessment
- **Keep**: Multi-modal concept, Expert Choice routing, progressive training
- **Replace**: EfficientNet → ConvNeXt/MaxViT, SignalFormerIQ → ResNet1D
- **Drop**: Cyclostationary expert (55% — worst performer, slow SCF extraction)
- **Add**: Confidence-weighted routing fallback, adaptive expert depth

### 6.5 Ensemble Methods
- Simple voting (100%) matches oracle on RTL-ML — learned gating overfits (98.8%)
- **Confidence-weighted routing**: Best practical MoE adaptation at small scale
- MoE gating shines when no single expert dominates AND data is abundant (10K+)

---

## 7. Infrastructure

### 7.1 Local Machine
- **GPU**: GTX 1660 Ti (6GB VRAM)
- **PyTorch**: 2.11.0+cu126 at /opt1/ml/pylibs
- **OS**: Manjaro Linux 6.18.12-1
- **Disk**: /home 12GB free, /opt1 24GB free

### 7.2 Remote Server (rax@129.212.188.94)
- **GPU**: AMD Instinct MI300X VF (206 GB HBM3, 750W TDP)
- **ROCm**: 6.3.42134
- **PyTorch**: 2.9.1+rocm6.3
- **Storage**: 5 TB at /home/rax/mtp (1.1 TB used, 3.7 TB free)
- **Note**: User must be in `render` group for GPU access — use `sg render -c "command"`
- **Installed**: ultralytics, timm (1000+ models), h5py, albumentations, scipy

### 7.3 Remote Server Directory Structure
```
/home/rax/mtp/
├── raw/                    # RFUAV extracted IQ data (263 GB, 37 drone folders)
├── spectrograms/           # Generated spectrogram images
│   ├── train/              # 2623 images, 37 classes
│   └── val/                # 890 images, 37 classes
├── models/                 # Trained model checkpoints (*_best.pt)
├── results/                # JSON results + logs
│   └── all_results.json    # Final combined results
├── scripts/
│   ├── gen_spectrograms.py # Fast spectrogram generator (no matplotlib)
│   ├── train_rfuav.py      # Main training pipeline (13 timm + 3 YOLO)
│   └── snr_benchmark.py    # SNR evaluation script (ready to run)
├── configs/                # Training configs
├── RFUAV/                  # Cloned RFUAV codebase
├── droneRFb/
│   └── extracted/twin_droneRF/  # DroneRFb-DIR dataset (65 GB)
│       ├── train/          # 2177 .mat files
│       ├── test/           # 2513 .mat files
│       ├── train_labels.txt  # filename class_name (A1,A2,B,C1,C2,D1,D2,E1,E2,F1,F2,G1,G2)
│       └── test_labels.txt   # filename index
├── droneRFa/               # DroneRFa (574 GB, downloading)
└── drffr2/                 # DRFF-R2 (14/400 GB downloaded)
```

---

## 8. SciDB API Reference (for china.scidb.cn datasets)

### File Tree Listing
```
POST https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath
Headers: authorization: {JWT}, content-type: application/json, username: {email}, traceid: {email}
Body: {"dataSetId": "...", "version": "V3", "path": "/V3/...", "lastIndex": 0, "pageSize": 200}
Response: {"data": [{"id": "...", "fileName": "...", "size": ..., "path": "...", "dir": true/false}]}
```

### File Download
```
GET https://china.scidb.cn/download?fileId={id}&username={email}&traceId={email}
No auth required — public CDN. Rate limited (~4 concurrent connections).
```

### Auth Token (expires ~48h)
```
eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzU3MDMyMjEsInVzZXJfbmFtZSI6InJha2VzaC5wZXRlckBnbWFpbC5jb20i...
```

---

## 9. Pending Work & Next Steps

### 9.1 In Progress
- [x] RFUAV 37-class training — **COMPLETE** (16 models trained)
- [ ] DroneRFa download — 574 GB (still downloading)
- [ ] DRFF-R2 download — 14/400 GB
- [ ] Papers to analyze: "From Lab to Field Trials", "VMD and GAF Empowered CNN", "Time-Frequency Multiscale CNN"

### 9.2 Ready to Run
- **SNR benchmark** (`/home/rax/mtp/scripts/snr_benchmark.py`): Evaluate all trained models across -20 to +20 dB SNR
- **DroneRFb-DIR training**: Generate spectrograms from .mat files, train classifiers on 13-class individual identification
- **DRFF-R2 multi-scenario evaluation**: Cross-scenario generalization testing

### 9.3 Planned
- Full RFML-MoE implementation on MI300X with RFUAV data
- Integration of YOLO/RT-DETR as spectrogram expert in MoE
- VMD + GAF preprocessing pipeline for improved low-SNR performance
- Time-frequency multiscale CNN implementation
- Cross-dataset generalization testing (train on RFUAV, test on DroneRFb/DRFF-R2)
- Edge deployment analysis (MobileNetV3 at 97.1%, 4.2M params)

---

## 10. Recommendations for RFML-MoE Architecture Updates

Based on all experiments:

1. **Replace EfficientNet-B2 → MaxViT-Base** (97.8% on RFUAV) or ConvNeXt-Base (97.5%, fewer params)
2. **Replace SignalFormerIQ → ResNet1D** (86.9% > 80.6% on RTL-ML raw IQ)
3. **Drop cyclostationary expert** (55% — worst, slow SCF extraction)
4. **Add YOLOv11-cls as spectrogram expert** (97.4% on RFUAV with pretrained backbone)
5. **Add confidence-weighted routing** as fallback when learned router entropy is high
6. **Add adaptive expert depth** — spectrogram-only for confident predictions (>95%), full MoE for uncertain
7. **Use Hot colormap + FFT=256 + Hamming** for spectrogram generation (RFUAV paper's optimal)
8. **MobileNetV3-Large for edge** (97.1%, 4.2M params — within 0.7% of MaxViT)

---

## 11. Code Files in Repo

| File | Purpose |
|------|---------|
| `rfml_comparison.py` | Phase 1-2: Statistical features + ensemble methods (5 modalities, 6 ensembles) |
| `nn_comparison.py` | Phase 3: Neural network architectures (11 models, PyTorch) |
| `nn_rerun_failed.py` | Re-run failed models from nn_comparison |
| `yolo_detr_comparison.py` | Phase 4: YOLO/RT-DETR on spectrograms |
| `comparison_results.json` | Statistical feature + ensemble results |
| `ensemble_results.json` | Ensemble method results |
| `nn_comparison_results.json` | Neural network results |
| `yolo_detr_results.json` | YOLO/DETR results |
| `COMPARISON_REPORT.md` | Phase 1-2 report |
| `expert_analysis.md` | Expert specialization deep dive |
| `architecture_proposals.md` | Tested improvement proposals |
| `NN_ARCHITECTURE_REPORT.md` | Phase 3 NN report |
| `FULL_STUDY_REPORT.md` | Comprehensive 1146-line report covering all 18 approaches |
| `RFUAV_IMPLEMENTATION_PLAN.md` | MI300X training plan |

---

*Last updated: April 8, 2026*
