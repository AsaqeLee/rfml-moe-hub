# Datasets Overview

This directory contains documentation for the 5 datasets used in the Drone-RFML-Hub research. The actual data is NOT stored in this repository due to size constraints (total ~2.4 TB). See each dataset's documentation and [INSTALL.md](../INSTALL.md) for download instructions.

## Dataset Comparison

| Dataset | Classes | Samples | Raw Size | Frequency | Sample Rate | Format | Task |
|---------|---------|---------|----------|-----------|-------------|--------|------|
| [RTL-ML](rtl_ml.md) | 7 signals | 800 | 6.2 GB | 88-462 MHz | 1.024 MSps | .npy | Signal classification |
| [RFUAV](rfuav.md) | 37 drones | 356 files | 1.3 TB | 5.8 GHz | 100 MSps | Binary IQ | Drone identification |
| [DroneRFb-DIR](droneRFb_dir.md) | 13 (6 types x2 + BG) | 4,690 | 65 GB | 2.4 GHz | 80 MSps | MATLAB .mat | Cross-individual ID |
| [DroneRFa](droneRFa.md) | TBD | TBD | 574 GB | 2.4 GHz | 80 MSps | MATLAB .mat | Dual-receiver |
| [DRFF-R2](drffr2.md) | 26 drones | 730 files | 400 GB | Multi | Varies | MATLAB .mat | Multi-scenario |

## Dataset Difficulty Hierarchy

```
Easy:     RTL-ML (7 classes, 800 samples, within-session)
          - Statistical features achieve 100%
          - Good for algorithm development and prototyping

Medium:   RFUAV (37 classes, 3553 spectrograms, single receiver)
          - DL required for best results (MaxViT 97.8%)
          - Statistical features plateau at 95.7%

Hard:     DroneRFb-DIR (13 classes, cross-individual generalization)
          - Train on individuals 1&2, test on individual 3
          - Statistical features collapse to 42.3%
          - Best DL: ConvNeXt-Base 92.0%

Research: DroneRFa (dual-receiver, 574 GB)
          - Companion to DroneRFb, dual-receiver configuration
          - Enables receiver diversity and fusion studies

Complex:  DRFF-R2 (26 drones, 7 scenarios, 400 GB)
          - Multi-scenario: indoor, outdoor, WiFi interference, shading
          - Cross-scenario generalization is the key challenge
```

## Storage Planning

| Dataset | Compressed | Extracted | Spectrograms | Total Needed |
|---------|------------|-----------|-------------|-------------|
| RTL-ML | -- | 6.2 GB | ~500 MB | ~7 GB |
| RFUAV | 102 GB | 263 GB | ~5 GB | ~370 GB |
| DroneRFb | 64 GB | 65 GB | ~3 GB | ~132 GB |
| DroneRFa | -- | 574 GB | ~10 GB | ~584 GB |
| DRFF-R2 | -- | 400 GB | ~8 GB | ~408 GB |
| **Total** | | | | **~1.5 TB** |

## Download Scripts

The `download_scripts/` directory contains automated downloaders for each dataset:

| Script | Dataset | Method |
|--------|---------|--------|
| `download_rfuav.py` | RFUAV | HuggingFace Hub |
| `download_droneRFb.py` | DroneRFb-DIR | SciDB API + split zip |
| `download_droneRFa.sh` | DroneRFa | aria2c direct download |
| `download_drffr2.py` | DRFF-R2 | SciDB API (730 files) |

RTL-ML uses `huggingface-hub` directly -- see [INSTALL.md](../INSTALL.md).
