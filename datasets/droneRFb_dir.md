# DroneRFb-DIR Dataset

## Overview

| Property | Value |
|----------|-------|
| **Name** | DroneRFb-DIR (Drone RF Fingerprinting - Dataset for Individual Recognition) |
| **Source** | [SciDB China](https://china.scidb.cn/) -- Dataset ID `84cf9101e739402784b1396783881202` |
| **Paper** | JEIT 2025, DOI: 10.11999/JEIT240804 |
| **Size** | 64 GB compressed (32-part split zip), 65 GB extracted |
| **Format** | MATLAB v7.3 (HDF5) .mat files |
| **Sample Rate** | 80 MSps |
| **Frequency** | 2.4--2.48 GHz ISM band |
| **Classes** | 13: 6 drone types (A-G) x 2 individuals + B (background) |
| **Task** | Cross-individual drone identification |
| **Capture Device** | Software Defined Radio (SDR) |

## What Makes This Dataset Special

DroneRFb-DIR tests **cross-individual generalization**: models train on individuals 1 and 2 of each drone type, then are tested on individual 3 (never seen during training). This simulates the real-world scenario where a detector must identify a drone type it has seen before, but from a specific unit it has never encountered.

This is substantially harder than within-individual classification. Statistical features that achieve 100% on RTL-ML collapse to 42.3% here.

## Classes

| Class | Drone Type | Individual | Split |
|-------|-----------|------------|-------|
| A1 | Type A | Individual 1 | Train |
| A2 | Type A | Individual 2 | Train |
| B | Background (no drone) | -- | Both |
| C1 | Type C | Individual 1 | Train |
| C2 | Type C | Individual 2 | Train |
| D1 | Type D | Individual 1 | Train |
| D2 | Type D | Individual 2 | Train |
| E1 | Type E | Individual 1 | Train |
| E2 | Type E | Individual 2 | Train |
| F1 | Type F | Individual 1 | Train |
| F2 | Type F | Individual 2 | Train |
| G1 | Type G | Individual 1 | Train |
| G2 | Type G | Individual 2 | Train |

Test set uses individual 3 for each drone type (A3, C3, D3, E3, F3, G3) plus background.

## Signal Types

Each drone produces two signal types:
- **FCS** (Flight Control Signal): Command/control link
- **VTS** (Video Transmission Signal): Video downlink

Captures include both LOS (Line of Sight) and NLOS (Non-Line of Sight), indoor and outdoor conditions.

## File Format

MATLAB v7.3 files (HDF5 format), readable with `h5py`:

```python
# Each .mat file contains:
{
    "I": np.ndarray,    # shape (1, 4000000), float32 -- In-phase
    "Q": np.ndarray     # shape (1, 4000000), float32 -- Quadrature
}
```

Total: 4,000,000 samples at 80 MSps = 50 ms per capture.

## Directory Structure

```
twin_droneRF/
├── train/
│   ├── A1_FCS_LOS_indoor_001.mat
│   ├── A1_FCS_LOS_indoor_002.mat
│   ├── A1_VTS_LOS_indoor_001.mat
│   ├── A2_FCS_LOS_outdoor_001.mat
│   ├── B_indoor_001.mat
│   ├── C1_FCS_NLOS_indoor_001.mat
│   └── ... (2177 files total)
├── test/
│   ├── A3_FCS_LOS_indoor_001.mat
│   ├── A3_VTS_NLOS_outdoor_001.mat
│   └── ... (2513 files total)
├── train_labels.txt    # Format: filename class_name
└── test_labels.txt     # Format: filename index
```

### Label File Format

**train_labels.txt**:
```
A1_FCS_LOS_indoor_001.mat A1
A1_FCS_LOS_indoor_002.mat A1
A2_FCS_LOS_outdoor_001.mat A2
B_indoor_001.mat B
```

**test_labels.txt**:
```
A3_FCS_LOS_indoor_001.mat 0
A3_VTS_NLOS_outdoor_001.mat 0
B_outdoor_001.mat 2
```

## Loading Example

```python
import h5py
import numpy as np

filepath = "data/droneRFb/twin_droneRF/train/A1_FCS_LOS_indoor_001.mat"

with h5py.File(filepath, 'r') as f:
    print(f"Keys: {list(f.keys())}")          # ['I', 'Q']

    i_channel = f['I'][0, :]                   # (4000000,) float32
    q_channel = f['Q'][0, :]                   # (4000000,) float32
    iq_complex = i_channel + 1j * q_channel

    print(f"Samples: {len(iq_complex):,}")     # 4,000,000
    print(f"Duration: {len(iq_complex) / 80e6 * 1000:.1f} ms")  # 50.0 ms
    print(f"Sample rate: 80 MSps")
    print(f"Frequency: 2.4 GHz band")

# Parse filename for metadata
# Format: {class}_{signal_type}_{LOS/NLOS}_{indoor/outdoor}_{index}.mat
parts = "A1_FCS_LOS_indoor_001".split("_")
drone_class = parts[0]       # A1
signal_type = parts[1]       # FCS or VTS
los_condition = parts[2]     # LOS or NLOS
environment = parts[3]       # indoor or outdoor
```

## Download

```bash
# 32-part split zip from SciDB
python datasets/download_scripts/download_droneRFb.py

# After download, combine and extract:
cat twin_droneRF.zip.* > twin_droneRF_combined.zip
unzip twin_droneRF_combined.zip -d data/droneRFb/
```

## Best Results

### 13-Class Individual Identification (Cross-Individual)

| Model | Accuracy | Notes |
|-------|----------|-------|
| ConvNeXt-Base | 92.0% | Best overall |
| MaxViT-Base | 89.5% | |
| SpectrogramExpert (RFML) | ~88% | |
| Statistical features (Combined) | 42.3% | Complete failure |

### 7-Class Type-Level Classification

| Model | Accuracy |
|-------|----------|
| ConvNeXt-Base | 94.2% |
| MaxViT-Base | 93.1% |
| SpectrogramExpert (RFML) | 90.4% |

## Citation

```bibtex
@article{droneRFb2025,
  title={Cross-Individual Drone Identification via RF Fingerprinting},
  journal={Journal of Electronics and Information Technology (JEIT)},
  year={2025},
  doi={10.11999/JEIT240804}
}
```
