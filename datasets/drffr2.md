# DRFF-R2 Dataset

## Overview

| Property | Value |
|----------|-------|
| **Name** | DRFF-R2 (Drone RF Fingerprinting - Round 2, Multi-Scenario) |
| **Source** | [SciDB China](https://china.scidb.cn/) -- Dataset ID `b8a16448c1284fd1be1ded9ccc45be20` |
| **Paper** | arXiv:2603.00106 -- A Multi-Scenario UAV RF Dataset with Real-World Acquisition |
| **Size** | 400.6 GB, 730 files |
| **Format** | MATLAB .mat files |
| **Classes** | 26 drones across 8 models |
| **Scenarios** | 7 distinct capture scenarios |
| **Task** | Multi-scenario cross-generalization |
| **Status** | Download in progress |

## Drone Models (8 Types, 26 Individuals)

| Model | Count | Manufacturer |
|-------|-------|-------------|
| mavic3 | 3-4 | DJI |
| mavic3C | 3-4 | DJI |
| mavic3S | 3-4 | DJI |
| mavicAir2 | 3-4 | DJI |
| mavicAir2s | 3-4 | DJI |
| mini3pro | 3-4 | DJI |
| mini4PRO | 3-4 | DJI |
| mini5PRO | 3-4 | DJI |

26 individual drone units across 8 DJI drone models, enabling both type-level and individual-level classification experiments.

## Scenarios (7 Capture Conditions)

| Dataset | Scenario | Description |
|---------|----------|-------------|
| dataset1 | Single drone states | Cruise, ascend, descend, takeoff, landing, shading |
| dataset2 | Drone mixed | Multiple drone types active simultaneously |
| dataset3 | Hover | Stationary hovering captures |
| dataset4 | Dual frequency | Captures at two different frequency bands |
| dataset5 | Absorbent cotton | Signals through RF-absorbing material |
| dataset6 | WiFi mixed | Drone signals with WiFi interference |
| dataset7 | Environment | Varying environmental conditions |

This multi-scenario design enables critical research questions:
- **Cross-scenario generalization**: Train on scenario 1, test on scenarios 2-7
- **Interference robustness**: WiFi mixed vs clean captures
- **State-invariant features**: Same drone in different flight states
- **Attenuation effects**: Signal through absorbent cotton

## Directory Structure

```
V3/
├── dataset1/                    # Single drone states
│   ├── mavic3_001_cruise.mat
│   ├── mavic3_001_ascend.mat
│   ├── mavic3_001_descend.mat
│   ├── mavic3_001_takeoff.mat
│   ├── mavic3_001_landing.mat
│   ├── mavic3_001_shading.mat
│   └── ...
├── dataset2/                    # Drone mixed
├── dataset3/                    # Hover
├── dataset4/                    # Dual frequency
├── dataset5/                    # Absorbent cotton
├── dataset6/                    # WiFi mixed
├── dataset7/                    # Environment
└── code/
    └── Technical_Validation/
        ├── experiment1.m        # MATLAB validation
        └── experiment2.py       # Python validation
```

## File Format

MATLAB .mat files containing IQ data:

```python
import h5py
import numpy as np

with h5py.File("dataset1/mavic3_001_cruise.mat", 'r') as f:
    print(list(f.keys()))
    # Access IQ data (structure may vary by scenario)
    data = f['data'][:]
    # Or for some files:
    i_channel = f['I'][:]
    q_channel = f['Q'][:]
```

## Download

The dataset consists of 730 files distributed across 7 scenario directories. Download requires the SciDB API for file listing:

```bash
# Automated download script (handles API pagination and parallel download)
python datasets/download_scripts/download_drffr2.py

# The script:
# 1. Lists all files via SciDB API (POST to gin-sdb-filetree endpoint)
# 2. Downloads each file via direct URL
# 3. Organizes into scenario directories
# 4. Supports resume on interruption
```

### SciDB API Details

```
File listing endpoint:
  POST https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath
  Body: {"dataSetId": "b8a16448c1284fd1be1ded9ccc45be20", "version": "V3", "path": "/V3/...", "lastIndex": 0, "pageSize": 200}

Download endpoint:
  GET https://china.scidb.cn/download?fileId={id}&username={email}&traceId={email}
  Rate limit: ~4 concurrent connections
```

## Storage Requirements

| Stage | Size |
|-------|------|
| Downloaded .mat files | 400.6 GB |
| Spectrograms (estimated) | ~8 GB |
| **Total needed** | **~410 GB** |

## Planned Experiments

1. **Within-scenario classification**: Train and test within each scenario
2. **Cross-scenario generalization**: Train on dataset1 (clean), test on dataset6 (WiFi interference)
3. **Flight state invariance**: Can models identify drones regardless of cruise/hover/ascend state?
4. **WiFi robustness**: Performance degradation under WiFi interference (dataset6)
5. **Attenuation robustness**: Signal through absorbent cotton (dataset5)
6. **Multi-task learning**: Joint drone type + flight state classification
7. **Cross-dataset transfer**: Models trained on RFUAV tested on DRFF-R2 (and vice versa)

## Citation

```bibtex
@article{drffr2-2026,
  title={A Multi-Scenario UAV RF Dataset with Real-World Acquisition},
  journal={arXiv:2603.00106},
  year={2026}
}
```
