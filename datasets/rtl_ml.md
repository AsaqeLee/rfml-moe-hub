# RTL-ML Dataset

## Overview

| Property | Value |
|----------|-------|
| **Name** | RTL-ML (Real-Time Learning for Machine Learning) |
| **Source** | [HuggingFace - TrevTron/rtl-ml-dataset](https://huggingface.co/datasets/TrevTron/rtl-ml-dataset) |
| **Size** | 800 samples, 6.2 GB |
| **Format** | NumPy `.npy` dictionaries |
| **Sample Rate** | 1.024 MSps (ARM-optimized to prevent USB overflow) |
| **Capture Duration** | 0.5 seconds per sample (512,000 complex IQ samples) |
| **Hardware** | RTL-SDR Blog V4 + Indiedroid Nova / Raspberry Pi 5 |
| **Location** | Temecula, CA (real-world captures, not synthetic) |
| **Quality Gates** | DC offset removed, 6 dB minimum SNR, per-class validation |
| **License** | MIT |

## Signal Classes

| Signal | Frequency | Samples | Nature | Modulation |
|--------|-----------|---------|--------|------------|
| FM Broadcast | 88.5--105.7 MHz | 200 | Continuous | Wideband FM |
| NOAA Weather | 162.4 MHz | 100 | Continuous | Narrowband FM |
| APRS | 144.39 MHz | 100 | Bursty/Sporadic | AFSK 1200 baud |
| Pager | 152.84 MHz | 100 | Bursty/Periodic | POCSAG/FLEX |
| ISM Sensors | 433.92 MHz | 100 | Bursty/Short | OOK/FSK |
| FRS/GMRS | 462.5625 MHz | 100 | Bursty/Voice | Narrowband FM |
| Noise | 145.0 MHz | 100 | Continuous | Thermal noise |

FM Broadcast has 200 samples across 5 different station frequencies to test whether models learn "FM-ness" vs memorizing a specific frequency.

## File Format

Each `.npy` file is a pickled Python dictionary:

```python
{
    "samples": np.ndarray,      # (512000,) complex128 IQ samples
    "center_freq": float,       # Center frequency in Hz
    "sample_rate": float,       # 1024000.0
    "timestamp": str,           # ISO format capture time
    "label": str,               # Class name
    "snr_db": float,            # Estimated SNR in dB
    "version": str              # Dataset version
}
```

## Directory Structure

```
datasets_validated/
├── APRS/
│   ├── APRS_144390000_20250101_120000.npy
│   └── ... (100 files)
├── FM_broadcast/
│   ├── FM_broadcast_88500000_20250101_120100.npy
│   └── ... (200 files, 5 frequencies)
├── FRS_GMRS/       (100 files)
├── ISM_sensors/    (100 files)
├── NOAA_weather/   (100 files)
├── noise/          (100 files)
└── pager/          (100 files)
```

## Data Split

Temporal split to prevent data leakage (signals captured at similar times may be correlated):

```
Per-class temporal ordering:
    Train (64%)  |  Val (16%)  |  Test (20%)
    Earlier captures    -->     Later captures

Total: 512 train, 128 val, 160 test
```

## Loading Example

```python
import numpy as np
import os

data_dir = "data/rtl_ml/datasets_validated"

# Load a single sample
sample = np.load(
    os.path.join(data_dir, "FM_broadcast", "FM_broadcast_88500000_20250101_120100.npy"),
    allow_pickle=True
).item()

iq = sample["samples"]          # (512000,) complex128
freq = sample["center_freq"]    # e.g., 88500000.0
sr = sample["sample_rate"]      # 1024000.0
label = sample["label"]         # "FM_broadcast"
snr = sample["snr_db"]          # e.g., 17.5

print(f"Class: {label}, Freq: {freq/1e6:.1f} MHz, SNR: {snr:.1f} dB")
print(f"IQ shape: {iq.shape}, dtype: {iq.dtype}")
print(f"Duration: {len(iq)/sr:.3f} seconds")

# Extract I and Q channels
i_channel = iq.real   # In-phase
q_channel = iq.imag   # Quadrature
```

## Best Results on This Dataset

| Approach | Method | Accuracy |
|----------|--------|----------|
| Statistical | Spectrogram features (37) + RF | 100.0% |
| Ensemble | Majority vote (5 experts) | 100.0% |
| DL | YOLOv11n-cls (spectrogram) | 99.4% |
| DL | ConvNeXt-Tiny (spectrogram) | 98.1% |
| DL (raw IQ) | ResNet1D | 86.9% |

## Citation

```bibtex
@dataset{rtl-ml-2026,
  title={RTL-ML Dataset: Real-World RF Signal Classification},
  author={Trevor Unland},
  year={2026},
  url={https://huggingface.co/datasets/TrevTron/rtl-ml-dataset}
}
```
