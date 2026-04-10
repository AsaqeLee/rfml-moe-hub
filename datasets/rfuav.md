# RFUAV Dataset

## Overview

| Property | Value |
|----------|-------|
| **Name** | RFUAV -- RF-based UAV Detection and Identification Benchmark |
| **Source** | [HuggingFace - kitofrank/RFUAV](https://huggingface.co/datasets/kitofrank/RFUAV) |
| **Code** | [GitHub - kitoweeknd/RFUAV](https://github.com/kitoweeknd/RFUAV) |
| **Paper** | arXiv:2503.09033 |
| **Size** | 102 GB compressed (37 .rar files), ~263 GB extracted, ~1.3 TB total raw |
| **Format** | Binary float32 interleaved I/Q (.iq files) |
| **Sample Rate** | 100 MSps (USRP X310) |
| **Frequency** | 5.765 GHz (5.8 GHz ISM band) |
| **Classes** | 37 drone/RC controller types |
| **Capture Device** | USRP X310 Software Defined Radio |

## Drone Classes (37 Types)

| # | Drone/RC Type | Manufacturer |
|---|--------------|-------------|
| 1 | dautel_evo_nano+ | Autel |
| 2 | dji_avata2 | DJI |
| 3 | dji_fpv_combo | DJI |
| 4 | dji_mavic3_pro | DJI |
| 5 | dji_mavic3pro | DJI |
| 6 | dji_mini3 | DJI |
| 7 | dji_mini4_pro | DJI |
| 8 | dji_mini4pro | DJI |
| 9 | flysky_el_18 | FlySky |
| 10 | flysky_nv_14 | FlySky |
| 11 | frsky_x14 | FrSky |
| 12 | frsky_x9dp2019 | FrSky |
| 13 | futaba_t14sg | Futaba |
| 14 | futaba_t16iz | Futaba |
| 15 | futaba_t18sz | Futaba |
| 16 | herelink_v1_1 | CubePilot |
| 17 | jr_propo_xg7 | JR |
| 18 | jr_propo_xg14 | JR |
| 19 | jumper_t14 | Jumper |
| 20 | jumper_tprov2 | Jumper |
| 21 | radiolink_at10_ii | RadioLink |
| 22 | radiomaster_tx16s | RadioMaster |
| 23 | siyi_ft24 | SIYI |
| 24 | siyi_mk15 | SIYI |
| 25 | siyi_mk32 | SIYI |
| 26 | skydroid_h12 | SkyDroid |
| 27 | wfly_et10 | WFLY |
| 28 | wfly_wft09sii | WFLY |
| 29 | yunzhuo_h16 | YunZhuo |
| 30 | yunzhuo_h30 | YunZhuo |
| 31-37 | Additional variants | Various |

## File Format

Each `.iq` file is raw binary with float32 interleaved I/Q samples:

```
[I_0, Q_0, I_1, Q_1, I_2, Q_2, ...]
```

- File size: ~763 MB per file (100M complex samples = 1 second at 100 MSps)
- Data type: float32 (4 bytes per value, 8 bytes per complex sample)

## Directory Structure

```
raw/
├── dautel_evo_nano+/
│   └── dautel_evo_nano+/
│       └── VTSBW=20/
│           ├── segment_0001.iq
│           ├── segment_0002.iq
│           └── ...
├── dji_avata2/
│   └── dji_avata2/
│       ├── VTSBW=10/
│       │   └── *.iq
│       ├── VTSBW=20/
│       │   └── *.iq
│       └── VTSBW=40/
│           └── *.iq
...
└── yunzhuo_h30/
    └── yunzhuo_h30/
        └── VTSBW=20/
            └── *.iq
```

Subfolders are organized by VTS (Video Transmission System) bandwidth: 10, 20, 40, or 60 MHz.

## XML Metadata

Each drone folder contains XML metadata files with capture parameters:

```xml
<capture>
  <center_frequency>5765000000</center_frequency>
  <sample_rate>100000000</sample_rate>
  <gain>40</gain>
  <bandwidth>20000000</bandwidth>
</capture>
```

## Loading Example

```python
import numpy as np

# Load a single .iq file
filepath = "data/rfuav_raw/dji_mini4_pro/dji_mini4_pro/VTSBW=20/segment_0001.iq"

# Read binary float32
raw = np.fromfile(filepath, dtype=np.float32)

# Deinterleave to complex
i_channel = raw[0::2]
q_channel = raw[1::2]
iq_complex = i_channel + 1j * q_channel

print(f"Complex samples: {len(iq_complex):,}")  # ~100,000,000
print(f"Duration: {len(iq_complex) / 100e6:.2f} seconds")  # ~1.0s
print(f"File size: {raw.nbytes / 1e6:.1f} MB")  # ~763 MB
```

## Spectrogram Generation Parameters (Optimal)

As determined by the RFUAV paper:

| Parameter | Value | Notes |
|-----------|-------|-------|
| FFT size (STFTP) | 256 | Paper's optimum |
| Window | Hamming | Standard for RF |
| Colormap | Hot | 58.16% vs 56.44% Parula |
| Samples per spectrogram | 1,000,000 | 0.01s at 100 MSps |
| Overlap | 50% | Standard |
| Output size | 640x640 | Classifier input standard |

## Best Results

| Approach | Model | Accuracy | Params |
|----------|-------|----------|--------|
| Spectrogram DL | MaxViT-Base | 97.8% | 118.7M |
| Spectrogram DL | ConvNeXt-Base | 97.5% | 87.6M |
| Spectrogram DL | MobileNetV3-Large | 97.1% | 4.2M |
| Raw IQ DL | LWMExpert | 94.1% | 1.3M |
| Statistical | Combined RF | 95.7% | ~200KB |

## RFUAV Paper Results (5-Class, SNR-Averaged)

From the original paper (arXiv:2503.09033), tested across all SNR levels:

| Model | Overall Accuracy | High-SNR (>=10dB) |
|-------|-----------------|-------------------|
| ViT-L-16 | 56.44% | 98.55% |
| ResNet18 | 54.78% | 99.93% |

Note: Our 37-class results (97.8%) significantly outperform the paper's 5-class SNR-averaged results because we train and evaluate on high-SNR data only.

## Citation

```bibtex
@article{rfuav2025,
  title={RFUAV: A Benchmark Dataset for UAV Detection and Identification},
  author={Kito et al.},
  journal={arXiv:2503.09033},
  year={2025}
}
```
