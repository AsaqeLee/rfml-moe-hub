# Installation & Dataset Download Guide

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU 8GB VRAM (CUDA 12+) | AMD Instinct MI300X (192 GB HBM3) |
| **RAM** | 32 GB | 64 GB+ |
| **Storage** | 100 GB (RTL-ML only) | 5 TB (all datasets) |
| **CPU** | 8 cores | 16+ cores (feature extraction is CPU-bound) |

### Software

| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.10+ | 3.11 recommended |
| PyTorch | 2.x | ROCm 6.3 or CUDA 12.x |
| OS | Linux (Ubuntu 22.04+ / Manjaro) | Windows untested |

---

## 1. Python Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Or with conda
conda create -n drone-rfml python=3.11
conda activate drone-rfml
```

## 2. PyTorch Installation

### For AMD ROCm (MI300X)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# Verify ROCm
python -c "import torch; print(f'ROCm: {torch.version.hip}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected: ROCm: 6.3.42134, GPU: AMD Instinct MI300X VF
```

**GPU access on multi-user systems** -- you may need the `render` group:

```bash
# Check if you have GPU access
rocm-smi

# If permission denied, add yourself to the render group
sudo usermod -aG render $USER
# Then either re-login or use:
sg render -c "python your_script.py"
```

### For NVIDIA CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## 3. Python Dependencies

```bash
# Core ML
pip install timm>=1.0.0          # 1000+ pretrained models (MaxViT, ConvNeXt, etc.)
pip install ultralytics>=8.0.0   # YOLO models
pip install scikit-learn>=1.2.0  # Random Forest, GBM, metrics

# Data processing
pip install numpy>=1.24.0
pip install scipy>=1.9.0
pip install h5py>=3.0.0          # MATLAB v7.3 .mat file reading (DroneRFb)
pip install pandas>=1.5.0

# Image processing
pip install albumentations>=1.3.0  # Training augmentations
pip install opencv-python>=4.7.0
pip install Pillow>=9.0.0

# Visualization
pip install matplotlib>=3.5.0
pip install seaborn>=0.12.0
pip install tqdm>=4.60.0

# Dataset download
pip install huggingface-hub>=0.20.0  # RFUAV and RTL-ML datasets
pip install requests>=2.28.0         # SciDB API downloads

# Optional
pip install tensorboard>=2.12.0      # Training visualization
pip install wandb>=0.15.0            # Experiment tracking
pip install pyyaml>=6.0.0            # Config files
```

Or install all at once:

```bash
pip install timm ultralytics scikit-learn numpy scipy h5py pandas \
    albumentations opencv-python Pillow matplotlib seaborn tqdm \
    huggingface-hub requests tensorboard pyyaml
```

## 4. aria2c Installation (Fast Parallel Downloads)

```bash
# Ubuntu/Debian
sudo apt install aria2

# Manjaro/Arch
sudo pacman -S aria2

# Verify
aria2c --version
```

---

## 5. Dataset Downloads

### 5.1 RTL-ML (6.2 GB) -- Recommended First

The smallest dataset. Good for verifying your setup works.

**Source**: [HuggingFace - TrevTron/rtl-ml-dataset](https://huggingface.co/datasets/TrevTron/rtl-ml-dataset)

```bash
# Method 1: HuggingFace Hub (recommended)
pip install huggingface-hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('TrevTron/rtl-ml-dataset', local_dir='data/rtl_ml')
"

# Method 2: Git LFS
git lfs install
git clone https://huggingface.co/datasets/TrevTron/rtl-ml-dataset data/rtl_ml
```

**Verification**:

```bash
python -c "
import os, numpy as np
data_dir = 'data/rtl_ml/datasets_validated'
classes = sorted(os.listdir(data_dir))
print(f'Classes: {classes}')
total = sum(len(os.listdir(os.path.join(data_dir, c))) for c in classes)
print(f'Total samples: {total}')
# Expected: 7 classes, 800 samples
sample = np.load(os.path.join(data_dir, classes[0], os.listdir(os.path.join(data_dir, classes[0]))[0]), allow_pickle=True).item()
print(f'Keys: {list(sample.keys())}')
print(f'IQ shape: {sample[\"samples\"].shape}, dtype: {sample[\"samples\"].dtype}')
"
```

### 5.2 RFUAV (1.3 TB) -- Primary Drone Dataset

**Source**: [HuggingFace - kitofrank/RFUAV](https://huggingface.co/datasets/kitofrank/RFUAV) + [GitHub - kitoweeknd/RFUAV](https://github.com/kitoweeknd/RFUAV)

```bash
# Download via huggingface-hub (37 .rar files, ~102 GB compressed)
python datasets/download_scripts/download_rfuav.py

# Or manually:
huggingface-cli download kitofrank/RFUAV --local-dir data/rfuav_compressed --repo-type dataset

# Extract (requires unrar)
sudo apt install unrar  # or: sudo pacman -S unrar
cd data/rfuav_compressed
for f in *.rar; do
    unrar x "$f" ../rfuav_raw/
done
```

**Expected structure after extraction**:

```
data/rfuav_raw/
├── dautel_evo_nano+/
│   └── dautel_evo_nano+/
│       └── VTSBW=20/
│           ├── segment_0001.iq
│           ├── segment_0002.iq
│           └── ...
├── dji_avata2/
├── dji_fpv_combo/
├── dji_mavic3_pro/
...
└── yunzhuo_h30/       # 37 drone types total
```

Each `.iq` file is binary float32 interleaved I/Q at 100 MSps (~763 MB = 1 second of capture).

### 5.3 DroneRFb-DIR (65 GB) -- Cross-Individual

**Source**: [SciDB China](https://china.scidb.cn/) -- Dataset ID `84cf9101e739402784b1396783881202`

```bash
# Download using provided script (handles split-zip format)
python datasets/download_scripts/download_droneRFb.py

# Manual download: 32-part split zip from SciDB
# After download, combine and extract:
cat twin_droneRF.zip.* > twin_droneRF_combined.zip
unzip twin_droneRF_combined.zip -d data/droneRFb/
```

**Expected structure**:

```
data/droneRFb/twin_droneRF/
├── train/
│   ├── A1_FCS_LOS_indoor_001.mat    # 2177 files
│   └── ...
├── test/
│   ├── A3_FCS_LOS_indoor_001.mat    # 2513 files (individual 3)
│   └── ...
├── train_labels.txt                  # filename class_name
└── test_labels.txt                   # filename index
```

Each `.mat` file (MATLAB v7.3 / HDF5) contains keys `I` and `Q`, each `1x4000000` float32 at 80 MSps.

### 5.4 DroneRFa (574 GB) -- Dual-Receiver Companion

**Source**: [SciDB China](https://china.scidb.cn/) -- File ID `c403fc76444e4b9989e4f3ff570f3b3d`

```bash
# Download using aria2c (large single file)
bash datasets/download_scripts/download_droneRFa.sh

# Or manually:
aria2c -x 4 -s 4 --max-tries=0 --retry-wait=30 \
    "https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d&username=YOUR_EMAIL&traceId=YOUR_EMAIL" \
    -o data/droneRFa.rar

# Extract
cd data && unrar x droneRFa.rar droneRFa/
```

### 5.5 DRFF-R2 (400 GB) -- Multi-Scenario

**Source**: [SciDB China](https://china.scidb.cn/) -- Dataset ID `b8a16448c1284fd1be1ded9ccc45be20`

730 files across 7 scenario directories. Requires API-based file listing and download.

```bash
# Download using provided script (handles API pagination and parallel download)
python datasets/download_scripts/download_drffr2.py

# Files are organized by scenario:
# V3/dataset1/ - Single drone states (cruise, ascend, descend, takeoff, landing, shading)
# V3/dataset2/ - Drone mixed
# V3/dataset3/ - Hover
# V3/dataset4/ - Dual frequency
# V3/dataset5/ - Inside absorbent cotton
# V3/dataset6/ - WiFi mixed
# V3/dataset7/ - Environment
```

---

## 6. Post-Download: Spectrogram Generation

After downloading raw IQ data, generate spectrograms for the vision model pipeline:

```bash
# RFUAV spectrograms (FFT=256, Hamming, Hot colormap)
python preprocessing/rfuav_specgen.py \
    --input data/rfuav_raw/ \
    --output intermediate/spectrograms/rfuav/ \
    --fft-size 256 --colormap hot --samples-per-spec 1000000

# DroneRFb spectrograms (from HDF5 .mat files)
python preprocessing/droneRFb_specgen.py \
    --input data/droneRFb/twin_droneRF/train/ \
    --output intermediate/spectrograms/droneRFb/train/

# DRFF-R2 spectrograms
python preprocessing/drffr2_specgen.py \
    --input data/drffr2/ \
    --output intermediate/spectrograms/drffr2/
```

---

## 7. Verification

```bash
# Check PyTorch and GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# Check key libraries
python -c "
import timm, ultralytics, sklearn, albumentations, h5py
print(f'timm: {timm.__version__} ({len(timm.list_models())} models)')
print(f'ultralytics: {ultralytics.__version__}')
print(f'sklearn: {sklearn.__version__}')
print(f'albumentations: {albumentations.__version__}')
print(f'h5py: {h5py.__version__}')
"

# Quick test: RTL-ML statistical features
python experiments/rtl_ml/rfml_comparison.py --quick-test
```

---

## Troubleshooting

### ROCm: "No GPU detected"

```bash
# Check ROCm installation
rocm-smi
# If permission denied:
sudo usermod -aG render $USER
newgrp render
```

### HDF5 errors reading DroneRFb .mat files

```bash
# MATLAB v7.3 files require h5py, not scipy.io.loadmat
pip install h5py
python -c "
import h5py
with h5py.File('data/droneRFb/twin_droneRF/train/A1_FCS_LOS_indoor_001.mat', 'r') as f:
    print(list(f.keys()))  # Should show ['I', 'Q']
    print(f['I'].shape)    # Should show (1, 4000000)
"
```

### Out of VRAM during training

```bash
# Reduce batch size in training configs
# Or use gradient accumulation:
python experiments/rfuav/train_rfuav.py --batch-size 8 --grad-accum 4

# For RTL-ML (small dataset), CPU training is feasible:
python experiments/rtl_ml/nn_comparison.py --device cpu
```

### aria2c connection issues with SciDB

```bash
# SciDB China rate-limits to ~4 concurrent connections
# Use --max-connection-per-server=4 and add retry logic
aria2c -x 4 -s 4 --max-tries=0 --retry-wait=30 --timeout=600 URL
```
