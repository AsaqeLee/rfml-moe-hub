# RFML-MoE: Multi-modal Mixture-of-Experts Drone RF Signal Detection Pipeline

RFML-MoE is a research-grade pipeline for detecting, classifying, and fingerprinting drone radio-frequency emissions using a multi-modal Mixture-of-Experts (MoE) architecture. The system ingests raw IQ samples, extracts four complementary feature representations, routes them through domain-specialist expert networks, and produces hierarchical classifications at three levels of granularity: binary drone/no-drone detection, drone-type and link-type identification, and fine-grained make/model/protocol fingerprinting.

## Why Mixture-of-Experts

No single feature representation dominates across all SNR regimes or signal modalities. At high SNR, raw IQ time-domain features are highly discriminative; at low SNR, statistical features such as higher-order spectra and cyclostationary signatures are more robust. Frequency-domain representations capture modulation structure that is invisible in the time domain, while cyclostationary features reveal protocol-specific periodicities that are buried in magnitude spectrograms. A hard-coded fusion of these modalities would require manual weighting that degrades under distribution shift. Expert Choice routing lets the model learn, per sample, which combination of representations is most informative, and the shared expert captures cross-modal structure that no single specialist sees in isolation.

---

## Architecture Overview

### Expert Network Stack

The model comprises four domain-specialist experts plus shared components:

| Expert | Architecture | Input | Role |
|---|---|---|---|
| IQ Expert (`SignalFormerIQ`) | Complex CNN + Transformer | Raw IQ samples [2, N] | Time-domain waveform analysis |
| Spectrogram Expert (`EfficientNetSpec`) | EfficientNet-B2 | 3-channel STFT image [3, 512, 512] | Frequency-domain modulation structure |
| HOS Expert (`FTTransformerHOS`) | FT-Transformer | Normalized cumulants [20] | Statistical modulation fingerprinting |
| Cyclostationary Expert (`TCNCyclo`) | Temporal CNN (TCN) | SCF feature vector [512] | Protocol fingerprinting via spectral correlation |

**IQ Expert** — A stack of three 1D complex convolutional layers (channels: 64, 128, 256; kernel size 7) followed by four Transformer encoder layers (8 heads, embed dim 512). Processes the full 32 768-sample IQ segment in the time domain.

**Spectrogram Expert** — EfficientNet-B2 pretrained on ImageNet, adapted for 3-channel input. The three channels encode log-magnitude, unwrapped phase, and instantaneous frequency (phase derivative), computed from a complex STFT with 512-point FFT and 50% overlap. The model's final representation is projected to 512 dimensions.

**HOS Expert** — A Feature Tokenizer Transformer (FT-Transformer) operating on nine normalized cumulants (C20, C21, C40, C41, C42, C60, C61, C62, C63) plus derived statistics, for a 20-dimensional input. The transformer uses 3 layers, 8 heads, and embed dim 192 with FFN width 768. Cumulants are power-invariant normalized to remove amplitude ambiguity.

**Cyclostationary Expert** — A dilated Temporal Convolutional Network with dilation schedule [1, 2, 4, 8, 16, 32], 64 filters per layer, and kernel size 8. Operates on the 512-dimensional output of the Spectral Correlation Function (SCF) computed via the Frequency-smoothed Averaging Method (FAM) with 256 cycle frequencies and 1024-point FFT.

### Routing and Fusion

**Expert Choice routing** (Zhou et al., 2022) — Each expert selects its top-K preferred samples from the batch rather than each sample selecting experts. This eliminates load imbalance by construction and uses a router Z-loss to further regularize routing logits. With top-K=2 and capacity factor 1.25, each sample is processed by an average of two experts.

**Cross-attention fusion** — Two layers of bidirectional multi-head cross-attention (8 heads, embed dim 512) fuse the four weighted expert embeddings. Each expert embedding attends to all others, capturing inter-modal dependencies before the final classification step.

**Shared expert (DeepSeek-style)** — An always-active two-layer MLP operates on the concatenation of all four expert embeddings (2048-dim input, 512-dim output) and its output is added to the fused representation. This ensures that features common across modalities are never discarded by the routing mechanism.

### Hierarchical Classification

Three classification heads operate on the same 512-dimensional fused representation:

```
Level 1 (binary)   :  2 classes  — drone / no-drone
Level 2 (type)     : 15 classes  — drone type + link type
Level 3 (full)     : 50 classes  — make / model / protocol
```

Loss weights transition smoothly from emphasizing coarse predictions early in training ([0.5, 0.3, 0.2]) to emphasizing fine-grained predictions late ([0.1, 0.2, 0.7]) via a cosine schedule centered at epoch 100.

### Total Parameter Count (approximate)

| Component | Parameters |
|---|---|
| IQ Expert | 15–25 M |
| Spectrogram Expert | ~9 M |
| HOS Expert | 2–5 M |
| Cyclostationary Expert | 5–10 M |
| Router + Fusion + Heads | ~5 M |
| **Total** | **~36–54 M** |

---

## Project Structure

```
rfml/
├── main.py                        # CLI entry point (click)
├── setup.py                       # Package installation
├── requirements.txt               # Pinned dependencies
├── configs/
│   └── default.yaml               # Full pipeline configuration
├── data/
│   ├── download.py                # Dataset downloaders (RFUAV, DroneDetect, CardRF, DroneRF, Tampere)
│   └── dataset.py                 # PyTorch Dataset + DataLoader factory
├── features/
│   ├── pipeline.py                # Parallel feature extraction orchestrator
│   ├── spectrogram.py             # Complex STFT -> 3-channel spectrogram
│   ├── hos.py                     # Higher-order statistics / cumulants
│   ├── cyclostationary.py         # Spectral Correlation Function via FAM
│   ├── wavelet.py                 # Wavelet scattering (Kymatio)
│   └── augmentation.py            # RF-aware augmentation pipeline
├── models/
│   ├── experts/
│   │   ├── iq_expert.py           # SignalFormerIQ (Complex CNN + Transformer)
│   │   ├── spectrogram_expert.py  # EfficientNetSpec (EfficientNet-B2)
│   │   ├── hos_expert.py          # FTTransformerHOS (FT-Transformer)
│   │   └── cyclo_expert.py        # TCNCyclo (dilated TCN)
│   ├── fusion/
│   │   └── cross_attention.py     # Bidirectional cross-attention fusion
│   └── moe/
│       ├── moe_model.py           # DroneRFMoE top-level model
│       ├── router.py              # ExpertChoiceRouter
│       ├── losses.py              # HierarchicalLoss + auxiliary losses
│       └── load_balance.py        # Z-loss, utilization tracking
├── training/
│   ├── trainer.py                 # MoETrainer (4-phase loop)
│   ├── pretraining.py             # MaskedAutoencoder + ContrastiveLearning (MoCo-v3)
│   ├── curriculum.py              # SNRCurriculum + HierarchyScheduler
│   └── schedulers.py              # LR scheduler utilities
├── evaluation/
│   ├── evaluator.py               # MoEEvaluator (SNR-stratified, routing analysis)
│   ├── metrics.py                 # Hierarchical F1, balanced accuracy
│   ├── openset.py                 # OpenMax open-set recognition
│   └── visualization.py           # Confusion matrices, routing plots
├── utils/
│   ├── config.py                  # YAML config loader (OmegaConf)
│   ├── helpers.py                 # Seeding, device selection, checkpoint I/O
│   └── logging.py                 # Structured logging setup
└── scripts/
    ├── preprocess.py              # Raw data normalization and segmentation
    ├── create_shards.py           # WebDataset shard creation
    └── run_pipeline.sh            # Full pipeline bash runner
```

---

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.2.0
- CUDA 12.x (NVIDIA) or ROCm 6.x (AMD)

### Install

```bash
git clone <repository-url>
cd rfml

# Create and activate environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### CUDA (NVIDIA)

Standard PyTorch CUDA installation is sufficient:

```bash
pip install torch>=2.2.0 torchvision>=0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

### ROCm (AMD MI300X)

Install the ROCm-enabled PyTorch wheel and set the required environment variables:

```bash
pip install torch>=2.2.0 torchvision>=0.17.0 --index-url https://download.pytorch.org/whl/rocm6.0

export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=2
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3
```

These variables are also set automatically by `scripts/run_pipeline.sh` and by the training command when a ROCm configuration is detected.

### Optional Dependencies

| Package | Purpose |
|---|---|
| `kymatio>=0.4.0` | Wavelet scattering transform features |
| `wandb>=0.16.0` | Experiment tracking |
| `tensorboard>=2.15.0` | Local training metrics |

---

## Datasets

Five public drone RF datasets are supported. Combined they span 2.4 GHz and 5.8 GHz ISM bands, 37+ drone models, and multiple signal formats.

| Dataset | Size | Source | Format | Bands | Notes |
|---|---|---|---|---|---|
| **RFUAV** | 1.3 TB | HuggingFace (`RFUAV/RFUAV-1.3T`) | Binary IQ FP32 | 2.4 GHz, 5.8 GHz | 37 drone models, 100 MSps |
| **DroneDetect v2** | 10 GB | IEEE DataPort (ID: 17703) | Complex IQ `.dat` | 2.4375 GHz | 7 DJI/Parrot drones, manual download |
| **CardRF** | 65 GB | IEEE DataPort | MATLAB `.mat` | 2.4 GHz | 5 UAVs + 15 RF devices |
| **DroneRF** | 40 GB | Mendeley Data (DOI: 10.17632/s3c4gf5ng2.1) | CSV | 2.4 GHz | Automatic API download |
| **Tampere/Zenodo** | 5 GB | Zenodo (record: 4264467) | IQ int16 | 2.44 GHz, 5.8 GHz | Dual-band, checksum-verified |

**Download notes:**
- RFUAV is fetched automatically via `huggingface-hub` with resume support.
- DroneRF is fetched automatically via the Mendeley public REST API.
- Tampere/Zenodo is fetched via the Zenodo REST API with MD5/SHA-256 verification.
- DroneDetect v2 and CardRF require a free IEEE account and manual browser download from IEEE DataPort. The pipeline will print instructions and continue with other datasets.

---

## Feature Extraction

Each IQ segment (32 768 samples) is transformed into four independent feature representations that serve as inputs to the corresponding expert networks.

### Spectrograms

A complex STFT is computed by running separate STFTs on the I and Q channels, then recombining as `S_I + j*S_Q`. Three feature channels are derived from the complex spectrogram:

| Channel | Description |
|---|---|
| Magnitude | `log1p(|S|)` — log-compressed power |
| Phase | `angle(S)` — wrapped phase in `[-pi, pi]` |
| Instantaneous Frequency | Phase derivative along time, wrapped to `[-pi, pi]` |

Parameters: FFT size 512, hop length 256 (50% overlap), Hann window, output resized to 512x512 via bilinear interpolation, per-channel mean/std normalization.

### Higher-Order Statistics (HOS)

Nine complex cumulants are computed up to order six:

```
C20, C21, C40, C41, C42, C60, C61, C62, C63
```

All cumulants are power-invariant normalized (divided by the appropriate power of the second-order moment) to remove amplitude ambiguity. The resulting 20-dimensional feature vector is robust to gain fluctuations and is particularly discriminative for modulation-order classification.

### Cyclostationary Features

The Spectral Correlation Function (SCF) is estimated via the Frequency-smoothed Averaging Method (FAM). Parameters: 256 cycle frequencies, 1024-point FFT. The SCF captures periodic spectral structure introduced by modulation, coding, and framing, producing a 512-dimensional feature vector that encodes protocol-specific signatures invisible to standard power spectral density estimates.

### Wavelet Scattering

Second-order wavelet scattering coefficients are computed using Kymatio with J=8 octaves and Q=8 wavelets per octave. Scattering coefficients are stable under small deformations of the signal and complement the STFT by capturing transient structures at multiple time-frequency resolutions.

### RF-Aware Augmentation Pipeline

The `RFAugmentor` class applies a composable chain of channel impairments during training, each with a configurable probability:

| Augmentation | Default Probability | Description |
|---|---|---|
| AWGN | 0.80 | Additive white Gaussian noise, SNR drawn uniformly from [-20, 20] dB |
| Cyclic time shift | 0.50 | Random circular shift up to 50% of segment length |
| Carrier frequency offset (CFO) | 0.50 | Phase rotation `e^{j2pi*df*t}`, max 1000 Hz at 100 MSps |
| Amplitude scale | 0.50 | Uniform random gain in [0.5, 2.0] |
| Multipath fading | 0.30 | Rayleigh or Rician fading, 3 paths, max 10-sample delay |
| Interference injection | 0.20 | CW tone or narrowband noise, configurable SIR |

All augmentations operate directly on the `[2, N]` IQ tensor and are applied independently per sample. The pipeline accepts both single samples and batches.

---

## Training Pipeline

Training proceeds in four sequential phases of progressive difficulty and increasing end-to-end coupling.

### Phase 1: Self-supervised Pretraining (75 epochs)

Each expert is pretrained independently on 30% of the training data using two complementary objectives:

**Masked Autoencoder (MAE)** — 75% of input patches are masked; a lightweight Transformer decoder reconstructs the original signal from visible patches only. For IQ data, patches are 256-sample segments of the time axis. For spectrograms, patches are 32x32 spatial tiles. MSE reconstruction loss is computed only on masked patches.

**MoCo-v3 Contrastive Learning** — Two augmented views of each signal are encoded by an online encoder and a momentum encoder (EMA coefficient 0.999). InfoNCE loss with a learnable temperature is minimized symmetrically between online and momentum projections. RF-specific augmentations (cyclic time shift, frequency offset, SNR variation) are used to generate positive pairs.

| Hyperparameter | Value |
|---|---|
| Learning rate | 1e-3 |
| Subset fraction | 0.30 |
| MAE mask ratio | 0.75 |
| MoCo momentum | 0.999 |
| Projection dim | 256 |

### Phase 2: Supervised Curriculum Training (75 epochs)

Experts are trained on labeled data with SNR-based curriculum learning. Training begins with only high-SNR samples (>= +20 dB) and linearly introduces harder samples until the full range down to -10 dB is included by the final epoch.

```
Epoch 0  : threshold = +20 dB  (easy, clean signals)
Epoch 37 : threshold =  +5 dB  (mid-range)
Epoch 74 : threshold = -10 dB  (all samples)
```

The hierarchical loss uses early weights [0.5, 0.3, 0.2] for levels 1/2/3, transitioning to late weights [0.1, 0.2, 0.7] via a cosine schedule centered at epoch 100.

| Hyperparameter | Value |
|---|---|
| Learning rate | 5e-4 |
| SNR start | +20 dB |
| SNR end | -10 dB |

### Phase 3: Gating Network Training (35 epochs)

All expert parameters are frozen. Only the router, cross-attention fusion layers, shared expert, and classification heads are trained. This allows the routing mechanism to learn which experts to trust for which input types without disrupting the pretrained expert representations.

| Hyperparameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Expert parameters | Frozen |
| Router Z-loss coefficient | 0.001 |

### Phase 4: End-to-end Fine-tuning (15 epochs)

All parameters are unfrozen and trained jointly at a low learning rate. Gradient clipping at 1.0 prevents large updates from destabilizing the routing mechanism.

| Hyperparameter | Value |
|---|---|
| Learning rate | 1e-5 |
| Expert parameters | Unfrozen |
| Gradient clip | 1.0 |

### Shared Training Configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW (betas=[0.9, 0.999], weight decay=0.01) |
| Scheduler | Cosine annealing with warm restarts (T_0=10, T_mult=2, eta_min=1e-7) |
| Batch size | 128 |
| Gradient accumulation | 4 steps (effective batch = 512) |
| Precision | BF16 |
| Checkpoint interval | Every 5 epochs, keep top 3 |

### Expected Training Times (single A100 / MI300X, BF16)

| Phase | Duration |
|---|---|
| Pretraining | ~12–18 hours |
| Supervised curriculum | ~18–24 hours |
| Gating | ~4–6 hours |
| Fine-tuning | ~2–3 hours |
| **Total** | **~36–51 hours** |

---

## Evaluation

The evaluator produces a structured report with multiple analysis dimensions.

### Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall per-level accuracy |
| Balanced accuracy | Macro-averaged per-class recall |
| Precision / Recall | Per-class and macro-averaged |
| F1 (macro, weighted) | Macro and weighted F1 |
| Hierarchical F1 | F1 computed accounting for label hierarchy |
| Confusion matrix | Full NxN confusion matrix per level |

### SNR-Stratified Evaluation

Performance is measured separately at each SNR level from -20 dB to +30 dB in 2 dB steps. This produces accuracy-vs-SNR curves that reveal where each expert's contribution is most significant and at what SNR the system degrades gracefully.

### Open-set Recognition

OpenMax open-set recognition identifies inputs that do not belong to any training class. A Weibull distribution is fit to the tail of activation vectors (tail size 20) from training data. At inference, samples with low maximum class probability are flagged as unknown RF emitters.

### Expert Routing Analysis

For each test batch, the evaluator records which experts were selected and with what weights. Routing entropy, per-expert utilization fractions, and per-class routing preferences are reported. This analysis reveals whether the model has learned semantically meaningful specializations (e.g., the cyclostationary expert preferentially processing FHSS signals).

### Cross-dataset Generalization

When multiple datasets are available, the evaluator runs cross-dataset evaluation: training on a subset of datasets and testing on held-out datasets. This measures domain shift robustness across different receiver hardware, environments, and drone models.

---

## Usage

### Show System and Configuration Info

```bash
python main.py info
python main.py --config configs/default.yaml info
```

### Download Datasets

```bash
# Download all enabled datasets
python main.py download

# Download a specific dataset
python main.py download --datasets rfuav
python main.py download --datasets dronerf --datasets tampere_zenodo
```

### Extract Features

```bash
# Extract features for all datasets
python main.py extract-features

# Extract for a specific dataset with custom worker count
python main.py extract-features --dataset rfuav --workers 16
```

### Training

```bash
# Full 4-phase progressive training
python main.py train --phase all

# Run a specific phase only
python main.py train --phase pretrain
python main.py train --phase supervised
python main.py train --phase gating
python main.py train --phase finetune

# Resume from checkpoint
python main.py train --phase all --resume checkpoints/epoch_050.pt

# Enable torch.compile for additional throughput
python main.py train --phase all --compile
```

### Evaluation

```bash
# Standard evaluation
python main.py evaluate --checkpoint checkpoints/best.pt --output results/

# Include open-set recognition
python main.py evaluate --checkpoint checkpoints/best.pt --output results/ --open-set
```

### Full Pipeline

```bash
# Run all stages end-to-end via CLI
python main.py run-all

# Or via the shell script (also sets ROCm environment variables)
bash scripts/run_pipeline.sh
bash scripts/run_pipeline.sh configs/default.yaml DEBUG
```

### Multi-GPU Training

Wrap the training command with `torchrun` for distributed data-parallel training:

```bash
torchrun --nproc_per_node=4 main.py train --phase all --compile
```

### Custom Configuration

```bash
# Point to a custom config file
python main.py --config configs/my_experiment.yaml train --phase all
```

---

## GPU Support

### CUDA (NVIDIA)

No special configuration is required beyond a standard CUDA PyTorch installation. BF16 precision is used by default and requires Ampere or newer (A100, RTX 3090+). To use FP16 or FP32, set `project.precision` in the config.

```bash
python main.py train --phase all --compile
```

`torch.compile` with `mode="max-autotune"` is recommended for sustained throughput and is enabled via the `--compile` flag.

### ROCm (AMD MI300X)

Set the following environment variables before running any training or inference command:

```bash
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=2
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3
```

These variables are defined in `configs/default.yaml` under the `rocm.env` block and are applied automatically during training when the config is loaded. The shell script `scripts/run_pipeline.sh` also exports them unconditionally.

**Variable reference:**

| Variable | Purpose |
|---|---|
| `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | Reduces memory fragmentation for large-batch workloads |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | Prefers hipBLASLt for GEMM operations (better throughput) |
| `HIP_FORCE_DEV_KERNARG=1` | Forces device-side kernel arguments (stability fix) |
| `GPU_MAX_HW_QUEUES=2` | Limits hardware queues to avoid scheduler contention |
| `MIOPEN_FIND_MODE=3` | Uses exhaustive MIOpen kernel search |
| `MIOPEN_FIND_ENFORCE=3` | Enforces exhaustive search result caching |

### Memory Optimization Tips

- BF16 precision (`project.precision: bf16`) halves activation memory vs FP32.
- Gradient accumulation (`training.accumulation_steps: 4`) allows large effective batch sizes without proportional VRAM growth.
- WebDataset shards (`data/shards/`) stream data from disk rather than loading full datasets into RAM.
- The `pin_memory: true` and `num_workers: 12` settings in the config minimize data-loading latency on high-throughput GPUs.

---

## Configuration

The complete configuration is in `configs/default.yaml`. Key sections and their most commonly adjusted parameters:

### Data

```yaml
data:
  sample_length: 32768        # IQ samples per segment
  sampling_rate: 100_000_000  # samples per second (100 MSps for RFUAV)
  shard_size_mb: 1000         # WebDataset shard size
```

### Feature Extraction

```yaml
features:
  spectrogram:
    fft_size: 512
    hop_length: 256
    output_size: [512, 512]
  hos:
    max_order: 6
    feature_dim: 20
  cyclostationary:
    num_cycle_freqs: 256
    output_dim: 512
  wavelet:
    J: 8      # octaves
    Q: 8      # wavelets per octave
    order: 2  # second-order scattering
```

### MoE Router

```yaml
moe:
  num_experts: 4
  top_k: 2
  capacity_factor: 1.25
  router_type: "expert_choice"
  shared_expert: true
  load_balancing:
    method: "router_z_loss"
    z_loss_coeff: 0.001
```

### Classification Hierarchy

```yaml
classification:
  hierarchy:
    level1_classes: 2    # drone / no-drone
    level2_classes: 15   # drone type + link type
    level3_classes: 50   # make / model / protocol
```

### Augmentation

```yaml
data:
  augmentation:
    awgn_snr_range: [-20, 20]
    cfo_max_hz: 1000
    amplitude_scale: [0.5, 2.0]
    multipath_fading: true
    interference_injection: true
```

### Training Phases

Each phase block under `training.phases` accepts `epochs`, `lr`, and phase-specific keys:

```yaml
training:
  phases:
    pretrain:
      epochs: 75
      lr: 1e-3
      subset_fraction: 0.3
    supervised:
      epochs: 75
      lr: 5e-4
      curriculum: true
      snr_start_db: 20
      snr_end_db: -10
    gating:
      epochs: 35
      lr: 1e-4
      freeze_experts: true
    finetune:
      epochs: 15
      lr: 1e-5
```

---

## References

### Architecture

- Zhou, Y., et al. (2022). **Mixture-of-Experts with Expert Choice Routing.** NeurIPS 2022. — Expert Choice routing algorithm used in the MoE router.
- Dai, D., et al. (2024). **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.** — DeepSeek-style shared expert design.
- He, K., et al. (2022). **Masked Autoencoders Are Scalable Vision Learners.** CVPR 2022. — MAE pretraining objective.
- Chen, X., & He, K. (2021). **Exploring Simple Siamese Representation Learning.** — MoCo-v3 contrastive learning framework.
- Gorishniy, Y., et al. (2021). **Revisiting Deep Learning Models for Tabular Data.** NeurIPS 2021. — FT-Transformer used in HOS Expert.
- Tan, M., & Le, Q. (2019). **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.** ICML 2019. — EfficientNet-B2 backbone for Spectrogram Expert.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.** — TCN architecture used in Cyclostationary Expert.

### Signal Processing

- Gardner, W. A. (1994). **Cyclostationarity in Communications and Signal Processing.** IEEE Press. — Cyclostationary signal theory.
- Roberts, R. S., et al. (1991). **Computationally Efficient Algorithms for Cyclic Spectral Analysis.** IEEE Signal Processing Magazine. — FAM algorithm for SCF estimation.
- Mallat, S. (2012). **Group Invariant Scattering.** Communications on Pure and Applied Mathematics. — Wavelet scattering transform theory.

### Datasets

- **RFUAV**: HuggingFace `RFUAV/RFUAV-1.3T` — large-scale drone RF dataset, 37 models, 100 MSps.
- **DroneDetect v2**: IEEE DataPort ID 17703 — 7 DJI/Parrot drones, complex IQ.
- **CardRF**: IEEE DataPort — 5 UAVs + 15 RF devices, MATLAB format.
- **DroneRF**: Mendeley Data DOI `10.17632/s3c4gf5ng2.1` — CSV-format drone RF recordings.
- **Tampere/Zenodo**: Zenodo record 4264467 — dual-band IQ int16, University of Tampere.

### Open-set Recognition

- Bendale, A., & Boult, T. E. (2016). **Towards Open Set Deep Networks.** CVPR 2016. — OpenMax algorithm used for open-set evaluation.
