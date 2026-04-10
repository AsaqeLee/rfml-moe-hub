# Drone RF signal detection pipeline for AMD MI300X with FPGA edge deployment

**A production-grade training pipeline for drone RF classification is now feasible using the 1.3 TB RFUAV benchmark dataset, a multi-modal Mixture-of-Experts architecture, and AMD's MI300X cluster, with edge inference achievable at sub-10-microsecond latency on Xilinx RFSoC.** This blueprint synthesizes 2023–2025 advances across datasets, feature engineering, model architectures, distributed training, MoE gating, FPGA deployment, and evaluation to provide a complete implementation roadmap. The core insight is that no single feature representation dominates across all SNR regimes—spectrograms excel below 0 dB, raw IQ wins above 10 dB, and cyclostationary features generalize best across environments—which makes the MoE architecture not merely desirable but architecturally necessary. What follows is a seven-part technical blueprint covering every component from data ingestion to field deployment.

---

## Part A: Dataset catalog and synthetic augmentation strategy

### Publicly available datasets

The drone RF dataset landscape transformed in 2025 with the release of **RFUAV**, a 1.3 TB benchmark containing 37 distinct UAV types with raw IQ data at 100 MSps—an order-of-magnitude leap over all prior datasets. The complete catalog of usable public datasets:

| Dataset | Year | Size | Drones | Format | Bands | Source |
|---------|------|------|--------|--------|-------|--------|
| **RFUAV** | 2025 | 1.3 TB | 37 types | Binary IQ (fp32) | 2.4/5.8 GHz | HuggingFace, GitHub |
| **DroneDetect v2** | 2021 | Multi-GB | 7 DJI+Parrot | Complex IQ (.dat) | 2.4375 GHz | IEEE DataPort |
| **CardRF** | 2022 | 65 GB | 5 UAVs + 15 devices | .mat | 2.4 GHz | IEEE DataPort |
| **MPACT DroneRC** | 2020 | 124 GB | 17 controllers | .mat (20 GHz osc.) | 2.4 GHz | IEEE DataPort |
| **DroneRF** | 2019 | 40 GB | 3 (Parrot, DJI) | CSV amplitude | 2.4 GHz | Mendeley Data |
| **Tampere/Zenodo** | 2020 | Multi-GB | 10 models | IQ int16 | 2.44 + 5.8 GHz | Zenodo (CC-BY) |
| **KU Leuven/RMA** | 2024 | Multi-GB | Multiple | .mat IQ complex | 2.44 GHz | KU Leuven RDR |
| **DroneRFb-Spectra** | 2024 | 14,460 spectrograms | 7 brands | 512×512 STFT images | 3 ISM bands | IEEE DataPort |
| **DroneRFa** | 2024 | Large | ~25 types | RF signal | ISM bands | JEIT |
| **AirID** | 2020 | Multi-GB | 4 UAV-mounted radios | Binary IQ + SigMF | 2.4 GHz | GENESYS Lab |
| **DJI DroneID samples** | 2023 | Small | N/A (protocol) | IQ float32 | 2.4/5.8 GHz | GitHub (proto17, RUB-SysSec) |
| **Kaggle Noisy Drone RF** | 2024 | Moderate | Derived from KU Leuven | Spectrograms | 2.4 GHz | Kaggle |

**Critical gap analysis.** Most datasets cover only **2.4 GHz**. Only RFUAV and Tampere/Zenodo include 5.8 GHz. C-band and 900 MHz are entirely absent from public datasets, requiring synthetic generation or custom data collection. No public dataset includes ELRS, Crossfire, or custom FHSS protocols—these must be captured via SDR or synthesized.

**Recommended primary training corpus.** Combine RFUAV (37 drone types, 1.3 TB, both 2.4/5.8 GHz) with DroneDetect v2 (interference conditions), CardRF (Wi-Fi/Bluetooth discrimination), and Tampere/Zenodo (dual-band anechoic captures). Total: approximately **1.5 TB** raw data, fitting comfortably within the 40 TB NVMe scratch.

### Synthetic data generation

**RF-Diffusion** (ACM MobiCom 2024) represents the state-of-the-art for synthetic RF signal generation. It adapts denoising diffusion probabilistic models to the RF domain using a Hierarchical Diffusion Transformer with complex-valued operators. Cross-domain evaluation showed **4.7%–11.5% accuracy improvement** when used for augmentation. The open-source implementation (`github.com/mobicom24/RF-Diffusion`) can be retrained on drone IQ data to generate labeled synthetic samples for underrepresented protocols.

For 900 MHz and C-band gap-filling, physics-based channel simulation using GNU Radio with 3GPP/ITU multipath models applied to resampled RFUAV data provides a pragmatic alternative. RFUAV's built-in augmentation toolkit (`utils.preprocessor.data_augmentation`) supports AWGN injection from -20 to +20 dB in 2 dB steps.

### IQ data augmentation pipeline

The augmentation strategy must be signal-aware. Key techniques validated in the literature:

- **Cyclic time shift** (from IQFM, 2025): circular shifting of IQ samples preserving inter-antenna phase structure—the single most effective augmentation for contrastive pre-training
- **SNR-stratified noise injection**: calibrated AWGN at target SNR levels, enabling curriculum learning from clean to noisy signals
- **Carrier frequency offset simulation**: applying random phase rotations `x(t)·exp(j2πΔft)` to simulate receiver CFO, critical for cross-hardware generalization
- **Multipath channel augmentation**: Rayleigh/Rician fading applied to IQ data simulates real propagation environments
- **Interference injection**: mixing real Wi-Fi and Bluetooth captures from DroneDetect's interference subsets with clean drone signals
- **Amplitude scaling**: random gain adjustment (0.5×–2.0×) prevents models from relying on absolute power levels

---

## Part B: Multi-modal feature extraction architecture

The strongest empirical evidence indicates that **no single representation dominates across all operating conditions**. Complex spectrograms achieve 0.842 balanced accuracy at -12 dB SNR where raw IQ achieves only 0.413 (Glüge et al., 2024). Conversely, raw IQ networks outperform at high SNR by preserving fine-grained phase information. Cyclic cumulant features uniquely **generalize perfectly** across datasets with different CFO distributions where IQ networks fail entirely (Snoap et al., MILCOM 2022). This motivates the multi-modal MoE design.

### Raw IQ representation (Expert 1)

Input format: 2×N tensor (I channel, Q channel) with N = 2048–32768 samples depending on protocol bandwidth. Normalization: per-sample RMS power normalization followed by zero-mean scaling per channel. Complex-valued convolutions preserve the coupling between I and Q that real-valued networks lose—**up to 34% higher classification accuracy** demonstrated for device identification (arXiv 2202.09777). Implementation: `complextorch` library (arXiv 2309.07948) for PyTorch complex-valued layers.

### Spectrogram representation (Expert 2)

STFT parameters optimized for drone protocols: **FFT size 512, 50% overlap, Hann window**. For 60 MSps sampling, this yields approximately 234 time frames per 2-millisecond observation window. Use complex spectrograms (separate FFT of real and imaginary parts) rather than magnitude-only—validated by Glüge et al. to preserve phase information critical for protocol discrimination. Linear frequency scale preferred over mel-scale, as mel was designed for human auditory perception and misaligns with drone RF spectral structure.

### Higher-order statistics features (Expert 3)

Extract normalized cumulants through 6th order: **Ĉ₂₀, Ĉ₂₁, Ĉ₄₀, Ĉ₄₁, Ĉ₄₂, Ĉ₆₀, Ĉ₆₁, Ĉ₆₂, Ĉ₆₃**. The normalization `Ĉ_{p,q} = C_{p,q}/(C₂₁)^{p/2}` cancels power level effects, providing distance-invariant features. Fourth-order cumulants separate PSK/QAM families; sixth-order separates 16-QAM from 64-QAM. Feature vector dimensionality: 9–20 features per segment (compact, suitable for tabular models). Computation is O(N) per cumulant and runs efficiently on CPU during data preprocessing.

### Cyclostationary features (Expert 4)

The Spectral Correlation Function (SCF) and cyclic cumulants provide the strongest cross-domain generalization. The key finding from cyclostationary.blog (W.A. Gardner's group): **"IQ-input networks do not generalize, but cyclic-cumulant-input networks generalize very well."** Cyclic features extract OFDM parameters (subcarrier count, cyclic prefix length, symbol time) that fingerprint drone protocols regardless of receiver hardware or channel conditions. Computational complexity is O(N·log(N)·K) where K = number of cycle frequencies tested—GPU acceleration essential via custom CUDA/HIP kernels.

### Wavelet scattering transform (supplementary)

Implemented via the **Kymatio** library (v0.4, supports PyTorch backend with GPU acceleration). Second-order scattering `S₂x(t, λ₁, λ₂) = ||x * ψ_{λ₁}| * ψ_{λ₂}| * φ_J(t)` captures non-stationary features missed by STFT—critical for frequency-hopping protocols like FHSS and ELRS. Computational cost is moderate: O(N·J·Q) where J = octaves, Q = wavelets per octave. Use as an additional input channel to the spectrogram expert rather than a separate expert.

### Bispectrum features (for RF fingerprinting)

The bispectrum `B(f₁,f₂) = E[X(f₁)·X(f₂)·X*(f₁+f₂)]` is blind to Gaussian noise and captures hardware nonlinearities producing device-specific patterns. A 2024 Nature Scientific Reports paper demonstrated **near-zero error rate** on 352 emitter classes using 224×224 bispectral images with EfficientNetB0, from very short signal subsamples (~56 ppm of original). Bispectrum is computationally heavy (O(N²)) but valuable for the make/model/individual discrimination task at the finest taxonomy level.

---

## Part C: Model architecture specifications

### IQ-domain expert — SignalFormer hybrid

**SignalFormer** (Sensors 2023, PMC) is purpose-built for drone RF identification with CNN-based tokenization, dilation time-frequency convolution blocks (D-TFCB), and gated self-attention. It achieved **97.57% accuracy under Gaussian noise and 98.03% under co-frequency interference** for coarse-grained drone identification. Architecture: CNN tokenizer → T/F-encoder (time and frequency transformer blocks) → classification head. Alternative for maximum IQ performance: complex-valued CNN (DC-CNN) achieving 99.5% on 4-class drone recognition with smaller model size than real-valued equivalents.

For the IQ expert backbone, use a **1D complex-valued CNN** (7 conv layers, 64→128→256 filters, kernel size 7, GroupNorm) feeding into a 4-layer transformer encoder with 8 attention heads and 512-dimensional embeddings. Parameter count: approximately **15–25M**. The `complextorch` package provides drop-in complex Conv1d, BatchNorm, and Linear layers.

### Spectrogram-domain expert — EfficientNet-B4

EfficientNet offers the best accuracy-efficiency tradeoff for spectrogram classification, achieving **96.31% accuracy** on drone spectrograms while outperforming ResNet50 (94.22%) and ViT (73.69%) in comparative evaluations. The compound scaling (depth/width/resolution) naturally adapts to different spectrogram resolutions. For maximum capacity, Swin Transformer is an option, but EfficientNet trains 2–3× faster with comparable accuracy. Input: 512×512 linear-frequency spectrograms with 3 channels (magnitude, phase, instantaneous frequency). Parameter count: approximately **19M** for B4 variant.

The RFUAV benchmark (2025) validates this choice, benchmarking ViT, Swin, ResNet, EfficientNet, and MobileNet across 37 drone types on spectrograms and finding EfficientNet competitive across all scales.

### HOS-domain expert — FT-Transformer

**FT-Transformer** (Gorishniy et al., NeurIPS 2021) consistently outperforms other deep learning models on tabular data and is the recommended architecture for the compact HOS feature vectors. Configuration: numerical features embedded via learned linear transformations, 3 transformer layers, 8 attention heads, embedding dimension 192. For comparison, XGBoost achieves **99.96% binary drone detection** on DroneRF statistical features—use as an ensemble baseline. Parameter count: approximately **2–5M**.

### Cyclostationary expert — temporal convolutional network

A TCN with dilated causal convolutions processes the spectral correlation function output. Configuration: `nb_filters=64, kernel_size=8, dilations=(1,2,4,8,16,32)`, providing a receptive field covering the full cycle frequency range. TCNs are parallelizable (unlike LSTMs) and capture long-range dependencies in cyclostationary features. Library: `keras-tcn` or PyTorch `locuslab/TCN`. Parameter count: approximately **5–10M**.

### Multi-modal fusion via cross-attention

Each expert produces a 512-dimensional embedding. Cross-attention fusion learns inter-modal dependencies:

```
Q_iq = E_iq · W_q,  K_spec = E_spec · W_k,  V_spec = E_spec · W_v
Attention(Q_iq, K_spec, V_spec) = softmax(Q_iq · K_spec^T / √d) · V_spec
```

This is applied bidirectionally between all modality pairs. A 2-layer cross-attention module with 8 heads fuses the four expert embeddings before the MoE gating layer. Alternative: late fusion via confidence-weighted voting provides a simpler baseline (typically 1–3% below learned fusion).

### Hierarchical classification head

Implement a **branching DNN** (LH-DNN architecture, IEEE TNNLS 2024) with shared backbone → level-specific branches:

- **Level 1 head**: Binary (drone / no-drone) — 2 outputs
- **Level 2 head**: Multi-class (drone type + link type) — ~15 outputs  
- **Level 3 head**: Full taxonomy (make/model/protocol) — ~50+ outputs

The key innovation from LH-DNN is **lexicographic projection**: coarser classifications are guaranteed not to degrade when training finer levels. Hierarchical loss with progressive weighting:

```python
# Early training: emphasize coarse
loss = 0.5 * L_binary + 0.3 * L_type + 0.2 * L_full
# Late training: emphasize fine-grained  
loss = 0.1 * L_binary + 0.2 * L_type + 0.7 * L_full
```

---

## Part D: Mixture of Experts gating network

### Architecture design

The MoE layer combines four modality-specific experts via a learned gating network. The recommended architecture draws from **MoE-AMC** (arXiv 2312.02298)—the first MoE applied to automatic modulation classification, achieving **71.76% on RML2018.01a** (surpassing prior SOTA by ~10%)—and **FuseMoE** (NeurIPS 2024) for multi-modal sensor fusion.

**Gating network**: MLP-based router taking concatenated low-dimensional projections from all four expert encoders as input:

```python
class SignalMoERouter(nn.Module):
    def __init__(self, input_dim=2048, num_experts=4, top_k=2):
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_experts)
        )
    def forward(self, x):
        logits = self.gate(x)
        # Add Jitter noise during training for exploration
        if self.training:
            logits += torch.randn_like(logits) * 0.01
        weights, indices = torch.topk(F.softmax(logits, dim=-1), k=self.top_k)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # renormalize
        return weights, indices
```

### Routing strategy — Expert Choice with top-2

**Expert Choice routing** (Zhou et al., NeurIPS 2022) is recommended over standard token-choice routing. In Expert Choice, each expert selects its top-c samples from the batch rather than each sample choosing experts. This guarantees **perfect load balance by construction**, eliminating the need for auxiliary load-balancing losses and achieving **2× training efficiency** improvement over GShard and Switch Transformer approaches. Expert Choice is ideal for signal classification since it is non-autoregressive.

For the multi-modal signal pipeline, **top-2 routing** provides the best tradeoff: two experts are activated per sample, enabling complementary representations (e.g., spectrogram expert for low-SNR samples plus IQ expert for fine-grained discrimination) while keeping compute at 2× a single expert.

### Load balancing and stability

Even with Expert Choice, monitor and enforce expert utilization via:

- **Router Z-loss** (ST-MoE): `L_z = (1/B) · Σ(log Σ exp(h_j))²` with coefficient 0.001, preventing sharp routing distributions
- **DeepSeek-V3 loss-free balancing** (arXiv 2408.15664) as an alternative: dynamic per-expert bias adjusted after each step based on recent load—achieves better performance AND balance than auxiliary losses
- **Expert specialization monitoring**: track pairwise cosine similarity between expert weight matrices; if similarity exceeds 0.7, increase routing noise or apply orthogonality loss

### Shared expert (DeepSeek-style)

Add one **always-active shared expert** alongside the four specialist experts. The shared expert captures common signal features (noise floor characteristics, general spectral shape) while specialists focus on modality-specific patterns. This follows the DeepSeek-V2/V3 architecture where shared experts improve overall performance by 1–2% without additional routing complexity.

### Implementation

Use **Tutel** (`github.com/microsoft/Tutel`) for the MoE layer—it has explicit ROCm support (fp64/fp32/fp16) and supports DeepSeek gating functions:

```python
from tutel import moe as tutel_moe
moe_layer = tutel_moe.moe_layer(
    gate_type={'type': 'top', 'k': 2, 'capacity_factor': 1.5},
    model_dim=512,
    experts={'type': 'ffn', 'count_per_node': 5,  # 4 specialists + 1 shared
             'hidden_size_per_expert': 2048}
)
```

**DeepSpeed MoE** (`deepspeedai/DeepSpeed`) is the alternative, with full ROCm support since v0.6 and native MI300X compatibility.

---

## Part E: MI300X x8 training pipeline

### ROCm environment and configuration

Use **ROCm 7.2.0** (production) with the official Docker image `rocm/pytorch-training:v25.4`. This image includes PyTorch 2.7+, ROCm libraries, and all MI300X-specific optimizations. Critical environment variables:

```bash
export TORCH_NCCL_HIGH_PRIORITY=1        # High-priority RCCL streams
export GPU_MAX_HW_QUEUES=2               # Limit compute + RCCL stream contention
export HIP_FORCE_DEV_KERNARG=1           # Faster kernel argument passing
export TORCH_BLAS_PREFER_HIPBLASLT=1     # hipBLASLt for optimized GEMM
export NCCL_MIN_NCHANNELS=112            # Full xGMI bandwidth utilization
export HSA_NO_SCRATCH_RECLAIM=1          # Prevent scratch memory overhead
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
echo 0 > /proc/sys/kernel/numa_balancing # Disable NUMA auto-balancing
```

MI300X provides **192 GB HBM3 per GPU** (1.536 TB aggregate across 8 GPUs), **5.3 TB/s memory bandwidth per GPU**, and **1,307.4 TFLOPS BF16** peak per GPU. The 8 GPUs are fully connected via Infinity Fabric (xGMI) with **336 GB/s** aggregate bandwidth per GPU.

### Distributed training — FSDP with full sharding

**FSDP (Fully Sharded Data Parallel)** is the recommended strategy for single-node 8×MI300X, per AMD's official guidance. Key configuration:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
    mixed_precision=bf16_policy,
    device_id=torch.cuda.current_device(),
    use_orig_params=True,
)
```

Critical rules: always use **TP=1** (no tensor parallelism—FSDP handles everything), always use all **8 GPUs**, and enable gradient checkpointing only if OOM occurs (unlikely given 192 GB per GPU for signal classification models).

### Precision — BF16 as default

**BF16** is the optimal default for MI300X signal classification training. It matches FP32's exponent range (8 bits), eliminating the need for loss scaling while delivering **1,307.4 TFLOPS** per GPU—8× faster than FP32 matrix operations. The wide dynamic range of RF data (spanning >60 dB) makes BF16's exponent range essential; FP16 risks overflow. FP8 (E5M2/E4M3) is available on MI300X at 2,614.9 TFLOPS but adds calibration complexity inappropriate for signal models under 1B parameters.

### Data loading — FFCV on NVMe

The 40 TB NVMe scratch can host the entire training corpus locally. **FFCV** (.beton format) is recommended for maximum throughput:

```python
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField

writer = DatasetWriter('/scratch/beton/iq_train.beton', {
    'iq_data': NDArrayField(dtype=np.float32, shape=(2, 32768)),
    'spectrogram': NDArrayField(dtype=np.float32, shape=(3, 512, 512)),
    'hos_features': NDArrayField(dtype=np.float32, shape=(20,)),
    'label_binary': IntField(),
    'label_type': IntField(),
    'label_full': IntField(),
    'snr_db': NDArrayField(dtype=np.float32, shape=(1,)),
})
```

FFCV provides **1.6× speedup** over standard PyTorch DataLoader via page-organized storage, multithreaded workers sharing CUDA context, and async GPU transfers. Configure 16 DataLoader workers (160 vCPU ÷ 8 GPUs ≈ 20, minus compute overhead), `pin_memory=True`, `prefetch_factor=4`. With **1920 GB RAM**, the OS page cache holds a substantial fraction of the preprocessed dataset in memory.

### Memory budget per GPU

For a combined MoE model (~200M total parameters, 50M active per sample):

| Component | With FSDP (per GPU) |
|-----------|-------------------|
| Sharded parameters (200M × 2B ÷ 8) | ~50 MB |
| Sharded optimizer (Adam, 2× FP32) | ~200 MB |
| Sharded gradients | ~50 MB |
| Activations (batch-dependent) | 10–80 GB |
| RCCL buffers | ~3 GB |
| PyTorch context | ~2 GB |
| **Available for data/compute** | **~100–175 GB** |

This leaves enormous headroom. Practical batch size: **4096–8192 per GPU** (IQ samples at 2×32768×4B = 256 KB each). Effective global batch with 4-step gradient accumulation: **131,072–262,144 samples**.

### Three-phase training curriculum

**Phase 1 — Self-supervised pre-training (epochs 0–50).** Masked autoencoder on all unlabeled IQ data. Mask 75% of patches (patch size 256 samples), reconstruct via lightweight transformer decoder. This learns general RF signal structure without labels. Follow with contrastive fine-tuning (MoCo-v3) using RF-specific augmentations: cyclic time shift, frequency offset, SNR variation, amplitude scaling, phase rotation. Pre-training increases sample efficiency by **~10×** and improves few-shot adaptation to new drone types.

**Phase 2 — Curriculum supervised training (epochs 50–150).** Progressive label refinement:
- Epochs 50–70: Binary classification (drone/no-drone), lr=1e-3
- Epochs 70–100: Multi-class (drone type + link), lr=5e-4
- Epochs 100–150: Full taxonomy (make/model/protocol), lr=1e-4

SNR-based self-paced learning starts with high-SNR samples (>20 dB), progressively introducing lower-SNR data. The SNR threshold decreases linearly from +20 dB to -10 dB over training.

**Phase 3 — MoE joint fine-tuning (epochs 150–200).** Initialize expert backbones from Phase 2 weights. Train the gating network from scratch. Apply Expert Choice routing with capacity factor 1.25. Monitor expert utilization—each expert should handle 15–30% of samples. Apply router Z-loss (coefficient 0.001) for stability.

---

## Part F: Edge deployment on Xilinx RFSoC

### Knowledge distillation pipeline

Distill the MoE teacher ensemble into a compact student for FPGA deployment. The **"Every Expert Matters"** approach (arXiv 2502.12947) is critical: sample from non-activated experts during distillation because they possess valuable knowledge that standard KD misses. Combine logit-based distillation (temperature T=4, KL divergence) with feature-based distillation (L2 alignment of intermediate representations).

Target student architecture: **1D-CNN VGG10-variant** with 7 convolutional layers (64→128→128→256→256→256→256 filters, kernel size 7) + 3 fully-connected layers. Parameter count: **100K–600K**. This architecture was validated on RFSoC by Tridgell et al. at the University of Sydney, achieving **488,000 classifications/second at 8 μs latency** on the ZCU111.

A prior AMC distillation study demonstrated that an Inception-ResNet teacher (311.77 MB, 93.09% accuracy) can be distilled into a CNN3 student (**0.37 MB**) with accuracy boosted from 79.81% to **89.36%** through KD—the student can approach or exceed standalone teacher accuracy with proper technique.

### Quantization-aware training

Two deployment paths with different quantization approaches:

**Path A — Ultra-low latency (custom RTL via hls4ml).** Train with **Brevitas** (Xilinx/AMD QAT library) targeting ternary weights + INCRA (incrementally increasing activation precision). INCRA applies lower precision activations in early layers (where throughput is highest) and higher precision in deeper layers (where throughput drops after pooling), yielding **~5% accuracy boost with <1% hardware overhead**. Export via `hls4ml.converters.convert_from_pytorch_model()` → Vivado HLS → bitstream.

**Path B — Standard DPU deployment (Vitis AI).** Quantize with `vai_q_pytorch` to INT8, using 1000 calibration samples from the training set. Vitis AI Compiler generates .xmodel targeting the DPUCZDX8G configuration. INT4 is feasible via Xilinx's 4-bit XDPU (WP521), achieving **1.5–2.0× performance over INT8** with ~1.66% mAP reduction after progressive fine-tuning.

For both paths, complex-valued operations are decomposed into separate I and Q channels before quantization—this is the standard approach used in all published AMC-on-FPGA implementations.

### RFSoC system architecture

The Xilinx RFSoC (e.g., ZCU111 with ZU28DR) integrates RF-ADCs (up to **5 GSPS, 14-bit**), FPGA fabric (~425K LUTs, ~4272 DSP slices), and ARM Cortex-A53 processors on a single chip. The complete inference pipeline:

```
RF Antenna → ADC (4–5 GSPS) → DDC/Decimation (to 60–100 MSps) → 
I/Q Ring Buffer → Channelizer (per-band splitting) →
CNN Inference Core (PL fabric) → Classification Result → 
DMA → ARM CPU (aggregation, alerting, logging)
```

**Latency analysis for <10 ms target.** The benchmark RFSoC AMC implementation achieves **8 μs** classification latency—a **1250× margin** below the 10 ms target. Even the slower Vitis AI DPU path delivers 1–10 ms for small signal classifiers. The system is decisively real-time.

| Deployment Approach | Latency | Throughput | Model Size |
|---|---|---|---|
| Custom RTL (hls4ml, ternary) | **5–8 μs** | 488K cls/sec | 100K–600K params |
| Vitis AI DPU (INT8) | **1–10 ms** | Varies by config | 100K–2M params |
| ARM CPU fallback | 10–100 ms | ~1K cls/sec | Any |

Resource utilization for the VGG10-INCRA design on ZCU111: **211K LUTs (49.6%)**, 324K FFs (38.1%), 512 BRAMs (48.3%), 1407 DSPs (32.9%)—leaving sufficient fabric for the RF data converter IP and DMA infrastructure.

### Structured pruning for FPGA

The hls4ml team's **FPGA resource-aware structured pruning** (arXiv 2308.05170) formulates pruning as a knapsack problem mapping weight groups to DSP blocks and BRAM. Results: **55–92% DSP reduction, up to 81% BRAM reduction** with minimal accuracy loss. The implementation is open-sourced in hls4ml (`github.com/fastmachinelearning/hls4ml/tree/hardware-aware-pruning`).

Recommended compression pipeline order: **Train → Distill → Prune (structured) → Fine-tune → Quantize (QAT) → Export → Deploy**. Power consumption for the complete RFSoC system including RF subsystem: **10–25W total**, with the CNN inference core consuming 3–8W—over **2.3× more efficient** in GOP/watt than equivalent GPU solutions.

---

## Part G: Evaluation framework and benchmarking protocol

### Multi-level evaluation table structure

A comprehensive drone RF evaluation requires six standardized tables:

**Classification performance** — report per-level accuracy, per-class precision/recall/F1, macro/micro/weighted F1, and confusion matrices at each hierarchy level. The RFUAV benchmark (2025) with 37 drone types and open-source evaluation tools (`github.com/kitoweeknd/RFUAV`) should serve as the primary benchmark, supplemented by DroneDetect v2 for interference robustness.

**SNR-stratified performance** — accuracy-vs-SNR curves from -20 to +30 dB in 2 dB steps, following the RadioML evaluation protocol. Report minimum detectable SNR at 80%, 90%, and 95% accuracy thresholds. Per-class accuracy at critical SNR levels (-10, 0, +10 dB) reveals which drone types fail first. **Balanced accuracy** (not overall accuracy) is the correct metric for imbalanced SNR strata.

**Hierarchical metrics** — beyond level-wise accuracy, compute hierarchical F1: `hF1 = 2·hP·hR/(hP+hR)` where hP and hR use ancestor-augmented prediction/ground-truth sets. Tree-distance weighted error `d(pred, true) = path_length(pred, LCA) + path_length(true, LCA)` penalizes taxonomy-distant errors more heavily. The `hiclass` Python library and `github.com/RomanPlaud/revisitingHTC` provide implementations.

**Latency and throughput** — single-sample (batch=1) inference latency is the relevant metric for real-time detection. Report end-to-end latency including FFT/STFT preprocessing. For FPGA: report resource utilization (LUTs, DSPs, BRAMs, FFs), clock frequency, and power. Warmup-excluded timings with 95th-percentile latency.

### Cross-dataset generalization

**Leave-one-environment-out** testing is essential: train on N-1 capture environments, test on the held-out environment. Report accuracy gap between in-domain and cross-domain. Key finding: cyclic cumulant features generalize **perfectly** across datasets (Snoap et al.), while raw IQ features suffer 15–30% accuracy drops. Cross-receiver evaluation trains on data from receivers R₁...Rₖ and tests on unseen receiver R_{k+1}. Domain generalization via Fourier phase alignment + knowledge distillation improves by ~5% over standard domain adaptation.

**Open-set recognition** is critical for unknown drone types. Use **OpenMax** (Bendale & Boult, CVPR 2016) with Weibull-fitted activation vectors—validated at 98% accuracy for radar signals at -10 to +10 dB SNR. Report AUROC for unknown detection, FPR@TPR95, and the "openness" metric `O = 1 - sqrt(C_train/C_test)`.

### Adversarial robustness

RF adversarial attacks are **much more powerful than classical jamming**: at perturbation-to-noise ratio (PNR) ≤ 1, adversarial perturbations achieve 100% misclassification (Sadeghi & Larsson, IEEE JSAC 2019). The PNR metric `||r||²/||n||²` replaces the ε-ball from image adversarial ML. Evaluate under FGSM, PGD, and universal adversarial perturbations at PNR values from 0.01 to 10. Multi-modal MoE architectures inherently provide **cross-domain non-transferability**: attacks crafted for time-domain classifiers transfer poorly to frequency-domain classifiers, making the multi-expert design a natural defense. Use the **Adversarial Robustness Toolbox** (`github.com/Trusted-AI/adversarial-robustness-toolbox`) for standardized evaluation.

---

## Conclusion: implementation roadmap and key architectural decisions

This pipeline achieves drone RF detection across the full taxonomy—from binary presence detection through make/model/protocol discrimination—by combining four complementary signal representations in a Mixture-of-Experts architecture trained on the MI300X cluster and deployed on Xilinx RFSoC at microsecond latencies.

**Three decisions most impact success.** First, the choice of **Expert Choice routing** over standard token-choice routing eliminates load-balancing issues that plague sparse MoE training, providing 2× training efficiency. Second, the **three-phase curriculum** (self-supervised → hierarchical supervised → MoE joint fine-tuning) is essential because the hierarchical label structure and SNR variation create training instabilities when attacked simultaneously. Third, the **INCRA quantization strategy** for FPGA deployment—using lower precision in early layers where throughput is highest—provides 5% accuracy gains over uniform quantization at negligible hardware cost.

**The 900 MHz and C-band coverage gap** is the most significant limitation. No public dataset covers these bands for drone signals. Addressing this requires either custom SDR collection campaigns (USRP B210 at 900 MHz for Crossfire/ELRS, USRP X310 at C-band for DJI OcuSync 3) or physics-based synthetic generation using RF-Diffusion retrained on the target bands. The RFUAV team's data collection methodology (documented in arXiv 2503.09033) provides a reproducible protocol.

The **MI300X cluster's 1.5 TB aggregate VRAM** is dramatically oversized for signal classification models—even the full MoE ensemble at 200M parameters with FSDP consumes under 1 GB per GPU for model state. The true advantage is enabling **massive batch sizes** (131K+ effective) that accelerate contrastive pre-training convergence and allow exhaustive SNR-stratified curriculum sampling within each batch. The 40 TB NVMe scratch comfortably hosts the entire 1.5 TB combined training corpus with room for augmented variants, preprocessed spectrograms, and checkpoint storage.

**Total estimated training time**: 3–5 days for the complete three-phase pipeline on 8×MI300X, based on scaling from reported training times for similar signal classification models on smaller GPU clusters. Edge deployment latency of 5–8 μs on RFSoC provides a 1250× margin below the 10 ms real-time requirement, leaving substantial compute headroom for multi-band parallel classification and confidence aggregation.