# RFML-MoE Paper Integration Plan
## Three-Paper Synthesis for Enhanced Drone RF Detection

**Date**: 2026-04-08
**Papers Analyzed**:
1. Tanveer et al. (2026) — "From Lab to Field Trials: Real-Time Multimodel Drone Detection in Low-SNR Environments" — IEEE AES Magazine
2. Fu et al. (2026) — "VMD and Gramian Angular Field Empowered CNN for Drone RF Signal Recognition" — IEEE Comm. Letters
3. Mandal & Satija (2023) — "Time-Frequency Multiscale CNN for RF-Based Drone Detection and Identification" — IEEE Sensors Letters

---

## Executive Summary

This plan integrates three complementary techniques into the existing 4-expert RFML-MoE framework to produce a **6-expert, SNR-hardened, field-deployable** drone RF detection system. The additions address three concrete gaps:

| Gap | Current State | Paper Solution | Expected Gain |
|-----|--------------|----------------|---------------|
| No energy-domain pre-filter | All samples run full pipeline | Neyman-Pearson energy gate (Tanveer) | P_fa bounded to 1%, ~60% compute savings on noise |
| STFT fails on non-stationary signals | SpectrogramExpert uses fixed-window STFT | VMD + GAF temporal correlation images (Fu) | +10-15% type accuracy at SNR < 5 dB |
| No joint time-frequency 1D learning | IQ (time-only) and Spec (freq-only) are separate | TFMS dual-branch CNN (Mandal) | +5-7% fine-grained classification F1 |

**Architecture evolution**: 4 experts → 6 experts + energy gate + EMD denoising

```
                        ┌─────────────────────────────────────────────────────┐
                        │                  PROPOSED PIPELINE                  │
                        └─────────────────────────────────────────────────────┘

Raw IQ [2, 32768]
    │
    ▼
┌──────────────────┐    NO     ┌────────────────────┐
│  Energy Detector │──────────►│ Return: "No Drone" │
│  (Neyman-Pearson)│           │ (skip all experts) │
│  P_fa = 0.01     │           └────────────────────┘
└────────┬─────────┘
         │ YES (drone detected)
         ▼
┌──────────────────┐
│ SNR Estimation   │──► snr_db tensor (passed to router)
│ + EMD Denoising  │──► iq_clean (if SNR < -5 dB)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION (parallel)             │
│                                                              │
│  IQ ──────────► [2, 32768]     (raw or EMD-denoised)        │
│  Spectrogram ─► [3, 512, 512]  (STFT: mag, phase, IF)      │
│  HOS ─────────► [20]           (cumulants C20..C63)         │
│  Cyclo ───────► [512]          (SCF via FAM)                │
│  VMD-GAF ─────► [3, 256, 256]  (GASF + GADF + raw GASF) NEW│
│  TFMS IQ ─────► [2, 32768]    (same IQ, different expert)NEW│
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    6 EXPERT EMBEDDINGS                        │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐         │
│  │SignalForm│ │EfficientN│ │FT-Trans │ │TCN      │         │
│  │IQ Expert │ │Spec Expt │ │HOS Expt │ │Cyclo Ex │         │
│  │15-25M    │ │~9M       │ │2-5M     │ │5-10M    │         │
│  └────┬─────┘ └────┬─────┘ └────┬────┘ └────┬────┘         │
│       │512         │512         │512        │512            │
│                                                              │
│  ┌──────────┐ ┌──────────┐                          NEW     │
│  │VMD-GAF   │ │TFMS-CNN  │                                  │
│  │CNN Expert│ │Dual-Brch │                                  │
│  │~5.8M     │ │~4.5M     │                                  │
│  └────┬─────┘ └────┬─────┘                                  │
│       │512         │512                                      │
└───────┼────────────┼─────────────────────────────────────────┘
        │            │
        ▼            ▼
┌──────────────────────────────────────────────────────────────┐
│              SNR-ADAPTIVE EXPERT CHOICE ROUTER                │
│                                                              │
│  Input: 6×512 = 3072-dim concat + 32-dim SNR embedding      │
│  SNR bias: low SNR → prefer VMD-GAF, TFMS                   │
│            high SNR → prefer IQ, Spectrogram                 │
│  Output: top-k=2 expert selection per sample                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│         CROSS-ATTENTION FUSION (sparse, top-k only)          │
│         + SHARED EXPERT (DeepSeek-style)                     │
│         → Fused 512-dim representation                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              HIERARCHICAL CLASSIFICATION HEADS                │
│                                                              │
│  Level 1: Binary (drone/no-drone)      ← 2 classes          │
│  Level 2: Type (DJI/Parrot/etc+link)   ← 15 classes         │
│  Level 3: Model (exact make/model)     ← 50 classes         │
└──────────────────────────────────────────────────────────────┘
```

---

## 1. Novel Techniques: What Each Paper Contributes

### 1.1 Paper 1 (Tanveer) — Energy Gate + EMD Denoising

**What the framework lacks**: No energy-domain detection gate. The pipeline unconditionally runs all experts on every sample, including pure noise. In field deployments at SNR < 0 dB, this wastes compute and produces false positives.

**EMD Denoising** — Empirical Mode Decomposition decomposes the analytic signal `z(t) = I(t) + jQ(t)` into Intrinsic Mode Functions (IMFs) via the sifting algorithm. The denoised signal discards high-frequency noise IMFs:

```
z_denoised(t) = Σ_{k=k_min}^{K} IMF_k(t)
```

Unlike STFT, EMD makes no stationarity assumption — critical for drone RF signals with rapid frequency variations from motor speed changes and FHSS protocols.

**Energy Detection** — Neyman-Pearson threshold for binary signal presence:

```
T = μ_noise + Q^{-1}(p_fa) · σ_noise        where p_fa = 0.01
Y_RF > T  →  H_p (drone present)
Y_RF ≤ T  →  H_a (noise only)
```

Self-calibrating via rolling buffer of recent "quiet" samples. At N=32768, detection probability P_d ≈ 1.0 for SNR > -40 dB due to sqrt(N) averaging gain.

**Lightweight FTLW Classifiers** — FTLW-RF (57K params, 1.4ms CPU inference) provides a fast binary decision ideal for edge Tier 1 deployment.

### 1.2 Paper 2 (Fu) — VMD + Gramian Angular Field

**What the framework lacks**: SpectrogramExpert uses STFT with fixed window, trading time-frequency resolution via the uncertainty principle. Drone RF signals are non-stationary — STFT is fundamentally mismatched.

**VMD** — Variational Mode Decomposition solves a constrained optimization that decomposes the signal into K narrowband AM-FM modes centered at learned frequencies ω_k:

```
min_{u_k, ω_k} Σ_k ‖∂_t [(δ(t) + j/πt) * u_k(t)] e^{-jω_k t}‖²_2
subject to: Σ_k u_k = f
```

Solved via ADMM. Each IMF `u_k` is a physically meaningful narrowband component. Effective IMFs are selected by Pearson Correlation Coefficient with the original signal (threshold ρ > 0.3).

**Gramian Angular Field** — Transforms 1D denoised signal to 2D temporal correlation image:

```
Step 1: Rescale x̃ to [-1, 1]
Step 2: Polar encoding → θ_n = arccos(x̃_n)
Step 3: GASF[i,j] = cos(θ_i + θ_j)    — angular summation (temporal similarity)
         GADF[i,j] = sin(θ_i - θ_j)    — angular difference (rate of change)
```

GAF encodes **temporal correlations** — fundamentally different from STFT's time-frequency representation. STFT loses long-range temporal structure by windowing; GAF preserves it as 2D structure that CNNs exploit. This is complementary to, not redundant with, the existing SpectrogramExpert.

**PAA** — Piecewise Aggregation Approximation compresses N-sample signal to n points (e.g., 32768 → 64), producing manageable 64×64 images. Complexity: O(N²_max) for full GAF, O(n²) after PAA.

### 1.3 Paper 3 (Mandal & Satija) — Time-Frequency Multiscale CNN

**What the framework lacks**: IQExpert is time-domain only (complex conv on raw IQ). SpectrogramExpert is frequency-domain only (STFT image). Neither jointly learns cross-domain features.

**TFMS-CNN** — Two parallel 1D CNN branches with shared gradient flow:

```
Branch 1 (Time):  |z(t)|      → L Conv1D blocks → GlobalAvgPool → [256]
Branch 2 (Freq):  |FFT(z(f))| → L Conv1D blocks → GlobalAvgPool → [256]
                                    ↓  merge (concat)  ↓
                                         [512]
                                     D dense layers
                                      → embed_dim
```

Conv block structure: `Conv1D(2^{i+4} filters, kernel k, stride 2) → ReLU → MaxPool(2) → Dropout(0.25)`

The key innovation is **gradient coupling**: errors in the frequency branch backpropagate through the merge layer and reshape the time branch's filters, creating automatically co-adapted representations. Ablation studies show 7.9–8.8% F1 improvement vs. single-branch networks.

**Critical distinction from IQExpert**: TFMS processes magnitude `|z(t)|` (scalar 1D), not complex `I+jQ` (2-channel). IQExpert preserves phase structure; TFMS preserves magnitude/power structure. They are complementary.

---

## 2. New Signal Processing Modules

### 2.1 Energy Detector — `features/energy_gate.py` (NEW)

```python
class EnergyDetector:
    """Neyman-Pearson energy detector for drone signal presence.
    
    Self-calibrating via rolling buffer of recent observations.
    Provides binary detection and SNR estimation.
    """
    
    def __init__(self, p_fa=0.01, buffer_size=1024, calibration_percentile=10.0):
        self.p_fa = p_fa
        self.buffer = torch.zeros(buffer_size)
        self.buf_ptr = 0
        self.buf_full = False
        self._mu = None    # noise mean power
        self._sigma = None # noise std power

    def calibrate(self, noise_iq: Tensor):
        """Initial calibration from known noise-only samples."""
        psd = (noise_iq[0]**2 + noise_iq[1]**2)  # |z|^2
        self._mu = psd.mean().item()
        self._sigma = psd.std().item()

    def detect(self, iq: Tensor) -> bool:
        """Returns True if drone signal present (energy > threshold)."""
        energy = (iq[0]**2 + iq[1]**2).mean().item()
        self._update_buffer(energy)
        T = self._mu + q_inv(self.p_fa) * self._sigma
        return energy > T

    def snr_estimate_db(self, iq: Tensor) -> float:
        """Estimate SNR relative to calibrated noise floor."""
        energy = (iq[0]**2 + iq[1]**2).mean().item()
        snr_linear = max(energy / self._mu - 1, 1e-10)
        return 10 * math.log10(snr_linear)
```

**Compute cost**: O(N) — negligible. <1ms on any platform including ESP32-P4.

### 2.2 EMD Denoiser — `features/emd_denoising.py` (NEW)

```python
class EMDDenoiser:
    """Empirical Mode Decomposition denoiser for IQ RF signals.
    
    Decomposes magnitude signal into IMFs via sifting, discards
    high-frequency noise IMFs, reconstructs clean IQ.
    """
    
    def __init__(self, num_imfs=4, noise_imfs=2, max_sifting=20):
        self.num_imfs = num_imfs       # total IMFs to extract
        self.noise_imfs = noise_imfs   # high-freq IMFs to discard

    def denoise(self, iq: Tensor) -> Tensor:
        """Denoise IQ by removing high-frequency IMFs from magnitude."""
        z = torch.complex(iq[0], iq[1])
        mag = z.abs()
        imfs = self._decompose(mag)
        denoised_mag = sum(imfs[self.noise_imfs:])
        scale = denoised_mag / (mag + 1e-10)
        return torch.stack([iq[0] * scale, iq[1] * scale], dim=0)
```

**Compute cost**: O(K × max_sifting × N) — ~50ms on ARM, too slow for ESP32. Use only on GPU/server path.

**Production note**: Use `PyEMD` library (`pip install EMD-signal`) for proper cubic spline interpolation instead of the linear approximation above.

### 2.3 VMD Extractor — `features/vmd.py` (NEW)

```python
class VMDExtractor:
    """Variational Mode Decomposition via ADMM optimization.
    
    Decomposes signal into K narrowband IMFs with adaptive
    center frequencies. Selects effective IMFs via PCC.
    
    Parameters:
        K=5 modes, α=2000 (bandwidth penalty), max_iter=500
    """
    
    def decompose(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Returns IMFs in freq domain [K, N] and center freqs [K]."""
        # ADMM loop: update u_k, ω_k, λ alternately
        # u_k^{n+1}(ω) = [f̂ - Σ_{i≠k} û_i + λ̂/2] / [1 + 2α(ω - ω_k)²]
        # ω_k^{n+1}   = ∫ ω|û_k|² dω / ∫ |û_k|² dω
        ...

    def select_effective_imfs(self, imfs, original, threshold=0.3):
        """Select IMFs with |PCC| > threshold vs original signal."""
        # ρ_k = Cov(IMF_k, s) / (σ_{IMF_k} · σ_s)
        ...
        return denoised_signal  # sum of effective IMFs
```

**Compute cost**: O(max_iter × K × N log N) — ~12ms GPU, ~200ms CPU per segment. **Must be precomputed** during dataset creation, not at inference time.

### 2.4 GAF Extractor — `features/gaf.py` (NEW)

```python
class GAFExtractor:
    """Gramian Angular Field image generation from 1D signal.
    
    Transforms denoised 1D signal into [3, n, n] image:
      Channel 0: GASF of VMD-denoised signal
      Channel 1: GADF of VMD-denoised signal  
      Channel 2: GASF of raw signal (SNR proxy for router)
    """
    
    def __init__(self, n=256, method="both"):
        self.n = n  # PAA target length → n×n output image (256×256 per Fu et al.)

    def extract(self, x_denoised: Tensor, x_raw: Tensor) -> Tensor:
        """Generate 3-channel GAF image [3, n, n]."""
        x_paa = self._paa(x_denoised, self.n)     # [n]
        x_scaled = self._rescale(x_paa)            # [-1, 1]
        phi = torch.arccos(x_scaled.clamp(-1+1e-6, 1-1e-6))

        GASF = torch.cos(phi.unsqueeze(0) + phi.unsqueeze(1))  # [n, n]
        GADF = torch.sin(phi.unsqueeze(0) - phi.unsqueeze(1))  # [n, n]

        # Channel 2: raw signal GASF (undenoised) — gives router SNR info
        x_raw_paa = self._paa(x_raw, self.n)
        phi_raw = torch.arccos(self._rescale(x_raw_paa).clamp(-1+1e-6, 1-1e-6))
        GASF_raw = torch.cos(phi_raw.unsqueeze(0) + phi_raw.unsqueeze(1))

        return torch.stack([GASF, GADF, GASF_raw], dim=0)  # [3, n, n]
```

**Compute cost**: O(n²) after PAA — ~20ms on ARM for 256×256. Matches the spectrogram's resolution class.

**Design choice (not in Fu's paper)**: Channel 2 (raw GASF) encodes the difference between denoised and raw signals — this is a strong SNR proxy that the router can learn to exploit for SNR-adaptive expert selection.

---

## 3. New Expert Architectures

### 3.1 VMD-GAF Expert — `models/experts/vmd_gaf_expert.py` (NEW)

**Architecture** (adapted from Fu et al., matching paper's 256×256 resolution):

```
Input: [B, 3, 256, 256]  (GASF_denoised, GADF_denoised, GASF_raw)

Conv Block 1: Conv2d(3, 32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)   → [32, 128, 128]
Conv Block 2: Conv2d(32, 64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → [64, 64, 64]
Conv Block 3: Conv2d(64, 128, 3×3, pad=1) → BN → ReLU → MaxPool(2×2) → [128, 32, 32]
Conv Block 4: Conv2d(128, 256, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)→ [256, 16, 16]

AdaptiveAvgPool2d(4) → [256, 4, 4]
Flatten: 256 × 4 × 4 = 4096
FC(4096, 512) → GELU → LayerNorm → Dropout(0.5)
FC(512, embed_dim=512)
```

**Parameters**: ~5.8M (comparable to SpectrogramExpert's ~9M)

**Key interface methods** (matching existing expert protocol):
- `get_embedding(x) → [B, 512]` — for router/fusion
- `forward(x) → [B, num_classes]` — standalone classification
- `freeze()` / `unfreeze()` — for phased training

### 3.2 TFMS-CNN Expert — `models/experts/tfms_expert.py` (NEW)

**Architecture** (adapted from Mandal & Satija):

```
Time Branch:                                  Frequency Branch:
  Input: |z(t)| [B, 1, N]                      Input: |FFT(z)| [B, 1, N//2+1]
  │                                             │
  Conv1D(1→16, k=5, s=2) → ReLU → Pool(2)     Conv1D(1→16, k=5, s=2) → ReLU → Pool(2)
  Conv1D(16→32, k=5, s=2) → ReLU → Pool(2)    Conv1D(16→32, k=5, s=2) → ReLU → Pool(2)
  Conv1D(32→64, k=5, s=2) → ReLU → Pool(2)    Conv1D(32→64, k=5, s=2) → ReLU → Pool(2)
  Conv1D(64→128, k=5, s=2) → ReLU → Pool(2)   Conv1D(64→128, k=5, s=2) → ReLU → Pool(2)
  Conv1D(128→256, k=5, s=2) → ReLU → Pool(2)  Conv1D(128→256, k=5, s=2) → ReLU → Pool(2)
  │                                             │
  AdaptiveAvgPool1d(1) → [B, 256]              AdaptiveAvgPool1d(1) → [B, 256]
                          │
                    Concatenate → [B, 512]
                          │
                    FC(512, 128) → ReLU → Dropout(0.25)
                    FC(128, 128) → ReLU → Dropout(0.25)
                    FC(128, 128) → ReLU → Dropout(0.25)
                          │
                    FC(128, embed_dim=512) → LayerNorm → GELU
```

**Parameters**: ~4.5M total (2.1M per branch + 0.3M dense/projection)

**Input construction**:
- Time branch: `|z(t)| = sqrt(I² + Q²)` — magnitude envelope
- Freq branch: `|FFT(I + jQ)|` — magnitude spectrum (one-sided)

**ONNX-exportable**: All ops are standard (Conv1D, MaxPool1d, Linear, ReLU). Unlike IQExpert's ComplexConv1d, TFMS can be directly exported for ARM deployment.

---

## 4. Router & Fusion Modifications

### 4.1 Dimensional Changes in `models/moe/moe_model.py`

```python
# Current → Proposed changes:

self.experts = nn.ModuleDict({
    "iq": IQExpert(),
    "spectrogram": SpectrogramExpert(),
    "hos": HOSExpert(),
    "cyclo": CycloExpert(),
    "vmd_gaf": VMDGAFExpert(),      # NEW (+5.8M params)
    "tfms": TFMSExpert(),            # NEW (+4.5M params)
})

# Router input: 6 × 512 = 3072 (was 4 × 512 = 2048)
self.router = SNRAdaptiveRouter(input_dim=3072, num_experts=6, top_k=2)

# Fusion: 6 modalities (was 4)
self.fusion = CrossAttentionFusion(num_modalities=6, embed_dim=512, num_layers=1)
#                                                                    ↑ reduce from 2 to 1
# O(M²) scaling: 4→6 experts = 12→30 cross-attention pairs per layer
# Keeping layers=2 would mean 60 pairs — too expensive. Use layers=1.

# SharedExpert input: 3072 (was 2048)
self.shared_expert = SharedExpert(input_dim=3072, embed_dim=512)
```

### 4.2 SNR-Adaptive Router — `models/moe/router.py` (EXTEND)

```python
class SNRAdaptiveRouter(ExpertChoiceRouter):
    """Expert Choice router conditioned on estimated SNR.
    
    At low SNR: biases toward VMD-GAF (index 4) and TFMS (index 5)
    At high SNR: biases toward IQ (index 0) and Spectrogram (index 1)
    
    The bias is learned during Phase 3 gating training.
    """
    
    def __init__(self, input_dim=3072, num_experts=6, top_k=2, snr_dim=32):
        super().__init__(input_dim + snr_dim, num_experts, top_k)
        self.snr_embedder = nn.Sequential(nn.Linear(1, snr_dim), nn.Tanh())
        self.snr_bias = nn.Parameter(
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.3])  # initial: prefer new experts at low SNR
        )

    def forward(self, x, snr_db=None):
        if snr_db is not None:
            snr_norm = (snr_db.unsqueeze(-1) / 30.0).clamp(-1, 1)  # normalize [-30,30]→[-1,1]
            snr_emb = self.snr_embedder(snr_norm)                   # [B, 32]
            x_aug = torch.cat([x, snr_emb], dim=-1)                 # [B, 3104]
            # Low SNR → negative snr_norm → positive bias scale
            bias_scale = (-snr_norm).clamp(0, 1)
            effective_bias = self.snr_bias * bias_scale              # [B, 6]
        else:
            x_aug = F.pad(x, (0, 32))
            effective_bias = None
        
        out = super().forward(x_aug)
        if effective_bias is not None:
            out.router_logits = out.router_logits + effective_bias
        return out
```

### 4.3 Sparse Cross-Attention Optimization

With 6 experts, full cross-attention (30 pairs) is expensive. Optimization: only compute cross-attention between the top-k=2 **routed** experts per sample, not all 6:

```python
# In CrossAttentionFusion.forward():
# Only fuse embeddings selected by router (2 of 6)
# Zero-pad unrouted expert positions
# Reduces 30 attention pairs → 2 pairs per sample
```

---

## 5. Low-SNR Hardening Strategy

### 5.1 Two-Stage Detection Architecture

The energy gate sits **before** the MoE model (not inside the router). It prevents the entire pipeline from running on pure noise:

```python
# In inference wrapper:
if energy_gate.detect(iq):
    snr_est = energy_gate.snr_estimate_db(iq)
    if snr_est < EMD_THRESHOLD:  # e.g., -5 dB
        iq = emd_denoiser.denoise(iq)  # clean IQ for all experts
    return moe_model(extract_features(iq, snr_db=snr_est))
else:
    return {"level1": NO_DRONE, "confidence": 0.99}
```

### 5.2 EMD as Universal Preprocessor at Low SNR

When the energy gate fires but SNR < -5 dB, EMD-denoised IQ replaces raw IQ for **all** downstream feature extractors (spectrogram, HOS, cyclostationary, VMD-GAF, TFMS). This compounds with each expert's own noise handling.

### 5.3 SNR-Conditioned Expert Routing

The `snr_db` estimate from the energy gate propagates to the router as an auxiliary feature. The router learns:
- **Low SNR** (< 0 dB): Route to VMD-GAF (denoising + temporal correlation) and TFMS (dual-domain redundancy)
- **High SNR** (> 10 dB): Route to IQ (phase-sensitive) and Spectrogram (high-resolution)
- **Mid SNR** (0–10 dB): Mixed routing, data-driven

---

## 6. Training Strategy Updates

### 6.1 Phase 1: Self-Supervised Pretraining (75 epochs)

**Existing experts**: Unchanged (MAE for IQ, MoCo-v3 for Spectrogram, etc.)

**VMD-GAF Expert — Contrastive Pretraining**:
- Positive pairs: Apply VMD with different random ω_k initializations to same signal → two different GAF images of the same underlying drone
- Negative pairs: GAF images from different signals
- Loss: InfoNCE with learnable temperature (MoCo-v3 framework)

**TFMS Expert — Masked Frequency Prediction**:
- Mask 20% of frequency bins in the FFT branch input (set to zero)
- Train to predict masked frequency content from remaining time + frequency features
- This is the frequency-domain equivalent of MAE, natural for TFMS's explicit frequency branch

### 6.2 Phase 2: Supervised Curriculum (75 epochs)

**SNR Curriculum**: Unchanged — all experts receive progressively harder (lower-SNR) samples.

**VMD-specific curriculum**: PCC threshold for IMF selection decreases over training:
- Epochs 1–25: threshold = 0.7 (strict, high-quality IMFs only)
- Epochs 26–50: threshold = 0.5
- Epochs 51–75: threshold = 0.2 (accept noisier IMFs for robustness)

### 6.3 Phase 3: Gating Training (35 epochs)

All 6 expert parameters **frozen**. Train only:
- SNRAdaptiveRouter (MLP + SNR embedder + bias)
- CrossAttentionFusion (1-layer sparse)
- SharedExpert
- Classification heads

**Z-loss coefficient**: Increase from 0.001 → 0.002 (more experts = stronger load balancing needed)

**Utilization monitoring**: Ensure VMD-GAF and TFMS experts are utilized ≥25% (not starved by routing collapse)

### 6.4 Phase 4: End-to-End Fine-Tuning (15 epochs)

**Differential learning rates**:
- New experts (VMD-GAF, TFMS): 2× base LR (less pre-training to catch up)
- Existing experts (IQ, Spec, HOS, Cyclo): 0.5× base LR (protect learned features)
- Router + fusion: 1× base LR

---

## 7. Dataset Integration

### 7.1 LowSNR_DroneRF (Tanveer)

| Parameter | Value |
|-----------|-------|
| Samples | 137.56 million |
| Sample rate | 10 MHz |
| Bandwidth | 20 MHz |
| Center freq | 2.4 / 2.44 GHz |
| Hardware | USRP B210, VERT2450 |
| Classes | drone vs. no-drone (binary) |
| Modes | OFF, connected, armed, flying |
| Source | Kaggle (Lowsnr_dronerf) |

**Integration**: Add to `data/download.py`. Primary use: train energy gate calibration, binary detection head, and low-SNR robustness for all experts. The 137.56M samples provide massive training data specifically for the SNR regime (-20 to +20 dB) where existing datasets are sparse.

### 7.2 CardRF (Fu)

| Parameter | Value |
|-----------|-------|
| Sample rate | 20 GSa/s |
| Segment size | 1024 points per slice |
| Classes | 5 UAVs, 6 controllers, BT×3, WiFi×2 |
| Images per class | 1000 (GAF @ 256×256) |
| Source | IEEE DataPort |

**Integration**: Requires resampling from 20 GSa/s to RFML-MoE's operating rate. Use sliding windows to concatenate consecutive 1024-point slices into longer segments. Train VMD-GAF expert supervised on CardRF first (validated >90.6% in paper), then integrate into MoE fine-tuning.

### 7.3 DroneRF (Mandal & Satija)

| Parameter | Value |
|-----------|-------|
| Drones | Phantom 3, Bebop, AR Drone |
| Segments | 227 × 5.25s |
| Modes | 4 (controller, hovering, flight, flight+video) |
| Classes | 2 (binary), 4 (type), 10 (flight mode) |
| Source | IEEE DataPort [18] |

**Integration**: Sample rate mismatch — DroneRF is ~1 kHz baseband. Add `nn.AdaptiveAvgPool1d(5000)` at TFMS input to downsample from 32768 to paper's validated 5000 samples.

### 7.4 Unified Batch Schema

```python
batch = {
    "iq":           Tensor[B, 2, 32768],      # raw or EMD-denoised IQ
    "spectrogram":  Tensor[B, 3, 512, 512],   # STFT (mag, phase, IF)
    "hos":          Tensor[B, 20],             # cumulants
    "cyclo":        Tensor[B, 512],            # SCF features
    "vmd_gaf":      Tensor[B, 3, 256, 256],    # NEW: GASF + GADF + raw GASF
    "tfms_iq":      Tensor[B, 2, 32768],      # NEW: same as iq (separate key)
    "label_binary": Tensor[B],                 # 0=no-drone, 1=drone
    "label_type":   Tensor[B],                 # 15-class
    "label_full":   Tensor[B],                 # 50-class
    "snr_db":       Tensor[B],                 # NEW: estimated or annotated SNR
    "dataset_id":   Tensor[B],                 # NEW: which dataset (for weighting)
}
```

---

## 8. Edge Deployment: Three-Tier Architecture

### Feasibility Matrix

| Technique | ESP32-P4 | ARM (Pi 5) | GPU Server | Latency (ARM) |
|-----------|----------|------------|------------|---------------|
| Energy Detector | **YES** | YES | YES | <1ms |
| EMD Denoising | NO | MAYBE | YES | ~50ms |
| VMD Decomposition | NO | NO | YES | ~12ms GPU |
| GAF Transform | NO | **YES** | YES | ~5ms |
| TFMS-CNN (INT8 ONNX) | NO | **YES** | YES | ~30ms |
| VMD-GAF Expert (INT8) | NO | MAYBE | YES | ~80ms |
| Full 6-Expert MoE | NO | NO | **YES** | ~60ms |
| FTLW-RF (from Paper 1) | **YES** | YES | YES | 1.4ms |

### Tier 1: ESP32-P4 (Edge Detection)

```
Energy Detector → binary "drone present?"
Model: Statistical features + FTLW-RF (200KB, 57K params)
Latency: <15ms
Trigger: Always-on monitoring at 2.4 GHz
Action: If drone detected → wake Tier 2 via UART/SPI
```

### Tier 2: ARM Pi 5 / Indiedroid Nova (Type Classification)

```
GAF Transform + TFMS-CNN (INT8 ONNX, ~1.8MB)
→ 4-class type identification (DJI / Parrot / hobby / unknown)
Latency: ~45ms
Trigger: Tier 1 "drone detected" alert
Action: If confidence < 0.7 → forward to Tier 3
```

TFMS is the best candidate for ARM because:
- All ops are ONNX-standard (Conv1D, MaxPool1d, Linear)
- No custom complex arithmetic (unlike IQExpert)
- INT8 quantization preserves accuracy (Conv1D + ReLU is quantization-friendly)

### Tier 3: GPU Server (Full Identification)

```
Full 6-Expert RFML-MoE
→ 50-class model identification + flight mode
Latency: ~60ms on T4
Trigger: Tier 2 low-confidence or fine-grained ID needed
```

---

## 9. Performance Projections

### 9.1 Binary Detection (Level 1)

| Condition | Current (4-expert) | With Energy Gate | Improvement |
|-----------|-------------------|-----------------|-------------|
| SNR > 10 dB | ~97% | ~98% | +1% |
| SNR 0–10 dB | ~92% | ~95% | +3% |
| SNR -5–0 dB | ~75% | ~90% | **+15%** |
| SNR < -10 dB | ~60% | ~85% | **+25%** |
| False alarm (noise) | ~5-10% | **≤1%** | Bounded by P_fa |
| Compute on noise | Full pipeline | Gate only | **~60% savings** |

### 9.2 Type Classification (Level 2, 15 classes)

| Condition | Current (4-expert) | With VMD-GAF + TFMS | Improvement |
|-----------|-------------------|---------------------|-------------|
| SNR > 10 dB | ~88% | ~90% | +2% |
| SNR 0–10 dB | ~78% | ~85% | **+7%** |
| SNR < 0 dB | ~65% | ~78% | **+13%** |

Based on: Fu et al. STFT baseline 78.3% vs. VMD-GAF 90.6% at SNR=0 dB (Δ=12.3%)

### 9.3 Fine-Grained Classification (Level 3, 50 classes)

| Condition | Current (4-expert) | With TFMS + 6-expert routing | Improvement |
|-----------|-------------------|------------------------------|-------------|
| All SNRs | ~72% | ~77% | **+5%** |

Based on: Mandal & Satija dual-branch F1 gain of 7.9–8.8% vs. single-branch

### 9.4 Total Parameter Budget

| Component | Current | Proposed | Delta |
|-----------|---------|----------|-------|
| IQ Expert | 15-25M | 15-25M | 0 |
| Spectrogram Expert | ~9M | ~9M | 0 |
| HOS Expert | 2-5M | 2-5M | 0 |
| Cyclostationary Expert | 5-10M | 5-10M | 0 |
| **VMD-GAF Expert** | — | **5.8M** | **+5.8M** |
| **TFMS Expert** | — | **4.5M** | **+4.5M** |
| Router | ~0.5M | ~0.8M | +0.3M |
| Fusion | ~2M | ~2.5M | +0.5M |
| SharedExpert | ~1M | ~1.5M | +0.5M |
| **Total** | **36-54M** | **47-65M** | **+11.6M** |

The +11.6M parameter increase (21-30%) is modest relative to the capability gains.

---

## 10. Implementation Schedule & Dependencies

```
Week 1-2 ─── PRIORITY 1: Energy Gate + EMD Preprocessing
│             ├── features/energy_gate.py (NEW)
│             ├── features/emd_denoising.py (NEW)
│             ├── features/pipeline.py (MODIFY — integrate gate + denoiser)
│             └── Validation: <1% false alarm on noise, SNR estimation accuracy
│
Week 2-3 ─── PRIORITY 2: VMD + GAF Infrastructure
│             ├── features/vmd.py (NEW)
│             ├── features/gaf.py (NEW)
│             ├── features/pipeline.py (MODIFY — add vmd_gaf output)
│             ├── data/dataset.py (MODIFY — add vmd_gaf tensor to batch)
│             └── Validation: VMD on chirp → K IMFs with ordered ω_k; GAF of sine → stripes
│
Week 3-4 ─── PRIORITY 3: VMD-GAF Expert
│             ├── models/experts/vmd_gaf_expert.py (NEW)
│             ├── models/experts/__init__.py (MODIFY)
│             ├── models/moe/moe_model.py (MODIFY — 5 experts interim)
│             ├── training/trainer.py (MODIFY — contrastive SSL for VMD-GAF)
│             └── Validation: standalone VMD-GAF accuracy on CardRF ≥ 85%
│
Week 4-5 ─── PRIORITY 4: TFMS Expert
│             ├── models/experts/tfms_expert.py (NEW)
│             ├── models/moe/moe_model.py (MODIFY — 6 experts final)
│             ├── training/pretraining.py (MODIFY — MaskedFrequencyPredictor)
│             └── Validation: standalone TFMS accuracy on DroneRF ≥ 95% binary
│
Week 5-6 ─── PRIORITY 5: SNR-Adaptive Router
│             ├── models/moe/router.py (MODIFY — add SNRAdaptiveRouter)
│             ├── models/moe/moe_model.py (MODIFY — use SNRAdaptiveRouter)
│             ├── training/trainer.py (MODIFY — pass snr_db to model)
│             └── Validation: router correctly biases to VMD-GAF/TFMS at low SNR
│
Week 6-7 ─── PRIORITY 6: Dataset Integration (parallel with 2-5)
│             ├── data/download.py (MODIFY — LowSNR_DroneRF + CardRF)
│             ├── data/dataset.py (MODIFY — unified schema with snr_db, dataset_id)
│             ├── scripts/preprocess.py (MODIFY — precompute VMD features)
│             └── configs/default.yaml (MODIFY — 6-expert config)
│
Week 7-8 ─── PRIORITY 7: Edge Export
              ├── rtl-ml-exp/src/energy_detector.py (NEW — port gate to edge)
              ├── scripts/export_tfms_onnx.py (NEW — ONNX + INT8 quantization)
              └── Validation: TFMS INT8 on Pi 5 < 50ms, energy gate on ESP32 < 1ms
```

### Dependency Graph

```
P1 (Energy Gate) ──────────────────────────────────────► P5 (SNR Router)
                                                              │
P2 (VMD + GAF) ──► P3 (VMD-GAF Expert) ──► P4 (TFMS) ──► P5 ──► P7 (Edge)
                                                              │
P6 (Datasets) ─────────────────────────────────────────► P5
```

P1 and P2 can start in parallel. P3 depends on P2. P4 depends on P3 (router dims). P5 depends on P4. P6 can run alongside P2-P5. P7 depends on P4+P5.

---

## 11. Key Architectural Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | VMD-GAF is **preprocessing-dependent** (receives pre-computed [3,64,64], not raw IQ) | VMD is too slow for real-time (~200ms CPU). Precompute during dataset creation, not inference. |
| 2 | TFMS is **IQ-receiving** (same raw IQ as IQExpert, processed differently) | Magnitude + FFT computation is fast (<1ms). No preprocessing bottleneck. |
| 3 | Energy gate is **pre-router** (prevents entire model from running) | Implements Tanveer's binary detection as a hard gate, not a soft weight. Saves ~60% compute on noise. |
| 4 | CrossAttentionFusion uses **1 layer** (not 2) for 6 experts | O(M²) scaling: 4→6 = 12→30 pairs per layer. 2 layers would be 60 pairs — too expensive. |
| 5 | Channel 2 of GAF image is **raw GASF** (not in Fu's paper) | Encodes denoised-vs-raw difference. Gives router an SNR proxy without explicit SNR annotation. |
| 6 | **Sparse cross-attention** (only top-k routed experts) | Full 6×5=30 pair attention is wasteful when only 2 experts are active per sample. |
| 7 | VMD center frequencies ω_k are **cached per dataset** | Stable per signal class, expensive to compute. Store as metadata alongside features. |
| 8 | TFMS for edge (not IQExpert) | TFMS uses standard ONNX ops. IQExpert's ComplexConv1d needs custom op registration. |

---

## 12. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| VMD too slow for dataset preprocessing | Medium | High | GPU-accelerated VMD via batched FFT; precompute once, cache as .pt files |
| GAF 64×64 too low resolution | Low | Medium | Make n configurable (64→128→256); benchmark accuracy vs. compute |
| Router collapse to 4 original experts | Medium | High | Monitor utilization per expert; increase z-loss coeff; curriculum warmup for new experts |
| EMD sifting divergence on short segments | Low | Low | Cap max_sifting iterations; fallback to bandpass filter |
| LowSNR_DroneRF label noise (field data) | Medium | Medium | Cross-validate with energy gate labels; discard ambiguous samples |
| TFMS frequency branch input size mismatch | Low | Low | AdaptiveAvgPool1d(5000) at input handles any N |

---

## 13. Validation Criteria

Each priority phase must pass before proceeding:

**P1**: Energy gate achieves P_fa ≤ 0.01 on 10K noise samples, P_d ≥ 0.90 at SNR = -5 dB

**P2**: VMD produces K=5 ordered IMFs on synthetic chirp; GAF of sine wave matches expected pattern

**P3**: Standalone VMD-GAF expert achieves ≥85% on CardRF 5-class (paper reports 90.6%)

**P4**: Standalone TFMS-CNN achieves ≥95% binary detection on DroneRF (paper reports 99.89%)

**P5**: Router utilization of each expert is ≥15% (no expert starved); SNR bias correctly shifts routing

**P6**: Unified dataset loader produces correct batch shapes; precomputed features load without errors

**P7**: ONNX TFMS INT8 on Pi 5 runs in <50ms; accuracy drop from FP32 < 2%

---

## References

1. L. Tanveer et al., "From Lab to Field Trials: Real-Time Multimodel Drone Detection in Low-SNR Environments," IEEE AES Magazine, Feb. 2026. DOI: 10.1109/MAES.2025.3624717
2. Y. Fu, C. Zhang, Z. He, "VMD and Gramian Angular Field Empowered CNN for Drone RF Signal Recognition," IEEE Comm. Letters, 2026. DOI: 10.1109/LCOMM.2026.3680926
3. S. Mandal, U. Satija, "Time-Frequency Multiscale Convolutional Neural Network for RF-Based Drone Detection and Identification," IEEE Sensors Letters, vol. 7, no. 7, Jul. 2023. DOI: 10.1109/LSENS.2023.3289145
