# Training a 200M MoE drone detector on one MI300X

**A single AMD MI300X can train the full 200M-parameter MoE pipeline in approximately 5.5–7 days — and the model's memory footprint barely scratches the surface of the GPU's 192 GB VRAM.** The static memory for model weights, optimizer states, and gradients totals just **3.2 GB (1.7% of capacity)**, leaving ~188 GB for activations and large batch sizes. This makes the MI300X one of the few single GPUs where this entire pipeline is not only feasible but comfortable. No distributed frameworks, no memory offloading, no heroic optimization needed — just plain PyTorch with BF16 autocast and a few ROCm-specific tuning flags.

The key trade-off is time, not memory. An 8×MI300X cluster completes the same training in ~1 day at comparable cost. But for iterative research, debugging, and architecture search, the single-GPU setup is simpler, faster to set up, and avoids MI300X's known multi-GPU networking pain points.

---

## The memory math is overwhelmingly favorable

The 200M-parameter MoE model is trivially small relative to 192 GB of HBM3. Here is the complete static memory budget:

| Component | Calculation | Size |
|-----------|------------|------|
| BF16 model weights | 200M × 2 bytes | 400 MB |
| FP32 master copy (AdamW) | 200M × 4 bytes | 800 MB |
| FP32 momentum (AdamW) | 200M × 4 bytes | 800 MB |
| FP32 variance (AdamW) | 200M × 4 bytes | 800 MB |
| BF16 gradients | 200M × 2 bytes | 400 MB |
| **Total static** | | **3.2 GB** |
| HIP/ROCm context + fragmentation | | ~2 GB |
| **Total non-activation** | | **~5.2 GB** |

That leaves **~187 GB purely for activations and data buffers**. Activation memory scales linearly with batch size and depends heavily on the spectrogram expert's input resolution. Per-sample activation estimates across the four experts plus gating network:

- **Conservative** (224×224 spectrograms): ~300 MB/sample → max batch size **~530** without checkpointing
- **Moderate** (128×128 RF spectrograms): ~200 MB/sample → max batch size **~800**
- **Optimistic** (smaller inputs, sparse MoE routing): ~100 MB/sample → max batch size **~1,600**

With gradient checkpointing enabled (60% activation reduction, ~25% compute overhead), these limits roughly double — reaching batch sizes of **1,350–4,000**. However, **gradient checkpointing is probably unnecessary** for this configuration. Even at batch size 256 with conservative activation estimates, total memory usage is under 95 GB — barely half the available VRAM.

The practical recommendation: start at batch size 128, measure actual memory with `torch.cuda.max_memory_allocated()`, then binary-search upward. Larger batches improve MI300X utilization since its matrix cores need problem sizes above ~200 GFLOPs to approach peak throughput. **Batch sizes of 512–1024 should be the target**, with gradient accumulation to reach the desired effective batch size.

---

## Training completes in roughly one week with progressive phasing

The critical variable is dataset sample count. For 1.5 TB of RF IQ data at ~10 KB per segment (the most realistic assumption for spectral analysis), the dataset contains approximately **150 million samples**. At the MI300X's effective BF16 throughput of ~500 TFLOPS (40% MFU), the model processes roughly **35,000 samples/second**.

The recommended progressive training pipeline cuts total time by ~15% compared to naïve end-to-end training:

| Phase | Description | Active params | Epochs | Estimated time |
|-------|------------|--------------|--------|---------------|
| 1a–1d | Self-supervised pre-training (each expert individually) | 5–25M each | 75 (30% subset) | ~1.5 days |
| 2a–2d | Supervised curriculum training (each expert individually) | 5–25M each | 75 | ~1.5 days |
| 3 | Freeze experts, train gating + shared layers only | ~141M (no expert gradients) | 35 | ~1.0 day |
| 4 | Unfreeze all, end-to-end fine-tuning with small LR | Full 200M | 15 | ~1.0 day |
| **Total** | | | | **~5–5.5 days** |

Training experts individually in early phases enables larger batch sizes (smaller models → lower activation memory → better GPU utilization). Phase 3 trains only the gating network while expert backbones are frozen, dramatically reducing gradient computation. Phase 4 performs short end-to-end fine-tuning to allow expert co-adaptation.

For comparison, an **8×MI300X cluster** completes the same pipeline in ~18–23 hours (6.8–7.6× speedup accounting for ~90% data-parallel scaling efficiency). At current cloud pricing, single-GPU costs ~$293 (Vultr, $1.85/hr × 158 hrs) versus ~$276 for 8-GPU (TensorWave, $12/hr × 23 hrs). **The 8-GPU option is actually cheaper and 7× faster** — making single-GPU training primarily valuable for development, debugging, and architecture iteration rather than final production runs.

---

## The optimal single-GPU stack eliminates distributed complexity entirely

No DeepSpeed. No FSDP. No DDP. For a 200M-parameter model on 192 GB VRAM, distributed frameworks add only overhead. The entire training stack reduces to:

**Core training loop:**
```python
model = YourMoEModel().cuda()
model = torch.compile(model, mode="max-autotune")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
accumulation_steps = 4  # effective batch = batch_size × 4

for epoch in range(num_epochs):
    optimizer.zero_grad(set_to_none=True)
    for step, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets) / accumulation_steps
        loss.backward()
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
```

Key design decisions and their rationale:

**BF16 without GradScaler.** BF16 shares FP32's 8-bit exponent range, making gradient underflow extremely rare. GradScaler is unnecessary — it exists only for FP16's 5-bit exponent limitation. Use `torch.amp.autocast('cuda', dtype=torch.bfloat16)` which automatically keeps master weights in FP32.

**`torch.compile` with `max-autotune`.** AMD's ROCm blog reports **2.3–3.5× speedup** over eager mode for vision and transformer architectures. The `max-autotune` mode benchmarks multiple Triton kernels per operation and selects the fastest. Compilation warm-up takes ~100+ seconds but amortizes over days of training.

**Gradient accumulation replaces data parallelism.** To match an 8-GPU effective batch size of 512, use `batch_size=128` with `accumulation_steps=4`. The MoE load-balancing loss should be computed per micro-batch (not averaged over the full accumulated batch) to maintain correct expert routing statistics.

**`set_to_none=True` for `zero_grad`.** Frees gradient memory between optimizer steps rather than zeroing tensors in-place, reducing peak memory by one gradient copy (~400 MB).

---

## Data pipeline design leverages WebDataset, not FFCV

**FFCV is incompatible with ROCm** — it depends on Numba's CUDA target and CuPy, neither of which support AMD GPUs. WebDataset is the correct choice: GPU-agnostic, sequential-I/O optimized, and supports multi-modal samples natively.

**Storage budget on 5 TB NVMe scratch:**

| Item | Size | Notes |
|------|------|-------|
| Raw IQ data (RFUAV + supplementary) | 1.5 TB | Original dataset |
| Pre-computed spectrograms (float16) | 0.5 TB | STFT output, half-precision |
| Pre-computed HOS + cyclostationary features | 0.1 TB | Much smaller than raw IQ |
| WebDataset .tar shards (all modalities) | 1.6 TB | ~3–5% tar overhead |
| Model checkpoints (10 × 2.4 GB) | 24 GB | Including optimizer states |
| **Total** | **~3.7 TB** | **1.3 TB headroom** |

Pre-compute spectrograms, HOS features, and cyclostationary features **offline** before training. These operations are CPU-intensive (FFTs, cumulant estimation, spectral correlation) and would bottleneck 20 vCPUs during training. An alternative for spectrograms is computing them on-GPU via `torch.stft()` during the forward pass, offloading work from the CPU bottleneck to the MI300X's massive compute headroom.

**DataLoader configuration for 20 vCPU:**
```python
dataset = (
    wds.WebDataset("rfiq-{000000..001499}.tar", shardshuffle=True)
    .shuffle(5000)  # buffer shuffle for randomization
    .decode()
    .to_tuple("iq.npy", "spec.npy", "hos.npy", "label.cls")
    .map(online_augment)  # noise injection, phase offset, time shift
    .batched(batch_size)
)
loader = DataLoader(dataset, batch_size=None, num_workers=12,
                    prefetch_factor=3, pin_memory=True, persistent_workers=True)
```

Use **~1 GB shards** (~1,500 total) for optimal NVMe sequential read performance. Reserve 12 of 20 vCPUs for data workers, leaving 8 for the main process and system overhead. Two-level shuffling (shard-level + 5,000-sample buffer) provides adequate randomization while maintaining sequential I/O patterns. For curriculum learning phases, disable shard shuffle and control shard ordering explicitly.

---

## ROCm tuning flags make a measurable difference

MI300X performance is sensitive to environment configuration in ways that NVIDIA GPUs are not. The following settings should be applied before every training run:

```bash
# Memory management — reduces fragmentation by up to 35%
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

# GEMM performance — forces optimal BLAS library selection
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIPBLAS_WORKSPACE_CONFIG=:4096:2:16:8

# Kernel launch optimization
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=2

# Convolution auto-tuning (critical for EfficientNet)
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3

# FlashAttention Triton backend with auto-tuning
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"

# Single GPU — no distributed overhead
export ROCR_VISIBLE_DEVICES=0
```

**GEMM tuning is uniquely critical on MI300X.** Unlike NVIDIA's cuBLAS, AMD's hipBLASLt heuristic model frequently selects suboptimal algorithms. Enabling `PYTORCH_TUNABLEOP_ENABLED=1` for an initial tuning pass (one epoch with algorithm search) followed by loading the cached results can yield **up to 7.2× throughput improvement** for specific matrix sizes. However, TunableOp has been observed to cause 25+ GB memory leaks on MI300X — set explicit workspace sizes and monitor memory during tuning.

**FlashAttention-2 is fully supported** on MI300X through both Composable Kernel (CK) and Triton backends. The Triton backend with autotune is fastest and supports BF16 and FP8. For the 5M-parameter FT-Transformer expert, PyTorch's built-in `scaled_dot_product_attention` (using AOTriton 0.11b) is the simplest path — it automatically dispatches to an optimized flash attention kernel. FlashAttention-3 and FlashAttention-4 remain NVIDIA-only.

**MIOpen auto-tuning** provides **15–20% performance improvement** for convolutions, directly benefiting the EfficientNet and TCN experts. The first forward pass through each unique tensor shape triggers algorithm search; results are cached in a local database for subsequent runs.

---

## Architecture modifications that matter for single-GPU efficiency

**Start with EfficientNet-B2, not B4.** B2 has **9.1M parameters** versus B4's **19M**, with **4.2× fewer FLOPs** at native resolution. The ImageNet accuracy gap is 2.8% (80.1% vs 82.9%), but for RF spectrogram classification — with fewer classes, simpler textures, and lower input resolution — the practical gap shrinks to **1–2%**. Published drone RF detection work achieves 95–100% accuracy with models as small as 2–5M parameters. B2 saves ~15% of total training time when the spectrogram expert represents ~20% of compute. Upgrade to B3 (12M params) or B4 only if B2 proves insufficient.

**The 200M total parameter count warrants scrutiny.** The literature shows that even complex multi-class drone type + flight mode classification achieves **95%+ accuracy with 10–50M parameters**. The ~141M parameters allocated to gating and shared layers seems disproportionate. Consider:

- Reducing shared layers to bring total parameters to ~80–100M
- Using a simple single-FC-layer gating network (~10K params) with softmax routing instead of a learned deep gating network
- Starting soft MoE (weighted average of all experts) for the first 20% of training, then transitioning to top-2 sparse routing to prevent early expert collapse

**MoE-specific single-GPU considerations:** With top-2 of 4 experts, only 50% of expert parameters activate per sample, providing ~30–40% FLOPs savings versus running all experts. The load-balancing auxiliary loss (α = 0.01, capacity factor 1.25) is essential to prevent expert collapse. Monitor per-expert utilization — if any expert handles <10% of samples, increase the balancing coefficient. Keep gating softmax in FP32 (routing precision is critical per Switch Transformer findings).

---

## Concrete comparison: single-GPU versus 8-GPU pipeline changes

| Configuration | 8×MI300X Pipeline | 1×MI300X Pipeline |
|---------------|-------------------|-------------------|
| Distributed framework | FSDP or DDP | None (plain PyTorch) |
| Effective batch size | 512 (64 × 8 GPUs) | 512 (128 × 4 accum steps) |
| DeepSpeed | Optional ZeRO Stage 1 | Not needed |
| Gradient checkpointing | Optional | Not needed |
| Training time (150M samples) | ~23 hours | ~5.5 days (progressive) |
| Cloud cost | ~$276 (TensorWave) | ~$293 (Vultr) |
| Memory utilization | ~20–40 GB per GPU | ~30–80 GB of 192 GB |
| Data pipeline | Distributed sampler | WebDataset, single-node |
| NCCL/RCCL comms | Required (known MI300X pain point) | None |
| Debugging complexity | High (distributed deadlocks) | Low |
| Recommended for | Production training, tight deadlines | Research iteration, arch search |

**Environment differences:** Remove all NCCL/RCCL environment variables. Remove `torch.distributed.init_process_group()`. Remove `DistributedSampler`. Replace `model = FSDP(model)` or `model = DDP(model)` with `model = torch.compile(model)`. Replace per-GPU batch size with gradient accumulation to match effective batch size.

## Conclusion

The single MI300X's **192 GB VRAM converts a distributed systems problem into a straightforward single-process training job**. The 200M-parameter MoE model uses under 2% of available memory for its parameters and optimizer — an almost absurd ratio that eliminates the need for any memory optimization techniques typically associated with large-model training. The real constraints are training time (~5.5 days with progressive phasing) and CPU-side data preprocessing (pre-compute expensive features offline, use WebDataset for I/O, keep DataLoader workers to 12).

The most impactful optimizations are not memory-related but compute-related: `torch.compile` (2–3.5× speedup), GEMM tuning via TunableOp (up to 7× for specific operations), MIOpen convolution auto-tuning (15–20%), and large batch sizes that push matrix operations into the MI300X's compute-bound regime. Progressive 4-phase training (individual experts → frozen-backbone gating → end-to-end fine-tuning) saves ~15% wall-clock time while producing comparable model quality. And given that 8×MI300X training costs roughly the same but runs 7× faster, the single-GPU setup is best positioned as a research and development platform, with final production training scaled to a multi-GPU cluster for the last full run.