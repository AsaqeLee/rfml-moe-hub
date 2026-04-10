# RF machine learning for drone detection and edge deployment: a practitioner's technical survey (2023–2026)

**The field of RF-based drone detection via deep learning has reached a pivotal maturity point.** The release of the **RFUAV benchmark** (2025, 37 drone types, 1.3 TB) alongside complex-valued YOLO architectures achieving **93.8% same-model fingerprinting accuracy** signals that production-grade ML pipelines for RF drone detection are now feasible. Simultaneously, RF foundation models have emerged from multiple research groups, with **IQFM achieving 99.67% modulation accuracy from just 1 labeled sample per class** via contrastive self-supervised pretraining on raw IQ. For edge deployment on your RFSoC 4x2 (ZU48DR), the most proven path remains **Brevitas QAT → FINN** for streaming CNN inference at **~8 µs latency and 488k classifications/sec**, while Vitis AI provides a faster development cycle for INT8 models through the DPU soft-core. This survey covers every major model, architecture, dataset, and deployment pathway relevant to building a real-time drone detection pipeline on RFSoC hardware.

---

## Part 1a — Drone presence detection from raw IQ: the current landscape

The dominant approach for detecting drone RF presence has shifted decisively from raw 1D IQ classification toward **2D spectrogram-based CNNs**, which outperform raw IQ by a wide margin at low SNR. Glüge et al. (arXiv:2406.18624, 2024, ZHAW/armasuisse) demonstrated this gap quantitatively: at **−12 dB SNR, spectrograms achieve 84.2% balanced accuracy versus 41.3% for raw IQ** on a 6-drone dataset collected with a USRP B210 at 14 MSps. Their system uses consecutive FFTs on I and Q components separately to produce a 2-channel spectrogram input, with signal windows of ~75 ms (1,048,576 samples) to capture burst repetition patterns. All CNN variants tested achieved ≥85% balanced accuracy above −12 dB, with field tests maintaining >80% using a 20 dBi LHCP directional antenna.

The **RFUAV benchmark** (Shi et al., arXiv:2503.09033, 2025) represents the most ambitious open dataset to date: **37 distinct UAV types** including DJI Mavic 3 Pro, Mini 3/4 Pro, FPV Combo, Avata, and numerous others, totaling approximately **1.3 TB** of raw IQ collected via USRP at 2.4/5.8 GHz ISM bands. Their two-stage pipeline uses **YOLOv5 on STFT spectrograms** for signal detection/localization in the time-frequency plane, followed by classification using ViT, SwinTransformer, ResNet, EfficientNet, MobileNet, DenseNet, or VGG — all benchmarked with full training code. The repository at `github.com/kitoweeknd/RFUAV` includes training configs, dataloaders, YOLOv5 detection, and inference pipelines, with pretrained weights on HuggingFace (`kitofrank/RFUAV`). STFT colormap (CMAP) optimization experiments and SNR adjustment tools for augmentation are included. Their "RF drone fingerprint" is defined as a sequence of frequency-hop bandwidth, hop duration, duty cycle, pattern period, and video transmit bandwidth parameters.

The **FDGAF-CNN** (UESTC, MDPI Drones 8(9):511, 2024) introduces a novel encoding: raw RF → STFT → 1D frequency spectrum → modified Gramian Angular Field transform → 2D image, preserving both time and frequency domain features in a single representation. This achieves **98.72% on DroneRF and 98.67% on DroneRFa** using a custom CNN classifier with LR=0.0001, batch size 32, 70/10/20 split, and 5 epochs.

**FLEDNet** (MDPI Drones 9(243), 2025) applies fuzzy logic edge detection to spectrograms as a preprocessing step before CNN/CRNN classification, yielding **+13.41% accuracy for drone-type identification** over baselines. This was validated on NVIDIA Jetson Orin NX + USRP-2954 for real-time inference on VTI_DroneSET, demonstrating stability across −10 to 20 dB SNR.

For RF-specific data augmentation, recent work shows that applying SpecAugment-style techniques (frequency masking, time masking, random scaling) to RF spectrograms improves accuracy by **+24.7% at −14 dB SNR**, with residual blocks adding +14.1%, for a combined +38.8% gain over baseline CNNs.

### Key datasets for drone detection

| Dataset | Year | Format | Drones | Size | Access |
|---------|------|--------|--------|------|--------|
| DroneRF | 2019 | Raw amplitude (2×40 MHz) | 3 (Parrot Bebop, AR, DJI Phantom 3) | 3.75 GB | Mendeley Data |
| DroneDetect | 2020 | Raw IQ (BladeRF) | 7 drones + 7 controllers | 66 GB | IEEE DataPort |
| CardRF | 2022 | Raw time-series | 17 controllers, 8 manufacturers | 65+ GB | IEEE DataPort |
| DroneRFa | 2024 | Raw IQ | 9 outdoor + 15 indoor types | 100M+ samples/segment | J. Electronics & Info Tech |
| Noisy Drone RF | 2023 | IQ + spectrograms | 6 drones + 4 controllers + interference | 23 GB | Kaggle (sgluege) |
| UAVSig | 2024 | Raw IQ (50 MSps) | 4 identical DJI M100 + controllers | 720 files | UCLA Dataverse |
| **RFUAV** | **2025** | **Raw IQ (USRP)** | **37 UAV types** | **~1.3 TB** | **github.com/kitoweeknd/RFUAV** |

---

## Part 1b — DJI DroneID: protocol internals and ML-augmented demodulation

### The protocol baseline

The definitive reverse engineering of DJI DroneID comes from the NDSS 2023 paper "Drone Security and the Mysterious Case of DJI's DroneID" (Schiller et al., RUB-SysSec), with code at `github.com/RUB-SysSec/DroneSecurity`. The protocol specifics are critical for any ML-augmented pipeline:

The DroneID signal is OFDM-based, embedded within OcuSync, transmitted in 2.4/5.8 GHz ISM bands with dynamic hopping. The signal is **~10 MHz wide (15.36 MHz with guard carriers)**, broadcast every ~600–640 ms. Each frame contains **9 OFDM symbols** (sometimes 8 when the first is zeroed by the scrambler). Symbols 4 and 6 carry **Zadoff-Chu sequences** with roots **600 and 147** respectively, sequence length **601**, used for synchronization and channel estimation. The remaining 7 symbols carry **QPSK-modulated data** across 600 active subcarriers. The FFT is **1024-point** with **15 kHz subcarrier spacing** (LTE-like), using long cyclic prefixes. Data is **unencrypted** (contradicting DJI's earlier claims) and contains drone GPS, operator GPS, serial number, altitude, velocity, and yaw. Forward error correction uses turbo coding with LTE-compatible rate matching.

The conventional receiver pipeline is: energy detection → frame candidate extraction → ZC sequence correlation (threshold ~0.7) → coarse CFO correction via CP autocorrelation → fine CFO correction via ZC phase → OFDM symbol extraction → QPSK demodulation → turbo decoding → CRC check. Proto17's community decoder (`github.com/proto17/dji_droneid`, 473 stars) implements this in MATLAB/Octave + C++ for DJI Mini 2 at **30.72 MSps** with Ettus B205-mini.

**Critical note for RFSoC implementors:** The required sample rates are 15.36 or 30.72 MSps, which must divide evenly by the 15 kHz LTE subcarrier spacing into power-of-two FFT sizes. The ZU48DR's 14-bit ADCs running at these rates provide more than sufficient dynamic range.

### TranSIC-Net: the first ML model validated on DroneID

**TranSIC-Net** ("An End-to-End Transformer Network for OFDM Symbol Demodulation with Validation on DroneID Signals," MDPI Sensors 25(20):6488, 2025) is the most directly relevant ML model found. It **unifies channel estimation and symbol detection** in a single Transformer architecture, eliminating the conventional LS/MMSE equalization pipeline. The self-attention mechanism captures inter-subcarrier correlations, using the ZC pilot symbols at positions 4 and 6 as implicit channel references. The input is frequency-domain OFDM symbols after FFT, and the output is symbol probabilities directly. It was validated specifically on DroneID's 9-symbol frame with ZC roots 600/147, 600 active subcarriers, and QPSK modulation. It compares favorably against LS/MMSE + equalization + demodulation baselines and SCBiGNet. Code availability is not confirmed — this is a key gap for implementors.

### ML components applicable to the DroneID pipeline

For **timing synchronization**, Qing et al. (arXiv:2209.06451) present a lightweight 1D CNN that transforms OFDM timing sync into a classification task. A single CNN layer + FC layer, with filter size matched to CP length, achieves lower complexity than compressed sensing approaches. The cascaded variant (PA-Net + RS-Net, 2023) uses two 1D CNN subnetworks for coarse-then-fine timing estimation from 2M×1 sample vectors, suitable for multi-path scenarios.

For **ZC sequence detection**, Mohammadi et al. (arXiv:2110.02738) propose NN-based blind coherent combining across antennas/time instances for ZC preamble detection — the NN learns optimal combining weights without channel knowledge, outperforming non-coherent (power summing) approaches. The aggregate ZC DNN from ShanghaiTech/UBC (IEEE 2021, 10.1109/JIOT.2021.3066797) decodes aggregate preambles containing two ZC sequences from different roots at half power — **directly analogous to DroneID's dual ZC structure** (roots 600 and 147). The DNN classifier takes correlation outputs as input and outperforms threshold-based detection for misdetection and false alarm probability.

For **OFDM channel estimation**, the foundational DNN approach (Ye et al., IEEE WCL 2018) uses a 5-layer FC network with ReLU on interleaved real/imaginary OFDM samples, matching MMSE performance while being robust to non-linear distortions. LSTM-based variants (2022-2023) outperform both LS and MMSE under Rayleigh fading.

### DJI O4 encryption challenge

Newer DJI drones (Mini 5 and onwards) encrypt DroneID via the O4 protocol. The `alphafox02/antsdr_dji_droneid` firmware handles this through **hash-based session identification** — the OFDM physical layer structure (ZC sequences, 601 subcarriers, timing patterns) remains an unforgeable hardware fingerprint even when payload is encrypted. ML-based detection of encrypted DroneID frames via RF fingerprinting rather than payload decoding is an emerging research direction.

---

## Part 1c — Drone type and model fingerprinting via RF

**SignalFormer** (MDPI Sensors 23(22):9098, 2023) introduces a hybrid CNN-Transformer that processes spectrograms with phase information. A **CNN-based C-tokenizer** generates time-frequency tokens enriched with local context, which are then processed by a **T/F-encoder** using Gated Self-Attention (GSA) for global time and frequency correlations. Phase inclusion substantially impacts performance — a key finding for anyone discarding phase in their preprocessing pipeline.

**Deep Complex-Valued CNN (DC-CNN)** (Zhang et al., MDPI Drones 6(12):374, 2022) processes I and Q channels jointly in the complex domain using complex convolutions: W*h = (A*x − B*y) + i(B*x + A*y). On DroneRF this achieves **99.5% accuracy (4-class)** and 74.1% (8-class), outperforming real-valued networks by preserving phase-amplitude relationships natively. This is the architecture to consider if you want to avoid the STFT preprocessing step entirely and work directly on raw IQ.

A **Lightweight Hybrid CNN-Transformer** (IEEE Xplore 11105417, 2025) combines Partial Convolution → PointWise Convolution → Squeeze-and-Excitation (SE) modules for local features with Super Token Transformer blocks for global dependencies, achieving **96.89% average accuracy across −5 to 20 dB SNR** with limited training data. The partial convolution strategy specifically reduces computation for edge deployment.

The **VMD-Transformer** (Han et al., IET Communications, 2025) uses Variational Mode Decomposition to separate phase and amplitude of ADS-B preambles into Intrinsic Mode Function components at different frequencies, then feeds these into a Transformer encoder with multi-head attention — a noteworthy technique for separating modulation effects from hardware-specific fingerprints.

For the CardRF dataset specifically, wavelet scattering transform + SqueezeNet achieves **98.9% accuracy at 10 dB SNR** for distinguishing among 17 drone controllers from 8 manufacturers, with **0.37 ms inference time per signal** — an important edge-deployment benchmark.

---

## Part 1d — Pilot identification and specific emitter identification for drones

**CV-YOLO** (Zhao & Cabric, UCLA CORES Lab, 2025) is the standout architecture here: a **complex-valued YOLOv5** with complex convolution, complex max pooling (on magnitudes), complex batch normalization (whitening), and magnitude conversion at the output layer. Operating on raw complex IQ from the UAVSig dataset (50 MHz bandwidth, 50 MSps, USRP B205mini), it achieves:

- **93.8% accuracy** for single-drone fingerprinting (vs. 81.6% baseline — **+12.2%**)
- **67.7% accuracy** for two-drone generalization (vs. 37.0% baseline — **+30.7%**)
- **86.5% accuracy** for cross-temporal evaluation (May→July data)

The backpropagation uses separate partial derivatives for real and imaginary parts. This architecture simultaneously detects and fingerprints FHSS drone signals in wideband spectrograms and can discriminate **identical DJI M100 drones** — true individual device identification, not just model classification.

**CrossRF** (arXiv:2505.18200, 2025) tackles the critical practical challenge that RF fingerprints change across frequency channels. Using Adversarial Discriminative Domain Adaptation (ADDA), it achieves **99.03% accuracy** when adapting from Channel 3 to Channel 4 on UAVSig, versus 26.39% without adaptation. This is essential for any real deployment where you cannot guarantee training and test data come from the same frequency channel.

**Multi-Domain Supervised Contrastive Learning for UAV Open-Set Recognition** (arXiv:2508.12689, 2025) uses ResNet + TransformerEncoder with texture + time-frequency position feature fusion, trained with supervised contrastive loss + IG-OpenMax for open-set classification (20 known + 5 unknown UAV classes). Collected at **100 MS/s**, processed on Tesla V100 — relevant for scenarios where unknown drone types must be flagged rather than misclassified.

---

## Part 2a — Automatic modulation classification has moved beyond ResNets

The SOTA on **RadioML 2018.01A** (24 modulations, 26 SNR levels, ~2.55M samples) as of 2025 is **98.9% peak accuracy at 22 dB, 63.7% average accuracy across all SNRs**, achieved by SE-ResNet with dilated convolutions + statistics pooling + squeeze-and-excitation blocks (Harper et al., Electronics 2023). Input is raw IQ [2, 1024]. Code: `github.com/caharper/Automatic-Modulation-Classification-with-Deep-Neural-Networks`.

**Transformer-based AMC** has matured rapidly. **IQFormer** (Shao et al., IEEE TCCN 11(3):1623-1634, 2025) fuses raw IQ signals with time-frequency distribution matrices through staged Transformer blocks with GRU for sequential patterns, achieving SOTA on RML2016.10a/b and HisarMod2019.1 (`github.com/WestdoorSad/IQFormer`). **AMC-Transformer** (2024) tokenizes raw IQ into fixed-length patches with learnable positional embeddings and pure multi-head self-attention — no convolutions, no handcrafted features — reaching 98.8% at ≥10 dB SNR, outperforming CNN by 4.44% and ResNet by 1.96%. **Meta-Transformer** (Jang et al., IEEE Access 2024) adds few-shot learning to identify unseen modulations (`github.com/cheeseBG/meta-transformer-amc`).

**Mamba/SSM architectures** have entered the RF domain. **MAMC** (Zhang et al., IEEE Comm. Letters, arXiv:2405.11263, 2024) uses the Selective SSM backbone with a denoising unit based on Deep Residual Shrinkage Networks. The key advantage is **O(n) complexity** versus Transformer's O(n²) for long IQ sequences, with d_model standard, d_state=16, d_conv=4, expand=2. It demonstrates optimal accuracy-efficiency tradeoff for extended signal lengths with low GPU occupancy. Code: `github.com/ZhangYezhuo/MAMC` (52 stars).

**TENN from BrainChip** (arXiv:2501.13230, 2025) is a state-space model with only **~276K parameters and ~3.7M MACs** — **100× more efficient than CLDNN** — while matching or exceeding CLDNN accuracy on RadioML 2018.01A at mid-to-high SNR. Designed for neuromorphic/event-driven edge deployment on drones and CubeSats, this is the model to evaluate first if you need minimal FPGA resources.

For **self-supervised and contrastive AMC**: GAF-MAE (Shi et al., IEEE TCCN 10(1):94-106, 2024) converts IQ to Gramian Angular Field images and applies masked autoencoders; MCLHN (Xiao et al., IEEE TWC 23(10):14304-14319, 2024) uses masked contrastive learning with hard negative mining; SSCL-AMC (ICASSP 2025) combines frequency-aware Transformer + LSTM with adversarial augmentation and ensemble voting.

**Complex-valued networks** for AMC include LDCVNN (2025, dual-branch with phase information + complex-scaling-equivariant pathways, complex depthwise separable convolutions), MCCSAN (2025, multiscale complex convolutions + spatiotemporal attention, trained with cross-entropy + center loss), and CC-MSNet (Scientific Reports 2024, multi-stream spatial-temporal with complex convolution achieving 71.12% on RML2016.04c).

---

## Part 2b — Specific emitter identification beyond drones

**Federated LoRa-RFFI** (Peng et al., IEEE TIFS 2024) achieves **95% accuracy across 60 LoRa transmitters and 6 SDR receivers** using unsupervised contrastive pretraining in a federated setting — improving from 63% without pretraining. CFO compensation is critical for stability. **DeepCRF** (Kong et al., IEEE TIFS 20:264-278, 2025) extracts micro-CSI features from WiFi's 52 OFDM subcarriers across 879,943 measurements in 4 environments over multiple months, addressing the channel resilience problem that plagues most RFFI systems.

**TF-CSS** (MDPI Electronics 2025) uses an Asymmetric Masked Auto-Encoder with complex-valued neural networks for few-shot SEI on 30 LoRa classes, with code at `github.com/jackcomnet/TF-CSS`. **ProSSL** (ScienceDirect 2025) achieves **96.48% with 90% labels, 59.88% with only 5% labels** through progressive semi-supervised learning with iterative clustering + contrastive learning.

Key domain tricks across SEI research: **CFO compensation** is essential for LoRa; **differential constellation trace figures (DCTF)** provide synchronization-free 2D representations; **bispectrum features** capture higher-order statistics as device fingerprints; **multi-receiver processing** frameworks mitigate receiver-specific distortions; and **phase-rotation augmentation** improves robustness.

---

## Part 2c — YOLO-on-spectrograms and wideband signal detection

**Spec-YOLO** (IEEE Signal Processing Letters 2025) modifies YOLOv8 with a Selective Feature Fusion Module, Enhanced BiFPN, and Light Convolution Module, achieving **mAP₅₀=92.1%, mAP₉₅=85.5%** for 5G NR and LTE signal detection with **5.9M parameters**. **RF-YOLO** (Telecommunication Systems, Springer, 2025) targets drone controller RF signals specifically, achieving **mAP=0.9213, precision=0.9800, recall=0.9750** — outperforming YOLOv3/v5/v8 and RT-DETR. Idaho National Laboratory's YOLO for RF (DHS-funded, 2024) uses YOLOv7 with bounding-box labeled spectrogram images.

For **semantic segmentation**: SRNet (IEEE WCL 2025) uses a deep encoder-decoder for 5G NR/LTE identification; the MathWorks/NI pipeline uses ResNet50-based segmentation with FFT length 4096, 256×256 RGB images at 61.44 MHz sampling, achieving ~95% mean accuracy and deploying on USRP + Jetson. Qoherent's open-source PyTorch/Lightning implementation (`github.com/qoherent/spectrogram-segmentation`) provides a practical starting point for 5G NR/LTE segmentation.

---

## Part 2d — RF foundation models are now real, not theoretical

The RF foundation model landscape has exploded since late 2024. The most significant efforts come from a single prolific group led by Hatem Abou-Zeid:

**IQFM** (arXiv:2506.06718, June 2025) is the **first wireless foundation model operating directly on raw MIMO IQ streams**. A shared encoder (CNN/ResNet-style) processes real-valued tensors of shape [antennas × 2(I/Q) × T] with contrastive self-supervised pretraining using task-aware augmentations (cyclic time shifting as core, spatial/temporal as task-specific). With **just 1 labeled sample per class**, it achieves **99.67% modulation accuracy** (vs. 14.27% supervised baseline — a **7× improvement**) and 65.45% AoA accuracy (vs. 0.45% — **145× improvement**). LoRA fine-tuning on out-of-distribution tasks yields 94.15% beam prediction, 96.05% RF fingerprinting. Code and weights: `github.com/haoruizhao/IQFM`.

**WavesFM** (IEEE OJCOMS 6:6792-6807, 2025; arXiv:2504.14100) uses a **ViT backbone** with task-specific MLP heads, pretrained via masked wireless modeling (self-supervised MAE). It handles both spectrograms and IQ-as-OFDM-resource-grids, sharing **80% of parameters** across 5G positioning, MIMO channel estimation, human activity sensing, and RF classification. Pretraining accelerates convergence by up to **5×**. Weights and code: `wavesfm.waveslab.ai`.

**WirelessJEPA** (arXiv:2601.20190, January 2026) applies Joint Embedding Predictive Architecture to wireless, using 2D antenna-time representations with block masking. The non-contrastive approach predicts latent representations of masked regions — temporal masks favor waveform structure, antenna masks emphasize spatial cues. Strong OOD generalization surpasses contrastive baselines.

**SpectrumFM** (arXiv:2505.06256, May 2025) uses a **CNN + Multi-Head Self-Attention hybrid** pretrained with masked reconstruction + next-slot signal prediction on RML2018.01A and TechRec datasets. LoRA fine-tuning (~2% of parameters) improves AMC by **up to 12.1%**, spectrum sensing AUC to 0.97 at −4 dB SNR, and anomaly detection AUC by >10%.

**EMind** (arXiv:2508.18785, August 2025) is a Transformer encoder-decoder with length-adaptive multi-signal packing, per-sample masking, and sampling-rate tokens for heterogeneous data — pretrained via MAE at 75% masking ratio across 7 downstream tasks including AMC, radar waveform classification, RF fingerprinting, blind source separation, and signal denoising.

**LWM-Spectro** (arXiv:2601.08780, January 2026) uses a **Transformer with Mixture-of-Experts (MoE)** pretrained via joint masked spectrogram modeling + contrastive learning on 9.2 million WiFi/LTE/5G samples from DeepMIMO across 20 city scenarios. The MoE-Router reaches **89.8% F1 with only 100 samples per class**, within 98% of saturated performance — deep CNN baselines require **30× more data**.

For generative RF models: **RF-Diffusion** (Tsinghua, MobiCom 2024) uses a hierarchical Diffusion Transformer with time-frequency diffusion theory and complex-valued operators; **ReFormer** (Nokia, arXiv:2501.00282, 2025) combines VQ-VAE + decoder-only Transformer for autoregressive RF signal generation, useful for data augmentation.

### Self-supervised RF learning specifically

**RIS-MAE** (arXiv:2508.00274, August 2025) applies masked autoencoders directly to raw IQ sequences (no time-frequency conversion), with random masking + reconstruction capturing amplitude and phase features. It outperforms existing methods in few-shot and cross-domain AMC with minimal fine-tuning. **Contrastive learning for RF fingerprinting** (Chen et al., arXiv:2403.04036, Oregon State/NSF) treats signals from the same transmission as positive pairs, achieving **10.8–27.8% accuracy improvement** over baselines under domain shift. **FLA-CL** (Electronics 2025) adds feature-level augmentation in high-dimensional space, matching SOTA with only 10% labeled data.

---

## Part 2e — RF anomaly and jamming detection

Digital twin-based approaches dominate recent work: XGBoost on DT-derived features achieves **0.99 accuracy** for jamming and signal drift detection (Scientific Reports 2025). The TU Dresden/Fettweis group (Globecom Workshops 2023) compares expected RSS from digital twins with real measurements for Industry 4.0 jammer detection. For **GPS spoofing**, CTDNN-Spoof (Scientific Reports, Feb 2025) provides a TinyML architecture (64-32-4 sequential NN) deployable on STM32 microcontrollers; PCA-CNN-LSTM (2023) achieves **99.49% accuracy** on UAV spoofing datasets. Dual autoencoders for spectrum anomaly detection (IEEE/NSF 2024) use LSTM autoencoder for frequency domain + dense autoencoder for time domain on real Wi-Fi/LTE IQ data.

---

## Part 3 — Edge deployment on RFSoC: proven pathways and quantization realities

### The FINN + Brevitas path (most proven for RF)

**"RadioML Meets FINN"** (Jentzsch et al., IEEE Micro 42(6), 2022, AMD-Xilinx Research Labs) is the flagship demonstration: Brevitas QAT → FINN compiler → FPGA bitstream for RadioML 2018.01A (24 modulations). The VGG-like CNN with various reduced-precision configurations (binary, ternary, low-bit) achieves **3.5× throughput** over alternative FPGA approaches. The ITU AI/ML 5G Grand Challenge "Lightning-Fast Modulation Classification" (2021) validated this pipeline competitively, with the winning BacalhauNET team achieving **62× better inference cost** than baseline — all top 3 teams used Brevitas + FINN.

Optimal bit-per-layer analysis (Göez et al., Algorithms 2022) shows that FINN-targeted models can achieve **75.8% model size reduction with only 0.06% accuracy loss** through mixed-precision quantization (2-8 bits across layers), exported via QONNX.

### RFSoC-specific deployments

**Tridgell et al. (IEEE IPDPSW 2020)** deployed a VGG10-based 1D-CNN on the **Xilinx ZCU111 RFSoC** with ternary weights ({-1, 0, 1}) and multi-bit activations using INCRA (incrementally increasing activation precision). Results:

- **Latency: ~8 µs** per classification
- **Throughput: 488k classifications/sec**, accepting 2× I/Q samples per clock at 500 MHz
- **Accuracy: TW-INCRA-128** recovers +4.3% accuracy over baseline at same hardware utilization
- Convolutions use **zero DSP slices** (ternary weights → XNOR operations); DSPs used only for batch normalization in dense layers
- Full loopback validated: Modulator → DAC → coax → ADC → CNN → DMA → CPU, zero test error on 4 modulation types

Compared to NVIDIA RTX 2080 Ti (~30k classifications/sec at batch=256), this represents a **~16× throughput advantage** at the critical batch_size=1 latency point relevant for real-time RF.

**MacLellan et al. (IEEE OJCAS 6:38-49, January 2025, University of Strathclyde/StrathSDR)** deployed on the **AMD RFSoC2x2** (XCZU28DR, Gen1 RFSoC — the same generation as your ZU48DR). Key details:

- **Model:** CNN for 8 modulation schemes (~260K parameters), streaming architecture
- **QAT with Brevitas** at 16w16a, 8w8a, and **4w4a** precisions
- **Sampling rate:** 128 MHz at ADC with decimation rate of 8
- The **16w16a** model shows ~4% accuracy reduction vs. floating-point baseline
- Custom **"DeepRFSoC" dataset** incorporating the RFSoC DAC→channel→ADC in the generation loop
- Built with MathWorks HDL Coder + Vivado
- **PYNQ framework** with Jupyter-based interactive app for live classification
- On unseen RadioML data at SNR >4 dB: 65% average accuracy

### hls4ml: sub-microsecond but RF examples still emerging

The hls4ml platform (v1.3.0, arXiv:2512.01463, December 2025) supports Keras, PyTorch, ONNX models with QKeras, HGQ, Brevitas, and QONNX frontends. Fixed-point arithmetic uses arbitrary bit-widths (`ap_fixed<total, integer>`). Two deployment modes: **io_parallel** (maximum throughput, minimum latency) and **io_stream** (lower resources). Achieved latencies: **85 ns for small MLPs on Alveo U50**, **~5 µs for CNNs**, with >1.4M inferences/sec in multi-accelerator configurations. On RFSoC for quantum control (QICK platform): **32 ns latency** multi-layer networks with >96% fidelity. The QKeras→hls4ml workflow automatically parses quantization parameters to generate optimized HLS, enabling mixed-precision per-layer optimization guided by Hessian-aware quantization (HAWQ).

No published RF-specific hls4ml case study exists yet — the primary user community is high-energy physics. However, the design philosophy (sub-µs latency, on-chip weight storage, streaming I/O) maps directly to your RF inference requirements. For a DroneID detection pipeline, an hls4ml-compiled ZC correlator enhancement network or low-latency signal classifier would sit between the ADC data path and the PS-side demodulation stack.

### Vitis AI: faster development, higher latency

The official **"RF Modulation Recognition with Vitis AI"** tutorial (Vitis-AI-Tutorials branch 2.5) deploys a CNN modulation classifier on the DPUCZ B4096 configuration on ZCU102/ZCU104. Vitis AI quantizer converts FP32 → INT8 automatically. A community project by Matjaz Zibert integrates GNU Radio via a custom OOT module (`gr-fpga-ai`) with the DPU on ZCU104 + RTL-SDR for live 2m band classification. A YOLO-based spectral analysis project (`github.com/Devjyoti-D/ML_for_Spectral_Analysis_on_SoC`) uses TorchSig + 8-bit quantization/pruning via Vitis AI compiled for the DPUCVDX8H.

**Important limitation for your use case:** The Vitis AI DPU is a systolic-array architecture optimized for INT8 batch CNN inference. It introduces ms-range latency (not µs), processes one sample per core at a time, and is **not designed for ultra-low-latency streaming** — use dedicated HLS/FINN IP for the latency-critical path and reserve the DPU for higher-level classification tasks.

### Quantization accuracy impact across precisions

| Precision | Typical accuracy retention | Resource impact | Best framework |
|-----------|---------------------------|-----------------|----------------|
| FP32 (baseline) | 100% | Maximum | Training only |
| 16-bit fixed | ~96% of baseline | Moderate reduction | Brevitas/HDL Coder |
| **INT8** | **~95-99% of baseline** | **Major reduction** | **Vitis AI / Brevitas** |
| INT4 | ~85-95% of baseline | Dramatic reduction | Brevitas + FINN |
| Ternary (2-bit) | ~80-92% of baseline | No DSP (XNOR) | Brevitas + FINN |
| Binary (1-bit) | ~72-80% of baseline | LUT-only | FINN |

**Key finding:** For RadioML-class tasks, **INT8 QAT fully recovers baseline accuracy**. Below INT6, significant degradation occurs. The sweet spot for your ZU48DR is likely **8-bit weights / 8-bit activations** via Brevitas → FINN for the detection CNN, with INT8 DPU via Vitis AI for the classification network.

### Spiking neural networks on FPGA for AMC

**SAOCDS** (arXiv:2601.02613, 2025) deploys streaming spiking neural networks for RadioML 2016 AMC on FPGA, achieving **23.5 MS/s throughput** — **2× higher than FINN baseline** at 41.3% of dynamic power. LUT utilization increases only 11.1% versus baseline. The sparsity-aware output-channel dataflow exploits the inherent sparsity of spike-based computation.

---

## Part 4 — Implementation specifics: what you need to build each model

### Recommended architecture for your ZU48DR DroneID pipeline

Based on this survey, the optimal architecture for your RFSoC 4x2 drone detection pipeline decomposes into three inference stages, each targeting different hardware:

**Stage 1 — Signal detection (PL fabric, hls4ml or HLS):** A lightweight 1D CNN or matched-filter-plus-NN for ZC sequence detection and frame synchronization, compiled via hls4ml in io_stream mode. Target **<1 µs latency**. Input: streaming IQ at 30.72 MSps. The aggregate ZC DNN architecture (dual ZC roots) is directly applicable. QKeras at 8-bit, ~10K parameters.

**Stage 2 — OFDM demodulation (PL fabric, FINN):** A TranSIC-Net-inspired Transformer or lighter CNN for joint channel estimation + symbol detection on extracted DroneID frames. Brevitas QAT at 4-8 bits → FINN streaming dataflow. Target **~8 µs per frame**. Input: 9×600 complex subcarrier values after FFT (which runs as standard HLS IP).

**Stage 3 — Drone classification/fingerprinting (DPU via Vitis AI):** A ViT, SwinTransformer, or ResNet from the RFUAV benchmark for drone type classification on STFT spectrograms. INT8 via Vitis AI. Latency requirement relaxed to ~10 ms. The RFUAV pretrained weights provide the fastest path to deployment.

### Cross-cutting implementation details

**Input preprocessing consensus across papers:** STFT with FFT size 1024-4096, hop length 256-512, Hanning window, log-power normalization. For raw IQ: normalize to unit variance, shape [2, 1024] (I/Q channels × samples). For spectrograms: 256×256 or 128×128 RGB/grayscale images. Sample rates range from 14-128 MSps depending on band coverage requirements.

**Training consensus:** Adam optimizer dominates (LR=0.001 typical, 0.0001 for fine-tuning). Cross-entropy loss for classification; YOLO loss for detection; contrastive (NT-Xent) for self-supervised pretraining. Batch size 32-64. PyTorch is the framework of choice for 2024-2026 work. RF-specific augmentations (frequency masking, time masking, Gaussian noise injection, phase rotation) are essential for robustness.

**Complex-valued networks** consistently outperform real-valued counterparts by **+5-12%** on fingerprinting tasks. Implement as paired real tensors with the complex multiplication rule applied in convolution kernels: (A*x − B*y) + i(B*x + A*y). For FPGA deployment, this doubles the real-valued multiplications but preserves phase information that spectrograms discard.

---

## Key repositories and tools for immediate use

| Repository | Purpose | Notes |
|-----------|---------|-------|
| `github.com/kitoweeknd/RFUAV` | 37-drone benchmark + training code | YOLOv5 + ViT/Swin/ResNet, weights on HuggingFace |
| `github.com/proto17/dji_droneid` | DroneID decoder (MATLAB/C++) | 473 stars, community standard |
| `github.com/RUB-SysSec/DroneSecurity` | DroneID receiver (Python) + samples | NDSS 2023 reference implementation |
| `github.com/alphafox02/antsdr_dji_droneid` | AntSDR E200 DroneID + ZMQ/TAK | Production-ready firmware |
| `github.com/haoruizhao/IQFM` | RF foundation model (raw IQ) | Contrastive SSL, LoRA fine-tuning |
| `wavesfm.waveslab.ai` | WavesFM foundation model | ViT backbone, weights available |
| `github.com/ZhangYezhuo/MAMC` | Mamba for AMC | 52 stars, RadioML benchmark |
| `github.com/brysef/rfml` | DARPA RFMLS library (PyTorch) | Adversarial ML, signal classification |
| `github.com/neu-spiral/RFMLS-NEU` | WiFi/ADS-B RF fingerprinting | DARPA RFMLS program output |
| `github.com/fastmachinelearning/hls4ml` | ML→HLS→FPGA, v1.3.0 | Sub-µs latency, QKeras/Brevitas support |
| `github.com/Xilinx/finn` | QNN dataflow compiler | RadioML examples included |
| `github.com/Xilinx/brevitas` | QAT library, v0.12.1 | QONNX export → FINN/hls4ml |
| `github.com/qoherent/spectrogram-segmentation` | 5G/LTE spectrogram segmentation | PyTorch/Lightning |
| `github.com/jackcomnet/TF-CSS` | Few-shot SEI (LoRa) | AMAE + complex-valued NN |
| `github.com/caharper/Automatic-Modulation-Classification-with-Deep-Neural-Networks` | RadioML 2018 SOTA ablation | SE-ResNet, 63.7% avg accuracy |

---

## Conclusion: where the gaps remain and what to build first

Three observations stand out from this survey that should inform your implementation priorities. First, **no public ML-augmented DroneID decoder exists on FPGA** — TranSIC-Net is validated on DroneID data but has no released code, and all existing FPGA DroneID implementations (AntSDR E200, proto17) use conventional DSP. An hls4ml-compiled ZC detection network feeding into a FINN-deployed channel estimation + demodulation network on your ZU48DR would be genuinely novel. Second, the **RFUAV dataset** provides the most comprehensive drone fingerprinting benchmark available, and its two-stage YOLOv5→classifier architecture maps naturally onto a PL-side signal detector + DPU-side classifier split on RFSoC. Third, **RF foundation models** (IQFM, WavesFM) are mature enough to serve as pretrained encoders for your downstream drone detection task — especially valuable when you have limited labeled data from your specific deployment environment. LoRA fine-tuning adds only ~2% of parameters, making frozen-encoder + lightweight-head deployment on the DPU practical.

The most impactful near-term build would be: RFUAV-trained YOLOv5 signal detector (Brevitas 8-bit → FINN, PL fabric) → proto17-based OFDM demodulation (HLS IP) → TranSIC-Net-inspired channel estimation (hls4ml, PL fabric) → IQFM-pretrained fingerprinting classifier (Vitis AI INT8, DPU) — all running on your ZU48DR with ADC input at 30.72 MSps and <10 ms end-to-end pipeline latency. The field has finally produced both the models and the toolchains to make this real.