# Expert Specialization Analysis

## Expert Hierarchy (by accuracy)

```
Spectrogram (100%) > IQ Statistical (98.8%) > Baseline (97.5%) > Cyclo (95.6%) > HOS (93.8%)
```

## Expert Dominance Map

| Signal Type | Dominant Expert | Runner-Up | Weakest | Difficulty |
|-------------|----------------|-----------|---------|------------|
| FM Broadcast | All tied (100%) | — | — | Easy |
| NOAA Weather | All tied (100%) | — | — | Easy |
| Noise | All tied (100%) | — | — | Easy |
| ISM Sensors | 4 tied (100%) | — | HOS (90%) | Easy |
| APRS | 3 tied (100%) | HOS (95%) | Cyclo (85%) | Medium |
| Pager | Spec/Cyclo (100%) | IQ (95%) | Baseline/HOS (90%) | Medium |
| FRS/GMRS | Spectrogram (100%) | IQ (95%) | HOS (75%) | **Hard** |

## Why Each Expert Succeeds or Fails

### Spectrogram Expert (100% - Perfect)
**Strengths**: Captures time-frequency structure that uniquely identifies each signal type. FM has characteristic wideband energy spread, APRS shows discrete packet bursts in time, FRS/GMRS has narrow-band burst patterns distinguishable from ISM in the time-frequency plane.

**Why it dominates FRS/GMRS**: The spectrogram reveals temporal burst patterns and spectral occupancy width simultaneously — FRS uses narrower channels with voice-like temporal structure, while ISM has shorter, more regular bursts. Other experts see only one dimension.

### IQ Statistical Expert (98.8%)
**Strengths**: Kurtosis, skewness, and autocorrelation capture amplitude distribution shapes. FM has near-constant envelope (low kurtosis), while bursty signals have high kurtosis.

**Weakness**: 2 misclassifications (1 FRS→pager, 1 pager→APRS). The amplitude statistics of bursty signals with similar duty cycles overlap.

### Baseline RTL-ML (97.5%)
**Strengths**: Power and FFT features efficiently separate signals with different spectral footprints. Bandwidth ratio alone separates FM (wide) from narrow-band signals.

**Weakness**: 4 misclassifications (2 FRS→ISM, 2 pager→APRS). Without time-frequency or higher-order features, bursty signals at different frequencies can look identical in aggregate statistics.

### Cyclostationary Expert (95.6%)
**Strengths**: Detects periodic structure in FM carriers, ISM sensor repetition rates, and pager timing. Perfect on FM, ISM, NOAA, noise, pager.

**Weakness**: APRS (85%) and FRS/GMRS (80%). These signals are sporadic — APRS transmissions are irregular, FRS is human-initiated. Without strong periodicity, the SCF features degrade to noise-like patterns.

### HOS Expert (93.8% - Weakest)
**Strengths**: Cumulants distinguish modulation types (FM vs AM vs digital). Perfect on FM, NOAA, noise.

**Weakness**: FRS/GMRS (75%), ISM (90%), APRS (95%), pager (90%). Higher-order statistics require large sample counts for stable estimation. With 512K samples per capture, 6th-order cumulants have high variance. Signals with similar modulation schemes (GFSK variants in FRS/ISM/pager) produce overlapping cumulant profiles.

## Expert Complementarity

The disagreement analysis reveals a clear **trust hierarchy**:

```
Spectrogram (always right when disagreeing)
    └── IQ Statistical (right 75-90% vs others)
         └── Baseline (right 67-71% vs HOS/cyclo)
              └── Cyclostationary (right 57% vs HOS)
                   └── HOS (least reliable in disagreements)
```

This hierarchy suggests the optimal routing strategy is **"spectrogram first, escalate to IQ when uncertain, ignore HOS when it disagrees with others"**.

## Feature Importance by Modality

### Baseline (17 Features)

| Rank | Feature | Importance | What It Captures |
|------|---------|------------|-----------------|
| 1 | power_max | 0.155 | Peak signal strength (FM >> noise) |
| 2 | phase_diff_std | 0.139 | Modulation rate (FM fast, noise random) |
| 3 | q_std | 0.103 | Quadrature spread (modulation depth) |
| 4 | power_mean | 0.097 | Average signal level |
| 5 | fft_mean | 0.073 | Overall spectral energy |
| 6 | fft_std | 0.072 | Spectral shape variation |
| 7 | power_std | 0.069 | Amplitude stability (continuous vs bursty) |
| 8 | bandwidth_ratio | 0.057 | Spectral occupancy width |
| 9 | fft_max | 0.050 | Peak spectral component |
| 10 | i_std | 0.044 | In-phase spread |

The top 3 features (power_max, phase_diff_std, q_std) account for 39.7% of total importance.

### IQ Statistical (37 Features)
Key discriminators: amplitude kurtosis (bursty vs continuous), crest factor (FM constant envelope vs pager spikes), zero-crossing rate (modulation bandwidth proxy), autocorrelation lag-1 and lag-100 (repetition structure).

### HOS Cumulants (20 Features)
Key discriminators: normalized C42 (kurtosis-like, separates Gaussian noise from modulated signals), C40/C42 ratio (distinguishes FM from digital modulations), phase of C40 (modulation type indicator).

### Cyclostationary (64 Features)
Key discriminators: SCF peak values at low cycle frequencies (carrier detection), SCF mean at mid-range frequencies (symbol rate detection). High redundancy in the 64-dim vector — effective dimensionality is ~10-15.

### Spectrogram (37 Features)
Key discriminators: spectral bandwidth mean/std (FM wideband vs narrowband), band energy ratios (frequency occupancy pattern), spectral flatness (noise-like vs tonal), temporal envelope kurtosis (continuous vs bursty). The combination of spectral and temporal statistics is what makes spectrogram features dominate — they encode both frequency structure AND timing patterns simultaneously.

*Full per-modality feature importance rankings are printed by `rfml_comparison.py` during execution.*
