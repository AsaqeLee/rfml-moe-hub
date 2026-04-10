# DroneRFa Dataset

## Overview

| Property | Value |
|----------|-------|
| **Name** | DroneRFa (Drone RF Fingerprinting - Dataset A, Dual-Receiver) |
| **Source** | [SciDB China](https://china.scidb.cn/) -- File ID `c403fc76444e4b9989e4f3ff570f3b3d` |
| **Paper** | JEIT 2025, DOI: 10.11999/JEIT240804 (companion to DroneRFb) |
| **Size** | 574 GB (single compressed archive) |
| **Format** | MATLAB .mat files |
| **Sample Rate** | 80 MSps |
| **Frequency** | 2.4--2.48 GHz ISM band |
| **Classes** | Same drone types as DroneRFb |
| **Task** | Dual-receiver drone identification |
| **Status** | Download in progress |

## Relationship to DroneRFb

DroneRFa and DroneRFb are companion datasets from the same research group. While DroneRFb focuses on cross-individual generalization (training on individuals 1&2, testing on individual 3), DroneRFa provides dual-receiver captures of the same drone types, enabling:

- **Receiver diversity studies**: Train on receiver A, test on receiver B
- **Fusion experiments**: Combine signals from both receivers for improved classification
- **Receiver-invariant feature learning**: Develop features that generalize across different SDR hardware

## File Format

MATLAB .mat files with IQ data, consistent with DroneRFb format:

```python
import h5py

with h5py.File("droneRFa_sample.mat", 'r') as f:
    i_channel = f['I'][0, :]    # float32
    q_channel = f['Q'][0, :]    # float32
    iq = i_channel + 1j * q_channel
```

## Download

```bash
# Single large file download via aria2c
bash datasets/download_scripts/download_droneRFa.sh

# Or manually:
aria2c -x 4 -s 4 --max-tries=0 --retry-wait=30 \
    "https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d&username=YOUR_EMAIL&traceId=YOUR_EMAIL" \
    -o data/droneRFa.rar

# Extract (requires unrar, takes significant time at 574 GB)
cd data && unrar x droneRFa.rar droneRFa/
```

## Storage Requirements

| Stage | Size |
|-------|------|
| Compressed download | ~574 GB |
| Extracted | ~574 GB |
| Spectrograms (estimated) | ~10 GB |
| **Total needed** | **~1.2 TB** (keep compressed during extraction) |

Ensure at least 1.2 TB free before starting download and extraction.

## Planned Experiments

1. **Receiver diversity**: Train models on DroneRFa, test on DroneRFb (and vice versa)
2. **Multi-receiver fusion**: Combine DroneRFa + DroneRFb features for improved accuracy
3. **Receiver-invariant representations**: Learn embeddings that are stable across receivers
4. **Transfer learning**: Fine-tune RFUAV-trained models on DroneRFa

## Citation

```bibtex
@article{droneRFa2025,
  title={Cross-Individual Drone Identification via RF Fingerprinting},
  journal={Journal of Electronics and Information Technology (JEIT)},
  year={2025},
  doi={10.11999/JEIT240804}
}
```
