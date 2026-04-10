#!/usr/bin/env python3
"""
Create WebDataset shards from preprocessed HDF5 data.

Combines all processed datasets into WebDataset .tar shards for efficient
sequential I/O during training. Each shard is ~1 GB.

Output: data/shards/{split}/rfiq-{000000..NNNNNN}.tar
Each sample in a shard contains:
  - {key}.iq.npy: IQ data (2, N)
  - {key}.spec.npy: precomputed spectrogram (3, 512, 512) [if available]
  - {key}.hos.npy: HOS features (20,)
  - {key}.cyclo.npy: cyclostationary features (1, 512)
  - {key}.label.json: {"binary": int, "type": int, "full": int, "snr_db": float}
"""

import io
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

log = logging.getLogger("rfml.create_shards")

try:
    import webdataset as wds
except ImportError:
    log.error("webdataset not installed. Run: pip install webdataset")
    sys.exit(1)


def create_shards(
    processed_dir: str = "data/processed",
    features_dir: str = "data/features",
    output_dir: str = "data/shards",
    shard_size_mb: int = 1000,
    split: str = "train",
):
    """Create WebDataset shards from processed data."""
    processed = Path(processed_dir)
    features = Path(features_dir)
    output = Path(output_dir) / split
    output.mkdir(parents=True, exist_ok=True)

    shard_pattern = str(output / "rfiq-%06d.tar")
    max_shard_bytes = shard_size_mb * 1024 * 1024

    # Collect all HDF5 files for this split
    h5_files = sorted(processed.rglob(f"*/{split}/samples.h5"))
    if not h5_files:
        log.warning(f"No processed data found for split '{split}'")
        return

    log.info(f"Found {len(h5_files)} HDF5 files for split '{split}'")

    sample_idx = 0
    with wds.ShardWriter(shard_pattern, maxsize=max_shard_bytes) as sink:
        for h5_path in h5_files:
            dataset_name = h5_path.parent.parent.name
            log.info(f"Processing {dataset_name}/{split}...")

            with h5py.File(h5_path, "r") as f:
                n = f["iq"].shape[0]

                for i in tqdm(range(n), desc=f"  {dataset_name}"):
                    key = f"{sample_idx:010d}"
                    sample = {"__key__": key}

                    # IQ data
                    iq = f["iq"][i]
                    buf = io.BytesIO()
                    np.save(buf, iq)
                    sample["iq.npy"] = buf.getvalue()

                    # Precomputed features (if available)
                    feat_dir = features / dataset_name / split
                    spec_path = feat_dir / f"spec_{i:08d}.npy"
                    if spec_path.exists():
                        sample["spec.npy"] = spec_path.read_bytes()

                    hos_path = feat_dir / f"hos_{i:08d}.npy"
                    if hos_path.exists():
                        sample["hos.npy"] = hos_path.read_bytes()

                    cyclo_path = feat_dir / f"cyclo_{i:08d}.npy"
                    if cyclo_path.exists():
                        sample["cyclo.npy"] = cyclo_path.read_bytes()

                    # Labels
                    label_data = {
                        "binary": int(f["label_binary"][i]),
                        "type": int(f["label_type"][i]),
                        "full": int(f["label_full"][i]),
                        "snr_db": float(f["snr_db"][i]),
                        "dataset": dataset_name,
                    }
                    sample["label.json"] = json.dumps(label_data).encode()

                    sink.write(sample)
                    sample_idx += 1

    log.info(f"Created shards for {split}: {sample_idx} samples total")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.logging import setup_logging

    setup_logging()

    for split in ["train", "val", "test"]:
        create_shards(split=split)
