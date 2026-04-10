"""Multi-modal RF Dataset for drone signal detection.

Provides both standard map-style and WebDataset-based iterable datasets
for loading preprocessed IQ, spectrogram, HOS, and cyclostationary features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("rfml.data")


class RFDataset(Dataset):
    """Map-style dataset that lazily loads preprocessed numpy files.

    Directory structure expected::

        processed_dir/
            {dataset_name}/
                {split}/
                    iq/          -> {idx}.npy  shape (2, N)
                    spectrogram/ -> {idx}.npy  shape (3, 512, 512)
                    hos/         -> {idx}.npy  shape (20,)
                    cyclo/       -> {idx}.npy  shape (1, 512)
                    labels.npy   -> (num_samples, 4) columns: binary, type, full, snr_db

    Args:
        processed_dir: Root directory containing preprocessed data.
        datasets: List of dataset names to include. If None, use all found.
        split: One of "train", "val", "test".
        min_snr: Minimum SNR in dB for curriculum filtering. None = no filter.
        max_snr: Maximum SNR in dB for curriculum filtering. None = no filter.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        datasets: Optional[List[str]] = None,
        split: str = "train",
        min_snr: Optional[float] = None,
        max_snr: Optional[float] = None,
    ):
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.min_snr = min_snr
        self.max_snr = max_snr

        # Discover samples across all requested datasets
        self._samples: List[Tuple[Path, int]] = []  # (dataset_split_dir, local_idx)
        self._labels: Optional[np.ndarray] = None

        self._build_index(datasets)

    def _build_index(self, datasets: Optional[List[str]]):
        """Scan the processed directory and build a flat sample index."""
        if datasets is None:
            dataset_dirs = sorted(
                d for d in self.processed_dir.iterdir()
                if d.is_dir() and (d / self.split).is_dir()
            )
        else:
            dataset_dirs = [
                self.processed_dir / name
                for name in datasets
                if (self.processed_dir / name / self.split).is_dir()
            ]

        all_labels = []
        for ds_dir in dataset_dirs:
            split_dir = ds_dir / self.split
            labels_path = split_dir / "labels.npy"
            if not labels_path.exists():
                logger.warning("No labels.npy in %s, skipping", split_dir)
                continue

            labels = np.load(labels_path)  # (num_samples, 4)
            num_samples = labels.shape[0]

            for idx in range(num_samples):
                snr = float(labels[idx, 3])
                if self.min_snr is not None and snr < self.min_snr:
                    continue
                if self.max_snr is not None and snr > self.max_snr:
                    continue
                self._samples.append((split_dir, idx))
                all_labels.append(labels[idx])

        if all_labels:
            self._labels = np.stack(all_labels, axis=0)
        else:
            self._labels = np.empty((0, 4), dtype=np.float32)

        logger.info(
            "RFDataset[%s]: %d samples from %d dataset(s), SNR filter=[%s, %s]",
            self.split,
            len(self._samples),
            len(dataset_dirs),
            self.min_snr,
            self.max_snr,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        split_dir, local_idx = self._samples[index]
        label_row = self._labels[index]

        # Lazy-load each modality from disk
        iq = np.load(split_dir / "iq" / f"{local_idx}.npy")
        spectrogram = np.load(split_dir / "spectrogram" / f"{local_idx}.npy")
        hos = np.load(split_dir / "hos" / f"{local_idx}.npy")
        cyclo = np.load(split_dir / "cyclo" / f"{local_idx}.npy")

        return {
            "iq": torch.from_numpy(iq).float(),                    # (2, N)
            "spectrogram": torch.from_numpy(spectrogram).float(),  # (3, 512, 512)
            "hos": torch.from_numpy(hos).float(),                  # (20,)
            "cyclo": torch.from_numpy(cyclo).float(),              # (1, 512)
            "label_binary": int(label_row[0]),
            "label_type": int(label_row[1]),
            "label_full": int(label_row[2]),
            "snr_db": float(label_row[3]),
        }


class RFWebDataset:
    """WebDataset-based iterable loader for sharded .tar files.

    Each tar shard contains samples stored as::

        {idx}.iq.npy
        {idx}.spectrogram.npy
        {idx}.hos.npy
        {idx}.cyclo.npy
        {idx}.labels.npy

    Args:
        shards_pattern: Glob pattern for tar shards (e.g. "data/shards/train-{0000..0099}.tar").
        batch_size: Batch size for internal batching.
        shuffle_shards: Whether to shuffle shard order.
        shuffle_buffer: Size of sample shuffle buffer.
    """

    def __init__(
        self,
        shards_pattern: str,
        batch_size: int = 128,
        shuffle_shards: bool = True,
        shuffle_buffer: int = 5000,
    ):
        try:
            import webdataset as wds
        except ImportError:
            raise ImportError(
                "webdataset is required for RFWebDataset. "
                "Install with: pip install webdataset"
            )

        self.batch_size = batch_size

        pipeline = wds.WebDataset(shards_pattern, shardshuffle=shuffle_shards)
        if shuffle_buffer > 0:
            pipeline = pipeline.shuffle(shuffle_buffer)

        pipeline = pipeline.decode().map(self._decode_sample)
        pipeline = pipeline.batched(batch_size, collation_fn=self._collate)

        self._pipeline = pipeline
        logger.info(
            "RFWebDataset: shards=%s batch_size=%d shuffle_buffer=%d",
            shards_pattern,
            batch_size,
            shuffle_buffer,
        )

    @staticmethod
    def _decode_sample(sample: dict) -> Dict[str, torch.Tensor]:
        """Decode a single sample from tar entry dict."""

        def _load_npy(data):
            """Load numpy array from bytes or already-decoded array."""
            if isinstance(data, np.ndarray):
                return data
            return np.load(
                __import__("io").BytesIO(data), allow_pickle=False
            )

        iq = _load_npy(sample["iq.npy"])
        spectrogram = _load_npy(sample["spectrogram.npy"])
        hos = _load_npy(sample["hos.npy"])
        cyclo = _load_npy(sample["cyclo.npy"])
        labels = _load_npy(sample["labels.npy"])

        return {
            "iq": torch.from_numpy(iq).float(),
            "spectrogram": torch.from_numpy(spectrogram).float(),
            "hos": torch.from_numpy(hos).float(),
            "cyclo": torch.from_numpy(cyclo).float(),
            "label_binary": int(labels[0]),
            "label_type": int(labels[1]),
            "label_full": int(labels[2]),
            "snr_db": float(labels[3]),
        }

    @staticmethod
    def _collate(samples: List[dict]) -> Dict[str, torch.Tensor]:
        """Collate a list of sample dicts into a batched dict."""
        return {
            "iq": torch.stack([s["iq"] for s in samples]),
            "spectrogram": torch.stack([s["spectrogram"] for s in samples]),
            "hos": torch.stack([s["hos"] for s in samples]),
            "cyclo": torch.stack([s["cyclo"] for s in samples]),
            "label_binary": torch.tensor([s["label_binary"] for s in samples], dtype=torch.long),
            "label_type": torch.tensor([s["label_type"] for s in samples], dtype=torch.long),
            "label_full": torch.tensor([s["label_full"] for s in samples], dtype=torch.long),
            "snr_db": torch.tensor([s["snr_db"] for s in samples], dtype=torch.float32),
        }

    def __iter__(self):
        return iter(self._pipeline)


def create_dataloaders(config) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders from config.

    Args:
        config: Config object with data.processed_dir, training.batch_size, etc.

    Returns:
        Dict with keys "train", "val", "test" mapping to DataLoaders.
    """
    processed_dir = config.get_nested("data.processed_dir", "data/processed")
    batch_size = config.get_nested("training.batch_size", 128)
    num_workers = config.get_nested("project.num_workers", 12)
    pin_memory = config.get_nested("project.pin_memory", True)

    # Determine enabled datasets
    datasets_cfg = config.get_nested("data.datasets", {})
    enabled_datasets = [
        name for name, ds_cfg in datasets_cfg.items()
        if isinstance(ds_cfg, dict) and ds_cfg.get("enabled", False)
    ]

    loaders = {}
    for split in ("train", "val", "test"):
        dataset = RFDataset(
            processed_dir=processed_dir,
            datasets=enabled_datasets if enabled_datasets else None,
            split=split,
        )

        is_train = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=3 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            drop_last=is_train,
        )
        logger.info(
            "DataLoader[%s]: %d samples, batch_size=%d, workers=%d",
            split,
            len(dataset),
            batch_size,
            num_workers,
        )

    return loaders
