"""
Dataset download system for RF ML drone detection pipeline.

Handles downloading, verification, and extraction of:
  - RFUAV (1.3 TB) from HuggingFace
  - DroneDetect v2 from IEEE DataPort (manual)
  - CardRF (65 GB) from IEEE DataPort (manual)
  - DroneRF (40 GB) from Mendeley Data
  - Tampere/Zenodo from Zenodo REST API
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import textwrap
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
import yaml
from tqdm import tqdm

logger = logging.getLogger("rfml.data.download")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return hex SHA-256 of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _download_file(
    url: str,
    dest: Path,
    session: Optional[requests.Session] = None,
    chunk_size: int = 1 << 20,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> Path:
    """
    Download *url* to *dest*, resuming if the file already exists.

    Returns the destination path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()
    req_headers: Dict[str, str] = dict(headers or {})

    existing_size = dest.stat().st_size if dest.exists() else 0
    if existing_size:
        req_headers["Range"] = f"bytes={existing_size}-"
        logger.debug("Resuming %s from byte %d", url, existing_size)

    try:
        resp = sess.get(url, headers=req_headers, stream=True, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc

    if resp.status_code == 416:
        # Server says range not satisfiable → file already complete
        logger.info("File already fully downloaded: %s", dest.name)
        return dest

    if resp.status_code not in (200, 206):
        raise RuntimeError(
            f"GET {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    total_raw = resp.headers.get("Content-Length")
    total = int(total_raw) if total_raw else None
    mode = "ab" if resp.status_code == 206 else "wb"

    desc = dest.name[:40]
    with (
        open(dest, mode) as fh,
        tqdm(
            total=total,
            initial=existing_size if mode == "ab" else 0,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            leave=False,
        ) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))

    return dest


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    name: str
    source: str
    raw_dir: Path
    size_hint: str = "unknown"
    bands: List[str] = field(default_factory=list)
    format: str = "unknown"
    extra: Dict[str, Any] = field(default_factory=dict)


class DatasetDownloader(ABC):
    """Abstract base for dataset downloaders."""

    def __init__(self, info: DatasetInfo) -> None:
        self.info = info
        self.raw_dir = info.raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    @abstractmethod
    def download(self) -> bool:
        """
        Fetch the dataset.

        Returns True on success, False if skipped / requires manual action.
        """

    @abstractmethod
    def verify(self) -> bool:
        """Return True if the downloaded files appear valid."""

    @abstractmethod
    def extract(self) -> bool:
        """Extract / unpack archives if necessary. Return True on success."""

    @abstractmethod
    def get_info(self) -> DatasetInfo:
        """Return metadata about this dataset."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _log_banner(self, action: str) -> None:
        logger.info("[%s] %s — %s", self.info.name, action, self.info.size_hint)

    def _verify_sha256(self, path: Path, expected: str) -> bool:
        logger.debug("SHA-256 checking %s", path)
        actual = _sha256_file(path)
        if actual.lower() != expected.lower():
            logger.error(
                "SHA-256 mismatch for %s: expected %s, got %s",
                path.name,
                expected,
                actual,
            )
            return False
        logger.info("SHA-256 OK: %s", path.name)
        return True


# ---------------------------------------------------------------------------
# RFUAV — HuggingFace
# ---------------------------------------------------------------------------

class RFUAVDownloader(DatasetDownloader):
    """Downloads RFUAV-1.3T from HuggingFace Hub."""

    REPO_ID = "RFUAV/RFUAV-1.3T"

    def download(self) -> bool:
        self._log_banner("download")
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import]
        except ImportError:
            logger.error(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )
            return False

        try:
            repo_id = self.info.extra.get("repo_id", self.REPO_ID)
            logger.info("Downloading HuggingFace repo %s → %s", repo_id, self.raw_dir)
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(self.raw_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                ignore_patterns=["*.md", "*.txt"],  # skip docs to save bandwidth
            )
            logger.info("[rfuav] snapshot_download complete")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("[rfuav] Download failed: %s", exc)
            return False

    def verify(self) -> bool:
        """Check that at least some binary IQ files are present."""
        files = list(self.raw_dir.rglob("*.bin")) + list(self.raw_dir.rglob("*.iq"))
        if not files:
            # HuggingFace may use arbitrary names; fall back to any file
            files = [p for p in self.raw_dir.rglob("*") if p.is_file()]
        ok = len(files) > 0
        if ok:
            logger.info("[rfuav] verify OK — %d file(s) present", len(files))
        else:
            logger.warning("[rfuav] verify FAIL — no files found in %s", self.raw_dir)
        return ok

    def extract(self) -> bool:
        # HuggingFace delivers files directly; no extraction needed
        logger.debug("[rfuav] No extraction required")
        return True

    def get_info(self) -> DatasetInfo:
        return self.info


# ---------------------------------------------------------------------------
# IEEE DataPort — manual-download base
# ---------------------------------------------------------------------------

class IEEEDataPortDownloader(DatasetDownloader):
    """
    IEEE DataPort datasets require a manual download step.

    This class provides instructions and checks whether the files are present.
    """

    _DATAPORT_URL = "https://ieee-dataport.org/open-access/"

    def _dataset_url(self) -> str:
        ds_id = self.info.extra.get("dataset_id", "")
        return f"{self._DATAPORT_URL}{ds_id}"

    def download(self) -> bool:
        self._log_banner("download (manual required)")
        url = self._dataset_url()
        logger.warning(
            textwrap.dedent(
                f"""
                ╔══════════════════════════════════════════════════════════╗
                  {self.info.name} — MANUAL DOWNLOAD REQUIRED
                ╠══════════════════════════════════════════════════════════╣
                  IEEE DataPort requires a free account and browser login.

                  1. Visit: {url}
                  2. Sign in (or create a free IEEE account)
                  3. Click "Download" and save to:
                       {self.raw_dir}
                  4. Re-run the pipeline — this step will be skipped if
                     files are detected in the target directory.
                ╚══════════════════════════════════════════════════════════╝
                """
            ).strip()
        )
        # Return True so the pipeline continues; verify() gates actual use
        return True

    def verify(self) -> bool:
        files = [p for p in self.raw_dir.rglob("*") if p.is_file()]
        if not files:
            logger.warning(
                "[%s] verify FAIL — directory empty: %s\n"
                "  Please download manually from IEEE DataPort.",
                self.info.name,
                self.raw_dir,
            )
            return False
        total = sum(p.stat().st_size for p in files)
        logger.info(
            "[%s] verify OK — %d file(s), %s",
            self.info.name,
            len(files),
            _human_bytes(total),
        )
        return True

    def extract(self) -> bool:
        """Attempt to unzip/untar any archives found in raw_dir."""
        archives = (
            list(self.raw_dir.glob("*.zip"))
            + list(self.raw_dir.glob("*.tar.gz"))
            + list(self.raw_dir.glob("*.tgz"))
            + list(self.raw_dir.glob("*.tar"))
        )
        if not archives:
            logger.debug("[%s] No archives to extract", self.info.name)
            return True
        for arc in archives:
            logger.info("[%s] Extracting %s", self.info.name, arc.name)
            shutil.unpack_archive(str(arc), str(self.raw_dir))
        return True

    def get_info(self) -> DatasetInfo:
        return self.info


class DroneDetectV2Downloader(IEEEDataPortDownloader):
    """DroneDetect v2 — 7 DJI/Parrot drones, complex IQ .dat files."""


class CardRFDownloader(IEEEDataPortDownloader):
    """CardRF — 5 UAVs + 15 devices, .mat format."""


# ---------------------------------------------------------------------------
# DroneRF — Mendeley Data
# ---------------------------------------------------------------------------

class DroneRFDownloader(DatasetDownloader):
    """
    Downloads DroneRF from Mendeley Data via their public API.

    DOI: 10.17632/s3c4gf5ng2.1
    """

    MENDELEY_API = "https://data.mendeley.com/public-api/datasets"
    DOI_PREFIX = "10.17632/"

    def _dataset_id(self) -> str:
        doi: str = self.info.extra.get("doi", "10.17632/s3c4gf5ng2.1")
        # e.g. "10.17632/s3c4gf5ng2.1" → identifier "s3c4gf5ng2", version "1"
        parts = doi.replace(self.DOI_PREFIX, "").split(".")
        return parts[0] if parts else doi

    def _get_files(self, session: requests.Session) -> List[Dict[str, Any]]:
        """Fetch file listing from Mendeley public API."""
        ds_id = self._dataset_id()
        url = f"{self.MENDELEY_API}/{ds_id}/files"
        logger.debug("Mendeley API: GET %s", url)
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Mendeley API error {resp.status_code} for dataset '{ds_id}': "
                f"{resp.text[:300]}"
            )
        return resp.json()  # type: ignore[return-value]

    def download(self) -> bool:
        self._log_banner("download")
        session = requests.Session()
        session.headers["User-Agent"] = "rfml-pipeline/1.0"

        try:
            files = self._get_files(session)
        except RuntimeError as exc:
            logger.error("[dronerf] %s", exc)
            return False

        if not files:
            logger.error("[dronerf] API returned no files")
            return False

        logger.info("[dronerf] %d file(s) to download", len(files))
        success = True
        for entry in tqdm(files, desc="DroneRF", unit="file"):
            fname = entry.get("filename") or entry.get("name") or "unknown"
            dl_url = entry.get("content_details", {}).get("download_url") or entry.get(
                "download_url"
            )
            if not dl_url:
                logger.warning("[dronerf] No download URL for %s, skipping", fname)
                continue

            dest = self.raw_dir / fname
            try:
                _download_file(dl_url, dest, session=session)
                logger.debug("[dronerf] downloaded %s", fname)
            except RuntimeError as exc:
                logger.error("[dronerf] Failed to download %s: %s", fname, exc)
                success = False

        return success

    def verify(self) -> bool:
        csv_files = list(self.raw_dir.rglob("*.csv"))
        if not csv_files:
            logger.warning("[dronerf] verify FAIL — no CSV files in %s", self.raw_dir)
            return False
        total = sum(p.stat().st_size for p in csv_files)
        logger.info(
            "[dronerf] verify OK — %d CSV file(s), %s",
            len(csv_files),
            _human_bytes(total),
        )
        return True

    def extract(self) -> bool:
        # Mendeley files are raw CSVs; no extraction needed
        logger.debug("[dronerf] No extraction required")
        return True

    def get_info(self) -> DatasetInfo:
        return self.info


# ---------------------------------------------------------------------------
# Tampere / Zenodo
# ---------------------------------------------------------------------------

class TampereZenodoDownloader(DatasetDownloader):
    """
    Downloads the Tampere UAV dataset from Zenodo.

    Record: https://zenodo.org/record/4264467
    Format: IQ int16, dual-band (2.44 GHz + 5.8 GHz)
    """

    ZENODO_API = "https://zenodo.org/api/records"

    def _get_record(self, session: requests.Session) -> Dict[str, Any]:
        record_id = self.info.extra.get("record_id", "4264467")
        url = f"{self.ZENODO_API}/{record_id}"
        logger.debug("Zenodo API: GET %s", url)
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Zenodo API error {resp.status_code} for record '{record_id}': "
                f"{resp.text[:300]}"
            )
        return resp.json()  # type: ignore[return-value]

    def download(self) -> bool:
        self._log_banner("download")
        session = requests.Session()
        session.headers["User-Agent"] = "rfml-pipeline/1.0"

        try:
            record = self._get_record(session)
        except RuntimeError as exc:
            logger.error("[tampere_zenodo] %s", exc)
            return False

        files: List[Dict[str, Any]] = record.get("files", [])
        if not files:
            logger.error(
                "[tampere_zenodo] No files listed in Zenodo record %s",
                self.info.extra.get("record_id"),
            )
            return False

        logger.info("[tampere_zenodo] %d file(s) to download", len(files))
        success = True

        for entry in tqdm(files, desc="Tampere/Zenodo", unit="file"):
            fname = entry.get("key") or entry.get("filename") or "unknown"
            links = entry.get("links", {})
            dl_url = links.get("self") or links.get("download")
            checksum: Optional[str] = entry.get("checksum")  # "md5:..." or "sha256:..."

            if not dl_url:
                logger.warning(
                    "[tampere_zenodo] No download link for %s, skipping", fname
                )
                continue

            dest = self.raw_dir / fname
            try:
                _download_file(dl_url, dest, session=session)
            except RuntimeError as exc:
                logger.error(
                    "[tampere_zenodo] Failed to download %s: %s", fname, exc
                )
                success = False
                continue

            # Zenodo checksums are usually "md5:<hex>" or "sha256:<hex>"
            if checksum and ":" in checksum:
                algo, expected_hex = checksum.split(":", 1)
                if algo.lower() == "sha256":
                    if not self._verify_sha256(dest, expected_hex):
                        success = False
                elif algo.lower() == "md5":
                    actual = hashlib.md5(dest.read_bytes()).hexdigest()  # noqa: S324
                    if actual != expected_hex:
                        logger.error(
                            "[tampere_zenodo] MD5 mismatch for %s", fname
                        )
                        success = False
                    else:
                        logger.info("[tampere_zenodo] MD5 OK: %s", fname)

        return success

    def verify(self) -> bool:
        # Expect IQ int16 files (various extensions)
        candidates = (
            list(self.raw_dir.rglob("*.dat"))
            + list(self.raw_dir.rglob("*.bin"))
            + list(self.raw_dir.rglob("*.iq"))
            + list(self.raw_dir.rglob("*.zip"))  # may be zipped
        )
        if not candidates:
            candidates = [p for p in self.raw_dir.rglob("*") if p.is_file()]

        if not candidates:
            logger.warning(
                "[tampere_zenodo] verify FAIL — no files in %s", self.raw_dir
            )
            return False

        total = sum(p.stat().st_size for p in candidates)
        logger.info(
            "[tampere_zenodo] verify OK — %d file(s), %s",
            len(candidates),
            _human_bytes(total),
        )
        return True

    def extract(self) -> bool:
        archives = (
            list(self.raw_dir.glob("*.zip"))
            + list(self.raw_dir.glob("*.tar.gz"))
            + list(self.raw_dir.glob("*.tgz"))
        )
        if not archives:
            logger.debug("[tampere_zenodo] No archives to extract")
            return True
        for arc in archives:
            logger.info("[tampere_zenodo] Extracting %s", arc.name)
            shutil.unpack_archive(str(arc), str(self.raw_dir))
        return True

    def get_info(self) -> DatasetInfo:
        return self.info


# ---------------------------------------------------------------------------
# Download Manager
# ---------------------------------------------------------------------------

_DOWNLOADER_REGISTRY: Dict[str, type] = {
    "rfuav": RFUAVDownloader,
    "dronedetect_v2": DroneDetectV2Downloader,
    "cardrf": CardRFDownloader,
    "dronerf": DroneRFDownloader,
    "tampere_zenodo": TampereZenodoDownloader,
}


def _build_info(name: str, ds_cfg: Dict[str, Any], raw_root: Path) -> DatasetInfo:
    size_parts = []
    if "size_tb" in ds_cfg:
        size_parts.append(f"{ds_cfg['size_tb']} TB")
    elif "size_gb" in ds_cfg:
        size_parts.append(f"{ds_cfg['size_gb']} GB")
    size_hint = size_parts[0] if size_parts else "unknown"

    return DatasetInfo(
        name=name,
        source=ds_cfg.get("source", "unknown"),
        raw_dir=raw_root / name,
        size_hint=size_hint,
        bands=ds_cfg.get("bands", []),
        format=ds_cfg.get("format", "unknown"),
        extra=ds_cfg,
    )


class DownloadManager:
    """
    Orchestrates downloading all enabled datasets as specified in a config dict.

    Config structure mirrors ``configs/default.yaml`` under the ``data`` key.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        data_cfg: Dict[str, Any] = config.get("data", config)
        self.raw_root = Path(data_cfg.get("raw_dir", "data/raw"))
        self.datasets_cfg: Dict[str, Any] = data_cfg.get("datasets", {})

    # ------------------------------------------------------------------

    def _build_downloader(self, name: str) -> Optional[DatasetDownloader]:
        ds_cfg = self.datasets_cfg.get(name, {})
        cls = _DOWNLOADER_REGISTRY.get(name)
        if cls is None:
            logger.warning("No downloader registered for '%s', skipping", name)
            return None
        info = _build_info(name, ds_cfg, self.raw_root)
        return cls(info)

    def _enabled_names(self) -> List[str]:
        return [
            name
            for name, cfg in self.datasets_cfg.items()
            if cfg.get("enabled", True)
        ]

    # ------------------------------------------------------------------

    def download_all(self) -> Dict[str, bool]:
        """
        Download all enabled datasets.

        Returns a mapping of dataset name → success flag.
        """
        names = self._enabled_names()
        if not names:
            logger.warning("No datasets enabled in config")
            return {}

        results: Dict[str, bool] = {}
        logger.info("Starting download of %d dataset(s): %s", len(names), ", ".join(names))

        for name in names:
            logger.info("=" * 60)
            logger.info("Dataset: %s", name)
            logger.info("=" * 60)
            downloader = self._build_downloader(name)
            if downloader is None:
                results[name] = False
                continue

            t0 = time.monotonic()
            try:
                ok = downloader.download()
            except Exception as exc:  # noqa: BLE001
                logger.exception("[%s] Unexpected error during download: %s", name, exc)
                ok = False

            if ok:
                try:
                    downloader.extract()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s] Extract error (non-fatal): %s", name, exc)

            elapsed = time.monotonic() - t0
            results[name] = ok
            status = "OK" if ok else "FAILED/MANUAL"
            logger.info("[%s] %s (%.1fs)", name, status, elapsed)

        # Summary
        logger.info("=" * 60)
        logger.info("Download summary:")
        for name, ok in results.items():
            logger.info("  %-25s %s", name, "OK" if ok else "FAILED/MANUAL")
        logger.info("=" * 60)

        return results

    def verify_all(self) -> Dict[str, bool]:
        """Verify all enabled datasets (post-download integrity check)."""
        names = self._enabled_names()
        results: Dict[str, bool] = {}
        for name in names:
            downloader = self._build_downloader(name)
            if downloader is None:
                results[name] = False
                continue
            try:
                results[name] = downloader.verify()
            except Exception as exc:  # noqa: BLE001
                logger.exception("[%s] Verify error: %s", name, exc)
                results[name] = False
        return results

    def get_dataset_info(self) -> List[DatasetInfo]:
        """Return DatasetInfo for all known datasets regardless of enabled flag."""
        infos = []
        for name, ds_cfg in self.datasets_cfg.items():
            info = _build_info(name, ds_cfg, self.raw_root)
            infos.append(info)
        return infos


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def download_all(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Top-level convenience function.

    Parameters
    ----------
    config:
        Full pipeline config dict (as loaded from ``configs/default.yaml``).

    Returns
    -------
    dict
        Mapping of dataset name → download success (True/False).

    Example
    -------
    >>> import yaml
    >>> with open("configs/default.yaml") as f:
    ...     cfg = yaml.safe_load(f)
    >>> from data.download import download_all
    >>> results = download_all(cfg)
    """
    manager = DownloadManager(config)
    return manager.download_all()


def load_config_and_download(config_path: str = "configs/default.yaml") -> Dict[str, bool]:
    """Load config from *config_path* and download all enabled datasets."""
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    return download_all(config)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Download RF-ML drone detection datasets"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip downloading; only verify existing files",
    )
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        help="Download a single dataset by name (e.g. rfuav, dronerf)",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    if args.dataset:
        # Restrict to the requested dataset
        data_cfg = cfg.get("data", cfg)
        ds_cfgs = data_cfg.get("datasets", {})
        if args.dataset not in ds_cfgs:
            parser.error(f"Unknown dataset '{args.dataset}'. Known: {list(ds_cfgs)}")
        # Disable everything else
        for k in ds_cfgs:
            ds_cfgs[k]["enabled"] = k == args.dataset

    manager = DownloadManager(cfg)

    if args.verify_only:
        results = manager.verify_all()
    else:
        results = manager.download_all()

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        import sys
        logger.warning(
            "%d dataset(s) need attention: %s",
            len(failed),
            ", ".join(failed),
        )
        sys.exit(1)
