"""Feature extraction pipeline orchestrator for the RF ML system."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import torch

from .augmentation import RFAugmentor
from .cyclostationary import CycloExtractor
from .emd_denoising import EMDDenoiser
from .energy_gate import EnergyDetector
from .gaf import GAFExtractor
from .hos import HOSExtractor
from .spectrogram import SpectrogramExtractor
from .vmd import VMDExtractor
from .wavelet import WaveletExtractor

logger = logging.getLogger("rfml.features")


class FeaturePipeline:
    """Orchestrate all feature extractors and optional augmentation.

    Given a raw IQ tensor, produces a dictionary of feature modalities used
    by the MoE experts:

        "iq"          - [2, N] float (pass-through, optionally augmented)
        "spectrogram" - [3, H, W] float
        "hos"         - [D_hos] float
        "cyclo"       - [D_cyclo] float
        "vmd_gaf"     - [3, n, n] float (GASF + GADF + raw GASF)
        "tfms_iq"     - [2, N] float (same as iq, separate key for TFMS expert)
        "snr_db"      - [] float scalar (estimated SNR in dB)

    The pipeline includes an energy gate for binary signal detection and
    optional EMD denoising at low SNR.

    Args:
        config: dict matching the ``features`` section of default.yaml.
            Top-level keys: spectrogram, hos, cyclostationary, wavelet,
                           energy_gate, emd, vmd, gaf.
            Additional pipeline-level keys:
                device          (str)  - override device for all extractors
                augment         (bool) - apply augmentation during extraction
                augmentation    (dict) - passed to RFAugmentor
                parallel        (bool) - run extractors in parallel threads
                include_wavelet (bool) - append wavelet channel to spectrogram
                emd_threshold   (float) - SNR (dB) below which EMD denoising is applied
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}

        # Resolve device: pipeline-level overrides individual extractor configs
        device = str(cfg.get("device", "cpu"))

        def _merge(sub_key: str) -> dict:
            sub = dict(cfg.get(sub_key, {}))
            if "device" not in sub:
                sub["device"] = device
            return sub

        # Original extractors
        self.spec_extractor = SpectrogramExtractor(_merge("spectrogram"))
        self.hos_extractor = HOSExtractor(_merge("hos"))
        self.cyclo_extractor = CycloExtractor(_merge("cyclostationary"))
        self.wavelet_extractor = WaveletExtractor(_merge("wavelet"))

        # New extractors (Paper Integration)
        eg_cfg = cfg.get("energy_gate", {})
        self.energy_gate = EnergyDetector(
            p_fa=float(eg_cfg.get("p_fa", 0.01)),
            buffer_size=int(eg_cfg.get("buffer_size", 1024)),
            calibration_percentile=float(eg_cfg.get("calibration_percentile", 10.0)),
        )

        emd_cfg = cfg.get("emd", {})
        self.emd_denoiser = EMDDenoiser(
            num_imfs=int(emd_cfg.get("num_imfs", 4)),
            noise_imfs=int(emd_cfg.get("noise_imfs", 2)),
            max_sifting=int(emd_cfg.get("max_sifting", 20)),
        )

        vmd_cfg = cfg.get("vmd", {})
        self.vmd_extractor = VMDExtractor(
            K=int(vmd_cfg.get("K", 5)),
            alpha=float(vmd_cfg.get("alpha", 2000.0)),
            tau=float(vmd_cfg.get("tau", 0.0)),
            max_iter=int(vmd_cfg.get("max_iter", 500)),
            tol=float(vmd_cfg.get("tol", 1e-7)),
        )

        gaf_cfg = cfg.get("gaf", {})
        self.gaf_extractor = GAFExtractor(
            n=int(gaf_cfg.get("n", 256)),
            method=str(gaf_cfg.get("method", "both")),
        )

        self.emd_threshold = float(cfg.get("emd_threshold", -5.0))
        self.enable_energy_gate = bool(cfg.get("enable_energy_gate", True))
        self.enable_vmd_gaf = bool(cfg.get("enable_vmd_gaf", True))

        self.do_augment = bool(cfg.get("augment", False))
        aug_cfg = dict(cfg.get("augmentation", {}))
        if "device" not in aug_cfg:
            aug_cfg["device"] = device
        self.augmentor = RFAugmentor(aug_cfg)

        self.parallel = bool(cfg.get("parallel", False))
        self.include_wavelet = bool(cfg.get("include_wavelet", False))

        self._device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )

        logger.info(
            "FeaturePipeline initialised: device=%s augment=%s parallel=%s "
            "wavelet=%s energy_gate=%s vmd_gaf=%s emd_threshold=%.1f dB",
            self._device,
            self.do_augment,
            self.parallel,
            self.include_wavelet,
            self.enable_energy_gate,
            self.enable_vmd_gaf,
            self.emd_threshold,
        )

    # ------------------------------------------------------------------
    # Internal extraction helpers
    # ------------------------------------------------------------------

    def _extract_spectrogram(self, iq: torch.Tensor) -> torch.Tensor:
        spec = self.spec_extractor.extract(iq)
        if self.include_wavelet:
            try:
                wv = self.wavelet_extractor.extract(iq)
                # Resize wavelet vector to match spectrogram spatial dims
                H, W = spec.shape[1], spec.shape[2]
                wv_map = wv[: H * W].reshape(1, H, W) if wv.numel() >= H * W else (
                    torch.nn.functional.pad(wv, (0, H * W - wv.numel())).reshape(1, H, W)
                )
                spec = torch.cat([spec, wv_map], dim=0)
            except Exception as exc:
                logger.warning("Wavelet extraction failed, skipping: %s", exc)
        return spec

    def _extract_vmd_gaf(self, iq: torch.Tensor) -> torch.Tensor:
        """Extract VMD-GAF image from IQ signal.

        Runs VMD decomposition, selects effective IMFs via PCC,
        then generates a 3-channel GAF image (GASF_denoised, GADF, GASF_raw).

        Args:
            iq: [2, N] IQ tensor.

        Returns:
            [3, n, n] GAF image tensor.
        """
        try:
            # VMD on magnitude signal
            denoised_mag = self.vmd_extractor.extract(iq)
            # Raw magnitude for channel 2
            z = torch.complex(iq[0], iq[1])
            raw_mag = z.abs()
            # Generate 3-channel GAF image
            return self.gaf_extractor.extract(denoised_mag, raw_mag)
        except Exception as exc:
            logger.warning("VMD-GAF extraction failed, returning zeros: %s", exc)
            n = self.gaf_extractor.n
            return torch.zeros(3, n, n, device=iq.device)

    def _extract_all_serial(
        self, iq: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        result = {
            "iq": iq,
            "spectrogram": self._extract_spectrogram(iq),
            "hos": self.hos_extractor.extract(iq),
            "cyclo": self.cyclo_extractor.extract(iq),
            "tfms_iq": iq,  # TFMS expert receives raw IQ
        }
        if self.enable_vmd_gaf:
            result["vmd_gaf"] = self._extract_vmd_gaf(iq)
        return result

    def _extract_all_parallel(
        self, iq: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        results: Dict[str, torch.Tensor] = {"iq": iq, "tfms_iq": iq}

        def run_spec():
            return ("spectrogram", self._extract_spectrogram(iq))

        def run_hos():
            return ("hos", self.hos_extractor.extract(iq))

        def run_cyclo():
            return ("cyclo", self.cyclo_extractor.extract(iq))

        def run_vmd_gaf():
            return ("vmd_gaf", self._extract_vmd_gaf(iq))

        workers = [run_spec, run_hos, run_cyclo]
        if self.enable_vmd_gaf:
            workers.append(run_vmd_gaf)

        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = [pool.submit(fn) for fn in workers]
            for future in futures:
                key, val = future.result()
                results[key] = val

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, iq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract all feature modalities from a single IQ segment.

        Applies energy detection gate, optional EMD denoising at low SNR,
        and extracts all modalities including VMD-GAF features.

        Args:
            iq: [2, N] float tensor (I and Q rows)

        Returns:
            dict with keys "iq", "spectrogram", "hos", "cyclo", "vmd_gaf",
            "tfms_iq", "snr_db". Returns a minimal dict with
            "energy_gate_pass": False if no drone signal detected.
        """
        iq = iq.to(self._device).float()

        # Energy gate: skip full extraction if no signal detected
        if self.enable_energy_gate and not self.energy_gate.detect(iq):
            n = self.gaf_extractor.n
            return {
                "energy_gate_pass": False,
                "snr_db": torch.tensor(float("-inf"), device=self._device),
            }

        # Estimate SNR for routing and conditional denoising
        snr_db = self.energy_gate.snr_estimate_db(iq)
        snr_tensor = torch.tensor(snr_db, device=self._device)

        # Apply EMD denoising at low SNR
        if snr_db < self.emd_threshold:
            try:
                iq = self.emd_denoiser.denoise(iq)
                logger.debug("EMD denoising applied at SNR=%.1f dB", snr_db)
            except Exception as exc:
                logger.warning("EMD denoising failed at SNR=%.1f dB: %s", snr_db, exc)

        if self.do_augment:
            iq = self.augmentor.augment(iq)

        if self.parallel:
            result = self._extract_all_parallel(iq)
        else:
            result = self._extract_all_serial(iq)

        result["snr_db"] = snr_tensor
        result["energy_gate_pass"] = True
        return result

    def extract_batch(
        self, iq_batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract all feature modalities for a batch of IQ segments.

        Args:
            iq_batch: [B, 2, N] float tensor

        Returns:
            dict with keys "iq", "spectrogram", "hos", "cyclo", "vmd_gaf",
            "tfms_iq", "snr_db", each value having a leading batch dimension B.
        """
        iq_batch = iq_batch.to(self._device).float()

        if self.do_augment:
            iq_batch = self.augmentor.augment_batch(iq_batch)

        # Collect all tensor keys from first sample to determine structure
        first_sample = self.extract(iq_batch[0])
        tensor_keys = [k for k, v in first_sample.items() if isinstance(v, torch.Tensor)]

        batch_results: Dict[str, list] = {k: [first_sample[k]] for k in tensor_keys}

        for b in range(1, iq_batch.shape[0]):
            sample = self.extract(iq_batch[b])
            for key in tensor_keys:
                if key in sample:
                    batch_results[key].append(sample[key])

        return {k: torch.stack(v, dim=0) for k, v in batch_results.items()}

    def __call__(
        self, iq: torch.Tensor, batch: bool = False
    ) -> Dict[str, torch.Tensor]:
        if batch:
            return self.extract_batch(iq)
        return self.extract(iq)
