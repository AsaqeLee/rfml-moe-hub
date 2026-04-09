"""Higher-Order Statistics (HOS) feature extractor for RF IQ signals."""

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger("rfml.features")

# Canonical cumulant names and their (p, q) orders
_CUMULANT_ORDERS: Dict[str, tuple] = {
    "C20": (2, 0),
    "C21": (2, 1),
    "C40": (4, 0),
    "C41": (4, 1),
    "C42": (4, 2),
    "C60": (6, 0),
    "C61": (6, 1),
    "C62": (6, 2),
    "C63": (6, 3),
}


class HOSExtractor:
    """Extract higher-order cumulant features from complex IQ signals.

    Cumulants are computed up to 6th order using O(N) moment estimators.
    Power-invariant normalization is applied:
        Ĉ_{p,q} = C_{p,q} / (C21)^(p/2)

    Args:
        config: dict with keys:
            max_order   (int)   - maximum cumulant order, default 6
            cumulants   (list)  - list of cumulant names to extract,
                                  default all 9 defined above
            normalize   (bool)  - apply power-invariant normalization,
                                  default True
            feature_dim (int)   - target output dim via zero-padding or
                                  truncation; None = use raw cumulant count
            device      (str)   - computation device
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.max_order = int(cfg.get("max_order", 6))
        requested = cfg.get(
            "cumulants",
            ["C20", "C21", "C40", "C41", "C42", "C60", "C61", "C62", "C63"],
        )
        self.cumulant_names: List[str] = [
            c for c in requested if c in _CUMULANT_ORDERS
        ]
        self.normalize = bool(cfg.get("normalize", True))
        self.feature_dim: Optional[int] = cfg.get("feature_dim", None)
        if self.feature_dim is not None:
            self.feature_dim = int(self.feature_dim)

        device_str = str(cfg.get("device", "cpu"))
        self.device = torch.device(
            device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        logger.debug(
            "HOSExtractor: cumulants=%s normalize=%s feature_dim=%s device=%s",
            self.cumulant_names,
            self.normalize,
            self.feature_dim,
            self.device,
        )

    # ------------------------------------------------------------------
    # Moment computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _moments(z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute raw complex moments needed for cumulants up to 6th order.

        Uses O(N) estimators.  z is 1-D complex tensor.
        Returns a dict of moment tensors (scalar complex values).
        """
        N = z.shape[0]
        zc = z.conj()

        m10 = z.mean()                         # E[z]
        m11 = (z * zc).real.mean().to(z.dtype) # E[|z|^2], cast to complex dtype
        m20 = (z * z).mean()                   # E[z^2]
        m21 = (z * z * zc).mean()              # E[z^2 z*]
        m22 = (z * zc).pow(2).real.mean().to(z.dtype)  # E[|z|^4]
        m30 = (z ** 3).mean()
        m31 = (z ** 3 * zc).mean()
        m32 = (z ** 2 * zc ** 2).mean()  # = E[|z^2|^2 * z^{-0}... approx
        m40 = (z ** 4).mean()
        m41 = (z ** 4 * zc).mean()
        m42 = (z ** 2 * zc ** 2).mean()  # same as m32 for zero-mean
        m60 = (z ** 6).mean()
        m61 = (z ** 6 * zc).mean()
        m62 = (z ** 4 * zc ** 2).mean()
        m63 = (z ** 3 * zc ** 3).mean()

        return {
            "m10": m10, "m11": m11, "m20": m20, "m21": m21, "m22": m22,
            "m30": m30, "m31": m31, "m32": m32,
            "m40": m40, "m41": m41, "m42": m42,
            "m60": m60, "m61": m61, "m62": m62, "m63": m63,
        }

    @staticmethod
    def _cumulants_from_moments(
        m: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Convert moments to cumulants using standard moment-cumulant relations.

        For zero-mean signals: C_{pq} = M_{pq} minus lower-order products.
        We assume near-zero mean (typical after DC removal) and use simplified
        zero-mean relations which are exact for stationary modulations.
        """
        # Second-order
        C20 = m["m20"]
        C21 = m["m11"]  # E[|z|^2] is the variance = C21 for zero-mean

        # Fourth-order (zero-mean cumulants)
        C40 = m["m40"] - 3 * m["m20"] ** 2
        C41 = m["m41"] - 3 * m["m20"] * m["m21"]
        C42 = m["m42"] - torch.abs(m["m20"]) ** 2 - 2 * m["m21"] ** 2

        # Sixth-order (zero-mean; simplified Brillinger relations)
        C60 = (
            m["m60"]
            - 15 * m["m40"] * m["m20"]
            - 10 * m["m30"] ** 2
            + 30 * m["m20"] ** 3
        )
        C61 = (
            m["m61"]
            - 5 * m["m41"] * m["m21"]
            - 10 * m["m31"] * m["m30"]
            - 5 * m["m40"] * C20.conj()
            + 30 * m["m20"] ** 2 * m["m21"]
        )
        C62 = (
            m["m62"]
            - 6 * m["m42"] * m["m21"]
            - 8 * m["m32"] * m["m31"]
            - m["m40"] * m["m22"]
            + 6 * m["m20"] * m["m22"] * m["m20"].conj()
            + 24 * m["m20"] * m["m21"] ** 2
        )
        C63 = (
            m["m63"]
            - 9 * m["m42"] * m["m21"]
            - 12 * m["m32"] ** 2
            + 12 * m["m22"] * m["m21"] ** 2
            + 18 * m["m42"] * m["m21"]
        )

        return {
            "C20": C20,
            "C21": C21,
            "C40": C40,
            "C41": C41,
            "C42": C42,
            "C60": C60,
            "C61": C61,
            "C62": C62,
            "C63": C63,
        }

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_cumulants(
        cumulants: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Power-invariant normalization: Ĉ_{p,q} = C_{p,q} / (C21)^(p/2)."""
        c21_mag = torch.abs(cumulants["C21"]).clamp(min=1e-10)
        normalized: Dict[str, torch.Tensor] = {}
        for name, val in cumulants.items():
            p, _ = _CUMULANT_ORDERS[name]
            scale = c21_mag ** (p / 2.0)
            normalized[name] = val / scale
        return normalized

    # ------------------------------------------------------------------
    # Feature vector assembly
    # ------------------------------------------------------------------

    def _to_feature_vector(
        self, cumulants: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Stack cumulant magnitudes into a real feature vector."""
        parts = []
        for name in self.cumulant_names:
            val = cumulants[name]
            # Use real part, imaginary part, and magnitude for complex cumulants
            parts.append(torch.abs(val).unsqueeze(0))
        vec = torch.cat(parts, dim=0)  # [num_cumulants]

        if self.feature_dim is not None:
            if vec.shape[0] < self.feature_dim:
                pad = torch.zeros(
                    self.feature_dim - vec.shape[0],
                    dtype=vec.dtype,
                    device=vec.device,
                )
                vec = torch.cat([vec, pad], dim=0)
            else:
                vec = vec[: self.feature_dim]
        return vec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, iq: torch.Tensor) -> torch.Tensor:
        """Extract HOS features from a single IQ segment.

        Args:
            iq: float tensor [2, N] (I and Q rows) or complex [N]

        Returns:
            float tensor [D] where D = len(cumulant_names) or feature_dim
        """
        iq = iq.to(self.device)

        if iq.is_complex():
            z = iq
        else:
            z = torch.complex(iq[0], iq[1])

        # Center signal (remove DC)
        z = z - z.mean()

        moments = self._moments(z)
        cumulants = self._cumulants_from_moments(moments)

        if self.normalize:
            cumulants = self._normalize_cumulants(cumulants)

        return self._to_feature_vector(cumulants).float()

    def extract_batch(self, iq_batch: torch.Tensor) -> torch.Tensor:
        """Batch HOS extraction.

        Args:
            iq_batch: [B, 2, N] float or [B, N] complex

        Returns:
            [B, D] float tensor
        """
        iq_batch = iq_batch.to(self.device)
        results = [self.extract(iq_batch[b]) for b in range(iq_batch.shape[0])]
        return torch.stack(results, dim=0)

    def __call__(
        self, iq: torch.Tensor, batch: bool = False
    ) -> torch.Tensor:
        if batch:
            return self.extract_batch(iq)
        return self.extract(iq)
