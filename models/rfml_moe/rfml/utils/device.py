"""Unified device abstraction layer supporting NVIDIA CUDA and AMD ROCm (HIP)."""

import os
import torch
from contextlib import contextmanager
from typing import Optional


def detect_backend() -> str:
    """Detect available GPU backend.

    Returns one of "cuda", "rocm", or "cpu".
    """
    if torch.cuda.is_available():
        # torch.version.hip is set when PyTorch was built with ROCm support
        if torch.version.hip is not None:
            return "rocm"
        return "cuda"
    return "cpu"


def get_device_info() -> dict:
    """Return a dict with GPU/backend diagnostics."""
    backend = detect_backend()
    info: dict = {
        "backend": backend,
        "pytorch_version": torch.__version__,
        "torch_compile_available": hasattr(torch, "compile"),
    }

    if backend in ("cuda", "rocm"):
        dev = torch.device("cuda", 0)
        props = torch.cuda.get_device_properties(dev)
        info["gpu_name"] = props.name
        info["total_memory_gb"] = props.total_memory / (1024 ** 3)
        if backend == "cuda":
            info["compute_capability"] = f"{props.major}.{props.minor}"
            info["backend_version"] = torch.version.cuda or "unknown"
        else:
            info["backend_version"] = torch.version.hip or "unknown"
    else:
        info["gpu_name"] = "CPU"
        info["total_memory_gb"] = 0.0
        info["backend_version"] = "N/A"

    return info


def setup_distributed(backend: str = "auto") -> None:
    """Initialise torch.distributed with an appropriate backend.

    For CUDA uses NCCL; for ROCm uses RCCL (API-compatible with NCCL).
    Skips setup when only one GPU is visible or no GPU is available.
    """
    if not torch.cuda.is_available():
        return
    if torch.cuda.device_count() < 2:
        return

    detected = detect_backend()

    if backend == "auto":
        # RCCL ships as the "nccl" backend string inside ROCm PyTorch builds
        dist_backend = "nccl"
    elif backend in ("nccl", "rccl"):
        dist_backend = "nccl"
    else:
        dist_backend = backend

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=dist_backend)


class DeviceManager:
    """Unified device management for CUDA (NVIDIA) and ROCm (AMD MI300X)."""

    def __init__(self, device_str: str = "auto", gpu_id: int = 0):
        """Initialise the manager.

        Parameters
        ----------
        device_str:
            "auto"       – detect available backend
            "cuda"       – force NVIDIA CUDA
            "rocm"/"hip" – force AMD ROCm
            "cpu"        – force CPU
        gpu_id:
            Ordinal of the GPU to use (ignored for CPU).
        """
        self._gpu_id = gpu_id
        self._backend, self._device = self._resolve(device_str, gpu_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(device_str: str, gpu_id: int):
        """Return (backend_str, torch.device)."""
        ds = device_str.lower().strip()

        if ds == "cpu":
            return "cpu", torch.device("cpu")

        detected = detect_backend()

        if ds == "auto":
            if detected == "cpu":
                return "cpu", torch.device("cpu")
            return detected, torch.device("cuda", gpu_id)

        if ds in ("rocm", "hip"):
            if detected != "rocm":
                raise RuntimeError(
                    "ROCm backend requested but not available. "
                    "Ensure PyTorch was built with ROCm support."
                )
            return "rocm", torch.device("cuda", gpu_id)

        if ds == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA backend requested but torch.cuda.is_available() is False."
                )
            if detected == "rocm":
                raise RuntimeError(
                    "CUDA backend requested but the installed PyTorch uses ROCm. "
                    "Use device_str='rocm' or 'auto'."
                )
            return "cuda", torch.device("cuda", gpu_id)

        raise ValueError(
            f"Unknown device_str {device_str!r}. "
            "Expected 'auto', 'cuda', 'rocm', 'hip', or 'cpu'."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """The resolved torch.device."""
        return self._device

    @property
    def backend(self) -> str:
        """Active backend: 'cuda', 'rocm', or 'cpu'."""
        return self._backend

    @property
    def device_name(self) -> str:
        """Human-readable GPU name, or 'CPU'."""
        if self._backend == "cpu":
            return "CPU"
        return torch.cuda.get_device_properties(self._device).name

    @property
    def total_memory_gb(self) -> float:
        """Total VRAM in GiB, or 0 for CPU."""
        if self._backend == "cpu":
            return 0.0
        return torch.cuda.get_device_properties(self._device).total_memory / (1024 ** 3)

    @property
    def is_gpu(self) -> bool:
        """True when a GPU backend is active."""
        return self._backend != "cpu"

    # ------------------------------------------------------------------
    # Environment / optimisation setup
    # ------------------------------------------------------------------

    def setup_environment(self) -> None:
        """Set backend-specific environment variables and PyTorch flags."""
        if self._backend == "rocm":
            # MI300X tuning knobs
            os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
            os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
            os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")
            os.environ.setdefault("GPU_MAX_HW_QUEUES", "2")
            os.environ.setdefault("MIOPEN_FIND_MODE", "3")

        elif self._backend == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------------
    # Dtype / autocast helpers
    # ------------------------------------------------------------------

    def get_autocast_dtype(self) -> torch.dtype:
        """Return bf16 for modern CUDA / ROCm; fp16 for older CUDA; cpu uses bf16."""
        if self._backend == "cpu":
            return torch.bfloat16

        if self._backend == "rocm":
            return torch.bfloat16

        # CUDA: bf16 requires compute capability >= 8.0 (Ampere+)
        props = torch.cuda.get_device_properties(self._device)
        if props.major >= 8:
            return torch.bfloat16
        return torch.float16

    @contextmanager
    def get_autocast_context(self):
        """Yield an autocast context appropriate for the active backend."""
        if self._backend == "cpu":
            yield
            return

        dtype = self.get_autocast_dtype()
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def memory_stats(self) -> dict:
        """Return current GPU memory usage statistics."""
        if self._backend == "cpu":
            return {"backend": "cpu"}

        allocated = torch.cuda.memory_allocated(self._device)
        reserved = torch.cuda.memory_reserved(self._device)
        total = torch.cuda.get_device_properties(self._device).total_memory

        return {
            "backend": self._backend,
            "device": str(self._device),
            "allocated_gb": allocated / (1024 ** 3),
            "reserved_gb": reserved / (1024 ** 3),
            "total_gb": total / (1024 ** 3),
            "free_gb": (total - reserved) / (1024 ** 3),
        }

    def optimal_batch_size(
        self,
        model_params_m: float,
        sample_size_kb: float,
    ) -> int:
        """Estimate a safe maximum batch size given available VRAM.

        Parameters
        ----------
        model_params_m:
            Number of model parameters in millions.
        sample_size_kb:
            Memory footprint of a single sample in kilobytes.

        Returns
        -------
        Estimated batch size (at least 1).
        """
        if self._backend == "cpu":
            return 32  # conservative default for CPU

        total_bytes = torch.cuda.get_device_properties(self._device).total_memory
        # Reserve ~20 % for framework overhead and gradients
        usable_bytes = total_bytes * 0.80

        # Model weights + optimizer states: ~16 bytes/param (fp32 weights + Adam states)
        model_bytes = model_params_m * 1e6 * 16
        remaining = usable_bytes - model_bytes

        if remaining <= 0:
            return 1

        sample_bytes = sample_size_kb * 1024
        # Account for activations (roughly 3× sample size during forward/backward)
        batch_size = int(remaining / (sample_bytes * 3))
        return max(1, batch_size)

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile_model(self, model: torch.nn.Module, mode: str = "max-autotune"):
        """Wrap model with torch.compile using backend-appropriate settings.

        Returns the original model unchanged if torch.compile is unavailable.
        """
        if not hasattr(torch, "compile"):
            return model

        if self._backend == "rocm":
            # ROCm / HIP uses the "inductor" backend but may need explicit hints
            return torch.compile(model, mode=mode, backend="inductor")

        # CUDA standard path
        return torch.compile(model, mode=mode)

    # ------------------------------------------------------------------
    # Gradient scaling
    # ------------------------------------------------------------------

    def create_grad_scaler(self) -> Optional["torch.cuda.amp.GradScaler"]:
        """Return a GradScaler for fp16, or None when bf16/CPU is used.

        bf16 does not suffer from the underflow issues that require loss scaling,
        so None is returned and callers should skip the scaler entirely.
        """
        if self._backend == "cpu":
            return None

        dtype = self.get_autocast_dtype()
        if dtype == torch.float16:
            return torch.cuda.amp.GradScaler()

        # bf16 — no scaling needed
        return None
