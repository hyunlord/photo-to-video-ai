"""
GPU Manager for Local AI

Handles CUDA device management, memory monitoring, and OOM prevention.
Provides utilities for efficient GPU resource utilization.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    device_id: int
    name: str
    total_memory: int  # bytes
    allocated_memory: int  # bytes
    cached_memory: int  # bytes
    free_memory: int  # bytes
    compute_capability: Tuple[int, int]

    @property
    def memory_usage_percent(self) -> float:
        """Get memory usage as percentage."""
        return (self.allocated_memory / self.total_memory) * 100 if self.total_memory > 0 else 0

    @property
    def total_memory_gb(self) -> float:
        """Get total memory in GB."""
        return self.total_memory / (1024 ** 3)

    @property
    def free_memory_gb(self) -> float:
        """Get free memory in GB."""
        return self.free_memory / (1024 ** 3)


class GPUManager:
    """
    Manages GPU resources for local AI inference.

    Features:
    - CUDA availability detection
    - Memory monitoring and management
    - OOM prevention with automatic cache clearing
    - Multi-GPU device selection
    """

    _instance = None
    _torch = None

    def __new__(cls):
        """Singleton pattern for GPU manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._device = None
        self._torch = None
        self._cuda_available = False

        self._init_torch()

    def _init_torch(self):
        """Initialize PyTorch and check CUDA availability."""
        try:
            import torch
            self._torch = torch
            self._cuda_available = torch.cuda.is_available()

            if self._cuda_available:
                # Set default device
                self._device = torch.device("cuda:0")
                device_name = torch.cuda.get_device_name(0)
                total_mem = torch.cuda.get_device_properties(0).total_memory
                logger.info(f"CUDA available: {device_name} ({total_mem / 1024**3:.1f}GB)")
            else:
                self._device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU (very slow)")

        except ImportError:
            logger.error("PyTorch not installed")
            self._cuda_available = False

    @property
    def is_available(self) -> bool:
        """Check if GPU is available for inference."""
        return self._cuda_available

    @property
    def device(self):
        """Get the current device (cuda or cpu)."""
        return self._device

    @property
    def device_name(self) -> str:
        """Get the device name string."""
        if self._cuda_available:
            return f"cuda:{self._torch.cuda.current_device()}"
        return "cpu"

    def get_gpu_info(self, device_id: int = 0) -> Optional[GPUInfo]:
        """
        Get detailed information about a GPU.

        Args:
            device_id: CUDA device ID

        Returns:
            GPUInfo object or None if unavailable
        """
        if not self._cuda_available:
            return None

        try:
            props = self._torch.cuda.get_device_properties(device_id)

            return GPUInfo(
                device_id=device_id,
                name=props.name,
                total_memory=props.total_memory,
                allocated_memory=self._torch.cuda.memory_allocated(device_id),
                cached_memory=self._torch.cuda.memory_reserved(device_id),
                free_memory=props.total_memory - self._torch.cuda.memory_allocated(device_id),
                compute_capability=(props.major, props.minor),
            )
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return None

    def get_all_gpus(self) -> list[GPUInfo]:
        """Get information about all available GPUs."""
        if not self._cuda_available:
            return []

        gpus = []
        for i in range(self._torch.cuda.device_count()):
            info = self.get_gpu_info(i)
            if info:
                gpus.append(info)
        return gpus

    def select_best_gpu(self, min_memory_gb: float = 6.0) -> Optional[int]:
        """
        Select the best GPU for inference.

        Args:
            min_memory_gb: Minimum required free memory in GB

        Returns:
            Device ID of the best GPU, or None if none suitable
        """
        if not self._cuda_available:
            return None

        best_gpu = None
        max_free_memory = 0

        for gpu in self.get_all_gpus():
            if gpu.free_memory_gb >= min_memory_gb and gpu.free_memory > max_free_memory:
                best_gpu = gpu.device_id
                max_free_memory = gpu.free_memory

        return best_gpu

    def clear_cache(self):
        """Clear GPU memory cache."""
        if self._cuda_available:
            self._torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")

    def synchronize(self):
        """Synchronize CUDA operations."""
        if self._cuda_available:
            self._torch.cuda.synchronize()

    @contextmanager
    def memory_efficient_context(self, clear_before: bool = True, clear_after: bool = True):
        """
        Context manager for memory-efficient inference.

        Args:
            clear_before: Clear cache before entering context
            clear_after: Clear cache after exiting context
        """
        if clear_before:
            self.clear_cache()

        try:
            yield self.device
        finally:
            if clear_after:
                self.clear_cache()

    def check_memory_for_model(self, required_gb: float) -> Dict[str, Any]:
        """
        Check if there's enough memory for a model.

        Args:
            required_gb: Required VRAM in GB

        Returns:
            Dict with 'available', 'message', and memory details
        """
        if not self._cuda_available:
            return {
                "available": False,
                "message": "CUDA not available",
                "required_gb": required_gb,
            }

        info = self.get_gpu_info()
        if not info:
            return {
                "available": False,
                "message": "Could not get GPU info",
                "required_gb": required_gb,
            }

        # Consider some overhead
        available_gb = info.free_memory_gb * 0.9  # 10% safety margin

        if available_gb >= required_gb:
            return {
                "available": True,
                "message": f"Sufficient memory: {available_gb:.1f}GB available, {required_gb:.1f}GB required",
                "required_gb": required_gb,
                "available_gb": available_gb,
                "total_gb": info.total_memory_gb,
            }
        else:
            return {
                "available": False,
                "message": f"Insufficient memory: {available_gb:.1f}GB available, {required_gb:.1f}GB required",
                "required_gb": required_gb,
                "available_gb": available_gb,
                "total_gb": info.total_memory_gb,
                "suggestion": "Try clearing GPU cache or closing other applications",
            }

    def optimize_for_inference(self):
        """Apply optimizations for inference."""
        if not self._cuda_available:
            return

        # Enable TF32 for faster computation on Ampere+ GPUs
        self._torch.backends.cuda.matmul.allow_tf32 = True
        self._torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmark for consistent input sizes
        self._torch.backends.cudnn.benchmark = True

        logger.debug("GPU optimizations applied")

    def get_status(self) -> Dict[str, Any]:
        """Get overall GPU status."""
        if not self._cuda_available:
            return {
                "available": False,
                "reason": "CUDA not available",
                "device": "cpu",
            }

        info = self.get_gpu_info()
        return {
            "available": True,
            "device": self.device_name,
            "gpu_name": info.name if info else "Unknown",
            "total_memory_gb": info.total_memory_gb if info else 0,
            "free_memory_gb": info.free_memory_gb if info else 0,
            "memory_usage_percent": info.memory_usage_percent if info else 0,
            "compute_capability": f"{info.compute_capability[0]}.{info.compute_capability[1]}" if info else "N/A",
        }


# Global instance
gpu_manager = GPUManager()


def check_cuda_available() -> Dict[str, Any]:
    """Quick check for CUDA availability."""
    return gpu_manager.get_status()
