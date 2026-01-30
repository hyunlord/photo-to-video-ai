"""
Local AI Services

Provides local GPU-based video generation using Stable Video Diffusion.
Requires NVIDIA GPU with CUDA support and sufficient VRAM (8GB+ recommended).

Usage:
    from app.services.local_ai import get_svd_service, check_local_ai_available

    # Check availability
    status = check_local_ai_available()
    if status["available"]:
        svd = get_svd_service()
        video_path = svd.generate_video("input.jpg", "output.mp4")
"""

from app.services.local_ai.gpu_manager import (
    GPUManager,
    GPUInfo,
    gpu_manager,
    check_cuda_available,
)

from app.services.local_ai.model_manager import (
    ModelManager,
    model_manager,
    get_model_manager,
    MODEL_CONFIGS,
)

from app.services.local_ai.svd_service import (
    SVDService,
    get_svd_service,
    check_local_ai_available,
    SVD_DIMENSIONS,
)

__all__ = [
    # GPU Management
    "GPUManager",
    "GPUInfo",
    "gpu_manager",
    "check_cuda_available",
    # Model Management
    "ModelManager",
    "model_manager",
    "get_model_manager",
    "MODEL_CONFIGS",
    # SVD Service
    "SVDService",
    "get_svd_service",
    "check_local_ai_available",
    "SVD_DIMENSIONS",
]
