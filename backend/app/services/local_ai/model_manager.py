"""
Model Manager for Local AI

Handles loading, caching, and management of Stable Video Diffusion models.
Supports lazy loading to minimize memory usage.
"""

import logging
import gc
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Lock

from app.config import settings
from app.services.local_ai.gpu_manager import gpu_manager

logger = logging.getLogger(__name__)


# Model configurations
MODEL_CONFIGS = {
    "svd": {
        "repo_id": "stabilityai/stable-video-diffusion-img2vid",
        "num_frames": 14,
        "decode_chunk_size": 8,
        "motion_bucket_range": (1, 127),
        "vram_required_gb": 6,
    },
    "svd-xt": {
        "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
        "num_frames": 25,
        "decode_chunk_size": 8,
        "motion_bucket_range": (1, 255),
        "vram_required_gb": 8,
    },
    "svd-xt-1.1": {
        "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        "num_frames": 25,
        "decode_chunk_size": 8,
        "motion_bucket_range": (1, 255),
        "vram_required_gb": 8,
    },
}


class ModelManager:
    """
    Manages SVD model loading and caching.

    Features:
    - Lazy loading (models loaded on first use)
    - Automatic unloading when memory is low
    - Thread-safe model access
    - Support for multiple model variants
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern for model manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._loaded_models: Dict[str, Any] = {}
        self._model_lock = Lock()
        self._model_dir = Path(settings.LOCAL_MODEL_PATH) if settings.LOCAL_MODEL_PATH else None

        logger.info(f"ModelManager initialized. Model dir: {self._model_dir}")

    def _get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the local path for a model."""
        if self._model_dir and (self._model_dir / model_name).exists():
            return self._model_dir / model_name
        return None

    def _get_model_source(self, model_name: str) -> str:
        """Get the model source (local path or HuggingFace repo)."""
        local_path = self._get_model_path(model_name)
        if local_path:
            logger.info(f"Using local model: {local_path}")
            return str(local_path)

        # Fall back to HuggingFace
        if model_name in MODEL_CONFIGS:
            repo_id = MODEL_CONFIGS[model_name]["repo_id"]
            logger.info(f"Using HuggingFace model: {repo_id}")
            return repo_id

        raise ValueError(f"Unknown model: {model_name}")

    def load_model(self, model_name: str = "svd-xt-1.1", force_reload: bool = False) -> Any:
        """
        Load a model (lazy loading with caching).

        Args:
            model_name: Name of the model to load
            force_reload: Force reload even if cached

        Returns:
            Loaded model pipeline
        """
        with self._model_lock:
            # Return cached model if available
            if model_name in self._loaded_models and not force_reload:
                logger.debug(f"Using cached model: {model_name}")
                return self._loaded_models[model_name]

            # Check GPU memory
            config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["svd-xt-1.1"])
            required_gb = config["vram_required_gb"]

            memory_check = gpu_manager.check_memory_for_model(required_gb)
            if not memory_check["available"]:
                # Try to free up memory
                self.unload_all_models()
                gpu_manager.clear_cache()

                # Check again
                memory_check = gpu_manager.check_memory_for_model(required_gb)
                if not memory_check["available"]:
                    raise RuntimeError(
                        f"Insufficient GPU memory for {model_name}. "
                        f"Need {required_gb}GB, have {memory_check.get('available_gb', 0):.1f}GB. "
                        f"Try closing other applications or using Cloud AI instead."
                    )

            logger.info(f"Loading model: {model_name}")

            try:
                import torch
                from diffusers import StableVideoDiffusionPipeline
                from diffusers.utils import export_to_video

                model_source = self._get_model_source(model_name)

                # Load pipeline with optimizations
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    model_source,
                    torch_dtype=torch.float16,
                    variant="fp16",
                )

                # Move to GPU
                pipe = pipe.to(gpu_manager.device)

                # Enable memory optimizations
                pipe.enable_model_cpu_offload()

                # Try to enable xformers if available
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xformers memory efficient attention enabled")
                except Exception:
                    logger.debug("xformers not available, using default attention")

                # Cache the model
                self._loaded_models[model_name] = {
                    "pipeline": pipe,
                    "config": config,
                    "export_to_video": export_to_video,
                }

                logger.info(f"Model {model_name} loaded successfully")
                return self._loaded_models[model_name]

            except ImportError as e:
                logger.error(f"Missing dependency for local AI: {e}")
                raise RuntimeError(
                    f"Local AI dependencies not installed. "
                    f"Install with: pip install diffusers transformers accelerate"
                ) from e

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

    def unload_model(self, model_name: str):
        """Unload a specific model from memory."""
        with self._model_lock:
            if model_name in self._loaded_models:
                del self._loaded_models[model_name]
                gc.collect()
                gpu_manager.clear_cache()
                logger.info(f"Model {model_name} unloaded")

    def unload_all_models(self):
        """Unload all models from memory."""
        with self._model_lock:
            self._loaded_models.clear()
            gc.collect()
            gpu_manager.clear_cache()
            logger.info("All models unloaded")

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded."""
        return model_name in self._loaded_models

    def get_loaded_models(self) -> list:
        """Get list of currently loaded models."""
        return list(self._loaded_models.keys())

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a model."""
        return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["svd-xt-1.1"])

    def check_model_available(self, model_name: str) -> Dict[str, Any]:
        """
        Check if a model is available (either locally or downloadable).

        Returns:
            Dict with 'available', 'source', and details
        """
        local_path = self._get_model_path(model_name)

        if local_path:
            return {
                "available": True,
                "source": "local",
                "path": str(local_path),
                "model_name": model_name,
            }

        if model_name in MODEL_CONFIGS:
            return {
                "available": True,
                "source": "huggingface",
                "repo_id": MODEL_CONFIGS[model_name]["repo_id"],
                "model_name": model_name,
                "note": "Will be downloaded on first use (~9GB)",
            }

        return {
            "available": False,
            "model_name": model_name,
            "error": f"Unknown model: {model_name}",
            "supported_models": list(MODEL_CONFIGS.keys()),
        }


# Global instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get the model manager instance."""
    return model_manager
