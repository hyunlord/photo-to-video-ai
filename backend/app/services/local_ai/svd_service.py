"""
Stable Video Diffusion Service

Generates videos from images using local SVD models.
Handles image preprocessing, inference, and video encoding.
"""

import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from PIL import Image
import tempfile

from app.config import settings
from app.services.local_ai.gpu_manager import gpu_manager
from app.services.local_ai.model_manager import model_manager, MODEL_CONFIGS

logger = logging.getLogger(__name__)


# SVD supported dimensions (width, height)
SVD_DIMENSIONS = (1024, 576)  # 16:9 landscape


class SVDService:
    """
    Service for generating videos using Stable Video Diffusion.

    Features:
    - Image preprocessing (resize, crop)
    - SVD inference with progress callbacks
    - Frame to video conversion
    - Memory-efficient generation
    """

    def __init__(self, model_name: str = "svd-xt-1.1"):
        """
        Initialize SVD service.

        Args:
            model_name: Name of the SVD model to use
        """
        self.model_name = model_name
        self.config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["svd-xt-1.1"])
        self._model = None

    def _ensure_model_loaded(self):
        """Ensure the model is loaded before inference."""
        if self._model is None:
            self._model = model_manager.load_model(self.model_name)
        return self._model

    def _preprocess_image(self, image_path: str) -> Image.Image:
        """
        Preprocess image for SVD inference.

        Resizes and center-crops to 1024x576.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed PIL Image
        """
        img = Image.open(image_path).convert("RGB")

        target_width, target_height = SVD_DIMENSIONS
        target_aspect = target_width / target_height

        # Get current dimensions
        width, height = img.size
        current_aspect = width / height

        # Resize to fit target aspect ratio
        if current_aspect > target_aspect:
            # Image is wider - fit by height
            new_height = target_height
            new_width = int(new_height * current_aspect)
        else:
            # Image is taller - fit by width
            new_width = target_width
            new_height = int(new_width / current_aspect)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop to target dimensions
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        img = img.crop((left, top, right, bottom))

        logger.debug(f"Preprocessed image: {width}x{height} -> {target_width}x{target_height}")
        return img

    def _convert_frames_to_video(
        self,
        frames: list,
        output_path: str,
        fps: int = 7,
        target_fps: int = 30
    ) -> str:
        """
        Convert frames to video using FFmpeg.

        Args:
            frames: List of PIL Images or numpy arrays
            output_path: Path for output video
            fps: Original frame rate from SVD
            target_fps: Target frame rate with interpolation

        Returns:
            Path to output video
        """
        import numpy as np

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Save frames
            for i, frame in enumerate(frames):
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame)
                frame_path = temp_dir / f"frame_{i:04d}.png"
                frame.save(frame_path, "PNG")

            # Use FFmpeg to create video with frame interpolation
            input_pattern = str(temp_dir / "frame_%04d.png")

            # Build FFmpeg command
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-framerate", str(fps),  # Input framerate
                "-i", input_pattern,
                "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                "-c:v", settings.VIDEO_CODEC,
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                str(output_path)
            ]

            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.warning(f"FFmpeg interpolation failed: {result.stderr}")
                # Fall back to simple conversion without interpolation
                cmd_simple = [
                    "ffmpeg",
                    "-y",
                    "-framerate", str(fps),
                    "-i", input_pattern,
                    "-c:v", settings.VIDEO_CODEC,
                    "-pix_fmt", "yuv420p",
                    "-r", str(target_fps),  # Output framerate
                    str(output_path)
                ]
                subprocess.run(cmd_simple, capture_output=True, check=True)

        logger.debug(f"Created video: {output_path}")
        return output_path

    def generate_video(
        self,
        image_path: str,
        output_path: str,
        motion_intensity: float = 0.7,
        fps: int = 30,
        duration_seconds: float = 4.0,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> str:
        """
        Generate a video from an image using SVD.

        Args:
            image_path: Path to input image
            output_path: Path for output video
            motion_intensity: Motion amount (0.0-1.0)
            fps: Target FPS for output video
            duration_seconds: Desired duration (limited by model)
            seed: Random seed for reproducibility
            progress_callback: Function to call with (progress_percent, message)

        Returns:
            Path to generated video
        """
        import torch

        if progress_callback:
            progress_callback(0, "Initializing...")

        # Check GPU availability
        if not gpu_manager.is_available:
            raise RuntimeError(
                "GPU not available for local AI. "
                "Please use Cloud AI mode or install CUDA."
            )

        if progress_callback:
            progress_callback(5, "Loading model...")

        # Load model
        model_data = self._ensure_model_loaded()
        pipeline = model_data["pipeline"]
        config = model_data["config"]

        if progress_callback:
            progress_callback(15, "Preprocessing image...")

        # Preprocess image
        image = self._preprocess_image(image_path)

        # Calculate motion bucket ID (1-255 range mapped from 0-1)
        min_motion, max_motion = config["motion_bucket_range"]
        motion_bucket_id = int(min_motion + motion_intensity * (max_motion - min_motion))
        motion_bucket_id = max(min_motion, min(max_motion, motion_bucket_id))

        if progress_callback:
            progress_callback(20, "Generating frames with AI...")

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=gpu_manager.device).manual_seed(seed)

        # Generate frames
        with gpu_manager.memory_efficient_context():
            try:
                # Run inference
                with torch.inference_mode():
                    output = pipeline(
                        image,
                        decode_chunk_size=config["decode_chunk_size"],
                        motion_bucket_id=motion_bucket_id,
                        noise_aug_strength=0.02,  # Slight noise for better motion
                        num_frames=config["num_frames"],
                        generator=generator,
                    )

                frames = output.frames[0]  # Get frames from output

                if progress_callback:
                    progress_callback(80, "Converting to video...")

            except torch.cuda.OutOfMemoryError:
                gpu_manager.clear_cache()
                raise RuntimeError(
                    "GPU out of memory during inference. "
                    "Try closing other applications or using Cloud AI."
                )

        # SVD generates at ~7fps, interpolate to target fps
        svd_fps = 7
        video_path = self._convert_frames_to_video(
            frames,
            output_path,
            fps=svd_fps,
            target_fps=fps
        )

        if progress_callback:
            progress_callback(100, "Complete!")

        logger.info(f"Generated video: {video_path}")
        return video_path

    def estimate_generation_time(self) -> Dict[str, Any]:
        """
        Estimate video generation time based on hardware.

        Returns:
            Dict with estimated time and factors
        """
        gpu_info = gpu_manager.get_gpu_info()

        if not gpu_info:
            return {
                "estimated_seconds": None,
                "note": "GPU not available",
            }

        # Rough estimates based on GPU capability
        # RTX 3090: ~30s, RTX 3080: ~45s, RTX 3070: ~60s
        base_time = 60  # seconds

        # Adjust based on VRAM (more VRAM usually means faster GPU)
        if gpu_info.total_memory_gb >= 20:
            multiplier = 0.5  # High-end GPU
        elif gpu_info.total_memory_gb >= 10:
            multiplier = 0.75  # Mid-range
        else:
            multiplier = 1.0  # Entry level

        estimated = int(base_time * multiplier)

        return {
            "estimated_seconds": estimated,
            "gpu_name": gpu_info.name,
            "vram_gb": gpu_info.total_memory_gb,
            "note": f"Estimated {estimated}s per image on {gpu_info.name}",
        }

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "model_name": self.model_name,
            "model_loaded": model_manager.is_model_loaded(self.model_name),
            "gpu_available": gpu_manager.is_available,
            "gpu_status": gpu_manager.get_status(),
            "supported_dimensions": SVD_DIMENSIONS,
            "output_frames": self.config["num_frames"],
        }


# Service factory
_svd_services: Dict[str, SVDService] = {}


def get_svd_service(model_name: str = "svd-xt-1.1") -> SVDService:
    """
    Get or create an SVD service instance.

    Args:
        model_name: Name of the model to use

    Returns:
        SVDService instance
    """
    if model_name not in _svd_services:
        _svd_services[model_name] = SVDService(model_name)
    return _svd_services[model_name]


def check_local_ai_available() -> Dict[str, Any]:
    """
    Check if local AI is available and ready.

    Returns:
        Dict with 'available' status and details
    """
    # Check if local AI is enabled
    if not settings.ENABLE_LOCAL_MODELS:
        return {
            "available": False,
            "reason": "Local AI models disabled in settings",
            "setting": "ENABLE_LOCAL_MODELS",
        }

    # Check GPU
    gpu_status = gpu_manager.get_status()
    if not gpu_status["available"]:
        return {
            "available": False,
            "reason": "No GPU available",
            "gpu_status": gpu_status,
        }

    # Check memory
    memory_check = gpu_manager.check_memory_for_model(6.0)  # Minimum for SVD
    if not memory_check["available"]:
        return {
            "available": False,
            "reason": "Insufficient GPU memory",
            "memory_status": memory_check,
        }

    # Check dependencies
    try:
        import torch
        import diffusers
        deps_ok = True
    except ImportError as e:
        return {
            "available": False,
            "reason": f"Missing dependency: {e}",
            "install_hint": "pip install torch diffusers transformers accelerate",
        }

    return {
        "available": True,
        "gpu_status": gpu_status,
        "memory_status": memory_check,
        "default_model": "svd-xt-1.1",
    }
