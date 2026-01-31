"""
Wan 2.1 FLF2V Service
여러 사진 사이의 자연스러운 전환 영상 생성

First+Last Frame Video Generation for natural transitions between photos.
"""

import torch
from pathlib import Path
from typing import List, Optional, Callable, Union
from PIL import Image
import logging
import tempfile
import subprocess
import shutil

from app.config import settings

logger = logging.getLogger(__name__)


class WanService:
    """
    Wan 2.1 First-Last Frame to Video Service

    여러 이미지를 순서대로 받아 각 쌍 사이의 전환 영상 생성
    """

    MODEL_ID = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"

    # 품질 프리셋
    QUALITY_PRESETS = {
        "fast": {"num_inference_steps": 30, "guidance_scale": 5.0},
        "balanced": {"num_inference_steps": 50, "guidance_scale": 6.0},
        "quality": {"num_inference_steps": 75, "guidance_scale": 7.0},
    }

    # 기본 네거티브 프롬프트
    DEFAULT_NEGATIVE_PROMPT = (
        "blurry, low quality, distorted face, deformed, "
        "disfigured, bad anatomy, watermark, text, ugly, "
        "duplicate, morbid, mutilated, poorly drawn face"
    )

    def __init__(self, models_dir: Optional[str] = None):
        self.pipe = None
        self.device = None
        self.models_dir = Path(models_dir or settings.MODELS_DIR)
        self._initialized = False

    def _get_device(self) -> torch.device:
        """Get best available device (CUDA > MPS > CPU)"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _get_device_type(self) -> str:
        """Get device type string"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_memory_gb(self) -> float:
        """Get available GPU/Unified memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        elif torch.backends.mps.is_available():
            # MPS uses Unified Memory - get system RAM as estimate
            try:
                import psutil
                return psutil.virtual_memory().total / 1e9
            except ImportError:
                return 64.0  # Default assumption for Apple Silicon
        return 0

    def _clear_memory_cache(self):
        """Clear GPU/MPS memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

    def _get_vram_gb(self) -> float:
        """Get available VRAM in GB (alias for _get_memory_gb)"""
        return self._get_memory_gb()

    def load_model(self):
        """모델 로드 (CUDA/MPS/CPU 지원)"""
        if self._initialized:
            return

        try:
            from diffusers import WanPipeline
        except ImportError:
            # Fallback: diffusers might use different class name
            from diffusers import DiffusionPipeline as WanPipeline

        self.device = self._get_device()
        device_type = self._get_device_type()
        model_path = self.models_dir / "wan" / "wan2.1_flf2v_720p"

        # 메모리에 따라 precision 결정
        memory_gb = self._get_memory_gb()

        # MPS는 FP8을 지원하지 않음
        if device_type == "mps":
            dtype = torch.float16
            logger.info(f"MPS device detected, using FP16 (Unified Memory: {memory_gb:.1f}GB)")
        elif memory_gb < 12:
            logger.info(f"Low VRAM ({memory_gb:.1f}GB), using FP8 quantization")
            dtype = torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16
        else:
            dtype = torch.float16

        logger.info(f"Loading Wan 2.1 from {model_path}")

        self.pipe = WanPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype
        )
        self.pipe.to(self.device)

        # 메모리 최적화 (MPS에서는 일부 기능 미지원)
        if device_type != "mps":
            if hasattr(self.pipe, 'enable_model_cpu_offload'):
                self.pipe.enable_model_cpu_offload()
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()

        self._initialized = True
        logger.info(f"Wan 2.1 loaded on {self.device} ({memory_gb:.1f}GB {'Unified Memory' if device_type == 'mps' else 'VRAM'})")

    def generate_single_transition(
        self,
        first_image: Image.Image,
        last_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        quality_preset: str = "balanced",
        duration: float = 5.0,
        fps: int = 24,
    ) -> bytes:
        """
        두 이미지 사이의 전환 영상 생성

        Args:
            first_image: 시작 이미지
            last_image: 끝 이미지
            prompt: 영상 설명 프롬프트
            negative_prompt: 네거티브 프롬프트
            quality_preset: fast/balanced/quality
            duration: 영상 길이 (초)
            fps: 프레임 레이트

        Returns:
            영상 바이트 데이터
        """
        self.load_model()

        # 720p 리사이즈
        target_size = (1280, 720)
        first_image = first_image.resize(target_size, Image.LANCZOS)
        last_image = last_image.resize(target_size, Image.LANCZOS)

        params = self.QUALITY_PRESETS.get(quality_preset, self.QUALITY_PRESETS["balanced"])
        num_frames = int(duration * fps)

        # 네거티브 프롬프트
        if not negative_prompt:
            negative_prompt = self.DEFAULT_NEGATIVE_PROMPT

        logger.info(f"Generating transition: prompt='{prompt[:50]}...', frames={num_frames}, preset={quality_preset}")

        output = self.pipe(
            first_frame=first_image,
            last_frame=last_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
        )

        # 영상 바이트로 변환
        from diffusers.utils import export_to_video

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            export_to_video(output.frames[0], temp_path, fps=fps)
            return Path(temp_path).read_bytes()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def generate_multi_transition(
        self,
        images: List[Image.Image],
        prompts: Union[str, List[str]],
        quality_preset: str = "balanced",
        duration_per_transition: float = 5.0,
        fps: int = 24,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> bytes:
        """
        여러 이미지 사이의 연속 전환 영상 생성

        Args:
            images: 순서대로 정렬된 이미지 리스트 [img1, img2, img3, ...]
            prompts: 각 전환에 대한 프롬프트 (len = len(images) - 1)
                     또는 단일 프롬프트 (모든 전환에 동일 적용)
            quality_preset: fast/balanced/quality
            duration_per_transition: 각 전환 영상 길이 (초)
            fps: 프레임 레이트
            progress_callback: 진행률 콜백 (percent, message)

        Returns:
            최종 연결된 영상 바이트
        """
        if len(images) < 2:
            raise ValueError("At least 2 images required")

        # 프롬프트 처리
        num_transitions = len(images) - 1
        if isinstance(prompts, str):
            prompts = [prompts] * num_transitions
        elif len(prompts) < num_transitions:
            # 부족하면 마지막 프롬프트 반복
            prompts = list(prompts) + [prompts[-1]] * (num_transitions - len(prompts))

        temp_dir = Path(tempfile.mkdtemp())
        clip_paths = []

        try:
            # 각 전환 생성
            for i in range(num_transitions):
                if progress_callback:
                    pct = int((i / num_transitions) * 80)
                    progress_callback(pct, f"Generating transition {i+1}/{num_transitions}...")

                logger.info(f"Generating transition {i+1}/{num_transitions}")

                clip_bytes = self.generate_single_transition(
                    first_image=images[i],
                    last_image=images[i + 1],
                    prompt=prompts[i],
                    quality_preset=quality_preset,
                    duration=duration_per_transition,
                    fps=fps,
                )

                clip_path = temp_dir / f"clip_{i:03d}.mp4"
                clip_path.write_bytes(clip_bytes)
                clip_paths.append(clip_path)

            # FFmpeg로 연결
            if progress_callback:
                progress_callback(85, "Concatenating clips...")

            output_bytes = self._concatenate_clips(clip_paths, temp_dir)

            if progress_callback:
                progress_callback(100, "Complete!")

            return output_bytes

        finally:
            # 정리
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _concatenate_clips(self, clip_paths: List[Path], temp_dir: Path) -> bytes:
        """FFmpeg로 클립 연결"""
        concat_file = temp_dir / "concat.txt"
        with open(concat_file, 'w') as f:
            for clip_path in clip_paths:
                f.write(f"file '{clip_path}'\n")

        output_path = temp_dir / "final.mp4"

        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            '-y',
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"Failed to concatenate videos: {result.stderr}")

        return output_path.read_bytes()

    def unload_model(self):
        """메모리 해제 (CUDA/MPS 지원)"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._initialized = False

            self._clear_memory_cache()

            logger.info("Wan 2.1 model unloaded")


# Singleton
_wan_service: Optional[WanService] = None


def get_wan_service() -> WanService:
    """Get singleton WanService instance"""
    global _wan_service
    if _wan_service is None:
        _wan_service = WanService()
    return _wan_service


def check_wan_available() -> dict:
    """
    Wan 서비스 가용성 확인 (CUDA/MPS 지원)

    Returns:
        Dict with availability status and details
    """
    model_dir = Path(settings.MODELS_DIR) / "wan" / "wan2.1_flf2v_720p"
    model_exists = model_dir.exists() and any(model_dir.iterdir()) if model_dir.exists() else False

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    gpu_available = cuda_available or mps_available

    # 디바이스 타입 결정
    if cuda_available:
        device_type = "cuda"
    elif mps_available:
        device_type = "mps"
    else:
        device_type = "cpu"

    # 메모리 확인
    memory_gb = 0
    if cuda_available:
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    elif mps_available:
        # MPS uses Unified Memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / 1e9
        except ImportError:
            memory_gb = 64.0  # Default assumption for Apple Silicon

    # 추천 프리셋 결정
    if memory_gb < 12:
        recommended_preset = "fast"
    elif memory_gb < 24:
        recommended_preset = "balanced"
    else:
        recommended_preset = "quality"

    return {
        "available": model_exists and gpu_available and settings.ENABLE_LOCAL_MODELS,
        "model_exists": model_exists,
        "model_dir": str(model_dir),
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "gpu_available": gpu_available,
        "device_type": device_type,
        "memory_gb": round(memory_gb, 1),
        "vram_gb": round(memory_gb, 1),  # Alias for backward compatibility
        "recommended_preset": recommended_preset,
        "enabled": settings.ENABLE_LOCAL_MODELS,
    }
