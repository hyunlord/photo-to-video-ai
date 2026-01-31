"""
Video Generation Pipeline
사진 → 영상 생성의 전체 흐름 관리

Main pipeline for processing photos into video with AI-generated transitions.
"""

import io
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from PIL import Image
import logging
import shutil

from app.services.wan_service import get_wan_service, check_wan_available
from app.services.prompt_enhancer import get_prompt_enhancer
from app.services.storage.s3_client import s3_client

logger = logging.getLogger(__name__)


class VideoPipeline:
    """
    사진 → 영상 생성 파이프라인

    Input: N개의 사진 (순서대로) + 프롬프트 (선택)
    Output: 전환 영상

    Flow:
    1. Download photos from S3
    2. Prepare prompts (with optional AI enhancement)
    3. Generate transition videos using Wan 2.1
    4. Apply post-processing (optional)
    5. Upload result to S3
    """

    def __init__(self, project_id: str, job_id: str):
        self.project_id = project_id
        self.job_id = job_id
        self.temp_dir = Path(f"/tmp/video_gen/{job_id}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def process(
        self,
        photo_paths: List[str],
        settings: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> str:
        """
        메인 처리 파이프라인

        Args:
            photo_paths: S3에 저장된 사진 경로들 (순서대로)
            settings: {
                "prompt": str,              # 사용자 프롬프트 (선택)
                "enhancePrompt": bool,      # AI 프롬프트 보강 여부 (default: True)
                "enhanceMode": str,         # "none" / "append" / "auto" (default: "append")
                "qualityPreset": str,       # "fast" / "balanced" / "quality" (default: "balanced")
                "durationPerTransition": float,  # 각 전환 길이 초 (default: 5.0)
                "fps": int,                 # 프레임 레이트 (default: 24)
                "enablePostProcess": bool,  # 후처리 활성화 (default: False)
            }
            progress_callback: 진행률 콜백 (percent, message)

        Returns:
            생성된 영상의 S3 경로
        """
        try:
            # 1. 사진 다운로드
            if progress_callback:
                progress_callback(5, "Downloading photos...")

            images = await self._download_photos(photo_paths)

            if len(images) < 2:
                raise ValueError("At least 2 photos required for video generation")

            # 2. Wan 가용성 확인
            wan_status = check_wan_available()
            if not wan_status["available"]:
                raise RuntimeError(
                    f"Wan 2.1 not available: model_exists={wan_status['model_exists']}, "
                    f"cuda={wan_status['cuda_available']}, enabled={wan_status['enabled']}"
                )

            # 3. 프롬프트 처리
            if progress_callback:
                progress_callback(10, "Preparing prompts...")

            prompts = self._prepare_prompts(images, settings)

            # 4. 영상 생성
            if progress_callback:
                progress_callback(15, "Generating video with Wan 2.1...")

            wan_service = get_wan_service()

            # 진행률 콜백 래핑
            def wan_progress(pct: int, msg: str):
                if progress_callback:
                    # 15% ~ 85% 구간 매핑
                    adjusted = 15 + int(pct * 0.7)
                    progress_callback(adjusted, msg)

            video_bytes = wan_service.generate_multi_transition(
                images=images,
                prompts=prompts,
                quality_preset=settings.get("qualityPreset", wan_status["recommended_preset"]),
                duration_per_transition=settings.get("durationPerTransition", 5.0),
                fps=settings.get("fps", 24),
                progress_callback=wan_progress
            )

            # 5. 후처리 (선택)
            if settings.get("enablePostProcess", False):
                if progress_callback:
                    progress_callback(88, "Post-processing...")
                video_bytes = await self._post_process(video_bytes)

            # 6. S3 업로드
            if progress_callback:
                progress_callback(95, "Uploading result...")

            output_path = await self._upload_result(video_bytes)

            if progress_callback:
                progress_callback(100, "Complete!")

            logger.info(f"Pipeline completed: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            self._cleanup()

    async def _download_photos(self, photo_paths: List[str]) -> List[Image.Image]:
        """S3에서 사진 다운로드"""
        images = []

        for i, path in enumerate(photo_paths):
            local_path = self.temp_dir / f"photo_{i:03d}.jpg"

            try:
                await s3_client.download_file(path, str(local_path))
                img = Image.open(local_path).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.error(f"Failed to download photo {i}: {path}, error: {e}")
                raise

        logger.info(f"Downloaded {len(images)} photos")
        return images

    def _prepare_prompts(
        self,
        images: List[Image.Image],
        settings: Dict[str, Any]
    ) -> List[str]:
        """프롬프트 준비"""
        user_prompt = settings.get("prompt", "")
        enhance_prompt = settings.get("enhancePrompt", True)
        enhance_mode = settings.get("enhanceMode", "append")

        # 프롬프트 보강 비활성화시 mode를 none으로
        if not enhance_prompt:
            enhance_mode = "none"

        # auto 모드는 vision 모델 필요
        use_vision = enhance_mode == "auto"

        enhancer = get_prompt_enhancer(use_vision=use_vision)
        prompts = enhancer.generate_prompts_for_sequence(
            images=images,
            user_prompt=user_prompt,
            enhancement_mode=enhance_mode
        )

        logger.info(f"Prepared {len(prompts)} prompts (mode: {enhance_mode})")
        return prompts

    async def _post_process(self, video_bytes: bytes) -> bytes:
        """
        후처리 적용 (선택적)

        TODO: Real-ESRGAN, GFPGAN 등 통합
        """
        # 현재는 bypass
        logger.info("Post-processing skipped (not implemented)")
        return video_bytes

    async def _upload_result(self, video_bytes: bytes) -> str:
        """S3에 결과 업로드"""
        object_name = f"projects/{self.project_id}/videos/{self.job_id}.mp4"

        file_obj = io.BytesIO(video_bytes)

        url = await s3_client.upload_file(
            file_obj,
            object_name,
            content_type="video/mp4"
        )

        logger.info(f"Uploaded video: {object_name} ({len(video_bytes)} bytes)")
        return object_name

    def _cleanup(self):
        """임시 파일 정리"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temp dir: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def generate_video(
    project_id: str,
    job_id: str,
    photo_paths: List[str],
    settings: Dict[str, Any],
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> str:
    """
    편의 함수: 영상 생성 실행

    Args:
        project_id: 프로젝트 ID
        job_id: 작업 ID
        photo_paths: S3 사진 경로 리스트 (순서대로)
        settings: 생성 설정
        progress_callback: 진행률 콜백

    Returns:
        생성된 영상의 S3 경로
    """
    pipeline = VideoPipeline(project_id, job_id)
    return await pipeline.process(photo_paths, settings, progress_callback)
