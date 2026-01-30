"""
Video Processing Pipeline

Orchestrates the video generation process from photos to final video.
Supports both Cloud AI (Stability AI) and Local AI (Stable Video Diffusion).
"""

import os
import subprocess
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
import logging
from PIL import Image
import uuid

from app.config import settings
from app.services.storage.s3_client import s3_client

logger = logging.getLogger(__name__)


class VideoProcessingPipeline:
    """
    Pipeline for processing photos into animated video.

    Supports:
    - Cloud AI (Stability AI, Runway, Pika)
    - Local AI (Stable Video Diffusion)
    - FFmpeg fallback for demo/testing
    """

    def __init__(self, project_id: str, job_id: str):
        self.project_id = project_id
        self.job_id = job_id
        self.temp_dir = Path(f"/tmp/video_gen/{job_id}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def process(
        self,
        photos: List[Dict[str, Any]],
        animation_settings: Dict[str, Any],
        progress_callback: Callable[[int, str], None] = None
    ) -> str:
        """
        Main processing pipeline.

        Args:
            photos: List of photo objects with file_path
            animation_settings: Animation settings (mode, motionIntensity, fps, etc.)
            progress_callback: Function to call with (progress_percentage, message)

        Returns:
            S3 path to the generated video file
        """
        try:
            # Step 1: Download photos from S3
            if progress_callback:
                progress_callback(10, "Downloading photos...")
            image_paths = await self._download_photos(photos)

            # Step 2: Generate animated clips
            mode = animation_settings.get("mode", "cloud")

            if mode == "cloud":
                if progress_callback:
                    progress_callback(20, "Generating animations with Cloud AI...")
                video_clips = await self._generate_with_cloud_ai(
                    image_paths, animation_settings, progress_callback
                )
            else:
                if progress_callback:
                    progress_callback(20, "Generating animations with Local AI...")
                video_clips = await self._generate_with_local_ai(
                    image_paths, animation_settings, progress_callback
                )

            # Step 3: Apply transitions
            if progress_callback:
                progress_callback(70, "Applying transitions...")
            final_clips = await self._apply_transitions(video_clips, animation_settings)

            # Step 4: Concatenate videos
            if progress_callback:
                progress_callback(80, "Merging video clips...")
            output_path = await self._concatenate_videos(final_clips, animation_settings)

            # Step 5: Upload to S3
            if progress_callback:
                progress_callback(90, "Uploading video...")
            video_url = await self._upload_result(output_path)

            # Cleanup
            self._cleanup()

            if progress_callback:
                progress_callback(100, "Completed!")

            return video_url

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._cleanup()
            raise

    async def _download_photos(self, photos: List[Dict[str, Any]]) -> List[str]:
        """Download photos from S3 to local temp directory."""
        image_paths = []

        for i, photo in enumerate(photos):
            file_path = photo["file_path"]
            local_path = self.temp_dir / f"photo_{i}.jpg"

            await s3_client.download_file(file_path, str(local_path))
            image_paths.append(str(local_path))

        logger.info(f"Downloaded {len(image_paths)} photos")
        return image_paths

    async def _generate_with_cloud_ai(
        self,
        image_paths: List[str],
        animation_settings: Dict[str, Any],
        progress_callback: Callable[[int, str], None] = None
    ) -> List[str]:
        """
        Generate video clips using Cloud AI service (Stability AI).

        Falls back to FFmpeg zoom effect if API key is not configured.
        """
        from app.services.cloud_ai.stability import (
            StabilityAIService,
            StabilityAPIAuthError,
            check_stability_available
        )

        cloud_service_name = animation_settings.get("cloudService", "stability")
        duration = animation_settings.get("durationPerPhoto", 4.0)
        fps = animation_settings.get("fps", 30)
        motion_intensity = animation_settings.get("motionIntensity", 70)

        video_clips = []
        total_images = len(image_paths)

        # Check if Stability AI is available
        availability = check_stability_available()

        if not availability["available"]:
            logger.warning("Stability AI not available (no API key), using FFmpeg fallback")
            return await self._generate_with_ffmpeg_fallback(
                image_paths, animation_settings, progress_callback
            )

        # Initialize Stability AI service
        try:
            stability_service = StabilityAIService()
        except StabilityAPIAuthError as e:
            logger.warning(f"Stability AI auth error: {e}, using FFmpeg fallback")
            return await self._generate_with_ffmpeg_fallback(
                image_paths, animation_settings, progress_callback
            )

        # Generate video for each image
        for i, image_path in enumerate(image_paths):
            # Calculate progress: 20% start + up to 50% for all images
            base_progress = 20 + int((i / total_images) * 50)

            if progress_callback:
                progress_callback(base_progress, f"Generating AI animation {i+1}/{total_images}...")

            # Per-image progress callback
            def image_progress_callback(pct: int, msg: str):
                if progress_callback:
                    image_portion = 50 / total_images
                    adjusted_progress = base_progress + int((pct / 100) * image_portion)
                    progress_callback(adjusted_progress, f"Photo {i+1}/{total_images}: {msg}")

            output_path = self.temp_dir / f"clip_{i}.mp4"

            try:
                # Generate video using Stability AI
                video_bytes = await stability_service.generate_video_with_retry(
                    image_path=image_path,
                    motion_intensity=motion_intensity,
                    seed=i,  # Different seed per image for variety
                    progress_callback=image_progress_callback
                )

                # Save video to temp file
                with open(output_path, 'wb') as f:
                    f.write(video_bytes)

                video_clips.append(str(output_path))
                logger.info(f"Generated AI clip {i+1}/{total_images}: {output_path}")

            except Exception as e:
                logger.error(f"Failed to generate clip {i+1} with Cloud AI: {e}")

                # Fallback to FFmpeg zoom effect for this image
                logger.warning(f"Falling back to FFmpeg for image {i+1}")
                await self._create_video_from_image(
                    image_path, str(output_path), duration, fps
                )
                video_clips.append(str(output_path))

        return video_clips

    async def _generate_with_local_ai(
        self,
        image_paths: List[str],
        animation_settings: Dict[str, Any],
        progress_callback: Callable[[int, str], None] = None
    ) -> List[str]:
        """
        Generate video clips using Local AI model (Stable Video Diffusion).

        Falls back to Cloud AI or FFmpeg if local AI is not available.
        """
        # Check if local AI is enabled and available
        if not settings.ENABLE_LOCAL_MODELS:
            logger.warning("Local AI models disabled, falling back to Cloud AI")
            return await self._generate_with_cloud_ai(
                image_paths, animation_settings, progress_callback
            )

        try:
            from app.services.local_ai import get_svd_service, check_local_ai_available

            # Check availability
            status = check_local_ai_available()
            if not status["available"]:
                logger.warning(f"Local AI not available: {status}, falling back to Cloud AI")
                if progress_callback:
                    progress_callback(20, "Local AI unavailable, using Cloud AI...")
                return await self._generate_with_cloud_ai(
                    image_paths, animation_settings, progress_callback
                )

            # Get settings
            model_name = animation_settings.get("localModel", "svd-xt-1.1")
            motion_intensity = animation_settings.get("motionIntensity", 70) / 100.0
            fps = animation_settings.get("fps", 30)
            duration = animation_settings.get("durationPerPhoto", 4.0)

            # Initialize service
            svd_service = get_svd_service(model_name)

            video_clips = []
            total_images = len(image_paths)

            for i, image_path in enumerate(image_paths):
                if progress_callback:
                    base_progress = 20 + int((i / total_images) * 50)
                    progress_callback(base_progress, f"Generating clip {i+1}/{total_images}...")

                output_path = self.temp_dir / f"clip_{i}.mp4"

                # Per-image progress callback
                def image_progress(pct: int, msg: str):
                    if progress_callback:
                        image_start = 20 + int((i / total_images) * 50)
                        image_range = 50 / total_images
                        overall_pct = image_start + int((pct / 100) * image_range)
                        progress_callback(overall_pct, f"Image {i+1}: {msg}")

                try:
                    result = svd_service.generate_video(
                        image_path=image_path,
                        output_path=str(output_path),
                        motion_intensity=motion_intensity,
                        fps=fps,
                        duration_seconds=duration,
                        progress_callback=image_progress
                    )
                    video_clips.append(result)

                except Exception as e:
                    logger.error(f"Failed to generate clip {i} with Local AI: {e}")

                    # Fall back to FFmpeg zoom effect
                    logger.info(f"Falling back to FFmpeg for image {i}")
                    await self._create_video_from_image(
                        image_path, str(output_path), duration, fps
                    )
                    video_clips.append(str(output_path))

            return video_clips

        except ImportError:
            logger.warning("Local AI module not available, falling back to Cloud AI")
            return await self._generate_with_cloud_ai(
                image_paths, animation_settings, progress_callback
            )

    async def _generate_with_ffmpeg_fallback(
        self,
        image_paths: List[str],
        animation_settings: Dict[str, Any],
        progress_callback: Callable[[int, str], None] = None
    ) -> List[str]:
        """
        Generate video clips using FFmpeg zoom effect (fallback/demo mode).
        """
        duration = animation_settings.get("durationPerPhoto", 4.0)
        fps = animation_settings.get("fps", 30)

        video_clips = []
        total_images = len(image_paths)

        for i, image_path in enumerate(image_paths):
            if progress_callback:
                progress = 20 + int((i / total_images) * 50)
                progress_callback(progress, f"Creating animation {i+1}/{total_images} (demo mode)...")

            output_path = self.temp_dir / f"clip_{i}.mp4"

            await self._create_video_from_image(
                image_path, str(output_path), duration, fps
            )

            video_clips.append(str(output_path))

        return video_clips

    async def _create_video_from_image(
        self,
        image_path: str,
        output_path: str,
        duration: float = 4.0,
        fps: int = 30
    ):
        """
        Create a video from a single image with zoom/pan effect using FFmpeg.

        This is the fallback method when AI services are not available.
        """
        # Simple zoom in effect
        zoom_filter = (
            f"scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            f"zoompan=z='min(zoom+0.0015,1.5)':d={int(duration*fps)}:s=1920x1080:fps={fps}"
        )

        cmd = [
            'ffmpeg',
            '-loop', '1',
            '-i', image_path,
            '-vf', zoom_filter,
            '-t', str(duration),
            '-c:v', settings.VIDEO_CODEC,
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]

        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            logger.error(f"FFmpeg error: {process.stderr}")
            raise RuntimeError(f"Failed to create video from image: {process.stderr}")

        logger.debug(f"Created video from image: {output_path}")

    async def _apply_transitions(
        self,
        video_clips: List[str],
        animation_settings: Dict[str, Any]
    ) -> List[str]:
        """
        Apply transitions between video clips.

        Currently returns clips as-is. Future implementation would add:
        - Crossfade transitions
        - Morph transitions
        - Zoom transitions
        """
        transition_type = animation_settings.get("transitionType", "none")
        transition_speed = animation_settings.get("transitionSpeed", "medium")

        if transition_type == "none" or len(video_clips) <= 1:
            return video_clips

        # TODO: Implement actual transitions
        # For now, return clips as-is
        logger.debug(f"Transitions not yet implemented, returning {len(video_clips)} clips as-is")
        return video_clips

    async def _concatenate_videos(
        self,
        video_clips: List[str],
        animation_settings: Dict[str, Any]
    ) -> str:
        """
        Concatenate multiple video clips into one final video.
        """
        if len(video_clips) == 1:
            return video_clips[0]

        output_path = self.temp_dir / "final_output.mp4"

        # Create concat file for ffmpeg
        concat_file = self.temp_dir / "concat.txt"
        with open(concat_file, 'w') as f:
            for clip in video_clips:
                f.write(f"file '{clip}'\n")

        # Concatenate videos
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            '-y',
            str(output_path)
        ]

        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            logger.error(f"FFmpeg concat error: {process.stderr}")
            raise RuntimeError(f"Failed to concatenate videos: {process.stderr}")

        logger.info(f"Concatenated {len(video_clips)} clips into {output_path}")
        return str(output_path)

    async def _upload_result(self, video_path: str) -> str:
        """Upload final video to S3."""
        object_name = f"projects/{self.project_id}/videos/{self.job_id}.mp4"

        with open(video_path, 'rb') as f:
            url = await s3_client.upload_file(
                f,
                object_name,
                content_type="video/mp4"
            )

        logger.info(f"Uploaded video to S3: {object_name}")
        return object_name

    def _cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
