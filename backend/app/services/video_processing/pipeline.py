import os
import subprocess
from typing import List, Dict, Any, Callable
from pathlib import Path
import logging
from PIL import Image
import uuid

from app.config import settings
from app.services.cloud_ai.stability import get_cloud_service
from app.services.storage.s3_client import s3_client

logger = logging.getLogger(__name__)

class VideoProcessingPipeline:
    """
    Pipeline for processing photos into animated video
    """

    def __init__(self, project_id: str, job_id: str):
        self.project_id = project_id
        self.job_id = job_id
        self.temp_dir = Path(f"/tmp/video_gen/{job_id}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def process(
        self,
        photos: List[Dict[str, Any]],
        settings: Dict[str, Any],
        progress_callback: Callable[[int, str], None] = None
    ) -> str:
        """
        Main processing pipeline

        Args:
            photos: List of photo objects with file_path
            settings: Animation settings
            progress_callback: Function to call with (progress_percentage, message)

        Returns:
            Path to the generated video file
        """
        try:
            # Step 1: Download photos from S3
            if progress_callback:
                progress_callback(10, "Downloading photos...")
            image_paths = await self._download_photos(photos)

            # Step 2: Generate animated clips
            mode = settings.get("mode", "cloud")

            if mode == "cloud":
                if progress_callback:
                    progress_callback(20, "Generating animations with Cloud AI...")
                video_clips = await self._generate_with_cloud_ai(
                    image_paths, settings, progress_callback
                )
            else:
                if progress_callback:
                    progress_callback(20, "Generating animations with Local AI...")
                video_clips = await self._generate_with_local_ai(
                    image_paths, settings, progress_callback
                )

            # Step 3: Apply transitions
            if progress_callback:
                progress_callback(70, "Applying transitions...")
            final_clips = await self._apply_transitions(video_clips, settings)

            # Step 4: Concatenate videos
            if progress_callback:
                progress_callback(80, "Merging video clips...")
            output_path = await self._concatenate_videos(final_clips, settings)

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
        """Download photos from S3 to local temp directory"""
        image_paths = []

        for i, photo in enumerate(photos):
            file_path = photo["file_path"]
            local_path = self.temp_dir / f"photo_{i}.jpg"

            await s3_client.download_file(file_path, str(local_path))
            image_paths.append(str(local_path))

        return image_paths

    async def _generate_with_cloud_ai(
        self,
        image_paths: List[str],
        settings: Dict[str, Any],
        progress_callback: Callable[[int, str], None] = None
    ) -> List[str]:
        """
        Generate video clips using cloud AI service

        For now, this is a simplified version that creates static video clips
        from images. In production, this would call the actual AI service.
        """
        cloud_service_name = settings.get("cloudService", "stability")
        duration = settings.get("durationPerPhoto", 4.0)
        fps = settings.get("fps", 30)

        video_clips = []

        for i, image_path in enumerate(image_paths):
            if progress_callback:
                progress = 20 + int((i / len(image_paths)) * 50)
                progress_callback(progress, f"Animating photo {i+1}/{len(image_paths)}...")

            # For demo: create a simple video from image using ffmpeg
            # In production, this would use the cloud AI service
            output_path = self.temp_dir / f"clip_{i}.mp4"

            # Create video from image with zoom effect
            await self._create_video_from_image(
                image_path,
                str(output_path),
                duration=duration,
                fps=fps
            )

            video_clips.append(str(output_path))

        return video_clips

    async def _generate_with_local_ai(
        self,
        image_paths: List[str],
        settings: Dict[str, Any],
        progress_callback: Callable[[int, str], None] = None
    ) -> List[str]:
        """
        Generate video clips using local AI model

        This would use Stable Video Diffusion or AnimateDiff locally
        For now, falls back to the same method as cloud
        """
        # TODO: Implement actual local AI model inference
        # This would load the model and generate frames

        return await self._generate_with_cloud_ai(
            image_paths, settings, progress_callback
        )

    async def _create_video_from_image(
        self,
        image_path: str,
        output_path: str,
        duration: float = 4.0,
        fps: int = 30
    ):
        """
        Create a video from a single image with zoom/pan effect using FFmpeg
        """
        # Simple zoom in effect
        cmd = [
            'ffmpeg',
            '-loop', '1',
            '-i', image_path,
            '-vf', f'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,zoompan=z=\'min(zoom+0.0015,1.5)\':d={int(duration*fps)}:s=1920x1080:fps={fps}',
            '-t', str(duration),
            '-c:v', settings.VIDEO_CODEC,
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]

        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if process.returncode != 0:
            logger.error(f"FFmpeg error: {process.stderr}")
            raise RuntimeError(f"Failed to create video from image: {process.stderr}")

    async def _apply_transitions(
        self,
        video_clips: List[str],
        settings: Dict[str, Any]
    ) -> List[str]:
        """
        Apply transitions between video clips

        For demo, returns clips as-is
        In production, would add crossfade, morph, or other transitions
        """
        # TODO: Implement actual transitions
        # transition_type = settings.get("transitionType", "fade")
        # transition_speed = settings.get("transitionSpeed", "medium")

        return video_clips

    async def _concatenate_videos(
        self,
        video_clips: List[str],
        settings: Dict[str, Any]
    ) -> str:
        """
        Concatenate multiple video clips into one final video
        """
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

        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if process.returncode != 0:
            logger.error(f"FFmpeg concat error: {process.stderr}")
            raise RuntimeError(f"Failed to concatenate videos: {process.stderr}")

        return str(output_path)

    async def _upload_result(self, video_path: str) -> str:
        """Upload final video to S3"""
        object_name = f"projects/{self.project_id}/videos/{self.job_id}.mp4"

        with open(video_path, 'rb') as f:
            url = await s3_client.upload_file(
                f,
                object_name,
                content_type="video/mp4"
            )

        return object_name

    def _cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
