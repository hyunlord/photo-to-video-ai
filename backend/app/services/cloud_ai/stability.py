"""
Stability AI Video Generation Service
Using Stable Video Diffusion API (v2beta)

API Documentation: https://platform.stability.ai/docs/api-reference
"""

import httpx
import asyncio
import io
import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from PIL import Image

from app.services.cloud_ai.base import BaseCloudAIService, JobStatus
from app.config import settings

logger = logging.getLogger(__name__)


# Custom exceptions for Stability AI
class StabilityAIError(Exception):
    """Base exception for Stability AI errors"""
    pass


class StabilityAPIAuthError(StabilityAIError):
    """Authentication failed"""
    pass


class StabilityAPIRateLimitError(StabilityAIError):
    """Rate limit exceeded"""
    pass


class StabilityAPIQuotaError(StabilityAIError):
    """Credit quota exceeded"""
    pass


class StabilityAPIInvalidImageError(StabilityAIError):
    """Invalid image format or dimensions"""
    pass


class StabilityAIService(BaseCloudAIService):
    """
    Stability AI Video Generation Service
    Using Stable Video Diffusion API

    Supported dimensions: 1024x576, 576x1024, 768x768
    Output: ~4 second video (25 frames at 6 fps)
    Pricing: $0.20 per 5 seconds of video
    """

    # API Configuration
    BASE_URL = "https://api.stability.ai/v2beta"
    IMAGE_TO_VIDEO_ENDPOINT = f"{BASE_URL}/image-to-video"
    RESULT_ENDPOINT = f"{BASE_URL}/image-to-video/result"

    # Supported image dimensions (width, height)
    SUPPORTED_DIMENSIONS = [(1024, 576), (576, 1024), (768, 768)]

    # Polling configuration
    POLL_INTERVAL = 10  # seconds between status checks
    MAX_POLL_TIME = 300  # 5 minutes max wait time
    MAX_RETRIES = 3

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.STABILITY_API_KEY

        # Allow initialization without API key for checking availability
        if self.api_key:
            # Validate API key format
            if not self.api_key.startswith("sk-"):
                logger.warning("API key format may be invalid - expected format: sk-...")

    def _check_api_key(self):
        """Ensure API key is available before making requests"""
        if not self.api_key:
            raise StabilityAPIAuthError(
                "Stability API key is required. "
                "Set STABILITY_API_KEY environment variable or pass api_key parameter."
            )

    def _get_headers(self, accept: str = "application/json") -> Dict[str, str]:
        """Get request headers with authentication"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": accept
        }

    def _parse_api_error(self, status_code: int, response_text: str) -> StabilityAIError:
        """Parse API error response into appropriate exception"""
        try:
            import json
            error_data = json.loads(response_text)
            message = error_data.get("message", response_text)
        except:
            message = response_text

        if status_code == 401:
            return StabilityAPIAuthError(f"Authentication failed: {message}")
        elif status_code == 403:
            return StabilityAPIQuotaError(f"Insufficient credits or forbidden: {message}")
        elif status_code == 429:
            return StabilityAPIRateLimitError(f"Rate limit exceeded: {message}")
        elif status_code == 400:
            return StabilityAPIInvalidImageError(f"Invalid request: {message}")
        else:
            return StabilityAIError(f"API error {status_code}: {message}")

    async def _prepare_image(self, image_path: str) -> bytes:
        """
        Prepare image for Stability AI API.
        Resizes to supported dimensions (1024x576, 576x1024, or 768x768).

        Args:
            image_path: Path to the image file

        Returns:
            Image data as PNG bytes
        """
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            width, height = img.size
            aspect_ratio = width / height

            # Determine best target dimensions based on aspect ratio
            if aspect_ratio > 1.5:  # Wide landscape (16:9 or wider)
                target = (1024, 576)
            elif aspect_ratio < 0.67:  # Tall portrait (9:16 or taller)
                target = (576, 1024)
            else:  # Close to square
                target = (768, 768)

            # Calculate crop box to match target aspect ratio
            target_ratio = target[0] / target[1]

            if aspect_ratio > target_ratio:
                # Image is wider than target, crop width
                new_width = int(height * target_ratio)
                left = (width - new_width) // 2
                img = img.crop((left, 0, left + new_width, height))
            elif aspect_ratio < target_ratio:
                # Image is taller than target, crop height
                new_height = int(width / target_ratio)
                top = (height - new_height) // 2
                img = img.crop((0, top, width, top + new_height))

            # Resize to exact target dimensions
            img_resized = img.resize(target, Image.Resampling.LANCZOS)

            # Convert to PNG bytes
            buffer = io.BytesIO()
            img_resized.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)

            logger.debug(f"Prepared image: {image_path} -> {target[0]}x{target[1]}")
            return buffer.read()

    async def generate_video_from_image(
        self,
        image_path: str,
        motion_intensity: int = 70,
        seed: int = 0
    ) -> str:
        """
        Submit video generation request for a single image.

        Args:
            image_path: Path to source image
            motion_intensity: Motion intensity (0-100), maps to motion_bucket_id
            seed: Random seed for reproducibility

        Returns:
            generation_id: ID to poll for results
        """
        self._check_api_key()

        # Prepare image
        image_data = await self._prepare_image(image_path)

        # Map motion_intensity (0-100) to motion_bucket_id (1-255)
        motion_bucket_id = max(1, min(255, int(motion_intensity * 2.55)))

        # cfg_scale: typically 1.0-2.0 for natural motion
        cfg_scale = 1.8

        logger.info(f"Submitting to Stability AI: motion_bucket_id={motion_bucket_id}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.IMAGE_TO_VIDEO_ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"image": ("image.png", image_data, "image/png")},
                data={
                    "seed": str(seed),
                    "cfg_scale": str(cfg_scale),
                    "motion_bucket_id": str(motion_bucket_id)
                }
            )

            if response.status_code != 200:
                raise self._parse_api_error(response.status_code, response.text)

            result = response.json()
            generation_id = result["id"]

            logger.info(f"Stability AI generation started: {generation_id}")
            return generation_id

    async def poll_for_result(
        self,
        generation_id: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> bytes:
        """
        Poll Stability AI API until video generation completes.

        Args:
            generation_id: ID from generate_video_from_image
            progress_callback: Optional callback for progress updates (percentage, message)

        Returns:
            Video bytes on success

        Raises:
            TimeoutError: If max poll time exceeded
            StabilityAIError: If generation fails
        """
        self._check_api_key()

        start_time = asyncio.get_event_loop().time()
        poll_count = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time

                if elapsed > self.MAX_POLL_TIME:
                    raise TimeoutError(
                        f"Generation {generation_id} timed out after {self.MAX_POLL_TIME}s"
                    )

                response = await client.get(
                    f"{self.RESULT_ENDPOINT}/{generation_id}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "video/*"
                    }
                )

                if response.status_code == 202:
                    # Still processing
                    poll_count += 1

                    if progress_callback:
                        # Estimate progress (0-95% over expected generation time)
                        estimated_progress = min(95, int(elapsed / 120 * 100))
                        progress_callback(
                            estimated_progress,
                            f"AI generating video... ({int(elapsed)}s)"
                        )

                    logger.debug(
                        f"Generation {generation_id} still processing "
                        f"(poll #{poll_count}, {int(elapsed)}s elapsed)"
                    )
                    await asyncio.sleep(self.POLL_INTERVAL)

                elif response.status_code == 200:
                    # Complete!
                    logger.info(f"Generation {generation_id} completed")

                    if progress_callback:
                        progress_callback(100, "Video generation complete")

                    return response.content

                else:
                    # Error
                    raise self._parse_api_error(response.status_code, response.text)

    async def generate_video_with_retry(
        self,
        image_path: str,
        motion_intensity: int = 70,
        seed: int = 0,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> bytes:
        """
        Generate video with automatic retry on failure.

        This is the main method to use for video generation.

        Args:
            image_path: Path to input image
            motion_intensity: Motion intensity (0-100)
            seed: Random seed
            progress_callback: Progress callback

        Returns:
            Video bytes
        """
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                if attempt > 0:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 10s, 20s, 40s
                    logger.info(f"Retry attempt {attempt + 1}/{self.MAX_RETRIES}, waiting {wait_time}s")

                    if progress_callback:
                        progress_callback(0, f"Retrying... (attempt {attempt + 1}/{self.MAX_RETRIES})")

                    await asyncio.sleep(wait_time)

                # Submit generation request
                if progress_callback:
                    progress_callback(5, "Submitting to Stability AI...")

                generation_id = await self.generate_video_from_image(
                    image_path, motion_intensity, seed
                )

                # Poll for result
                video_bytes = await self.poll_for_result(generation_id, progress_callback)

                return video_bytes

            except StabilityAPIRateLimitError as e:
                last_error = e
                logger.warning(f"Rate limit on attempt {attempt + 1}: {e}")

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                logger.warning(f"Network error on attempt {attempt + 1}: {e}")

            except StabilityAIError as e:
                # Check if it's a retryable server error (5xx)
                if "5" == str(e).split()[-1][0:1]:
                    last_error = e
                    logger.warning(f"Server error on attempt {attempt + 1}: {e}")
                else:
                    raise  # Non-retryable error

        raise StabilityAIError(f"Failed after {self.MAX_RETRIES} attempts: {last_error}")

    # =========================================================================
    # BaseCloudAIService interface implementation
    # =========================================================================

    async def generate_video(
        self,
        images: List[str],
        settings: Dict[str, Any]
    ) -> str:
        """
        Generate video from multiple images.

        Note: This starts the generation process. The actual video clips
        are generated in the video processing pipeline.
        """
        if not images:
            raise ValueError("At least one image is required")

        self._check_api_key()

        logger.info(f"Starting Stability AI video generation for {len(images)} images")

        # For tracking purposes, return the first image's generation ID
        # The pipeline handles the actual multi-image processing
        motion_intensity = settings.get("motionIntensity", 70)
        generation_id = await self.generate_video_from_image(images[0], motion_intensity)

        return generation_id

    async def check_status(
        self,
        job_id: str
    ) -> tuple[JobStatus, Optional[str], Optional[str]]:
        """
        Check status of Stability AI generation job.
        """
        self._check_api_key()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.RESULT_ENDPOINT}/{job_id}",
                headers=self._get_headers()
            )

            if response.status_code == 202:
                return JobStatus.PROCESSING, None, None
            elif response.status_code == 200:
                return JobStatus.COMPLETED, job_id, None
            else:
                error_msg = response.text
                return JobStatus.FAILED, None, error_msg

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a Stability AI job.

        Note: Stability AI doesn't have a cancel endpoint.
        Jobs complete automatically after ~2 minutes.
        """
        logger.info(f"Cancel requested for job {job_id} (not supported by Stability AI)")
        return True


# Factory function
def get_cloud_service(service_name: str = "stability") -> BaseCloudAIService:
    """
    Get cloud AI service instance.

    Args:
        service_name: Name of the service (stability, runway, pika)

    Returns:
        Instance of the cloud service
    """
    if service_name == "stability":
        return StabilityAIService()
    elif service_name == "runway":
        raise NotImplementedError("Runway service not yet implemented")
    elif service_name == "pika":
        raise NotImplementedError("Pika service not yet implemented")
    else:
        raise ValueError(f"Unknown service: {service_name}")


def check_stability_available() -> Dict[str, Any]:
    """
    Check if Stability AI service is available.

    Returns:
        Dict with availability status
    """
    api_key = settings.STABILITY_API_KEY

    return {
        "available": bool(api_key),
        "service": "stability",
        "has_api_key": bool(api_key),
        "api_key_preview": f"{api_key[:8]}..." if api_key and len(api_key) > 8 else None
    }
