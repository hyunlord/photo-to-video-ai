from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class BaseCloudAIService(ABC):
    """Base class for cloud AI service providers"""

    @abstractmethod
    async def generate_video(
        self,
        images: List[str],
        settings: Dict[str, Any]
    ) -> str:
        """
        Generate video from images

        Args:
            images: List of image file paths or URLs
            settings: Animation settings

        Returns:
            job_id: ID to track the generation job
        """
        pass

    @abstractmethod
    async def check_status(self, job_id: str) -> tuple[JobStatus, Optional[str], Optional[str]]:
        """
        Check the status of a generation job

        Args:
            job_id: The job ID returned from generate_video

        Returns:
            (status, video_url, error_message)
        """
        pass

    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job

        Args:
            job_id: The job ID to cancel

        Returns:
            success: True if cancelled successfully
        """
        pass
