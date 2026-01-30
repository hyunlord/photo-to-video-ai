from celery import Task
from sqlalchemy.orm import Session
import logging
from datetime import datetime
from typing import Dict, Any
import redis

from app.workers.celery_app import celery_app
from app.models import get_db, Job, Photo, Project
from app.services.video_processing.pipeline import VideoProcessingPipeline
from app.config import settings

logger = logging.getLogger(__name__)

# Redis client for progress updates
redis_client = redis.from_url(settings.REDIS_URL)

class VideoGenerationTask(Task):
    """Custom task class with database session"""

    def __call__(self, *args, **kwargs):
        # Get database session
        db = next(get_db())
        try:
            return self.run(*args, db=db, **kwargs)
        finally:
            db.close()

@celery_app.task(
    bind=True,
    base=VideoGenerationTask,
    name='app.workers.tasks.generate_video_task',
    max_retries=3,
    default_retry_delay=60
)
def generate_video_task(
    self,
    job_id: str,
    project_id: str,
    settings_dict: Dict[str, Any],
    db: Session = None
):
    """
    Celery task to generate video from photos

    Args:
        job_id: UUID of the job
        project_id: UUID of the project
        settings_dict: Animation settings
        db: Database session (injected by custom task class)
    """
    logger.info(f"Starting video generation task for job {job_id}")

    # Get job from database
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        logger.error(f"Job {job_id} not found")
        return

    # Get photos
    photos = db.query(Photo).filter(
        Photo.project_id == project_id
    ).order_by(Photo.order_index).all()

    if len(photos) < 2:
        job.status = "failed"
        job.error_message = "At least 2 photos are required"
        db.commit()
        logger.error(f"Not enough photos for job {job_id}")
        return

    # Convert photos to dict format
    photos_data = [
        {
            "id": str(photo.id),
            "file_path": photo.file_path,
            "order_index": photo.order_index
        }
        for photo in photos
    ]

    # Progress callback to update Redis and database
    def progress_callback(percentage: int, message: str):
        # Update Redis for WebSocket broadcast
        redis_client.setex(
            f"job_progress:{job_id}",
            3600,  # Expire after 1 hour
            f"{percentage}|{message}"
        )

        # Update database
        job.progress = percentage
        db.commit()

        logger.info(f"Job {job_id}: {percentage}% - {message}")

    try:
        # Update job status to processing
        job.status = "processing"
        job.progress = 0
        db.commit()

        # Create pipeline
        pipeline = VideoProcessingPipeline(project_id, job_id)

        # Process video (this is synchronous for now)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result_path = loop.run_until_complete(
            pipeline.process(photos_data, settings_dict, progress_callback)
        )

        # Update job with result
        job.status = "completed"
        job.progress = 100
        job.result_path = result_path
        job.completed_at = datetime.utcnow()
        db.commit()

        logger.info(f"Job {job_id} completed successfully")

        # Notify via Redis
        redis_client.publish(
            f"job_updates:{job_id}",
            f"completed|{result_path}"
        )

        return {"status": "completed", "result_path": result_path}

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)

        # Update job status
        job.status = "failed"
        job.error_message = str(e)
        db.commit()

        # Notify via Redis
        redis_client.publish(
            f"job_updates:{job_id}",
            f"failed|{str(e)}"
        )

        # Retry if retries remaining
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        return {"status": "failed", "error": str(e)}
