from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
import uuid
import logging

from app.models import get_db, Job, Project, Photo
from app.workers.tasks import generate_video_task
from app.services.storage.s3_client import s3_client
from app.api.routes.websocket import broadcast_job_update
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

class GenerateVideoRequest(BaseModel):
    project_id: str
    settings: dict

@router.post("/")
async def start_generation(
    request: GenerateVideoRequest,
    db: Session = Depends(get_db)
):
    """
    Start video generation for a project

    Creates a job and queues it for processing
    """
    project_id = UUID(request.project_id)

    # Validate project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check if enough photos
    photo_count = db.query(Photo).filter(Photo.project_id == project_id).count()
    if photo_count < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 photos are required to generate a video"
        )

    # Create job
    job = Job(
        id=uuid.uuid4(),
        project_id=project_id,
        status="pending",
        progress=0,
        mode=request.settings.get("mode", "cloud"),
        model_name=request.settings.get("cloudService")
                   if request.settings.get("mode") == "cloud"
                   else request.settings.get("localModel")
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Queue task
    try:
        generate_video_task.delay(
            str(job.id),
            str(project_id),
            request.settings
        )

        logger.info(f"Queued video generation job {job.id}")

        # Broadcast job started
        await broadcast_job_update(
            str(job.id),
            str(project_id),
            "started",
            timestamp=job.created_at.isoformat()
        )

        return {
            "job_id": str(job.id),
            "status": job.status,
            "message": "Video generation started"
        }

    except Exception as e:
        logger.error(f"Failed to queue job: {e}")
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {str(e)}")

@router.get("/{job_id}/status")
async def get_job_status(
    job_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get the status of a generation job
    """
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": str(job.id),
        "status": job.status,
        "progress": job.progress,
        "result_path": job.result_path,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None
    }

@router.get("/{job_id}/video")
async def download_video(
    job_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Download the generated video
    """
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready. Current status: {job.status}"
        )

    if not job.result_path:
        raise HTTPException(status_code=404, detail="Video file not found")

    # Generate presigned URL for download
    try:
        url = await s3_client.get_presigned_url(job.result_path, expiration=3600)
        return {"download_url": url}

    except Exception as e:
        logger.error(f"Failed to generate download URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate download URL")

@router.delete("/{job_id}")
async def cancel_job(
    job_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Cancel a running job
    """
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}"
        )

    # Update job status
    job.status = "failed"
    job.error_message = "Cancelled by user"
    db.commit()

    # TODO: Actually cancel the Celery task
    # from app.workers.celery_app import celery_app
    # celery_app.control.revoke(str(job.id), terminate=True)

    # Broadcast cancellation
    await broadcast_job_update(
        str(job.id),
        str(job.project_id),
        "failed",
        error_message="Cancelled by user"
    )

    return {"message": "Job cancelled"}
