from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.models import get_db, Project, Photo
from app.services.storage.upload_service import upload_service
from app.config import settings

router = APIRouter()

@router.post("/projects/{project_id}/photos")
async def upload_photos(
    project_id: UUID,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload photos to a project"""
    # Check project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check number of photos limit
    existing_photos_count = db.query(Photo).filter(Photo.project_id == project_id).count()
    if existing_photos_count + len(files) > settings.MAX_PHOTOS_PER_PROJECT:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.MAX_PHOTOS_PER_PROJECT} photos per project"
        )

    uploaded_photos = []

    try:
        for idx, file in enumerate(files):
            # Upload photo and thumbnail
            file_path, thumbnail_path, metadata = await upload_service.upload_photo(
                file, project_id
            )

            # Create database entry
            photo = Photo(
                project_id=project_id,
                file_path=file_path,
                thumbnail_path=thumbnail_path,
                order_index=existing_photos_count + idx,
                metadata=metadata
            )
            db.add(photo)
            db.flush()  # Get the ID without committing

            uploaded_photos.append({
                "id": str(photo.id),
                "file_path": file_path,
                "thumbnail_path": thumbnail_path,
                "order_index": photo.order_index,
                "metadata": metadata
            })

        db.commit()

        return {
            "message": f"Successfully uploaded {len(files)} photos",
            "photos": uploaded_photos
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.delete("/projects/{project_id}/photos/{photo_id}")
async def delete_photo(
    project_id: UUID,
    photo_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a photo from a project"""
    photo = db.query(Photo).filter(
        Photo.id == photo_id,
        Photo.project_id == project_id
    ).first()

    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    # Delete from S3
    try:
        from app.services.storage.s3_client import s3_client
        await s3_client.delete_file(photo.file_path)
        if photo.thumbnail_path:
            await s3_client.delete_file(photo.thumbnail_path)
    except Exception as e:
        # Log error but continue with database deletion
        print(f"Error deleting from S3: {e}")

    # Delete from database
    db.delete(photo)
    db.commit()

    return {"message": "Photo deleted successfully"}

@router.put("/projects/{project_id}/photos/order")
async def reorder_photos(
    project_id: UUID,
    photo_order: List[dict],  # [{"id": "uuid", "order_index": 0}, ...]
    db: Session = Depends(get_db)
):
    """Update the order of photos in a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        for item in photo_order:
            photo_id = UUID(item["id"])
            new_index = item["order_index"]

            photo = db.query(Photo).filter(
                Photo.id == photo_id,
                Photo.project_id == project_id
            ).first()

            if photo:
                photo.order_index = new_index

        db.commit()
        return {"message": "Photo order updated successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update order: {str(e)}")
