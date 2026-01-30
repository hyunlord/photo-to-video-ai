from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from typing import List
from uuid import UUID

from app.models import get_db, Project
from app.schemas.project import ProjectCreate, ProjectResponse

router = APIRouter()

@router.post("/", response_model=ProjectResponse, status_code=201)
async def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db)
):
    """Create a new project"""
    db_project = Project(name=project.name, settings=project.settings or {})
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@router.get("/{project_id}")
async def get_project(
    project_id: UUID,
    db: Session = Depends(get_db)
):
    """Get project details with photos"""
    project = db.query(Project).options(
        joinedload(Project.photos)
    ).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Convert to dict to include photos
    return {
        "id": str(project.id),
        "name": project.name,
        "settings": project.settings,
        "created_at": project.created_at.isoformat(),
        "updated_at": project.updated_at.isoformat(),
        "photos": [
            {
                "id": str(photo.id),
                "file_path": photo.file_path,
                "thumbnail_path": photo.thumbnail_path,
                "order_index": photo.order_index,
                "metadata": photo.metadata
            }
            for photo in sorted(project.photos, key=lambda p: p.order_index)
        ]
    }

@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(project)
    db.commit()
    return None
