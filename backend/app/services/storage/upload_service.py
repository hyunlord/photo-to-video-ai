from fastapi import UploadFile, HTTPException
from PIL import Image
import io
import uuid
from typing import Tuple
import logging
from app.config import settings
from app.services.storage.s3_client import s3_client

logger = logging.getLogger(__name__)

class UploadService:
    @staticmethod
    def validate_image(file: UploadFile) -> None:
        """Validate uploaded image file"""
        # Check content type
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(settings.ALLOWED_IMAGE_TYPES)}"
            )

        # Check file size (converted to bytes)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning

        max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file_size > max_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB"
            )

    @staticmethod
    async def create_thumbnail(
        image_data: bytes,
        size: Tuple[int, int] = (300, 300)
    ) -> bytes:
        """Create a thumbnail from image data"""
        try:
            img = Image.open(io.BytesIO(image_data))

            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background

            # Create thumbnail
            img.thumbnail(size, Image.Resampling.LANCZOS)

            # Save to bytes
            thumbnail_io = io.BytesIO()
            img.save(thumbnail_io, format='JPEG', quality=85)
            thumbnail_io.seek(0)

            return thumbnail_io.read()

        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            raise HTTPException(status_code=500, detail="Error creating thumbnail")

    @staticmethod
    async def upload_photo(
        file: UploadFile,
        project_id: uuid.UUID
    ) -> Tuple[str, str, dict]:
        """
        Upload a photo and its thumbnail to S3
        Returns: (file_path, thumbnail_path, metadata)
        """
        # Validate file
        UploadService.validate_image(file)

        # Read file data
        file_data = await file.read()

        # Get image metadata
        img = Image.open(io.BytesIO(file_data))
        metadata = {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_bytes": len(file_data)
        }

        # Generate unique filename
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        unique_filename = f"{uuid.uuid4()}.{file_ext}"
        object_name = f"projects/{project_id}/photos/{unique_filename}"

        # Upload original
        file_io = io.BytesIO(file_data)
        file_url = await s3_client.upload_file(
            file_io,
            object_name,
            content_type=file.content_type
        )

        # Create and upload thumbnail
        thumbnail_data = await UploadService.create_thumbnail(file_data)
        thumbnail_name = f"projects/{project_id}/thumbnails/{unique_filename}"
        thumbnail_io = io.BytesIO(thumbnail_data)
        thumbnail_url = await s3_client.upload_file(
            thumbnail_io,
            thumbnail_name,
            content_type="image/jpeg"
        )

        return object_name, thumbnail_name, metadata

upload_service = UploadService()
