from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://photouser:photopass@localhost:5432/photovideo"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Storage
    S3_BUCKET: str = "photo-to-video"
    S3_ENDPOINT: str = "http://localhost:9000"
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin"

    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 100
    MAX_PHOTOS_PER_PROJECT: int = 20
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/jpg", "image/webp"]

    # AI Services
    RUNWAY_API_KEY: Optional[str] = None
    PIKA_API_KEY: Optional[str] = None
    STABILITY_API_KEY: Optional[str] = None

    # Local Models
    MODELS_DIR: str = "./models"
    ENABLE_LOCAL_MODELS: bool = True
    GPU_DEVICE: str = "cuda:0"

    # Processing
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Video Settings
    DEFAULT_FPS: int = 30
    DEFAULT_RESOLUTION: str = "1024x1024"
    VIDEO_CODEC: str = "libx264"
    VIDEO_CRF: int = 23  # Quality: 0-51, lower is better

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
