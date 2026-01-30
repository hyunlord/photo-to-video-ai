import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from typing import BinaryIO, Optional
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class S3Client:
    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        self.bucket = settings.S3_BUCKET

    async def upload_file(
        self,
        file_obj: BinaryIO,
        object_name: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to S3
        Returns the URL of the uploaded file
        """
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type

            self.client.upload_fileobj(
                file_obj,
                self.bucket,
                object_name,
                ExtraArgs=extra_args
            )

            # Generate URL
            url = f"{settings.S3_ENDPOINT}/{self.bucket}/{object_name}"
            logger.info(f"Uploaded file to {url}")
            return url

        except ClientError as e:
            logger.error(f"Error uploading file: {e}")
            raise

    async def download_file(self, object_name: str, file_path: str):
        """Download a file from S3"""
        try:
            self.client.download_file(self.bucket, object_name, file_path)
            logger.info(f"Downloaded {object_name} to {file_path}")
        except ClientError as e:
            logger.error(f"Error downloading file: {e}")
            raise

    async def delete_file(self, object_name: str):
        """Delete a file from S3"""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=object_name)
            logger.info(f"Deleted {object_name}")
        except ClientError as e:
            logger.error(f"Error deleting file: {e}")
            raise

    async def get_presigned_url(self, object_name: str, expiration: int = 3600) -> str:
        """Generate a presigned URL for downloading"""
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': object_name},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise

s3_client = S3Client()
