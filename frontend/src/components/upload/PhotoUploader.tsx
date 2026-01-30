"use client";

import { useState, useCallback } from 'react';
import { photoApi } from '@/lib/api-client';
import { Photo, UploadProgress } from '@/types';

interface PhotoUploaderProps {
  projectId: string;
  onUploadComplete: (photos: Photo[]) => void;
  maxFiles?: number;
  maxSizeMB?: number;
}

export default function PhotoUploader({
  projectId,
  onUploadComplete,
  maxFiles = 20,
  maxSizeMB = 100,
}: PhotoUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([]);
  const [error, setError] = useState<string | null>(null);

  const validateFiles = (files: File[]): boolean => {
    // Check number of files
    if (files.length > maxFiles) {
      setError(`Maximum ${maxFiles} files allowed`);
      return false;
    }

    // Check file types and sizes
    const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
    for (const file of files) {
      if (!allowedTypes.includes(file.type)) {
        setError(`Invalid file type: ${file.name}. Only JPEG, PNG, and WebP are allowed.`);
        return false;
      }

      const sizeMB = file.size / (1024 * 1024);
      if (sizeMB > maxSizeMB) {
        setError(`File ${file.name} is too large. Maximum size: ${maxSizeMB}MB`);
        return false;
      }
    }

    setError(null);
    return true;
  };

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const fileArray = Array.from(files);

    if (!validateFiles(fileArray)) {
      return;
    }

    // Initialize upload progress
    const initialProgress: UploadProgress[] = fileArray.map(file => ({
      file,
      progress: 0,
      status: 'pending',
    }));
    setUploadProgress(initialProgress);

    try {
      // Update status to uploading
      setUploadProgress(prev =>
        prev.map(p => ({ ...p, status: 'uploading' as const }))
      );

      // Upload files
      const result = await photoApi.upload(projectId, fileArray);

      // Update status to completed
      setUploadProgress(prev =>
        prev.map(p => ({ ...p, status: 'completed' as const, progress: 100 }))
      );

      // Notify parent component
      if (result.photos) {
        onUploadComplete(result.photos);
      }

      // Clear progress after a delay
      setTimeout(() => {
        setUploadProgress([]);
      }, 2000);

    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed');
      setUploadProgress(prev =>
        prev.map(p => ({
          ...p,
          status: 'error' as const,
          error: err.message,
        }))
      );
    }
  }, [projectId, onUploadComplete, maxFiles, maxSizeMB]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFiles(files);
    }
  }, [handleFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFiles(files);
    }
    // Reset input value to allow uploading the same file again
    e.target.value = '';
  }, [handleFiles]);

  return (
    <div className="w-full">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          border-2 border-dashed rounded-lg p-12 text-center transition-colors
          ${isDragging
            ? 'border-primary bg-primary/10'
            : 'border-border hover:border-primary/50'
          }
        `}
      >
        <div className="flex flex-col items-center gap-4">
          <svg
            className="w-16 h-16 text-muted-foreground"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>

          <div>
            <p className="text-lg font-medium mb-2">
              Drag and drop your photos here
            </p>
            <p className="text-sm text-muted-foreground mb-4">
              or click to browse files
            </p>
            <p className="text-xs text-muted-foreground">
              Maximum {maxFiles} files, {maxSizeMB}MB each
              <br />
              Supported formats: JPEG, PNG, WebP
            </p>
          </div>

          <input
            type="file"
            accept="image/jpeg,image/png,image/jpg,image/webp"
            multiple
            onChange={handleFileInput}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            Choose Files
          </label>
        </div>
      </div>

      {error && (
        <div className="mt-4 p-4 bg-destructive/10 border border-destructive rounded-lg">
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      {uploadProgress.length > 0 && (
        <div className="mt-4 space-y-2">
          {uploadProgress.map((item, index) => (
            <div key={index} className="p-3 bg-card border border-border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium truncate flex-1">
                  {item.file.name}
                </span>
                <span className="text-xs text-muted-foreground ml-2">
                  {(item.file.size / (1024 * 1024)).toFixed(2)} MB
                </span>
              </div>
              <div className="w-full bg-secondary rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all ${
                    item.status === 'error'
                      ? 'bg-destructive'
                      : item.status === 'completed'
                      ? 'bg-green-500'
                      : 'bg-primary'
                  }`}
                  style={{ width: `${item.progress}%` }}
                />
              </div>
              {item.error && (
                <p className="text-xs text-destructive mt-1">{item.error}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
