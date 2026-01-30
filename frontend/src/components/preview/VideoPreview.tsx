"use client";

import { useEffect, useState } from 'react';
import { Job } from '@/types';

interface VideoPreviewProps {
  job: Job | null;
  onDownload?: (downloadUrl: string) => void;
}

export default function VideoPreview({ job, onDownload }: VideoPreviewProps) {
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);

  useEffect(() => {
    if (job?.status === 'completed' && job.result_path) {
      // Get download URL from S3
      const url = `http://localhost:9000/photo-to-video/${job.result_path}`;
      setDownloadUrl(url);
    }
  }, [job]);

  if (!job) {
    return (
      <div className="bg-card border border-border rounded-lg p-8 text-center">
        <p className="text-muted-foreground">
          No video generation in progress
        </p>
      </div>
    );
  }

  const getStatusColor = () => {
    switch (job.status) {
      case 'completed':
        return 'text-green-500';
      case 'failed':
        return 'text-destructive';
      case 'processing':
        return 'text-primary';
      default:
        return 'text-muted-foreground';
    }
  };

  const getStatusIcon = () => {
    if (job.status === 'completed') {
      return (
        <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      );
    }
    if (job.status === 'failed') {
      return (
        <svg className="w-6 h-6 text-destructive" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      );
    }
    if (job.status === 'processing') {
      return (
        <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      );
    }
    return null;
  };

  return (
    <div className="bg-card border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <h3 className="font-semibold capitalize">{job.status}</h3>
              <p className="text-xs text-muted-foreground">
                {job.mode === 'cloud' ? 'Cloud AI' : 'Local Model'} • {job.model_name}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      {(job.status === 'pending' || job.status === 'processing') && (
        <div className="p-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Generating...</span>
              <span className="font-semibold">{job.progress}%</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-3 overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-300 ease-out"
                style={{ width: `${job.progress}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {job.status === 'failed' && job.error_message && (
        <div className="p-4 bg-destructive/10">
          <p className="text-sm text-destructive">{job.error_message}</p>
        </div>
      )}

      {/* Video Player */}
      {job.status === 'completed' && downloadUrl && (
        <div className="relative">
          <video
            controls
            className="w-full aspect-video bg-black"
            src={downloadUrl}
          >
            Your browser does not support the video tag.
          </video>

          <div className="p-4 bg-muted/50">
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                <p>Video generation completed!</p>
                <p className="text-xs">
                  Started: {new Date(job.created_at).toLocaleString()}
                </p>
              </div>
              <a
                href={downloadUrl}
                download="generated-video.mp4"
                className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors inline-flex items-center gap-2"
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                  />
                </svg>
                Download
              </a>
            </div>
          </div>
        </div>
      )}

      {/* Job Details */}
      <div className="p-4 bg-muted/30 text-xs text-muted-foreground">
        <div className="flex items-center justify-between">
          <span>Job ID: {job.id.substring(0, 8)}...</span>
          {job.completed_at && (
            <span>Completed: {new Date(job.completed_at).toLocaleTimeString()}</span>
          )}
        </div>
      </div>
    </div>
  );
}
