import { useState, useCallback } from 'react';
import { generationApi } from '@/lib/api-client';
import { Job } from '@/types';
import { useWebSocket } from './useWebSocket';

export function useVideoGeneration(projectId: string) {
  const [currentJob, setCurrentJob] = useState<Job | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // WebSocket for real-time updates
  const { isConnected } = useWebSocket(projectId, (message) => {
    if (message.type === 'job.started') {
      setIsGenerating(true);
      setCurrentJob((prev) => prev ? { ...prev, status: 'processing' } : null);
    } else if (message.type === 'job.progress') {
      setCurrentJob((prev) =>
        prev
          ? { ...prev, progress: message.progress || 0 }
          : null
      );
    } else if (message.type === 'job.completed') {
      setCurrentJob((prev) =>
        prev
          ? {
              ...prev,
              status: 'completed',
              progress: 100,
              result_path: message.result_path,
            }
          : null
      );
      setIsGenerating(false);
    } else if (message.type === 'job.failed') {
      setCurrentJob((prev) =>
        prev
          ? {
              ...prev,
              status: 'failed',
              error_message: message.error_message,
            }
          : null
      );
      setIsGenerating(false);
      setError(message.error_message || 'Video generation failed');
    }
  });

  const startGeneration = useCallback(
    async (settings: any) => {
      try {
        setIsGenerating(true);
        setError(null);

        const response = await generationApi.start(projectId, settings);

        setCurrentJob({
          id: response.job_id,
          project_id: projectId,
          status: 'pending',
          progress: 0,
          mode: settings.mode || 'cloud',
          model_name:
            settings.mode === 'cloud' ? settings.cloudService : settings.localModel,
          created_at: new Date().toISOString(),
        });

        return response.job_id;
      } catch (err: any) {
        const errorMsg = err.response?.data?.detail || 'Failed to start generation';
        setError(errorMsg);
        setIsGenerating(false);
        throw new Error(errorMsg);
      }
    },
    [projectId]
  );

  const checkStatus = useCallback(
    async (jobId: string) => {
      try {
        const status = await generationApi.getStatus(jobId);
        setCurrentJob(status);
        return status;
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to check status');
        throw err;
      }
    },
    []
  );

  const downloadVideo = useCallback(
    async (jobId: string) => {
      try {
        const response = await generationApi.getVideo(jobId);
        return response.download_url;
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to download video');
        throw err;
      }
    },
    []
  );

  return {
    currentJob,
    isGenerating,
    error,
    isWebSocketConnected: isConnected,
    startGeneration,
    checkStatus,
    downloadVideo,
  };
}
