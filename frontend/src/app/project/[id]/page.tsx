"use client";

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import PhotoUploader from '@/components/upload/PhotoUploader';
import PhotoOrdering from '@/components/editor/PhotoOrdering';
import AnimationSettings from '@/components/editor/AnimationSettings';
import VideoPreview from '@/components/preview/VideoPreview';
import { projectApi } from '@/lib/api-client';
import { Project, Photo, AnimationSettings as Settings } from '@/types';
import { useVideoGeneration } from '@/hooks/useVideoGeneration';

export default function ProjectPage() {
  const params = useParams();
  const projectId = params.id as string;

  const [project, setProject] = useState<Project | null>(null);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [settings, setSettings] = useState<Settings>({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Video generation hook
  const {
    currentJob,
    isGenerating,
    error: generationError,
    isWebSocketConnected,
    startGeneration,
  } = useVideoGeneration(projectId);

  useEffect(() => {
    loadProject();
  }, [projectId]);

  const loadProject = async () => {
    try {
      setIsLoading(true);
      const data = await projectApi.get(projectId);
      setProject(data);
      setSettings(data.settings || {});

      // Photos will be loaded with the project
      if (data.photos) {
        setPhotos(data.photos.sort((a: Photo, b: Photo) => a.order_index - b.order_index));
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load project');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadComplete = (newPhotos: Photo[]) => {
    setPhotos((prev) => [...prev, ...newPhotos].sort((a, b) => a.order_index - b.order_index));
  };

  const handleReorder = (reorderedPhotos: Photo[]) => {
    setPhotos(reorderedPhotos);
  };

  const handleSettingsChange = async (newSettings: Settings) => {
    setSettings(newSettings);
    // Auto-save settings could be implemented here
  };

  const handleGenerateVideo = async () => {
    try {
      await startGeneration(settings);
    } catch (err) {
      console.error('Failed to start generation:', err);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading project...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-destructive mb-4">{error}</p>
          <button
            onClick={loadProject}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card sticky top-0 z-40">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">{project?.name}</h1>
              <p className="text-sm text-muted-foreground">
                {photos.length} photo{photos.length !== 1 ? 's' : ''} uploaded
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => window.location.href = '/'}
                className="px-4 py-2 border border-border rounded-lg hover:bg-accent transition-colors"
              >
                Back
              </button>
              <button
                onClick={handleGenerateVideo}
                disabled={photos.length < 2 || isGenerating}
                className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isGenerating ? (
                  <>
                    <div className="w-4 h-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
                    Generating...
                  </>
                ) : (
                  'Generate Video'
                )}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Column - Photos */}
          <div className="lg:col-span-2 space-y-8">
            {/* Upload Section */}
            <section>
              <h2 className="text-xl font-semibold mb-4">Upload Photos</h2>
              <PhotoUploader
                projectId={projectId}
                onUploadComplete={handleUploadComplete}
                maxFiles={20}
                maxSizeMB={100}
              />
            </section>

            {/* Photo Ordering Section */}
            {photos.length > 0 && (
              <section>
                <h2 className="text-xl font-semibold mb-4">Arrange Photos</h2>
                <PhotoOrdering
                  projectId={projectId}
                  photos={photos}
                  onReorder={handleReorder}
                  onDelete={(photoId) => {
                    setPhotos((prev) => prev.filter((p) => p.id !== photoId));
                  }}
                />
              </section>
            )}

            {/* Instructions */}
            {photos.length === 0 && (
              <div className="bg-muted rounded-lg p-6">
                <h3 className="font-semibold mb-2">Getting Started</h3>
                <ol className="text-sm text-muted-foreground space-y-2 list-decimal list-inside">
                  <li>Upload at least 2 photos of a person</li>
                  <li>Arrange photos in the order you want</li>
                  <li>Adjust animation settings on the right</li>
                  <li>Click "Generate Video" to create your animated video</li>
                </ol>
              </div>
            )}
          </div>

          {/* Right Column - Settings */}
          <div className="lg:col-span-1">
            <div className="sticky top-24 space-y-6">
              <AnimationSettings
                settings={settings}
                onChange={handleSettingsChange}
              />

              {/* Video Preview */}
              {(currentJob || isGenerating) && (
                <div>
                  <h3 className="text-lg font-semibold mb-4">Video Generation</h3>
                  <VideoPreview job={currentJob} />
                  {generationError && (
                    <div className="mt-4 p-4 bg-destructive/10 border border-destructive rounded-lg">
                      <p className="text-sm text-destructive">{generationError}</p>
                    </div>
                  )}
                </div>
              )}

              {/* Info Card */}
              <div className="p-4 bg-muted rounded-lg">
                <h4 className="text-sm font-semibold mb-2">Tips</h4>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>• Use photos with similar lighting and angles</li>
                  <li>• Higher motion intensity = more dynamic movement</li>
                  <li>• Cloud API is faster but requires API key</li>
                  <li>• Local models are free but need GPU</li>
                  {isWebSocketConnected && (
                    <li className="text-green-500">• ✓ Real-time updates connected</li>
                  )}
                </ul>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
