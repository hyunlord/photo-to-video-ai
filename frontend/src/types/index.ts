export interface Photo {
  id: string;
  file_path: string;
  thumbnail_path: string;
  order_index: number;
  metadata: {
    width: number;
    height: number;
    format: string;
    mode: string;
    size_bytes: number;
  };
}

export interface Project {
  id: string;
  name: string;
  settings: AnimationSettings;
  created_at: string;
  updated_at: string;
  photos?: Photo[];
}

export interface AnimationSettings {
  motionIntensity?: number;  // 0-100
  transitionSpeed?: 'slow' | 'medium' | 'fast';
  durationPerPhoto?: number;  // seconds
  mode?: 'cloud' | 'local';
  cloudService?: 'stability' | 'runway' | 'pika';
  localModel?: 'svd' | 'animatediff';
  fps?: number;
  transitionType?: 'fade' | 'morph' | 'zoom';
}

export interface Job {
  id: string;
  project_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  mode: 'cloud' | 'local';
  model_name: string;
  result_path?: string;
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

export interface UploadProgress {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'completed' | 'error';
  error?: string;
}
