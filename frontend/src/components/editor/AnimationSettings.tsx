"use client";

import { useState, useEffect } from 'react';
import { AnimationSettings as Settings } from '@/types';

interface AnimationSettingsProps {
  settings: Settings;
  onChange: (settings: Settings) => void;
}

export default function AnimationSettings({
  settings: initialSettings,
  onChange,
}: AnimationSettingsProps) {
  const [settings, setSettings] = useState<Settings>({
    motionIntensity: 70,
    transitionSpeed: 'medium',
    durationPerPhoto: 4,
    mode: 'cloud',
    cloudService: 'stability',
    localModel: 'svd',
    fps: 30,
    transitionType: 'fade',
    ...initialSettings,
  });

  useEffect(() => {
    onChange(settings);
  }, [settings, onChange]);

  const updateSetting = <K extends keyof Settings>(
    key: K,
    value: Settings[K]
  ) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="space-y-6 bg-card border border-border rounded-lg p-6">
      <h3 className="text-lg font-semibold mb-4">Animation Settings</h3>

      {/* AI Mode Selection */}
      <div className="space-y-3">
        <label className="block text-sm font-medium">AI Mode</label>
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => updateSetting('mode', 'cloud')}
            className={`p-4 rounded-lg border-2 transition-all ${
              settings.mode === 'cloud'
                ? 'border-primary bg-primary/10'
                : 'border-border hover:border-primary/50'
            }`}
          >
            <div className="text-sm font-semibold mb-1">Cloud API</div>
            <div className="text-xs text-muted-foreground">
              Fast, high quality
            </div>
          </button>
          <button
            onClick={() => updateSetting('mode', 'local')}
            className={`p-4 rounded-lg border-2 transition-all ${
              settings.mode === 'local'
                ? 'border-primary bg-primary/10'
                : 'border-border hover:border-primary/50'
            }`}
          >
            <div className="text-sm font-semibold mb-1">Local Model</div>
            <div className="text-xs text-muted-foreground">
              Free, requires GPU
            </div>
          </button>
        </div>
      </div>

      {/* Cloud Service Selection */}
      {settings.mode === 'cloud' && (
        <div className="space-y-3">
          <label className="block text-sm font-medium">Cloud Service</label>
          <select
            value={settings.cloudService}
            onChange={(e) =>
              updateSetting('cloudService', e.target.value as any)
            }
            className="w-full p-3 rounded-lg border border-border bg-background focus:border-primary focus:outline-none"
          >
            <option value="stability">Stability AI (Recommended)</option>
            <option value="runway">Runway Gen-3 (Premium)</option>
            <option value="pika">Pika Labs (Creative)</option>
          </select>
        </div>
      )}

      {/* Local Model Selection */}
      {settings.mode === 'local' && (
        <div className="space-y-3">
          <label className="block text-sm font-medium">Local Model</label>
          <select
            value={settings.localModel}
            onChange={(e) =>
              updateSetting('localModel', e.target.value as any)
            }
            className="w-full p-3 rounded-lg border border-border bg-background focus:border-primary focus:outline-none"
          >
            <option value="svd">Stable Video Diffusion (Recommended)</option>
            <option value="animatediff">AnimateDiff</option>
          </select>
        </div>
      )}

      {/* Motion Intensity */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="block text-sm font-medium">Motion Intensity</label>
          <span className="text-sm text-muted-foreground">
            {settings.motionIntensity}%
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          value={settings.motionIntensity}
          onChange={(e) =>
            updateSetting('motionIntensity', parseInt(e.target.value))
          }
          className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Subtle</span>
          <span>Dynamic</span>
        </div>
      </div>

      {/* Transition Speed */}
      <div className="space-y-3">
        <label className="block text-sm font-medium">Transition Speed</label>
        <div className="grid grid-cols-3 gap-2">
          {(['slow', 'medium', 'fast'] as const).map((speed) => (
            <button
              key={speed}
              onClick={() => updateSetting('transitionSpeed', speed)}
              className={`p-3 rounded-lg border-2 transition-all capitalize ${
                settings.transitionSpeed === speed
                  ? 'border-primary bg-primary/10'
                  : 'border-border hover:border-primary/50'
              }`}
            >
              {speed}
            </button>
          ))}
        </div>
      </div>

      {/* Duration Per Photo */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="block text-sm font-medium">
            Duration Per Photo
          </label>
          <span className="text-sm text-muted-foreground">
            {settings.durationPerPhoto}s
          </span>
        </div>
        <input
          type="range"
          min="2"
          max="10"
          step="0.5"
          value={settings.durationPerPhoto}
          onChange={(e) =>
            updateSetting('durationPerPhoto', parseFloat(e.target.value))
          }
          className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>2s</span>
          <span>10s</span>
        </div>
      </div>

      {/* Transition Type */}
      <div className="space-y-3">
        <label className="block text-sm font-medium">Transition Type</label>
        <div className="grid grid-cols-3 gap-2">
          {(['fade', 'morph', 'zoom'] as const).map((type) => (
            <button
              key={type}
              onClick={() => updateSetting('transitionType', type)}
              className={`p-3 rounded-lg border-2 transition-all capitalize ${
                settings.transitionType === type
                  ? 'border-primary bg-primary/10'
                  : 'border-border hover:border-primary/50'
              }`}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Frame Rate */}
      <div className="space-y-3">
        <label className="block text-sm font-medium">Frame Rate (FPS)</label>
        <select
          value={settings.fps}
          onChange={(e) => updateSetting('fps', parseInt(e.target.value))}
          className="w-full p-3 rounded-lg border border-border bg-background focus:border-primary focus:outline-none"
        >
          <option value={24}>24 FPS (Cinematic)</option>
          <option value={30}>30 FPS (Standard)</option>
          <option value={60}>60 FPS (Smooth)</option>
        </select>
      </div>

      {/* Settings Summary */}
      <div className="mt-6 p-4 bg-muted rounded-lg">
        <p className="text-xs font-medium mb-2">Preview:</p>
        <ul className="text-xs text-muted-foreground space-y-1">
          <li>
            • Mode: {settings.mode === 'cloud' ? 'Cloud API' : 'Local Model'}
          </li>
          <li>
            • Service:{' '}
            {settings.mode === 'cloud'
              ? settings.cloudService
              : settings.localModel}
          </li>
          <li>• Motion: {settings.motionIntensity}% intensity</li>
          <li>• Transition: {settings.transitionType} ({settings.transitionSpeed})</li>
          <li>• Duration: {settings.durationPerPhoto}s per photo</li>
          <li>• Output: {settings.fps} FPS</li>
        </ul>
      </div>
    </div>
  );
}
