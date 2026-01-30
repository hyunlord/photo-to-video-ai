"use client";

import { useState, useEffect } from 'react';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  rectSortingStrategy,
  useSortable,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Photo } from '@/types';
import { photoApi } from '@/lib/api-client';

interface PhotoOrderingProps {
  projectId: string;
  photos: Photo[];
  onReorder: (photos: Photo[]) => void;
  onDelete?: (photoId: string) => void;
}

interface SortablePhotoItemProps {
  photo: Photo;
  onDelete?: (photoId: string) => void;
}

function SortablePhotoItem({ photo, onDelete }: SortablePhotoItemProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: photo.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const thumbnailUrl = `http://localhost:9000/photo-to-video/${photo.thumbnail_path}`;

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`
        relative group rounded-lg overflow-hidden bg-card border-2 border-border
        hover:border-primary transition-all cursor-move
        ${isDragging ? 'shadow-lg z-50' : 'shadow-sm'}
      `}
      {...attributes}
      {...listeners}
    >
      <div className="aspect-square relative">
        <img
          src={thumbnailUrl}
          alt={`Photo ${photo.order_index + 1}`}
          className="w-full h-full object-cover"
          draggable={false}
        />

        {/* Order number badge */}
        <div className="absolute top-2 left-2 bg-primary text-primary-foreground w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm shadow-lg">
          {photo.order_index + 1}
        </div>

        {/* Delete button */}
        {onDelete && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(photo.id);
            }}
            className="absolute top-2 right-2 bg-destructive text-destructive-foreground w-8 h-8 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-lg hover:bg-destructive/90"
            type="button"
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
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        )}

        {/* Drag handle indicator */}
        <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <svg
            className="w-6 h-6 text-white drop-shadow-lg"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path d="M9 3h2v2H9V3zm0 4h2v2H9V7zm0 4h2v2H9v-2zm0 4h2v2H9v-2zm0 4h2v2H9v-2zM13 3h2v2h-2V3zm0 4h2v2h-2V7zm0 4h2v2h-2v-2zm0 4h2v2h-2v-2zm0 4h2v2h-2v-2z" />
          </svg>
        </div>
      </div>

      {/* Photo info */}
      <div className="p-2 bg-card">
        <p className="text-xs text-muted-foreground truncate">
          {photo.metadata.width} × {photo.metadata.height}
        </p>
      </div>
    </div>
  );
}

export default function PhotoOrdering({
  projectId,
  photos: initialPhotos,
  onReorder,
  onDelete,
}: PhotoOrderingProps) {
  const [photos, setPhotos] = useState<Photo[]>(initialPhotos);
  const [isSaving, setIsSaving] = useState(false);

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  useEffect(() => {
    setPhotos(initialPhotos);
  }, [initialPhotos]);

  const handleDragEnd = async (event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      const oldIndex = photos.findIndex((p) => p.id === active.id);
      const newIndex = photos.findIndex((p) => p.id === over.id);

      const newPhotos = arrayMove(photos, oldIndex, newIndex);

      // Update order_index for all photos
      const updatedPhotos = newPhotos.map((photo, index) => ({
        ...photo,
        order_index: index,
      }));

      setPhotos(updatedPhotos);

      // Save to backend
      try {
        setIsSaving(true);
        const photoOrder = updatedPhotos.map((photo) => ({
          id: photo.id,
          order_index: photo.order_index,
        }));

        await photoApi.reorder(projectId, photoOrder);
        onReorder(updatedPhotos);
      } catch (error) {
        console.error('Failed to save photo order:', error);
        // Revert on error
        setPhotos(initialPhotos);
      } finally {
        setIsSaving(false);
      }
    }
  };

  const handleDelete = async (photoId: string) => {
    if (!confirm('Are you sure you want to delete this photo?')) {
      return;
    }

    try {
      await photoApi.delete(projectId, photoId);
      const newPhotos = photos
        .filter((p) => p.id !== photoId)
        .map((photo, index) => ({
          ...photo,
          order_index: index,
        }));

      setPhotos(newPhotos);
      if (onDelete) {
        onDelete(photoId);
      }
      onReorder(newPhotos);
    } catch (error) {
      console.error('Failed to delete photo:', error);
      alert('Failed to delete photo');
    }
  };

  if (photos.length === 0) {
    return (
      <div className="text-center py-12 border-2 border-dashed border-border rounded-lg">
        <p className="text-muted-foreground">No photos uploaded yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">
          Photos ({photos.length})
        </h3>
        {isSaving && (
          <span className="text-sm text-muted-foreground">Saving...</span>
        )}
      </div>

      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragEnd={handleDragEnd}
      >
        <SortableContext
          items={photos.map((p) => p.id)}
          strategy={rectSortingStrategy}
        >
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {photos.map((photo) => (
              <SortablePhotoItem
                key={photo.id}
                photo={photo}
                onDelete={onDelete ? handleDelete : undefined}
              />
            ))}
          </div>
        </SortableContext>
      </DndContext>

      <div className="text-sm text-muted-foreground text-center mt-4">
        Drag photos to reorder • Click X to delete
      </div>
    </div>
  );
}
