# Photo to Video AI

> Transform static photos into dynamic, AI-animated videos with natural movement and smooth transitions.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-teal.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core Features
- **Photo Upload**: Drag & drop multiple photos with instant preview
- **Photo Ordering**: Intuitive drag & drop reordering interface
- **AI Video Generation**: Transform photos into animated videos using AI
- **Real-time Progress**: WebSocket-based live progress tracking
- **Video Download**: Download generated videos in MP4 format

### AI Support
- **Cloud AI APIs**: Stability AI, Runway Gen-3, Pika Labs
- **Local AI Models**: Stable Video Diffusion (requires NVIDIA GPU)
- **Flexible Mode Switching**: Choose between cloud and local processing

### Animation Settings
- Motion intensity control
- Transition speed selection
- FPS configuration (24/30/60)
- Multiple resolution options

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│                    Next.js 14 + React 18                        │
│         ┌──────────────────────────────────────────┐            │
│         │  PhotoUploader → PhotoOrdering → Preview  │            │
│         │        ↓              ↓            ↓      │            │
│         │   AnimationSettings  WebSocket   Download │            │
│         └──────────────────────────────────────────┘            │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP / WebSocket
┌─────────────────────────▼───────────────────────────────────────┐
│                         Backend                                  │
│                    FastAPI + Python 3.11                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │  Projects  │  │   Upload   │  │ Generation │                │
│  │    API     │  │    API     │  │    API     │                │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                │
│        │               │               │                        │
│  ┌─────▼───────────────▼───────────────▼──────┐                │
│  │              Celery Workers                 │                │
│  │    ┌─────────────┐  ┌─────────────┐        │                │
│  │    │  Cloud AI   │  │  Local AI   │        │                │
│  │    │ (Stability) │  │   (SVD)     │        │                │
│  │    └─────────────┘  └─────────────┘        │                │
│  └─────────────────────────────────────────────┘                │
└───────────┬─────────────────┬─────────────────┬─────────────────┘
            │                 │                 │
   ┌────────▼──────┐ ┌───────▼───────┐ ┌──────▼──────┐
   │  PostgreSQL   │ │     Redis     │ │    MinIO    │
   │   Database    │ │  Cache/Queue  │ │   Storage   │
   └───────────────┘ └───────────────┘ └─────────────┘
```

---

## Tech Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 14.2 | React framework with App Router |
| React | 18.3 | UI library |
| TypeScript | 5.5 | Type safety |
| Tailwind CSS | 3.4 | Styling |
| dnd-kit | 6.1 | Drag and drop functionality |
| Zustand | 4.5 | State management |
| Socket.IO Client | 4.7 | Real-time communication |
| Axios | 1.6 | HTTP client |

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.109 | API framework |
| SQLAlchemy | 2.0 | ORM |
| Celery | 5.3 | Task queue |
| Redis | 5.0 | Caching and message broker |
| PostgreSQL | 15 | Database |
| MinIO | - | S3-compatible object storage |

### AI & Video Processing
| Technology | Version | Purpose |
|------------|---------|---------|
| PyTorch | 2.2 | Deep learning framework |
| Diffusers | 0.26 | Stable Video Diffusion |
| OpenCV | 4.9 | Image processing |
| FFmpeg | - | Video encoding |
| Pillow | 10.2 | Image manipulation |

---

## Project Structure

```
photo-to-video-app/
├── backend/                      # FastAPI Backend
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── generation.py # Video generation endpoints
│   │   │       ├── projects.py   # Project CRUD endpoints
│   │   │       ├── upload.py     # Photo upload endpoints
│   │   │       └── websocket.py  # Real-time updates
│   │   ├── models/
│   │   │   ├── job.py            # Video generation job model
│   │   │   ├── photo.py          # Photo model
│   │   │   └── project.py        # Project model
│   │   ├── services/
│   │   │   ├── cloud_ai/         # Cloud AI integrations
│   │   │   │   ├── base.py       # Abstract base class
│   │   │   │   └── stability.py  # Stability AI integration
│   │   │   ├── local_ai/         # Local model support
│   │   │   ├── storage/          # S3/MinIO storage
│   │   │   └── video_processing/ # FFmpeg pipeline
│   │   ├── workers/
│   │   │   ├── celery_app.py     # Celery configuration
│   │   │   └── tasks.py          # Async video generation tasks
│   │   ├── config.py             # Application settings
│   │   └── main.py               # FastAPI entry point
│   └── requirements.txt
│
├── frontend/                     # Next.js Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx          # Home page
│   │   │   └── project/[id]/     # Project editor page
│   │   ├── components/
│   │   │   ├── editor/
│   │   │   │   ├── AnimationSettings.tsx
│   │   │   │   └── PhotoOrdering.tsx
│   │   │   ├── preview/
│   │   │   │   └── VideoPreview.tsx
│   │   │   └── upload/
│   │   │       └── PhotoUploader.tsx
│   │   ├── hooks/
│   │   │   ├── useVideoGeneration.ts
│   │   │   └── useWebSocket.ts
│   │   ├── lib/
│   │   │   └── api-client.ts     # API client wrapper
│   │   └── types/
│   │       └── index.ts          # TypeScript types
│   └── package.json
│
├── docker/
│   └── docker-compose.yml        # PostgreSQL, Redis, MinIO
│
├── scripts/
│   └── download_models.py        # Download AI models
│
├── setup.bat                     # Automated setup script
├── start-backend.bat             # Start FastAPI server
├── start-frontend.bat            # Start Next.js server
├── start-worker.bat              # Start Celery worker
├── check-dependencies.bat        # Check prerequisites
│
├── QUICKSTART.md                 # Quick start guide
├── GITHUB_SETUP.md               # GitHub upload guide
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## Prerequisites

### Required Software
| Software | Version | Download |
|----------|---------|----------|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| Docker Desktop | Latest | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Git | Latest | [git-scm.com](https://git-scm.com/) |

### Optional (for Local AI)
| Software | Version | Notes |
|----------|---------|-------|
| NVIDIA GPU | 8GB+ VRAM | RTX 3060 or better |
| CUDA | 11.8+ | [nvidia.com](https://developer.nvidia.com/cuda-downloads) |
| cuDNN | 8.6+ | [nvidia.com](https://developer.nvidia.com/cudnn) |

### Check Dependencies
```bash
# Run this to verify all prerequisites
check-dependencies.bat
```

---

## Installation

### Quick Setup (Windows)
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/photo-to-video-ai.git
cd photo-to-video-ai

# 2. Run automated setup
setup.bat
```

The setup script will:
1. Create Python virtual environment
2. Install all Python dependencies
3. Install Node.js dependencies
4. Start Docker services (PostgreSQL, Redis, MinIO)
5. Initialize the database

### Manual Setup

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install
```

#### Docker Services
```bash
cd docker

# Start services
docker-compose up -d

# Verify services are running
docker-compose ps
```

---

## Running the Application

### Start All Services

Open 3 separate terminals and run:

**Terminal 1 - Backend API:**
```bash
start-backend.bat
# or manually:
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Celery Worker:**
```bash
start-worker.bat
# or manually:
cd backend
venv\Scripts\activate
celery -A app.workers.celery_app worker --loglevel=info --pool=solo
```

**Terminal 3 - Frontend:**
```bash
start-frontend.bat
# or manually:
cd frontend
npm run dev
```

### Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | Main application |
| Backend API | http://localhost:8000 | FastAPI server |
| API Docs | http://localhost:8000/docs | Swagger UI |
| MinIO Console | http://localhost:9001 | Object storage admin |

### MinIO Credentials
- **URL**: http://localhost:9001
- **Username**: `minioadmin`
- **Password**: `minioadmin`

---

## Usage Guide

### 1. Create a Project
1. Open http://localhost:3000
2. Enter a project name
3. Click "Create Project"

### 2. Upload Photos
1. Drag & drop photos onto the upload area
2. Or click "Choose Files" to select
3. Supported formats: JPEG, PNG, WebP
4. Maximum 20 photos per project

### 3. Arrange Photos
1. Drag photos to reorder the sequence
2. Click X to remove unwanted photos
3. Photos will animate in the order shown

### 4. Configure Animation Settings
| Setting | Description | Options |
|---------|-------------|---------|
| AI Mode | Cloud API or Local GPU | Cloud / Local |
| Motion Intensity | Amount of movement | 0-100% |
| Transition Speed | Speed between photos | Slow / Medium / Fast |
| FPS | Frames per second | 24 / 30 / 60 |

### 5. Generate Video
1. Click "Generate Video"
2. Watch real-time progress
3. Download when complete

---

## API Documentation

### Projects API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/projects` | Create new project |
| `GET` | `/api/projects/{id}` | Get project with photos |
| `PUT` | `/api/projects/{id}` | Update project |
| `DELETE` | `/api/projects/{id}` | Delete project |

#### Create Project
```bash
curl -X POST http://localhost:8000/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "My Project"}'
```

### Upload API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload/projects/{id}/photos` | Upload photos |
| `DELETE` | `/api/upload/projects/{id}/photos/{photo_id}` | Delete photo |
| `PUT` | `/api/upload/projects/{id}/photos/order` | Reorder photos |

#### Upload Photos
```bash
curl -X POST http://localhost:8000/api/upload/projects/{id}/photos \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg"
```

### Generation API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/generate` | Start video generation |
| `GET` | `/api/generate/{job_id}/status` | Check job status |
| `GET` | `/api/generate/{job_id}/video` | Download video |
| `DELETE` | `/api/generate/{job_id}` | Cancel job |

#### Start Generation
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "uuid",
    "mode": "cloud",
    "settings": {
      "motion_intensity": 70,
      "transition_speed": "medium",
      "fps": 30
    }
  }'
```

### WebSocket API

| Endpoint | Description |
|----------|-------------|
| `WS /ws/projects/{id}` | Real-time progress updates |

#### WebSocket Events
```javascript
// Connect
const ws = new WebSocket('ws://localhost:8000/ws/projects/{id}')

// Events received:
{
  "event": "generation_started",
  "job_id": "uuid"
}

{
  "event": "progress",
  "job_id": "uuid",
  "progress": 45,
  "message": "Generating frame 45/100"
}

{
  "event": "generation_complete",
  "job_id": "uuid",
  "video_url": "/api/generate/{job_id}/video"
}
```

---

## Configuration

### Environment Variables

Create `backend/.env` file:

```env
# Database
DATABASE_URL=postgresql://photouser:photopass@localhost:5432/photovideo

# Redis
REDIS_URL=redis://localhost:6379/0

# Storage (MinIO)
S3_BUCKET=photo-to-video
S3_ENDPOINT=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin

# AI API Keys (add your own)
STABILITY_API_KEY=your-stability-api-key
RUNWAY_API_KEY=your-runway-api-key
PIKA_API_KEY=your-pika-api-key

# Local AI Settings
ENABLE_LOCAL_MODELS=true
GPU_DEVICE=cuda:0
MODELS_DIR=./models

# Video Settings
DEFAULT_FPS=30
DEFAULT_RESOLUTION=1024x1024

# Security
SECRET_KEY=your-secret-key-change-in-production
```

### Docker Services Configuration

Edit `docker/docker-compose.yml` to customize:

| Service | Port | Environment Variables |
|---------|------|----------------------|
| PostgreSQL | 5432 | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` |
| Redis | 6379 | - |
| MinIO | 9000, 9001 | `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` |

---

## Troubleshooting

### Backend Issues

#### Database Connection Error
```bash
# Restart PostgreSQL
cd docker
docker-compose restart db

# Check if PostgreSQL is running
docker ps | grep photovideo-db
```

#### Import Errors
```bash
cd backend
venv\Scripts\activate
pip install -r requirements.txt --force-reinstall
```

### Frontend Issues

#### Module Not Found
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

#### Port Already in Use
```bash
# Find process using port 3000
netstat -ano | findstr :3000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Celery Worker Issues

#### Task Not Running
1. Ensure Redis is running: `docker ps | grep redis`
2. Check worker logs in terminal
3. On Windows, must use `--pool=solo` flag

#### Worker Connection Error
```bash
# Restart Redis
cd docker
docker-compose restart redis
```

### MinIO Issues

#### Cannot Upload Files
```bash
# Check MinIO is running
docker ps | grep minio

# Restart MinIO
cd docker
docker-compose restart minio

# Access MinIO console
# URL: http://localhost:9001
# Login: minioadmin / minioadmin
```

#### Bucket Not Found
```bash
# Recreate the bucket
docker-compose restart minio-init
```

### GPU Issues (Local AI)

#### CUDA Not Available
```python
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### Out of Memory
- Close other GPU applications
- Reduce batch size in settings
- Use cloud AI mode instead

---

## Hardware Requirements

### Minimum (Cloud AI Only)
| Component | Requirement |
|-----------|-------------|
| CPU | 4 cores |
| RAM | 8GB |
| Storage | 50GB SSD |
| GPU | Not required |

### Recommended (Local AI)
| Component | Requirement |
|-----------|-------------|
| CPU | 8+ cores |
| RAM | 32GB |
| Storage | 100GB NVMe SSD |
| GPU | NVIDIA RTX 3060 (12GB VRAM) |

### Optimal (Production)
| Component | Requirement |
|-----------|-------------|
| CPU | 16+ cores |
| RAM | 64GB |
| Storage | 500GB NVMe SSD |
| GPU | NVIDIA RTX 4090 (24GB VRAM) |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [GITHUB_SETUP.md](GITHUB_SETUP.md) for detailed GitHub setup instructions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Quick Links

- [Quick Start Guide](QUICKSTART.md) - Detailed setup and usage instructions
- [GitHub Setup](GITHUB_SETUP.md) - How to upload to GitHub
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)

---

Made with Claude AI
