# Photo to Video AI - Quick Start Guide

## 🚀 Quick Setup (Windows)

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker Desktop (for PostgreSQL, Redis, MinIO)
- (Optional) NVIDIA GPU with CUDA for local AI models

### One-Command Setup

```bash
setup.bat
```

This will:
1. Create Python virtual environment
2. Install all Python dependencies
3. Initialize database
4. Install Node.js dependencies
5. Start Docker services (PostgreSQL, Redis, MinIO)

## 🎬 Running the Application

### Option 1: Separate Terminals (Recommended for Development)

**Terminal 1 - Backend API:**
```bash
start-backend.bat
```
- FastAPI server at http://localhost:8000
- API docs at http://localhost:8000/docs

**Terminal 2 - Celery Worker:**
```bash
start-worker.bat
```
- Processes video generation jobs

**Terminal 3 - Frontend:**
```bash
start-frontend.bat
```
- Next.js app at http://localhost:3000

### Option 2: Check Services

**Docker Services:**
```bash
cd docker
docker-compose ps
```

Should show:
- `photovideo-db` (PostgreSQL) - Port 5432
- `photovideo-redis` (Redis) - Port 6379
- `photovideo-minio` (MinIO) - Ports 9000, 9001

**MinIO Console:**
- URL: http://localhost:9001
- Username: `minioadmin`
- Password: `minioadmin`

## 📝 Usage

1. **Open http://localhost:3000**

2. **Create a Project:**
   - Enter project name
   - Click "Create Project"

3. **Upload Photos:**
   - Drag & drop 2+ photos
   - Or click "Choose Files"

4. **Arrange Photos:**
   - Drag photos to reorder
   - Click X to delete

5. **Configure Settings:**
   - Choose AI Mode (Cloud/Local)
   - Adjust motion intensity
   - Set transition speed
   - Select FPS

6. **Generate Video:**
   - Click "Generate Video"
   - Watch real-time progress
   - Download when complete

## 🔧 Troubleshooting

### Backend Issues

**Database Connection Error:**
```bash
cd docker
docker-compose restart db
```

**Import Errors:**
```bash
cd backend
venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Issues

**Module Not Found:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Celery Worker Issues

**Task Not Running:**
- Ensure Redis is running: `docker ps`
- Check worker logs in terminal
- On Windows, must use `--pool=solo`

### MinIO Connection Issues

**Cannot Upload:**
- Check MinIO console: http://localhost:9001
- Verify bucket exists: `photo-to-video`
- Restart MinIO: `docker-compose restart minio`

## 📚 API Endpoints

### Projects
- `POST /api/projects` - Create project
- `GET /api/projects/{id}` - Get project with photos
- `DELETE /api/projects/{id}` - Delete project

### Photos
- `POST /api/upload/projects/{id}/photos` - Upload photos
- `DELETE /api/upload/projects/{id}/photos/{photo_id}` - Delete photo
- `PUT /api/upload/projects/{id}/photos/order` - Reorder photos

### Video Generation
- `POST /api/generate` - Start generation
- `GET /api/generate/{job_id}/status` - Check status
- `GET /api/generate/{job_id}/video` - Download video
- `DELETE /api/generate/{job_id}` - Cancel job

### WebSocket
- `WS /ws/projects/{id}` - Real-time updates

## 🎯 Next Steps

### Add API Keys (for Cloud AI)

Edit `backend/.env`:
```env
STABILITY_API_KEY=your-key-here
RUNWAY_API_KEY=your-key-here
PIKA_API_KEY=your-key-here
```

### Enable Local AI Models

1. Ensure NVIDIA GPU with CUDA
2. Install PyTorch with CUDA
3. Download models:
```bash
cd backend
python scripts/download_models.py --model stable-video-diffusion
```

### Production Deployment

See `README.md` for full deployment guide.

## 🐛 Logs

**Backend Logs:**
- Terminal running `start-backend.bat`

**Worker Logs:**
- Terminal running `start-worker.bat`

**Docker Logs:**
```bash
docker-compose logs -f
```

## 🆘 Support

For issues:
1. Check logs
2. Verify all services are running
3. Restart Docker services
4. Check GitHub issues

## 📦 Project Structure

```
photo-to-video-app/
├── backend/          # FastAPI backend
│   ├── app/
│   │   ├── api/     # API routes
│   │   ├── models/  # Database models
│   │   ├── services/# Business logic
│   │   └── workers/ # Celery tasks
│   └── requirements.txt
├── frontend/         # Next.js frontend
│   ├── src/
│   │   ├── app/     # Pages
│   │   ├── components/
│   │   └── hooks/
│   └── package.json
├── docker/          # Docker services
│   └── docker-compose.yml
└── README.md
```

---

Made with ❤️ using Claude AI
