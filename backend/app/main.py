import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import projects, upload, generation, websocket
from app.config import settings
from app.models import Base, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables on startup
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(
    lifespan=lifespan,
    title="Photo to Video API",
    description="AI-powered photo to video conversion API",
    version="0.1.0"
)

# CORS middleware - allow both local and Docker hostnames
allowed_origins = [
    "http://localhost:3000",      # Local development
    "http://127.0.0.1:3000",      # Alternative local
    "http://frontend:3000",       # Docker internal
]

# Add custom frontend URL if provided
custom_frontend = os.getenv("FRONTEND_URL")
if custom_frontend:
    allowed_origins.append(custom_frontend)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(generation.router, prefix="/api/generate", tags=["generation"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

@app.get("/")
async def root():
    return {"message": "Photo to Video API", "version": "0.1.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
