from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import projects, upload, generation, websocket
from app.config import settings

app = FastAPI(
    title="Photo to Video API",
    description="AI-powered photo to video conversion API",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
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
