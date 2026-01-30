@echo off
echo ========================================
echo Git Repository Initialization
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed!
    echo Please install Git from: https://git-scm.com/
    echo.
    pause
    exit /b 1
)

echo [OK] Git is installed
echo.

REM Check if already initialized
if exist .git (
    echo [WARNING] Git repository already initialized!
    echo.
    set /p REINIT="Do you want to reinitialize? (y/n): "
    if /i not "%REINIT%"=="y" (
        echo Cancelled.
        pause
        exit /b 0
    )
    rmdir /s /q .git
)

echo Step 1: Initializing Git repository...
git init
if errorlevel 1 (
    echo [ERROR] Failed to initialize git
    pause
    exit /b 1
)
echo [OK] Repository initialized
echo.

echo Step 2: Adding all files...
git add .
if errorlevel 1 (
    echo [ERROR] Failed to add files
    pause
    exit /b 1
)
echo [OK] Files staged
echo.

echo Step 3: Creating initial commit...
git commit -m "Initial commit: Photo to Video AI application

Features:
- Full-stack web application (FastAPI + Next.js)
- AI-powered video generation from photos
- Real-time progress tracking via WebSocket
- Support for Cloud AI APIs and Local models
- Drag-and-drop photo management
- Customizable animation settings
- Docker-based development environment
- Comprehensive documentation

Tech Stack:
- Backend: FastAPI, Python 3.11, PostgreSQL, Redis, Celery
- Frontend: Next.js 14, TypeScript, Tailwind CSS, shadcn/ui
- AI: Stable Video Diffusion, FFmpeg
- Infrastructure: Docker, MinIO (S3-compatible)"

if errorlevel 1 (
    echo [ERROR] Failed to create commit
    pause
    exit /b 1
)
echo [OK] Initial commit created
echo.

echo ========================================
echo Git repository initialized successfully!
echo ========================================
echo.
echo Next steps:
echo.
echo Option 1: Using GitHub CLI (gh)
echo   1. gh auth login
echo   2. gh repo create photo-to-video-ai --public --source=. --push
echo.
echo Option 2: Using GitHub website
echo   1. Create repository on GitHub: https://github.com/new
echo   2. git remote add origin https://github.com/YOUR_USERNAME/photo-to-video-ai.git
echo   3. git branch -M main
echo   4. git push -u origin main
echo.
echo See GITHUB_SETUP.md for detailed instructions
echo.

pause
