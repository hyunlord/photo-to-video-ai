@echo off
echo ========================================
echo Checking Dependencies...
echo ========================================
echo.

REM Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.11+ from https://www.python.org/
    set MISSING=1
) else (
    python --version
    echo [OK] Python installed
)
echo.

REM Check Node.js
echo [2/5] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found!
    echo Please install Node.js 18+ from https://nodejs.org/
    set MISSING=1
) else (
    node --version
    echo [OK] Node.js installed
)
echo.

REM Check Docker
echo [3/5] Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not found!
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    set MISSING=1
) else (
    docker --version
    echo [OK] Docker installed
)
echo.

REM Check FFmpeg
echo [4/5] Checking FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] FFmpeg not found!
    echo FFmpeg is required for video processing.
    echo.
    echo Installation options:
    echo 1. Using Chocolatey: choco install ffmpeg
    echo 2. Download from: https://www.gyan.dev/ffmpeg/builds/
    echo 3. Or use: winget install ffmpeg
    echo.
    set MISSING=1
) else (
    ffmpeg -version | findstr "version"
    echo [OK] FFmpeg installed
)
echo.

REM Check Git
echo [5/5] Checking Git...
git --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Git not found!
    echo Git is optional but recommended.
    echo Install from: https://git-scm.com/
) else (
    git --version
    echo [OK] Git installed
)
echo.

echo ========================================
if defined MISSING (
    echo [INCOMPLETE] Some dependencies are missing!
    echo Please install them before running setup.bat
) else (
    echo [COMPLETE] All required dependencies found!
    echo You can now run: setup.bat
)
echo ========================================
echo.

pause
