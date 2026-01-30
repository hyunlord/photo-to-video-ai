@echo off
echo ========================================
echo Photo to Video AI - Setup Script
echo ========================================
echo.

echo Step 1: Setting up Backend...
cd backend

if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo Installing Python dependencies...
    pip install -r requirements.txt
) else (
    echo Virtual environment already exists!
    call venv\Scripts\activate
    echo Updating Python dependencies...
    pip install -r requirements.txt
)

echo.
echo Initializing database...
python -c "from app.models import Base, engine; Base.metadata.create_all(bind=engine)"

cd ..

echo.
echo Step 2: Setting up Frontend...
cd frontend

if not exist node_modules (
    echo Installing Node.js dependencies...
    npm install
) else (
    echo Node modules already installed!
    echo Updating dependencies...
    npm install
)

cd ..

echo.
echo Step 3: Starting Docker services...
cd docker
docker-compose up -d
cd ..

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Services running:
echo - PostgreSQL: localhost:5432
echo - Redis: localhost:6379
echo - MinIO: localhost:9000 (Console: localhost:9001)
echo.
echo To start the application:
echo 1. Run start-backend.bat (in one terminal)
echo 2. Run start-worker.bat (in another terminal)
echo 3. Run start-frontend.bat (in a third terminal)
echo.
echo Or simply run: start-all.bat
echo.
pause
