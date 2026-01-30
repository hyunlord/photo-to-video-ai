@echo off
echo Starting Photo to Video Backend...
echo.

cd backend

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Checking database...
python -c "from app.models import Base, engine; Base.metadata.create_all(bind=engine)" 2>nul
if errorlevel 1 (
    echo Database setup complete!
) else (
    echo Database already exists!
)

echo.
echo Starting FastAPI server...
echo API will be available at http://localhost:8000
echo API Docs at http://localhost:8000/docs
echo.

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
