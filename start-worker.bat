@echo off
echo Starting Celery Worker...
echo.

cd backend

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Starting Celery worker...
echo Worker will process video generation tasks
echo.

celery -A app.workers.celery_app worker --loglevel=info --pool=solo

pause
