@echo off
echo Starting Photo to Video Frontend...
echo.

cd frontend

if not exist node_modules (
    echo Installing dependencies...
    npm install
)

echo.
echo Starting Next.js development server...
echo Frontend will be available at http://localhost:3000
echo.

npm run dev
