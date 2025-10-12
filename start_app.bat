@echo off
echo Starting Android Malware AI Dashboard...
echo.

echo Starting Backend API Server...
start "Backend API" cmd /k "cd /d "%~dp0" && python src/backend/app.py"

echo Waiting for backend to start...
timeout /t 5

echo Starting Frontend React App...
start "Frontend React" cmd /k "cd /d "%~dp0src\frontend" && npm start"

echo.
echo Both services are starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3002
echo.
pause