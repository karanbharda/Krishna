@echo off
echo 🚀 Starting Indian Stock Trading Bot - Full Stack Application
echo ==============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

echo ✅ Node.js found
node --version
echo.

REM Check if required files exist
if not exist "backend\web_backend.py" (
    echo ❌ Backend file not found: backend\web_backend.py
    echo Please make sure all required files are in the correct directories
    pause
    exit /b 1
)

if not exist "frontend\package.json" (
    echo ❌ Frontend package.json not found
    echo Please make sure the React frontend is properly set up
    pause
    exit /b 1
)

echo ✅ Required files found
echo.

REM Install frontend dependencies if node_modules doesn't exist
if not exist "frontend\node_modules" (
    echo 📦 Installing React dependencies...
    cd frontend
    npm install
    if errorlevel 1 (
        echo ❌ Failed to install frontend dependencies
        pause
        exit /b 1
    )
    cd ..
    echo ✅ Frontend dependencies installed
    echo.
)

echo 🌟 Starting Full Stack Application...
echo.
echo 📚 Application URLs:
echo    Backend API: http://127.0.0.1:5000
echo    Frontend UI: http://localhost:3000
echo    API Docs: http://127.0.0.1:5000/docs
echo.
echo 🔧 Starting backend server...

REM Start backend in a new window
start "Trading Bot Backend" cmd /k "cd /d %cd% && python backend\run_web_interface.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

echo 🎨 Starting React frontend...

REM Start frontend in a new window
start "Trading Bot Frontend" cmd /k "cd /d %cd%\frontend && npm start"

echo.
echo 🎉 Full Stack Application Started!
echo.
echo 📋 What's running:
echo    ✅ FastAPI Backend Server (Port 5000)
echo    ✅ React Development Server (Port 3000)
echo.
echo 🌐 Open your browser and go to:
echo    http://localhost:3000
echo.
echo 🛑 To stop the application:
echo    Close both terminal windows or press Ctrl+C in each
echo.
echo 📊 Monitor the terminal windows for logs and status updates
echo.
pause
