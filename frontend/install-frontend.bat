@echo off
echo 🚀 Installing React Frontend for Indian Stock Trading Bot
echo ========================================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    echo Recommended version: Node.js 16 or higher
    pause
    exit /b 1
)

echo ✅ Node.js found
node --version
echo.

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ npm found
npm --version
echo.

REM Install dependencies
echo 📦 Installing React dependencies...
echo This may take a few minutes...
echo.

npm install

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully!
echo.
echo 🎉 React frontend is ready!
echo.
echo 🚀 To start the development server:
echo    npm start
echo.
echo 🏗️  To build for production:
echo    npm run build
echo.
echo 🧪 To run tests:
echo    npm test
echo.
echo 📚 The frontend will be available at:
echo    http://localhost:3000
echo.
echo 🔗 Make sure the backend is running at:
echo    http://127.0.0.1:5000
echo.
pause
