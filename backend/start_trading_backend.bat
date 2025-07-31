@echo off
echo ================================================================
echo TRADING BOT BACKEND - MAIN APPLICATION SERVER
echo ================================================================
echo.
echo Starting Trading Bot Backend on port 5000...
echo This is the main application server for the trading bot
echo.
echo IMPORTANT: Make sure Data Service is running first!
echo.
echo ================================================================

cd /d "%~dp0"
python web_backend.py --host 127.0.0.1 --port 5000

echo.
echo Backend stopped. Press any key to exit...
pause >nul
