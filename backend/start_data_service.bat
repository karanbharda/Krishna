@echo off
echo ================================================================
echo FYERS DATA SERVICE - STANDALONE MARKET DATA PROVIDER
echo ================================================================
echo.
echo Starting Fyers Data Service on port 8002...
echo This service provides real-time market data to the trading bot
echo.
echo IMPORTANT: Keep this terminal open while using the trading bot
echo.
echo ================================================================

cd /d "%~dp0"
python fyers_data_service.py --host 127.0.0.1 --port 8002

echo.
echo Data service stopped. Press any key to exit...
pause >nul