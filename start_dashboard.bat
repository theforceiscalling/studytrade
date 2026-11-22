@echo off
setlocal
cd /d "%~dp0"
title StudyTrade - Quant Option Trading Dashboard
color 0B

echo =========================================================================
echo                   StudyTrade - Systematic Option Buying
echo               NSE India Quant Engine and Gamma Squeeze Cockpit
echo =========================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python not found in .venv\Scripts\python.exe!
    echo Please make sure the virtual environment is set up.
    pause
    exit /b 1
)

echo [1/3] Applying database migrations...
.venv\Scripts\python.exe web\manage.py migrate --noinput
if errorlevel 1 (
    echo [WARNING] Database migration had a warning or error.
)

echo [2/3] Checking and populating market data and paper journal...
.venv\Scripts\python.exe -m studytrade.data.seed_db
if errorlevel 1 (
    echo [WARNING] Seeder script had a warning or error.
)

echo [3/3] Launching web browser and starting localhost server...
echo.
echo =========================================================================
echo   Dashboard URL:     http://127.0.0.1:8000/
echo   Gamma Squeeze URL: http://127.0.0.1:8000/gamma-squeeze/
echo   Server is running. Press CTRL+C to stop.
echo =========================================================================
echo.

start "" http://127.0.0.1:8000/gamma-squeeze/

.venv\Scripts\python.exe web\manage.py runserver 127.0.0.1:8000

echo.
echo Server stopped.
pause