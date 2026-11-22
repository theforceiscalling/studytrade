@echo off
title StudyTrade — Automated Live Market Session Engine (09:15 AM - 03:30 PM IST)
color 0b

echo =====================================================================
echo  StudyTrade -- Automated Live Session Engine
echo  NSE India: 09:15 AM to 03:30 PM IST (Mon-Fri)
echo  WebSocket Streaming / Zero Rate-Limit Mode
echo =====================================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv.
    echo Please make sure .venv exists.
    pause
    exit /b 1
)

echo Starting Live Session Engine daemon...
.venv\Scripts\python.exe -c "from studytrade.live.session_engine import get_live_session_engine; import time; engine = get_live_session_engine(); engine.start(); print('Live Session Engine is ACTIVE.'); print('Status:', engine.get_status()); [time.sleep(1) for _ in iter(int, 1)]"

pause
