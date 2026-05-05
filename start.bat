@echo off
:: ============================================================
:: start.bat  –  Launch both microservices (Windows)
:: ============================================================
:: Usage:
::   start.bat           -> start both services
::   start.bat audio     -> start service_audio only (port 8001)
::   start.bat web       -> start service_web only   (port 8000)
:: ============================================================

SET ROOT=%~dp0

IF "%1"=="audio" GOTO AUDIO_ONLY
IF "%1"=="web"   GOTO WEB_ONLY

:: ── Both services ──────────────────────────────────────────
:BOTH
echo Starting service_audio on port 8001...
IF EXIST "%ROOT%.venv_audio\Scripts\activate.bat" (
    CALL "%ROOT%.venv_audio\Scripts\activate.bat"
)
START "service_audio" cmd /k "cd /d %ROOT%service_audio && uvicorn main:app --host 0.0.0.0 --port 8001 --reload"

timeout /t 3 /nobreak >nul

echo Starting service_web on port 8000...
IF EXIST "%ROOT%.venv_web\Scripts\activate.bat" (
    CALL "%ROOT%.venv_web\Scripts\activate.bat"
)
START "service_web" cmd /k "cd /d %ROOT%service_web && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo.
echo Both services launched in separate windows.
echo   service_web   -^> http://localhost:8000/docs
echo   service_audio -^> http://localhost:8001/docs
GOTO END

:: ── Audio only ─────────────────────────────────────────────
:AUDIO_ONLY
echo Starting service_audio on port 8001...
IF EXIST "%ROOT%.venv_audio\Scripts\activate.bat" (
    CALL "%ROOT%.venv_audio\Scripts\activate.bat"
)
cd /d %ROOT%service_audio
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
GOTO END

:: ── Web only ───────────────────────────────────────────────
:WEB_ONLY
echo Starting service_web on port 8000...
IF EXIST "%ROOT%.venv_web\Scripts\activate.bat" (
    CALL "%ROOT%.venv_web\Scripts\activate.bat"
)
cd /d %ROOT%service_web
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
GOTO END

:END
