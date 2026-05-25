@echo off
REM Start the FastAPI server for the medical transcription pipeline.
REM Uses the venv Python (GPU-enabled) for BART and any local models.
REM
REM Usage:
REM   run_server.bat                        # default port 8000
REM   set PORT=9000 && run_server.bat       # custom port
REM   set USE_LLM=0 && run_server.bat       # disable LLM

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\
    echo.
    echo         Create it with:
    echo           python -m venv .venv
    echo           .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: Defaults — overridable from the environment
if "%WHISPER_MODEL_SIZE%"=="" set WHISPER_MODEL_SIZE=large-v3-turbo
if "%WHISPER_LANGUAGE%"==""   set WHISPER_LANGUAGE=en
if "%USE_LLM%"==""            set USE_LLM=1
if "%HOST%"==""               set HOST=127.0.0.1
if "%PORT%"==""               set PORT=8000

echo [run_server.bat] Starting server on http://%HOST%:%PORT%
echo [run_server.bat] Whisper: %WHISPER_MODEL_SIZE%  (lang=%WHISPER_LANGUAGE%)
echo [run_server.bat] LLM:     %USE_LLM%  (1=on / 0=off)
echo [run_server.bat] GPU pipeline active — using CUDA-enabled PyTorch from .venv
echo.

.venv\Scripts\python.exe -m uvicorn app.main:app --host "%HOST%" --port "%PORT%"
