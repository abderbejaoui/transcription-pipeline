@echo off
REM Run the medical transcript correction CLI using the venv Python (GPU-enabled).
REM
REM Usage:
REM   run.bat --transcript "The patient has dolly prahn"           # correct a transcript
REM   run.bat --transcript "..." --no-interactive                  # skip HITL prompts
REM   run.bat --help                                               # show full help

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

echo [run.bat] GPU pipeline active — using CUDA-enabled PyTorch from .venv
.venv\Scripts\python.exe main.py %*
