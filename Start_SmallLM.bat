@echo off
setlocal
cd /d "%~dp0"
title SmallLM Terminal

set "SMALLLM_NEEDS_SETUP=0"
if not exist ".venv\Scripts\python.exe" set "SMALLLM_NEEDS_SETUP=1"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import torch, numpy, tokenizers, sklearn, joblib, tqdm, rich" >nul 2>&1
    if errorlevel 1 set "SMALLLM_NEEDS_SETUP=1"
)

if "%SMALLLM_NEEDS_SETUP%"=="1" (
    where uv >nul 2>&1
    if errorlevel 1 (
        echo SmallLM needs uv for its one-time setup.
        echo Install uv from https://docs.astral.sh/uv/getting-started/installation/
        echo Then double-click Start_SmallLM.bat again.
        echo.
        pause
        exit /b 1
    )

    echo Preparing SmallLM for its first launch...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    if errorlevel 1 (
        echo.
        echo SmallLM setup did not complete.
        pause
        exit /b 1
    )
    echo.
)

".venv\Scripts\python.exe" "Main_Run_Program.py"
if errorlevel 1 pause
