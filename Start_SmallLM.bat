@echo off
setlocal
cd /d "%~dp0"
title SmallLM Terminal

if not exist ".venv\Scripts\python.exe" (
    echo SmallLM is not set up yet.
    echo Run setup.ps1 once, then double-click Start_SmallLM.bat again.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "Main_Run_Program.py"
if errorlevel 1 pause
