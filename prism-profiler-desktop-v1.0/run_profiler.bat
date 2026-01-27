@echo off
title Profiler Desktop

REM === Find conda base path dynamically ===
for /f "delims=" %%i in ('conda info --base 2^>nul') do set CONDA_BASE=%%i

IF "%CONDA_BASE%"=="" (
    echo ERROR: Conda not found in PATH.
    echo Please open Anaconda Prompt once, then retry.
    pause
    exit /b 1
)

REM === Activate conda ===
call "%CONDA_BASE%\Scripts\activate.bat"

REM === Activate profiler environment ===
call conda activate profiler

IF ERRORLEVEL 1 (
    echo ERROR: Conda environment "profiler" not found.
    echo Please reinstall Profiler Desktop.
    pause
    exit /b 1
)

REM === Go to application directory ===
cd /d "%~dp0"

REM === Launch Streamlit app ===
streamlit run Profiler_Desktop_Gui.py

pause
