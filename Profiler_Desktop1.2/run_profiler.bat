@echo off
title Profiler Desktop

REM ============================================================
REM  Profiler Desktop — launcher
REM  © 2025 PRISM U1192 Laboratory (INSERM / CHU de Lille / Univ. de Lille)
REM ============================================================

REM === Locate conda base dynamically ===
for /f "delims=" %%i in ('conda info --base 2^>nul') do set CONDA_BASE=%%i

IF "%CONDA_BASE%"=="" (
    echo ERROR: Conda not found in PATH.
    echo Please open Anaconda Prompt once, then retry.
    pause
    exit /b 1
)

REM === Activate conda base + profiler environment ===
call "%CONDA_BASE%\Scripts\activate.bat"
call conda activate profiler

IF ERRORLEVEL 1 (
    echo ERROR: Conda environment "profiler" not found.
    echo Please run install_profiler.bat to set it up.
    pause
    exit /b 1
)

REM === Go to Profiler root directory ===
cd /d "%~dp0"

REM === Fix: tensorflow 2.10.1 is incompatible with protobuf >= 3.20
REM     This env var forces pure-Python protobuf (slightly slower but works)
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

REM === Launch Profiler via new entry point ===
python app\main.py

pause
