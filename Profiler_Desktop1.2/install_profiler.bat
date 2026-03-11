@echo off
title Profiler Desktop Installer

echo ============================================================
echo  Profiler Desktop - Installation (User Mode)
echo  (c) 2025 PRISM U1192 - INSERM / CHU de Lille / Univ. Lille
echo ============================================================
echo.

REM Get script directory (Profiler root)
set SCRIPT_DIR=%~dp0

REM Run PowerShell setup script with ExecutionPolicy Bypass
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%setup_environment.ps1"

pause
