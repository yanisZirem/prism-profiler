@echo off
title Profiler Desktop Installer

echo ===============================================
echo  Profiler Desktop - Installation (User Mode)
echo ===============================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0

REM Run PowerShell installer with ExecutionPolicy Bypass
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_profiler.ps1"

echo.
echo ===============================================
echo Installation finished.
echo If there were errors, read messages above.
echo.
pause
