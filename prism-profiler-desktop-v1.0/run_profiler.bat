@echo off
SET SCRIPT_DIR=%~dp0

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" profiler
python "%SCRIPT_DIR%\profiler_desktop.py"