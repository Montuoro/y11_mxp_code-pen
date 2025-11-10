@echo off
REM Year 11 Performance Analysis Tool Launcher
REM This batch file launches the Python application

echo ========================================
echo Year 11 Performance Analysis Tool
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher
    pause
    exit /b 1
)

REM Run the application
echo Starting application...
python app.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    echo Check the error message above
    pause
)
