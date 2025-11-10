@echo off
REM Setup script for Year 11 Performance Analysis Tool
REM This will install all required Python dependencies

echo ========================================
echo Year 11 Analysis Tool - Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo Installing required packages...
echo.

REM Install dependencies
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some packages
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo You can now run the application by:
echo 1. Double-clicking run_analysis.bat
echo 2. Or running: python app.py
echo.
echo Note: Make sure ODBC Driver 17 for SQL Server is installed
echo Download from: https://aka.ms/downloadmsodbcsql
echo.
pause
