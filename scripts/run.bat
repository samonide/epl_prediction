@echo off
REM EPL Prediction System Launcher (Windows)

echo 🚀 Starting EPL Prediction System...

REM Navigate to the project directory
cd /d "%~dp0\.."

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed.
    echo Please install Python 3.8 or later from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo 📥 Installing dependencies...
    pip install -r requirements.txt
)

REM Run the application
echo ⚽ Launching EPL Prediction System...
python main.py %*

echo 👋 Thanks for using EPL Prediction System!
pause
