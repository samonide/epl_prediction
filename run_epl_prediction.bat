@echo off
REM ⚽ EPL Prediction Runner - Windows Batch Script
REM Works on Windows Command Prompt and PowerShell

setlocal enabledelayedexpansion

REM Set console colors (Windows 10+)
set "RED=31"
set "GREEN=32"
set "YELLOW=33"
set "BLUE=34"
set "PURPLE=35"
set "CYAN=36"

REM Function to print colored text (Windows 10+ only)
REM For older Windows, colors won't show but script will work

echo.
echo ================================================================
echo ⚽ EPL PREDICTION SYSTEM LAUNCHER
echo ================================================================
echo 🚀 Windows runner for EPL match predictions
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.7+
    echo 📥 Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python detected

REM Navigate to script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated

REM Check if dependencies are installed
python -c "import numpy, pandas, sklearn, requests, joblib" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing dependencies...
    pip install --upgrade pip
    if exist "requirements.txt" (
        pip install -r requirements.txt
    ) else (
        pip install numpy pandas scikit-learn requests joblib
    )
)

REM Check if main script exists
if not exist "epl_prediction.py" (
    echo ❌ epl_prediction.py not found! Make sure you're in the correct directory.
    pause
    exit /b 1
)

REM Check API keys
echo.
echo 🔑 Checking API keys...
if "%FBR_API_KEY%"=="" (
    echo ⚠️  FBR_API_KEY not set
    set "KEYS_MISSING=1"
) else (
    echo ✅ FBR_API_KEY is set
)

if "%API_FOOTBALL_KEY%"=="" (
    echo ⚠️  API_FOOTBALL_KEY not set
    set "KEYS_MISSING=1"
) else (
    echo ✅ API_FOOTBALL_KEY is set
)

if "%BOOKMAKER_API_KEY%"=="" (
    echo ⚠️  BOOKMAKER_API_KEY not set
    set "KEYS_MISSING=1"
) else (
    echo ✅ BOOKMAKER_API_KEY is set
)

if defined KEYS_MISSING (
    echo.
    echo 💡 Set up API keys in Windows:
    echo    set FBR_API_KEY=your_fbr_key
    echo    set API_FOOTBALL_KEY=02eb00e7497de4d328fa72e3365791b5
    echo    set BOOKMAKER_API_KEY=e66a648eb21c685297c1df4c8e0304cc
    echo.
    echo 🔧 Generate FBR key with: python epl_prediction.py generate-key
    echo.
)

REM Check if first-time setup is needed
if not exist "cache" mkdir cache
if not exist "models" mkdir models
if not exist "models\epl_result_model.joblib" (
    echo.
    echo 🚀 First-time setup required!
    echo.
    echo 1️⃣  Generating FBR API key...
    python epl_prediction.py generate-key
    
    echo 2️⃣  Syncing match data (3 seasons)...
    python epl_prediction.py sync --seasons 3
    
    echo 3️⃣  Syncing enhanced data...
    python epl_prediction.py sync-enhanced --seasons 2
    
    echo 4️⃣  Training ML model...
    python epl_prediction.py train
    
    echo ✅ Setup complete!
)

echo.
echo 🚀 Launching EPL Prediction Interactive Mode...
echo.
python epl_prediction.py --interactive

echo.
echo 👋 Thanks for using EPL Predictor! Good luck! ⚽
pause
