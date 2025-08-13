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

:MAIN_MENU
echo.
echo ================================================================
echo ⚽ EPL PREDICTION QUICK LAUNCHER
echo ================================================================
echo.
echo Choose an action:
echo.
echo 1️⃣  🎯 Predict next EPL match
echo 2️⃣  📅 Show upcoming fixtures predictions  
echo 3️⃣  ⚔️  Predict specific match
echo 4️⃣  🔄 Update all data
echo 5️⃣  🖥️  Interactive mode (full menu)
echo 6️⃣  🛠️  Setup/Maintenance
echo 7️⃣  ❓ Help
echo 8️⃣  🚪 Exit
echo.
set /p "choice=Enter your choice (1-8): "

if "%choice%"=="1" (
    echo.
    echo ================================================================
    echo ⚽ NEXT MATCH PREDICTION
    echo ================================================================
    python epl_prediction.py --next
    goto CONTINUE_PROMPT
)

if "%choice%"=="2" (
    echo.
    echo ================================================================ 
    echo ⚽ UPCOMING FIXTURES
    echo ================================================================
    set /p "num_fixtures=How many fixtures to predict? (default: 5): "
    if "!num_fixtures!"=="" set "num_fixtures=5"
    python epl_prediction.py predict-fixtures --top !num_fixtures!
    goto CONTINUE_PROMPT
)

if "%choice%"=="3" (
    echo.
    echo ================================================================
    echo ⚽ SPECIFIC MATCH PREDICTION  
    echo ================================================================
    set /p "home_team=Home team: "
    set /p "away_team=Away team: "
    python epl_prediction.py predict-match --home "!home_team!" --away "!away_team!"
    goto CONTINUE_PROMPT
)

if "%choice%"=="4" (
    echo.
    echo ================================================================
    echo ⚽ UPDATING ALL DATA
    echo ================================================================
    python epl_prediction.py --update
    goto CONTINUE_PROMPT
)

if "%choice%"=="5" (
    echo.
    echo ================================================================
    echo ⚽ INTERACTIVE MODE
    echo ================================================================
    python epl_prediction.py --interactive
    goto CONTINUE_PROMPT
)

if "%choice%"=="6" (
    goto SETUP_MENU
)

if "%choice%"=="7" (
    echo.
    echo ================================================================
    echo ⚽ HELP
    echo ================================================================
    python epl_prediction.py --help
    goto CONTINUE_PROMPT
)

if "%choice%"=="8" (
    echo.
    echo 👋 Goodbye!
    goto END
)

echo ❌ Invalid choice. Please try again.
goto MAIN_MENU

:SETUP_MENU
echo.
echo ================================================================
echo ⚽ SETUP & MAINTENANCE
echo ================================================================
echo.
echo Choose an action:
echo.
echo 1️⃣  🔑 Generate FBR API key
echo 2️⃣  📥 Sync match data
echo 3️⃣  📊 Sync enhanced data  
echo 4️⃣  🧠 Train ML model
echo 5️⃣  🧹 Clean cache
echo 6️⃣  🔍 Check system status
echo 7️⃣  🔙 Back to main menu
echo.
set /p "setup_choice=Enter your choice (1-7): "

if "%setup_choice%"=="1" (
    python epl_prediction.py generate-key
    goto CONTINUE_PROMPT
)

if "%setup_choice%"=="2" (
    set /p "seasons=Number of seasons to sync (default: 3): "
    if "!seasons!"=="" set "seasons=3"
    python epl_prediction.py sync --seasons !seasons!
    goto CONTINUE_PROMPT
)

if "%setup_choice%"=="3" (
    set /p "seasons=Number of seasons for enhanced data (default: 2): "
    if "!seasons!"=="" set "seasons=2"
    python epl_prediction.py sync-enhanced --seasons !seasons!
    goto CONTINUE_PROMPT
)

if "%setup_choice%"=="4" (
    python epl_prediction.py train
    goto CONTINUE_PROMPT
)

if "%setup_choice%"=="5" (
    echo 🧹 Cleaning cache...
    if exist "cache" rmdir /s /q "cache"
    mkdir cache
    echo ✅ Cache cleaned
    goto CONTINUE_PROMPT
)

if "%setup_choice%"=="6" (
    echo.
    echo ================================================================
    echo ⚽ SYSTEM STATUS
    echo ================================================================
    echo.
    
    python --version
    
    if exist ".venv" (
        echo ✅ Virtual environment exists
    ) else (
        echo ⚠️  Virtual environment not found
    )
    
    if exist "epl_prediction.py" (
        echo ✅ Main script found
    ) else (
        echo ❌ epl_prediction.py not found
    )
    
    if exist "models\epl_result_model.joblib" (
        echo ✅ ML model trained
    ) else (
        echo ⚠️  ML model not found - run training
    )
    
    if exist "cache" (
        echo ✅ Cache directory exists
    ) else (
        echo ⚠️  Cache directory not found
    )
    
    echo.
    echo 📊 System ready for predictions!
    goto CONTINUE_PROMPT
)

if "%setup_choice%"=="7" (
    goto MAIN_MENU
)

echo ❌ Invalid choice. Please try again.
goto SETUP_MENU

:CONTINUE_PROMPT
echo.
set /p "continue_choice=🔄 Run another action? (y/N): "
if /i "%continue_choice%"=="y" goto MAIN_MENU
if /i "%continue_choice%"=="yes" goto MAIN_MENU

:END
echo.
echo 👋 Thanks for using EPL Predictor! Good luck! ⚽
pause
