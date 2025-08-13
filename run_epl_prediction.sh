#!/bin/bash

# ⚽ EPL Prediction Runner - Cross-Platform Shell Script
# Works on Linux, Mac, and Windows (with Git Bash/WSL)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    echo -e "${1}${2}${NC}"
}

# Function to print section headers
print_header() {
    echo
    print_color $CYAN "================================================================"
    print_color $CYAN "⚽ $1"
    print_color $CYAN "================================================================"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "mac"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to activate virtual environment
activate_venv() {
    local os=$(detect_os)
    
    if [ -d ".venv" ]; then
        print_color $GREEN "📦 Activating virtual environment..."
        
        case $os in
            "windows")
                source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
                ;;
            *)
                source .venv/bin/activate
                ;;
        esac
        
        print_color $GREEN "✅ Virtual environment activated"
    else
        print_color $YELLOW "⚠️  Virtual environment not found. Creating one..."
        create_venv
    fi
}

# Function to create virtual environment
create_venv() {
    print_color $BLUE "🛠️  Creating virtual environment..."
    
    if command_exists python3; then
        python3 -m venv .venv
    elif command_exists python; then
        python -m venv .venv
    else
        print_color $RED "❌ Python not found! Please install Python 3.7+"
        exit 1
    fi
    
    activate_venv
    
    print_color $BLUE "📦 Installing dependencies..."
    pip install --upgrade pip
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        pip install numpy pandas scikit-learn requests joblib
    fi
    
    print_color $GREEN "✅ Virtual environment created and dependencies installed"
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_color $RED "❌ Python not found! Please install Python 3.7+"
        exit 1
    fi
    
    # Check Python version
    python_version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    major_version=$(echo $python_version | cut -d'.' -f1)
    minor_version=$(echo $python_version | cut -d'.' -f2)
    
    if [[ $major_version -lt 3 ]] || [[ $major_version -eq 3 && $minor_version -lt 7 ]]; then
        print_color $RED "❌ Python 3.7+ required. Found: $python_version"
        exit 1
    fi
    
    print_color $GREEN "✅ Python $python_version detected"
}

# Function to check API keys
check_api_keys() {
    print_color $BLUE "🔑 Checking API keys..."
    
    keys_missing=false
    
    if [ -z "$FBR_API_KEY" ]; then
        print_color $YELLOW "⚠️  FBR_API_KEY not set"
        keys_missing=true
    else
        print_color $GREEN "✅ FBR_API_KEY is set"
    fi
    
    if [ -z "$API_FOOTBALL_KEY" ]; then
        print_color $YELLOW "⚠️  API_FOOTBALL_KEY not set"
        keys_missing=true
    else
        print_color $GREEN "✅ API_FOOTBALL_KEY is set"
    fi
    
    if [ -z "$BOOKMAKER_API_KEY" ]; then
        print_color $YELLOW "⚠️  BOOKMAKER_API_KEY not set"
        keys_missing=true
    else
        print_color $GREEN "✅ BOOKMAKER_API_KEY is set"
    fi
    
    if [ "$keys_missing" = true ]; then
        print_color $CYAN "💡 Set up API keys:"
        echo "   export FBR_API_KEY=\"your_fbr_key\""
        echo "   export API_FOOTBALL_KEY=\"02eb00e7497de4d328fa72e3365791b5\""
        echo "   export BOOKMAKER_API_KEY=\"e66a648eb21c685297c1df4c8e0304cc\""
        echo
        print_color $CYAN "🔧 Generate FBR key with: $PYTHON_CMD epl_prediction.py generate-key"
    fi
}

# Function to check if first time setup is needed
check_setup() {
    if [ ! -f "epl_prediction.py" ]; then
        print_color $RED "❌ epl_prediction.py not found! Make sure you're in the correct directory."
        exit 1
    fi
    
    if [ ! -d "cache" ] || [ ! -d "models" ] || [ ! -f "models/epl_result_model.joblib" ]; then
        print_color $YELLOW "🚀 First-time setup required!"
        return 1
    fi
    
    return 0
}

# Function to run first-time setup
first_time_setup() {
    print_header "FIRST-TIME SETUP"
    
    print_color $BLUE "1️⃣  Generating FBR API key..."
    $PYTHON_CMD epl_prediction.py generate-key
    
    print_color $BLUE "2️⃣  Syncing match data (3 seasons)..."
    $PYTHON_CMD epl_prediction.py sync --seasons 3
    
    print_color $BLUE "3️⃣  Syncing enhanced data (players & team stats)..."
    $PYTHON_CMD epl_prediction.py sync-enhanced --seasons 2
    
    print_color $BLUE "4️⃣  Training ML model..."
    $PYTHON_CMD epl_prediction.py train
    
    print_color $GREEN "✅ Setup complete! You're ready to make predictions!"
}

# Function to show quick actions menu
show_quick_menu() {
    print_header "EPL PREDICTION QUICK LAUNCHER"
    
    echo "Choose an action:"
    echo
    print_color $GREEN "1️⃣  🎯 Predict next EPL match"
    print_color $GREEN "2️⃣  📅 Show upcoming fixtures predictions"
    print_color $GREEN "3️⃣  ⚔️  Predict specific match"
    print_color $GREEN "4️⃣  🔄 Update all data"
    print_color $GREEN "5️⃣  🖥️  Interactive mode (full menu)"
    print_color $GREEN "6️⃣  🛠️  Setup/Maintenance"
    print_color $GREEN "7️⃣  ❓ Help"
    print_color $GREEN "8️⃣  🚪 Exit"
    echo
    print_color $CYAN "Enter your choice (1-8): "
    read -r choice
    
    case $choice in
        1)
            print_header "NEXT MATCH PREDICTION"
            $PYTHON_CMD epl_prediction.py --next
            ;;
        2)
            print_header "UPCOMING FIXTURES"
            print_color $CYAN "How many fixtures to predict? (default: 5): "
            read -r num_fixtures
            num_fixtures=${num_fixtures:-5}
            $PYTHON_CMD epl_prediction.py predict-fixtures --top $num_fixtures
            ;;
        3)
            print_header "SPECIFIC MATCH PREDICTION"
            print_color $CYAN "Home team: "
            read -r home_team
            print_color $CYAN "Away team: "
            read -r away_team
            $PYTHON_CMD epl_prediction.py predict-match --home "$home_team" --away "$away_team"
            ;;
        4)
            print_header "UPDATING ALL DATA"
            $PYTHON_CMD epl_prediction.py --update
            ;;
        5)
            print_header "INTERACTIVE MODE"
            $PYTHON_CMD epl_prediction.py --interactive
            ;;
        6)
            setup_menu
            ;;
        7)
            print_header "HELP"
            $PYTHON_CMD epl_prediction.py --help
            ;;
        8)
            print_color $GREEN "👋 Goodbye!"
            exit 0
            ;;
        *)
            print_color $RED "❌ Invalid choice. Please try again."
            show_quick_menu
            ;;
    esac
}

# Function to show setup/maintenance menu
setup_menu() {
    print_header "SETUP & MAINTENANCE"
    
    echo "Choose an action:"
    echo
    print_color $YELLOW "1️⃣  🔑 Generate FBR API key"
    print_color $YELLOW "2️⃣  📥 Sync match data"
    print_color $YELLOW "3️⃣  📊 Sync enhanced data"
    print_color $YELLOW "4️⃣  🧠 Train ML model"
    print_color $YELLOW "5️⃣  🧹 Clean cache"
    print_color $YELLOW "6️⃣  🔍 Check system status"
    print_color $YELLOW "7️⃣  🔙 Back to main menu"
    echo
    print_color $CYAN "Enter your choice (1-7): "
    read -r choice
    
    case $choice in
        1)
            $PYTHON_CMD epl_prediction.py generate-key
            ;;
        2)
            print_color $CYAN "Number of seasons to sync (default: 3): "
            read -r seasons
            seasons=${seasons:-3}
            $PYTHON_CMD epl_prediction.py sync --seasons $seasons
            ;;
        3)
            print_color $CYAN "Number of seasons for enhanced data (default: 2): "
            read -r seasons
            seasons=${seasons:-2}
            $PYTHON_CMD epl_prediction.py sync-enhanced --seasons $seasons
            ;;
        4)
            $PYTHON_CMD epl_prediction.py train
            ;;
        5)
            print_color $YELLOW "🧹 Cleaning cache..."
            rm -rf cache/*
            print_color $GREEN "✅ Cache cleaned"
            ;;
        6)
            check_system_status
            ;;
        7)
            show_quick_menu
            ;;
        *)
            print_color $RED "❌ Invalid choice. Please try again."
            setup_menu
            ;;
    esac
}

# Function to check system status
check_system_status() {
    print_header "SYSTEM STATUS"
    
    # Check Python
    check_python
    
    # Check virtual environment
    if [ -d ".venv" ]; then
        print_color $GREEN "✅ Virtual environment exists"
    else
        print_color $YELLOW "⚠️  Virtual environment not found"
    fi
    
    # Check main file
    if [ -f "epl_prediction.py" ]; then
        print_color $GREEN "✅ Main script found"
    else
        print_color $RED "❌ epl_prediction.py not found"
    fi
    
    # Check model
    if [ -f "models/epl_result_model.joblib" ]; then
        print_color $GREEN "✅ ML model trained"
    else
        print_color $YELLOW "⚠️  ML model not found - run training"
    fi
    
    # Check cache
    if [ -d "cache" ] && [ "$(ls -A cache)" ]; then
        print_color $GREEN "✅ Cache directory has data"
        cache_size=$(du -sh cache 2>/dev/null | cut -f1)
        echo "   Cache size: $cache_size"
    else
        print_color $YELLOW "⚠️  Cache is empty - run sync"
    fi
    
    # Check API keys
    check_api_keys
    
    echo
    print_color $CYAN "📊 System ready for predictions!"
}

# Main function
main() {
    # Clear screen for better presentation
    clear
    
    print_header "EPL PREDICTION SYSTEM LAUNCHER"
    print_color $PURPLE "🚀 Cross-platform runner for Linux, Mac & Windows"
    print_color $PURPLE "📍 OS Detected: $(detect_os)"
    
    # Check Python
    check_python
    
    # Navigate to script directory
    cd "$(dirname "$0")"
    
    # Activate virtual environment
    activate_venv
    
    # Check if setup is needed
    if ! check_setup; then
        print_color $YELLOW "🔧 Running first-time setup..."
        first_time_setup
    fi
    
    # Launch the beautiful interactive mode directly
    print_color $GREEN "� Launching EPL Prediction Interactive Mode..."
    echo
    $PYTHON_CMD epl_prediction.py --interactive
}

# Handle script arguments
if [ $# -eq 0 ]; then
    # No arguments - run interactive mode
    main
else
    # Arguments provided - pass through to Python script
    cd "$(dirname "$0")"
    check_python
    activate_venv
    $PYTHON_CMD epl_prediction.py "$@"
fi
