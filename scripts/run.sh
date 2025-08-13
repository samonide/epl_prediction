#!/bin/bash
# EPL Prediction System Launcher (Linux/macOS)

echo "🚀 Starting EPL Prediction System..."

# Navigate to the project directory
cd "$(dirname "$0")/.."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8 or later"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application
echo "⚽ Launching EPL Prediction System..."
python main.py "$@"

echo "👋 Thanks for using EPL Prediction System!"
