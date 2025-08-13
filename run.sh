#!/bin/bash
# EPL Prediction System Launcher
# Automatically activates virtual environment and runs the system

# Change to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment and run the system
source .venv/bin/activate
python main.py "$@"
