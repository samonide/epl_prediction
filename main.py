#!/usr/bin/env python3
"""
EPL Prediction System - Main Entry Point
Professional Football Analytics & Prediction Engine

Usage:
    python main.py                 # Interactive mode
    python main.py --predict       # Quick prediction mode
    python main.py --squad         # Squad management mode
    python main.py --test          # Run system tests
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Main entry point for the EPL Prediction System"""
    
    parser = argparse.ArgumentParser(
        description='EPL Prediction System - Advanced Football Analytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start interactive mode
  python main.py --predict          # Quick prediction mode
  python main.py --squad            # Squad management
  python main.py --test             # Run tests
  python main.py --version          # Show version
        """
    )
    
    parser.add_argument('--predict', action='store_true', 
                       help='Start in prediction mode')
    parser.add_argument('--squad', action='store_true',
                       help='Start squad management mode')
    parser.add_argument('--test', action='store_true',
                       help='Run production tests')
    parser.add_argument('--version', action='store_true',
                       help='Show version information')
    
    args = parser.parse_args()
    
    if args.version:
        from src import __version__
        print(f"EPL Prediction System v{__version__}")
        return
    
    if args.test:
        print("🧪 Running production tests...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from tests.production_test import run_comprehensive_tests
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    
    # Import the main system
    try:
        from src.core.epl_prediction_advanced import EPLPredictionSystem
        from src.core.squad_manager import SquadManager
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    
    try:
        if args.squad:
            print("🏆 Starting Squad Management Mode...")
            manager = SquadManager()
            manager.interactive_menu()
        elif args.predict:
            print("⚽ Starting Prediction Mode...")
            predictor = EPLPredictionSystem()
            predictor.interactive_mode()
        else:
            # Default interactive mode
            predictor = EPLPredictionSystem()
            predictor.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using EPL Prediction System!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("For support, please check the documentation in docs/")
        sys.exit(1)

if __name__ == "__main__":
    main()
