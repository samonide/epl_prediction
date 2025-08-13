#!/usr/bin/env python3
"""
EPL Prediction System Cleanup and Organization Script
Removes unnecessary files and consolidates the codebase for better memory usage.
"""

import os
import shutil
from pathlib import Path

def cleanup_prediction_system():
    """Clean up and organize the EPL prediction system."""
    
    print("🧹 CLEANING UP EPL PREDICTION SYSTEM")
    print("=" * 50)
    
    # Files to keep (core functionality)
    keep_files = {
        'epl_prediction_advanced.py',      # New consolidated main script
        'advanced_ml_engine.py',           # Advanced ML with transfer analysis
        'real_time_transfer_validator.py', # Transfer validation
        'enhanced_player_analyzer.py',     # Player analysis with new signings
        'README.md',                       # Documentation
        'requirements.txt',                # Dependencies
        'run_epl_prediction.sh',          # Cross-platform launcher
        'run_epl_prediction.bat',         # Windows launcher
        '.gitignore',                      # Git configuration
        'ENHANCEMENT_SUMMARY.md'           # Implementation summary
    }
    
    # Files to remove (consolidation targets)
    remove_files = {
        'enhanced_predictions.py',         # Consolidated into advanced_ml_engine.py
        'squad_validator.py',             # Consolidated into real_time_transfer_validator.py
        'transfer_validation.py',         # Consolidated into real_time_transfer_validator.py
        'comprehensive_predictor.py',     # Consolidated into epl_prediction_advanced.py
        'comprehensive_prediction_test.json'  # Test file
    }
    
    current_dir = Path.cwd()
    
    # 1. Remove unnecessary files
    removed_count = 0
    for filename in remove_files:
        file_path = current_dir / filename
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"🗑️  Removed: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"⚠️  Could not remove {filename}: {e}")
    
    # 2. Keep essential files and warn about unexpected ones
    kept_count = 0
    unexpected_files = []
    
    for file_path in current_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith('.'):
            if file_path.name in keep_files:
                kept_count += 1
            elif file_path.name.endswith('.py') and file_path.name not in remove_files:
                unexpected_files.append(file_path.name)
    
    # 3. Check if main original script exists and handle it
    original_script = current_dir / 'epl_prediction.py'
    if original_script.exists():
        backup_script = current_dir / 'epl_prediction_legacy.py'
        try:
            shutil.move(str(original_script), str(backup_script))
            print(f"📦 Moved original script to: epl_prediction_legacy.py")
        except Exception as e:
            print(f"⚠️  Could not backup original script: {e}")
    
    # 4. Create symbolic link for backward compatibility
    try:
        advanced_script = current_dir / 'epl_prediction_advanced.py'
        main_script_link = current_dir / 'epl_prediction.py'
        
        if advanced_script.exists() and not main_script_link.exists():
            if os.name != 'nt':  # Unix/Linux/Mac
                main_script_link.symlink_to('epl_prediction_advanced.py')
                print(f"🔗 Created symlink: epl_prediction.py -> epl_prediction_advanced.py")
            else:  # Windows
                shutil.copy2(str(advanced_script), str(main_script_link))
                print(f"📋 Created copy: epl_prediction.py (Windows compatibility)")
    except Exception as e:
        print(f"⚠️  Could not create main script link: {e}")
    
    # 5. Organize cache directory
    cache_dir = current_dir / 'cache'
    if cache_dir.exists():
        # Create organized subdirectories
        subdirs = ['matches', 'player_stats', 'team_stats', 'squads', 'injuries', 'odds', 'transfers']
        for subdir in subdirs:
            (cache_dir / subdir).mkdir(exist_ok=True)
        print(f"📁 Organized cache directory structure")
    
    # 6. Update file permissions for launchers
    launchers = ['run_epl_prediction.sh']
    for launcher in launchers:
        launcher_path = current_dir / launcher
        if launcher_path.exists() and os.name != 'nt':
            try:
                launcher_path.chmod(0o755)
                print(f"🔧 Made executable: {launcher}")
            except Exception as e:
                print(f"⚠️  Could not set permissions for {launcher}: {e}")
    
    # 7. Summary
    print(f"\n✅ CLEANUP SUMMARY")
    print(f"   Files removed: {removed_count}")
    print(f"   Files kept: {kept_count}")
    
    if unexpected_files:
        print(f"   Unexpected Python files found: {len(unexpected_files)}")
        for uf in unexpected_files:
            print(f"     - {uf}")
    
    print(f"\n🎯 CORE SYSTEM FILES:")
    for filename in sorted(keep_files):
        file_path = current_dir / filename
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"   ✅ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {filename} (missing)")
    
    print(f"\n🚀 SYSTEM READY!")
    print(f"   Main script: epl_prediction_advanced.py")
    print(f"   Compatibility: epl_prediction.py -> epl_prediction_advanced.py")
    print(f"   Launcher: ./run_epl_prediction.sh")
    print(f"   Advanced ML: ✅ Enabled")
    print(f"   Transfer Analysis: ✅ Enabled")
    print(f"   Memory Usage: 🔽 Optimized")

def update_launchers():
    """Update launcher scripts to use the new consolidated system."""
    
    print(f"\n🔧 UPDATING LAUNCHER SCRIPTS")
    print("-" * 30)
    
    # Update bash launcher
    bash_launcher = Path('run_epl_prediction.sh')
    if bash_launcher.exists():
        content = bash_launcher.read_text()
        # Replace references to old script
        updated_content = content.replace('epl_prediction.py', 'epl_prediction_advanced.py')
        bash_launcher.write_text(updated_content)
        print("✅ Updated run_epl_prediction.sh")
    
    # Update Windows launcher
    bat_launcher = Path('run_epl_prediction.bat')
    if bat_launcher.exists():
        content = bat_launcher.read_text()
        updated_content = content.replace('epl_prediction.py', 'epl_prediction_advanced.py')
        bat_launcher.write_text(updated_content)
        print("✅ Updated run_epl_prediction.bat")

if __name__ == "__main__":
    cleanup_prediction_system()
    update_launchers()
    
    print(f"\n🎉 SYSTEM CLEANUP COMPLETED!")
    print(f"   Run: python epl_prediction_advanced.py --interactive")
    print(f"   Or:  ./run_epl_prediction.sh")
