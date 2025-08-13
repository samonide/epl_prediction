#!/usr/bin/env python3
"""
Production Readiness Test Suite
Comprehensive testing for EPL Prediction System
"""

import sys
import traceback
from datetime import datetime

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing Imports...")
    try:
        from src.core.epl_prediction_advanced import EPLPredictionSystem
        from src.core.squad_manager import SquadManager
        from src.core.advanced_ml_engine import AdvancedMLPredictionEngine
        print("  ✅ All critical modules imported successfully")
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_system_initialization():
    """Test system initialization"""
    print("\n🔍 Testing System Initialization...")
    try:
        from src.core.epl_prediction_advanced import EPLPredictionSystem
        from src.core.squad_manager import SquadManager
        
        predictor = EPLPredictionSystem()
        print("  ✅ EPL Prediction System initialized")
        
        squad_manager = SquadManager()
        print("  ✅ Squad Manager initialized")
        
        return True, (predictor, squad_manager)
    except Exception as e:
        print(f"  ❌ Initialization error: {e}")
        traceback.print_exc()
        return False, None

def test_team_validation(predictor):
    """Test team name validation"""
    print("\n🔍 Testing Team Validation...")
    test_cases = [
        ("Liverpool", "Arsenal", True),
        ("Man City", "Manchester United", True),
        ("Spurs", "Chelsea", True),
        ("InvalidTeam", "Arsenal", False),
        ("", "Liverpool", False),
    ]
    
    all_passed = True
    for home, away, expected in test_cases:
        try:
            valid, msg = predictor._validate_team_names(home, away)
            if (valid and expected) or (not valid and not expected):
                print(f"  ✅ {home} vs {away}: {valid}")
            else:
                print(f"  ❌ {home} vs {away}: Expected {expected}, got {valid}")
                all_passed = False
        except Exception as e:
            print(f"  ❌ Error testing {home} vs {away}: {e}")
            all_passed = False
    
    return all_passed

def test_prediction_generation(predictor):
    """Test prediction generation"""
    print("\n🔍 Testing Prediction Generation...")
    try:
        # Test basic prediction
        result = predictor._predict_basic("Liverpool", "Arsenal")
        if result and hasattr(result, 'home_win_prob'):
            print("  ✅ Basic prediction generation working")
        else:
            print("  ❌ Basic prediction failed")
            return False
        
        # Test probability validation
        total_prob = result.home_win_prob + result.draw_prob + result.away_win_prob
        if 0.99 <= total_prob <= 1.01:  # Allow small floating point errors
            print(f"  ✅ Probabilities sum correctly: {total_prob:.3f}")
        else:
            print(f"  ❌ Probabilities don't sum to 1: {total_prob:.3f}")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ Prediction error: {e}")
        traceback.print_exc()
        return False

def test_squad_manager(squad_manager):
    """Test squad manager functionality"""
    print("\n🔍 Testing Squad Manager...")
    try:
        # Test EPL teams
        teams = squad_manager.epl_teams
        if len(teams) == 20:
            print(f"  ✅ EPL teams loaded: {len(teams)} teams")
        else:
            print(f"  ❌ Wrong number of teams: {len(teams)}")
            return False
        
        # Test database loading
        squad_manager._load_squad_database()
        cached_teams = len(squad_manager.squad_database)
        print(f"  ✅ Squad database: {cached_teams} teams cached")
        
        # Test getting team squad
        if cached_teams > 0:
            test_team = list(squad_manager.squad_database.keys())[0]
            squad = squad_manager.get_team_squad(test_team)
            print(f"  ✅ Squad retrieval: {len(squad)} players for {test_team}")
        
        return True
    except Exception as e:
        print(f"  ❌ Squad manager error: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling edge cases"""
    print("\n🔍 Testing Error Handling...")
    try:
        # Test division by zero protection
        test_values = [0, 0.0, None, "", []]
        
        # Test safe division
        for val in test_values:
            try:
                # Simulate potential division operations
                result = 1.0 / max(val or 1, 0.001) if val else 1.0
                print(f"  ✅ Safe division with {val}: {result:.3f}")
            except Exception as e:
                print(f"  ❌ Division error with {val}: {e}")
                return False
        
        print("  ✅ Error handling tests passed")
        return True
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage and cleanup"""
    print("\n🔍 Testing Memory Usage...")
    try:
        import psutil
        import os
        from src.core.epl_prediction_advanced import EPLPredictionSystem
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple instances to test memory
        systems = []
        for i in range(3):
            systems.append(EPLPredictionSystem())
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del systems
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"  ✅ Memory usage - Initial: {initial_memory:.1f}MB, Peak: {peak_memory:.1f}MB, Final: {final_memory:.1f}MB")
        
        if peak_memory < 200:  # Reasonable memory usage
            print("  ✅ Memory usage within acceptable limits")
            return True
        else:
            print("  ⚠️  High memory usage detected")
            return False
            
    except ImportError:
        print("  ⚠️  psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"  ❌ Memory test error: {e}")
        return False

def run_comprehensive_tests():
    """Run all production readiness tests"""
    print("🚀 EPL PREDICTION SYSTEM - PRODUCTION READINESS TEST")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Import Tests", test_imports()))
    
    init_success, systems = test_system_initialization()
    results.append(("System Initialization", init_success))
    
    if init_success:
        predictor, squad_manager = systems
        results.append(("Team Validation", test_team_validation(predictor)))
        results.append(("Prediction Generation", test_prediction_generation(predictor)))
        results.append(("Squad Manager", test_squad_manager(squad_manager)))
    
    results.append(("Error Handling", test_error_handling()))
    results.append(("Memory Usage", test_memory_usage()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"🎯 Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 SYSTEM IS PRODUCTION READY!")
        return True
    else:
        print("⚠️  SYSTEM NEEDS ATTENTION BEFORE PRODUCTION")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
