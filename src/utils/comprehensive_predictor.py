#!/usr/bin/env python3
"""
Comprehensive EPL match prediction system with all enhancements.
Includes new signings, transfers, current form, injuries, and advanced analytics.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import requests
import json

class ComprehensiveEPLPredictor:
    """
    Complete EPL prediction system integrating all improvements:
    - Real-time squad validation
    - New signings detection (Wirtz, etc.)
    - Transfer validation
    - Enhanced scoreline prediction
    - Form analysis
    - Injury impact
    - Opponent-specific adjustments
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        
        # Initialize all sub-systems
        try:
            from enhanced_player_analyzer import EnhancedPlayerAnalyzer
            self.player_analyzer = EnhancedPlayerAnalyzer(api_keys)
            print("✅ Enhanced Player Analyzer loaded")
        except ImportError:
            self.player_analyzer = None
            print("⚠️  Enhanced Player Analyzer not available")
        
        try:
            from squad_validator import RealTimeSquadValidator
            self.squad_validator = RealTimeSquadValidator(api_keys.get('api_football', ''))
            print("✅ Real-time Squad Validator loaded")
        except ImportError:
            self.squad_validator = None
            print("⚠️  Squad Validator not available")
        
        try:
            from enhanced_predictions import EnhancedPredictionEngine
            self.prediction_engine = EnhancedPredictionEngine(api_keys)
            print("✅ Enhanced Prediction Engine loaded")
        except ImportError:
            self.prediction_engine = None
            print("⚠️  Enhanced Prediction Engine not available")
    
    def predict_match_comprehensive(self, home_team: str, away_team: str, 
                                  include_detailed_analysis: bool = True) -> Dict:
        """
        Comprehensive match prediction with all enhancements.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            include_detailed_analysis: Whether to include detailed breakdowns
            
        Returns:
            Dict: Complete prediction with all analysis
        """
        print(f"\n🔍 Comprehensive Analysis: {home_team} vs {away_team}")
        print("=" * 60)
        
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'prediction_timestamp': datetime.now().isoformat(),
            'match_result': {},
            'scoreline': {},
            'goal_scorers': {},
            'analysis': {},
            'confidence': {},
            'warnings': []
        }
        
        # 1. Enhanced Squad Analysis
        if self.squad_validator:
            try:
                home_squad_analysis = self.squad_validator.validate_team_squad(home_team, detailed=True)
                away_squad_analysis = self.squad_validator.validate_team_squad(away_team, detailed=True)
                
                result['squad_analysis'] = {
                    'home': home_squad_analysis,
                    'away': away_squad_analysis
                }
                
                # Check for new signings
                home_new_signings = home_squad_analysis.get('new_signings', [])
                away_new_signings = away_squad_analysis.get('new_signings', [])
                
                if home_new_signings:
                    print(f"🆕 {home_team} new signings detected: {', '.join([p['name'] for p in home_new_signings])}")
                if away_new_signings:
                    print(f"🆕 {away_team} new signings detected: {', '.join([p['name'] for p in away_new_signings])}")
                
            except Exception as e:
                result['warnings'].append(f"Squad analysis failed: {e}")
                print(f"⚠️  Squad analysis failed: {e}")
        
        # 2. Enhanced Player Predictions
        if self.player_analyzer:
            try:
                # Mock historical data (in real implementation, this comes from your existing system)
                home_historical = self._get_mock_historical_players(home_team)
                away_historical = self._get_mock_historical_players(away_team)
                
                # Get enhanced predictions
                home_predictions = self.player_analyzer.get_enhanced_player_predictions(
                    home_team, home_historical
                )
                away_predictions = self.player_analyzer.get_enhanced_player_predictions(
                    away_team, away_historical
                )
                
                result['goal_scorers'] = {
                    'home': home_predictions,
                    'away': away_predictions
                }
                
                # Display top scorers
                print(f"\n⚽ Top Goal Scorer Predictions:")
                print(f"{home_team}:")
                for i, player in enumerate(home_predictions[:3], 1):
                    prob_pct = player['scoring_probability'] * 100
                    player_type = player.get('player_type', 'existing')
                    type_indicator = "🆕" if player_type == 'new_signing' else "📊"
                    print(f"  {i}. {type_indicator} {player['name']} ({player.get('position', 'N/A')}) - {prob_pct:.1f}%")
                
                print(f"{away_team}:")
                for i, player in enumerate(away_predictions[:3], 1):
                    prob_pct = player['scoring_probability'] * 100
                    player_type = player.get('player_type', 'existing')
                    type_indicator = "🆕" if player_type == 'new_signing' else "📊"
                    print(f"  {i}. {type_indicator} {player['name']} ({player.get('position', 'N/A')}) - {prob_pct:.1f}%")
                
                # Team attacking analysis
                home_attack_analysis = self.player_analyzer.analyze_team_attacking_changes(
                    home_team, home_historical
                )
                away_attack_analysis = self.player_analyzer.analyze_team_attacking_changes(
                    away_team, away_historical
                )
                
                result['attacking_analysis'] = {
                    'home': home_attack_analysis,
                    'away': away_attack_analysis
                }
                
                print(f"\n📈 Attacking Changes:")
                print(f"{home_team}: {home_attack_analysis['outlook']} ({home_attack_analysis['percentage_change']:+.1f}%)")
                for change in home_attack_analysis.get('key_changes', []):
                    print(f"  • {change}")
                
                print(f"{away_team}: {away_attack_analysis['outlook']} ({away_attack_analysis['percentage_change']:+.1f}%)")
                for change in away_attack_analysis.get('key_changes', []):
                    print(f"  • {change}")
                
            except Exception as e:
                result['warnings'].append(f"Player analysis failed: {e}")
                print(f"⚠️  Player analysis failed: {e}")
        
        # 3. Enhanced Match Prediction
        if self.prediction_engine:
            try:
                # Mock input data (in real implementation, this comes from your existing system)
                mock_match_data = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_form': {'ppg': 2.1, 'gf_avg': 2.3, 'ga_avg': 1.1},
                    'away_form': {'ppg': 1.8, 'gf_avg': 1.9, 'ga_avg': 1.4},
                    'h2h_recent': {'meetings': 5, 'home_wins': 2, 'draws': 2, 'away_wins': 1}
                }
                
                enhanced_prediction = self.prediction_engine.predict_match_comprehensive(
                    mock_match_data
                )
                
                result['match_result'] = enhanced_prediction.get('match_result', {})
                result['scoreline'] = enhanced_prediction.get('scoreline', {})
                result['confidence'] = enhanced_prediction.get('confidence', {})
                
                # Display match prediction
                print(f"\n🎯 Match Prediction:")
                match_pred = enhanced_prediction.get('match_result', {})
                if match_pred:
                    home_win_pct = match_pred.get('home_win', 0) * 100
                    draw_pct = match_pred.get('draw', 0) * 100
                    away_win_pct = match_pred.get('away_win', 0) * 100
                    
                    print(f"  {home_team} Win: {home_win_pct:.1f}%")
                    print(f"  Draw: {draw_pct:.1f}%")
                    print(f"  {away_team} Win: {away_win_pct:.1f}%")
                
                scoreline_pred = enhanced_prediction.get('scoreline', {})
                if scoreline_pred:
                    most_likely = scoreline_pred.get('most_likely', '1-1')
                    probability = scoreline_pred.get('probability', 0) * 100
                    print(f"  Most Likely Score: {most_likely} ({probability:.1f}%)")
                
            except Exception as e:
                result['warnings'].append(f"Enhanced prediction failed: {e}")
                print(f"⚠️  Enhanced prediction failed: {e}")
        
        # 4. Summary and Recommendations
        print(f"\n💡 Key Insights:")
        
        # New signings impact
        if result.get('squad_analysis', {}).get('home', {}).get('new_signings'):
            home_new = result['squad_analysis']['home']['new_signings']
            attacking_new = [p for p in home_new if any(pos in p.get('position', '').upper() 
                           for pos in ['FW', 'AM', 'LW', 'RW'])]
            if attacking_new:
                print(f"  • {home_team} has {len(attacking_new)} new attacking signing(s) - increased goal threat")
        
        if result.get('squad_analysis', {}).get('away', {}).get('new_signings'):
            away_new = result['squad_analysis']['away']['new_signings']
            attacking_new = [p for p in away_new if any(pos in p.get('position', '').upper() 
                           for pos in ['FW', 'AM', 'LW', 'RW'])]
            if attacking_new:
                print(f"  • {away_team} has {len(attacking_new)} new attacking signing(s) - increased goal threat")
        
        # Attacking changes impact
        home_attack = result.get('attacking_analysis', {}).get('home', {})
        away_attack = result.get('attacking_analysis', {}).get('away', {})
        
        if home_attack.get('percentage_change', 0) > 15:
            print(f"  • {home_team} significantly strengthened attack (+{home_attack['percentage_change']:.1f}%)")
        elif home_attack.get('percentage_change', 0) < -15:
            print(f"  • {home_team} weakened attack ({home_attack['percentage_change']:.1f}%)")
        
        if away_attack.get('percentage_change', 0) > 15:
            print(f"  • {away_team} significantly strengthened attack (+{away_attack['percentage_change']:.1f}%)")
        elif away_attack.get('percentage_change', 0) < -15:
            print(f"  • {away_team} weakened attack ({away_attack['percentage_change']:.1f}%)")
        
        # Confidence assessment
        total_warnings = len(result.get('warnings', []))
        if total_warnings == 0:
            confidence_level = "High"
        elif total_warnings <= 2:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        result['confidence']['overall'] = confidence_level
        print(f"\n🎯 Prediction Confidence: {confidence_level}")
        
        if result.get('warnings'):
            print(f"⚠️  Warnings: {len(result['warnings'])} issues detected")
        
        return result
    
    def _get_mock_historical_players(self, team_name: str) -> List[Dict]:
        """Mock historical player data - replace with real data integration."""
        # This is simplified mock data - in real implementation, integrate with your existing player stats
        mock_data = {
            'liverpool': [
                {'name': 'Mohamed Salah', 'goals': 18, 'matches': 32, 'xg': 20.5, 'position': 'RW'},
                {'name': 'Luis Díaz', 'goals': 13, 'matches': 30, 'xg': 12.0, 'position': 'LW'},
                {'name': 'Dominik Szoboszlai', 'goals': 6, 'matches': 28, 'xg': 7.3, 'position': 'AM'},
                {'name': 'Darwin Núñez', 'goals': 11, 'matches': 29, 'xg': 14.2, 'position': 'CF'}
            ],
            'chelsea': [
                {'name': 'João Pedro', 'goals': 10, 'matches': 25, 'xg': 12.1, 'position': 'FW'},  # Now at Chelsea
                {'name': 'Cole Palmer', 'goals': 15, 'matches': 31, 'xg': 16.8, 'position': 'AM'},
                {'name': 'Nicolas Jackson', 'goals': 12, 'matches': 28, 'xg': 13.5, 'position': 'CF'},
                {'name': 'Raheem Sterling', 'goals': 8, 'matches': 24, 'xg': 9.2, 'position': 'LW'}  # Now at Arsenal
            ],
            'arsenal': [
                {'name': 'Bukayo Saka', 'goals': 14, 'matches': 30, 'xg': 15.2, 'position': 'RW'},
                {'name': 'Gabriel Jesus', 'goals': 8, 'matches': 22, 'xg': 11.4, 'position': 'CF'},
                {'name': 'Martin Ødegaard', 'goals': 10, 'matches': 28, 'xg': 8.9, 'position': 'AM'},
                {'name': 'Kai Havertz', 'goals': 12, 'matches': 29, 'xg': 10.8, 'position': 'CF'}
            ]
        }
        
        team_key = team_name.lower()
        historical = mock_data.get(team_key, [])
        
        # Convert to expected format
        for player in historical:
            if 'goals_per_game' not in player:
                player['goals_per_game'] = player['goals'] / player['matches'] if player['matches'] > 0 else 0
            if 'assists' not in player:
                player['assists'] = max(1, player['goals'] // 3)  # Mock assists
            if 'scoring_probability' not in player:
                player['scoring_probability'] = min(player['goals_per_game'] * 0.4, 0.4)
        
        return historical

# Example usage and testing
if __name__ == "__main__":
    # Test the comprehensive system
    predictor = ComprehensiveEPLPredictor({
        'api_football': 'your_api_key_here',  # Replace with real key
        'fbr': 'your_fbr_key_here'
    })
    
    # Test with Liverpool vs Chelsea (includes new signings)
    print("Testing comprehensive prediction system...")
    result = predictor.predict_match_comprehensive("Liverpool", "Chelsea")
    
    print("\n" + "="*60)
    print("🎉 Comprehensive prediction completed!")
    print(f"Confidence: {result['confidence'].get('overall', 'Unknown')}")
    print(f"Warnings: {len(result.get('warnings', []))}")
    
    # Save result for debugging
    with open('comprehensive_prediction_test.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
