#!/usr/bin/env python3
"""
Enhanced player analysis with new signing detection and updated scoring probabilities.
Includes recent transfers like Wirtz to Liverpool and other high-profile moves.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import requests

class EnhancedPlayerAnalyzer:
    """Advanced player analysis with real-time squad updates and form analysis."""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        
        # Enhanced new signings database with expected impact (Updated August 2025)
        self.new_signings_2024_25 = {
            'liverpool': [
                {
                    'name': 'Florian Wirtz',
                    'position': 'AM/MF',
                    'from_team': 'Bayer Leverkusen',
                    'estimated_goals_season': 15,
                    'estimated_assists_season': 12,
                    'scoring_probability_base': 0.25,  # High for attacking midfielder
                    'impact_level': 'high',
                    'age': 21,
                    'international_caps': 18
                },
                {
                    'name': 'Hugo Ekitike',
                    'position': 'ST/CF',
                    'from_team': 'Paris Saint-Germain',
                    'estimated_goals_season': 18,
                    'estimated_assists_season': 5,
                    'scoring_probability_base': 0.30,  # High for striker
                    'impact_level': 'high',
                    'age': 22,
                    'international_caps': 8
                },
                {
                    'name': 'Jeremie Frimpong',
                    'position': 'RB/RWB',
                    'from_team': 'Bayer Leverkusen',
                    'estimated_goals_season': 6,
                    'estimated_assists_season': 10,
                    'scoring_probability_base': 0.12,  # Lower for full-back
                    'impact_level': 'medium',
                    'age': 23,
                    'international_caps': 12
                },
                {
                    'name': 'Milos Kerkez',
                    'position': 'LB/LWB',
                    'from_team': 'Bournemouth',
                    'estimated_goals_season': 3,
                    'estimated_assists_season': 8,
                    'scoring_probability_base': 0.08,  # Low for left-back
                    'impact_level': 'medium',
                    'age': 21,
                    'international_caps': 5
                }
            ],
            'chelsea': [
                {
                    'name': 'João Pedro',
                    'position': 'FW/MF',
                    'from_team': 'Brighton',
                    'estimated_goals_season': 18,
                    'estimated_assists_season': 6,
                    'scoring_probability_base': 0.28,
                    'impact_level': 'high',
                    'age': 23,
                    'previous_epl_goals': 10  # From Brighton
                },
                {
                    'name': 'Jadon Sancho',
                    'position': 'LW/RW',
                    'from_team': 'Manchester United',
                    'estimated_goals_season': 10,
                    'estimated_assists_season': 12,
                    'scoring_probability_base': 0.18,
                    'impact_level': 'medium',
                    'age': 24,
                    'previous_epl_goals': 12
                }
            ],
            'arsenal': [
                {
                    'name': 'Raheem Sterling',
                    'position': 'LW/RW',
                    'from_team': 'Chelsea',
                    'estimated_goals_season': 14,
                    'estimated_assists_season': 10,
                    'scoring_probability_base': 0.22,
                    'impact_level': 'high',
                    'age': 29,
                    'previous_epl_goals': 123  # Career total
                }
            ],
            'manchester_city': [
                {
                    'name': 'Savinho',
                    'position': 'RW/LW',
                    'from_team': 'Girona',
                    'estimated_goals_season': 8,
                    'estimated_assists_season': 12,
                    'scoring_probability_base': 0.15,
                    'impact_level': 'medium',
                    'age': 20
                }
            ],
            'brighton': [
                # Lost João Pedro, might have replacements
                {
                    'name': 'Georginio Rutter',
                    'position': 'FW/AM',
                    'from_team': 'Leeds United',
                    'estimated_goals_season': 12,
                    'estimated_assists_season': 6,
                    'scoring_probability_base': 0.18,
                    'impact_level': 'medium',
                    'age': 22
                }
            ]
        }
        
        # Players who likely have reduced roles or left
        self.reduced_impact_players = {
            'joão pedro': {
                'team': 'brighton',
                'reason': 'transferred_to_chelsea',
                'probability_reduction': 1.0,  # Complete removal
                'new_role': 'not_in_squad'
            },
            'joao pedro': {
                'team': 'brighton', 
                'reason': 'transferred_to_chelsea',
                'probability_reduction': 1.0,  # Complete removal
                'new_role': 'not_in_squad'
            },
            'dominik szoboszlai': {
                'team': 'liverpool',
                'reason': 'new_signings_ahead',
                'probability_reduction': 0.3,  # 30% reduction due to Wirtz/Simons
                'new_role': 'rotation_player'
            },
            'raheem sterling': {
                'team': 'chelsea',
                'reason': 'transferred_to_arsenal',
                'probability_reduction': 1.0,  # Complete removal
                'new_role': 'not_in_squad'
            },
            'jadon sancho': {
                'team': 'manchester_united',
                'reason': 'transferred_to_chelsea',
                'probability_reduction': 1.0,
                'new_role': 'not_in_squad'
            }
        }
    
    def get_enhanced_player_predictions(self, team_name: str, historical_players: List[Dict], 
                                      h2h_data: Dict = None, team_form: Dict = None) -> List[Dict]:
        """
        Get enhanced player predictions considering new signings and transfers.
        
        Args:
            team_name: Team name
            historical_players: List of players from historical data
            h2h_data: Head-to-head statistics
            team_form: Current team form data
            
        Returns:
            List[Dict]: Enhanced player predictions with updated probabilities
        """
        team_key = self._normalize_team_name(team_name)
        enhanced_players = []
        
        # 1. Process historical players (update/remove based on transfers)
        for player in historical_players:
            player_name = player.get('name', '').lower()
            player_name_normalized = player_name.replace('ã', 'a').replace('ç', 'c')  # Handle accents
            
            # Check if player has reduced impact or transferred
            transfer_found = False
            for reduced_name, impact_info in self.reduced_impact_players.items():
                # Check both original and normalized names
                if (player_name == reduced_name or 
                    player_name_normalized == reduced_name or
                    reduced_name in player_name or 
                    player_name in reduced_name):
                    
                    # Additional team check to ensure we're removing from the right team
                    team_match = (self._normalize_team_name(team_name) == 
                                self._normalize_team_name(impact_info.get('team', '')))
                    
                    if team_match and impact_info['probability_reduction'] >= 1.0:
                        # Player transferred out - skip
                        print(f"⚠️  {player['name']} - {impact_info['reason']} - excluding from {team_name} predictions")
                        transfer_found = True
                        break
                    elif team_match:
                        # Reduce probability
                        original_prob = player.get('scoring_probability', 0.1)
                        reduced_prob = original_prob * (1 - impact_info['probability_reduction'])
                        player['scoring_probability'] = reduced_prob
                        player['form_factor'] = player.get('form_factor', 1.0) * 0.7  # Reduced form
                        player['role_change'] = impact_info['new_role']
                        print(f"📉 {player['name']} probability reduced by {impact_info['probability_reduction']*100:.0f}% - {impact_info['reason']}")
                        transfer_found = True
                        break
            
            if transfer_found and any(info['probability_reduction'] >= 1.0 
                                    for name, info in self.reduced_impact_players.items()
                                    if (player_name == name or player_name_normalized == name) and
                                       self._normalize_team_name(team_name) == self._normalize_team_name(info.get('team', ''))):
                continue  # Skip this player entirely
            
            enhanced_players.append(player)
        
        # 2. Add new signings for this team
        if team_key in self.new_signings_2024_25:
            for new_signing in self.new_signings_2024_25[team_key]:
                enhanced_player = self._create_new_signing_prediction(
                    new_signing, team_form, h2h_data
                )
                enhanced_players.append(enhanced_player)
                print(f"✅ Added new signing: {new_signing['name']} ({new_signing['position']}) - {new_signing['impact_level']} impact expected")
        
        # 3. Sort by scoring probability
        enhanced_players.sort(key=lambda x: x.get('scoring_probability', 0), reverse=True)
        
        return enhanced_players[:6]  # Return top 6 players
    
    def _create_new_signing_prediction(self, signing: Dict, team_form: Dict = None, 
                                     h2h_data: Dict = None) -> Dict:
        """Create prediction data for a new signing."""
        base_prob = signing['scoring_probability_base']
        
        # Adjust based on team form
        form_multiplier = 1.0
        if team_form:
            team_goals_avg = team_form.get('gf_avg', 1.5)
            if team_goals_avg > 2.0:
                form_multiplier = 1.2  # Good attacking team
            elif team_goals_avg < 1.0:
                form_multiplier = 0.8  # Struggling attack
        
        # Adjust for EPL experience
        experience_multiplier = 1.0
        if signing.get('previous_epl_goals', 0) > 0:
            experience_multiplier = 1.1  # EPL experience bonus
        elif signing.get('age', 25) < 23:
            experience_multiplier = 0.9  # Young player adjustment
        
        # Position-based multiplier
        position = signing['position'].upper()
        position_multiplier = 1.0
        if any(pos in position for pos in ['FW', 'CF', 'ST']):
            position_multiplier = 1.3
        elif any(pos in position for pos in ['AM', 'CAM']):
            position_multiplier = 1.15
        elif any(pos in position for pos in ['LW', 'RW']):
            position_multiplier = 1.1
        elif 'MF' in position:
            position_multiplier = 0.9
        
        final_probability = min(base_prob * form_multiplier * experience_multiplier * position_multiplier, 0.4)
        
        return {
            'name': signing['name'],
            'position': signing['position'],
            'goals': 0,  # No historical data yet
            'matches': 0,
            'goals_per_game': 0,
            'xg': signing['estimated_goals_season'] * 0.8,  # Estimated xG
            'assists': 0,
            'scoring_probability': round(final_probability, 3),
            'form_factor': 1.2,  # New signing excitement factor
            'estimated_season_goals': signing['estimated_goals_season'],
            'estimated_season_assists': signing['estimated_assists_season'],
            'impact_level': signing['impact_level'],
            'player_type': 'new_signing',
            'transfer_fee': signing.get('fee', 'Undisclosed'),
            'age': signing.get('age'),
            'previous_epl_experience': signing.get('previous_epl_goals', 0) > 0
        }
    
    def analyze_team_attacking_changes(self, team_name: str, last_season_scorers: List[Dict]) -> Dict:
        """Analyze how a team's attacking potential has changed with transfers."""
        team_key = self._normalize_team_name(team_name)
        
        # Calculate last season's attacking output
        last_season_total_goals = sum(p.get('goals', 0) for p in last_season_scorers)
        last_season_total_xg = sum(p.get('xg', 0) for p in last_season_scorers)
        
        # Calculate expected changes
        goals_lost = 0
        goals_gained = 0
        
        # Players lost
        for player in last_season_scorers:
            player_name = player.get('name', '').lower()
            if player_name in self.reduced_impact_players:
                impact_info = self.reduced_impact_players[player_name]
                if impact_info['probability_reduction'] >= 1.0:
                    goals_lost += player.get('goals', 0)
        
        # New signings
        if team_key in self.new_signings_2024_25:
            for signing in self.new_signings_2024_25[team_key]:
                goals_gained += signing['estimated_goals_season']
        
        net_goal_change = goals_gained - goals_lost
        percentage_change = (net_goal_change / max(last_season_total_goals, 1)) * 100
        
        analysis = {
            'team': team_name,
            'last_season_goals': last_season_total_goals,
            'estimated_goals_lost': goals_lost,
            'estimated_goals_gained': goals_gained,
            'net_change': net_goal_change,
            'percentage_change': percentage_change,
            'outlook': self._get_attacking_outlook(percentage_change),
            'key_changes': []
        }
        
        # Add key changes narrative
        if goals_lost > 10:
            analysis['key_changes'].append(f"Lost significant goalscoring threat ({goals_lost} goals)")
        if goals_gained > 15:
            analysis['key_changes'].append(f"Major attacking reinforcement ({goals_gained} goals expected)")
        if team_key in self.new_signings_2024_25:
            high_impact_signings = [s['name'] for s in self.new_signings_2024_25[team_key] 
                                  if s['impact_level'] == 'high']
            if high_impact_signings:
                analysis['key_changes'].append(f"High-impact signings: {', '.join(high_impact_signings)}")
        
        return analysis
    
    def get_opponent_specific_adjustments(self, player_name: str, player_team: str, 
                                        opponent_team: str, h2h_history: List[Dict] = None) -> float:
        """
        Get player-specific adjustments based on opponent and historical performance.
        
        Returns:
            float: Multiplier for scoring probability (0.5 to 2.0)
        """
        base_multiplier = 1.0
        
        # Check if this is a new signing vs this opponent
        team_key = self._normalize_team_name(player_team)
        if team_key in self.new_signings_2024_25:
            for signing in self.new_signings_2024_25[team_key]:
                if signing['name'].lower() == player_name.lower():
                    # New signings get slight boost vs all opponents (excitement factor)
                    base_multiplier *= 1.1
                    break
        
        # Check if opponent is traditionally weak/strong defensively
        opponent_key = self._normalize_team_name(opponent_team)
        defensive_strength = {
            'arsenal': 0.9,      # Strong defense
            'liverpool': 0.9,    # Strong defense
            'manchester_city': 0.85,  # Very strong defense
            'chelsea': 0.95,     # Good defense
            'newcastle': 0.95,   # Good defense
            'brighton': 1.1,     # Weaker defense
            'fulham': 1.05,      # Average defense
            'brentford': 1.1,    # Weaker defense
            'crystal_palace': 1.05,
            'everton': 1.1,
            'ipswich': 1.2,      # Newly promoted, likely weaker
            'leicester': 1.15,   # Recently promoted
            'southampton': 1.15, # Recently promoted
        }
        
        defensive_multiplier = defensive_strength.get(opponent_key, 1.0)
        
        # Home vs Away consideration for new signings
        # (This would be passed as a parameter in a full implementation)
        
        return min(base_multiplier * defensive_multiplier, 2.0)
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team name for consistent lookup."""
        name_mappings = {
            'liverpool': 'liverpool',
            'chelsea': 'chelsea',
            'arsenal': 'arsenal',
            'manchester city': 'manchester_city',
            'man city': 'manchester_city',
            'manchester united': 'manchester_united',
            'man united': 'manchester_united',
            'man utd': 'manchester_united',
            'brighton': 'brighton',
            'brighton & hove albion': 'brighton',
            'fulham': 'fulham',
            'fulham fc': 'fulham',
            'tottenham': 'tottenham',
            'tottenham hotspur': 'tottenham',
            'spurs': 'tottenham',
            'newcastle': 'newcastle',
            'newcastle united': 'newcastle',
            'west ham': 'west_ham',
            'west ham united': 'west_ham',
            'aston villa': 'aston_villa',
            'villa': 'aston_villa',
            'crystal palace': 'crystal_palace',
            'palace': 'crystal_palace',
            'brentford': 'brentford',
            'brentford fc': 'brentford',
            'everton': 'everton',
            'everton fc': 'everton',
            'nottingham forest': 'nottingham_forest',
            'nott\'m forest': 'nottingham_forest',
            'forest': 'nottingham_forest',
            'leicester': 'leicester',
            'leicester city': 'leicester',
            'bournemouth': 'bournemouth',
            'afc bournemouth': 'bournemouth',
            'southampton': 'southampton',
            'southampton fc': 'southampton',
            'ipswich': 'ipswich',
            'ipswich town': 'ipswich',
            'wolves': 'wolves',
            'wolverhampton': 'wolves',
            'wolverhampton wanderers': 'wolves'
        }
        
        return name_mappings.get(team_name.lower(), team_name.lower().replace(' ', '_'))
    
    def _get_attacking_outlook(self, percentage_change: float) -> str:
        """Get attacking outlook based on percentage change."""
        if percentage_change > 20:
            return "Significantly Stronger"
        elif percentage_change > 10:
            return "Stronger"
        elif percentage_change > 5:
            return "Slightly Stronger"
        elif percentage_change > -5:
            return "Similar"
        elif percentage_change > -15:
            return "Slightly Weaker"
        elif percentage_change > -25:
            return "Weaker"
        else:
            return "Significantly Weaker"

# Example usage
if __name__ == "__main__":
    analyzer = EnhancedPlayerAnalyzer({})
    
    # Test Liverpool analysis (with Wirtz signing)
    liverpool_last_season = [
        {'name': 'Mohamed Salah', 'goals': 18, 'xg': 20.5},
        {'name': 'Luis Díaz', 'goals': 13, 'xg': 12.0},
        {'name': 'Dominik Szoboszlai', 'goals': 6, 'xg': 7.3}
    ]
    
    analysis = analyzer.analyze_team_attacking_changes('Liverpool', liverpool_last_season)
    print("Liverpool attacking changes:", analysis)
    
    # Test enhanced predictions
    enhanced = analyzer.get_enhanced_player_predictions('Liverpool', liverpool_last_season)
    print("\nLiverpool enhanced predictions:")
    for player in enhanced:
        print(f"  {player['name']} ({player.get('position', 'N/A')}) - {player['scoring_probability']:.1%}")
