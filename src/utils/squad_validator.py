#!/usr/bin/env python3
"""
Real-time squad validation and transfer detection using current team rosters.
This approach is more reliable than maintaining a transfer database.
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

class RealTimeSquadValidator:
    """Validates player squad status using real-time API data."""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.squad_cache = {}
        self.cache_duration = timedelta(hours=6)  # Cache squads for 6 hours
        
    def get_current_squad(self, team_id: str, team_name: str = None) -> List[Dict]:
        """Get current squad for a team using multiple API sources."""
        cache_key = f"squad_{team_id}"
        
        # Check cache first
        if cache_key in self.squad_cache:
            cached_time, cached_data = self.squad_cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_data
        
        squad_data = []
        
        try:
            # Try API-Football first
            if 'rapidapi' in self.api_keys:
                squad_data = self._get_api_football_squad(team_id)
            
            # If no data, try other sources
            if not squad_data and team_name:
                squad_data = self._get_alternative_squad_data(team_name)
            
            # Cache the result
            if squad_data:
                self.squad_cache[cache_key] = (datetime.now(), squad_data)
                
        except Exception as e:
            print(f"Error fetching squad for team {team_id}: {e}")
            
        return squad_data
    
    def is_player_in_current_squad(self, player_name: str, team_id: str, team_name: str = None) -> Dict[str, any]:
        """
        Check if player is currently in team squad.
        
        Returns:
            dict: {
                'in_squad': bool,
                'confidence': float,
                'match_type': str,  # 'exact', 'fuzzy', 'not_found'
                'squad_name': str or None
            }
        """
        squad = self.get_current_squad(team_id, team_name)
        
        if not squad:
            return {
                'in_squad': False,
                'confidence': 0.0,
                'match_type': 'no_squad_data',
                'squad_name': None
            }
        
        player_name_clean = self._normalize_name(player_name)
        
        # Try different matching strategies
        for player in squad:
            squad_name = player.get('name', '')
            squad_name_clean = self._normalize_name(squad_name)
            
            # 1. Exact match
            if player_name_clean == squad_name_clean:
                return {
                    'in_squad': True,
                    'confidence': 1.0,
                    'match_type': 'exact',
                    'squad_name': squad_name
                }
            
            # 2. Contains match (both directions)
            if (player_name_clean in squad_name_clean or 
                squad_name_clean in player_name_clean):
                return {
                    'in_squad': True,
                    'confidence': 0.8,
                    'match_type': 'fuzzy',
                    'squad_name': squad_name
                }
            
            # 3. Last name + first initial match
            if self._names_match_fuzzy(player_name_clean, squad_name_clean):
                return {
                    'in_squad': True,
                    'confidence': 0.7,
                    'match_type': 'fuzzy',
                    'squad_name': squad_name
                }
        
        return {
            'in_squad': False,
            'confidence': 0.9,  # High confidence they're not in squad
            'match_type': 'not_found',
            'squad_name': None
        }
    
    def validate_player_list(self, players: List[str], team_id: str, team_name: str = None) -> Dict[str, any]:
        """
        Validate entire list of players against current squad.
        
        Returns:
            dict: {
                'valid_players': List[str],
                'invalid_players': List[Dict],
                'new_signings': List[Dict],
                'validation_summary': Dict
            }
        """
        valid_players = []
        invalid_players = []
        
        squad = self.get_current_squad(team_id, team_name)
        squad_players = [p.get('name', '') for p in squad]
        
        # Check each provided player
        for player in players:
            validation = self.is_player_in_current_squad(player, team_id, team_name)
            
            if validation['in_squad']:
                valid_players.append(player)
            else:
                invalid_players.append({
                    'player': player,
                    'reason': 'not_in_current_squad',
                    'validation': validation
                })
        
        # Find potential new signings (in squad but not in provided list)
        provided_names = [self._normalize_name(p) for p in players]
        new_signings = []
        
        for squad_player in squad_players:
            squad_name_clean = self._normalize_name(squad_player)
            
            # If squad player is not in our provided list, they might be new
            if not any(self._names_match_fuzzy(squad_name_clean, provided) 
                      for provided in provided_names):
                
                # Check if they're a forward/midfielder (likely to score)
                player_info = next((p for p in squad if p.get('name') == squad_player), {})
                position = player_info.get('position', '').upper()
                
                if any(pos in position for pos in ['FW', 'MF', 'AM', 'CF', 'ST', 'LW', 'RW']):
                    new_signings.append({
                        'name': squad_player,
                        'position': position,
                        'age': player_info.get('age'),
                        'reason': 'new_in_squad_attacking_player'
                    })
        
        validation_summary = {
            'total_checked': len(players),
            'valid_count': len(valid_players),
            'invalid_count': len(invalid_players),
            'new_signings_count': len(new_signings),
            'squad_size': len(squad_players),
            'validation_confidence': sum(1 for p in players 
                                       if self.is_player_in_current_squad(p, team_id, team_name)['confidence'] > 0.7) / len(players) if players else 0
        }
        
        return {
            'valid_players': valid_players,
            'invalid_players': invalid_players,
            'new_signings': new_signings,
            'validation_summary': validation_summary
        }
    
    def get_team_transfer_analysis(self, team_id: str, team_name: str, 
                                 last_season_players: List[str]) -> Dict[str, any]:
        """
        Analyze transfers by comparing current squad with last season's players.
        
        Args:
            team_id: Team identifier
            team_name: Team name
            last_season_players: List of players from last season's data
            
        Returns:
            dict: Comprehensive transfer analysis
        """
        current_squad = self.get_current_squad(team_id, team_name)
        current_names = [p.get('name', '') for p in current_squad]
        
        # Find players who left (in last season but not in current)
        players_left = []
        for last_player in last_season_players:
            validation = self.is_player_in_current_squad(last_player, team_id, team_name)
            if not validation['in_squad'] and validation['confidence'] > 0.7:
                players_left.append({
                    'name': last_player,
                    'status': 'likely_transferred_out',
                    'confidence': validation['confidence']
                })
        
        # Find new players (in current but not similar to last season)
        last_names_clean = [self._normalize_name(p) for p in last_season_players]
        new_players = []
        
        for current_player in current_squad:
            current_name = current_player.get('name', '')
            current_name_clean = self._normalize_name(current_name)
            
            # Check if this player is similar to any last season player
            is_similar = any(self._names_match_fuzzy(current_name_clean, last_name) 
                           for last_name in last_names_clean)
            
            if not is_similar:
                position = current_player.get('position', '').upper()
                new_players.append({
                    'name': current_name,
                    'position': position,
                    'age': current_player.get('age'),
                    'status': 'likely_new_signing',
                    'attacking_potential': any(pos in position for pos in ['FW', 'MF', 'AM', 'CF', 'ST', 'LW', 'RW'])
                })
        
        return {
            'team_name': team_name,
            'players_left': players_left,
            'new_players': new_players,
            'net_change': len(new_players) - len(players_left),
            'current_squad_size': len(current_names),
            'attacking_signings': [p for p in new_players if p.get('attacking_potential', False)],
            'summary': self._generate_transfer_summary(players_left, new_players)
        }
    
    def _get_api_football_squad(self, team_id: str) -> List[Dict]:
        """Fetch squad from API-Football."""
        try:
            headers = {
                'X-RapidAPI-Key': self.api_keys.get('rapidapi', ''),
                'X-RapidAPI-Host': 'api-football-v1.p.rapidapi.com'
            }
            
            # Try team ID mapping if needed
            api_team_id = self._map_team_id_to_api_football(team_id)
            
            url = f"https://api-football-v1.p.rapidapi.com/v3/players/squads"
            params = {'team': api_team_id}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            squad_data = data.get('response', [])
            
            if squad_data:
                return squad_data[0].get('players', [])
                
        except Exception as e:
            print(f"API-Football squad fetch failed: {e}")
            
        return []
    
    def _get_alternative_squad_data(self, team_name: str) -> List[Dict]:
        """Get squad data from alternative sources."""
        # This could integrate with other APIs or scraping if needed
        # For now, return empty list
        return []
    
    def _map_team_id_to_api_football(self, team_id: str) -> str:
        """Map internal team ID to API-Football team ID."""
        # This would contain mappings between different API team IDs
        team_mappings = {
            'd07537b9': '51',   # Brighton
            'fd962109': '36',   # Fulham  
            'b2b47a98': '40',   # Liverpool
            'cff3d9bb': '49',   # Chelsea
            # Add more mappings as needed
        }
        
        return team_mappings.get(team_id, team_id)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize player name for comparison."""
        if not name:
            return ""
        
        # Convert to lowercase and remove extra spaces
        normalized = name.lower().strip()
        
        # Remove common prefixes/suffixes
        prefixes = ['de ', 'da ', 'del ', 'van ', 'von ', 'el ', 'al ']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        # Remove accents (basic)
        accent_map = {
            'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a', 'ä': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'õ': 'o', 'ô': 'o', 'ö': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c', 'ñ': 'n'
        }
        
        for accented, plain in accent_map.items():
            normalized = normalized.replace(accented, plain)
        
        return normalized
    
    def _names_match_fuzzy(self, name1: str, name2: str) -> bool:
        """Check if two names match using fuzzy logic."""
        if not name1 or not name2:
            return False
        
        # Split into parts
        parts1 = name1.split()
        parts2 = name2.split()
        
        if len(parts1) < 2 or len(parts2) < 2:
            return False
        
        # Check if last names match and first names have same initial
        last1, last2 = parts1[-1], parts2[-1]
        first1, first2 = parts1[0], parts2[0]
        
        return (last1 == last2 and 
                first1[0] == first2[0])
    
    def _generate_transfer_summary(self, players_left: List[Dict], new_players: List[Dict]) -> str:
        """Generate a human-readable transfer summary."""
        summary_parts = []
        
        if players_left:
            key_departures = [p['name'] for p in players_left[:3]]  # Top 3
            summary_parts.append(f"Lost: {', '.join(key_departures)}")
            if len(players_left) > 3:
                summary_parts[-1] += f" (+{len(players_left)-3} others)"
        
        if new_players:
            key_signings = [p['name'] for p in new_players[:3]]  # Top 3
            summary_parts.append(f"Signed: {', '.join(key_signings)}")
            if len(new_players) > 3:
                summary_parts[-1] += f" (+{len(new_players)-3} others)"
        
        if not summary_parts:
            return "No significant squad changes detected"
        
        return " | ".join(summary_parts)

# Example usage and testing
if __name__ == "__main__":
    # Test the squad validator
    api_keys = {
        'rapidapi': 'test_key'  # Would be real key in production
    }
    
    validator = RealTimeSquadValidator(api_keys)
    
    # Test with known transferred player
    result = validator.is_player_in_current_squad("João Pedro", "d07537b9", "Brighton")
    print("João Pedro at Brighton:", result)
    
    # Test squad validation
    brighton_players = ["João Pedro", "Danny Welbeck", "Kaoru Mitoma", "Lewis Dunk"]
    validation = validator.validate_player_list(brighton_players, "d07537b9", "Brighton")
    
    print("\nBrighton squad validation:")
    print(f"Valid players: {validation['valid_players']}")
    print(f"Invalid players: {validation['invalid_players']}")
    print(f"New signings: {validation['new_signings']}")
    print(f"Summary: {validation['validation_summary']}")
