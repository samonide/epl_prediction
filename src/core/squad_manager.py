#!/usr/bin/env python3
"""
EPL Squad Manager - Comprehensive player database with injury tracking
Caches all 20 EPL team squads to minimize API calls (100 limit consideration)
Supports transfers, injuries, and real-time squad updates
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
from dataclasses import dataclass, asdict

@dataclass
class Player:
    """Player data structure with comprehensive information"""
    id: int
    name: str
    position: str
    age: int
    nationality: str
    shirt_number: Optional[int] = None
    market_value: Optional[str] = None
    contract_until: Optional[str] = None
    injured: bool = False
    injury_type: Optional[str] = None
    injury_return_date: Optional[str] = None
    goals_season: int = 0
    assists_season: int = 0
    minutes_played: int = 0
    last_updated: str = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

@dataclass
class TeamSquad:
    """Team squad with metadata"""
    team_id: int
    team_name: str
    players: List[Player]
    last_updated: str
    transfer_window_updates: List[Dict] = None
    
    def __post_init__(self):
        if self.transfer_window_updates is None:
            self.transfer_window_updates = []

class SquadManager:
    """Manages EPL squad data with caching and real-time updates"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.cache_dir = "cache/squads"
        self.injury_cache_file = "cache/injuries/injuries.json"
        self.transfer_cache_file = "cache/transfers/transfers.json"
        self.squad_cache_file = "cache/squads/squad_database.json"
        
        # EPL Team IDs (2024-25 season)
        self.epl_teams = {
            'Arsenal': 42, 'Aston Villa': 66, 'Bournemouth': 35,
            'Brentford': 94, 'Brighton': 51, 'Chelsea': 61,
            'Crystal Palace': 52, 'Everton': 45, 'Fulham': 36,
            'Ipswich Town': 72, 'Leicester City': 46, 'Liverpool': 40,
            'Manchester City': 50, 'Manchester United': 33,
            'Newcastle': 34, 'Nottingham Forest': 65, 'Southampton': 41,
            'Tottenham': 47, 'West Ham': 48, 'Wolverhampton': 39
        }
        
        self._ensure_directories()
        self.squad_database = self._load_squad_database()
        
    def _ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs("cache/injuries", exist_ok=True)
        os.makedirs("cache/transfers", exist_ok=True)
    
    def _load_squad_database(self) -> Dict[str, TeamSquad]:
        """Load cached squad database"""
        if os.path.exists(self.squad_cache_file):
            try:
                with open(self.squad_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Convert back to TeamSquad objects
                database = {}
                for team_name, squad_data in data.items():
                    players = [Player(**player_data) for player_data in squad_data['players']]
                    database[team_name] = TeamSquad(
                        team_id=squad_data['team_id'],
                        team_name=squad_data['team_name'],
                        players=players,
                        last_updated=squad_data['last_updated'],
                        transfer_window_updates=squad_data.get('transfer_window_updates', [])
                    )
                return database
            except Exception as e:
                print(f"⚠️ Error loading squad database: {e}")
                return {}
        return {}
    
    def _save_squad_database(self):
        """Save squad database to cache"""
        try:
            # Convert to serializable format
            data = {}
            for team_name, squad in self.squad_database.items():
                data[team_name] = {
                    'team_id': squad.team_id,
                    'team_name': squad.team_name,
                    'players': [asdict(player) for player in squad.players],
                    'last_updated': squad.last_updated,
                    'transfer_window_updates': squad.transfer_window_updates
                }
            
            with open(self.squad_cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Squad database saved to {self.squad_cache_file}")
            
        except Exception as e:
            print(f"❌ Error saving squad database: {e}")
    
    def cache_all_squads(self, force_update: bool = False) -> bool:
        """Cache all 20 EPL team squads efficiently"""
        print("🔄 Caching all EPL squad data...")
        print(f"📊 Teams to process: {len(self.epl_teams)}")
        
        updated_teams = 0
        api_calls_made = 0
        
        for team_name, team_id in self.epl_teams.items():
            try:
                # Check if update needed
                needs_update = force_update
                if not needs_update and team_name in self.squad_database:
                    last_update = datetime.fromisoformat(self.squad_database[team_name].last_updated)
                    needs_update = (datetime.now() - last_update) > timedelta(days=7)
                
                if needs_update:
                    print(f"📥 Fetching {team_name} squad...")
                    squad = self._fetch_team_squad(team_id, team_name)
                    if squad:
                        self.squad_database[team_name] = squad
                        updated_teams += 1
                        api_calls_made += 1
                        
                        # Respect API rate limits
                        if api_calls_made >= 5:
                            print("⏳ Pausing to respect API limits...")
                            import time
                            time.sleep(2)
                            api_calls_made = 0
                else:
                    print(f"✅ {team_name} squad cache still fresh")
                    
            except Exception as e:
                print(f"❌ Error caching {team_name}: {e}")
                continue
        
        if updated_teams > 0:
            self._save_squad_database()
            
        print(f"🎯 Squad caching complete: {updated_teams} teams updated")
        return True
    
    def _fetch_team_squad(self, team_id: int, team_name: str) -> Optional[TeamSquad]:
        """Fetch team squad from API with comprehensive player data"""
        try:
            # Try to fetch real data from API first
            players = self._fetch_real_squad_from_api(team_id, team_name)
            
            # Fallback to mock data if API fails or returns insufficient data
            if not players or len(players) < 5:
                print(f"⚠️  API data insufficient for {team_name}, using mock data")
                players = self._get_mock_squad_data(team_name)
            else:
                print(f"✅ Real squad data fetched for {team_name}: {len(players)} players")
            
            squad = TeamSquad(
                team_id=team_id,
                team_name=team_name,
                players=players,
                last_updated=datetime.now().isoformat()
            )
            
            return squad
            
        except Exception as e:
            print(f"❌ Error fetching {team_name} squad: {e}")
            return None
            
    def _fetch_real_squad_from_api(self, team_id: int, team_name: str) -> List[Player]:
        """Fetch real squad data from Football API with backup key"""
        try:
            # Try primary API first, then backup
            api_keys = [
                self.api_keys.get('api_football'),
                'd9cfd371eace3833ef9c3a6011ffaaa8'  # Backup API key
            ]
            
            for api_key_idx, api_key in enumerate(api_keys):
                if not api_key:
                    continue
                    
                print(f"🔍 Trying API key {api_key_idx + 1} for {team_name}...")
                
                # Decode API key if it's base64 encoded
                try:
                    import base64
                    decoded_key = base64.b64decode(api_key).decode('utf-8')
                except:
                    decoded_key = api_key  # Use key as-is if not base64
                
                headers = {
                    'X-RapidAPI-Key': decoded_key,
                    'X-RapidAPI-Host': 'v3.football.api-sports.io'
                }
                
                # Get current season (2024-25)
                current_season = 2024
                
                # Fetch squad data
                url = "https://v3.football.api-sports.io/players/squads"
                params = {
                    'team': team_id,
                    'season': current_season
                }
                
                try:
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Check for API errors
                    if 'errors' in data and data['errors']:
                        error_msg = data['errors']
                        print(f"⚠️  API Error: {error_msg}")
                        continue  # Try next API key
                    
                    if not data.get('response'):
                        print(f"❌ No squad data returned for {team_name}")
                        continue  # Try next API key
                    
                    squad_data = data['response'][0] if data['response'] else {}
                    players_data = squad_data.get('players', [])
                    
                    if len(players_data) < 5:
                        print(f"⚠️  Insufficient player data for {team_name}: {len(players_data)} players")
                        continue  # Try next API key
                    
                    players = []
                    for i, player_data in enumerate(players_data):
                        try:
                            player = Player(
                                id=player_data.get('id', 1000 + i),
                                name=player_data.get('name', f'{team_name} Player {i+1}'),
                                position=self._map_position(player_data.get('position', 'M')),
                                age=player_data.get('age', 25),
                                nationality=player_data.get('nationality', 'Unknown'),
                                shirt_number=player_data.get('number'),
                                market_value=self._estimate_market_value(player_data.get('position', 'M')),
                                contract_until='2026',  # Default
                                goals_season=0,  # Would need separate stats API call
                                assists_season=0,
                                minutes_played=0
                            )
                            players.append(player)
                            
                        except Exception as e:
                            print(f"⚠️  Error processing player data for {team_name}: {e}")
                            continue
                    
                    if len(players) >= 5:
                        print(f"✅ Successfully fetched {len(players)} players for {team_name}")
                        return players
                    else:
                        print(f"⚠️  Processed data insufficient for {team_name}: {len(players)} players")
                        continue  # Try next API key
                        
                except requests.exceptions.RequestException as e:
                    print(f"❌ API request failed for {team_name}: {e}")
                    continue  # Try next API key
                except Exception as e:
                    print(f"❌ Error processing API response for {team_name}: {e}")
                    continue  # Try next API key
            
            print(f"❌ All API attempts failed for {team_name}")
            return []
            
        except Exception as e:
            print(f"❌ Error fetching squad for {team_name}: {e}")
            return []
    
    def _map_position(self, api_position: str) -> str:
        """Map API position to our standard positions"""
        mapping = {
            'Goalkeeper': 'GK',
            'Defender': 'CB', 
            'Midfielder': 'CM',
            'Attacker': 'CF'
        }
        return mapping.get(api_position, 'MF')
    
    def _estimate_market_value(self, position: str) -> str:
        """Estimate market value based on position"""
        values = {
            'GK': '£15M',
            'CB': '£25M', 
            'CM': '£20M',
            'CF': '£30M',
            'MF': '£15M'
        }
        return values.get(position, '£15M')

    def _get_mock_squad_data(self, team_name: str) -> List[Player]:
        """Generate comprehensive mock squad data for testing"""
        
        # Normalize team name for lookup
        team_key = team_name.lower().replace(' ', '_')
        
        mock_squads = {
            'newcastle': [
                Player(1, "Nick Pope", "GK", 32, "England", 22, "£15M", "2026", goals_season=0),
                Player(2, "Kieran Trippier", "RB", 34, "England", 2, "£8M", "2025", goals_season=1),
                Player(3, "Fabian Schär", "CB", 32, "Switzerland", 5, "£12M", "2025", goals_season=2),
                Player(4, "Sven Botman", "CB", 24, "Netherlands", 4, "£30M", "2027", injured=True, injury_type="ACL", injury_return_date="2025-01-15"),
                Player(5, "Dan Burn", "CB", 32, "England", 33, "£10M", "2026", goals_season=1),
                Player(6, "Lewis Hall", "LB", 20, "England", 20, "£25M", "2029", goals_season=0),
                Player(7, "Bruno Guimarães", "CM", 26, "Brazil", 39, "£80M", "2028", goals_season=4, assists_season=2),
                Player(8, "Joelinton", "CM", 28, "Brazil", 7, "£35M", "2026", goals_season=3, assists_season=1),
                Player(9, "Sean Longstaff", "CM", 26, "England", 36, "£15M", "2025", goals_season=2),
                Player(10, "Anthony Gordon", "LW", 23, "England", 10, "£40M", "2028", goals_season=8, assists_season=5),
                Player(11, "Miguel Almirón", "RW", 30, "Paraguay", 24, "£20M", "2026", goals_season=4, assists_season=3),
                Player(12, "Harvey Barnes", "LW", 26, "England", 11, "£35M", "2028", goals_season=6, assists_season=4),
                Player(13, "Alexander Isak", "CF", 25, "Sweden", 14, "£70M", "2028", goals_season=15, assists_season=3),
                Player(14, "Callum Wilson", "CF", 32, "England", 9, "£18M", "2025", injured=True, injury_type="Back", injury_return_date="2024-12-20"),
                Player(15, "Jacob Murphy", "RW", 29, "England", 23, "£12M", "2026", goals_season=3, assists_season=6)
            ],
            'west_ham': [
                Player(101, "Alphonse Areola", "GK", 31, "France", 23, "£12M", "2026", goals_season=0),
                Player(102, "Vladimir Coufal", "RB", 32, "Czech Republic", 5, "£8M", "2025", goals_season=0),
                Player(103, "Kurt Zouma", "CB", 29, "France", 4, "£25M", "2025", goals_season=1),
                Player(104, "Max Kilman", "CB", 27, "England", 26, "£40M", "2029", goals_season=2),
                Player(105, "Emerson", "LB", 25, "Italy", 33, "£15M", "2027", goals_season=1),
                Player(106, "Tomáš Souček", "CM", 29, "Czech Republic", 28, "£25M", "2026", goals_season=4, assists_season=2),
                Player(107, "Edson Álvarez", "DM", 26, "Mexico", 19, "£35M", "2028", goals_season=1, assists_season=1),
                Player(108, "Lucas Paquetá", "AM", 27, "Brazil", 10, "£45M", "2027", goals_season=6, assists_season=4),
                Player(109, "James Ward-Prowse", "CM", 29, "England", 7, "£30M", "2027", goals_season=3, assists_season=8),
                Player(110, "Jarrod Bowen", "RW", 27, "England", 20, "£40M", "2027", goals_season=14, assists_season=6),
                Player(111, "Mohammed Kudus", "LW", 24, "Ghana", 14, "£35M", "2028", goals_season=8, assists_season=3),
                Player(112, "Michail Antonio", "CF", 34, "Jamaica", 9, "£8M", "2025", goals_season=8, assists_season=2),
                Player(113, "Niclas Füllkrug", "CF", 31, "Germany", 11, "£25M", "2027", goals_season=12, assists_season=1),
                Player(114, "Crysencio Summerville", "LW", 22, "Netherlands", 27, "£25M", "2029", goals_season=5, assists_season=7)
            ],
            'liverpool': [
                Player(201, "Alisson", "GK", 31, "Brazil", 1, "£50M", "2027", goals_season=0),
                Player(202, "Trent Alexander-Arnold", "RB", 25, "England", 66, "£70M", "2025", goals_season=3, assists_season=12),
                Player(203, "Virgil van Dijk", "CB", 33, "Netherlands", 4, "£45M", "2025", goals_season=5, assists_season=2),
                Player(204, "Ibrahima Konaté", "CB", 25, "France", 5, "£40M", "2026", goals_season=2, assists_season=1),
                Player(205, "Andy Robertson", "LB", 30, "Scotland", 26, "£35M", "2026", goals_season=2, assists_season=8),
                Player(206, "Ryan Gravenberch", "CM", 22, "Netherlands", 38, "£35M", "2028", goals_season=3, assists_season=3),
                Player(207, "Alexis Mac Allister", "CM", 25, "Argentina", 10, "£55M", "2028", goals_season=6, assists_season=5),
                Player(208, "Dominik Szoboszlai", "AM", 23, "Hungary", 8, "£60M", "2028", goals_season=7, assists_season=4),
                Player(209, "Mohamed Salah", "RW", 32, "Egypt", 11, "£40M", "2025", goals_season=18, assists_season=10),
                Player(210, "Luis Díaz", "LW", 27, "Colombia", 7, "£50M", "2027", goals_season=13, assists_season=5),
                Player(211, "Darwin Núñez", "CF", 25, "Uruguay", 9, "£75M", "2028", goals_season=11, assists_season=8),
                Player(212, "Diogo Jota", "CF", 27, "Portugal", 20, "£45M", "2027", goals_season=10, assists_season=2),
                Player(213, "Cody Gakpo", "LW", 25, "Netherlands", 18, "£40M", "2027", goals_season=9, assists_season=6)
            ],
            'arsenal': [
                Player(301, "David Raya", "GK", 28, "Spain", 22, "£30M", "2028", goals_season=0),
                Player(302, "Ben White", "RB", 26, "England", 4, "£50M", "2026", goals_season=2, assists_season=4),
                Player(303, "William Saliba", "CB", 23, "France", 2, "£60M", "2027", goals_season=3, assists_season=1),
                Player(304, "Gabriel", "CB", 26, "Brazil", 6, "£45M", "2026", goals_season=4, assists_season=2),
                Player(305, "Oleksandr Zinchenko", "LB", 27, "Ukraine", 35, "£30M", "2026", goals_season=1, assists_season=6),
                Player(306, "Declan Rice", "DM", 25, "England", 41, "£105M", "2029", goals_season=7, assists_season=8),
                Player(307, "Martin Ødegaard", "AM", 25, "Norway", 8, "£80M", "2028", goals_season=8, assists_season=10),
                Player(308, "Kai Havertz", "CF", 25, "Germany", 29, "£65M", "2027", goals_season=13, assists_season=7),
                Player(309, "Bukayo Saka", "RW", 22, "England", 7, "£120M", "2027", goals_season=16, assists_season=9),
                Player(310, "Gabriel Martinelli", "LW", 23, "Brazil", 11, "£60M", "2027", goals_season=12, assists_season=5),
                Player(311, "Leandro Trossard", "LW", 29, "Belgium", 19, "£25M", "2026", goals_season=8, assists_season=2),
                Player(312, "Gabriel Jesus", "CF", 27, "Brazil", 9, "£45M", "2026", goals_season=4, assists_season=8),
                Player(313, "Thomas Partey", "CM", 31, "Ghana", 5, "£20M", "2025", goals_season=3, assists_season=3)
            ],
            'manchester_city': [
                Player(401, "Ederson", "GK", 30, "Brazil", 31, "£40M", "2026", goals_season=0),
                Player(402, "Kyle Walker", "RB", 34, "England", 2, "£15M", "2025", goals_season=1, assists_season=3),
                Player(403, "Rúben Dias", "CB", 27, "Portugal", 3, "£65M", "2027", goals_season=2, assists_season=1),
                Player(404, "John Stones", "CB", 30, "England", 5, "£40M", "2026", goals_season=3, assists_season=2),
                Player(405, "Joško Gvardiol", "LB", 22, "Croatia", 24, "£77M", "2029", goals_season=4, assists_season=3),
                Player(406, "Rodri", "DM", 28, "Spain", 16, "£90M", "2027", goals_season=8, assists_season=9),
                Player(407, "Kevin De Bruyne", "AM", 33, "Belgium", 17, "£50M", "2025", goals_season=6, assists_season=18),
                Player(408, "Bernardo Silva", "CM", 29, "Portugal", 20, "£70M", "2026", goals_season=7, assists_season=9),
                Player(409, "Phil Foden", "RW", 24, "England", 47, "£100M", "2027", goals_season=19, assists_season=8),
                Player(410, "Jack Grealish", "LW", 28, "England", 10, "£100M", "2027", goals_season=3, assists_season=11),
                Player(411, "Erling Haaland", "CF", 24, "Norway", 9, "£150M", "2029", goals_season=27, assists_season=5),
                Player(412, "Julián Álvarez", "CF", 24, "Argentina", 19, "£80M", "2028", goals_season=11, assists_season=9)
            ]
        }
        
        return mock_squads.get(team_key, [
            Player(999, f"{team_name} Player", "MF", 25, "England", 99, "£10M", "2026")
        ])
    
    def update_player_injury(self, team_name: str, player_name: str, 
                           injured: bool, injury_type: str = None, 
                           return_date: str = None) -> bool:
        """Update player injury status"""
        if team_name not in self.squad_database:
            print(f"❌ Team {team_name} not found in database")
            return False
        
        for player in self.squad_database[team_name].players:
            if player.name.lower() == player_name.lower():
                player.injured = injured
                player.injury_type = injury_type if injured else None
                player.injury_return_date = return_date if injured else None
                player.last_updated = datetime.now().isoformat()
                
                self._save_squad_database()
                
                status = "injured" if injured else "recovered"
                print(f"✅ {player_name} marked as {status}")
                if injured and injury_type:
                    print(f"   Injury: {injury_type}")
                    if return_date:
                        print(f"   Expected return: {return_date}")
                return True
        
        print(f"❌ Player {player_name} not found in {team_name}")
        return False
    
    def add_transfer(self, team_name: str, player_name: str, position: str,
                    transfer_type: str = "in", fee: str = None) -> bool:
        """Add new transfer to squad"""
        if team_name not in self.squad_database:
            print(f"❌ Team {team_name} not found")
            return False
        
        if transfer_type == "in":
            # Add new player
            new_id = max([p.id for p in self.squad_database[team_name].players], default=0) + 1
            new_player = Player(
                id=new_id,
                name=player_name,
                position=position,
                age=25,  # Default age
                nationality="Unknown",
                market_value=fee,
                contract_until="2029"
            )
            
            self.squad_database[team_name].players.append(new_player)
            
            # Log transfer
            transfer_record = {
                "date": datetime.now().isoformat(),
                "player": player_name,
                "type": "signing",
                "fee": fee,
                "position": position
            }
            self.squad_database[team_name].transfer_window_updates.append(transfer_record)
            
        elif transfer_type == "out":
            # Remove player
            for i, player in enumerate(self.squad_database[team_name].players):
                if player.name.lower() == player_name.lower():
                    self.squad_database[team_name].players.pop(i)
                    
                    # Log transfer
                    transfer_record = {
                        "date": datetime.now().isoformat(),
                        "player": player_name,
                        "type": "departure",
                        "fee": fee
                    }
                    self.squad_database[team_name].transfer_window_updates.append(transfer_record)
                    break
        
        self._save_squad_database()
        print(f"✅ Transfer completed: {player_name} {'joined' if transfer_type == 'in' else 'left'} {team_name}")
        return True
    
    def get_team_squad(self, team_name: str, include_injured: bool = True) -> List[Player]:
        """Get team squad with injury filtering"""
        if team_name not in self.squad_database:
            print(f"❌ Team {team_name} not found. Run cache_all_squads() first.")
            return []
        
        players = self.squad_database[team_name].players
        
        if not include_injured:
            players = [p for p in players if not p.injured]
        
        return players
    
    def get_available_players(self, team_name: str, position: str = None) -> List[Player]:
        """Get available (non-injured) players for a specific position"""
        players = self.get_team_squad(team_name, include_injured=False)
        
        if position:
            players = [p for p in players if position.upper() in p.position.upper()]
        
        return sorted(players, key=lambda x: x.goals_season, reverse=True)
    
    def display_squad_summary(self, team_name: str):
        """Display comprehensive squad summary"""
        if team_name not in self.squad_database:
            print(f"❌ Team {team_name} not found")
            return
        
        squad = self.squad_database[team_name]
        print(f"\n{'='*60}")
        print(f"🏆 {team_name.upper()} SQUAD OVERVIEW")
        print(f"{'='*60}")
        print(f"📅 Last Updated: {squad.last_updated[:10]}")
        print(f"👥 Total Players: {len(squad.players)}")
        
        # Injury report
        injured_players = [p for p in squad.players if p.injured]
        print(f"🏥 Injured Players: {len(injured_players)}")
        
        if injured_players:
            print("\n🚑 INJURY LIST:")
            for player in injured_players:
                return_info = f" (Return: {player.injury_return_date})" if player.injury_return_date else ""
                print(f"   • {player.name} ({player.position}) - {player.injury_type}{return_info}")
        
        # Top scorers
        top_scorers = sorted([p for p in squad.players if p.goals_season > 0], 
                           key=lambda x: x.goals_season, reverse=True)[:5]
        if top_scorers:
            print("\n⚽ TOP SCORERS:")
            for player in top_scorers:
                print(f"   • {player.name} ({player.position}) - {player.goals_season} goals")
        
        # Recent transfers
        if squad.transfer_window_updates:
            print(f"\n📈 RECENT TRANSFERS ({len(squad.transfer_window_updates)}):")
            for transfer in squad.transfer_window_updates[-3:]:  # Last 3
                date = transfer['date'][:10]
                print(f"   • {transfer['player']} ({transfer['type']}) - {date}")
    
    def _clear_screen(self):
        """Clear terminal screen for better UX"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def _print_header(self, title: str, subtitle: str = None):
        """Print styled header with consistent design"""
        self._clear_screen()
        print("=" * 70)
        print(f"🏆 {title.upper()}")
        if subtitle:
            print(f"📋 {subtitle}")
        print("=" * 70)
    
    def _print_menu_option(self, number: int, icon: str, text: str, description: str = None):
        """Print styled menu option"""
        if description:
            print(f"{number}. {icon} {text}")
            print(f"   └── {description}")
        else:
            print(f"{number}. {icon} {text}")
    
    def _wait_for_continue(self):
        """Wait for user to continue"""
        try:
            input("\n📱 Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            pass

    def squad_interactive_manager(self):
        """Enhanced interactive CLI for squad management"""
    def squad_interactive_manager(self):
        """Enhanced interactive CLI for squad management"""
        while True:
            self._print_header("EPL SQUAD MANAGER", "Professional Squad & Injury Management System")
            print()
            self._print_menu_option(1, "📥", "Cache All EPL Squads", "Download/update all 20 team rosters")
            self._print_menu_option(2, "👥", "View Team Squad", "Detailed squad overview with stats & injuries")
            self._print_menu_option(3, "🏥", "Update Injury Status", "Mark players as injured/recovered")
            self._print_menu_option(4, "📈", "Manage Transfers", "Add new signings or departures")
            self._print_menu_option(5, "🔍", "Search Available Players", "Find fit players by position")
            self._print_menu_option(6, "📊", "Squad Statistics", "Database overview & insights")
            self._print_menu_option(7, "�", "Advanced Options", "Data export, bulk operations")
            self._print_menu_option(8, "🚪", "Return to Main Menu", "Back to prediction system")
            print("=" * 70)
            
            try:
                choice = input("🎮 Select option (1-8): ").strip()
            except (EOFError, KeyboardInterrupt):
                self._clear_screen()
                print("📊 Returning to main menu...")
                break
            
            if choice == "1":
                self._handle_cache_squads()
                
            elif choice == "2":
                self._handle_view_squad()
                
            elif choice == "3":
                self._handle_injury_update()
                
            elif choice == "4":
                self._handle_transfer_management()
                
            elif choice == "5":
                self._handle_search_players()
                
            elif choice == "6":
                self._handle_squad_statistics()
                
            elif choice == "7":
                self._handle_advanced_options()
                
            elif choice == "8":
                self._clear_screen()
                break
            else:
                self._clear_screen()
                print("❌ Invalid choice. Please select 1-8.")
                self._wait_for_continue()
    
    def _handle_cache_squads(self):
        """Handle squad caching with better UX"""
        self._print_header("SQUAD CACHING", "Update EPL Team Rosters")
        print("📊 Current cache status:")
        print(f"   • Teams cached: {len(self.squad_database)}/20")
        
        if self.squad_database:
            oldest_update = min(
                datetime.fromisoformat(squad.last_updated) 
                for squad in self.squad_database.values()
            )
            days_old = (datetime.now() - oldest_update).days
            print(f"   • Oldest data: {days_old} days old")
        
        print("\n🔄 Cache Options:")
        print("1. 🔄 Update only stale data (>7 days)")
        print("2. 🆕 Force update all teams")
        print("3. 🚪 Cancel")
        
        try:
            cache_choice = input("\n🎮 Select option (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        
        if cache_choice == "1":
            self.cache_all_squads(force_update=False)
        elif cache_choice == "2":
            confirm = input("⚠️  This will use significant API calls. Continue? (y/N): ").lower()
            if confirm == 'y':
                self.cache_all_squads(force_update=True)
            else:
                print("❌ Operation cancelled")
        elif cache_choice == "3":
            return
        else:
            print("❌ Invalid choice")
        
        self._wait_for_continue()
    
    def _handle_view_squad(self):
        """Handle squad viewing with enhanced interface"""
        self._print_header("SQUAD VIEWER", "Detailed Team Information")
        
        if not self.squad_database:
            print("❌ No squad data available. Please cache squads first.")
            self._wait_for_continue()
            return
        
        print("🏆 Available Teams:")
        teams = list(self.squad_database.keys())
        for i, team in enumerate(teams, 1):
            squad = self.squad_database[team]
            injured_count = len([p for p in squad.players if p.injured])
            print(f"{i:2d}. {team} ({len(squad.players)} players, {injured_count} injured)")
        
        try:
            choice = input(f"\n🎮 Select team (1-{len(teams)}) or type name: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        
        # Handle numeric choice
        try:
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(teams):
                    team_name = teams[idx]
                else:
                    print("❌ Invalid team number")
                    self._wait_for_continue()
                    return
            else:
                # Handle name input
                team_name = choice
        except (ValueError, IndexError) as e:
            print(f"❌ Input error: {e}")
            self._wait_for_continue()
            return
        
        self._clear_screen()
        self.display_squad_summary(team_name)
        self._wait_for_continue()
    
    def _handle_injury_update(self):
        """Handle injury updates with validation"""
        self._print_header("INJURY MANAGEMENT", "Update Player Health Status")
        
        if not self.squad_database:
            print("❌ No squad data available. Please cache squads first.")
            self._wait_for_continue()
            return
        
        print("🏆 Select Team:")
        teams = list(self.squad_database.keys())
        for i, team in enumerate(teams, 1):
            print(f"{i:2d}. {team}")
        
        try:
            team_choice = input(f"\n🎮 Select team (1-{len(teams)}): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        
        if not team_choice.isdigit() or not (1 <= int(team_choice) <= len(teams)):
            print("❌ Invalid team selection")
            self._wait_for_continue()
            return
        
        team_name = teams[int(team_choice) - 1]
        squad = self.squad_database[team_name]
        
        self._clear_screen()
        print(f"👥 {team_name} Squad:")
        for i, player in enumerate(squad.players, 1):
            status = "🚑 INJURED" if player.injured else "✅ FIT"
            injury_info = f" ({player.injury_type})" if player.injured and player.injury_type else ""
            print(f"{i:2d}. {player.name} ({player.position}) - {status}{injury_info}")
        
        try:
            player_choice = input(f"\n🎮 Select player (1-{len(squad.players)}): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        
        try:
            if not player_choice.isdigit() or not (1 <= int(player_choice) <= len(squad.players)):
                print("❌ Invalid player selection")
                self._wait_for_continue()
                return
            
            player = squad.players[int(player_choice) - 1]
        except (ValueError, IndexError) as e:
            print(f"❌ Invalid selection: {e}")
            self._wait_for_continue()
            return
        current_status = "injured" if player.injured else "fit"
        
        print(f"\n👤 Player: {player.name}")
        print(f"📊 Current Status: {current_status.upper()}")
        
        try:
            new_status = input("🏥 New status (injured/fit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        
        if new_status not in ['injured', 'fit']:
            print("❌ Invalid status. Use 'injured' or 'fit'")
            self._wait_for_continue()
            return
        
        injured = new_status == 'injured'
        injury_type = None
        return_date = None
        
        if injured:
            try:
                injury_type = input("🩹 Injury type (optional): ").strip() or None
                return_date = input("📅 Expected return (YYYY-MM-DD, optional): ").strip() or None
            except (EOFError, KeyboardInterrupt):
                return
        
        success = self.update_player_injury(team_name, player.name, injured, injury_type, return_date)
        
        if success:
            print(f"✅ {player.name} updated successfully")
        else:
            print(f"❌ Failed to update {player.name}")
        
        self._wait_for_continue()
    
    def _handle_transfer_management(self):
        """Handle transfer operations"""
        self._print_header("TRANSFER MANAGEMENT", "Squad Changes & New Signings")
        print("📈 Transfer Options:")
        print("1. ➕ Add New Signing")
        print("2. ➖ Remove Player")
        print("3. 📋 View Recent Transfers")
        print("4. 🚪 Cancel")
        
        try:
            choice = input("\n🎮 Select option (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        
        if choice == "1":
            self._handle_add_signing()
        elif choice == "2":
            self._handle_remove_player()
        elif choice == "3":
            self._handle_view_transfers()
        elif choice == "4":
            return
        else:
            print("❌ Invalid choice")
            self._wait_for_continue()
    
    def _handle_search_players(self):
        """Handle player search with filters"""
        self._print_header("PLAYER SEARCH", "Find Available Players")
        
        if not self.squad_database:
            print("❌ No squad data available.")
            self._wait_for_continue()
            return
        
        teams = list(self.squad_database.keys())
        print("🏆 Available Teams:")
        for i, team in enumerate(teams, 1):
            print(f"{i:2d}. {team}")
        
        try:
            team_choice = input(f"\n🎮 Select team (1-{len(teams)}): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        
        if not team_choice.isdigit() or not (1 <= int(team_choice) <= len(teams)):
            print("❌ Invalid team selection")
            self._wait_for_continue()
            return
        
        team_name = teams[int(team_choice) - 1]
        
        try:
            position = input("📍 Position filter (GK/DEF/MID/FW or press Enter for all): ").strip().upper() or None
        except (EOFError, KeyboardInterrupt):
            return
        
        players = self.get_available_players(team_name, position)
        
        self._clear_screen()
        print(f"� Available Players - {team_name}")
        if position:
            print(f"📍 Position: {position}")
        print("=" * 50)
        
        if not players:
            print("❌ No players found matching criteria")
        else:
            for i, player in enumerate(players, 1):
                print(f"{i:2d}. {player.name} ({player.position}) - {player.goals_season}G/{player.assists_season}A")
        
        self._wait_for_continue()
    
    def _handle_squad_statistics(self):
        """Handle squad statistics display"""
        self._print_header("SQUAD STATISTICS", "Database Overview & Insights")
        
        if not self.squad_database:
            print("❌ No squad data available.")
            self._wait_for_continue()
            return
        
        total_players = sum(len(squad.players) for squad in self.squad_database.values())
        total_injured = sum(len([p for p in squad.players if p.injured]) for squad in self.squad_database.values())
        total_goals = sum(sum(p.goals_season for p in squad.players) for squad in self.squad_database.values())
        
        print("📊 Database Statistics:")
        print(f"   • Teams Cached: {len(self.squad_database)}/20")
        print(f"   • Total Players: {total_players}")
        injury_percentage = (total_injured/total_players*100) if total_players > 0 else 0
        print(f"   • Currently Injured: {total_injured} ({injury_percentage:.1f}%)")
        print(f"   • Total Season Goals: {total_goals}")
        
        print("\n🏥 Injury Report by Team:")
        for team_name, squad in self.squad_database.items():
            injured_players = [p for p in squad.players if p.injured]
            if injured_players:
                print(f"   • {team_name}: {len(injured_players)} injured")
                for player in injured_players:
                    injury_info = f" ({player.injury_type})" if player.injury_type else ""
                    print(f"     - {player.name}{injury_info}")
        
        print("\n⚽ Top Scorers Across League:")
        all_players = []
        for squad in self.squad_database.values():
            all_players.extend(squad.players)
        
        top_scorers = sorted([p for p in all_players if p.goals_season > 0], 
                           key=lambda x: x.goals_season, reverse=True)[:10]
        
        for i, player in enumerate(top_scorers, 1):
            team = next(squad.team_name for squad in self.squad_database.values() 
                       if player in squad.players)
            print(f"   {i:2d}. {player.name} ({team}) - {player.goals_season} goals")
        
        self._wait_for_continue()
    
    def _handle_advanced_options(self):
        """Handle advanced operations"""
        self._print_header("ADVANCED OPTIONS", "Data Management & Export")
        print("🔧 Advanced Operations:")
        print("1. 💾 Export Squad Data")
        print("2. 🔄 Rebuild Cache")
        print("3. 🧹 Clear All Data")
        print("4. 🚪 Cancel")
        
        try:
            choice = input("\n🎮 Select option (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        
        if choice == "1":
            # Export functionality would go here
            print("💾 Export feature coming soon...")
        elif choice == "2":
            confirm = input("🔄 This will rebuild the entire cache. Continue? (y/N): ").lower()
            if confirm == 'y':
                self.squad_database = {}
                self.cache_all_squads(force_update=True)
        elif choice == "3":
            confirm = input("⚠️  This will delete ALL squad data. Are you sure? (type YES): ")
            if confirm == "YES":
                self.squad_database = {}
                if os.path.exists(self.squad_cache_file):
                    os.remove(self.squad_cache_file)
                print("✅ All data cleared")
        elif choice == "4":
            return
        else:
            print("❌ Invalid choice")
        
        self._wait_for_continue()
    
    def _handle_add_signing(self):
        """Handle adding new signings"""
        # Implementation for adding signings
        print("➕ Add New Signing feature - Implementation needed")
        self._wait_for_continue()
    
    def _handle_remove_player(self):
        """Handle removing players"""
        # Implementation for removing players
        print("➖ Remove Player feature - Implementation needed")
        self._wait_for_continue()
    
    def _handle_view_transfers(self):
        """Handle viewing transfer history"""
        # Implementation for viewing transfers
        print("📋 Transfer History feature - Implementation needed")
        self._wait_for_continue()

# Example usage
if __name__ == "__main__":
    manager = SquadManager({})
    
    # Cache some squads for testing
    manager.cache_all_squads()
    
    # Display Newcastle squad
    manager.display_squad_summary("Newcastle")
    
    # Interactive mode
    # manager.squad_interactive_manager()
