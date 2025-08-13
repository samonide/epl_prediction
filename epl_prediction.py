#!/usr/bin/env python3
"""
EPL (Premier League) match result prediction pipeline using FBR API data.

Features:
- Fetch last N seasons of EPL fixtures/results from FBR API (/matches, league_id=9)
- Build features: recent form (PPG), goal diffs, home/away splits, head-to-head last 5,
  simple Elo rating pre-match (with home advantage), rest-days
- Train multinomial logistic regression on historical matches (time-aware split)
- Predict probabilities for upcoming fixtures or a specific match-up

Usage examples:
  # 1) Install deps and set API key
  export FBR_API_KEY=your_key_here

  # 2) Sync last 5 seasons and cache
  python epl_fpl_prediction.py sync --seasons 5

  # 3) Train the model
  python epl_fpl_prediction.py train

  # 4) Predict upcoming fixtures for current season
  python epl_fpl_prediction.py predict-fixtures --top 10

  # 5) Predict a specific match
  python epl_fpl_prediction.py predict-match --home "Arsenal" --away "Chelsea"

Notes:
- Respecting API rate limits: one request every 3 seconds. This script sleeps when needed.
- Data cached under ./cache/ to reduce API calls. Model saved to ./models/.
- League id for Premier League is 9 per FBR API.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Auto-activate virtual environment if it exists and we're not already in one
def auto_activate_venv():
    """Automatically activate virtual environment if available and not already active."""
    # Check if we're already in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return  # Already in a virtual environment
    
    # Look for virtual environment
    script_dir = Path(__file__).parent
    venv_paths = [
        script_dir / '.venv',
        script_dir / 'venv', 
        script_dir / 'env'
    ]
    
    for venv_path in venv_paths:
        if venv_path.exists():
            # Determine the correct activation script path
            if sys.platform == "win32":
                python_exe = venv_path / "Scripts" / "python.exe"
            else:
                python_exe = venv_path / "bin" / "python"
            
            if python_exe.exists():
                # Re-execute script with virtual environment Python
                import subprocess
                cmd = [str(python_exe)] + sys.argv
                try:
                    result = subprocess.run(cmd, check=False)
                    sys.exit(result.returncode)
                except KeyboardInterrupt:
                    sys.exit(1)
                except Exception as e:
                    print(f"⚠️  Warning: Could not activate virtual environment: {e}")
                    break

# Call auto-activation before importing dependencies
try:
    auto_activate_venv()
except Exception:
    pass  # Continue if auto-activation fails

# Now import other modules
import argparse
import gzip
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib


# --------------------------- Config ---------------------------
LEAGUE_ID_EPL = 9
CACHE_DIR = Path(__file__).resolve().parent / "cache"
MODELS_DIR = Path(__file__).resolve().parent / "models"
CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Organized cache subdirectories
CACHE_MATCHES_DIR = CACHE_DIR / "matches"
CACHE_PLAYER_STATS_DIR = CACHE_DIR / "player_stats"
CACHE_TEAM_SCHEDULES_DIR = CACHE_DIR / "team_schedules"
CACHE_TEAM_STATS_DIR = CACHE_DIR / "team_stats"
CACHE_SQUAD_DIR = CACHE_DIR / "squads"
CACHE_INJURIES_DIR = CACHE_DIR / "injuries"

# Create subdirectories
for subdir in [CACHE_MATCHES_DIR, CACHE_PLAYER_STATS_DIR, CACHE_TEAM_SCHEDULES_DIR, 
               CACHE_TEAM_STATS_DIR, CACHE_SQUAD_DIR, CACHE_INJURIES_DIR]:
    subdir.mkdir(exist_ok=True)


# --------------------------- API Client ---------------------------
def _rate_limit(min_interval_sec: float):
    """Decorator to enforce a minimum interval between function calls per-process."""
    def decorator(func):
        last_called = {"t": 0.0}

        def wrapper(*args, **kwargs):
            now = time.time()
            elapsed = now - last_called["t"]
            if elapsed < min_interval_sec:
                time.sleep(min_interval_sec - elapsed)
            result = func(*args, **kwargs)
            last_called["t"] = time.time()
            return result

        return wrapper

    return decorator


@dataclass
class FBRClient:
    """Client for Football Betting Results API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FBR_API_KEY")
        self.base_url = "https://fbrapi.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; EPL-Predictor/1.0)"
        })

    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        if not self.api_key:
            raise ValueError("API key is required. Set FBR_API_KEY environment variable or pass it to the constructor.")
        
        url = f"{self.base_url}{endpoint}"
        all_params = {"api_key": self.api_key}
        if params:
            all_params.update(params)
        
        def make_request():
            response = self.session.get(url, params=all_params, timeout=30)
            response.raise_for_status()
            return response.json()
        
        # Simple retry logic
        for attempt in range(3):
            try:
                return make_request()
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                time.sleep(2 * (attempt + 1))


class APIFootballClient:
    """Client for API-Football (current squad data)"""
    
    def __init__(self, api_key: str = "02eb00e7497de4d328fa72e3365791b5"):
        self.api_key = api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.session = requests.Session()
        self.session.headers.update({
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "v3.football.api-sports.io"
        })
    
    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        url = f"{self.base_url}{endpoint}"
        
        def make_request():
            response = self.session.get(url, params=params or {}, timeout=30)
            response.raise_for_status()
            return response.json()
        
        # Simple retry logic
        for attempt in range(3):
            try:
                return make_request()
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                time.sleep(2 * (attempt + 1))
    
    def get_team_squad(self, team_id: int, season: int = 2024, force_refresh: bool = False) -> List[Dict]:
        """Get current squad for a team with caching"""
        cache_file = CACHE_SQUAD_DIR / f"team_{team_id}_squad_{season}.json"
        
        # Check cache first (unless force refresh)
        if not force_refresh and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            # Cache for 24 hours
            if cache_age < 86400:
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    print(f"Using cached squad data for team {team_id}")
                    return cached_data
                except:
                    pass  # If cache is corrupted, fetch fresh
        
        params = {"team": team_id, "season": season}
        data = self._get("/players/squads", params=params)
        response = data.get("response", [])
        
        # Cache the response
        if response:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(response, f, indent=2)
                print(f"Cached squad data for team {team_id}")
            except Exception as e:
                print(f"Warning: Could not cache squad data: {e}")
        
        return response
    
    def get_team_injuries(self, team_id: int, force_refresh: bool = False) -> List[Dict]:
        """Get current injury list for a team with caching"""
        cache_file = CACHE_INJURIES_DIR / f"team_{team_id}_injuries.json"
        
        # Check cache first (unless force refresh)
        if not force_refresh and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            # Cache for 12 hours
            if cache_age < 43200:
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    print(f"Using cached injury data for team {team_id}")
                    return cached_data
                except:
                    pass  # If cache is corrupted, fetch fresh
        
        params = {"team": team_id}
        data = self._get("/injuries", params=params)
        response = data.get("response", [])
        
        # Cache the response
        if response:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(response, f, indent=2)
                print(f"Cached injury data for team {team_id}")
            except Exception as e:
                print(f"Warning: Could not cache injury data: {e}")
        
        return response

    def search_teams(self, league_id: int = 39, season: int = 2024) -> List[Dict]:
        """Get teams in Premier League"""
        params = {"league": league_id, "season": season}
        data = self._get("/teams", params=params)
        return data.get("response", [])


class BookmakerAPIClient:
    """Secondary API-Football client for bookmaker odds data"""
    
    def __init__(self, api_key: str = "e66a648eb21c685297c1df4c8e0304cc"):
        self.api_key = api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.session = requests.Session()
        self.session.headers.update({
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "v3.football.api-sports.io"
        })

    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make GET request with retry logic"""
        url = f"{self.base_url}{endpoint}"
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise
        return {}

    def get_match_odds(self, fixture_id: int = None, home_team: str = None, away_team: str = None, 
                      date: str = None, force_refresh: bool = False) -> Dict:
        """Get betting odds for a match with caching (very short cache due to real-time changes)"""
        # Create cache key based on available identifiers
        cache_key = f"odds_{fixture_id}" if fixture_id else f"odds_{home_team}_{away_team}_{date}".replace(" ", "_")
        cache_file = CACHE_DIR / "odds" / f"{cache_key}.json"
        cache_file.parent.mkdir(exist_ok=True)
        
        # Very short cache (30 minutes) since odds change rapidly
        if not force_refresh and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 1800:  # 30 minutes
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    print(f"Using cached odds data for {cache_key}")
                    return cached_data
                except:
                    pass
        
        # Try to get fixture ID if not provided
        if not fixture_id and home_team and away_team:
            fixtures = self.get_fixtures_by_teams(home_team, away_team, date)
            if fixtures:
                fixture_id = fixtures[0].get('fixture', {}).get('id')
        
        if not fixture_id:
            print(f"Could not find fixture ID for {home_team} vs {away_team}")
            return {}
        
        params = {"fixture": fixture_id}
        data = self._get("/odds", params=params)
        response = data.get("response", [])
        
        # Process odds from multiple bookmakers
        odds_data = {"fixture_id": fixture_id, "bookmakers": []}
        
        if response:
            for fixture_odds in response:
                bookmakers = fixture_odds.get("bookmakers", [])
                for bookmaker in bookmakers:
                    bm_name = bookmaker.get("name", "Unknown")
                    bets = bookmaker.get("bets", [])
                    
                    # Look for match winner odds (bet_id: 1)
                    for bet in bets:
                        if bet.get("id") == 1:  # Match Winner
                            values = bet.get("values", [])
                            odds_dict = {"name": bm_name, "odds": {}}
                            
                            for value in values:
                                outcome = value.get("value")  # Home, Draw, Away
                                odd = value.get("odd")
                                if outcome and odd:
                                    odds_dict["odds"][outcome] = float(odd)
                            
                            if odds_dict["odds"]:
                                odds_data["bookmakers"].append(odds_dict)
        
        # Cache the processed data
        try:
            with open(cache_file, 'w') as f:
                json.dump(odds_data, f, indent=2)
            print(f"Cached odds data for {cache_key}")
        except Exception as e:
            print(f"Warning: Could not cache odds data: {e}")
        
        return odds_data

    def get_fixtures_by_teams(self, home_team: str, away_team: str, date: str = None) -> List[Dict]:
        """Find fixtures by team names"""
        params = {"league": 39, "season": 2024}  # Premier League
        if date:
            params["date"] = date
        
        data = self._get("/fixtures", params=params)
        fixtures = data.get("response", [])
        
        # Filter by team names
        matching_fixtures = []
        for fixture in fixtures:
            teams = fixture.get("teams", {})
            home = teams.get("home", {}).get("name", "")
            away = teams.get("away", {}).get("name", "")
            
            if home_team.lower() in home.lower() and away_team.lower() in away.lower():
                matching_fixtures.append(fixture)
        
        return matching_fixtures


# Team mapping between FBR and API-Football
TEAM_MAPPING = {
    # FBR team_id -> API-Football team_id
    "822bd0ba": 40,    # Liverpool
    "cff3d9bb": 49,    # Chelsea  
    "18bb7c10": 42,    # Arsenal
    "19538871": 50,    # Manchester City
    "361ca564": 33,    # Manchester United
    "47c64c55": 47,    # Tottenham
    "8602292d": 55,    # Newcastle United
    "943e8050": 66,    # Aston Villa
    "8cec06e1": 48,    # West Ham
    "b8fd03ef": 35,    # Bournemouth
    "4ba7cbea": 51,    # Brighton
    "cd051869": 52,    # Crystal Palace
    "7c21e445": 45,    # Everton
    "fd962109": 36,    # Fulham
    "e4a775cb": 39,    # Wolves
    "d3fd31cc": 41,    # Southampton
    "b74092de": 57,    # Brentford
    "e297cd13": 46,    # Nottingham Forest
    "1df6b87e": 54,    # Leicester City
    "a2d435b3": 54,    # Ipswich Town (need correct ID)
}


def convert_odds_to_probabilities(odds_data: Dict) -> Dict:
    """Convert bookmaker odds to implied probabilities and average them"""
    if not odds_data or not odds_data.get("bookmakers"):
        return {}
    
    bookmakers = odds_data["bookmakers"]
    home_odds = []
    draw_odds = []
    away_odds = []
    
    # Collect odds from all bookmakers
    for bm in bookmakers:
        odds = bm.get("odds", {})
        if "Home" in odds:
            home_odds.append(odds["Home"])
        if "Draw" in odds:
            draw_odds.append(odds["Draw"])
        if "Away" in odds:
            away_odds.append(odds["Away"])
    
    # Calculate average odds
    avg_odds = {}
    if home_odds:
        avg_odds["Home"] = sum(home_odds) / len(home_odds)
    if draw_odds:
        avg_odds["Draw"] = sum(draw_odds) / len(draw_odds)
    if away_odds:
        avg_odds["Away"] = sum(away_odds) / len(away_odds)
    
    # Convert to implied probabilities
    probabilities = {}
    total_prob = 0
    
    for outcome, odd in avg_odds.items():
        prob = 1 / odd if odd > 0 else 0
        probabilities[outcome] = prob
        total_prob += prob
    
    # Normalize probabilities (remove bookmaker margin)
    if total_prob > 0:
        for outcome in probabilities:
            probabilities[outcome] = probabilities[outcome] / total_prob
    
    return {
        "bookmaker_probs": probabilities,
        "avg_odds": avg_odds,
        "num_bookmakers": len(bookmakers)
    }


def combine_ml_and_odds_predictions(ml_probs: Dict, odds_data: Dict, ml_weight: float = 0.7) -> Dict:
    """Combine ML model predictions with bookmaker odds for enhanced accuracy"""
    odds_analysis = convert_odds_to_probabilities(odds_data)
    
    if not odds_analysis or not odds_analysis.get("bookmaker_probs"):
        print("No valid bookmaker odds available, using ML predictions only")
        return {
            "final_probs": ml_probs,
            "method": "ML_ONLY",
            "odds_available": False
        }
    
    bookmaker_probs = odds_analysis["bookmaker_probs"]
    odds_weight = 1 - ml_weight
    
    # Map ML probability keys to bookmaker keys
    mapping = {"P(H)": "Home", "P(D)": "Draw", "P(A)": "Away"}
    
    combined_probs = {}
    for ml_key, bm_key in mapping.items():
        ml_prob = ml_probs.get(ml_key, 0.0)
        bm_prob = bookmaker_probs.get(bm_key, 0.0)
        
        # Weighted combination
        combined_prob = (ml_prob * ml_weight) + (bm_prob * odds_weight)
        combined_probs[ml_key] = combined_prob
    
    # Normalize combined probabilities
    total = sum(combined_probs.values())
    if total > 0:
        for key in combined_probs:
            combined_probs[key] = combined_probs[key] / total
    
    return {
        "final_probs": combined_probs,
        "ml_probs": ml_probs,
        "bookmaker_probs": bookmaker_probs,
        "avg_odds": odds_analysis.get("avg_odds", {}),
        "num_bookmakers": odds_analysis.get("num_bookmakers", 0),
        "ml_weight": ml_weight,
        "odds_weight": odds_weight,
        "method": "COMBINED",
        "odds_available": True
    }

@dataclass  
class FBRClient:
    api_key: Optional[str] = None
    base_url: str = "https://fbrapi.com"

    def __post_init__(self):
        if not self.api_key:
            # Try environment variable
            self.api_key = os.getenv("FBR_API_KEY")
        # Fallback to user-provided key if still missing
        if not self.api_key:
            self.api_key = "6t1DRADfXqi-iIUZJQZEj9_sv3oCvNt6yzVUbSGvYGk"
        self.session = requests.Session()

    @_rate_limit(3.0)
    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        last_exc = None
        for attempt in range(3):
            r = self.session.get(url, params=params or {}, headers=headers, timeout=60)
            if r.status_code == 401 and attempt == 0:
                # Try to generate a fresh key and retry once
                try:
                    new_key = self.generate_api_key()
                    self.api_key = new_key
                    headers = {"X-API-Key": self.api_key}
                    continue
                except Exception as e:
                    last_exc = e
            if 500 <= r.status_code < 600:
                last_exc = requests.HTTPError(f"{r.status_code} for {url}")
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        # If we exit loop, raise last error or generic
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Failed to GET {url}")

    @_rate_limit(3.0)
    def _post(self, path: str, data: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        r = self.session.post(url, json=data or {}, timeout=60)
        r.raise_for_status()
        return r.json()

    def generate_api_key(self) -> str:
        data = self._post("/generate_api_key")
        key = data.get("api_key")
        if not key:
            raise RuntimeError("Failed to generate API key.")
        return key

    def league_seasons(self, league_id: int) -> List[str]:
        data = self._get("/league-seasons", params={"league_id": league_id})
        seasons = [row.get("season_id") for row in data.get("data", []) if row.get("season_id")]
        return seasons

    def matches_by_league_season(self, league_id: int, season_id: str) -> List[Dict]:
        data = self._get("/matches", params={"league_id": league_id, "season_id": season_id})
        return data.get("data", [])

    def matches_by_team_season(self, team_id: str, league_id: Optional[int] = None, season_id: Optional[str] = None) -> List[Dict]:
        params: Dict[str, object] = {"team_id": team_id}
        if league_id is not None:
            params["league_id"] = league_id
        if season_id is not None:
            params["season_id"] = season_id
        data = self._get("/matches", params=params)
        return data.get("data", [])

    def league_standings(self, league_id: int, season_id: Optional[str] = None) -> List[Dict]:
        params = {"league_id": league_id}
        if season_id:
            params["season_id"] = season_id
        data = self._get("/league-standings", params=params)
        return data.get("data", [])

    def team_data(self, team_id: str, season_id: Optional[str] = None) -> Dict:
        params = {"team_id": team_id}
        if season_id:
            params["season_id"] = season_id
        data = self._get("/teams", params=params)
        return data

    def team_season_stats(self, league_id: int, season_id: Optional[str] = None) -> List[Dict]:
        params = {"league_id": league_id}
        if season_id:
            params["season_id"] = season_id
        data = self._get("/team-season-stats", params=params)
        return data.get("data", [])

    def player_season_stats(self, team_id: str, league_id: int, season_id: Optional[str] = None) -> List[Dict]:
        params = {"team_id": team_id, "league_id": league_id}
        if season_id:
            params["season_id"] = season_id
        data = self._get("/player-season-stats", params=params)
        return data.get("players", [])

    def player_match_stats(self, player_id: str, league_id: int, season_id: Optional[str] = None) -> List[Dict]:
        params = {"player_id": player_id, "league_id": league_id}
        if season_id:
            params["season_id"] = season_id
        data = self._get("/player-match-stats", params=params)
        return data.get("data", [])

    def all_players_match_stats(self, match_id: str) -> List[Dict]:
        data = self._get("/all-players-match-stats", params={"match_id": match_id})
        return data.get("data", [])
    
    def team_squad(self, team_id: str, season_id: Optional[str] = None) -> List[Dict]:
        """Get current squad for a team"""
        params = {"team_id": team_id}
        if season_id:
            params["season_id"] = season_id
        data = self._get("/team-squad", params=params)
        return data.get("data", [])


# --------------------------- Caching Helpers ---------------------------
def cache_path_for_matches(league_id: int, season_id: str) -> Path:
    """Get cache path for match data in organized subdirectory"""
    # Prefer compressed json.gz; maintain backward compatibility with existing .json
    gz = CACHE_MATCHES_DIR / f"matches_league{league_id}_{season_id}.json.gz"
    return gz


def cache_path_for_team_schedule(team_id: str, season_id: str) -> Path:
    """Get cache path for team schedule data in organized subdirectory"""
    return CACHE_TEAM_SCHEDULES_DIR / f"team_schedule_{team_id}_{season_id}.json"


def cache_path_for_player_stats(team_id: str, season_id: str) -> Path:
    """Get cache path for player stats data in organized subdirectory"""
    return CACHE_PLAYER_STATS_DIR / f"player_stats_{team_id}_{season_id}.json"


def cache_path_for_team_stats(league_id: int, season_id: str) -> Path:
    """Get cache path for team stats data in organized subdirectory"""
    return CACHE_TEAM_STATS_DIR / f"team_stats_league{league_id}_{season_id}.json"


def get_team_id_by_name(client: FBRClient, team_name: str, season_id: str = None) -> Optional[str]:
    """Get team ID by team name from cached data"""
    if not season_id:
        season_id = get_current_season()
    
    # Try to find team ID from matches cache first
    try:
        p_gz = cache_path_for_matches(LEAGUE_ID_EPL, season_id)
        p_json = Path(str(p_gz).replace(".json.gz", ".json"))
        cached = _read_json_any(p_gz, p_json)
        
        if cached:
            for match in cached:
                if match.get("home") == team_name and match.get("home_team_id"):
                    return match["home_team_id"]
                if match.get("away") == team_name and match.get("away_team_id"):
                    return match["away_team_id"]
    except Exception:
        pass
    
    # Try previous season as fallback
    try:
        year = int(season_id.split("-")[0])
        prev_season = f"{year-1}-{year}"
        p_gz = cache_path_for_matches(LEAGUE_ID_EPL, prev_season)
        p_json = Path(str(p_gz).replace(".json.gz", ".json"))
        cached = _read_json_any(p_gz, p_json)
        
        if cached:
            for match in cached:
                if match.get("home") == team_name and match.get("home_team_id"):
                    return match["home_team_id"]
                if match.get("away") == team_name and match.get("away_team_id"):
                    return match["away_team_id"]
    except Exception:
        pass
    
    return None


def extract_season_id_from_cache_path(p: Path) -> str:
    name = p.name
    if name.endswith(".json.gz"):
        name = name[:-8]  # remove .json.gz
    elif name.endswith(".json"):
        name = name[:-5]
    # now remove prefix
    if name.startswith("matches_league"):
        parts = name.split("_")
        return parts[-1]
    return name


def _read_json_any(path_gz: Path, fallback_json: Optional[Path] = None) -> Optional[List[Dict]]:
    if path_gz.exists():
        import gzip
        try:
            with gzip.open(path_gz, "rt", encoding="utf-8") as f:
                data = json.load(f)
                # If gzip file exists but is empty, try fallback
                if data:
                    return data
        except Exception:
            pass
    if fallback_json and fallback_json.exists():
        with open(fallback_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _write_json_gz(path_gz: Path, rows: List[Dict]) -> None:
    import gzip
    with gzip.open(path_gz, "wt", encoding="utf-8") as f:
        json.dump(rows, f)


def _write_json(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def load_or_fetch_team_schedule(client: FBRClient, team_id: str, season_id: str, cache_only: bool = False) -> List[Dict]:
    p = cache_path_for_team_schedule(team_id, season_id)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                pass
    if cache_only:
        return []
    # Prefer matches endpoint with team_id for reliability
    try:
        sch = client.matches_by_team_season(team_id, league_id=LEAGUE_ID_EPL, season_id=season_id)
    except Exception:
        # Fallback to /teams endpoint if available
        data = client.team_data(team_id, season_id)
        sch = data.get("team_schedule", {}).get("data", [])
    _write_json(p, sch)
    return sch


def load_or_fetch_player_stats(client: FBRClient, team_id: str, season_id: str, cache_only: bool = False) -> List[Dict]:
    """Load or fetch player season stats for a team."""
    p = cache_path_for_player_stats(team_id, season_id)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                pass
    if cache_only:
        return []
    try:
        players = client.player_season_stats(team_id, LEAGUE_ID_EPL, season_id)
        _write_json(p, players)
        return players
    except Exception:
        return []


def load_or_fetch_team_stats(client: FBRClient, league_id: int, season_id: str, cache_only: bool = False) -> List[Dict]:
    """Load or fetch team season stats for a league season."""
    p = cache_path_for_team_stats(league_id, season_id)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                pass
    if cache_only:
        return []
    try:
        stats = client.team_season_stats(league_id, season_id)
        _write_json(p, stats)
        return stats
    except Exception:
        return []


def _standings_team_map(standings_payload: List[Dict]) -> Dict[str, str]:
    team_map: Dict[str, str] = {}
    for table in standings_payload:
        rows = table.get("standings", [])
        for r in rows:
            tid = r.get("team_id")
            tname = r.get("team_name") or r.get("team")
            if tid and tname and tid not in team_map:
                team_map[tid] = tname
    return team_map


def build_season_matches_with_scores(client: FBRClient, league_id: int, season_id: str) -> List[Dict]:
    # Build from team schedules, using only Home entries to deduplicate
    team_map: Dict[str, str] = {}
    # Try standings first
    try:
        standings = client.league_standings(league_id, season_id)
        team_map = _standings_team_map(standings)
    except Exception:
        team_map = {}
    # Fallback: team-season-stats
    if not team_map:
        try:
            tstats = client.team_season_stats(league_id, season_id)
            for entry in tstats:
                meta = entry.get("meta_data") or entry.get("meta") or {}
                tid = (meta.get("team_id") or meta.get("teamid") or
                       (entry.get("meta_data", {}).get("team", {}).get("team_id") if isinstance(entry.get("meta_data"), dict) else None))
                tname = (meta.get("team_name") or meta.get("team") or
                         (entry.get("meta_data", {}).get("team", {}).get("team_name") if isinstance(entry.get("meta_data"), dict) else None))
                if tid and tname and tid not in team_map:
                    team_map[tid] = tname
        except Exception:
            pass
    if not team_map:
        # Fallback: infer from matches list (prefer local cache)
        base_rows = []
        p_gz = cache_path_for_matches(league_id, season_id)
        p_json = Path(str(p_gz).replace(".json.gz", ".json"))
        cached = _read_json_any(p_gz, p_json)
        if cached is None:
            try:
                base_rows = client.matches_by_league_season(league_id, season_id)
            except Exception:
                base_rows = []
        else:
            base_rows = cached
        for r in base_rows or []:
            if r.get("home_team_id") and r.get("home"):
                team_map[r["home_team_id"]] = r["home"]
            if r.get("away_team_id") and r.get("away"):
                team_map[r["away_team_id"]] = r["away"]
    # Fallback: derive team ids from existing schedule caches for the season
    if not team_map:
        import glob
        for path in glob.glob(str(CACHE_TEAM_SCHEDULES_DIR / f"team_schedule_*_{season_id}.json")):
            tid = Path(path).name.split("_")[2]
            if tid and tid not in team_map:
                team_map[tid] = None
    if not team_map:
        raise RuntimeError(f"Could not determine team list for league {league_id} season {season_id}")
    rows: List[Dict] = []
    for team_id, team_name in team_map.items():
        try:
            # Prefer using cache first to avoid API flakiness
            sch = load_or_fetch_team_schedule(client, team_id, season_id, cache_only=True)
            if not sch:
                sch = load_or_fetch_team_schedule(client, team_id, season_id)
        except Exception:
            continue
        for m in sch:
            try:
                if int(m.get("league_id", league_id)) != league_id:
                    continue
            except Exception:
                continue
            ha = str(m.get("home_away", "")).lower()
            if ha != "home":
                continue
            match_id = m.get("match_id")
            opp_id = m.get("opponent_id")
            opp_name = team_map.get(opp_id) or m.get("opponent")
            rows.append({
                "season_id": season_id,
                "match_id": match_id,
                "date": m.get("date"),
                "time": m.get("time"),
                "round": m.get("round"),
                "wk": m.get("wk"),
                "home": team_name,
                "home_team_id": team_id,
                "away": opp_name,
                "away_team_id": opp_id,
                "home_team_score": m.get("gf"),
                "away_team_score": m.get("ga"),
                "venue": None,
            })
    # Sort and return
    rows.sort(key=lambda r: (r.get("date") or "", r.get("time") or ""))
    return rows


def load_or_fetch_matches(client: FBRClient, league_id: int, season_id: str) -> List[Dict]:
    p_gz = cache_path_for_matches(league_id, season_id)
    p_json = Path(str(p_gz).replace(".json.gz", ".json"))
    cached = _read_json_any(p_gz, p_json)
    def has_scores(rows: Optional[List[Dict]]) -> bool:
        if not rows:
            return False
        for r in rows:
            if r.get("home_team_score") is not None and r.get("away_team_score") is not None:
                return True
        return False
    
    # For current/future seasons, return cached data even without scores
    current_season = get_current_season()
    if cached and season_id >= current_season:
        return cached
    
    if has_scores(cached):
        return cached  # already has labeled results
    # Build from team schedules and save
    built = build_season_matches_with_scores(client, league_id, season_id)
    _write_json_gz(p_gz, built)
    return built


def get_last_n_seasons(client: FBRClient, league_id: int, n: int = 5) -> List[str]:
    def season_sort_key(s: str):
        try:
            parts = str(s).split("-")
            return int(parts[0])
        except Exception:
            return 0

    def list_cached_seasons() -> List[str]:
        # From league match caches
        cached = sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json.gz")) or sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json"))
        seasons = {extract_season_id_from_cache_path(p) for p in cached}
        # From team schedule caches
        for p in CACHE_TEAM_SCHEDULES_DIR.glob("team_schedule_*_*.json"):
            try:
                season = p.stem.split("_")[-1]
                if season:
                    seasons.add(season)
            except Exception:
                continue
        return sorted(seasons, key=season_sort_key)

    def plausible_last_n() -> List[str]:
        # Heuristic seasons based on current date - prioritize current season
        current_season = get_current_season()
        today = datetime.now(UTC)
        start_year = today.year if today.month >= 7 else today.year - 1
        
        # Generate seasons with current season first
        seasons = [current_season]
        for i in range(1, n):
            year = start_year - i
            seasons.append(f"{year}-{year + 1}")
        
        # Remove duplicates and sort properly
        unique_seasons = list(dict.fromkeys(seasons))  # Preserves order, removes duplicates
        return unique_seasons[:n]

    try:
        seasons = client.league_seasons(league_id)
        seasons_sorted = sorted(seasons, key=season_sort_key)
        if seasons_sorted:
            # Ensure current season is included if available
            current_season = get_current_season()
            result_seasons = seasons_sorted[-n:]
            if current_season not in result_seasons and current_season in seasons_sorted:
                # Replace oldest with current season
                result_seasons = result_seasons[1:] + [current_season]
            return result_seasons
    except Exception:
        # Fall back to cache or heuristic
        pass
    cached_seasons = list_cached_seasons()
    if cached_seasons:
        # Prioritize current season in cached seasons too
        current_season = get_current_season()
        result_seasons = cached_seasons[-n:]
        if current_season not in result_seasons and current_season in cached_seasons:
            result_seasons = result_seasons[1:] + [current_season]
        return result_seasons
    return plausible_last_n()


def get_current_season_id(client: FBRClient, league_id: int) -> Optional[str]:
    """Return the most recent season_id for the league."""
    def season_sort_key(s: str):
        try:
            return int(str(s).split("-")[0])
        except Exception:
            return 0
    try:
        seasons = client.league_seasons(league_id)
        if seasons:
            return sorted(seasons, key=season_sort_key)[-1]
    except Exception:
        pass
    # Fallback to cache
    cached = sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json.gz")) or sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json"))
    seasons = [extract_season_id_from_cache_path(c) for c in cached]
    if seasons:
        return sorted(seasons, key=season_sort_key)[-1]
    # Heuristic current season
    today = datetime.now(UTC)
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year}-{start_year + 1}"


def get_season_teams_from_matches(client: FBRClient, league_id: int, season_id: str) -> Dict[str, str]:
    """Infer team_id -> team_name from league-level matches list (scores may be null)."""
    rows = client.matches_by_league_season(league_id, season_id)
    team_map: Dict[str, str] = {}
    for r in rows:
        hid, hname = r.get("home_team_id"), r.get("home")
        aid, aname = r.get("away_team_id"), r.get("away")
        if hid and hname and hid not in team_map:
            team_map[hid] = hname
        if aid and aname and aid not in team_map:
            team_map[aid] = aname
    return team_map


# --------------------------- Data and Features ---------------------------
def build_matches_dataframe(all_season_rows: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Build a DataFrame of matches across seasons with consistent columns."""
    records: List[Dict] = []
    for season_id, rows in all_season_rows.items():
        for r in rows:
            # League match data schema (see docs)
            # We only keep league matches with known teams, add season
            records.append({
                "season_id": season_id,
                "match_id": r.get("match_id"),
                "date": r.get("date"),
                "time": r.get("time"),
                "round": r.get("round"),
                "wk": r.get("wk"),
                "home": r.get("home"),
                "home_team_id": r.get("home_team_id"),
                "away": r.get("away"),
                "away_team_id": r.get("away_team_id"),
                "home_team_score": r.get("home_team_score"),
                "away_team_score": r.get("away_team_score"),
                "venue": r.get("venue"),
            })
    df = pd.DataFrame.from_records(records)
    # Parse date, filter obvious invalids
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "wk"]).reset_index(drop=True)
    return df


def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    def outcome_row(row):
        hs, as_ = row["home_team_score"], row["away_team_score"]
        if pd.isna(hs) or pd.isna(as_):
            return np.nan
        if hs > as_:
            return "H"
        if hs < as_:
            return "A"
        return "D"

    df = df.copy()
    df["outcome"] = df.apply(outcome_row, axis=1)
    return df


def compute_elo_features(df: pd.DataFrame, k_factor: float = 20.0, home_adv: float = 50.0) -> pd.DataFrame:
    """Compute simple Elo pre-match ratings for home/away teams and add elo_diff feature.
    Ratings initialized to 1500 per team per season-start, but we maintain across seasons.
    """
    df = df.copy()
    ratings: Dict[str, float] = {}
    pre_home_elo: List[float] = []
    pre_away_elo: List[float] = []

    def get_rating(team_id: str) -> float:
        return ratings.get(team_id, 1500.0)

    for _, row in df.iterrows():
        h_id = row["home_team_id"]
        a_id = row["away_team_id"]
        h_elo = get_rating(h_id)
        a_elo = get_rating(a_id)
        pre_home_elo.append(h_elo)
        pre_away_elo.append(a_elo)

        # Update post-match if result known
        hs, as_ = row["home_team_score"], row["away_team_score"]
        if pd.isna(hs) or pd.isna(as_):
            continue
        # Expected score with home advantage added to home rating
        exp_home = 1.0 / (1.0 + 10 ** (-(h_elo + home_adv - a_elo) / 400))
        exp_away = 1.0 - exp_home
        if hs > as_:
            s_home, s_away = 1.0, 0.0
        elif hs < as_:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5
        ratings[h_id] = h_elo + k_factor * (s_home - exp_home)
        ratings[a_id] = a_elo + k_factor * (s_away - exp_away)

    df["elo_home_pre"] = pre_home_elo
    df["elo_away_pre"] = pre_away_elo
    df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]
    return df


def compute_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute recent form PPG and goal-diff averages for each team before each match."""
    df = df.copy()

    # Build per-team timeline combining home/away into one series to compute rolling stats
    home_cols = [
        "date",
        "season_id",
        "home_team_id",
        "home_team_score",
        "away_team_score",
    ]
    away_cols = [
        "date",
        "season_id",
        "away_team_id",
        "home_team_score",
        "away_team_score",
    ]
    home_df = df[home_cols].rename(columns={
        "home_team_id": "team_id",
        "home_team_score": "gf",
        "away_team_score": "ga",
    })
    away_df = df[away_cols].rename(columns={
        "away_team_id": "team_id",
        "home_team_score": "ga",
        "away_team_score": "gf",
    })
    long_df = pd.concat([home_df, away_df], ignore_index=True).sort_values("date")
    # Points per match
    long_df["pts"] = np.where(long_df["gf"] > long_df["ga"], 3, np.where(long_df["gf"] < long_df["ga"], 0, 1))
    long_df["gd"] = long_df["gf"] - long_df["ga"]

    # Rolling features per team (excluding current match via shift)
    long_df = long_df.sort_values(["team_id", "date"]).reset_index(drop=True)
    long_df["ppg_last5"] = (
        long_df.groupby("team_id")["pts"].rolling(window, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    )
    long_df["gd_avg_last5"] = (
        long_df.groupby("team_id")["gd"].rolling(window, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    )
    long_df["gf_avg_last5"] = (
        long_df.groupby("team_id")["gf"].rolling(window, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    )
    long_df["ga_avg_last5"] = (
        long_df.groupby("team_id")["ga"].rolling(window, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    )

    # Home-only and Away-only rolling PPG
    # Create flags to later merge onto match df
    home_only = df[["date", "home_team_id"]].copy()
    home_only["is_home"] = 1
    away_only = df[["date", "away_team_id"]].copy()
    away_only["is_home"] = 0
    home_only = home_only.rename(columns={"home_team_id": "team_id"})
    away_only = away_only.rename(columns={"away_team_id": "team_id"})
    ha_timeline = pd.concat([home_only, away_only], ignore_index=True).sort_values(["team_id", "date"])  # occurrence per team per match

    # Merge pts back to ha_timeline
    ha_merge = ha_timeline.merge(long_df[["team_id", "date", "pts"]], on=["team_id", "date"], how="left")
    ha_merge["ppg_home_last5"] = (
        ha_merge[ha_merge["is_home"] == 1]
        .groupby("team_id")["pts"].rolling(window, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    )
    ha_merge["ppg_away_last5"] = (
        ha_merge[ha_merge["is_home"] == 0]
        .groupby("team_id")["pts"].rolling(window, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    )

    # For both rows (home and away entries), forward/back-fill per team
    ha_merge["ppg_home_last5"] = ha_merge.groupby("team_id")["ppg_home_last5"].ffill()
    ha_merge["ppg_away_last5"] = ha_merge.groupby("team_id")["ppg_away_last5"].ffill()

    # Join features back to original match df for both home and away teams
    # Merge general form/gd
    df = df.merge(
        long_df[["team_id", "date", "ppg_last5", "gd_avg_last5", "gf_avg_last5", "ga_avg_last5"]],
        left_on=["home_team_id", "date"],
        right_on=["team_id", "date"],
        how="left",
    ).rename(columns={
        "ppg_last5": "home_ppg_last5", 
        "gd_avg_last5": "home_gd_avg_last5",
        "gf_avg_last5": "home_gf_avg_last5",
        "ga_avg_last5": "home_ga_avg_last5"
    }).drop(columns=["team_id"])
    df = df.merge(
        long_df[["team_id", "date", "ppg_last5", "gd_avg_last5", "gf_avg_last5", "ga_avg_last5"]],
        left_on=["away_team_id", "date"],
        right_on=["team_id", "date"],
        how="left",
    ).rename(columns={
        "ppg_last5": "away_ppg_last5", 
        "gd_avg_last5": "away_gd_avg_last5",
        "gf_avg_last5": "away_gf_avg_last5",
        "ga_avg_last5": "away_ga_avg_last5"
    }).drop(columns=["team_id"])

    # Merge home/away split ppg
    df = df.merge(
        ha_merge[["team_id", "date", "ppg_home_last5", "ppg_away_last5"]],
        left_on=["home_team_id", "date"],
        right_on=["team_id", "date"],
        how="left",
    ).rename(columns={"ppg_home_last5": "home_home_ppg_last5", "ppg_away_last5": "home_away_ppg_last5"}).drop(columns=["team_id"])
    df = df.merge(
        ha_merge[["team_id", "date", "ppg_home_last5", "ppg_away_last5"]],
        left_on=["away_team_id", "date"],
        right_on=["team_id", "date"],
        how="left",
    ).rename(columns={"ppg_home_last5": "away_home_ppg_last5", "ppg_away_last5": "away_away_ppg_last5"}).drop(columns=["team_id"])

    return df


def compute_advanced_features(df: pd.DataFrame, client: FBRClient) -> pd.DataFrame:
    """Add advanced features like xG, team strength metrics, etc."""
    df = df.copy()
    
    # Initialize additional feature columns
    df["home_xg_for_avg"] = np.nan
    df["home_xg_ag_avg"] = np.nan
    df["away_xg_for_avg"] = np.nan
    df["away_xg_ag_avg"] = np.nan
    df["home_strength_rating"] = np.nan
    df["away_strength_rating"] = np.nan
    
    # Process each season to get team stats
    for season_id in df["season_id"].dropna().unique():
        try:
            team_stats = load_or_fetch_team_stats(client, LEAGUE_ID_EPL, season_id, cache_only=True)
            if not team_stats:
                continue
                
            # Build team strength mapping from advanced stats
            team_metrics = {}
            for team_stat in team_stats:
                meta = team_stat.get("meta_data", {})
                team_id = meta.get("team_id")
                if not team_id:
                    continue
                    
                stats = team_stat.get("stats", {}).get("stats", {})
                if not stats:
                    continue
                    
                # Calculate strength rating from multiple metrics
                xg_for = stats.get("avg_xg", 0)
                xga = stats.get("ttl_xg", 0) / max(stats.get("matches_played", 1), 1)  # avg xG against
                ppg = stats.get("ttl_gls", 0) * 3 / max(stats.get("matches_played", 1), 1) if stats.get("ttl_gls", 0) > stats.get("ttl_gls_ag", stats.get("ttl_gls", 0)) else 1
                
                # Composite strength rating
                strength = (xg_for * 0.4) + (ppg * 0.6) - (xga * 0.3)
                
                team_metrics[team_id] = {
                    "xg_for_avg": xg_for,
                    "xg_ag_avg": xga,
                    "strength": strength
                }
            
            # Apply to dataframe
            season_mask = df["season_id"] == season_id
            for idx in df[season_mask].index:
                home_id = df.loc[idx, "home_team_id"]
                away_id = df.loc[idx, "away_team_id"]
                
                if home_id in team_metrics:
                    df.loc[idx, "home_xg_for_avg"] = team_metrics[home_id]["xg_for_avg"]
                    df.loc[idx, "home_xg_ag_avg"] = team_metrics[home_id]["xg_ag_avg"]
                    df.loc[idx, "home_strength_rating"] = team_metrics[home_id]["strength"]
                    
                if away_id in team_metrics:
                    df.loc[idx, "away_xg_for_avg"] = team_metrics[away_id]["xg_for_avg"]
                    df.loc[idx, "away_xg_ag_avg"] = team_metrics[away_id]["xg_ag_avg"]
                    df.loc[idx, "away_strength_rating"] = team_metrics[away_id]["strength"]
                    
        except Exception as e:
            print(f"Warning: Could not load team stats for season {season_id}: {e}")
            continue
    
    # Calculate strength differential
    df["strength_differential"] = df["home_strength_rating"] - df["away_strength_rating"]
    df["xg_differential"] = (df["home_xg_for_avg"] + df["away_xg_ag_avg"]) - (df["away_xg_for_avg"] + df["home_xg_ag_avg"])
    
    return df


def compute_head_to_head_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute H2H PPG for the home team vs away team over last N meetings across seasons."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    key = df[["home_team_id", "away_team_id", "date", "home_team_score", "away_team_score"]].copy()
    # Outcome from home team perspective for h2h
    key["home_pts"] = np.where(
        key["home_team_score"] > key["away_team_score"], 3, np.where(key["home_team_score"] < key["away_team_score"], 0, 1)
    )

    # For each pair (unordered), build a timeline
    key["pair"] = key.apply(
        lambda r: "::".join(sorted([str(r["home_team_id"]), str(r["away_team_id"])])), axis=1
    )

    # Rolling mean of home points in previous meetings when home team is as recorded in the row
    h2h_ppg: List[float] = []
    last_meetings: Dict[str, List[float]] = {}
    last_dates: Dict[str, List[pd.Timestamp]] = {}
    for _, r in key.iterrows():
        pair = r["pair"]
        # Compute average of last N recorded home_pts for this ordered direction
        pts_list = last_meetings.get((pair, "home_as_left"), [])
        # If there were previous meetings, avg; else nan
        h2h_ppg.append(float(np.mean(pts_list[-window:])) if pts_list else np.nan)
        # Append current meeting from this row's perspective
        pts_list.append(r["home_pts"])
        last_meetings[(pair, "home_as_left")] = pts_list
    df["h2h_home_ppg_last5"] = h2h_ppg
    return df


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days since last match for each team before the match date."""
    df = df.copy()
    df = df.sort_values("date")

    def team_last_date_map(team_col: str) -> pd.Series:
        last_date_per_team: Dict[str, Optional[pd.Timestamp]] = {}
        rest_days: List[Optional[float]] = []
        for _, r in df.iterrows():
            team_id = r[team_col]
            d = r["date"]
            prev = last_date_per_team.get(team_id)
            rest_days.append((d - prev).days if prev is not None and pd.notna(prev) and pd.notna(d) else np.nan)
            last_date_per_team[team_id] = d
        return pd.Series(rest_days, index=df.index)

    df["home_rest_days"] = team_last_date_map("home_team_id")
    df["away_rest_days"] = team_last_date_map("away_team_id")
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (labels H/D/A) for rows with known outcomes."""
    df = add_outcome(df)
    # Known outcomes only
    train_df = df[pd.notna(df["outcome"])].copy()
    feature_cols = [
        "elo_diff",
        "home_ppg_last5",
        "away_ppg_last5",
        "home_gd_avg_last5",
        "away_gd_avg_last5",
        "home_gf_avg_last5",
        "home_ga_avg_last5",
        "away_gf_avg_last5",
        "away_ga_avg_last5",
        "home_home_ppg_last5",
        "away_away_ppg_last5",
        "h2h_home_ppg_last5",
        "strength_differential",
        "xg_differential",
    ]
    X = train_df[feature_cols].copy()
    y = train_df["outcome"].astype("category")
    return X, y


def get_enhanced_top_scorers(client: FBRClient, home_team_id: str, away_team_id: str, season_id: str, 
                           h2h_data: Dict = None, home_form: Dict = None, away_form: Dict = None) -> Dict[str, List[Dict]]:
    """Get top scorers for both teams with enhanced analysis including current squad, transfers, and form."""
    result = {"home": [], "away": []}
    
    # Initialize API-Football client for current squad data
    api_football = APIFootballClient()
    
    for team_key, team_id in [("home", home_team_id), ("away", away_team_id)]:
        try:
            # Get current squad from API-Football if team mapping exists
            current_squad = []
            api_team_id = TEAM_MAPPING.get(team_id)
            if api_team_id:
                try:
                    squad_data = api_football.get_team_squad(api_team_id, 2024)
                    if squad_data:
                        # Extract player data from API-Football response
                        squad_info = squad_data[0] if squad_data else {}
                        current_squad = squad_info.get("players", [])
                except Exception as e:
                    print(f"Could not fetch current squad from API-Football for team {team_id}: {e}")
            
            # Get player season stats (try fresh fetch first, then cache)
            season_stats = []
            try:
                season_stats = load_or_fetch_player_stats(client, team_id, season_id, cache_only=False)
            except:
                # Try cached stats
                try:
                    season_stats = load_or_fetch_player_stats(client, team_id, season_id, cache_only=True)
                except:
                    pass
            
            if not season_stats:
                continue
            
            # Create mapping of current squad players (if available)
            current_players = {}
            if current_squad:
                # Map by player name since APIs might use different IDs
                for player in current_squad:
                    player_name = player.get("name", "").lower()
                    if player_name:
                        current_players[player_name] = player
            
            # Extract scoring stats and calculate enhanced scoring probability
            scorers = []
            
            # Process season stats
            for player in season_stats:
                meta = player.get("meta_data", {})
                stats = player.get("stats", {}).get("stats", {})
                
                player_name = meta.get("player_name", "Unknown")
                player_name_lower = player_name.lower()
                
                # If we have current squad data, check if player is still in squad
                if current_squad:
                    # Try exact match first, then fuzzy match
                    is_in_squad = (
                        player_name_lower in current_players or 
                        any(player_name_lower in cp_name or cp_name in player_name_lower 
                            for cp_name in current_players.keys())
                    )
                    if not is_in_squad:
                        continue  # Player transferred out
                
                position = stats.get("positions", "")
                goals = stats.get("gls", 0) or 0
                minutes = stats.get("min", 0) or 0
                matches = stats.get("matches_played", 0) or 0
                xg = stats.get("xg", 0) or 0
                assists = stats.get("ast", 0) or 0
                
                if matches == 0:
                    continue
                
                # Basic scoring rates
                goals_per_game = goals / matches if matches > 0 else 0
                goals_per_90 = (goals * 90) / minutes if minutes > 0 else 0
                xg_per_game = xg / matches if matches > 0 else 0
                
                # Enhanced probability calculation with H2H and form factors
                base_prob = goals_per_game * 0.4 + xg_per_game * 0.3 + goals_per_90 * 0.3
                
                # Position multipliers (enhanced)
                position_multiplier = 1.0
                position_upper = position.upper()
                if any(pos in position_upper for pos in ["FW", "CF", "ST"]):
                    position_multiplier = 1.3
                elif any(pos in position_upper for pos in ["AM", "CAM", "LW", "RW"]):
                    position_multiplier = 1.1
                elif any(pos in position_upper for pos in ["MF", "CM", "DM"]):
                    position_multiplier = 0.7
                elif any(pos in position_upper for pos in ["DF", "CB", "LB", "RB"]):
                    position_multiplier = 0.2
                elif "GK" in position_upper:
                    position_multiplier = 0.05
                
                base_prob *= position_multiplier
                
                # Form factor (if available)
                form_multiplier = 1.0
                if team_key == "home" and home_form:
                    # Boost for good home form
                    if home_form.get('ppg', 0) > 2.0:
                        form_multiplier = 1.15
                    elif home_form.get('ppg', 0) < 1.0:
                        form_multiplier = 0.9
                elif team_key == "away" and away_form:
                    # Boost for good away form
                    if away_form.get('ppg', 0) > 1.5:
                        form_multiplier = 1.1
                    elif away_form.get('ppg', 0) < 0.8:
                        form_multiplier = 0.85
                
                base_prob *= form_multiplier
                
                # H2H factor (boost for players who score in this fixture historically)
                h2h_multiplier = 1.0
                if h2h_data and h2h_data.get('meetings', 0) > 0:
                    # If team has good scoring record against opponent
                    avg_goals = h2h_data.get('gf', 0) / h2h_data.get('meetings', 1)
                    if avg_goals > 1.5:
                        h2h_multiplier = 1.1
                    elif avg_goals < 0.8:
                        h2h_multiplier = 0.95
                
                base_prob *= h2h_multiplier
                
                # Minutes played factor (higher for regular starters)
                if minutes > 0 and matches > 0:
                    minutes_factor = min(minutes / (matches * 90), 1.0)
                    base_prob *= (0.7 + 0.3 * minutes_factor)
                
                # Cap probability at reasonable levels
                scoring_prob = min(base_prob, 0.65)
                
                if scoring_prob > 0.02:  # Only include players with meaningful probability
                    # Check if it's a current squad member
                    squad_status = player_name_lower in current_players if current_squad else True
                    
                    scorers.append({
                        "name": player_name,
                        "position": position,
                        "goals": goals,
                        "matches": matches,
                        "goals_per_game": round(goals_per_game, 2),
                        "xg": round(xg, 1),
                        "assists": assists,
                        "scoring_probability": round(scoring_prob, 3),
                        "form_factor": round(form_multiplier, 2),
                        "is_current_squad": squad_status
                    })
            
            # Sort by scoring probability and take top 3
            scorers.sort(key=lambda x: x["scoring_probability"], reverse=True)
            result[team_key] = scorers[:3]
            
        except Exception as e:
            # Fallback to basic stats
            try:
                players = load_or_fetch_player_stats(client, team_id, season_id, cache_only=True)
                if players:
                    basic_scorers = []
                    for player in players[:10]:  # Top 10 basic to find best 3
                        meta = player.get("meta_data", {})
                        stats = player.get("stats", {}).get("stats", {})
                        goals = stats.get("gls", 0) or 0
                        matches = stats.get("matches_played", 0) or 0
                        if matches > 0:
                            basic_scorers.append({
                                "name": meta.get("player_name", "Unknown"),
                                "position": stats.get("positions", ""),
                                "goals": goals,
                                "matches": matches,
                                "goals_per_game": round(goals/matches, 2),
                                "xg": round(stats.get("xg", 0) or 0, 1),
                                "assists": stats.get("ast", 0) or 0,
                                "scoring_probability": round(min(goals/matches * 0.5, 0.4), 3),
                                "form_factor": 1.0,
                                "is_current_squad": True
                            })
                    basic_scorers.sort(key=lambda x: x["scoring_probability"], reverse=True)
                    result[team_key] = basic_scorers[:3]
            except Exception as fallback_e:
                pass
                
    return result


def predict_exact_score_poisson(home_goals_avg: float, away_goals_avg: float) -> Tuple[int, int]:
    """Predict exact scoreline using Poisson distribution."""
    import math
    
    def poisson_prob(k: int, lam: float) -> float:
        """Calculate Poisson probability P(X = k) given rate λ."""
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    
    # Find most likely scoreline combinations (0-5 goals each team)
    max_prob = 0
    best_home = 1
    best_away = 1
    
    for home_goals in range(6):
        for away_goals in range(6):
            prob = poisson_prob(home_goals, home_goals_avg) * poisson_prob(away_goals, away_goals_avg)
            if prob > max_prob:
                max_prob = prob
                best_home = home_goals
                best_away = away_goals
    
    return best_home, best_away


def summarize_h2h(df: pd.DataFrame, home_team_id: str, away_team_id: str, seasons_window: int = 5) -> Dict[str, object]:
    # Last N seasons meetings between specific teams
    df2 = df.copy()
    # Filter last N seasons by season_id sort
    seasons_sorted = sorted([s for s in df2["season_id"].dropna().unique()], key=lambda t: int(str(t).split("-")[0]))
    lastN = set(seasons_sorted[-seasons_window:])
    df2 = df2[df2["season_id"].isin(lastN)]
    mask = ((df2["home_team_id"] == home_team_id) & (df2["away_team_id"] == away_team_id)) | (
        (df2["home_team_id"] == away_team_id) & (df2["away_team_id"] == home_team_id)
    )
    h2h = df2[mask].dropna(subset=["home_team_score", "away_team_score"]).sort_values("date")
    w = d = l = gf = ga = 0
    for _, r in h2h.iterrows():
        if r["home_team_id"] == home_team_id:
            hs, as_ = r["home_team_score"], r["away_team_score"]
        else:
            hs, as_ = r["away_team_score"], r["home_team_score"]
        gf += int(hs)
        ga += int(as_)
        if hs > as_:
            w += 1
        elif hs < as_:
            l += 1
        else:
            d += 1
    return {"meetings": int(len(h2h)), "record": f"{w}-{d}-{l}", "gf": int(gf), "ga": int(ga)}


def summarize_recent_form(df: pd.DataFrame, team_id: str, last_n: int = 5) -> Dict[str, object]:
    # Take most recent completed matches for team (home or away)
    team_rows = df[((df["home_team_id"] == team_id) | (df["away_team_id"] == team_id))].dropna(
        subset=["home_team_score", "away_team_score"]
    ).sort_values("date").tail(last_n)
    if team_rows.empty:
        return {"ppg": None, "gd_avg": None, "gf_avg": None, "ga_avg": None}
    pts = []
    gds = []
    gfs = []
    gas = []
    for _, r in team_rows.iterrows():
        if r["home_team_id"] == team_id:
            gf, ga = r["home_team_score"], r["away_team_score"]
        else:
            gf, ga = r["away_team_score"], r["home_team_score"]
        gfs.append(int(gf))
        gas.append(int(ga))
        gds.append(int(gf) - int(ga))
        pts.append(3 if gf > ga else (1 if gf == ga else 0))
    n = len(pts)
    return {
        "ppg": round(sum(pts) / n, 2),
        "gd_avg": round(sum(gds) / n, 2),
        "gf_avg": round(sum(gfs) / n, 2),
        "ga_avg": round(sum(gas) / n, 2),
    }


def summarize_home_away(df: pd.DataFrame, home_team_id: str, away_team_id: str, last_n: int = 5) -> Dict[str, object]:
    home_recent = df[(df["home_team_id"] == home_team_id)].dropna(subset=["home_team_score", "away_team_score"]).sort_values("date").tail(last_n)
    away_recent = df[(df["away_team_id"] == away_team_id)].dropna(subset=["home_team_score", "away_team_score"]).sort_values("date").tail(last_n)
    def ppg(rows, is_home: bool):
        pts = []
        for _, r in rows.iterrows():
            gf = r["home_team_score"] if is_home else r["away_team_score"]
            ga = r["away_team_score"] if is_home else r["home_team_score"]
            pts.append(3 if gf > ga else (1 if gf == ga else 0))
        return round(sum(pts)/len(pts), 2) if len(pts)>0 else None
    return {
        "home_ppg_last5_home": ppg(home_recent, True),
        "away_ppg_last5_away": ppg(away_recent, False),
    }


def summarize_venue_specific(df: pd.DataFrame, home_team_id: str, away_team_id: str, seasons_window: int = 5) -> Dict[str, object]:
    # Away team record at this specific opponent's home over last N seasons
    df2 = df.copy()
    seasons_sorted = sorted([s for s in df2["season_id"].dropna().unique()], key=lambda t: int(str(t).split("-")[0]))
    lastN = set(seasons_sorted[-seasons_window:])
    rows = df2[(df2["home_team_id"] == home_team_id) & (df2["away_team_id"] == away_team_id) & df2["season_id"].isin(lastN)]
    rows = rows.dropna(subset=["home_team_score", "away_team_score"]) 
    w = d = l = 0
    for _, r in rows.iterrows():
        hs, as_ = r["home_team_score"], r["away_team_score"]
        if hs > as_:
            l += 1  # from away perspective
        elif hs < as_:
            w += 1
        else:
            d += 1
    return {"away_at_venue": f"{w}-{d}-{l}", "meetings": int(len(rows))}


def estimate_expected_score(home_form: Dict[str, object], away_form: Dict[str, object]) -> Tuple[float, float]:
    """Estimate expected score using recent GF/GA averages.
    home_x = mean(home GF avg, away GA avg); away_x = mean(away GF avg, home GA avg). Fallback to league avg ~1.35.
    """
    def mean2(a, b):
        xs = [x for x in [a, b] if isinstance(x, (int, float))]
        if not xs:
            return None
        return sum(xs)/len(xs)
    home_x = mean2(home_form.get("gf_avg"), away_form.get("ga_avg"))
    away_x = mean2(away_form.get("gf_avg"), home_form.get("ga_avg"))
    if home_x is None:
        home_x = 1.35
    if away_x is None:
        away_x = 1.35
    return round(float(home_x), 1), round(float(away_x), 1)


def prepare_dataset(client: FBRClient, seasons: List[str]) -> pd.DataFrame:
    all_rows: Dict[str, List[Dict]] = {}
    for s in seasons:
        try:
            rows = load_or_fetch_matches(client, LEAGUE_ID_EPL, s)
        except Exception as e:
            print(f"Warning: skipping season {s} due to error: {e}")
            rows = []
        if rows:
            all_rows[s] = rows
    if not all_rows:
        raise RuntimeError("No seasons with matches available. Ensure schedule caches exist or try later.")
    df = build_matches_dataframe(all_rows)
    # Feature engineering chain
    df = compute_elo_features(df)
    df = compute_form_features(df)
    df = compute_head_to_head_features(df)
    df = compute_advanced_features(df, client)
    return df


def prepare_dataset_with_fresh_current(client: FBRClient, seasons: List[str]) -> pd.DataFrame:
    """Prepare dataset ensuring current season fixtures are freshly fetched (not only from cache)."""
    if not seasons:
        return prepare_dataset(client, seasons)
    current = seasons[-1]
    all_rows: Dict[str, List[Dict]] = {}
    for s in seasons[:-1]:
        rows = load_or_fetch_matches(client, LEAGUE_ID_EPL, s)
        all_rows[s] = rows
    # Fresh current: combine past (with scores) from schedules and upcoming from league matches
    past_scored = build_season_matches_with_scores(client, LEAGUE_ID_EPL, current)
    try:
        league_matches = client.matches_by_league_season(LEAGUE_ID_EPL, current)
    except Exception:
        league_matches = []
    existing_ids = {r.get("match_id") for r in past_scored}
    combined = list(past_scored)
    for m in league_matches:
        mid = m.get("match_id")
        if mid in existing_ids:
            continue
        combined.append(m)
    p = cache_path_for_matches(LEAGUE_ID_EPL, current)
    _write_json_gz(p, combined)
    all_rows[current] = combined
    df = build_matches_dataframe(all_rows)
    df = compute_elo_features(df)
    df = compute_form_features(df)
    df = compute_head_to_head_features(df)
    df = compute_advanced_features(df, client)
    return df


# --------------------------- Model ---------------------------
def train_model(df: pd.DataFrame, save_path: Path) -> Dict:
    X, y = build_feature_matrix(df)

    # Do not drop rows with missing features; the pipeline imputes them
    if X.shape[0] == 0:
        raise RuntimeError("No labeled samples with complete features. Sync more past seasons or wait for results.")

    # Ensure at least two classes exist for multinomial LR
    if y.nunique() < 2:
        raise RuntimeError("Training data has fewer than 2 outcome classes. Need more historical completed matches.")

    # Time-aware split: use last season as validation if present
    seasons_series = df.loc[X.index, "season_id"]
    unique_seasons = [s for s in seasons_series.dropna().unique()]
    unique_seasons_sorted = sorted(unique_seasons, key=lambda t: int(str(t).split("-")[0]))
    X_train = X_val = y_train = y_val = None
    if len(unique_seasons_sorted) >= 2:
        val_season = unique_seasons_sorted[-1]
        train_idx = X.index[seasons_series != val_season]
        val_idx = X.index[seasons_series == val_season]
        if len(train_idx) > 0 and len(val_idx) > 0:
            X_train, X_val = X.loc[train_idx], X.loc[val_idx]
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]
    if X_train is None:
        # Fallback to time-based split without season boundary
        if X.shape[0] < 2:
            raise RuntimeError("Not enough samples to split train/validation. Need at least 2.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    numeric_features = list(X.columns)
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
    ])

    # Try Random Forest for better performance
    from sklearn.ensemble import RandomForestClassifier
    
    # First try Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_pipe = Pipeline([
        ("pre", pre),
        ("clf", rf_clf),
    ])
    
    # Also try Logistic Regression for comparison
    lr_clf = LogisticRegression(
        max_iter=400,
        multi_class="multinomial",
        solver="saga",
        random_state=42,
        C=1.0
    )
    
    lr_pipe = Pipeline([
        ("pre", pre),
        ("clf", lr_clf),
    ])
    
    # Train both models
    rf_pipe.fit(X_train, y_train)
    lr_pipe.fit(X_train, y_train)
    
    # Evaluate both
    rf_pred = rf_pipe.predict(X_val)
    rf_proba = rf_pipe.predict_proba(X_val)
    rf_acc = accuracy_score(y_val, rf_pred)
    rf_ll = log_loss(y_val, rf_proba, labels=rf_pipe.named_steps["clf"].classes_)
    
    lr_pred = lr_pipe.predict(X_val)
    lr_proba = lr_pipe.predict_proba(X_val)
    lr_acc = accuracy_score(y_val, lr_pred)
    lr_ll = log_loss(y_val, lr_proba, labels=lr_pipe.named_steps["clf"].classes_)
    
    # Choose the better model
    if rf_acc > lr_acc:
        best_pipe = rf_pipe
        best_acc = rf_acc
        best_ll = rf_ll
        model_type = "RandomForest"
    else:
        best_pipe = lr_pipe
        best_acc = lr_acc
        best_ll = lr_ll
        model_type = "LogisticRegression"

    joblib.dump({
        "model": best_pipe,
        "feature_cols": numeric_features,
        "classes_": best_pipe.named_steps["clf"].classes_.tolist(),
        "model_type": model_type,
    }, save_path)

    return {"accuracy": best_acc, "log_loss": best_ll, "val_samples": int(len(y_val)), "model_type": model_type}
class ModelLoadError(Exception):
    pass


def load_model(path: Path):
    try:
        bundle = joblib.load(path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from {path}: {e}")
    # Handle both old and new model format
    model_type = bundle.get("model_type", "LogisticRegression")
    return bundle["model"], bundle["feature_cols"], bundle["classes_"], model_type


def predict_for_fixtures(model_bundle_path: Path, df_all: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    model, feature_cols, classes_, model_type = load_model(model_bundle_path)

    # Upcoming fixtures: rows with no scores AND future dates
    upcoming = df_all[df_all[["home_team_score", "away_team_score"]].isna().all(axis=1)].copy()
    if upcoming.empty:
        return upcoming
    
    # Filter for future dates only (from today onwards)
    today = datetime.now(UTC).date()
    upcoming["date"] = pd.to_datetime(upcoming["date"]).dt.date
    upcoming = upcoming[upcoming["date"] >= today].copy()
    
    if upcoming.empty:
        return upcoming

    # Build features on the full df (already computed), then select upcoming rows
    X = upcoming[feature_cols]
    # Fill NaNs with global means for prediction time
    X = X.fillna(X.mean(numeric_only=True))
    proba = model.predict_proba(X)
    proba_df = pd.DataFrame(proba, columns=[f"P({c})" for c in classes_], index=upcoming.index)
    out = pd.concat([upcoming[["date", "home", "away", "season_id"]], proba_df], axis=1).reset_index(drop=True)
    out = out.sort_values("date").head(top)
    return out


def predict_for_rows(df_rows: pd.DataFrame, model_bundle_path: Path) -> pd.DataFrame:
    """Return prediction probabilities for provided df rows (assumes features already engineered)."""
    model, feature_cols, classes_, model_type = load_model(model_bundle_path)
    if df_rows.empty:
        return df_rows
    X = df_rows[feature_cols].copy()
    X = X.fillna(X.mean(numeric_only=True))
    proba = model.predict_proba(X)
    proba_df = pd.DataFrame(proba, columns=[f"P({c})" for c in classes_], index=df_rows.index)
    out = pd.concat([df_rows[["date", "home", "away", "season_id", "wk"]], proba_df], axis=1)
    return out.reset_index(drop=True)


def predict_exact_score_enhanced(home_prob: float, draw_prob: float, away_prob: float) -> Tuple[int, int]:
    """Predict exact scoreline that aligns with win probabilities using enhanced Poisson logic."""
    # Estimate goals based on probabilities
    # Higher home prob = higher home goals, higher away prob = higher away goals
    
    # Base goals estimate (typical EPL average is ~2.7 goals per game)
    base_total_goals = 2.7
    
    # Adjust based on probabilities
    # If home advantage is strong, home team scores more
    home_advantage = home_prob - 0.33  # 0.33 is neutral probability
    away_advantage = away_prob - 0.33
    
    # Estimate lambda values for Poisson distribution
    home_lambda = max(0.3, base_total_goals * 0.55 + home_advantage * 2.0)
    away_lambda = max(0.3, base_total_goals * 0.45 + away_advantage * 2.0)
    
    # Sample from Poisson distributions (rounded to most likely values)
    home_goals = round(home_lambda)
    away_goals = round(away_lambda)
    
    # Ensure the result matches the most likely outcome
    if home_prob > max(draw_prob, away_prob):
        # Home win expected
        if home_goals <= away_goals:
            home_goals = away_goals + 1
    elif away_prob > max(home_prob, draw_prob):
        # Away win expected  
        if away_goals <= home_goals:
            away_goals = home_goals + 1
    else:
        # Draw expected
        if home_goals != away_goals:
            # Adjust to a draw - prefer the closer adjustment
            if abs(home_goals - away_goals) == 1:
                if home_goals > away_goals:
                    away_goals = home_goals
                else:
                    home_goals = away_goals
            else:
                # Default to 1-1 or 2-2 based on goal expectation
                avg_goals = round((home_goals + away_goals) / 2)
                home_goals = away_goals = max(1, avg_goals)
    
    return home_goals, away_goals


def predict_single_match(model_bundle_path: Path, df_all: pd.DataFrame, home_name: str, away_name: str) -> Optional[pd.Series]:
    model, feature_cols, classes_, model_type = load_model(model_bundle_path)
    
    # First, look for upcoming matches between these teams (no scores yet, future dates)
    current_date = datetime.now(UTC).date()
    current_timestamp = pd.Timestamp(current_date)
    
    upcoming_matches = df_all[
        (df_all["home"].str.lower() == home_name.lower()) & 
        (df_all["away"].str.lower() == away_name.lower()) &
        (df_all[["home_team_score", "away_team_score"]].isna().all(axis=1))
    ].copy()
    
    # Filter for future dates if date column exists
    if not upcoming_matches.empty and "date" in upcoming_matches.columns:
        upcoming_matches.loc[:, "date"] = pd.to_datetime(upcoming_matches["date"])
        future_matches = upcoming_matches[upcoming_matches["date"] >= current_timestamp]
        if not future_matches.empty:
            # Use the earliest upcoming match
            cand = future_matches.sort_values("date").head(1)
        else:
            cand = upcoming_matches.head(1)  # Use any upcoming match even if date is past
    else:
        cand = upcoming_matches.head(1) if not upcoming_matches.empty else pd.DataFrame()
    
    # If no upcoming matches found, look for any historical match to get team IDs
    if cand.empty:
        cand = df_all[(df_all["home"].str.lower() == home_name.lower()) & (df_all["away"].str.lower() == away_name.lower())]
    
    if cand.empty:
        # Construct a synthetic row using latest date in df_all to fetch team ids by name if possible
        # Try to infer team ids by most recent occurrence of those names
        home_row = df_all[df_all["home"].str.lower() == home_name.lower()].dropna(subset=["home_team_id"]).tail(1)
        away_row = df_all[df_all["away"].str.lower() == away_name.lower()].dropna(subset=["away_team_id"]).tail(1)
        if home_row.empty or away_row.empty:
            return None
        # Create synthetic fixture on current date + 7 days
        future_date = current_timestamp + pd.Timedelta(days=7)
        synth = {
            "date": future_date,
            "season_id": df_all["season_id"].dropna().max(),
            "home": home_name,
            "home_team_id": home_row.iloc[0]["home_team_id"],
            "away": away_name,
            "away_team_id": away_row.iloc[0]["away_team_id"],
            "home_team_score": np.nan,
            "away_team_score": np.nan,
        }
        cand = pd.DataFrame([synth])
    
    row = cand.iloc[0].copy()
    
    # If this is a historical match, update the date to show it as upcoming
    if pd.notna(row.get("home_team_score")) or pd.notna(row.get("away_team_score")):
        # This is a completed match, so create a future version
        row["date"] = current_timestamp + pd.Timedelta(days=7)  # Next week
        row["home_team_score"] = np.nan
        row["away_team_score"] = np.nan
        # Convert to DataFrame for consistency
        cand = pd.DataFrame([row])
    # Build features for this single row by adding it to df_all temporarily
    temp_df = pd.concat([df_all, cand], ignore_index=True)
    temp_df = compute_elo_features(temp_df)
    temp_df = compute_form_features(temp_df)
    temp_df = compute_head_to_head_features(temp_df)
    # We can't compute advanced features for a single synthetic row without client access
    # So we'll use the existing feature values if available, or fill with means
    
    # Get the last row (our target match)
    target_row = temp_df.iloc[-1]
    
    # Prepare features - ensure all values are scalar
    X_values = []
    for col in feature_cols:
        if col in target_row.index:
            val = target_row[col]
            # Ensure scalar value
            if hasattr(val, '__iter__') and not isinstance(val, str):
                val = val.iloc[0] if hasattr(val, 'iloc') and len(val) > 0 else 0.0
            # Check if the scalar value is not null
            if not pd.isna(val):
                X_values.append(float(val))
            else:
                # Use mean from the full dataset for missing features
                if col in temp_df.columns:
                    mean_val = temp_df[col].mean()
                    if hasattr(mean_val, '__iter__') and not isinstance(mean_val, str):
                        mean_val = mean_val.iloc[0] if hasattr(mean_val, 'iloc') and len(mean_val) > 0 else 0.0
                    X_values.append(float(mean_val) if not pd.isna(mean_val) else 0.0)
                else:
                    X_values.append(0.0)
        else:
            # Use mean from the full dataset for missing features
            if col in temp_df.columns:
                mean_val = temp_df[col].mean()
                if hasattr(mean_val, '__iter__') and not isinstance(mean_val, str):
                    mean_val = mean_val.iloc[0] if hasattr(mean_val, 'iloc') and len(mean_val) > 0 else 0.0
                X_values.append(float(mean_val) if not pd.isna(mean_val) else 0.0)
            else:
                X_values.append(0.0)
    
    X = np.array(X_values).reshape(1, -1)
    
    # Convert to DataFrame with proper column names for the model
    X_df = pd.DataFrame(X, columns=feature_cols)
    
    # Predict
    proba = model.predict_proba(X_df)[0]
    result = pd.Series(dict(zip([f"P({c})" for c in classes_], proba)))
    result["home"] = row["home"]
    result["away"] = row["away"]
    result["date"] = row.get("date")
    result["season_id"] = row.get("season_id")
    return result


def get_team_recent_form(client: FBRClient, team_id: str, season_id: str) -> str:
    """Get team's recent form from last 5 competitive matches."""
    try:
        # Try to get recent matches from API
        schedule = client.team_schedule(team_id, season_id)
        
        # Filter for completed matches (with scores) and sort by date
        completed = []
        for match in schedule:
            if (match.get('home_team_score') is not None and 
                match.get('away_team_score') is not None):
                completed.append(match)
        
        # Sort by date (most recent first) and take last 5
        completed.sort(key=lambda x: x.get('date', ''), reverse=True)
        recent_5 = completed[:5]
        
        if not recent_5:
            return "No recent data"
        
        # Calculate form
        form_string = ""
        points = 0
        
        for match in reversed(recent_5):  # Show oldest to newest
            is_home = match.get('home_away', '').lower() == 'home'
            home_score = int(match.get('home_team_score', 0))
            away_score = int(match.get('away_team_score', 0))
            
            if is_home:
                team_score, opp_score = home_score, away_score
            else:
                team_score, opp_score = away_score, home_score
            
            if team_score > opp_score:
                form_string += "W"
                points += 3
            elif team_score == opp_score:
                form_string += "D"
                points += 1
            else:
                form_string += "L"
        
        avg_points = points / len(recent_5)
        return f"{form_string} ({points}/{len(recent_5)*3} pts, {avg_points:.1f} avg)"
        
    except Exception as e:
        return f"Form unavailable"



    model, feature_cols, classes_, model_type = load_model(model_bundle_path)
    # Find the most recent scheduled or upcoming meeting in df_all for these names
    cand = df_all[(df_all["home"].str.lower() == home_name.lower()) & (df_all["away"].str.lower() == away_name.lower())]
    if cand.empty:
        # Construct a synthetic row using latest date in df_all to fetch team ids by name if possible
        # Try to infer team ids by most recent occurrence of those names
        home_row = df_all[df_all["home"].str.lower() == home_name.lower()].dropna(subset=["home_team_id"]).tail(1)
        away_row = df_all[df_all["away"].str.lower() == away_name.lower()].dropna(subset=["away_team_id"]).tail(1)
        if home_row.empty or away_row.empty:
            return None
        # Create synthetic fixture on max date + 1 day
        future_date = (pd.to_datetime(df_all["date"].dropna()).max() + pd.Timedelta(days=1)) if df_all["date"].notna().any() else pd.Timestamp(datetime.now(UTC).date())
        synth = {
            "date": future_date,
            "season_id": df_all["season_id"].dropna().max(),
            "home": home_name,
            "home_team_id": home_row.iloc[0]["home_team_id"],
            "away": away_name,
            "away_team_id": away_row.iloc[0]["away_team_id"],
        }
        # Recompute features would be expensive; instead, find nearest existing row for these teams and reuse features as proxy
        # Prefer latest home match for home team and away match for away team
        latest_home = df_all[df_all["home_team_id"] == synth["home_team_id"]].dropna(subset=model.feature_names_in_, how="any").tail(1)
        latest_away = df_all[df_all["away_team_id"] == synth["away_team_id"]].dropna(subset=model.feature_names_in_, how="any").tail(1)
        if latest_home.empty or latest_away.empty:
            return None
        # Take feature row from the most recent actual home vs away match for those teams separately, then average numeric features
        xh = latest_home[feature_cols]
        xa = latest_away[feature_cols]
        X = (xh.values + xa.values) / 2.0
        proba = model.predict_proba(X)[0]
        return pd.Series({"home": home_name, "away": away_name, **{f"P({c})": p for c, p in zip(classes_, proba)}})
    else:
        # Use the first upcoming match if several
        row = cand.iloc[0]
        X = row[feature_cols].to_frame().T
        X = X.fillna(X.mean(numeric_only=True))
        proba = model.predict_proba(X)[0]
        return pd.Series({"home": row["home"], "away": row["away"], **{f"P({c})": p for c, p in zip(classes_, proba)}})


# --------------------------- CLI ---------------------------
def cmd_generate_key(args):
    client = FBRClient()
    key = client.generate_api_key()
    print("API Key:", key)
    print("Tip: export FBR_API_KEY=", key)


def cmd_sync(args):
    client = FBRClient()
    if not client.api_key:
        print("Error: FBR_API_KEY not set. Set env var or run 'generate-key' first.")
        sys.exit(1)
    seasons = get_last_n_seasons(client, LEAGUE_ID_EPL, n=args.seasons)
    print("Seasons:", ", ".join(seasons))
    for s in seasons:
        p = cache_path_for_matches(LEAGUE_ID_EPL, s)
        if p.exists() and not args.force:
            print(f"Cached: {s} ({p.name})")
            continue
        print(f"Fetching matches for {s}...")
        _ = load_or_fetch_matches(client, LEAGUE_ID_EPL, s)
        print(f"Saved: {p}")


def cmd_sync_enhanced(args):
    """Sync team stats and player stats for enhanced predictions."""
    client = FBRClient()
    if not client.api_key:
        print("Error: FBR_API_KEY not set. Set env var or run 'generate-key' first.")
        sys.exit(1)
        
    seasons: List[str]
    if args.season:
        seasons = [args.season]
    else:
        seasons = get_last_n_seasons(client, LEAGUE_ID_EPL, n=args.seasons)
        
    print("Enhanced data sync seasons:", ", ".join(seasons))
    
    for s in seasons:
        print(f"\n=== Season {s} ===")
        
        # Sync team stats
        try:
            team_stats_path = cache_path_for_team_stats(LEAGUE_ID_EPL, s)
            if team_stats_path.exists() and not args.force:
                print(f"Team stats cached: {s}")
            else:
                print(f"Fetching team stats for {s}...")
                stats = load_or_fetch_team_stats(client, LEAGUE_ID_EPL, s)
                print(f"Saved team stats: {len(stats)} teams")
        except Exception as e:
            print(f"Failed to sync team stats for {s}: {e}")
        
        # Sync player stats for each team
        try:
            team_map = get_season_teams_from_matches(client, LEAGUE_ID_EPL, s)
            print(f"Syncing player stats for {len(team_map)} teams...")
            
            for tid, tname in sorted(team_map.items(), key=lambda kv: kv[1] or ''):
                player_stats_path = cache_path_for_player_stats(tid, s)
                if player_stats_path.exists() and not args.force:
                    print(f"  cached: {tname}")
                    continue
                    
                print(f"  fetching: {tname}...")
                try:
                    players = load_or_fetch_player_stats(client, tid, s)
                    print(f"    saved: {len(players)} players")
                except Exception as e:
                    print(f"    failed: {e}")
                    
        except Exception as e:
            print(f"Failed to sync player stats for {s}: {e}")


def cmd_sync_schedules(args):
    client = FBRClient()
    if not client.api_key:
        print("Error: FBR_API_KEY not set. Set env var or run 'generate-key' first.")
        sys.exit(1)
    seasons: List[str]
    if args.season:
        seasons = [args.season]
    else:
        seasons = get_last_n_seasons(client, LEAGUE_ID_EPL, n=args.seasons)
    print("Schedule sync seasons:", ", ".join(seasons))
    for s in seasons:
        try:
            team_map = get_season_teams_from_matches(client, LEAGUE_ID_EPL, s)
        except Exception as e:
            print(f"Failed to list teams for {s}: {e}")
            continue
        print(f"{s}: {len(team_map)} teams")
        for tid, tname in sorted(team_map.items(), key=lambda kv: kv[1] or ''):
            p = cache_path_for_team_schedule(tid, s)
            if p.exists() and not args.force:
                print(f"cached: {tname} ({tid})")
                continue
            print(f"fetching: {tname} ({tid}) @ {s} ...")
            try:
                _ = load_or_fetch_team_schedule(client, tid, s)
                print(f"saved: {p.name}")
            except Exception as e:
                print(f"warn: failed {tname} {s}: {e}")


def cmd_train(args):
    client = FBRClient()
    if not client.api_key:
        print("Warning: FBR_API_KEY not set. Using only cached data if available.")
    # Try to infer seasons from cache; if empty, fetch last 5 seasons (requires key)
    cached = sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json.gz")) or sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json"))
    seasons = [extract_season_id_from_cache_path(c) for c in cached]
    if not seasons:
        if not client.api_key:
            print("Error: No cached data and no API key. Run 'sync' with an API key first.")
            sys.exit(1)
        seasons = get_last_n_seasons(client, LEAGUE_ID_EPL, n=5)
        for s in seasons:
            load_or_fetch_matches(client, LEAGUE_ID_EPL, s)
    print("Preparing dataset...")
    df = prepare_dataset(client, seasons)
    print(f"Total matches: {len(df)}")
    model_path = MODELS_DIR / "epl_result_model.joblib"
    print("Training model...")
    metrics = train_model(df, model_path)
    print(f"Saved model to {model_path}")
    print(f"Validation accuracy: {metrics['accuracy']:.3f} | log_loss: {metrics['log_loss']:.3f} | n={metrics['val_samples']}")


def cmd_predict_fixtures(args):
    model_path = MODELS_DIR / "epl_result_model.joblib"
    if not model_path.exists():
        print("Error: Model not found. Train it first with 'train'.")
        sys.exit(1)
    client = FBRClient()
    # Validate model load; if incompatible/corrupted, retrain on cached seasons if possible
    try:
        _ = load_model(model_path)
    except ModelLoadError:
        print("Model file is incompatible. Retraining on cached data...")
        cached_for_train = sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json.gz")) or sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json"))
        if not cached_for_train:
            print("Error: No cached data to retrain. Run 'sync' first.")
            sys.exit(1)
        seasons_for_train = [c.stem.split("_")[-1] for c in cached_for_train]
        df_train = prepare_dataset(client, seasons_for_train)
        _ = train_model(df_train, model_path)
    # Build df from cached seasons (plus current season if available)
    cached = sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json.gz")) or sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json"))
    if not cached:
        print("Error: No cached data. Run 'sync' first.")
        sys.exit(1)
    seasons = [extract_season_id_from_cache_path(c) for c in cached]
    df = prepare_dataset(client, seasons)
    preds = predict_for_fixtures(model_path, df, top=args.top)
    if preds.empty:
        print("No upcoming fixtures found in cached data.")
        return

    # Enhanced output with detailed match predictions
    print(f"\n🔮 UPCOMING EPL FIXTURES PREDICTIONS (Top {args.top})")
    print("=" * 80)
    
    for _, r in preds.iterrows():
        probs = {k: r[k] for k in [c for c in preds.columns if c.startswith("P(")]} 
        best = max(probs.items(), key=lambda kv: kv[1])
        date_str = r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "TBD"
        
        # Get team IDs for recent form
        home_team_id = None
        away_team_id = None
        try:
            match_row = df[(df['home'] == r['home']) & (df['away'] == r['away']) & (df['season_id'] == r['season_id'])].iloc[0]
            home_team_id = match_row['home_team_id']
            away_team_id = match_row['away_team_id']
        except:
            pass

        print(f"\n🏟️  {r['home']} vs {r['away']}")
        print("─" * 80)
        print(f"📅 Date: {date_str}")
        
        # Win probabilities
        print("\n📊 WIN PROBABILITIES")
        print("─" * 40)
        home_prob = probs.get('P(H)', 0) * 100
        draw_prob = probs.get('P(D)', 0) * 100
        away_prob = probs.get('P(A)', 0) * 100
        
        print(f"🏠 {r['home']:<20} {home_prob:5.1f}% {'█' * max(1, int(home_prob / 3))}")
        print(f"🤝 Draw{'':<16} {draw_prob:5.1f}% {'█' * max(1, int(draw_prob / 3))}")
        print(f"✈️  {r['away']:<20} {away_prob:5.1f}% {'█' * max(1, int(away_prob / 3))}")
        
        # Most likely outcome
        if best[0] == 'P(H)':
            outcome = f"🏠 {r['home']}"
        elif best[0] == 'P(A)':
            outcome = f"✈️ {r['away']}"
        else:
            outcome = "🤝 Draw"
        print(f"\n🎯 Most Likely: {outcome} ({best[1]:.1%})")
        
        # Predicted scoreline (improved logic to match probabilities)
        home_goals, away_goals = predict_exact_score_enhanced(home_prob/100, draw_prob/100, away_prob/100)
        print("\n" + "─" * 80)
        print("⚽ PREDICTED SCORELINE")
        print("─" * 40)
        print(f"🥅 Final Score: {r['home']} {home_goals} - {away_goals} {r['away']}")
        
        # Recent form (if team IDs available)
        if home_team_id and away_team_id:
            print("\n📈 RECENT FORM (Last 5 Games)")
            print("─" * 40)
            try:
                home_form = get_team_recent_form(client, home_team_id, r['season_id'])
                away_form = get_team_recent_form(client, away_team_id, r['season_id'])
                print(f"🏠 {r['home']}: {home_form}")
                print(f"✈️  {r['away']}: {away_form}")
            except:
                print("Recent form data unavailable")
        
        print("─" * 80)


def cmd_predict_match(args):
    model_path = MODELS_DIR / "epl_result_model.joblib"
    if not model_path.exists():
        print("Error: Model not found. Train it first with 'train'.")
        sys.exit(1)
    client = FBRClient()
    # Validate model load; if incompatible/corrupted, retrain on cached seasons if possible
    try:
        _ = load_model(model_path)
    except ModelLoadError:
        print("Model file is incompatible. Retraining on cached data...")
        cached_for_train = sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json.gz")) or sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json"))
        if not cached_for_train:
            print("Error: No cached data to retrain. Run 'sync' first.")
            sys.exit(1)
        seasons_for_train = [c.stem.split("_")[-1] for c in cached_for_train]
        df_train = prepare_dataset(client, seasons_for_train)
        _ = train_model(df_train, model_path)
    cached = sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json.gz")) or sorted(CACHE_MATCHES_DIR.glob("matches_league9_*.json"))
    if not cached:
        print("Error: No cached data. Run 'sync' first.")
        sys.exit(1)
    seasons = [extract_season_id_from_cache_path(c) for c in cached]
    df = prepare_dataset(client, seasons)
    res = predict_single_match(model_path, df, args.home, args.away)
    if res is None:
        print("Could not infer enough data for the requested teams from cache.")
        sys.exit(2)
    
    
    # Enhanced Beautiful Match Prediction Output
    home, away = res['home'], res['away']
    ph = res['P(H)']
    pd_ = res['P(D)']
    pa = res['P(A)']
    
    print("\n" + "="*80)
    print("⚽" + " EPL MATCH PREDICTION ".center(76) + "⚽")
    print("="*80)
    
    # Match Header
    match_date = res.get('date')
    date_str = ""
    if match_date and not pd.isna(match_date):
        if hasattr(match_date, 'strftime'):
            date_str = f"\n📅 Date: {match_date.strftime('%Y-%m-%d')}"
        else:
            date_str = f"\n📅 Date: {match_date}"
    
    print(f"\n🏟️  {home} vs {away}{date_str}")
    print("\n" + "─"*80)
    
    # Result Probabilities with visual bars
    print("📊 WIN PROBABILITIES")
    print("─"*40)
    home_bar = "█" * int(ph * 30)
    draw_bar = "█" * int(pd_ * 30)
    away_bar = "█" * int(pa * 30)
    
    print(f"🏠 {home:20} {ph*100:5.1f}% {home_bar}")
    print(f"🤝 Draw{'':<16} {pd_*100:5.1f}% {draw_bar}")
    print(f"✈️  {away:20} {pa*100:5.1f}% {away_bar}")
    
    # Most Likely Result
    if ph > pd_ and ph > pa:
        winner = f"🏠 {home}"
        best_prob = ph
    elif pd_ > ph and pd_ > pa:
        winner = "🤝 Draw"
        best_prob = pd_
    else:
        winner = f"✈️ {away}"
        best_prob = pa
        
    print(f"\n🎯 Most Likely: {winner} ({best_prob*100:.1f}%)")
    
    # Predicted scoreline (improved logic to match probabilities)
    home_goals, away_goals = predict_exact_score_enhanced(ph, pd_, pa)
    print("\n" + "─"*80)
    print("⚽ PREDICTED SCORELINE")
    print("─"*40)
    print(f"🥅 Final Score: {home} {home_goals} - {away_goals} {away}")
    
    # Get player predictions with enhanced features
    try:
        # Get current season and team IDs  
        current_season = get_current_season()
        # Try current season first, fallback to previous season for player data
        seasons_to_try = [current_season]
        year = int(current_season.split("-")[0])
        prev_season = f"{year-1}-{year}"
        if prev_season != current_season:
            seasons_to_try.append(prev_season)
        
        home_team_id = None
        away_team_id = None
        player_season = None
        
        for season in seasons_to_try:
            home_team_id = get_team_id_by_name(client, home, season)
            away_team_id = get_team_id_by_name(client, away, season)
            if home_team_id and away_team_id:
                player_season = season
                break
        
        # Get detailed match analysis from DataFrame
        home_form = None
        away_form = None
        h2h = None
        ha_form = None
        venue = None
        
        # Find team IDs in dataframe for enhanced analysis
        home_matches = df[df['home'] == home]
        away_matches = df[df['away'] == away]
        
        if not home_matches.empty and not away_matches.empty:
            h_id = home_matches.iloc[0]['home_team_id'] if 'home_team_id' in home_matches.columns else home_team_id
            a_id = away_matches.iloc[0]['away_team_id'] if 'away_team_id' in away_matches.columns else away_team_id
            
            if h_id and a_id:
                h2h = summarize_h2h(df, h_id, a_id)
                home_form = summarize_recent_form(df, h_id)
                away_form = summarize_recent_form(df, a_id)
                ha_form = summarize_home_away(df, h_id, a_id)
                venue = summarize_venue_specific(df, h_id, a_id)
        
        # Enhanced Match Analysis (if data available)
        if h2h and home_form and away_form and ha_form and venue:
            print("\n" + "─"*80)
            print("📈 MATCH ANALYSIS")
            print("─"*40)
            print(f"📊 H2H Record: {h2h['record']} (Goals: {h2h['gf']}-{h2h['ga']}) in {h2h['meetings']} games")
            print(f"🔥 Recent Form: {home} {home_form['ppg']:.1f}PPG ({home_form['gf_avg']:.1f}GF {home_form['ga_avg']:.1f}GA) | {away} {away_form['ppg']:.1f}PPG ({away_form['gf_avg']:.1f}GF {away_form['ga_avg']:.1f}GA)")
            print(f"🏠 Home/Away: {home} {ha_form['home_ppg_last5_home']:.1f}PPG at home | {away} {ha_form['away_ppg_last5_away']:.1f}PPG away")
            print(f"🏟️  Venue History: {venue['away_at_venue']} in {venue['meetings']} visits")
        
        # Get top scorers prediction with enhanced analysis
        if home_team_id and away_team_id and player_season:
            try:
                # Try multiple seasons for player stats (current, then previous)
                seasons_to_try = [player_season]
                if player_season == "2025-2026":
                    seasons_to_try = ["2024-2025", "2023-2024"]
                elif player_season == "2024-2025":
                    seasons_to_try = ["2024-2025", "2023-2024"]
                
                top_scorers = None
                for try_season in seasons_to_try:
                    try:
                        # Pass H2H and form data for enhanced scoring prediction
                        top_scorers = get_enhanced_top_scorers(client, home_team_id, away_team_id, try_season, 
                                                             h2h, home_form, away_form)
                        if top_scorers and (top_scorers.get('home') or top_scorers.get('away')):
                            break
                    except:
                        continue
                
                if top_scorers and (top_scorers.get('home') or top_scorers.get('away')):
                    print("\n" + "─"*80)
                    print("⭐ TOP EXPECTED SCORERS")
                    print("─"*40)
                    for team_key, team_name in [('home', home), ('away', away)]:
                        if top_scorers.get(team_key):
                            emoji = "🏠" if team_key == 'home' else "✈️"
                            print(f"{emoji} {team_name}:")
                            for i, scorer in enumerate(top_scorers[team_key], 1):
                                # Enhanced display with additional stats
                                prob_percent = scorer['scoring_probability'] * 100
                                if scorer['matches'] > 0:
                                    stats_text = f"({scorer['goals']}G/{scorer['matches']}M"
                                    if scorer.get('assists', 0) > 0:
                                        stats_text += f", {scorer['assists']}A"
                                    if scorer.get('xg', 0) > 0:
                                        stats_text += f", {scorer['xg']}xG"
                                    stats_text += ")"
                                else:
                                    stats_text = "(New signing)"
                                
                                # Add form indicator if available
                                form_indicator = ""
                                if scorer.get('form_factor', 1.0) > 1.05:
                                    form_indicator = " 🔥"
                                elif scorer.get('form_factor', 1.0) < 0.95:
                                    form_indicator = " ❄️"
                                
                                print(f"  {i}. {scorer['name']} ({scorer['position']}) - {prob_percent:.1f}% {stats_text}{form_indicator}")
                else:
                    print("\n" + "─"*80)
                    print("⭐ TOP EXPECTED SCORERS")
                    print("─"*40)
                    print("📊 Fetching current squad and player statistics...")
                    # Try to get basic stats even without cached data
                    try:
                        basic_scorers = get_enhanced_top_scorers(client, home_team_id, away_team_id, player_season)
                        if basic_scorers and (basic_scorers.get('home') or basic_scorers.get('away')):
                            for team_key, team_name in [('home', home), ('away', away)]:
                                if basic_scorers.get(team_key):
                                    emoji = "🏠" if team_key == 'home' else "✈️"
                                    print(f"{emoji} {team_name}:")
                                    for i, scorer in enumerate(basic_scorers[team_key], 1):
                                        prob_percent = scorer['scoring_probability'] * 100
                                        stats_text = f"({scorer['goals']}G/{scorer['matches']}M)" if scorer['matches'] > 0 else "(New)"
                                        print(f"  {i}. {scorer['name']} ({scorer['position']}) - {prob_percent:.1f}% {stats_text}")
                        else:
                            print("📊 Player statistics not available for current squads")
                    except Exception as e:
                        print("📊 Unable to fetch current player statistics")
            except Exception:
                print("\n" + "─"*80)
                print("⭐ TOP EXPECTED SCORERS")
                print("─"*40)
                print("📊 Player statistics not available (run 'sync-enhanced' for predictions)")
        else:
            print("\n" + "─"*80)
            print("⭐ TOP EXPECTED SCORERS")
            print("─"*40)
            print("📊 Team ID mapping not available for enhanced predictions")
        
    except Exception as e:
        print("\n" + "─"*80)
        print("⭐ TOP EXPECTED SCORERS")
        print("─"*40)
        print("📊 Enhanced features temporarily unavailable")
    
    print("\n" + "="*80)
def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_loading(message="Processing"):
    """Show a simple loading message"""
    print(f"⏳ {message}...")
    print("─" * 50)

def show_success(message="Operation completed"):
    """Show a success message"""
    print(f"✅ {message}")
    print("─" * 50)

def show_error(message="Operation failed"):
    """Show an error message"""
    print(f"❌ {message}")
    print("─" * 50)

def get_current_season():
    """Get current EPL season based on current date"""
    now = datetime.now()
    # EPL season typically starts in August and ends in May
    # If current month is June-July, we're in the off-season, use next season
    # If current month is August-May, we're in the current season
    
    if now.month >= 8:  # August onwards - current season
        start_year = now.year
        end_year = now.year + 1
    elif now.month <= 5:  # January to May - season started previous year
        start_year = now.year - 1
        end_year = now.year
    else:  # June-July - off season, prepare for next season
        start_year = now.year
        end_year = now.year + 1
    
    return f"{start_year}-{end_year}"

def display_team_selection():
    """Display all 20 Premier League teams in a 3-per-line grid (7 rows)"""
    # Current 2024-25 Premier League teams
    teams = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crys Palace", "Everton", "Fulham", "Ipswich",
        "Leicester", "Liverpool", "Man City", "Man Utd",
        "Newcastle", "Nott' Forest", "Southampton", "Tottenham",
        "West Ham", "Wolves"
    ]
    
    print("\n⚽ SELECT PREMIER LEAGUE TEAMS")
    print("=" * 55)
    
    # Display in 7 rows of 3 teams each (with last row having only 2 teams)
    print("┌" + "─" * 53 + "┐")
    
    for row in range(7):
        print("│", end="")
        teams_in_row = 3 if row < 6 else 2  # Last row has only 2 teams (19, 20)
        
        for col in range(teams_in_row):
            team_index = row * 3 + col
            if team_index < len(teams):
                team_num = f"{team_index + 1:2d}"
                team_name = teams[team_index][:12].ljust(12)  # Longer team names, 12 chars
                print(f" {team_num}.{team_name}", end="")
                if col < teams_in_row - 1:
                    print("│", end="")
        
        # Pad the last row to maintain alignment
        if row == 6:  # Last row with only 2 teams
            print(" " * 17, end="")  # Padding for missing 3rd team
            
        print(" │")
        
        # Add separator between rows (except after last row)
        if row < 6:
            print("├" + "─" * 53 + "┤")
    
    print("└" + "─" * 53 + "┘")
    
    return teams

def select_team(prompt, teams):
    """Allow user to select a team by number"""
    while True:
        try:
            choice = input(f"\n{prompt} (1-20): ").strip()
            team_num = int(choice)
            if 1 <= team_num <= 20:
                selected_team = teams[team_num - 1]
                print(f"✅ Selected: {selected_team}")
                return selected_team
            else:
                print("❌ Please enter a number between 1 and 20")
        except ValueError:
            print("❌ Please enter a valid number")

def print_banner():
    """Print the EPL Predictor banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    ⚽ EPL MATCH PREDICTOR ⚽                  ║
║                                                               ║
║        Advanced Machine Learning Premier League Predictions   ║
║              With Player Stats & Exact Scorelines             ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print the main menu"""
    menu = """
┌─────────────────────────────────────────────────────────────┐
│                        MAIN MENU                            │
├─────────────────────────────────────────────────────────────┤
│  1. 🔑 generate-key     - Generate FBR API key              │
│  2. 📥 sync            - Sync match data (3 seasons)        │
│  3. 📊 sync-enhanced   - Sync advanced stats & players      │
│  4. 🔄 update          - Update all cached data             │
│  5. 🧠 train           - Train ML prediction model          │
│  6. ⚡ next            - Predict next EPL match              │
│  7. 📅 fixtures        - Predict upcoming fixtures          │
│  8. ⚔️  match           - Predict specific match            │
│  9. ❓ help            - Show detailed help                 │
│  10. 🚪 exit           - Exit application                   │
└─────────────────────────────────────────────────────────────┘
    """
    print(menu)

def run_interactive_mode():
    """Run the interactive CLI mode"""
    clear_screen()
    print_banner()
    
    while True:
        print_menu()
        choice = input("\n🎯 Enter your choice (1-10): ").strip()
        
        # Clear screen before each action
        clear_screen()
        
        if choice == '1':
            print_banner()
            show_loading("Generating API key")
            args = argparse.Namespace()
            args.func = cmd_generate_key
            try:
                cmd_generate_key(args)
                show_success("API key generated successfully")
            except Exception as e:
                show_error(f"Failed to generate API key: {e}")
        
        elif choice == '2':
            print_banner()
            show_loading("Syncing match data")
            seasons = input("Number of seasons (default 3): ").strip() or "3"
            try:
                seasons = int(seasons)
                args = argparse.Namespace(seasons=seasons, force=False)
                cmd_sync(args)
                show_success(f"Successfully synced {seasons} seasons of match data")
            except ValueError:
                show_error("Invalid number of seasons")
            except Exception as e:
                show_error(f"Failed to sync data: {e}")
        
        elif choice == '3':
            print_banner()
            show_loading("Syncing enhanced data (team stats & players)")
            seasons = input("Number of seasons (default 2): ").strip() or "2"
            try:
                seasons = int(seasons)
                args = argparse.Namespace(seasons=seasons, season=None, force=False)
                cmd_sync_enhanced(args)
                show_success(f"Successfully synced enhanced data for {seasons} seasons")
            except ValueError:
                show_error("Invalid number of seasons")
            except Exception as e:
                show_error(f"Failed to sync enhanced data: {e}")
        
        elif choice == '4':
            print_banner()
            show_loading("Updating all cached data")
            try:
                # Call main with --update flag
                main(['--update'])
                show_success("All cached data updated successfully")
            except Exception as e:
                show_error(f"Failed to update data: {e}")
        
        elif choice == '5':
            print_banner()
            show_loading("Training machine learning model")
            args = argparse.Namespace()
            try:
                cmd_train(args)
                show_success("Model training completed successfully")
            except Exception as e:
                show_error(f"Failed to train model: {e}")
        
        elif choice == '6':
            print_banner()
            show_loading("Generating next match prediction")
            args = argparse.Namespace(debug=False)
            try:
                cmd_next_match(args)
                show_success("Next match prediction generated")
            except Exception as e:
                show_error(f"Failed to generate prediction: {e}")
        
        elif choice == '7':
            print_banner()
            print("\n📅 Predicting fixtures...")
            print("=" * 50)
            top = input("Number of fixtures (default 10): ").strip() or "10"
            try:
                top = int(top)
                show_loading(f"Generating predictions for {top} fixtures")
                args = argparse.Namespace(top=top)
                cmd_predict_fixtures(args)
                show_success(f"Predictions generated for {top} fixtures")
            except ValueError:
                show_error("Invalid number of fixtures")
            except Exception as e:
                show_error(f"Failed to generate fixture predictions: {e}")
        
        elif choice == '8':
            print_banner()
            print("\n⚔️ Predicting specific match...")
            print("=" * 50)
            
            # Display team selection grid
            teams = display_team_selection()
            
            # Select home team
            home = select_team("🏠 Select HOME team", teams)
            
            # Select away team  
            away = select_team("✈️ Select AWAY team", teams)
            
            if home and away:
                try:
                    show_loading(f"Predicting {home} vs {away}")
                    args = argparse.Namespace(home=home, away=away)
                    cmd_predict_match(args)
                    show_success(f"Prediction generated for {home} vs {away}")
                except Exception as e:
                    show_error(f"Failed to generate match prediction: {e}")
            else:
                show_error("Both team names are required")
        
        elif choice == '9':
            print_banner()
            show_detailed_help()
        
        elif choice == '10':
            clear_screen()
            print_banner()
            print("\n👋 Thanks for using EPL Match Predictor!")
            print("=" * 50)
            break
        
        else:
            print_banner()
            print("❌ Invalid choice. Please enter 1-10.")
        
        print("\n" + "=" * 80)
        input("⏸️  Press Enter to continue...")
        clear_screen()

def show_detailed_help():
    """Show detailed help information"""
    help_text = """
┌─────────────────────────────────────────────────────────────┐
│                      DETAILED HELP                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 🔑 GENERATE-KEY                                             │
│    Creates a new API key for accessing FBR data             │
│                                                             │
│ 📥 SYNC                                                     │
│    Downloads and caches match data for recent seasons       │
│    • Fetches results, fixtures, and basic stats             │
│    • Required before training or predictions                │
│                                                             │
│ 📊 SYNC-ENHANCED                                            │
│    Downloads advanced statistics and player data            │
│    • Team performance metrics and xG data                   │
│    • Individual player statistics                           │
│    • Enables player scoring predictions                     │
│                                                             │
│ 🧠 TRAIN                                                    │
│    Trains the machine learning model                        │
│    • Uses Random Forest vs Logistic Regression              │
│    • Automatically selects best performing model            │
│    • Creates 14 advanced features for prediction            │
│                                                             │
│ ⚡ NEXT                                                      │
│    Predicts the immediate next EPL match                    │
│    • Shows exact scoreline prediction                       │
│    • Displays player scoring probabilities                  │
│    • Includes historical context and form                   │
│                                                             │
│ 📅 FIXTURES                                                 │
│    Shows predictions for upcoming matches                   │
│    • Customizable number of fixtures                        │
│    • Confidence percentages for each outcome                │
│                                                             │
│ ⚔️ MATCH                                                    │
│    Predicts any specific team matchup                       │
│    • Enter home and away team names                         │
│    • Returns detailed prediction with exact score           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
    """
    print(help_text)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="⚽ EPL Match Predictor - Advanced ML Premier League Predictions")
    
    # Interactive mode (default)
    p.add_argument("--interactive", action="store_true", default=False, help="Run in interactive mode")
    
    # Quick command modes
    p.add_argument("--next", action="store_true", help="Quick predict next EPL match")
    p.add_argument("--debug", action="store_true", help="Show debug information")
    p.add_argument("--update", action="store_true", help="Update all cached data (fixtures, squads, stats)")
    
    # Subcommands for direct CLI usage
    sub = p.add_subparsers(dest="cmd", required=False)

    sk = sub.add_parser("generate-key", help="🔑 Generate a new API key via FBR API")
    sk.set_defaults(func=cmd_generate_key)

    ss = sub.add_parser("sync", help="📥 Fetch and cache last N seasons of EPL matches")
    ss.add_argument("--seasons", type=int, default=5, help="Number of previous seasons to sync (default 5)")
    ss.add_argument("--force", action="store_true", help="Re-download even if cached")
    ss.set_defaults(func=cmd_sync)

    ssch = sub.add_parser("sync-schedules", help="📅 Fetch and cache per-team schedules (with scores) for seasons")
    ssch.add_argument("--seasons", type=int, default=5, help="Number of previous seasons to sync (default 5)")
    ssch.add_argument("--season", type=str, help="Sync a specific season id (e.g., 2024-2025)")
    ssch.add_argument("--force", action="store_true", help="Re-download even if cached")
    ssch.set_defaults(func=cmd_sync_schedules)

    senh = sub.add_parser("sync-enhanced", help="📊 Fetch and cache team stats and player stats for enhanced predictions")
    senh.add_argument("--seasons", type=int, default=3, help="Number of previous seasons to sync (default 3)")
    senh.add_argument("--season", type=str, help="Sync a specific season id (e.g., 2024-2025)")
    senh.add_argument("--force", action="store_true", help="Re-download even if cached")
    senh.set_defaults(func=cmd_sync_enhanced)

    tr = sub.add_parser("train", help="🧠 Train the prediction model on cached data")
    tr.set_defaults(func=cmd_train)

    pf = sub.add_parser("predict-fixtures", help="📅 Predict upcoming fixtures from cached current season")
    pf.add_argument("--top", type=int, default=10, help="Show top N upcoming fixtures (by date)")
    pf.set_defaults(func=cmd_predict_fixtures)

    pm = sub.add_parser("predict-match", help="⚔️ Predict a single specified match")
    pm.add_argument("--home", required=True, help="Home team name (as in FBR matches data)")
    pm.add_argument("--away", required=True, help="Away team name (as in FBR matches data)")
    pm.set_defaults(func=cmd_predict_match)

    return p


def ensure_current_season_fixtures(client: FBRClient) -> bool:
    """Ensure current season fixtures are available for predictions"""
    current_season = get_current_season()
    
    # Check if current season data exists
    current_cache_file = CACHE_DIR / f"matches_league{LEAGUE_ID_EPL}_{current_season}.json.gz"
    if current_cache_file.exists():
        # Check if file has fixture data (not just completed matches)
        try:
            with gzip.open(current_cache_file, 'rt') as f:
                data = json.load(f)
                # Look for matches without scores (upcoming fixtures)
                upcoming_count = sum(1 for match in data if not match.get('home_team_score') and not match.get('away_team_score'))
                if upcoming_count > 0:
                    return True
                else:
                    print(f"⚠️  Current season ({current_season}) has no upcoming fixtures. Refreshing...")
        except Exception:
            print(f"⚠️  Current season cache corrupted. Refreshing...")
    
    # Need to fetch current season data
    print(f"📥 Fetching current season fixtures ({current_season})...")
    try:
        matches = client.league_matches(LEAGUE_ID_EPL, current_season)
        if matches:
            # Save the matches
            current_cache_file.parent.mkdir(exist_ok=True)
            with gzip.open(current_cache_file, 'wt') as f:
                json.dump(matches, f)
            print(f"✅ Current season fixtures updated")
            return True
        else:
            print(f"⚠️  No fixtures found for current season {current_season}")
            return False
    except Exception as e:
        print(f"❌ Failed to fetch current season fixtures: {e}")
        return False

def cmd_next_match(args):
    """Command handler for next match prediction with auto-sync"""
    client = FBRClient()
    
    # Ensure current season fixtures are available
    if not ensure_current_season_fixtures(client):
        print("❌ Cannot predict next match without current season fixtures")
        return
    
    # Call the existing next match logic
    main(['--next'] + (['--debug'] if args.debug else []))

def main(argv: Optional[List[str]] = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    
    # Check for interactive mode (when no args or explicitly requested)
    if (not argv and len(sys.argv) == 1) or getattr(args, "interactive", False):
        run_interactive_mode()
        return
    
    # Handle --update flag
    if getattr(args, "update", False):
        print("🔄 Updating all cached data...")
        client = FBRClient()
        
        # Update fixtures and team data from FBR
        print("📥 Updating fixtures from FBR...")
        sync_args = type('Args', (), {'seasons': 3, 'force': True})()
        cmd_sync(sync_args)
        
        # Update enhanced data (team stats, player stats)
        print("📊 Updating enhanced data...")
        enhanced_args = type('Args', (), {'seasons': 3, 'season': None, 'force': True})()
        cmd_sync_enhanced(enhanced_args)
        
        # Update squad data from API-Football if available
        if os.getenv("API_FOOTBALL_KEY"):
            print("👥 Updating squad data from API-Football...")
            api_football = APIFootballClient()
            for team_id in TEAM_MAPPING.values():
                try:
                    api_football.get_team_squad(team_id, force_refresh=True)
                except Exception as e:
                    print(f"Could not update squad for team {team_id}: {e}")
        else:
            print("⚠️ API_FOOTBALL_KEY not set, skipping squad updates")
        
        # Clear odds cache (since odds change rapidly)
        print("💰 Clearing odds cache for fresh data...")
        odds_cache_dir = CACHE_DIR / "odds"
        if odds_cache_dir.exists():
            import shutil
            shutil.rmtree(odds_cache_dir)
            print("Odds cache cleared")
        
        print("✅ Update complete!")
        return

    if getattr(args, "cmd", None):
        # Subcommand path (generate-key, sync, train, predict-*)
        args.func(args)
        return    # Default behavior (no subcommand): quick prediction for next match or all matches of next gameweek
    client = FBRClient()
    # Ensure model exists; if not, train on last 5 seasons
    model_path = MODELS_DIR / "epl_result_model.joblib"
    seasons = get_last_n_seasons(client, LEAGUE_ID_EPL, n=5)
    if not model_path.exists():
        # Prepare data (may take some time due to rate limits)
        df_train = prepare_dataset(client, seasons)
        _ = train_model(df_train, model_path)
    else:
        # Validate model can be loaded; if not, retrain now
        try:
            _ = load_model(model_path)
        except ModelLoadError:
            print("Existing model file is incompatible. Retraining on last 5 seasons...")
            df_train = prepare_dataset(client, seasons)
            _ = train_model(df_train, model_path)

    # Prepare dataset with fresh current fixtures
    if not seasons:
        print("Could not determine seasons for EPL.")
        sys.exit(1)
    df_all = prepare_dataset_with_fresh_current(client, seasons)

    # Upcoming fixtures (choose the immediate next by date/time >= now UTC)
    upcoming = df_all[df_all[["home_team_score", "away_team_score"]].isna().all(axis=1)].copy()
    if upcoming.empty:
        print("No upcoming fixtures found.")
        sys.exit(0)
    now = pd.Timestamp(datetime.now(UTC).date())
    # If times are present, sort by date then time and pick first with date >= today
    upcoming_sorted = upcoming.sort_values(["date", "time"]) if "time" in upcoming.columns else upcoming.sort_values(["date"]) 
    next_match = upcoming_sorted[upcoming_sorted["date"] >= now].head(1)
    preds = predict_for_rows(next_match, model_path)
    if preds.empty:
        print("No upcoming fixtures found.")
        sys.exit(0)
    r = preds.iloc[0]
    if getattr(args, "debug", False):
        try:
            model, feature_cols, classes_, model_type = load_model(model_path)
            fr = next_match.iloc[0][feature_cols]
            print("[DEBUG] classes:", classes_)
            print("[DEBUG] model type:", model_type)
            print("[DEBUG] features:")
            for k in feature_cols:
                print(f"  {k}: {fr.get(k)}")
        except Exception as e:
            print("[DEBUG] unable to print features:", e)
    probs = {k: r[k] for k in preds.columns if k.startswith("P(")}
    best = max(probs.items(), key=lambda kv: kv[1])
    date_str = r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "TBD"
    
    # Get bookmaker odds and combine with ML predictions
    bookmaker_client = BookmakerAPIClient()
    odds_data = bookmaker_client.get_match_odds(
        home_team=r['home'], 
        away_team=r['away'], 
        date=date_str
    )
    
    # Combine ML predictions with bookmaker odds
    combined_result = combine_ml_and_odds_predictions(probs, odds_data, ml_weight=0.7)
    final_probs = combined_result.get("final_probs", probs)
    
    # Extract final probabilities
    ph = float(final_probs.get('P(H)', 0.0))
    pd_ = float(final_probs.get('P(D)', 0.0))
    pa = float(final_probs.get('P(A)', 0.0))
    
    # Update best prediction with final probabilities
    best = max(final_probs.items(), key=lambda kv: kv[1])
    
    # Enhanced Beautiful Output
    print("\n" + "="*80)
    print("⚽" + " EPL MATCH PREDICTION ".center(76) + "⚽")
    print("="*80)
    
    # Match Header
    print(f"\n🏟️  {r['home']} vs {r['away']}")
    print(f"📅 Date: {date_str}")
    
    # Show prediction method
    if combined_result.get("odds_available"):
        print(f"🎯 Method: ML + Bookmaker Odds ({combined_result.get('num_bookmakers', 0)} bookmakers)")
    else:
        print("🎯 Method: ML Model Only")
    
    print("\n" + "─"*80)
    
    # Result Probabilities with visual bars
    print("📊 WIN PROBABILITIES")
    print("─"*40)
    home_bar = "█" * int(ph * 30)
    draw_bar = "█" * int(pd_ * 30)
    away_bar = "█" * int(pa * 30)
    
    print(f"🏠 {r['home']:20} {ph*100:5.1f}% {home_bar}")
    print(f"🤝 Draw{'':<16} {pd_*100:5.1f}% {draw_bar}")
    print(f"✈️  {r['away']:20} {pa*100:5.1f}% {away_bar}")
    
    # Most Likely Result
    winner = "🏠 " + r['home'] if best[0] == 'P(H)' else ("🤝 Draw" if best[0] == 'P(D)' else "✈️ " + r['away'])
    print(f"\n🎯 Most Likely: {winner} ({best[1]*100:.1f}%)")

    # Display bookmaker odds information if available
    if combined_result.get("odds_available"):
        avg_odds = combined_result.get("avg_odds", {})
        ml_probs = combined_result.get("ml_probs", {})
        bm_probs = combined_result.get("bookmaker_probs", {})
        
        print("\n" + "─"*80)
        print("💰 BOOKMAKER ANALYSIS")
        print("─"*40)
        
        if avg_odds:
            print("📈 Average Odds:")
            for outcome, odd in avg_odds.items():
                print(f"   {outcome}: {odd:.2f}")
        
        print(f"\n🔬 Prediction Breakdown:")
        mapping = [("P(H)", "🏠 " + r['home'], "Home"), 
                  ("P(D)", "🤝 Draw", "Draw"), 
                  ("P(A)", "✈️ " + r['away'], "Away")]
        
        for ml_key, display_name, bm_key in mapping:
            ml_prob = ml_probs.get(ml_key, 0) * 100
            bm_prob = bm_probs.get(bm_key, 0) * 100
            final_prob = final_probs.get(ml_key, 0) * 100
            print(f"   {display_name:<20} ML: {ml_prob:5.1f}% | Odds: {bm_prob:5.1f}% | Final: {final_prob:5.1f}%")

    # Context summary
    # Find IDs
    match_row = df_all.loc[(df_all['date'] == r['date']) & (df_all['home'] == r['home']) & (df_all['away'] == r['away'])].head(1)
    if not match_row.empty:
        h_id = match_row.iloc[0]['home_team_id']
        a_id = match_row.iloc[0]['away_team_id']
        season = match_row.iloc[0]['season_id']
        h2h = summarize_h2h(df_all, h_id, a_id)
        home_form = summarize_recent_form(df_all, h_id)
        away_form = summarize_recent_form(df_all, a_id)
        ha_form = summarize_home_away(df_all, h_id, a_id)
        venue = summarize_venue_specific(df_all, h_id, a_id)
        
        # Enhanced score prediction using team averages
        home_goals_avg = (home_form['gf_avg'] + away_form['ga_avg']) / 2 if home_form['gf_avg'] and away_form['ga_avg'] else 1.5
        away_goals_avg = (away_form['gf_avg'] + home_form['ga_avg']) / 2 if away_form['gf_avg'] and home_form['ga_avg'] else 1.2
        
        # Get exact score prediction
        exact_home, exact_away = predict_exact_score_poisson(home_goals_avg, away_goals_avg)
        
        # Scoreline Prediction
        print("\n" + "─"*80)
        print("⚽ PREDICTED SCORELINE")
        print("─"*40)
        print(f"🥅 Final Score: {r['home']} {exact_home} - {exact_away} {r['away']}")
        
        # Get top scorers prediction
        try:
            # Try current season first, fallback to previous season for player data
            top_scorers = None
            seasons_to_try = [season]
            year = int(season.split("-")[0])
            prev_season = f"{year-1}-{year}"
            if prev_season != season:
                seasons_to_try.append(prev_season)
            
            for try_season in seasons_to_try:
                try:
                    top_scorers = get_enhanced_top_scorers(client, h_id, a_id, try_season)
                    if top_scorers and (top_scorers.get('home') or top_scorers.get('away')):
                        break
                except:
                    continue
            
            if not top_scorers:
                top_scorers = {'home': [], 'away': []}
        except:
            top_scorers = {'home': [], 'away': []}
        
        print("\n" + "─"*80)
        print("📈 MATCH ANALYSIS")
        print("─"*40)
        print(f"📊 H2H Record: {h2h['record']} (Goals: {h2h['gf']}-{h2h['ga']}) in {h2h['meetings']} games")
        print(f"🔥 Recent Form: {r['home']} {home_form['ppg']:.1f}PPG ({home_form['gf_avg']:.1f}GF {home_form['ga_avg']:.1f}GA) | {r['away']} {away_form['ppg']:.1f}PPG ({away_form['gf_avg']:.1f}GF {away_form['ga_avg']:.1f}GA)")
        print(f"🏠 Home/Away: {r['home']} {ha_form['home_ppg_last5_home']:.1f}PPG at home | {r['away']} {ha_form['away_ppg_last5_away']:.1f}PPG away")
        print(f"🏟️  Venue History: {venue['away_at_venue']} in {venue['meetings']} visits")
        
        # Top scorers predictions
        if top_scorers['home'] or top_scorers['away']:
            print("\n" + "─"*80)
            print("⭐ TOP EXPECTED SCORERS")
            print("─"*40)
            for team_key, team_name in [('home', r['home']), ('away', r['away'])]:
                if top_scorers[team_key]:
                    emoji = "🏠" if team_key == 'home' else "✈️"
                    print(f"{emoji} {team_name}:")
                    for i, scorer in enumerate(top_scorers[team_key], 1):
                        print(f"  {i}. {scorer['name']} ({scorer['position']}) - {scorer['scoring_probability']*100:.1f}% ({scorer['goals']}G/{scorer['matches']}M)")
                else:
                    emoji = "🏠" if team_key == 'home' else "✈️"
                    print(f"{emoji} {team_name}: Player data not available")
        else:
            print("\n" + "─"*80)
            print("⭐ TOP EXPECTED SCORERS")
            print("─"*40)
            print("📊 Player statistics not available (run 'sync-enhanced' for predictions)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
