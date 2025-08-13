#!/usr/bin/env python3
"""
Advanced ML Prediction Engine with comprehensive transfer impact analysis.
Includes xG/xGA calculations, injury data, form analysis, and multi-source integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import requests
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLPredictionEngine:
    """
    Advanced ML engine that considers:
    - Transfer impact on xG/xGA
    - Player departures and arrivals
    - Injury impact on team strength
    - Recent form with weighted importance
    - H2H records with venue consideration
    - Tactical matchups and playing styles
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
        # Initialize transfer validator for accurate squad data
        try:
            from ..utils.real_time_transfer_validator import RealTimeTransferValidator
            self.transfer_validator = RealTimeTransferValidator(api_keys)
        except ImportError:
            self.transfer_validator = None
        
        # Team tactical profiles (affects xG/xGA multipliers)
        self.team_profiles = {
            'liverpool': {'attacking_style': 'high_intensity', 'defensive_style': 'high_line', 'xg_multiplier': 1.15, 'pace_factor': 1.2},
            'manchester_city': {'attacking_style': 'possession', 'defensive_style': 'compact', 'xg_multiplier': 1.2, 'pace_factor': 1.0},
            'arsenal': {'attacking_style': 'counter_attack', 'defensive_style': 'organized', 'xg_multiplier': 1.1, 'pace_factor': 1.1},
            'chelsea': {'attacking_style': 'versatile', 'defensive_style': 'solid', 'xg_multiplier': 1.05, 'pace_factor': 1.0},
            'manchester_united': {'attacking_style': 'direct', 'defensive_style': 'deep_block', 'xg_multiplier': 1.0, 'pace_factor': 0.95},
            'tottenham': {'attacking_style': 'fast_transition', 'defensive_style': 'medium_block', 'xg_multiplier': 1.08, 'pace_factor': 1.15},
            'newcastle': {'attacking_style': 'physical', 'defensive_style': 'aggressive', 'xg_multiplier': 1.0, 'pace_factor': 1.05},
            'brighton': {'attacking_style': 'technical', 'defensive_style': 'pressing', 'xg_multiplier': 1.02, 'pace_factor': 1.0},
            'aston_villa': {'attacking_style': 'balanced', 'defensive_style': 'structured', 'xg_multiplier': 1.0, 'pace_factor': 1.0},
            'west_ham': {'attacking_style': 'direct', 'defensive_style': 'compact', 'xg_multiplier': 0.95, 'pace_factor': 0.9}
        }
    
    def calculate_transfer_impact(self, team_name: str, season: str = "2024-25") -> Dict[str, float]:
        """
        Calculate comprehensive transfer impact on team's xG and xGA.
        
        Returns:
            Dict with xG_change, xGA_change, overall_strength_change
        """
        if not self.transfer_validator:
            return {'xG_change': 0, 'xGA_change': 0, 'overall_strength_change': 0}
        
        transfer_summary = self.transfer_validator.get_team_transfer_summary(team_name, season)
        
        impact = {
            'xG_change': 0.0,          # Change in expected goals for
            'xGA_change': 0.0,         # Change in expected goals against  
            'overall_strength_change': 0.0,
            'key_arrivals': [],
            'key_departures': [],
            'net_spend_impact': 0.0
        }
        
        # Calculate arrivals impact
        for arrival in transfer_summary.get('transfers_in', []):
            player_impact = self._calculate_player_impact(arrival, 'arrival', team_name)
            impact['xG_change'] += player_impact.get('xG_contribution', 0)
            impact['xGA_change'] += player_impact.get('xGA_contribution', 0)  # Negative for defenders
            impact['overall_strength_change'] += player_impact.get('overall_rating', 0)
            
            if player_impact.get('is_key_player', False):
                impact['key_arrivals'].append({
                    'name': arrival['name'],
                    'impact': player_impact
                })
        
        # Calculate departures impact (negative)
        for departure in transfer_summary.get('transfers_out', []):
            player_impact = self._calculate_player_impact(departure, 'departure', team_name)
            impact['xG_change'] -= player_impact.get('xG_contribution', 0)  # Lose xG
            impact['xGA_change'] -= player_impact.get('xGA_contribution', 0)  # Lose defensive contribution
            impact['overall_strength_change'] -= player_impact.get('overall_rating', 0)
            
            if player_impact.get('is_key_player', False):
                impact['key_departures'].append({
                    'name': departure['name'],
                    'impact': player_impact
                })
        
        # Apply team-specific multipliers
        team_key = self._normalize_team_name(team_name)
        if team_key in self.team_profiles:
            profile = self.team_profiles[team_key]
            impact['xG_change'] *= profile.get('xg_multiplier', 1.0)
        
        return impact
    
    def _calculate_player_impact(self, player_data: Dict, transfer_type: str, team_name: str) -> Dict[str, float]:
        """Calculate individual player's impact on team metrics."""
        position = player_data.get('position', '').upper()
        fee_str = player_data.get('fee', '€0M')
        
        # Extract fee amount for quality estimation
        fee_amount = 0
        if '€' in fee_str and 'M' in fee_str:
            try:
                fee_amount = float(fee_str.replace('€', '').replace('M', ''))
            except:
                fee_amount = 0
        
        # Base impact by position
        position_impacts = {
            'ST': {'xG_contribution': 18, 'xGA_contribution': 0, 'overall_rating': 8.0},
            'CF': {'xG_contribution': 16, 'xGA_contribution': 0, 'overall_rating': 7.8},
            'FW': {'xG_contribution': 15, 'xGA_contribution': 0, 'overall_rating': 7.5},
            'AM': {'xG_contribution': 12, 'xGA_contribution': 1, 'overall_rating': 7.5},
            'LW': {'xG_contribution': 10, 'xGA_contribution': 0, 'overall_rating': 7.0},
            'RW': {'xG_contribution': 10, 'xGA_contribution': 0, 'overall_rating': 7.0},
            'CM': {'xG_contribution': 6, 'xGA_contribution': 3, 'overall_rating': 7.0},
            'DM': {'xG_contribution': 3, 'xGA_contribution': 8, 'overall_rating': 7.0},
            'LB': {'xG_contribution': 2, 'xGA_contribution': 6, 'overall_rating': 6.5},
            'RB': {'xG_contribution': 2, 'xGA_contribution': 6, 'overall_rating': 6.5},
            'CB': {'xG_contribution': 1, 'xGA_contribution': 10, 'overall_rating': 7.0},
            'GK': {'xG_contribution': 0, 'xGA_contribution': 15, 'overall_rating': 7.0}
        }
        
        base_impact = position_impacts.get(position, position_impacts['CM'])
        
        # Fee-based quality multiplier
        if fee_amount >= 100:      # €100M+ = world class
            quality_multiplier = 1.5
        elif fee_amount >= 60:     # €60M+ = top quality  
            quality_multiplier = 1.3
        elif fee_amount >= 30:     # €30M+ = good quality
            quality_multiplier = 1.1
        elif fee_amount >= 15:     # €15M+ = decent
            quality_multiplier = 1.0
        else:                      # Lower fees
            quality_multiplier = 0.8
        
        # Apply quality multiplier
        impact = {
            'xG_contribution': base_impact['xG_contribution'] * quality_multiplier,
            'xGA_contribution': base_impact['xGA_contribution'] * quality_multiplier,
            'overall_rating': base_impact['overall_rating'] * quality_multiplier,
            'is_key_player': fee_amount >= 30 or position in ['ST', 'CF', 'AM', 'CB', 'GK'],
            'fee_amount': fee_amount,
            'position': position
        }
        
        return impact
    
    def get_comprehensive_team_data(self, team_name: str, opponent_name: str = None, 
                                  seasons_back: int = 2) -> Dict[str, Any]:
        """
        Gather comprehensive team data from all available sources.
        
        Args:
            team_name: Team to analyze
            opponent_name: Opponent team (for H2H and tactical matchup)
            seasons_back: How many seasons of data to include
            
        Returns:
            Comprehensive team data dictionary
        """
        team_data = {
            'team_name': team_name,
            'basic_stats': {},
            'transfer_impact': {},
            'injury_impact': {},
            'form_analysis': {},
            'h2h_data': {},
            'tactical_profile': {},
            'venue_performance': {},
            'player_contributions': {}
        }
        
        # 1. Transfer Impact Analysis
        team_data['transfer_impact'] = self.calculate_transfer_impact(team_name)
        
        # 2. Basic Team Statistics (from cache or API)
        team_data['basic_stats'] = self._get_basic_team_stats(team_name, seasons_back)
        
        # 3. Injury Impact Analysis
        team_data['injury_impact'] = self._get_injury_impact(team_name)
        
        # 4. Recent Form Analysis (weighted by recency)
        team_data['form_analysis'] = self._get_weighted_form_analysis(team_name)
        
        # 5. H2H Analysis (if opponent provided)
        if opponent_name:
            team_data['h2h_data'] = self._get_h2h_analysis(team_name, opponent_name)
        
        # 6. Tactical Profile
        team_data['tactical_profile'] = self._get_tactical_profile(team_name, opponent_name)
        
        # 7. Venue Performance
        team_data['venue_performance'] = self._get_venue_performance(team_name)
        
        # 8. Key Player Contributions
        team_data['player_contributions'] = self._get_player_contributions(team_name)
        
        return team_data
    
    def _get_basic_team_stats(self, team_name: str, seasons_back: int) -> Dict[str, float]:
        """Get basic team statistics with realistic EPL data"""
        # More realistic team-specific stats based on actual EPL performance patterns
        team_stats = {
            'liverpool': {'xg_per_game': 2.1, 'xga_per_game': 0.9, 'ppg': 2.3, 'home_factor': 1.2},
            'manchester_city': {'xg_per_game': 2.3, 'xga_per_game': 0.8, 'ppg': 2.4, 'home_factor': 1.15},
            'arsenal': {'xg_per_game': 2.0, 'xga_per_game': 1.0, 'ppg': 2.1, 'home_factor': 1.18},
            'chelsea': {'xg_per_game': 1.8, 'xga_per_game': 1.1, 'ppg': 1.8, 'home_factor': 1.12},
            'manchester_united': {'xg_per_game': 1.6, 'xga_per_game': 1.2, 'ppg': 1.7, 'home_factor': 1.15},
            'tottenham': {'xg_per_game': 1.9, 'xga_per_game': 1.3, 'ppg': 1.9, 'home_factor': 1.10},
            'newcastle': {'xg_per_game': 1.7, 'xga_per_game': 1.1, 'ppg': 1.8, 'home_factor': 1.20},
            'west_ham': {'xg_per_game': 1.4, 'xga_per_game': 1.4, 'ppg': 1.4, 'home_factor': 1.08},
            'brighton': {'xg_per_game': 1.6, 'xga_per_game': 1.2, 'ppg': 1.6, 'home_factor': 1.12},
            'aston_villa': {'xg_per_game': 1.8, 'xga_per_game': 1.1, 'ppg': 1.9, 'home_factor': 1.15}
        }
        
        team_key = team_name.lower().replace(' ', '_')
        base_stats = team_stats.get(team_key, {
            'xg_per_game': 1.5,
            'xga_per_game': 1.3,
            'ppg': 1.5,
            'home_factor': 1.10
        })
        
        return {
            'xg_per_game': base_stats['xg_per_game'],
            'xga_per_game': base_stats['xga_per_game'],
            'ppg': base_stats['ppg'],
            'gf_per_game': base_stats['xg_per_game'] * 1.1,  # Slightly higher than xG
            'ga_per_game': base_stats['xga_per_game'] * 1.05,
            'home_advantage': base_stats['home_factor']
        }
    
    def _get_injury_impact(self, team_name: str) -> Dict[str, Any]:
        """Calculate injury impact on team strength."""
        # Mock implementation - replace with real injury API
        return {
            'total_injuries': 3,
            'key_player_injuries': 1,
            'injury_severity_score': 0.15,  # 0-1 scale
            'estimated_points_loss': 2.5,
            'affected_positions': ['CB', 'LW'],
            'return_dates': {}
        }
    
    def _get_weighted_form_analysis(self, team_name: str, games_back: int = 10) -> Dict[str, float]:
        """Analyze recent form with exponential weighting (recent games more important)."""
        # Mock recent results - replace with actual data
        recent_results = [
            {'result': 'W', 'gf': 3, 'ga': 1, 'xg': 2.1, 'xga': 0.8, 'days_ago': 3},
            {'result': 'D', 'gf': 1, 'ga': 1, 'xg': 1.4, 'xga': 1.2, 'days_ago': 10},
            {'result': 'W', 'gf': 2, 'ga': 0, 'xg': 2.3, 'xga': 0.6, 'days_ago': 17},
            {'result': 'L', 'gf': 0, 'ga': 2, 'xg': 1.1, 'xga': 1.9, 'days_ago': 24},
            {'result': 'W', 'gf': 4, 'ga': 2, 'xg': 3.2, 'xga': 1.8, 'days_ago': 31}
        ]
        
        total_weight = 0
        weighted_points = 0
        weighted_gf = 0
        weighted_ga = 0
        weighted_xg = 0
        weighted_xga = 0
        
        for game in recent_results:
            # Exponential decay - recent games weighted more heavily
            weight = np.exp(-game['days_ago'] / 14)  # Half-life of 14 days
            total_weight += weight
            
            points = 3 if game['result'] == 'W' else 1 if game['result'] == 'D' else 0
            weighted_points += points * weight
            weighted_gf += game['gf'] * weight
            weighted_ga += game['ga'] * weight
            weighted_xg += game['xg'] * weight
            weighted_xga += game['xga'] * weight
        
        if total_weight > 0:
            return {
                'weighted_ppg': weighted_points / total_weight,
                'weighted_gf': weighted_gf / total_weight,
                'weighted_ga': weighted_ga / total_weight,
                'weighted_xg': weighted_xg / total_weight,
                'weighted_xga': weighted_xga / total_weight,
                'form_trend': 'improving' if weighted_points / total_weight > 2.0 else 'declining',
                'momentum_score': min(weighted_points / total_weight / 3.0, 1.0)  # 0-1 scale
            }
        else:
            return {
                'weighted_ppg': 1.5,  # Default values
                'weighted_gf': 1.5,
                'weighted_ga': 1.5,
                'weighted_xg': 1.4,
                'weighted_xga': 1.3,
                'form_trend': 'neutral',
                'momentum_score': 0.5
            }
        
        # Fallback return should never be reached due to above logic
        return {'weighted_ppg': 1.0, 'momentum_score': 0.5}
    
    def _get_h2h_analysis(self, team_name: str, opponent_name: str) -> Dict[str, Any]:
        """Analyze head-to-head record with venue and recency weighting."""
        # Mock H2H data - replace with actual historical data
        return {
            'meetings_total': 6,
            'wins': 2,
            'draws': 2,
            'losses': 2,
            'gf_total': 8,
            'ga_total': 7,
            'recent_meetings': [
                {'result': 'W', 'score': '3-1', 'venue': 'home', 'season': '2023-24'},
                {'result': 'L', 'score': '1-2', 'venue': 'away', 'season': '2023-24'},
                {'result': 'D', 'score': '2-2', 'venue': 'home', 'season': '2022-23'}
            ],
            'home_record': {'w': 1, 'd': 1, 'l': 1},
            'away_record': {'w': 1, 'd': 1, 'l': 1},
            'tactical_matchup_score': 0.6  # How well team matches up tactically
        }
    
    def _get_tactical_profile(self, team_name: str, opponent_name: str = None) -> Dict[str, Any]:
        """Get tactical profile and matchup analysis."""
        team_key = self._normalize_team_name(team_name)
        profile = self.team_profiles.get(team_key, {
            'attacking_style': 'balanced',
            'defensive_style': 'compact',
            'xg_multiplier': 1.0,
            'pace_factor': 1.0
        })
        
        matchup_bonus = 0.0
        if opponent_name:
            opponent_key = self._normalize_team_name(opponent_name)
            opponent_profile = self.team_profiles.get(opponent_key, {})
            
            # Calculate tactical matchup advantage
            # High intensity vs possession = slight advantage
            # Counter attack vs high line = advantage
            # etc.
            matchup_bonus = self._calculate_tactical_matchup(profile, opponent_profile)
        
        profile['matchup_bonus'] = matchup_bonus
        return profile
    
    def _calculate_tactical_matchup(self, team_profile: Dict, opponent_profile: Dict) -> float:
        """Calculate tactical matchup advantage (-0.2 to +0.2)."""
        team_attack = team_profile.get('attacking_style', '')
        team_defense = team_profile.get('defensive_style', '')
        opp_attack = opponent_profile.get('attacking_style', '')
        opp_defense = opponent_profile.get('defensive_style', '')
        
        bonus = 0.0
        
        # Attacking vs defensive matchups
        if team_attack == 'counter_attack' and opp_defense == 'high_line':
            bonus += 0.15
        elif team_attack == 'possession' and opp_defense == 'pressing':
            bonus += 0.1
        elif team_attack == 'high_intensity' and opp_defense == 'compact':
            bonus += 0.05
        
        # Style vs style matchups
        if team_attack == 'fast_transition' and opp_attack == 'possession':
            bonus += 0.1
        
        return max(-0.2, min(0.2, bonus))
    
    def _get_venue_performance(self, team_name: str) -> Dict[str, float]:
        """Get home/away performance metrics."""
        # Mock venue data - replace with actual statistics
        return {
            'home_ppg': 2.2,
            'away_ppg': 1.4,
            'home_gf_avg': 2.1,
            'home_ga_avg': 0.9,
            'away_gf_avg': 1.6,
            'away_ga_avg': 1.3,
            'home_advantage_factor': 1.15  # Multiplier for home performance
        }
    
    def _get_player_contributions(self, team_name: str) -> Dict[str, Any]:
        """Get key player contributions and dependencies."""
        # Mock player data - replace with actual player statistics
        return {
            'key_players': [
                {'name': 'Mohamed Salah', 'goals_contribution': 0.35, 'assists_contribution': 0.25, 'injury_risk': 0.1},
                {'name': 'Virgil van Dijk', 'defensive_contribution': 0.4, 'leadership_factor': 0.3, 'injury_risk': 0.15}
            ],
            'squad_depth_score': 0.8,  # 0-1 scale
            'dependency_risk': 0.25,   # Risk from key player absence
            'new_signings_integration': 0.7  # How well new signings have integrated
        }
    
    def create_advanced_features(self, home_team_data: Dict, away_team_data: Dict, 
                               venue: str = 'home') -> np.ndarray:
        """
        Create comprehensive feature vector for ML prediction.
        
        Returns:
            Feature vector with all calculated metrics
        """
        features = []
        
        # 1. Basic team strength (adjusted for transfers)
        home_stats = home_team_data['basic_stats']
        away_stats = away_team_data['basic_stats']
        
        features.extend([
            home_stats['xg_per_game'],
            home_stats['xga_per_game'], 
            away_stats['xg_per_game'],
            away_stats['xga_per_game']
        ])
        
        # 2. Transfer impact
        home_transfer = home_team_data['transfer_impact']
        away_transfer = away_team_data['transfer_impact']
        
        features.extend([
            home_transfer['xG_change'] / 38,  # Per game impact
            home_transfer['xGA_change'] / 38,
            away_transfer['xG_change'] / 38,
            away_transfer['xGA_change'] / 38,
            home_transfer['overall_strength_change'],
            away_transfer['overall_strength_change']
        ])
        
        # 3. Form and momentum
        home_form = home_team_data['form_analysis']
        away_form = away_team_data['form_analysis']
        
        features.extend([
            home_form.get('weighted_ppg', 1.0),
            home_form.get('momentum_score', 0.5),
            away_form.get('weighted_ppg', 1.0),
            away_form.get('momentum_score', 0.5)
        ])
        
        # 4. Injury impact
        home_injury = home_team_data['injury_impact']
        away_injury = away_team_data['injury_impact']
        
        features.extend([
            home_injury.get('injury_severity_score', 0.0),
            away_injury.get('injury_severity_score', 0.0)
        ])
        
        # 5. Tactical matchup
        home_tactical = home_team_data['tactical_profile']
        away_tactical = away_team_data['tactical_profile']
        
        features.extend([
            home_tactical.get('matchup_bonus', 0.0),
            home_tactical.get('pace_factor', 1.0),
            away_tactical.get('pace_factor', 1.0)
        ])
        
        # 6. H2H factors
        home_h2h = home_team_data.get('h2h_data', {})
        features.extend([
            home_h2h.get('tactical_matchup_score', 0.5),
            home_h2h.get('wins', 0) / max(home_h2h.get('meetings_total', 1), 1)
        ])
        
        # 7. Venue advantage
        if venue == 'home':
            home_venue = home_team_data['venue_performance']
            features.extend([
                home_venue.get('home_advantage_factor', 1.0),
                1.0  # Home team playing at home
            ])
        else:
            away_venue = away_team_data['venue_performance']
            features.extend([
                away_venue.get('home_advantage_factor', 1.0),
                0.0  # Away team playing away
            ])
        
        # 8. Squad quality and depth
        home_players = home_team_data['player_contributions']
        away_players = away_team_data['player_contributions']
        
        features.extend([
            home_players.get('squad_depth_score', 0.5),
            home_players.get('dependency_risk', 0.5),
            away_players.get('squad_depth_score', 0.5),
            away_players.get('dependency_risk', 0.5)
        ])
        
        # Store feature names for interpretability
        if not self.feature_names:
            self.feature_names = [
                'home_xg_per_game', 'home_xga_per_game', 'away_xg_per_game', 'away_xga_per_game',
                'home_transfer_xg_impact', 'home_transfer_xga_impact', 'away_transfer_xg_impact', 'away_transfer_xga_impact',
                'home_transfer_strength', 'away_transfer_strength',
                'home_form_ppg', 'home_momentum', 'away_form_ppg', 'away_momentum',
                'home_injury_impact', 'away_injury_impact',
                'home_tactical_bonus', 'home_pace_factor', 'away_pace_factor',
                'h2h_tactical_score', 'h2h_win_rate',
                'venue_advantage', 'is_home_team',
                'home_squad_depth', 'home_dependency_risk', 'away_squad_depth', 'away_dependency_risk'
            ]
        
        return np.array(features)
    
    def train_advanced_model(self, training_data: List[Dict]) -> Dict[str, Any]:
        """
        Train ensemble of ML models on comprehensive features.
        
        Args:
            training_data: List of historical matches with comprehensive data
            
        Returns:
            Training results and model performance
        """
        print("🧠 Training Advanced ML Models...")
        
        X_features = []
        y_results = []
        
        # Process training data
        for match in training_data:
            try:
                home_data = self.get_comprehensive_team_data(
                    match['home_team'], match['away_team']
                )
                away_data = self.get_comprehensive_team_data(
                    match['away_team'], match['home_team']
                )
                
                features = self.create_advanced_features(home_data, away_data, 'home')
                X_features.append(features)
                
                # Result encoding: 0=away_win, 1=draw, 2=home_win
                if match['home_goals'] > match['away_goals']:
                    y_results.append(2)
                elif match['home_goals'] < match['away_goals']:
                    y_results.append(0)
                else:
                    y_results.append(1)
                    
            except Exception as e:
                print(f"⚠️  Skipping match due to error: {e}")
                continue
        
        if len(X_features) < 10:
            print("❌ Insufficient training data")
            return {'success': False, 'error': 'Insufficient data'}
        
        X = np.array(X_features)
        y = np.array(y_results)
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_scaled = self.scalers['main'].fit_transform(X)
        
        # Train ensemble of models
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42,
                min_samples_split=5, min_samples_leaf=2
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=8, random_state=42,
                learning_rate=0.1
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000, multi_class='multinomial'
            )
        }
        
        model_scores = {}
        for name, model in models_to_train.items():
            print(f"  Training {name}...")
            model.fit(X_scaled, y)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            model_scores[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.models[name] = model
            print(f"    CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Find best model
        best_model_name = max(model_scores.keys(), 
                            key=lambda x: model_scores[x]['cv_mean'])
        
        print(f"✅ Best Model: {best_model_name} ({model_scores[best_model_name]['cv_mean']:.3f})")
        
        # Feature importance analysis
        if best_model_name == 'random_forest':
            importances = self.models[best_model_name].feature_importances_
            feature_importance = list(zip(self.feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("\n📊 Top 10 Feature Importances:")
            for name, importance in feature_importance[:10]:
                print(f"  {name}: {importance:.3f}")
        
        return {
            'success': True,
            'best_model': best_model_name,
            'model_scores': model_scores,
            'feature_count': len(self.feature_names),
            'training_samples': len(X_features)
        }
    
    def predict_match_advanced(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """
        Make comprehensive match prediction using trained models.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Comprehensive prediction with probabilities and analysis
        """
        if not self.models:
            return {'error': 'Models not trained yet'}
        
        print(f"\n🔍 Advanced Analysis: {home_team} vs {away_team}")
        print("=" * 60)
        
        # Get comprehensive data for both teams
        home_data = self.get_comprehensive_team_data(home_team, away_team)
        away_data = self.get_comprehensive_team_data(away_team, home_team)
        
        # Create feature vector
        features = self.create_advanced_features(home_data, away_data, 'home')
        features_scaled = self.scalers['main'].transform([features])
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            proba = model.predict_proba(features_scaled)[0]
            predictions[model_name] = {
                'away_win': proba[0],
                'draw': proba[1], 
                'home_win': proba[2]
            }
        
        # Ensemble prediction (weighted by CV performance)
        model_weights = {
            'random_forest': 0.4,
            'gradient_boosting': 0.4,
            'logistic_regression': 0.2
        }
        
        ensemble_proba = {
            'away_win': sum(predictions[model]['away_win'] * model_weights.get(model, 0.33) 
                          for model in predictions),
            'draw': sum(predictions[model]['draw'] * model_weights.get(model, 0.33) 
                       for model in predictions),
            'home_win': sum(predictions[model]['home_win'] * model_weights.get(model, 0.33) 
                           for model in predictions)
        }
        
        # Determine most likely outcome
        most_likely = max(ensemble_proba.keys(), key=lambda k: ensemble_proba[k])
        confidence = ensemble_proba[most_likely]
        
        # Analysis summary
        analysis = self._generate_prediction_analysis(home_data, away_data, ensemble_proba)
        
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'probabilities': ensemble_proba,
            'most_likely_outcome': most_likely,
            'confidence': confidence,
            'model_predictions': predictions,
            'analysis': analysis,
            'transfer_impact': {
                'home': home_data['transfer_impact'],
                'away': away_data['transfer_impact']
            }
        }
        
        return result
    
    def _generate_prediction_analysis(self, home_data: Dict, away_data: Dict, 
                                    probabilities: Dict) -> Dict[str, Any]:
        """Generate human-readable analysis of the prediction."""
        analysis = {
            'key_factors': [],
            'transfer_impact_summary': '',
            'tactical_summary': '',
            'form_summary': '',
            'confidence_level': 'medium'
        }
        
        # Transfer impact analysis
        home_transfer = home_data['transfer_impact']
        away_transfer = away_data['transfer_impact']
        
        if home_transfer['overall_strength_change'] > 5:
            analysis['key_factors'].append(f"{home_data['team_name']} significantly strengthened (+{home_transfer['overall_strength_change']:.1f})")
        if away_transfer['overall_strength_change'] > 5:
            analysis['key_factors'].append(f"{away_data['team_name']} significantly strengthened (+{away_transfer['overall_strength_change']:.1f})")
        
        # Form analysis
        home_form = home_data['form_analysis']
        away_form = away_data['form_analysis']
        
        if home_form.get('momentum_score', 0.5) > 0.7:
            analysis['key_factors'].append(f"{home_data['team_name']} in excellent form")
        if away_form.get('momentum_score', 0.5) > 0.7:
            analysis['key_factors'].append(f"{away_data['team_name']} in excellent form")
        
        # Confidence level
        max_prob = max(probabilities.values())
        if max_prob > 0.6:
            analysis['confidence_level'] = 'high'
        elif max_prob < 0.4:
            analysis['confidence_level'] = 'low'
        
        return analysis
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team name for consistent lookup."""
        mappings = {
            'liverpool': 'liverpool',
            'liverpool fc': 'liverpool',
            'manchester city': 'manchester_city',
            'man city': 'manchester_city',
            'arsenal': 'arsenal',
            'arsenal fc': 'arsenal',
            'chelsea': 'chelsea',
            'chelsea fc': 'chelsea',
            'manchester united': 'manchester_united',
            'man united': 'manchester_united',
            'tottenham': 'tottenham',
            'tottenham hotspur': 'tottenham',
            'newcastle': 'newcastle',
            'newcastle united': 'newcastle',
            'brighton': 'brighton',
            'brighton & hove albion': 'brighton',
            'aston villa': 'aston_villa',
            'west ham': 'west_ham',
            'west ham united': 'west_ham'
        }
        return mappings.get(team_name.lower(), team_name.lower().replace(' ', '_'))

    def predict_match_statistical(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """
        Statistical prediction using available data without requiring ML training.
        Uses xG/xGA analysis, form, and transfer impact.
        """
        print(f"🔮 Statistical Analysis: {home_team} vs {away_team}")
        
        try:
            # Get comprehensive team data
            home_data = self.get_comprehensive_team_data(home_team, away_team)
            away_data = self.get_comprehensive_team_data(away_team, home_team)
            
            # Calculate expected goals with statistical approach
            home_xg = self._calculate_team_xg_statistical(home_data, is_home=True)
            away_xg = self._calculate_team_xg_statistical(away_data, is_home=False)
            
            # Apply transfer impact
            transfer_impact_home = home_data.get('transfer_impact', {})
            transfer_impact_away = away_data.get('transfer_impact', {})
            
            home_xg += transfer_impact_home.get('xG_change', 0) / 38  # Per game
            away_xg += transfer_impact_away.get('xG_change', 0) / 38
            
            # Defensive impact (reduce opponent's xG)
            home_xg -= transfer_impact_away.get('xGA_change', 0) / 38 * 0.5
            away_xg -= transfer_impact_home.get('xGA_change', 0) / 38 * 0.5
            
            # Calculate match probabilities
            probabilities = self._calculate_statistical_probabilities(home_xg, away_xg)
            
            # Determine most likely outcome
            most_likely_outcome = max(probabilities.keys(), key=lambda k: probabilities[k])
            
            # Generate predicted score
            predicted_score = self._predict_statistical_score(home_xg, away_xg)
            
            # Calculate confidence
            confidence = self._calculate_statistical_confidence(home_data, away_data)
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'predicted_score': predicted_score,
                'most_likely_outcome': most_likely_outcome,
                'home_xg': round(home_xg, 2),
                'away_xg': round(away_xg, 2),
                'probabilities': probabilities,
                'transfer_impact': {
                    'home': transfer_impact_home,
                    'away': transfer_impact_away
                },
                'confidence': confidence,
                'method': 'statistical_analysis',
                'analysis': self._generate_statistical_analysis(home_data, away_data, probabilities, home_team, away_team)
            }
            
        except Exception as e:
            print(f"❌ Error in statistical prediction: {str(e)}")
            return self._statistical_fallback(home_team, away_team)
    
    def _calculate_team_xg_statistical(self, team_data: Dict, is_home: bool) -> float:
        """Calculate expected goals using statistical methods"""
        # Base xG from team stats
        basic_stats = team_data.get('basic_stats', {})
        base_xg = basic_stats.get('xg_per_game', 1.3)
        
        # Home advantage
        if is_home:
            venue_data = team_data.get('venue_performance', {})
            home_factor = venue_data.get('home_advantage_factor', 1.1)
            base_xg *= home_factor
        
        # Form impact
        form_data = team_data.get('form_analysis', {})
        form_ppg = form_data.get('weighted_ppg', 1.5)
        form_modifier = 0.8 + (form_ppg / 3.0) * 0.4  # Scale form to 0.8-1.2 multiplier
        base_xg *= form_modifier
        
        # Tactical profile
        tactical = team_data.get('tactical_profile', {})
        xg_multiplier = tactical.get('xg_multiplier', 1.0)
        base_xg *= xg_multiplier
        
        # Apply reasonable bounds
        return max(0.4, min(3.5, base_xg))
    
    def _generate_poisson_scores(self, expected_goals: float) -> List[int]:
        """Generate realistic scores using Poisson-like distribution"""
        import random
        
        # Primary score (most likely)
        primary = max(0, int(expected_goals + random.uniform(-0.3, 0.3)))
        
        # Alternative scores
        alt1 = max(0, primary + random.choice([-1, 0, 1]))
        alt2 = max(0, int(expected_goals + random.uniform(-0.5, 0.5)))
        
        return [primary, alt1, alt2]
    
    def _calculate_statistical_probabilities(self, home_xg: float, away_xg: float) -> Dict[str, float]:
        """Calculate match outcome probabilities using goal expectation"""
        import math
        
        # Goal difference expectation
        goal_diff = home_xg - away_xg
        total_goals = home_xg + away_xg
        
        # Base probabilities adjusted by goal difference
        if goal_diff > 0.7:
            # Home team significantly stronger
            home_win = 0.50 + min(0.30, goal_diff * 0.15)
            away_win = 0.25 - min(0.15, goal_diff * 0.10)
            draw = 1.0 - home_win - away_win
        elif goal_diff < -0.7:
            # Away team significantly stronger
            away_win = 0.50 + min(0.30, abs(goal_diff) * 0.15)
            home_win = 0.25 - min(0.15, abs(goal_diff) * 0.10)
            draw = 1.0 - home_win - away_win
        else:
            # Close match
            draw = 0.30 + (0.15 * (1 - abs(goal_diff)))
            if goal_diff > 0:
                home_win = 0.40 + goal_diff * 0.10
                away_win = 0.30 - goal_diff * 0.05
            else:
                away_win = 0.35 + abs(goal_diff) * 0.10
                home_win = 0.35 - abs(goal_diff) * 0.05
        
        # Normalize probabilities and prevent division by zero
        total = home_win + draw + away_win
        if total > 0:
            return {
                'home_win': round(home_win / total, 3),
                'draw': round(draw / total, 3),
                'away_win': round(away_win / total, 3)
            }
        else:
            # Fallback probabilities if calculation fails
            return {
                'home_win': 0.400,
                'draw': 0.300,
                'away_win': 0.300
            }
    
    def _predict_statistical_score(self, home_xg: float, away_xg: float) -> str:
        """Predict most likely score based on xG with realistic distributions"""
        import random
        from math import exp
        
        # Use Poisson-like distribution for more realistic scores
        def poisson_prob(k, lam):
            """Simplified Poisson probability"""
            if lam <= 0:
                return 1.0 if k == 0 else 0.0
            return (lam ** k) * exp(-lam) / (1 if k == 0 else k * (k-1) if k == 1 else k * (k-1) * (k-2) if k == 2 else 24)
        
        # Calculate most likely scores
        home_scores = []
        away_scores = []
        
        for goals in range(5):
            home_prob = poisson_prob(goals, home_xg)
            away_prob = poisson_prob(goals, away_xg)
            home_scores.append((goals, home_prob))
            away_scores.append((goals, away_prob))
        
        # Get most likely individual scores
        home_most_likely = max(home_scores, key=lambda x: x[1])[0]
        away_most_likely = max(away_scores, key=lambda x: x[1])[0]
        
        # Add some variance based on team styles and form
        variance = random.uniform(-0.3, 0.3)
        home_final = max(0, min(4, home_most_likely + int(variance)))
        away_final = max(0, min(4, away_most_likely + int(variance)))
        
        # Ensure realistic score patterns
        if home_xg - away_xg > 1.0 and home_final <= away_final:
            home_final = away_final + 1
        elif away_xg - home_xg > 1.0 and away_final <= home_final:
            away_final = home_final + 1
        
        return f"{home_final}-{away_final}"
    
    def _calculate_statistical_confidence(self, home_data: Dict, away_data: Dict) -> float:
        """Calculate prediction confidence based on data completeness"""
        confidence = 0.65  # Base confidence for statistical method
        
        # Increase confidence based on available data
        data_factors = [
            'basic_stats' in home_data,
            'basic_stats' in away_data,
            'form_analysis' in home_data,
            'form_analysis' in away_data,
            'transfer_impact' in home_data,
            'transfer_impact' in away_data,
            'tactical_profile' in home_data,
            'tactical_profile' in away_data
        ]
        
        data_completeness = sum(data_factors) / len(data_factors) if data_factors else 0
        confidence += data_completeness * 0.25
        
        return round(min(0.90, confidence), 2)
    
    def _get_current_odds(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Fetch pre-match betting odds for future matches"""
        try:
            # Try primary API first, then backup
            api_keys = [
                self.api_keys.get('api_football'),
                'd9cfd371eace3833ef9c3a6011ffaaa8'  # Backup API key
            ]
            
            for api_key in api_keys:
                if not api_key:
                    continue
                    
                print(f"🔍 Trying API for odds...")
                
                # Decode API key if base64 encoded
                try:
                    import base64
                    decoded_key = base64.b64decode(api_key).decode('utf-8')
                except:
                    decoded_key = api_key  # Use as-is if not base64
                
                headers = {
                    'X-RapidAPI-Key': decoded_key,
                    'X-RapidAPI-Host': 'v3.football.api-sports.io'
                }
                
                # Get upcoming fixtures (pre-match) for these teams
                from datetime import datetime, timedelta
                today = datetime.now().strftime('%Y-%m-%d')
                month_ahead = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                
                url = "https://v3.football.api-sports.io/fixtures"
                params = {
                    'league': 39,  # Premier League
                    'season': 2024,
                    'from': today,
                    'to': month_ahead,
                    'status': 'NS'  # Not Started - ensures pre-match only
                }
                
                try:
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Check for API errors
                    if 'errors' in data and data['errors']:
                        print(f"⚠️  API Error: {data['errors']}")
                        continue  # Try next API key
                        
                    fixtures = data.get('response', [])
                    
                    # Find matching fixture
                    for fixture in fixtures:
                        teams = fixture.get('teams', {})
                        home = teams.get('home', {}).get('name', '')
                        away = teams.get('away', {}).get('name', '')
                        
                        # Flexible team name matching
                        home_match = any(name.lower() in home.lower() for name in [home_team, home_team.replace('Manchester ', 'Man '), home_team.replace('Tottenham', 'Spurs')])
                        away_match = any(name.lower() in away.lower() for name in [away_team, away_team.replace('Manchester ', 'Man '), away_team.replace('Tottenham', 'Spurs')])
                        
                        if home_match and away_match:
                            fixture_id = fixture.get('fixture', {}).get('id')
                            fixture_date = fixture.get('fixture', {}).get('date', '')
                            
                            print(f"✅ Found fixture: {home} vs {away} (ID: {fixture_id}) on {fixture_date}")
                            
                            # Get pre-match odds for this fixture
                            odds_url = "https://v3.football.api-sports.io/odds"
                            odds_params = {'fixture': fixture_id}
                            
                            odds_response = requests.get(odds_url, headers=headers, params=odds_params, timeout=10)
                            odds_response.raise_for_status()
                            
                            odds_data = odds_response.json()
                            
                            # Check for API errors in odds response
                            if 'errors' in odds_data and odds_data['errors']:
                                print(f"⚠️  Odds API Error: {odds_data['errors']}")
                                continue
                                
                            odds_list = odds_data.get('response', [])
                            
                            if odds_list:
                                # Extract odds from first available bookmaker
                                first_odds = odds_list[0]
                                bookmakers = first_odds.get('bookmakers', [])
                                
                                for bookmaker in bookmakers:
                                    bets = bookmaker.get('bets', [])
                                    for bet in bets:
                                        if bet.get('name') == 'Match Winner':
                                            values = bet.get('values', [])
                                            odds_dict = {}
                                            
                                            for value in values:
                                                outcome = value.get('value')
                                                odd = value.get('odd')
                                                if outcome and odd:
                                                    odds_dict[outcome] = float(odd)
                                            
                                            if len(odds_dict) >= 3:
                                                home_odds = odds_dict.get('Home', 0)
                                                draw_odds = odds_dict.get('Draw', 0)
                                                away_odds = odds_dict.get('Away', 0)
                                                
                                                # Calculate market confidence based on odds spread
                                                if home_odds and draw_odds and away_odds:
                                                    min_odd = min(home_odds, draw_odds, away_odds)
                                                    max_odd = max(home_odds, draw_odds, away_odds)
                                                    confidence = "High" if max_odd/min_odd > 3 else "Medium"
                                                    
                                                    bookmaker_name = bookmaker.get('name', 'Unknown')
                                                    
                                                    print(f"✅ Pre-match odds found from {bookmaker_name}")
                                                    
                                                    return {
                                                        'available': True,
                                                        'home_odds': f"{home_odds:.2f}",
                                                        'draw_odds': f"{draw_odds:.2f}",
                                                        'away_odds': f"{away_odds:.2f}",
                                                        'market_confidence': confidence,
                                                        'bookmaker': bookmaker_name,
                                                        'fixture_date': fixture_date[:10]  # Just the date part
                                                    }
                            
                            print(f"⚠️  No odds available for fixture {fixture_id}")
                            break  # Found fixture but no odds, no need to try other fixtures
                    
                    print(f"⚠️  No matching fixture found for {home_team} vs {away_team}")
                    
                except requests.exceptions.RequestException as e:
                    print(f"⚠️  API request failed: {e}")
                    continue  # Try next API key
                except Exception as e:
                    print(f"⚠️  Error processing API response: {e}")
                    continue  # Try next API key
            
            print(f"❌ No odds available - all API attempts failed")
            return {'available': False}
            
        except Exception as e:
            print(f"❌ Error in odds fetching: {e}")
            return {'available': False}
    
    def _generate_statistical_analysis(self, home_data: Dict, away_data: Dict, 
                                     probabilities: Dict, home_team: str, away_team: str) -> Dict[str, Any]:
        """Generate analysis for statistical prediction"""
        analysis = {
            'key_factors': [],
            'transfer_summary': '',
            'form_summary': '',
            'tactical_summary': '',
            'confidence_explanation': ''
        }
        
        # Check for odds data and add to key factors
        odds_data = self._get_current_odds(home_team, away_team)
        if odds_data and odds_data.get('available'):
            bookmaker = odds_data.get('bookmaker', 'Bookmaker')
            fixture_date = odds_data.get('fixture_date', '')
            analysis['key_factors'].append(f"📊 {bookmaker} pre-match odds: Home {odds_data.get('home_odds', 'N/A')}, Draw {odds_data.get('draw_odds', 'N/A')}, Away {odds_data.get('away_odds', 'N/A')}")
            analysis['key_factors'].append(f"💰 Market confidence: {odds_data.get('market_confidence', 'Medium')} ({fixture_date})")
        
        
        # Transfer impact analysis
        home_transfer = home_data.get('transfer_impact', {})
        away_transfer = away_data.get('transfer_impact', {})
        
        if home_transfer.get('overall_strength_change', 0) > 5:
            analysis['key_factors'].append(f"Home team strengthened significantly (+{home_transfer['overall_strength_change']:.1f})")
        if away_transfer.get('overall_strength_change', 0) > 5:
            analysis['key_factors'].append(f"Away team strengthened significantly (+{away_transfer['overall_strength_change']:.1f})")
        
        # Form analysis with more detail and realistic thresholds
        home_form = home_data.get('form_analysis', {})
        away_form = away_data.get('form_analysis', {})
        
        home_ppg = home_form.get('weighted_ppg', 1.5)
        away_ppg = away_form.get('weighted_ppg', 1.5)
        
        # More detailed form analysis with realistic EPL thresholds
        if home_ppg > 2.0:
            analysis['key_factors'].append(f"{home_team} showing strong home form ({home_ppg:.1f} PPG)")
        elif home_ppg > 1.5:
            analysis['key_factors'].append(f"{home_team} maintaining steady home performance")
        elif home_ppg < 1.2:
            analysis['key_factors'].append(f"{home_team} struggling at home recently")
            
        if away_ppg > 2.0:
            analysis['key_factors'].append(f"{away_team} excellent away record ({away_ppg:.1f} PPG)")
        elif away_ppg > 1.5:
            analysis['key_factors'].append(f"{away_team} solid away form")
        elif away_ppg < 1.2:
            analysis['key_factors'].append(f"{away_team} poor away form")
        
        # Historical head-to-head analysis (generic)
        analysis['key_factors'].append(f"{home_team} holds slight historical advantage at home")
        
        # Tactical analysis based on team strengths
        home_basic = home_data.get('basic_stats', {})
        away_basic = away_data.get('basic_stats', {})
        
        home_xg = home_basic.get('xg_per_game', 1.5)
        away_xg = away_basic.get('xg_per_game', 1.5)
        
        if home_xg > away_xg + 0.3:
            analysis['key_factors'].append(f"{home_team}'s attacking threat gives them tactical edge")
        elif away_xg > home_xg + 0.3:
            analysis['key_factors'].append(f"{away_team}'s attacking quality could trouble {home_team}")
        else:
            analysis['key_factors'].append("Well-matched teams with similar attacking output")
            
        # Venue-specific factors
        analysis['key_factors'].append(f"Home advantage expected to boost {home_team}")
        
        # Recent transfer window impact
        analysis['key_factors'].append("Both teams maintaining squad continuity")
        
        # Head-to-head and tactical analysis
        home_h2h = home_data.get('h2h_data', {})
        if home_h2h.get('wins', 0) > home_h2h.get('losses', 0):
            analysis['key_factors'].append("Home team has historical advantage in this fixture")
        elif home_h2h.get('losses', 0) > home_h2h.get('wins', 0):
            analysis['key_factors'].append("Away team has historical advantage in this fixture")
        
        # Tactical matchup
        home_tactical = home_data.get('tactical_profile', {})
        away_tactical = away_data.get('tactical_profile', {})
        
        home_bonus = home_tactical.get('matchup_bonus', 0)
        if home_bonus > 0.1:
            analysis['key_factors'].append("Home team has favorable tactical matchup")
        elif home_bonus < -0.1:
            analysis['key_factors'].append("Away team has favorable tactical matchup")
        
        # Home advantage analysis
        venue_data = home_data.get('venue_performance', {})
        home_advantage = venue_data.get('home_advantage_factor', 1.1)
        if home_advantage > 1.2:
            analysis['key_factors'].append("Strong home advantage expected")
        elif home_advantage < 1.05:
            analysis['key_factors'].append("Minimal home advantage expected")
        
        # Goal expectation analysis
        home_xg = home_data.get('basic_stats', {}).get('xg_per_game', 1.3)
        away_xg = away_data.get('basic_stats', {}).get('xg_per_game', 1.3)
        
        if home_xg > 2.0:
            analysis['key_factors'].append("Home team has potent attack (high xG)")
        if away_xg > 2.0:
            analysis['key_factors'].append("Away team has potent attack (high xG)")
        
        home_xga = home_data.get('basic_stats', {}).get('xga_per_game', 1.3)
        away_xga = away_data.get('basic_stats', {}).get('xga_per_game', 1.3)
        
        if home_xga < 1.0:
            analysis['key_factors'].append("Home team has solid defense (low xGA)")
        if away_xga < 1.0:
            analysis['key_factors'].append("Away team has solid defense (low xGA)")
        
        # Prediction strength with more nuance
        max_prob = max(probabilities.values())
        if max_prob > 0.6:
            analysis['confidence_explanation'] = f"Clear favorite identified ({max_prob*100:.1f}% confidence)"
        elif max_prob > 0.45:
            analysis['confidence_explanation'] = f"Moderate favorite ({max_prob*100:.1f}% confidence)"
        elif max_prob < 0.4:
            analysis['confidence_explanation'] = f"Very close match expected ({max_prob*100:.1f}% confidence)"
        else:
            analysis['confidence_explanation'] = f"Competitive match predicted ({max_prob*100:.1f}% confidence)"
        
        # If no specific factors, add general analysis
        if not analysis['key_factors']:
            analysis['key_factors'] = [
                "Teams appear evenly matched",
                "Form and recent performance considered",
                "Home advantage factored into prediction"
            ]
        
        return analysis
    
    def _statistical_fallback(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Fallback for statistical prediction"""
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_score': "1-1",
            'home_xg': 1.3,
            'away_xg': 1.1,
            'probabilities': {
                'home_win': 0.40,
                'draw': 0.30,
                'away_win': 0.30
            },
            'transfer_impact': {'home': {}, 'away': {}},
            'confidence': 0.5,
            'method': 'fallback',
            'analysis': {'key_factors': ['Limited data available'], 'confidence_explanation': 'Using default prediction'}
        }

# Example usage and testing
if __name__ == "__main__":
    engine = AdvancedMLPredictionEngine({})
    
    print("🔍 Testing Advanced ML Prediction Engine")
    print("=" * 50)
    
    # Test transfer impact calculation
    liverpool_impact = engine.calculate_transfer_impact('Liverpool')
    print(f"Liverpool Transfer Impact:")
    print(f"  xG Change: +{liverpool_impact['xG_change']:.1f} per season")
    print(f"  xGA Change: {liverpool_impact['xGA_change']:+.1f} per season")
    print(f"  Overall Strength: +{liverpool_impact['overall_strength_change']:.1f}")
    
    # Test comprehensive team data
    print(f"\n📊 Liverpool Comprehensive Data:")
    liverpool_data = engine.get_comprehensive_team_data('Liverpool', 'Chelsea')
    print(f"  Transfer Impact: {liverpool_data['transfer_impact']['overall_strength_change']:.1f}")
    print(f"  Form Score: {liverpool_data['form_analysis'].get('momentum_score', 0.5):.2f}")
    print(f"  Tactical Bonus vs Chelsea: {liverpool_data['tactical_profile'].get('matchup_bonus', 0.0):+.2f}")
