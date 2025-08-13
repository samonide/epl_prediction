"""
Transfer and squad validation utilities for EPL predictions.
Contains known transfers and validation logic.
"""

KNOWN_TRANSFERS_2024_25 = {
    # Major transfers affecting predictions
    "joão pedro": {
        "from_team": "brighton", 
        "to_team": "chelsea", 
        "season": "2024-25",
        "transfer_date": "2024-08-29",
        "fee": "£25m",
        "impact": "high"  # Key goalscorer
    },
    "joao pedro": {
        "from_team": "brighton", 
        "to_team": "chelsea", 
        "season": "2024-25",
        "transfer_date": "2024-08-29",
        "fee": "£25m",
        "impact": "high"
    },
    "raheem sterling": {
        "from_team": "chelsea",
        "to_team": "arsenal",
        "season": "2024-25", 
        "transfer_date": "2024-08-30",
        "fee": "loan",
        "impact": "high"
    },
    "jadon sancho": {
        "from_team": "manchester united",
        "to_team": "chelsea",
        "season": "2024-25",
        "transfer_date": "2024-08-30", 
        "fee": "loan",
        "impact": "medium"
    },
    "ivan toney": {
        "from_team": "brentford",
        "to_team": "al-ahli",
        "season": "2024-25",
        "transfer_date": "2024-08-30",
        "fee": "£40m",
        "impact": "high"
    },
    "scott mctominay": {
        "from_team": "manchester united",
        "to_team": "napoli",
        "season": "2024-25",
        "transfer_date": "2024-08-30",
        "fee": "£25.7m",
        "impact": "medium"
    },
    "eddie nketiah": {
        "from_team": "arsenal",
        "to_team": "crystal palace",
        "season": "2024-25",
        "transfer_date": "2024-08-30",
        "fee": "£30m",
        "impact": "medium"
    },
    "conor gallagher": {
        "from_team": "chelsea",
        "to_team": "atletico madrid",
        "season": "2024-25",
        "transfer_date": "2024-08-14",
        "fee": "£36m",
        "impact": "medium"
    },
    # Add more transfers as they happen
}

# Team name mappings for transfer validation
TEAM_NAME_MAPPINGS = {
    "brighton": ["brighton & hove albion", "brighton", "brighton & ha"],
    "chelsea": ["chelsea", "chelsea fc"],
    "arsenal": ["arsenal", "arsenal fc"],
    "manchester united": ["manchester united", "man united", "man utd"],
    "manchester city": ["manchester city", "man city"],
    "liverpool": ["liverpool", "liverpool fc"],
    "tottenham": ["tottenham hotspur", "tottenham", "spurs"],
    "newcastle": ["newcastle united", "newcastle"],
    "west ham": ["west ham united", "west ham"],
    "aston villa": ["aston villa", "villa"],
    "crystal palace": ["crystal palace", "palace"],
    "brentford": ["brentford", "brentford fc"],
    "fulham": ["fulham", "fulham fc"],
    "wolves": ["wolverhampton wanderers", "wolves", "wolverhampton"],
    "everton": ["everton", "everton fc"],
    "nottingham forest": ["nottingham forest", "nott'm forest", "forest"],
    "leicester": ["leicester city", "leicester"],
    "bournemouth": ["afc bournemouth", "bournemouth"],
    "southampton": ["southampton", "southampton fc"],
    "ipswich": ["ipswich town", "ipswich"],
}

def normalize_team_name(team_name: str) -> str:
    """Normalize team name for comparison."""
    team_lower = team_name.lower().strip()
    
    for canonical, variants in TEAM_NAME_MAPPINGS.items():
        if team_lower in variants:
            return canonical
    
    return team_lower

def check_player_transfer_status(player_name: str, current_team: str, season: str = "2024-25") -> dict:
    """
    Check if a player has been transferred and should not appear in predictions.
    
    Returns:
        dict: {
            'transferred': bool,
            'transfer_info': dict or None,
            'should_exclude': bool,
            'warning_message': str or None
        }
    """
    player_lower = player_name.lower().strip()
    current_team_normalized = normalize_team_name(current_team)
    
    result = {
        'transferred': False,
        'transfer_info': None,
        'should_exclude': False,
        'warning_message': None
    }
    
    if player_lower in KNOWN_TRANSFERS_2024_25:
        transfer_info = KNOWN_TRANSFERS_2024_25[player_lower]
        
        # Check if this transfer affects the current team
        from_team_normalized = normalize_team_name(transfer_info['from_team'])
        to_team_normalized = normalize_team_name(transfer_info['to_team'])
        
        result['transferred'] = True
        result['transfer_info'] = transfer_info
        
        # Should exclude if player transferred away from current team
        if from_team_normalized == current_team_normalized and transfer_info['season'] == season:
            result['should_exclude'] = True
            result['warning_message'] = (
                f"⚠️ {player_name} transferred from {transfer_info['from_team'].title()} "
                f"to {transfer_info['to_team'].title()} in {transfer_info['season']} "
                f"- excluding from predictions"
            )
        
        # Should include if player transferred TO current team
        elif to_team_normalized == current_team_normalized and transfer_info['season'] == season:
            result['should_exclude'] = False
            result['warning_message'] = (
                f"✅ {player_name} transferred from {transfer_info['from_team'].title()} "
                f"to {transfer_info['to_team'].title()} in {transfer_info['season']} "
                f"- now available for {current_team.title()}"
            )
    
    return result

def get_transfer_impact_summary(team_name: str, season: str = "2024-25") -> dict:
    """
    Get summary of transfers affecting a team's scoring potential.
    
    Returns:
        dict: {
            'players_lost': list,
            'players_gained': list,
            'net_goal_impact': float,  # Estimated goals per season impact
            'key_changes': list
        }
    """
    team_normalized = normalize_team_name(team_name)
    
    summary = {
        'players_lost': [],
        'players_gained': [],
        'net_goal_impact': 0.0,
        'key_changes': []
    }
    
    for player, transfer_info in KNOWN_TRANSFERS_2024_25.items():
        if transfer_info['season'] != season:
            continue
            
        from_team = normalize_team_name(transfer_info['from_team'])
        to_team = normalize_team_name(transfer_info['to_team'])
        
        # Estimate goal impact based on transfer impact level
        impact_values = {'high': 8, 'medium': 4, 'low': 1}
        goal_impact = impact_values.get(transfer_info.get('impact', 'low'), 1)
        
        if from_team == team_normalized:
            # Player left the team
            summary['players_lost'].append({
                'player': player.title(),
                'to_team': transfer_info['to_team'].title(),
                'impact': transfer_info.get('impact', 'low'),
                'estimated_goals_lost': goal_impact
            })
            summary['net_goal_impact'] -= goal_impact
            
        elif to_team == team_normalized:
            # Player joined the team
            summary['players_gained'].append({
                'player': player.title(),
                'from_team': transfer_info['from_team'].title(),
                'impact': transfer_info.get('impact', 'low'),
                'estimated_goals_gained': goal_impact
            })
            summary['net_goal_impact'] += goal_impact
    
    # Generate key changes summary
    if summary['players_lost']:
        high_impact_lost = [p for p in summary['players_lost'] if p['impact'] == 'high']
        if high_impact_lost:
            summary['key_changes'].append(f"Lost key goalscorer(s): {', '.join(p['player'] for p in high_impact_lost)}")
    
    if summary['players_gained']:
        high_impact_gained = [p for p in summary['players_gained'] if p['impact'] == 'high']
        if high_impact_gained:
            summary['key_changes'].append(f"Gained key player(s): {', '.join(p['player'] for p in high_impact_gained)}")
    
    if abs(summary['net_goal_impact']) >= 5:
        direction = "stronger" if summary['net_goal_impact'] > 0 else "weaker"
        summary['key_changes'].append(f"Squad significantly {direction} in attack")
    
    return summary

def validate_squad_for_prediction(team_name: str, players_list: list, season: str = "2024-25") -> dict:
    """
    Validate entire squad list for transfers and provide updated predictions.
    
    Args:
        team_name: Team name
        players_list: List of player names or player dicts
        season: Season to check
        
    Returns:
        dict: {
            'valid_players': list,
            'transferred_players': list,
            'warnings': list,
            'team_transfer_summary': dict
        }
    """
    valid_players = []
    transferred_players = []
    warnings = []
    
    for player in players_list:
        if isinstance(player, dict):
            player_name = player.get('name', '')
        else:
            player_name = str(player)
            
        if not player_name:
            continue
            
        transfer_status = check_player_transfer_status(player_name, team_name, season)
        
        if transfer_status['should_exclude']:
            transferred_players.append({
                'player': player_name,
                'transfer_info': transfer_status['transfer_info']
            })
            if transfer_status['warning_message']:
                warnings.append(transfer_status['warning_message'])
        else:
            valid_players.append(player)
            if transfer_status['warning_message'] and not transfer_status['should_exclude']:
                warnings.append(transfer_status['warning_message'])
    
    # Get team transfer summary
    team_summary = get_transfer_impact_summary(team_name, season)
    
    return {
        'valid_players': valid_players,
        'transferred_players': transferred_players,
        'warnings': warnings,
        'team_transfer_summary': team_summary
    }

if __name__ == "__main__":
    # Test the transfer validation
    print("Testing transfer validation...")
    
    # Test João Pedro case
    result = check_player_transfer_status("João Pedro", "Brighton", "2024-25")
    print("João Pedro at Brighton:", result)
    
    result = check_player_transfer_status("João Pedro", "Chelsea", "2024-25")
    print("João Pedro at Chelsea:", result)
    
    # Test team impact
    brighton_impact = get_transfer_impact_summary("Brighton", "2024-25")
    print("Brighton transfer impact:", brighton_impact)
    
    chelsea_impact = get_transfer_impact_summary("Chelsea", "2024-25")
    print("Chelsea transfer impact:", chelsea_impact)
