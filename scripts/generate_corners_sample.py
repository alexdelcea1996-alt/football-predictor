"""
Generate sample corners data for testing.

Creates realistic corner statistics based on Premier League patterns.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from datetime import datetime, timedelta
import pandas as pd
from rich.console import Console

console = Console()


def generate_corners_data(n_matches: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic corners data based on real patterns.
    
    Real Premier League stats (2023/24):
    - Average total corners: ~10.2 per match
    - Home team average: ~5.3 corners
    - Away team average: ~4.9 corners
    - Over 9.5 rate: ~55%
    """
    random.seed(seed)
    
    teams = [
        # High corner teams (attacking style)
        {"name": "Manchester City", "corner_rate": 6.5, "concede_rate": 4.0},
        {"name": "Liverpool", "corner_rate": 6.2, "concede_rate": 4.5},
        {"name": "Arsenal", "corner_rate": 6.0, "concede_rate": 4.2},
        {"name": "Chelsea", "corner_rate": 5.8, "concede_rate": 4.8},
        
        # Medium corner teams
        {"name": "Tottenham", "corner_rate": 5.5, "concede_rate": 5.0},
        {"name": "Newcastle", "corner_rate": 5.3, "concede_rate": 4.8},
        {"name": "Brighton", "corner_rate": 5.4, "concede_rate": 5.2},
        {"name": "Aston Villa", "corner_rate": 5.2, "concede_rate": 5.0},
        {"name": "West Ham", "corner_rate": 5.0, "concede_rate": 5.3},
        {"name": "Manchester United", "corner_rate": 5.1, "concede_rate": 5.0},
        
        # Lower corner teams (defensive style)
        {"name": "Crystal Palace", "corner_rate": 4.5, "concede_rate": 5.5},
        {"name": "Wolves", "corner_rate": 4.3, "concede_rate": 5.2},
        {"name": "Everton", "corner_rate": 4.2, "concede_rate": 5.8},
        {"name": "Bournemouth", "corner_rate": 4.4, "concede_rate": 5.5},
        {"name": "Fulham", "corner_rate": 4.6, "concede_rate": 5.3},
        {"name": "Brentford", "corner_rate": 4.8, "concede_rate": 5.0},
        {"name": "Nottingham Forest", "corner_rate": 4.0, "concede_rate": 6.0},
        {"name": "Burnley", "corner_rate": 3.8, "concede_rate": 6.5},
        {"name": "Sheffield United", "corner_rate": 3.5, "concede_rate": 6.8},
        {"name": "Luton", "corner_rate": 3.6, "concede_rate": 6.2},
    ]
    
    team_dict = {t["name"]: t for t in teams}
    team_names = [t["name"] for t in teams]
    
    rows = []
    start_date = datetime(2023, 8, 11)
    
    for i in range(n_matches):
        # Pick two random teams
        home_team, away_team = random.sample(team_names, 2)
        
        # Match date
        match_date = start_date + timedelta(days=random.randint(0, 280))
        
        # Calculate expected corners based on team rates
        home_data = team_dict[home_team]
        away_data = team_dict[away_team]
        
        # Expected corners with home advantage
        exp_home = (home_data["corner_rate"] * 1.1 + away_data["concede_rate"]) / 2
        exp_away = (away_data["corner_rate"] * 0.9 + home_data["concede_rate"]) / 2
        
        # Generate actual corners with variance
        home_corners = max(0, int(random.gauss(exp_home, 2.0)))
        away_corners = max(0, int(random.gauss(exp_away, 2.0)))
        
        # Goals (correlated with corners somewhat)
        home_goals = random.choices([0, 1, 2, 3, 4], weights=[20, 35, 25, 15, 5])[0]
        away_goals = random.choices([0, 1, 2, 3], weights=[30, 35, 25, 10])[0]
        
        rows.append({
            "date": match_date.strftime("%Y-%m-%d"),
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_corners": home_corners,
            "away_corners": away_corners,
            "total_corners": home_corners + away_corners,
            "league": "premier_league",
            "season": "2023/24",
        })
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    return df


def main():
    """Generate and save sample corners data."""
    console.print("[bold blue]Generating Sample Corners Data[/]")
    console.print("=" * 50)
    
    df = generate_corners_data(n_matches=500)
    
    # Stats
    console.print(f"\nGenerated {len(df)} matches")
    console.print(f"Average total corners: {df['total_corners'].mean():.1f}")
    console.print(f"Average home corners: {df['home_corners'].mean():.1f}")
    console.print(f"Average away corners: {df['away_corners'].mean():.1f}")
    console.print(f"Over 9.5 rate: {(df['total_corners'] > 9.5).mean():.1%}")
    console.print(f"Over 10.5 rate: {(df['total_corners'] > 10.5).mean():.1%}")
    
    # Save
    output_path = Path("data/corners_sample.csv")
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    console.print(f"\n[green]Saved to: {output_path}[/]")


if __name__ == "__main__":
    main()
