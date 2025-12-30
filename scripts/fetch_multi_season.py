"""
Fetch historical data from multiple seasons.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
import time

console = Console()


def main():
    """Fetch data from multiple seasons."""
    from football_predictor.data.football_data_client import FootballDataClient
    
    API_KEY = "1b3e20f99212447dba6016eb1810f054"
    client = FootballDataClient(API_KEY)
    
    console.print("[bold blue]Fetching Multi-Season Historical Data[/]")
    console.print("=" * 50)
    
    leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
    seasons = [2021, 2022, 2023]  # 2021/22, 2022/23, 2023/24
    
    all_matches = []
    
    for season in seasons:
        console.print(f"\n[cyan]Season {season}/{season+1}:[/]")
        
        for league in leagues:
            try:
                df = client.get_matches(league, season=season, status="FINISHED")
                if not df.empty:
                    all_matches.append(df)
                    console.print(f"  ✓ {league}: {len(df)} matches")
                time.sleep(6)  # Rate limit: 10 req/min for free tier
            except Exception as e:
                console.print(f"  ✗ {league}: {e}")
                time.sleep(10)
    
    if not all_matches:
        console.print("[red]No matches fetched![/]")
        return
    
    # Combine and sort
    df = pd.concat(all_matches, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    
    console.print(f"\n[bold green]Total matches: {len(df)}[/]")
    
    # Summary by season
    df["season_year"] = df["date"].dt.year.apply(
        lambda y: f"{y-1}/{y}" if pd.Timestamp.now().month < 8 else f"{y}/{y+1}"
    )
    
    console.print("\n[dim]Matches per league:[/]")
    for league in leagues:
        count = len(df[df["league"] == league])
        console.print(f"  {league}: {count}")
    
    # Save
    output_path = Path("data/matches_multi_season.csv")
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"\n[green]Saved to: {output_path}[/]")


if __name__ == "__main__":
    main()
