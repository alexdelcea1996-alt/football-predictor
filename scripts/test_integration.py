"""
Test script for Football Predictor with real data.

Uses football-data.org API to fetch matches and test the prediction pipeline.
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def test_api_connection():
    """Test API connectivity."""
    from football_predictor.data.football_data_client import FootballDataClient
    
    API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not API_KEY:
        raise SystemExit("Set FOOTBALL_DATA_API_KEY (free key at football-data.org)")
    
    console.print("\n[bold blue]Testing Football-Data.org API connection...[/]")
    
    client = FootballDataClient(API_KEY)
    
    # Test getting competitions
    try:
        competitions = client.get_competitions()
        console.print(f"  ✓ Connected! Found {len(competitions)} competitions")
        return client
    except Exception as e:
        console.print(f"  ✗ Connection failed: {e}")
        return None


def fetch_training_data(client):
    """Fetch historical match data."""
    console.print("\n[bold blue]Fetching historical matches (2023/24 season)...[/]")
    
    # Fetch from multiple leagues
    df = client.get_all_leagues_matches(
        leagues=["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"],
        season=2023,
    )
    
    console.print(f"\n[green]Total matches fetched: {len(df)}[/]")
    
    if len(df) > 0:
        # Show sample
        console.print("\n[dim]Sample matches:[/]")
        sample = df.head(5)[["date", "home_team", "away_team", "home_goals", "away_goals", "league"]]
        print(sample.to_string(index=False))
    
    return df


def test_feature_engineering(df):
    """Test feature engineering pipeline."""
    console.print("\n[bold blue]Testing feature engineering...[/]")
    
    from football_predictor.features.elo import EloRatingSystem
    from football_predictor.features.rolling_stats import RollingStatsCalculator
    
    # Test Elo
    elo = EloRatingSystem()
    df_elo = elo.process_matches(df.copy())
    console.print(f"  ✓ Elo features added: {['home_elo', 'away_elo', 'elo_diff']}")
    
    # Test rolling stats
    rolling = RollingStatsCalculator(windows=[5, 10])
    df_rolling = rolling.process_matches(df.copy())
    console.print(f"  ✓ Rolling stats added for windows [5, 10]")
    
    return df_elo


def test_prediction_pipeline(df):
    """Test the full prediction pipeline."""
    console.print("\n[bold blue]Testing prediction pipeline...[/]")
    
    from football_predictor.features.aggregator import FeatureAggregator
    from football_predictor.data.preprocessor import DataPreprocessor
    from football_predictor.evaluation.metrics import calculate_all_metrics
    
    # Prepare data
    preprocessor = DataPreprocessor()
    df = preprocessor.encode_outcome(df)
    
    # Compute features
    aggregator = FeatureAggregator()
    df = aggregator.process_matches(df)
    
    # Filter finished matches
    df = df[df["outcome"].notna()].reset_index(drop=True)
    console.print(f"  ✓ Processed {len(df)} finished matches")
    
    # Get feature matrix
    X, feature_names = aggregator.get_feature_matrix(df)
    y = df["outcome"].astype(int).values
    
    console.print(f"  ✓ Feature matrix shape: {X.shape}")
    console.print(f"  ✓ Features: {len(feature_names)}")
    
    # Show top features
    console.print(f"\n[dim]Sample features: {feature_names[:10]}...[/]")
    
    return X, y, feature_names, df


def main():
    """Run all tests."""
    console.print("[bold green]Football Predictor - Integration Test[/]")
    console.print("=" * 50)
    
    # Test API
    client = test_api_connection()
    if not client:
        console.print("\n[red]API connection failed. Exiting.[/]")
        return
    
    # Fetch data
    df = fetch_training_data(client)
    if len(df) < 100:
        console.print("\n[yellow]Warning: Less than 100 matches fetched. Results may be unreliable.[/]")
    
    # Save data for later use
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "matches_2023_24.csv", index=False)
    console.print(f"\n[dim]Data saved to: data/matches_2023_24.csv[/]")
    
    # Test feature engineering
    df_with_features = test_feature_engineering(df)
    
    # Test prediction pipeline
    X, y, features, processed_df = test_prediction_pipeline(df)
    
    # Summary
    console.print("\n" + "=" * 50)
    console.print("[bold green]✓ All tests passed![/]")
    console.print(f"\nTotal matches: {len(df)}")
    console.print(f"Features computed: {len(features)}")
    
    # Show outcome distribution
    from collections import Counter
    outcome_counts = Counter(y)
    table = Table(title="Outcome Distribution")
    table.add_column("Outcome")
    table.add_column("Count")
    table.add_column("Percentage")
    
    labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    for outcome in sorted(outcome_counts.keys()):
        count = outcome_counts[outcome]
        pct = count / len(y) * 100
        table.add_row(labels[outcome], str(count), f"{pct:.1f}%")
    
    console.print(table)


if __name__ == "__main__":
    main()
