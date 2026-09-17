"""
Predict today's Premier League matches.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from football_predictor.prediction.predictor import as_feature_value

console = Console()


def main():
    """Predict today's matches."""
    from football_predictor.data.football_data_client import FootballDataClient
    from football_predictor.features.aggregator import FeatureAggregator
    from football_predictor.data.preprocessor import DataPreprocessor
    
    API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not API_KEY:
        raise SystemExit("Set FOOTBALL_DATA_API_KEY (free key at football-data.org)")
    
    console.print("\n[bold blue]🔮 Premier League Match Predictions[/]")
    console.print(f"Date: {date.today()}\n")
    
    # 1. Fetch today's scheduled fixtures
    console.print("[dim]Fetching today's fixtures...[/]")
    client = FootballDataClient(API_KEY)
    
    today = date.today()
    fixtures = client.get_matches(
        "premier_league",
        date_from=today,
        date_to=today,
        status="SCHEDULED"
    )
    
    if fixtures.empty:
        # Also try TIMED status
        fixtures = client.get_matches(
            "premier_league",
            date_from=today,
            date_to=today,
        )
        fixtures = fixtures[fixtures["status"].isin(["SCHEDULED", "TIMED"])]
    
    if fixtures.empty:
        console.print("[yellow]No scheduled matches found for today.[/]")
        console.print("[dim]Checking upcoming matches...[/]")
        
        # Get next 7 days
        from datetime import timedelta
        fixtures = client.get_matches(
            "premier_league",
            date_from=today,
            date_to=today + timedelta(days=7),
        )
        fixtures = fixtures[fixtures["status"].isin(["SCHEDULED", "TIMED"])]
    
    if fixtures.empty:
        console.print("[red]No upcoming fixtures found.[/]")
        return
    
    console.print(f"Found {len(fixtures)} upcoming matches\n")
    
    # 2. Load historical data to build feature state
    console.print("[dim]Loading historical data...[/]")
    hist_path = Path("data/matches_2023_24.csv")
    if not hist_path.exists():
        console.print("[red]No historical data. Run test_integration.py first.[/]")
        return
    
    hist_df = pd.read_csv(hist_path)
    
    # 3. Process historical data to build feature state
    preprocessor = DataPreprocessor()
    hist_df = preprocessor.encode_outcome(hist_df)
    
    aggregator = FeatureAggregator()
    hist_df = aggregator.process_matches(hist_df)
    
    # Get finished matches for training
    train_df = hist_df[hist_df["outcome"].notna()].reset_index(drop=True)
    X_train, feature_names = aggregator.get_feature_matrix(train_df)
    y_train = train_df["outcome"].astype(int).values
    
    # 4. Train the ensemble (tuned params, calibration and blend weights are
    #    handled by the package, so predictions match what training evaluated)
    console.print("[dim]Training ensemble (tuned params + calibrated blend)...[/]")
    from football_predictor.models.ensemble import EnsemblePredictor

    ensemble = EnsemblePredictor().fit(X_train, y_train, feature_names)
    console.print(f"[dim]Blend: {ensemble.get_blend_info()}[/]")

    # 5. Predict fixtures
    console.print("[dim]Generating predictions...[/]\n")
    
    predictions = []
    for _, match in fixtures.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        match_date = match["date"]
        
        # Get features. `nan or 0` is nan (NaN is truthy), so missing values
        # have to be coerced explicitly or they reach the models as NaN.
        features = aggregator.get_features_for_match(home, away, match_date)
        X = np.array([[as_feature_value(features.get(name)) for name in feature_names]])
        
        probs = ensemble.predict_proba(X)[0]
        pred_class = int(probs.argmax())
        
        predictions.append({
            "home": home,
            "away": away,
            "date": str(match_date)[:10],
            "pred": pred_class,
            "probs": probs,
            "conf": max(probs),
        })
    
    # 6. Display results
    table = Table(title="⚽ Match Predictions", show_header=True)
    table.add_column("Match", style="bold")
    table.add_column("Prediction", justify="center")
    table.add_column("H%", justify="right", style="cyan")
    table.add_column("D%", justify="right", style="yellow")
    table.add_column("A%", justify="right", style="magenta")
    table.add_column("Conf", justify="right")
    
    outcome_map = {0: "🏠 HOME", 1: "🤝 DRAW", 2: "✈️ AWAY"}
    
    for p in predictions:
        table.add_row(
            f"{p['home'][:18]} vs {p['away'][:18]}",
            outcome_map[p["pred"]],
            f"{p['probs'][0]:.0%}",
            f"{p['probs'][1]:.0%}",
            f"{p['probs'][2]:.0%}",
            f"{p['conf']:.0%}",
        )
    
    console.print(table)
    
    # Summary
    console.print(f"\n[dim]Model: calibrated ensemble of {', '.join(ensemble.get_models_info())}[/]")
    console.print(f"[dim]Training data: {len(train_df)} matches from 2023/24[/]")


if __name__ == "__main__":
    main()
