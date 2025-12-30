"""
Predict today's Premier League matches.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def main():
    """Predict today's matches."""
    from football_predictor.data.football_data_client import FootballDataClient
    from football_predictor.features.aggregator import FeatureAggregator
    from football_predictor.data.preprocessor import DataPreprocessor
    
    API_KEY = "1b3e20f99212447dba6016eb1810f054"
    
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
    
    # 4. Train model with best params
    console.print("[dim]Loading optimized model...[/]")
    import json
    params_path = Path("models/best_params.json")
    
    if params_path.exists():
        with open(params_path) as f:
            best_params = json.load(f)
        xgb_params = best_params.get("xgboost", {})
        logreg_params = best_params.get("logreg", {})
    else:
        xgb_params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5}
        logreg_params = {"C": 0.001, "solver": "saga", "max_iter": 1000}
    
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        **xgb_params,
        objective="multi:softprob",
        num_class=3,
        verbosity=0,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    
    # Train LogReg
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logreg_model = LogisticRegression(**logreg_params, random_state=42)
        logreg_model.fit(X_train_scaled, y_train)
    
    # 5. Predict fixtures
    console.print("[dim]Generating predictions...[/]\n")
    
    predictions = []
    for _, match in fixtures.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        match_date = match["date"]
        
        # Get features
        features = aggregator.get_features_for_match(home, away, match_date)
        X = np.array([[features.get(f, 0) or 0 for f in feature_names]])
        
        # Predict with both models
        xgb_probs = xgb_model.predict_proba(X)[0]
        logreg_probs = logreg_model.predict_proba(scaler.transform(X))[0]
        
        # Ensemble
        probs = (xgb_probs + logreg_probs) / 2
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
    console.print("\n[dim]Model: XGBoost + LogReg ensemble (optimized)[/]")
    console.print(f"[dim]Training data: {len(train_df)} matches from 2023/24[/]")


if __name__ == "__main__":
    main()
