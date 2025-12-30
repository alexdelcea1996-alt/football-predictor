"""
Full training test with real data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    """Train model on real data."""
    console.print("[bold green]Football Predictor - Full Training Test[/]")
    console.print("=" * 50)
    
    # Load data
    data_path = Path("data/matches_2023_24.csv")
    if not data_path.exists():
        console.print("[red]Run test_integration.py first to fetch data[/]")
        return
    
    df = pd.read_csv(data_path)
    console.print(f"\nLoaded {len(df)} matches from {data_path}")
    
    # Import components
    from football_predictor.features.aggregator import FeatureAggregator
    from football_predictor.data.preprocessor import DataPreprocessor
    from football_predictor.evaluation.metrics import calculate_all_metrics, ranked_probability_score
    
    # Try to import ML models
    try:
        from football_predictor.models.xgboost_model import XGBoostModel
        HAS_XGBOOST = True
    except ImportError:
        HAS_XGBOOST = False
        console.print("[yellow]XGBoost not available[/]")
    
    try:
        from football_predictor.models.logistic_model import LogisticModel
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False
        console.print("[yellow]Scikit-learn not available[/]")
    
    # Preprocess
    console.print("\n[blue]Preprocessing data...[/]")
    preprocessor = DataPreprocessor()
    df = preprocessor.encode_outcome(df)
    
    # Compute features
    console.print("[blue]Computing features...[/]")
    aggregator = FeatureAggregator()
    df = aggregator.process_matches(df)
    
    # Filter finished matches
    df = df[df["outcome"].notna()].reset_index(drop=True)
    console.print(f"  Processed {len(df)} matches with {len(aggregator.feature_names)} features")
    
    # Train/test split (temporal)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    console.print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Get feature matrices
    X_train, feature_names = aggregator.get_feature_matrix(train_df)
    y_train = train_df["outcome"].astype(int).values
    
    X_test, _ = aggregator.get_feature_matrix(test_df)
    y_test = test_df["outcome"].astype(int).values
    
    # Train models
    results = []
    
    if HAS_XGBOOST:
        console.print("\n[blue]Training XGBoost...[/]")
        xgb = XGBoostModel(n_estimators=200, learning_rate=0.05, max_depth=5)
        xgb.fit(X_train, y_train, feature_names)
        
        xgb_probs = xgb.predict_proba(X_test)
        xgb_metrics = calculate_all_metrics(y_test, xgb_probs)
        results.append(("XGBoost", xgb_metrics))
        console.print(f"  ✓ Accuracy: {xgb_metrics['accuracy']:.2%}, RPS: {xgb_metrics['rps']:.4f}")
    
    if HAS_SKLEARN:
        console.print("\n[blue]Training Logistic Regression...[/]")
        logreg = LogisticModel()
        logreg.fit(X_train, y_train, feature_names)
        
        logreg_probs = logreg.predict_proba(X_test)
        logreg_metrics = calculate_all_metrics(y_test, logreg_probs)
        results.append(("LogReg", logreg_metrics))
        console.print(f"  ✓ Accuracy: {logreg_metrics['accuracy']:.2%}, RPS: {logreg_metrics['rps']:.4f}")
    
    # Show results table
    if results:
        console.print("\n" + "=" * 50)
        table = Table(title="Model Comparison")
        table.add_column("Model")
        table.add_column("Accuracy", justify="right")
        table.add_column("RPS", justify="right")
        table.add_column("Log Loss", justify="right")
        table.add_column("F1 Macro", justify="right")
        
        for name, m in results:
            table.add_row(
                name,
                f"{m['accuracy']:.2%}",
                f"{m['rps']:.4f}",
                f"{m['log_loss']:.4f}",
                f"{m['f1_macro']:.4f}",
            )
        
        console.print(table)
        
        # Show prediction samples
        if HAS_XGBOOST:
            console.print("\n[dim]Sample Predictions (XGBoost):[/]")
            sample_idx = np.random.choice(len(X_test), 5, replace=False)
            
            for i in sample_idx:
                probs = xgb_probs[i]
                actual = int(y_test[i])
                pred = int(probs.argmax())
                labels = {0: "H", 1: "D", 2: "A"}
                
                match = test_df.iloc[i]
                console.print(
                    f"  {match['home_team'][:15]:15} vs {match['away_team'][:15]:15} | "
                    f"Pred: {labels[pred]} ({max(probs):.0%}) | "
                    f"Actual: {labels[actual]} | "
                    f"{'✓' if pred == actual else '✗'}"
                )
    
    console.print("\n[bold green]✓ Training complete![/]")


if __name__ == "__main__":
    main()
