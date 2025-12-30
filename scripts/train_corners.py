"""
Train and evaluate the corners prediction model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    """Train corners predictor."""
    from football_predictor.models.corners_predictor import CornersPredictor
    
    console.print("[bold blue]🔲 Corners Prediction Model Training[/]")
    console.print("=" * 50)
    
    # Load data
    data_path = Path("data/corners_sample.csv")
    if not data_path.exists():
        console.print("[yellow]Sample data not found. Generating...[/]")
        from scripts.generate_corners_sample import generate_corners_data
        df = generate_corners_data(n_matches=500)
        data_path.parent.mkdir(exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    
    console.print(f"Loaded {len(df)} matches")
    console.print(f"Average total corners: {df['total_corners'].mean():.1f}")
    console.print(f"Over 9.5 rate: {(df['total_corners'] > 9.5).mean():.1%}\n")
    
    # Train model
    console.print("[blue]Training model (O/U 9.5)...[/]")
    predictor = CornersPredictor(threshold=9.5)
    results = predictor.fit(df)
    
    # Display results
    table = Table(title="Training Results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    
    table.add_row("Train samples", str(results["train_samples"]))
    table.add_row("Test samples", str(results["test_samples"]))
    table.add_row("Over 9.5 rate", f"{results['over_rate']:.1%}")
    table.add_row("", "")
    table.add_row("Accuracy", f"{results['accuracy']:.1%}")
    table.add_row("Baseline (majority)", f"{results['baseline_accuracy']:.1%}")
    table.add_row("[bold]Improvement[/]", f"[bold]+{results['improvement']:.1%}[/]")
    table.add_row("AUC", f"{results['auc']:.3f}")
    
    console.print(table)
    
    # Save model
    model_path = Path("models/corners")
    predictor.save(model_path)
    console.print(f"\n[green]Model saved to: {model_path}[/]")
    
    # Sample predictions
    console.print("\n[dim]Sample Predictions:[/]")
    sample_matches = [
        ("Manchester City", "Liverpool"),
        ("Burnley", "Sheffield United"),
        ("Arsenal", "Everton"),
    ]
    
    for home, away in sample_matches:
        pred = predictor.predict_match(home, away)
        exp = f"(exp: {pred['expected_total']:.1f})" if pred['expected_total'] else ""
        console.print(
            f"  {home} vs {away}: {pred['prediction']} {exp} "
            f"({pred['over_prob']:.0%} over)"
        )


if __name__ == "__main__":
    main()
