"""
Command-line interface for Football Predictor.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def main() -> None:
    """Football Match Prediction System - CLI"""
    pass


@main.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="models", help="Output directory for trained model")
@click.option("--test-size", default=0.2, help="Test set proportion (0.0-1.0)")
def train(data_file: str, output: str, test_size: float) -> None:
    """Train the prediction model on historical data."""
    import pandas as pd
    from football_predictor.training.trainer import ModelTrainer
    
    console.print(f"[bold blue]Loading data from {data_file}...[/]")
    df = pd.read_csv(data_file)
    console.print(f"Loaded {len(df)} matches")
    
    trainer = ModelTrainer(output_dir=output)
    
    console.print("[bold blue]Training ensemble model...[/]")
    results = trainer.train(df, test_size=test_size)
    
    # Display results
    console.print("\n[bold green]Training Complete![/]\n")
    
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Train", justify="right")
    table.add_column("Test", justify="right")
    
    for metric in ["rps", "accuracy", "log_loss"]:
        train_val = results["train_metrics"][metric]
        test_val = results["test_metrics"][metric]
        fmt = ".4f" if metric != "accuracy" else ".2%"
        table.add_row(
            metric.upper(),
            f"{train_val:{fmt}}",
            f"{test_val:{fmt}}"
        )
    
    console.print(table)
    console.print(f"\n[dim]Model saved to: {output}/[/]")


@main.command()
@click.argument("fixtures_file", type=click.Path(exists=True))
@click.option("--model", "-m", default="models", help="Model directory")
@click.option("--output", "-o", default=None, help="Output CSV file")
def predict(fixtures_file: str, model: str, output: str | None) -> None:
    """Predict outcomes for upcoming fixtures."""
    import pandas as pd
    from football_predictor.prediction.predictor import MatchPredictor
    
    console.print(f"[bold blue]Loading model from {model}...[/]")
    predictor = MatchPredictor()
    predictor.load(model)
    
    console.print(f"[bold blue]Loading fixtures from {fixtures_file}...[/]")
    df = pd.read_csv(fixtures_file)
    
    console.print("[bold blue]Generating predictions...[/]")
    predictions = predictor.predict_fixtures(df.to_dict("records"))
    
    # Display predictions
    table = Table(title="Predictions")
    table.add_column("Date", style="dim")
    table.add_column("Home Team", style="cyan")
    table.add_column("Away Team", style="magenta")
    table.add_column("Prediction", style="bold")
    table.add_column("H%", justify="right")
    table.add_column("D%", justify="right")
    table.add_column("A%", justify="right")
    
    for p in predictions:
        table.add_row(
            str(p["date"]),
            p["home_team"],
            p["away_team"],
            p["predicted_outcome"],
            f"{p['home_win_prob']:.0%}",
            f"{p['draw_prob']:.0%}",
            f"{p['away_win_prob']:.0%}",
        )
    
    console.print(table)
    
    if output:
        pd.DataFrame(predictions).to_csv(output, index=False)
        console.print(f"\n[dim]Saved to: {output}[/]")


@main.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("--folds", "-k", default=5, help="Number of CV folds")
def evaluate(data_file: str, folds: int) -> None:
    """Run cross-validation evaluation."""
    import pandas as pd
    from football_predictor.training.trainer import ModelTrainer
    
    console.print(f"[bold blue]Loading data from {data_file}...[/]")
    df = pd.read_csv(data_file)
    
    console.print(f"[bold blue]Running {folds}-fold temporal CV...[/]")
    trainer = ModelTrainer()
    results = trainer.cross_validate(df, n_folds=folds)
    
    # Display fold results
    table = Table(title="Cross-Validation Results")
    table.add_column("Fold", style="cyan")
    table.add_column("RPS", justify="right")
    table.add_column("Accuracy", justify="right")
    
    for m in results["fold_metrics"]:
        table.add_row(
            str(m["fold"]),
            f"{m['rps']:.4f}",
            f"{m['accuracy']:.2%}"
        )
    
    avg = results["average_metrics"]
    table.add_row("---", "---", "---")
    table.add_row(
        "[bold]Average[/]",
        f"[bold]{avg['rps']:.4f}[/]",
        f"[bold]{avg['accuracy']:.2%}[/]"
    )
    
    console.print(table)


@main.command()
@click.option("--output", "-o", default="data/sample_matches.csv", help="Output file")
@click.option("--matches", "-n", default=500, help="Number of sample matches")
def generate_sample(output: str, matches: int) -> None:
    """Generate sample match data for testing."""
    import random
    from datetime import datetime, timedelta
    import pandas as pd
    
    teams = [
        "Arsenal", "Chelsea", "Liverpool", "Man City", "Man United",
        "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham",
        "Crystal Palace", "Brentford", "Fulham", "Wolves", "Bournemouth",
        "Nottingham Forest", "Everton", "Leicester", "Leeds", "Southampton",
    ]
    
    random.seed(42)
    data = []
    start_date = datetime(2022, 8, 1)
    
    for i in range(matches):
        date = start_date + timedelta(days=random.randint(0, 500))
        home, away = random.sample(teams, 2)
        
        # Simulate goals (home advantage)
        home_goals = random.choices([0, 1, 2, 3, 4], weights=[20, 35, 25, 15, 5])[0]
        away_goals = random.choices([0, 1, 2, 3], weights=[30, 35, 25, 10])[0]
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "league": "premier_league",
            "season": "2022/23" if date < datetime(2023, 6, 1) else "2023/24",
        })
    
    df = pd.DataFrame(data)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    
    console.print(f"[green]Generated {matches} sample matches to {output}[/]")


if __name__ == "__main__":
    main()
