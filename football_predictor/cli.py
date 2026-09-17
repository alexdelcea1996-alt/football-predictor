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
@click.option("--test-size", default=0.2, help="Final holdout proportion (0.0-1.0)")
@click.option("--val-size", default=0.15, help="Validation block used for calibration and blending")
@click.option(
    "--blend",
    type=click.Choice(["weights", "stack", "equal"]),
    default="weights",
    help="How ensemble members are combined",
)
def train(data_file: str, output: str, test_size: float, val_size: float, blend: str) -> None:
    """Train the prediction model on historical data."""
    import pandas as pd
    from football_predictor.training.trainer import ModelTrainer

    console.print(f"[bold blue]Loading data from {data_file}...[/]")
    df = pd.read_csv(data_file)
    console.print(f"Loaded {len(df)} matches")

    trainer = ModelTrainer(output_dir=output)

    console.print("[bold blue]Training ensemble model...[/]")
    results = trainer.train(df, test_size=test_size, val_size=val_size, blend=blend)
    
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

    if results.get("member_test_metrics"):
        members = Table(title="Per-member holdout scores")
        members.add_column("Member", style="cyan")
        members.add_column("Accuracy", justify="right")
        members.add_column("RPS", justify="right")
        members.add_column("Weight", justify="right")

        weights = results.get("blend", {}).get("weights", {})
        for name, metrics in results["member_test_metrics"].items():
            members.add_row(
                name,
                f"{metrics['accuracy']:.2%}",
                f"{metrics['rps']:.4f}",
                f"{weights.get(name, float('nan')):.3f}" if weights else "-",
            )
        console.print(members)

    console.print(f"\n[dim]Blend: {results.get('blend')}[/]")
    console.print(f"[dim]Samples - train: {results['train_samples']}, "
                  f"val: {results['val_samples']}, test: {results['test_samples']}[/]")
    console.print(f"[dim]Model saved to: {output}/[/]")


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
@click.option(
    "--blend",
    type=click.Choice(["weights", "stack", "equal"]),
    default="weights",
    help="How ensemble members are combined",
)
def evaluate(data_file: str, folds: int, blend: str) -> None:
    """Run cross-validation evaluation."""
    import pandas as pd
    from football_predictor.training.trainer import ModelTrainer
    
    console.print(f"[bold blue]Loading data from {data_file}...[/]")
    df = pd.read_csv(data_file)
    
    console.print(f"[bold blue]Running {folds}-fold temporal CV...[/]")
    trainer = ModelTrainer()
    results = trainer.cross_validate(df, n_folds=folds, blend=blend)
    
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
@click.option("--matches", "-n", default=0, help="Cap on matches (0 = all)")
@click.option("--seasons", default=3, help="Seasons to simulate")
@click.option("--with-odds/--no-odds", default=True, help="Include simulated bookmaker odds")
def generate_sample(output: str, matches: int, seasons: int, with_odds: bool) -> None:
    """Generate sample match data for testing."""
    from football_predictor.data.simulator import simulate_league

    df = simulate_league(n_seasons=seasons, with_odds=with_odds)
    if matches and matches < len(df):
        df = df.head(matches)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    console.print(f"[green]Generated {len(df)} simulated matches to {output}[/]")
    console.print(
        "[dim]Teams have latent strengths, so the data carries learnable signal; "
        "it still says nothing about real-world accuracy.[/]"
    )


if __name__ == "__main__":
    main()
