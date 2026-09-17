"""
Hyperparameter optimization with Optuna.

Tuning runs on temporal cross-validation folds inside the training block; the
final holdout is scored once, after tuning, and never influences the search.
(The previous version optimized the objective *and* early-stopped directly on
the holdout, then reported those same numbers as results.)

Results are written to models/best_params.json, which the training pipeline
reads automatically via football_predictor.models.params.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from football_predictor.data.preprocessor import DataPreprocessor
from football_predictor.evaluation.metrics import calculate_all_metrics
from football_predictor.features.aggregator import FeatureAggregator
from football_predictor.models.catboost_model import CatBoostModel
from football_predictor.models.ensemble import EnsemblePredictor
from football_predictor.models.logistic_model import LogisticModel
from football_predictor.models.xgboost_model import XGBoostModel
from football_predictor.training.hyperopt import HyperparameterOptimizer

console = Console()

MODEL_CLASSES = {
    "catboost": CatBoostModel,
    "xgboost": XGBoostModel,
    "logistic": LogisticModel,
}


def load_data(path: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load the requested file, or the best available data file."""
    if path:
        return pd.read_csv(path), path

    for candidate in (
        "data/matches_with_odds.csv",
        "data/matches_multi_season.csv",
        "data/matches_2023_24.csv",
    ):
        if Path(candidate).exists():
            return pd.read_csv(candidate), candidate

    raise FileNotFoundError("No data found. Run a fetch script first.")


def prepare_features(df: pd.DataFrame, holdout_size: float = 0.2):
    """Compute features and split off a final holdout that tuning never sees."""
    df = DataPreprocessor().encode_outcome(df)

    aggregator = FeatureAggregator()
    df = aggregator.process_matches(df)
    df = df[df["outcome"].notna()].reset_index(drop=True)

    X, feature_names = aggregator.get_feature_matrix(df)
    y = df["outcome"].astype(int).values

    split_idx = int(len(y) * (1 - holdout_size))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:], feature_names


@click.command()
@click.option("--data", "data_path", default=None, help="Match data CSV")
@click.option("--trials", default=50, help="Optuna trials per model")
@click.option("--folds", default=3, help="Temporal CV folds used for scoring trials")
@click.option(
    "--models",
    default="catboost,xgboost,logistic",
    help="Comma-separated models to tune",
)
@click.option("--output", default=None, help="Where to write best_params.json")
def main(data_path: str | None, trials: int, folds: int, models: str, output: str | None) -> None:
    """Run hyperparameter optimization."""
    console.print("[bold green]Football Predictor - Hyperparameter Optimization[/]")
    console.print("=" * 60)

    df, source = load_data(data_path)
    console.print(f"Loaded {len(df)} matches from {source}")

    console.print("\n[blue]Preparing features...[/]")
    X_tune, y_tune, X_holdout, y_holdout, feature_names = prepare_features(df)
    console.print(f"  Tuning block: {len(y_tune)} | Final holdout: {len(y_holdout)}")
    console.print(f"  Features: {len(feature_names)}")

    requested = tuple(m.strip() for m in models.split(",") if m.strip())
    optimizer = HyperparameterOptimizer(n_trials=trials, n_splits=folds, show_progress_bar=True)

    best_params: dict[str, dict] = {}
    for name in requested:
        if name not in MODEL_CLASSES:
            console.print(f"[yellow]Skipping unknown model: {name}[/]")
            continue
        console.print(f"\n[bold blue]Optimizing {name} ({trials} trials, {folds}-fold temporal CV)...[/]")
        best_params[name] = optimizer.optimize_all(X_tune, y_tune, models=(name,))[name]
        console.print(f"  Best params: {best_params[name]}")

    if not best_params:
        console.print("[red]Nothing was tuned.[/]")
        return

    params_path = optimizer.save_params(best_params, path=output)
    console.print(f"\n[green]Best parameters saved to: {params_path}[/]")

    # Final, honest evaluation: models pick up the tuned params from the file
    console.print("\n[bold blue]Evaluating on the untouched holdout...[/]")
    results: dict[str, dict] = {}

    for name in best_params:
        model = MODEL_CLASSES[name](tuned_params_path=params_path)
        model.fit(X_tune, y_tune, feature_names)
        results[name] = calculate_all_metrics(y_holdout, model.predict_proba(X_holdout))

    ensemble = EnsemblePredictor().fit(X_tune, y_tune, feature_names)
    results["ensemble"] = calculate_all_metrics(y_holdout, ensemble.predict_proba(X_holdout))
    blend_info = ensemble.get_blend_info()

    table = Table(title="Holdout results with tuned parameters")
    table.add_column("Model")
    table.add_column("Accuracy", justify="right")
    table.add_column("RPS", justify="right")
    table.add_column("Log loss", justify="right")

    for name, m in results.items():
        table.add_row(
            name.upper(),
            f"{m['accuracy']:.2%}",
            f"{m['rps']:.4f}",
            f"{m['log_loss']:.4f}",
        )
    console.print(table)
    console.print(f"Ensemble blend: {blend_info}")

    optimizer.save_params(best_params, path=params_path, results=results)


if __name__ == "__main__":
    main()
