"""
Benchmark pipeline configurations with temporal cross-validation.

Every configuration sees identical folds: train on all earlier matches, score
the next block, never look forward. The "legacy" row reproduces the behaviour
this repository had before the audit (equal-weight blending, no calibration,
dead features kept) so each change can be judged on its own.

Reference rows are not models to deploy, they are yardsticks: the market row
shows what the bookmakers' own prices score, and the class-prior row shows
what predicting the base rates achieves.

Examples:
    python scripts/benchmark.py                          # simulated league
    python scripts/benchmark.py --data data/matches_with_odds.csv
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# Feature set -> (use_odds, use_poisson, drop_degenerate)
FEATURE_SETS = {
    "base_raw": (False, False, False),
    "base": (False, False, True),
    "poisson": (False, True, True),
    "odds": (True, False, True),
    "full": (True, True, True),
}

# name -> (feature set, ensemble kwargs, description)
_TUNED = {"blend": "weights", "calibrate": True}
_NO_MEMBERS = {"include_probability_members": False}

CONFIGS = {
    "legacy": (
        "base_raw",
        {
            "blend": "equal",
            "calibrate": False,
            "refit_on_full": False,
            "blend_strategy": "holdout",
            **_NO_MEMBERS,
        },
        "pre-audit behaviour",
    ),
    "fixed": ("base", {**_TUNED, **_NO_MEMBERS}, "fitted blend + calibration"),
    "fixed_no_refit": (
        "base",
        {**_TUNED, "refit_on_full": False, **_NO_MEMBERS},
        "same, members keep the val split",
    ),
    "poisson_features": ("poisson", {**_TUNED, **_NO_MEMBERS}, "Dixon-Coles as features"),
    "poisson_member": ("poisson", {**_TUNED}, "Dixon-Coles as ensemble member"),
    "odds_features": ("odds", {**_TUNED, **_NO_MEMBERS}, "market odds as features"),
    "odds_member": ("odds", {**_TUNED}, "market odds as ensemble member"),
    "full": ("full", {**_TUNED}, "odds + Dixon-Coles as members"),
    "full+stacking": ("full", {**_TUNED, "blend": "stack"}, "same, meta-learner blend"),
}


def load_matches(data_path: str | None, seasons: int) -> tuple[pd.DataFrame, str]:
    """Load real matches, or simulate a league when no data file is given."""
    from football_predictor.data.simulator import simulate_league

    if data_path:
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"])
        return df, f"{data_path} (real)"

    return simulate_league(n_seasons=seasons), f"simulated league ({seasons} seasons)"


def build_features(df: pd.DataFrame, feature_set: str):
    """Compute one feature variant; returns (processed_df, X, y, feature_names)."""
    from football_predictor.data.preprocessor import DataPreprocessor
    from football_predictor.features.aggregator import FeatureAggregator

    use_odds, use_poisson, drop_degenerate = FEATURE_SETS[feature_set]

    prepared = DataPreprocessor().encode_outcome(df)
    aggregator = FeatureAggregator(
        use_odds=use_odds,
        use_poisson=use_poisson,
        drop_degenerate=drop_degenerate,
    )
    processed = aggregator.process_matches(prepared)
    processed = processed[processed["outcome"].notna()].reset_index(drop=True)

    X, names = aggregator.get_feature_matrix(processed)
    y = processed["outcome"].astype(int).to_numpy()
    return processed, X, y, names


def evaluate_folds(probs_by_fold: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Pool fold predictions and compute the reported metrics."""
    from football_predictor.evaluation.metrics import calculate_all_metrics

    y_true = np.concatenate([y for y, _ in probs_by_fold])
    y_prob = np.vstack([p for _, p in probs_by_fold])

    metrics = calculate_all_metrics(y_true, y_prob)
    predicted = np.argmax(y_prob, axis=1)
    metrics["draw_predicted_rate"] = float((predicted == 1).mean())
    metrics["draw_actual_rate"] = float((y_true == 1).mean())
    metrics["n_predictions"] = int(len(y_true))
    return metrics


def run_pipeline_config(
    X, y, folds: int, ensemble_kwargs: dict, feature_names: list[str]
) -> dict[str, float]:
    """Temporal-CV evaluation of one ensemble configuration."""
    from football_predictor.models.ensemble import EnsemblePredictor
    from football_predictor.training.cv_strategy import TemporalCrossValidator

    cv = TemporalCrossValidator(n_splits=folds)
    results = []
    blend_info: dict = {}

    for train_idx, test_idx in cv.split(X, y):
        model = EnsemblePredictor(**ensemble_kwargs)
        model.fit(X[train_idx], y[train_idx], feature_names)
        results.append((y[test_idx], model.predict_proba(X[test_idx])))
        blend_info = model.get_blend_info()

    metrics = evaluate_folds(results)
    metrics["blend"] = blend_info.get("weights", blend_info.get("meta_learner", "n/a"))
    metrics["members"] = blend_info.get("members", [])
    return metrics


def run_market_reference(processed: pd.DataFrame, y, folds: int) -> dict[str, float] | None:
    """What the bookmakers' own de-vigged prices score on the same folds."""
    from football_predictor.training.cv_strategy import TemporalCrossValidator

    cols = ["odds_prob_home", "odds_prob_draw", "odds_prob_away"]
    if not all(col in processed.columns for col in cols):
        return None

    market = processed[cols].to_numpy(dtype=float)
    cv = TemporalCrossValidator(n_splits=folds)
    results = []

    for train_idx, test_idx in cv.split(market, y):
        probs = market[test_idx].copy()
        missing = ~np.isfinite(probs).all(axis=1)
        if missing.any():
            prior = np.bincount(y[train_idx], minlength=3) / len(train_idx)
            probs[missing] = prior
        results.append((y[test_idx], probs))

    return evaluate_folds(results)


def run_prior_reference(y, folds: int) -> dict[str, float]:
    """Predicting the training-set base rates for every match."""
    from football_predictor.training.cv_strategy import TemporalCrossValidator

    cv = TemporalCrossValidator(n_splits=folds)
    results = []

    for train_idx, test_idx in cv.split(np.zeros((len(y), 1)), y):
        prior = np.bincount(y[train_idx], minlength=3) / len(train_idx)
        results.append((y[test_idx], np.tile(prior, (len(test_idx), 1))))

    return evaluate_folds(results)


def run_dixon_coles_reference(processed: pd.DataFrame, y, folds: int) -> dict[str, float]:
    """The goal model alone, refitted per fold."""
    from football_predictor.models.poisson_model import DixonColesModel
    from football_predictor.training.cv_strategy import TemporalCrossValidator

    cv = TemporalCrossValidator(n_splits=folds)
    results = []

    for train_idx, test_idx in cv.split(np.zeros((len(y), 1)), y):
        model = DixonColesModel().fit(processed.iloc[train_idx])
        results.append((y[test_idx], model.predict_proba(processed.iloc[test_idx])))

    return evaluate_folds(results)


@click.command()
@click.option("--data", "data_path", default=None, help="Match CSV (default: simulate)")
@click.option("--seasons", default=4, help="Seasons to simulate when no data given")
@click.option("--folds", default=3, help="Temporal CV folds")
@click.option("--blend-folds", default=3, help="Folds used to fit calibration and blending")
@click.option("--output", default="models/benchmark.json", help="Where to write results")
@click.option("--tuned/--no-tuned", default=False, help="Use models/best_params.json")
def main(
    data_path: str | None,
    seasons: int,
    folds: int,
    blend_folds: int,
    output: str,
    tuned: bool,
) -> None:
    """Compare pipeline configurations on identical temporal folds."""
    if not tuned:
        # Keep stale tuned parameters out of a like-for-like comparison
        os.environ["MODELS_DIR"] = tempfile.mkdtemp(prefix="fp_benchmark_")

    df, source = load_matches(data_path, seasons)
    console.print("[bold green]Football Predictor - pipeline benchmark[/]")
    console.print(f"Data: {source} | {len(df)} matches | {folds} temporal folds")
    console.print(
        "[dim]Every configuration is trained from scratch on each fold, and the "
        "default blending walks folds inside the training block, so a full run "
        "takes a while (tens of minutes on a few thousand matches).[/]"
    )
    if not data_path:
        console.print(
            "[yellow]Simulated data validates the pipeline, not real-world accuracy. "
            "Run scripts/fetch_odds_data.py, then re-run with --data.[/]"
        )
    console.print()

    feature_cache: dict[str, tuple] = {}
    needed = {config[0] for config in CONFIGS.values()}

    for name in sorted(needed):
        started = time.time()
        feature_cache[name] = build_features(df, name)
        n_features = len(feature_cache[name][3])
        console.print(f"  features [{name}]: {n_features} columns ({time.time() - started:.1f}s)")

    results: dict[str, dict] = {}

    for name, (feature_set, kwargs, description) in CONFIGS.items():
        _, X, y, names = feature_cache[feature_set]
        started = time.time()
        results[name] = run_pipeline_config(
            X, y, folds, {**kwargs, "blend_folds": blend_folds}, names
        )
        results[name]["description"] = description
        results[name]["features"] = len(feature_cache[feature_set][3])
        console.print(f"  ran [{name}] in {time.time() - started:.1f}s")

    processed_full, _, y_full, _ = feature_cache["full"]
    results["reference:class_prior"] = {
        **run_prior_reference(y_full, folds),
        "description": "predict base rates",
    }
    results["reference:dixon_coles"] = {
        **run_dixon_coles_reference(processed_full, y_full, folds),
        "description": "goal model alone",
    }
    market = run_market_reference(processed_full, y_full, folds)
    if market:
        results["reference:market"] = {**market, "description": "bookmaker odds themselves"}

    table = Table(title="Temporal cross-validation (pooled over folds)")
    table.add_column("Configuration")
    table.add_column("Accuracy", justify="right")
    table.add_column("RPS", justify="right")
    table.add_column("Log loss", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("Draw F1", justify="right")
    table.add_column("Draws pred.", justify="right")
    table.add_column("Notes")

    best_rps = min(m["rps"] for m in results.values())
    for name, m in results.items():
        highlight = "[bold green]" if m["rps"] == best_rps else ""
        table.add_row(
            f"{highlight}{name}",
            f"{m['accuracy']:.1%}",
            f"{m['rps']:.4f}",
            f"{m['log_loss']:.4f}",
            f"{m['ece']:.3f}",
            f"{m['f1_draw']:.3f}",
            f"{m['draw_predicted_rate']:.1%}",
            m["description"],
        )

    console.print()
    console.print(table)
    console.print(
        f"[dim]Actual draw rate: {results['full']['draw_actual_rate']:.1%} | "
        f"predictions scored: {results['full']['n_predictions']}[/]"
    )

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump({"source": source, "folds": folds, "results": results}, f, indent=2)
    console.print(f"\n[green]Results written to {target}[/]")


if __name__ == "__main__":
    main()
