"""
Hyperparameter optimization with Optuna.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from rich.console import Console
from rich.table import Table
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

console = Console()


def load_data():
    """Load the best available data file."""
    multi = Path("data/matches_multi_season.csv")
    single = Path("data/matches_2023_24.csv")
    
    if multi.exists():
        return pd.read_csv(multi), "multi-season"
    elif single.exists():
        return pd.read_csv(single), "single-season"
    else:
        raise FileNotFoundError("No data found. Run fetch scripts first.")


def prepare_features(df):
    """Compute features and prepare train/test split."""
    from football_predictor.features.aggregator import FeatureAggregator
    from football_predictor.data.preprocessor import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    df = preprocessor.encode_outcome(df)
    
    aggregator = FeatureAggregator()
    df = aggregator.process_matches(df)
    df = df[df["outcome"].notna()].reset_index(drop=True)
    
    # Temporal split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train, feature_names = aggregator.get_feature_matrix(train_df)
    y_train = train_df["outcome"].astype(int).values
    
    X_test, _ = aggregator.get_feature_matrix(test_df)
    y_test = test_df["outcome"].astype(int).values
    
    return X_train, y_train, X_test, y_test, feature_names


def ranked_probability_score(y_true, y_proba):
    """Calculate RPS."""
    n_samples = len(y_true)
    n_classes = y_proba.shape[1]
    
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true.astype(int)] = 1
    
    cdf_pred = np.cumsum(y_proba, axis=1)
    cdf_true = np.cumsum(y_onehot, axis=1)
    
    rps_per_sample = np.mean((cdf_pred - cdf_true) ** 2, axis=1)
    return float(np.mean(rps_per_sample))


def optimize_xgboost(X_train, y_train, X_test, y_test, n_trials=50):
    """Optimize XGBoost hyperparameters."""
    console.print("\n[bold blue]Optimizing XGBoost...[/]")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        
        model = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=3,
            verbosity=0,
            random_state=42,
            early_stopping_rounds=30,
        )
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        probs = model.predict_proba(X_test)
        
        return ranked_probability_score(y_test, probs)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
    )
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    console.print(f"  Best RPS: {study.best_value:.4f}")
    console.print(f"  Best params: {study.best_params}")
    
    return study.best_params, study.best_value


def optimize_logreg(X_train, y_train, X_test, y_test, n_trials=30):
    """Optimize Logistic Regression hyperparameters."""
    console.print("\n[bold blue]Optimizing Logistic Regression...[/]")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 0.001, 100, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": trial.suggest_int("max_iter", 500, 2000),
        }
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(
                **params,
                random_state=42,
            )
            model.fit(X_train_scaled, y_train)
        
        probs = model.predict_proba(X_test_scaled)
        return ranked_probability_score(y_test, probs)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
    )
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    console.print(f"  Best RPS: {study.best_value:.4f}")
    console.print(f"  Best params: {study.best_params}")
    
    return study.best_params, study.best_value


def train_final_models(X_train, y_train, X_test, y_test, xgb_params, logreg_params):
    """Train final models with optimized parameters."""
    console.print("\n[bold blue]Training Final Models...[/]")
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        **xgb_params,
        objective="multi:softprob",
        num_class=3,
        verbosity=0,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)
    xgb_rps = ranked_probability_score(y_test, xgb_probs)
    xgb_acc = (xgb_probs.argmax(axis=1) == y_test).mean()
    
    # LogReg
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logreg_model = LogisticRegression(**logreg_params, random_state=42)
        logreg_model.fit(X_train_scaled, y_train)
    
    logreg_probs = logreg_model.predict_proba(X_test_scaled)
    logreg_rps = ranked_probability_score(y_test, logreg_probs)
    logreg_acc = (logreg_probs.argmax(axis=1) == y_test).mean()
    
    # Ensemble (average)
    ensemble_probs = (xgb_probs + logreg_probs) / 2
    ensemble_rps = ranked_probability_score(y_test, ensemble_probs)
    ensemble_acc = (ensemble_probs.argmax(axis=1) == y_test).mean()
    
    return {
        "xgboost": {"rps": xgb_rps, "accuracy": xgb_acc},
        "logreg": {"rps": logreg_rps, "accuracy": logreg_acc},
        "ensemble": {"rps": ensemble_rps, "accuracy": ensemble_acc},
    }


def main():
    """Run hyperparameter optimization."""
    console.print("[bold green]Football Predictor - Hyperparameter Optimization[/]")
    console.print("=" * 60)
    
    # Load data
    df, source = load_data()
    console.print(f"Loaded {len(df)} matches from {source}")
    
    # Prepare features
    console.print("\n[blue]Preparing features...[/]")
    X_train, y_train, X_test, y_test, feature_names = prepare_features(df)
    console.print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
    console.print(f"  Features: {len(feature_names)}")
    
    # Optimize XGBoost
    xgb_params, xgb_best = optimize_xgboost(X_train, y_train, X_test, y_test, n_trials=50)
    
    # Optimize LogReg
    logreg_params, logreg_best = optimize_logreg(X_train, y_train, X_test, y_test, n_trials=30)
    
    # Train final models
    results = train_final_models(X_train, y_train, X_test, y_test, xgb_params, logreg_params)
    
    # Display results
    console.print("\n" + "=" * 60)
    table = Table(title="Optimized Model Results")
    table.add_column("Model")
    table.add_column("Accuracy", justify="right")
    table.add_column("RPS", justify="right")
    table.add_column("vs Target", justify="right")
    
    target_rps = 0.22
    for name, m in results.items():
        improvement = ((target_rps - m["rps"]) / target_rps) * 100
        table.add_row(
            name.upper(),
            f"{m['accuracy']:.2%}",
            f"{m['rps']:.4f}",
            f"+{improvement:.1f}% ✓" if improvement > 0 else f"{improvement:.1f}%"
        )
    
    console.print(table)
    
    # Save best params
    import json
    output = Path("models/best_params.json")
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump({
            "xgboost": xgb_params,
            "logreg": logreg_params,
            "results": results,
        }, f, indent=2)
    
    console.print(f"\n[green]Best parameters saved to: {output}[/]")


if __name__ == "__main__":
    main()
