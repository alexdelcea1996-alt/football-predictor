"""
Hyperparameter optimization using Optuna.

Trials are scored by mean RPS across temporal cross-validation folds rather
than on a single split, so the chosen parameters are not an artifact of where
one split happened to fall. Results are written in the format consumed by
``football_predictor.models.params``, which feeds them back into training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
from optuna.samplers import TPESampler

from football_predictor.config import get_settings
from football_predictor.evaluation.metrics import ranked_probability_score
from football_predictor.training.cv_strategy import TemporalCrossValidator

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """Optuna-based hyperparameter tuning with a temporal-CV RPS objective."""

    def __init__(
        self,
        n_trials: int | None = None,
        timeout: int | None = None,
        random_state: int = 42,
        n_splits: int = 3,
        show_progress_bar: bool = False,
    ) -> None:
        settings = get_settings()
        self.n_trials = n_trials or settings.model.optuna_n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.n_splits = n_splits
        self.show_progress_bar = show_progress_bar

    # ------------------------------------------------------------ objective

    def cv_score(
        self,
        build_model: Callable[[], Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Mean RPS of a freshly built model across temporal folds.

        Args:
            build_model: Zero-argument factory returning an unfitted estimator
                exposing scikit-learn's fit/predict_proba
            X: Feature matrix in chronological order
            y: Labels
        """
        cv = TemporalCrossValidator(n_splits=self.n_splits)
        scores: list[float] = []

        for train_idx, test_idx in cv.split(X, y):
            model = build_model()
            model.fit(X[train_idx], y[train_idx])
            probs = model.predict_proba(X[test_idx])
            scores.append(ranked_probability_score(y[test_idx], probs))

        if not scores:
            raise ValueError("Temporal CV produced no folds; not enough data to tune")

        return float(np.mean(scores))

    def _run_study(self, objective: Callable[[optuna.Trial], float]) -> optuna.Study:
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.random_state),
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=self.show_progress_bar,
        )
        return study

    # --------------------------------------------------------------- models

    def optimize_catboost(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Optimize CatBoost hyperparameters."""
        from catboost import CatBoostClassifier

        def objective(trial: optuna.Trial) -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 200, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 12.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
            }
            return self.cv_score(
                lambda: CatBoostClassifier(
                    **params,
                    loss_function="MultiClass",
                    verbose=False,
                    random_state=self.random_state,
                ),
                X,
                y,
            )

        return self._run_study(objective).best_params

    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Optimize XGBoost hyperparameters."""
        import xgboost as xgb

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 12.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            return self.cv_score(
                lambda: xgb.XGBClassifier(
                    **params,
                    objective="multi:softprob",
                    num_class=3,
                    verbosity=0,
                    random_state=self.random_state,
                ),
                X,
                y,
            )

        return self._run_study(objective).best_params

    def optimize_logistic(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Optimize logistic regression hyperparameters (with scaling)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        def objective(trial: optuna.Trial) -> float:
            params = {
                "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
                "max_iter": trial.suggest_int("max_iter", 500, 2000),
            }
            return self.cv_score(
                lambda: make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        **params, solver="lbfgs", random_state=self.random_state
                    ),
                ),
                X,
                y,
            )

        best = self._run_study(objective).best_params
        best["solver"] = "lbfgs"
        return best

    def optimize_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: tuple[str, ...] = ("catboost", "xgboost", "logistic"),
    ) -> dict[str, dict[str, Any]]:
        """Tune every requested model and return a params dict ready to save."""
        optimizers = {
            "catboost": self.optimize_catboost,
            "xgboost": self.optimize_xgboost,
            "logistic": self.optimize_logistic,
        }
        return {name: optimizers[name](X, y) for name in models if name in optimizers}

    @staticmethod
    def save_params(
        params: dict[str, Any],
        path: str | Path | None = None,
        results: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write tuned parameters where the training pipeline will pick them up.

        Existing sections for models that were not tuned in this run are kept.
        """
        settings = get_settings()
        target = Path(path) if path else Path(settings.models_dir) / "best_params.json"
        target.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {}
        if target.exists():
            try:
                with open(target) as f:
                    payload = json.load(f)
            except (json.JSONDecodeError, OSError):
                payload = {}

        payload.update(params)
        if results is not None:
            payload["results"] = results

        with open(target, "w") as f:
            json.dump(payload, f, indent=2)

        return target
