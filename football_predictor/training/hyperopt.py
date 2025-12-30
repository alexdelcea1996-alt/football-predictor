"""
Hyperparameter optimization using Optuna.
"""

from typing import Any, Callable
import numpy as np
import optuna
from optuna.samplers import TPESampler

from football_predictor.config import get_settings
from football_predictor.evaluation.metrics import ranked_probability_score


class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter tuning with RPS objective.
    """
    
    def __init__(
        self,
        n_trials: int | None = None,
        timeout: int | None = None,
        random_state: int = 42,
    ) -> None:
        settings = get_settings()
        self.n_trials = n_trials or settings.model.optuna_n_trials
        self.timeout = timeout
        self.random_state = random_state
    
    def optimize_catboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Optimize CatBoost hyperparameters."""
        from catboost import CatBoostClassifier
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 200, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            }
            
            model = CatBoostClassifier(
                **params,
                loss_function="MultiClass",
                verbose=False,
                random_state=self.random_state,
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)
            
            return ranked_probability_score(y_val, probs)
        
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)
        
        return study.best_params
    
    def optimize_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Optimize XGBoost hyperparameters."""
        import xgboost as xgb
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            }
            
            model = xgb.XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=3,
                verbosity=0,
                random_state=self.random_state,
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)
            
            return ranked_probability_score(y_val, probs)
        
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)
        
        return study.best_params
