"""
CatBoost model wrapper for football prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostClassifier

from football_predictor.config import get_settings
from football_predictor.models.calibration import CalibrationMethod, ProbabilityCalibrator
from football_predictor.models.params import load_tuned_params, resolve_param

# Parameters handled explicitly; anything else in the tuned file is passed through
_EXPLICIT_KEYS = {"iterations", "learning_rate", "depth", "l2_leaf_reg"}


class CatBoostModel:
    """
    CatBoost classifier wrapper.

    Hyperparameters resolve as: explicit argument > models/best_params.json >
    config.py default. Probability calibration is fitted on an explicitly
    supplied, chronologically later block (see ``fit``) and is persisted with
    the model, so loaded models return the same probabilities that were
    evaluated at training time.
    """

    def __init__(
        self,
        iterations: int | None = None,
        learning_rate: float | None = None,
        depth: int | None = None,
        l2_leaf_reg: float | None = None,
        random_state: int = 42,
        tuned_params_path: str | Path | None = None,
        calibration_method: CalibrationMethod = "isotonic",
    ) -> None:
        settings = get_settings()
        tuned = load_tuned_params("catboost", tuned_params_path)

        self.iterations = resolve_param(
            iterations, tuned, "iterations", settings.model.catboost_iterations
        )
        self.learning_rate = resolve_param(
            learning_rate, tuned, "learning_rate", settings.model.catboost_learning_rate
        )
        self.depth = resolve_param(depth, tuned, "depth", settings.model.catboost_depth)
        self.l2_leaf_reg = resolve_param(
            l2_leaf_reg, tuned, "l2_leaf_reg", settings.model.catboost_l2_leaf_reg
        )
        self.extra_params = {k: v for k, v in tuned.items() if k not in _EXPLICIT_KEYS}
        self.random_state = random_state
        self.calibration_method: CalibrationMethod = calibration_method

        self.model: CatBoostClassifier | None = None
        self.calibrator: ProbabilityCalibrator | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        early_stopping_rounds: int | None = None,
        calibration_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "CatBoostModel":
        """
        Train the CatBoost model.

        Args:
            X: Feature matrix
            y: Target labels (0=Home, 1=Draw, 2=Away)
            feature_names: Names of features
            eval_set: Optional (X_val, y_val) for early stopping
            early_stopping_rounds: Stop if no improvement
            calibration_set: Optional (X_cal, y_cal) block, which must come
                chronologically after the training data, used to fit the
                probability calibration map
        """
        settings = get_settings()
        early_stop = early_stopping_rounds or settings.model.early_stopping_rounds

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_state=self.random_state,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            verbose=False,
            early_stopping_rounds=early_stop if eval_set else None,
            **self.extra_params,
        )

        if eval_set:
            self.model.fit(X, y, eval_set=[eval_set])
        else:
            self.model.fit(X, y)

        self.fit_calibration(calibration_set)
        return self

    def fit_calibration(
        self,
        calibration_set: tuple[np.ndarray, np.ndarray] | None,
    ) -> "CatBoostModel":
        """Fit the calibration map on held-out data (no-op when not supplied)."""
        if calibration_set is None:
            self.calibrator = None
            return self
        X_cal, y_cal = calibration_set
        self.calibrator = ProbabilityCalibrator(method=self.calibration_method)
        self.calibrator.fit(self.predict_proba_raw(X_cal), y_cal)
        return self

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        return np.asarray(self.model.predict_proba(X), dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities, calibrated when a calibrator was fitted.

        Returns:
            Array of shape (n_samples, 3) with [P(Home), P(Draw), P(Away)]
        """
        probs = self.predict_proba_raw(X)
        if self.calibrator is not None:
            return self.calibrator.transform(probs)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}
        importance = self.model.get_feature_importance()
        return dict(zip(self._feature_names, importance))

    def get_params(self) -> dict[str, Any]:
        """Hyperparameters actually in use (for reporting/reproducibility)."""
        return {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            **self.extra_params,
        }

    def save(self, path: str | Path) -> None:
        """Save model and its calibrator."""
        if self.model:
            self.model.save_model(str(path))
        if self.calibrator is not None:
            self.calibrator.save(_calibrator_path(path))

    def load(self, path: str | Path) -> "CatBoostModel":
        """Load model and its calibrator."""
        self.model = CatBoostClassifier()
        self.model.load_model(str(path))
        calib_path = _calibrator_path(path)
        self.calibrator = (
            ProbabilityCalibrator.load(calib_path) if Path(calib_path).exists() else None
        )
        return self


def _calibrator_path(path: str | Path) -> str:
    return f"{path}.calibrator.joblib"
