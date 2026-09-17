"""
XGBoost model wrapper for football prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from football_predictor.config import get_settings
from football_predictor.models.calibration import CalibrationMethod, ProbabilityCalibrator
from football_predictor.models.params import load_tuned_params, resolve_param

_EXPLICIT_KEYS = {
    "n_estimators", "learning_rate", "max_depth", "reg_lambda",
    "subsample", "colsample_bytree",
}


class XGBoostModel:
    """
    XGBoost classifier wrapper.

    Hyperparameters resolve as: explicit argument > models/best_params.json >
    config.py default. Calibration is fitted on an explicitly supplied,
    chronologically later block and persisted with the model.
    """

    def __init__(
        self,
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        max_depth: int | None = None,
        reg_lambda: float | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        random_state: int = 42,
        tuned_params_path: str | Path | None = None,
        calibration_method: CalibrationMethod = "isotonic",
    ) -> None:
        settings = get_settings()
        tuned = load_tuned_params("xgboost", tuned_params_path)

        self.n_estimators = resolve_param(
            n_estimators, tuned, "n_estimators", settings.model.xgboost_n_estimators
        )
        self.learning_rate = resolve_param(
            learning_rate, tuned, "learning_rate", settings.model.xgboost_learning_rate
        )
        self.max_depth = resolve_param(
            max_depth, tuned, "max_depth", settings.model.xgboost_max_depth
        )
        self.reg_lambda = resolve_param(
            reg_lambda, tuned, "reg_lambda", settings.model.xgboost_reg_lambda
        )
        self.subsample = resolve_param(subsample, tuned, "subsample", 0.8)
        self.colsample_bytree = resolve_param(
            colsample_bytree, tuned, "colsample_bytree", 0.8
        )
        self.extra_params = {k: v for k, v in tuned.items() if k not in _EXPLICIT_KEYS}
        self.random_state = random_state
        self.calibration_method: CalibrationMethod = calibration_method

        self.model: xgb.XGBClassifier | None = None
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
    ) -> "XGBoostModel":
        """Train the XGBoost model (see CatBoostModel.fit for argument semantics)."""
        settings = get_settings()
        early_stop = early_stopping_rounds or settings.model.early_stopping_rounds

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            early_stopping_rounds=early_stop if eval_set else None,
            verbosity=0,
            **self.extra_params,
        )

        if eval_set:
            self.model.fit(X, y, eval_set=[eval_set], verbose=False)
        else:
            self.model.fit(X, y)

        self.fit_calibration(calibration_set)
        return self

    def fit_calibration(
        self,
        calibration_set: tuple[np.ndarray, np.ndarray] | None,
    ) -> "XGBoostModel":
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
        """Predict class probabilities, calibrated when a calibrator was fitted."""
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
        return dict(zip(self._feature_names, self.model.feature_importances_))

    def get_params(self) -> dict[str, Any]:
        """Hyperparameters actually in use (for reporting/reproducibility)."""
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            **self.extra_params,
        }

    def save(self, path: str | Path) -> None:
        """Save model and its calibrator."""
        if self.model:
            self.model.save_model(str(path))
        if self.calibrator is not None:
            self.calibrator.save(_calibrator_path(path))

    def load(self, path: str | Path) -> "XGBoostModel":
        """Load model and its calibrator."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        calib_path = _calibrator_path(path)
        self.calibrator = (
            ProbabilityCalibrator.load(calib_path) if Path(calib_path).exists() else None
        )
        return self


def _calibrator_path(path: str | Path) -> str:
    return f"{path}.calibrator.joblib"
