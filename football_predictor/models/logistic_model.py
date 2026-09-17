"""
Logistic Regression model for football prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from football_predictor.config import get_settings
from football_predictor.models.calibration import CalibrationMethod, ProbabilityCalibrator
from football_predictor.models.params import load_tuned_params, resolve_param

_EXPLICIT_KEYS = {"C", "solver", "max_iter"}


class LogisticModel:
    """
    Multinomial logistic regression with built-in normalization.

    Interpretable, naturally well-calibrated, and a strong baseline on this
    problem. Hyperparameters resolve as: explicit argument >
    models/best_params.json > config.py default.
    """

    def __init__(
        self,
        C: float | None = None,
        solver: str | None = None,
        max_iter: int | None = None,
        random_state: int = 42,
        tuned_params_path: str | Path | None = None,
        calibration_method: CalibrationMethod = "vector",
    ) -> None:
        settings = get_settings()
        tuned = load_tuned_params("logistic", tuned_params_path)

        self.C = resolve_param(C, tuned, "C", settings.model.logreg_c)
        self.solver = resolve_param(solver, tuned, "solver", "lbfgs")
        self.max_iter = resolve_param(max_iter, tuned, "max_iter", 1000)
        self.extra_params = {k: v for k, v in tuned.items() if k not in _EXPLICIT_KEYS}
        self.random_state = random_state
        self.calibration_method: CalibrationMethod = calibration_method

        self.scaler = StandardScaler()
        self.model: LogisticRegression | None = None
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
    ) -> "LogisticModel":
        """
        Train the logistic regression model.

        ``eval_set`` and ``early_stopping_rounds`` are accepted for a uniform
        model interface and ignored (the solver runs to convergence).
        """
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        X_scaled = self.scaler.fit_transform(X)

        # scikit-learn >= 1.7 dropped multi_class; multinomial is the default
        # for multiclass targets with the solvers used here.
        self.model = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.extra_params,
        )
        self.model.fit(X_scaled, y)

        self.fit_calibration(calibration_set)
        return self

    def fit_calibration(
        self,
        calibration_set: tuple[np.ndarray, np.ndarray] | None,
    ) -> "LogisticModel":
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
        return np.asarray(self.model.predict_proba(self.scaler.transform(X)), dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities, calibrated when a calibrator was fitted."""
        probs = self.predict_proba_raw(X)
        if self.calibrator is not None:
            return self.calibrator.transform(probs)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)

    def get_coefficients(self) -> dict[str, Any]:
        """Get model coefficients for interpretability."""
        if self.model is None:
            return {}
        return {
            "classes": self.model.classes_.tolist(),
            "coefficients": dict(zip(self._feature_names, self.model.coef_.T.tolist())),
            "intercepts": self.model.intercept_.tolist(),
        }

    def get_params(self) -> dict[str, Any]:
        """Hyperparameters actually in use (for reporting/reproducibility)."""
        return {"C": self.C, "solver": self.solver, "max_iter": self.max_iter, **self.extra_params}

    def save(self, path: str | Path) -> None:
        """Save model, scaler and calibrator."""
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "calibrator": self.calibrator},
            str(path),
        )

    def load(self, path: str | Path) -> "LogisticModel":
        """Load model, scaler and calibrator."""
        data = joblib.load(str(path))
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.calibrator = data.get("calibrator")
        return self
