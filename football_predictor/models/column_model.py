"""
Ensemble members that read probabilities straight from feature columns.

Some of the strongest signals arrive already shaped as probabilities: the
Dixon-Coles goal model's 1X2 output and the de-vigged market prices. Feeding
them to the gradient-boosted models as features lets those models dilute or
overfit them; the benchmark showed the goal model alone scoring better than
the whole classifier ensemble.

This wrapper exposes such a probability triplet as an ensemble member in its
own right, so the blender can weight it directly against the learned models.
It fits nothing except its own calibration map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from football_predictor.models.calibration import CalibrationMethod, ProbabilityCalibrator

_EPS = 1e-9

# Probability triplets the ensemble can lift out of the feature matrix
KNOWN_SOURCES: dict[str, tuple[str, str, str]] = {
    "dixon_coles": ("poisson_prob_home", "poisson_prob_draw", "poisson_prob_away"),
    "market": ("odds_prob_home", "odds_prob_draw", "odds_prob_away"),
}


class ColumnProbabilityModel:
    """
    Ensemble member backed by three precomputed probability columns.

    Rows where the columns are missing (an unpriced fixture, or a match before
    the goal model had enough history) fall back to the training class priors,
    so the member always returns a valid distribution.
    """

    def __init__(
        self,
        columns: Sequence[str],
        name: str = "column_probabilities",
        calibration_method: CalibrationMethod = "vector",
    ) -> None:
        if len(columns) != 3:
            raise ValueError("Expected exactly three probability columns")
        self.columns = tuple(columns)
        self.name = name
        self.calibration_method: CalibrationMethod = calibration_method

        self.calibrator: ProbabilityCalibrator | None = None
        self._indices: tuple[int, int, int] | None = None
        self._prior = np.array([1 / 3, 1 / 3, 1 / 3])

    @staticmethod
    def available(columns: Sequence[str], feature_names: Sequence[str]) -> bool:
        """True when every required column is present in the feature matrix."""
        return all(col in feature_names for col in columns)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        early_stopping_rounds: int | None = None,
        calibration_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "ColumnProbabilityModel":
        """
        Locate the columns and learn the fallback prior.

        ``eval_set`` and ``early_stopping_rounds`` are accepted for a uniform
        member interface and ignored: there is nothing to early-stop.
        """
        if feature_names is None:
            raise ValueError("ColumnProbabilityModel requires feature names")

        names = list(feature_names)
        missing = [col for col in self.columns if col not in names]
        if missing:
            raise ValueError(f"Missing probability columns: {missing}")

        self._indices = tuple(names.index(col) for col in self.columns)

        counts = np.bincount(np.asarray(y).astype(int), minlength=3)
        self._prior = counts / max(counts.sum(), 1)

        self.fit_calibration(calibration_set)
        return self

    def fit_calibration(
        self,
        calibration_set: tuple[np.ndarray, np.ndarray] | None,
    ) -> "ColumnProbabilityModel":
        """Fit the calibration map on held-out data (no-op when not supplied)."""
        if calibration_set is None:
            self.calibrator = None
            return self
        X_cal, y_cal = calibration_set
        self.calibrator = ProbabilityCalibrator(method=self.calibration_method)
        self.calibrator.fit(self.predict_proba_raw(X_cal), y_cal)
        return self

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Probabilities read from the columns, normalized, priors where absent."""
        if self._indices is None:
            raise RuntimeError("Model not fitted")

        X = np.asarray(X, dtype=float)
        probs = X[:, list(self._indices)].copy()

        totals = probs.sum(axis=1)
        # Missing features arrive as zeros after fillna, which is not a distribution
        usable = np.isfinite(probs).all(axis=1) & (totals > _EPS) & (probs >= 0).all(axis=1)

        out = np.tile(self._prior, (len(probs), 1))
        if usable.any():
            out[usable] = probs[usable] / totals[usable, None]
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probabilities when a calibrator was fitted."""
        probs = self.predict_proba_raw(X)
        if self.calibrator is not None:
            return self.calibrator.transform(probs)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_feature_importance(self) -> dict[str, float]:
        """This member uses exactly its own columns; nothing is learned."""
        return {}

    def get_params(self) -> dict[str, Any]:
        return {"columns": list(self.columns), "name": self.name}

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "columns": self.columns,
                "name": self.name,
                "indices": self._indices,
                "prior": self._prior,
                "calibration_method": self.calibration_method,
                "calibrator": self.calibrator,
            },
            str(path),
        )

    def load(self, path: str | Path) -> "ColumnProbabilityModel":
        state = joblib.load(str(path))
        self.columns = tuple(state["columns"])
        self.name = state["name"]
        self._indices = state["indices"]
        self._prior = state["prior"]
        self.calibration_method = state.get("calibration_method", "vector")
        self.calibrator = state.get("calibrator")
        return self
