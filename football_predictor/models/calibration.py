"""
Probability calibration for multiclass match outcomes.

Why not ``CalibratedClassifierCV``: its cross-validated variants shuffle the
data into random K-folds, which on a time series means calibrating past
matches with information from future ones, and they refit the estimator
internally (discarding early stopping). This module instead fits calibration
maps on a chronologically held-out block of predictions, and can be persisted
alongside the model so production probabilities match the evaluated ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibrationMethod = Literal["isotonic", "vector", "none"]

_EPS = 1e-9
# Isotonic needs a reasonable number of points per class to not just memorize
_MIN_SAMPLES_PER_CLASS = 25


class ProbabilityCalibrator:
    """
    Fits a calibration map on held-out probabilities.

    Methods:
        isotonic: one-vs-rest isotonic regression per class, then renormalize.
            Flexible and non-parametric; needs a few hundred held-out samples.
        vector: multinomial logistic regression on log-probabilities (vector
            scaling). Only K*(K+1) parameters, so it works on small holdouts.
        none: identity (used as an automatic fallback when data is too thin).
    """

    def __init__(self, method: CalibrationMethod = "isotonic", n_classes: int = 3) -> None:
        self.method: CalibrationMethod = method
        self.n_classes = n_classes
        self._fitted = False
        self._effective_method: CalibrationMethod = "none"
        self._isotonics: list[IsotonicRegression] = []
        self._vector_model: LogisticRegression | None = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def effective_method(self) -> str:
        """The method actually used (may differ from the request on thin data)."""
        return self._effective_method

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fit the calibration map.

        Args:
            probs: Uncalibrated probabilities from held-out data, (n, K)
            y: True labels for the same held-out data, (n,)
        """
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y).astype(int)

        self.n_classes = probs.shape[1]
        self._isotonics = []
        self._vector_model = None
        self._fitted = True

        if self.method == "none" or len(y) == 0:
            self._effective_method = "none"
            return self

        counts = np.bincount(y, minlength=self.n_classes)
        method = self.method

        # Every class must be represented, otherwise a map cannot be learned
        if (counts == 0).any():
            self._effective_method = "none"
            return self

        # Isotonic on a thin holdout overfits badly; vector scaling degrades better
        if method == "isotonic" and counts.min() < _MIN_SAMPLES_PER_CLASS:
            method = "vector"

        if method == "isotonic":
            for k in range(self.n_classes):
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(probs[:, k], (y == k).astype(float))
                self._isotonics.append(iso)
            self._effective_method = "isotonic"
        else:
            log_probs = np.log(np.clip(probs, _EPS, 1.0))
            model = LogisticRegression(max_iter=1000, C=1.0)
            model.fit(log_probs, y)
            # A degenerate holdout can leave classes out of the fitted model
            if len(model.classes_) != self.n_classes:
                self._effective_method = "none"
                return self
            self._vector_model = model
            self._effective_method = "vector"

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply the fitted calibration map and renormalize to sum to 1."""
        probs = np.asarray(probs, dtype=float)

        if not self._fitted or self._effective_method == "none":
            return _normalize(probs)

        if self._effective_method == "isotonic":
            out = np.column_stack([
                iso.predict(probs[:, k]) for k, iso in enumerate(self._isotonics)
            ])
            return _normalize(out)

        assert self._vector_model is not None
        log_probs = np.log(np.clip(probs, _EPS, 1.0))
        return _normalize(self._vector_model.predict_proba(log_probs))

    def fit_transform(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(probs, y).transform(probs)

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "method": self.method,
                "effective_method": self._effective_method,
                "n_classes": self.n_classes,
                "isotonics": self._isotonics,
                "vector_model": self._vector_model,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        state: dict[str, Any] = joblib.load(str(path))
        obj = cls(method=state["method"], n_classes=state["n_classes"])
        obj._effective_method = state["effective_method"]
        obj._isotonics = state["isotonics"]
        obj._vector_model = state["vector_model"]
        obj._fitted = True
        return obj


def _normalize(probs: np.ndarray) -> np.ndarray:
    """Clip negatives and renormalize rows; uniform where a row sums to zero."""
    out = np.clip(np.asarray(probs, dtype=float), 0.0, None)
    totals = out.sum(axis=1, keepdims=True)
    degenerate = totals[:, 0] <= _EPS
    if degenerate.any():
        out[degenerate] = 1.0 / out.shape[1]
        totals = out.sum(axis=1, keepdims=True)
    return out / totals
