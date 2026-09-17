"""
Ensemble blenders.

The original ensemble averaged every member with equal weight regardless of
how well each one scored, so a weak member dragged the ensemble below its own
best model. These blenders are fitted on held-out predictions:

    WeightBlender:   simplex weights chosen to minimize RPS (convex problem)
    StackingBlender: multinomial logistic meta-learner over member probabilities
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from football_predictor.evaluation.metrics import ranked_probability_score

_EPS = 1e-9


def _stack_members(member_probs: Sequence[np.ndarray]) -> np.ndarray:
    """Stack member probabilities into an (n_models, n_samples, n_classes) array."""
    if not member_probs:
        raise ValueError("No member probabilities supplied")
    arr = np.stack([np.asarray(p, dtype=float) for p in member_probs], axis=0)
    if arr.ndim != 3:
        raise ValueError(f"Expected member probabilities of shape (n, K), got {arr.shape}")
    return arr


def _normalize(probs: np.ndarray) -> np.ndarray:
    out = np.clip(np.asarray(probs, dtype=float), 0.0, None)
    totals = out.sum(axis=1, keepdims=True)
    degenerate = totals[:, 0] <= _EPS
    if degenerate.any():
        out[degenerate] = 1.0 / out.shape[1]
        totals = out.sum(axis=1, keepdims=True)
    return out / totals


class WeightBlender:
    """Weighted average of member probabilities, weights fitted to minimize RPS."""

    def __init__(self, weights: Sequence[float] | None = None) -> None:
        self.weights: np.ndarray | None = (
            _normalize_weights(np.asarray(weights, dtype=float)) if weights is not None else None
        )
        self._fitted = weights is not None

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, member_probs: Sequence[np.ndarray], y: np.ndarray) -> "WeightBlender":
        """
        Fit weights on held-out member predictions.

        RPS is convex in the predicted CDF and the blend is linear in the
        weights, so the objective is convex over the simplex and a single
        SLSQP run from equal weights finds the optimum.
        """
        stacked = _stack_members(member_probs)
        y = np.asarray(y).astype(int)
        n_models = stacked.shape[0]

        if n_models == 1 or len(y) == 0:
            self.weights = np.ones(n_models) / n_models
            self._fitted = True
            return self

        def objective(w: np.ndarray) -> float:
            w = np.clip(w, 0.0, None)
            total = w.sum()
            if total <= _EPS:
                return 1.0
            blended = np.tensordot(w / total, stacked, axes=(0, 0))
            return ranked_probability_score(y, _normalize(blended))

        start = np.ones(n_models) / n_models
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 200, "ftol": 1e-9},
        )

        candidate = _normalize_weights(result.x) if result.success else start
        # Never accept a fit that scores worse than the equal-weight baseline
        self.weights = candidate if objective(candidate) <= objective(start) else start
        self._fitted = True
        return self

    def transform(self, member_probs: Sequence[np.ndarray]) -> np.ndarray:
        stacked = _stack_members(member_probs)
        if self.weights is None or len(self.weights) != stacked.shape[0]:
            weights = np.ones(stacked.shape[0]) / stacked.shape[0]
        else:
            weights = self.weights
        return _normalize(np.tensordot(weights, stacked, axes=(0, 0)))

    def save(self, path: str | Path) -> None:
        joblib.dump({"weights": self.weights}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "WeightBlender":
        state: dict[str, Any] = joblib.load(str(path))
        obj = cls()
        obj.weights = state["weights"]
        obj._fitted = obj.weights is not None
        return obj


class StackingBlender:
    """Multinomial logistic meta-learner over member probabilities."""

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self.model: LogisticRegression | None = None
        self.n_models: int | None = None

    @property
    def fitted(self) -> bool:
        return self.model is not None

    @staticmethod
    def _design_matrix(stacked: np.ndarray) -> np.ndarray:
        """(n_models, n, K) -> (n, n_models * K) meta-features."""
        return np.concatenate(list(stacked), axis=1)

    def fit(self, member_probs: Sequence[np.ndarray], y: np.ndarray) -> "StackingBlender":
        stacked = _stack_members(member_probs)
        y = np.asarray(y).astype(int)
        self.n_models = stacked.shape[0]

        X = self._design_matrix(stacked)
        model = LogisticRegression(C=self.C, max_iter=2000)
        model.fit(X, y)

        # A meta-learner that never saw a class cannot predict it
        if len(model.classes_) != stacked.shape[2]:
            self.model = None
            return self

        self.model = model
        return self

    def transform(self, member_probs: Sequence[np.ndarray]) -> np.ndarray:
        stacked = _stack_members(member_probs)
        if self.model is None:
            raise RuntimeError("StackingBlender is not fitted")
        if self.n_models is not None and stacked.shape[0] != self.n_models:
            raise ValueError(
                f"Expected {self.n_models} members, got {stacked.shape[0]}"
            )
        return _normalize(self.model.predict_proba(self._design_matrix(stacked)))

    def save(self, path: str | Path) -> None:
        joblib.dump({"model": self.model, "n_models": self.n_models, "C": self.C}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "StackingBlender":
        state: dict[str, Any] = joblib.load(str(path))
        obj = cls(C=state.get("C", 1.0))
        obj.model = state["model"]
        obj.n_models = state["n_models"]
        return obj


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = w.sum()
    if total <= _EPS:
        return np.ones_like(w) / len(w)
    return w / total
