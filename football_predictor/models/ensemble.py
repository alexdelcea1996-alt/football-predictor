"""
Ensemble predictor combining multiple models.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from football_predictor.config import get_settings
from football_predictor.models.blending import StackingBlender, WeightBlender

BlendMethod = Literal["weights", "stack", "equal"]

# Below this many validation rows, calibration and blend fitting are unreliable;
# the ensemble then trains on all data and averages members equally.
_MIN_VAL_SAMPLES = 60

_MODEL_EXTENSIONS = {"catboost": "cbm", "xgboost": "json", "logistic": "joblib"}


class EnsemblePredictor:
    """
    Ensemble of available models with a fitted blend.

    Training splits the (chronologically ordered) input into an earlier
    fitting block and a later validation block. Members are trained on the
    earlier block, their probabilities are calibrated on the validation block,
    and the blend is fitted there too. The validation block is never used to
    train member parameters, so blend weights reflect out-of-sample skill.

    This replaces the previous equal-weight average, which let a weak member
    drag the ensemble below its own best model.
    """

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        blend: BlendMethod = "weights",
        val_fraction: float = 0.2,
        calibrate: bool = True,
        refit_on_full: bool = False,
        random_state: int = 42,
    ) -> None:
        settings = get_settings()
        self.weights = list(weights) if weights is not None else list(settings.model.ensemble_weights)
        self.blend: BlendMethod = blend
        self.val_fraction = val_fraction
        self.calibrate = calibrate
        self.refit_on_full = refit_on_full
        self.random_state = random_state

        self._models: list[tuple[str, Any]] = []
        self._feature_names: list[str] = []
        self._blender: WeightBlender | StackingBlender | None = None
        self._fitted_on_val = False
        self._trained = False

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        val_set: tuple[np.ndarray, np.ndarray] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "EnsemblePredictor":
        """
        Train all available models and fit the blend.

        Args:
            X: Feature matrix in chronological order
            y: Labels (0=Home, 1=Draw, 2=Away)
            feature_names: Feature names
            val_set: Optional explicit (X_val, y_val) block that must come
                chronologically after X. When omitted, the last
                ``val_fraction`` of X is held out for calibration/blending.
            eval_set: Deprecated alias for ``val_set``. Passing the *test* set
                here (as the trainer used to) leaks it into early stopping and
                calibration.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        if eval_set is not None and val_set is None:
            warnings.warn(
                "eval_set is deprecated; pass val_set with a chronologically "
                "later validation block (never the test set).",
                DeprecationWarning,
                stacklevel=2,
            )
            val_set = eval_set

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        X_fit, y_fit, X_val, y_val = self._split_for_validation(X, y, val_set)
        use_val = X_val is not None and len(y_val) >= _MIN_VAL_SAMPLES

        if not use_val:
            X_fit, y_fit, X_val, y_val = X, y, None, None

        val_pair = (X_val, y_val) if use_val else None
        calib_pair = val_pair if (use_val and self.calibrate) else None

        self._models = self._build_models()
        if not self._models:
            raise RuntimeError("No models available. Install catboost, xgboost, or scikit-learn.")

        for _, model in self._models:
            model.fit(
                X_fit,
                y_fit,
                self._feature_names,
                eval_set=val_pair,
                calibration_set=calib_pair,
            )

        # Fit the blend on validation predictions (member-out-of-sample)
        if use_val:
            member_probs = [model.predict_proba(X_val) for _, model in self._models]
            self._blender = self._fit_blender(member_probs, y_val)
            self._fitted_on_val = True

            if self.refit_on_full:
                # Use every match for the final members; keep the calibrators
                # and blend learned on the held-out block.
                for _, model in self._models:
                    calibrator = getattr(model, "calibrator", None)
                    model.fit(X, y, self._feature_names)
                    model.calibrator = calibrator
        else:
            self._blender = WeightBlender([1.0] * len(self._models))
            self._fitted_on_val = False

        self._trained = True
        return self

    def _split_for_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_set: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return (X_fit, y_fit, X_val, y_val) using an explicit or temporal split."""
        if val_set is not None:
            X_val, y_val = np.asarray(val_set[0], dtype=float), np.asarray(val_set[1]).astype(int)
            return X, y, X_val, y_val

        if not 0.0 < self.val_fraction < 1.0:
            return X, y, None, None

        split_idx = int(len(y) * (1.0 - self.val_fraction))
        if split_idx <= 0 or split_idx >= len(y):
            return X, y, None, None

        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

    def _build_models(self) -> list[tuple[str, Any]]:
        """Instantiate whichever member models are installed."""
        models: list[tuple[str, Any]] = []

        try:
            from football_predictor.models.catboost_model import CatBoostModel
            models.append(("catboost", CatBoostModel(random_state=self.random_state)))
        except ImportError:
            pass

        try:
            from football_predictor.models.xgboost_model import XGBoostModel
            models.append(("xgboost", XGBoostModel(random_state=self.random_state)))
        except ImportError:
            pass

        try:
            from football_predictor.models.logistic_model import LogisticModel
            models.append(("logistic", LogisticModel(random_state=self.random_state)))
        except ImportError:
            pass

        return models

    def _fit_blender(
        self,
        member_probs: list[np.ndarray],
        y_val: np.ndarray,
    ) -> WeightBlender | StackingBlender:
        """Fit the configured blend on validation-set member probabilities."""
        if self.blend == "equal":
            return WeightBlender([1.0] * len(member_probs))

        if self.blend == "stack":
            stacker = StackingBlender()
            stacker.fit(member_probs, y_val)
            if stacker.fitted:
                return stacker
            # Degenerate meta-fit: fall back to weights rather than failing
            return WeightBlender().fit(member_probs, y_val)

        return WeightBlender().fit(member_probs, y_val)

    # -------------------------------------------------------------- predict

    def member_probabilities(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Per-member calibrated probabilities, for diagnostics."""
        if not self._trained:
            raise RuntimeError("Ensemble not trained")
        X = np.asarray(X, dtype=float)
        return {name: model.predict_proba(X) for name, model in self._models}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the fitted blend.

        Returns:
            Array of shape (n_samples, 3) with [P(Home), P(Draw), P(Away)]
        """
        if not self._trained:
            raise RuntimeError("Ensemble not trained")

        X = np.asarray(X, dtype=float)
        member_probs = [model.predict_proba(X) for _, model in self._models]

        if self._blender is None:
            self._blender = WeightBlender([1.0] * len(member_probs))

        return self._blender.transform(member_probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_with_confidence(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Predict with detailed probability breakdown."""
        probs = self.predict_proba(X)
        outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}

        return [
            {
                "predicted_outcome": outcome_map[int(np.argmax(prob))],
                "predicted_class": int(np.argmax(prob)),
                "home_win_prob": float(prob[0]),
                "draw_prob": float(prob[1]),
                "away_win_prob": float(prob[2]),
                "confidence": float(max(prob)),
            }
            for prob in probs
        ]

    # ------------------------------------------------------------ reporting

    def get_feature_importance(self) -> dict[str, float]:
        """Get averaged feature importance from tree models."""
        importance: dict[str, float] = {}
        count = 0

        for _, model in self._models:
            if hasattr(model, "get_feature_importance"):
                imp = model.get_feature_importance()
                if not imp:
                    continue
                for f, v in imp.items():
                    importance[f] = importance.get(f, 0) + float(v)
                count += 1

        if count > 0:
            for f in importance:
                importance[f] /= count

        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    def get_models_info(self) -> list[str]:
        """Get list of trained models."""
        return [name for name, _ in self._models]

    def get_blend_info(self) -> dict[str, Any]:
        """Describe the fitted blend (method, weights, whether it was fitted)."""
        info: dict[str, Any] = {
            "method": self.blend,
            "fitted_on_validation": self._fitted_on_val,
            "members": self.get_models_info(),
        }
        if isinstance(self._blender, WeightBlender) and self._blender.weights is not None:
            info["weights"] = {
                name: round(float(w), 4)
                for name, w in zip(self.get_models_info(), self._blender.weights)
            }
        elif isinstance(self._blender, StackingBlender):
            info["meta_learner"] = "logistic"
        return info

    # ------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> None:
        """Save all members, calibrators and the blend."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        for name, model in self._models:
            if hasattr(model, "save"):
                model.save(path / f"{name}.{_MODEL_EXTENSIONS.get(name, 'pkl')}")

        if self._blender is not None:
            self._blender.save(path / "blender.joblib")

        meta = {
            "weights": self.weights,
            "feature_names": self._feature_names,
            "models": [name for name, _ in self._models],
            "blend": self.blend,
            "calibrate": self.calibrate,
            "fitted_on_validation": self._fitted_on_val,
            "blend_info": self.get_blend_info(),
        }
        with open(path / "ensemble_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, directory: str | Path) -> "EnsemblePredictor":
        """Load all members, calibrators and the blend."""
        path = Path(directory)

        with open(path / "ensemble_meta.json") as f:
            meta = json.load(f)

        self.weights = meta.get("weights", self.weights)
        self._feature_names = meta.get("feature_names", [])
        self.blend = meta.get("blend", self.blend)
        self.calibrate = meta.get("calibrate", self.calibrate)
        self._fitted_on_val = meta.get("fitted_on_validation", False)
        self._models = []

        for name in meta.get("models", []):
            model_path = path / f"{name}.{_MODEL_EXTENSIONS.get(name, 'pkl')}"
            if not model_path.exists():
                continue
            try:
                if name == "catboost":
                    from football_predictor.models.catboost_model import CatBoostModel
                    self._models.append((name, CatBoostModel().load(model_path)))
                elif name == "xgboost":
                    from football_predictor.models.xgboost_model import XGBoostModel
                    self._models.append((name, XGBoostModel().load(model_path)))
                elif name == "logistic":
                    from football_predictor.models.logistic_model import LogisticModel
                    self._models.append((name, LogisticModel().load(model_path)))
            except ImportError:
                pass

        blender_path = path / "blender.joblib"
        if blender_path.exists():
            self._blender = (
                StackingBlender.load(blender_path)
                if self.blend == "stack"
                else WeightBlender.load(blender_path)
            )
        else:
            self._blender = None

        # A blend fitted for N members cannot be applied to a different N
        if (
            isinstance(self._blender, WeightBlender)
            and self._blender.weights is not None
            and len(self._blender.weights) != len(self._models)
        ):
            self._blender = None

        self._trained = bool(self._models)
        return self
