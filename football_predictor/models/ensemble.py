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
from football_predictor.evaluation.metrics import ranked_probability_score
from football_predictor.models.blending import StackingBlender, WeightBlender
from football_predictor.models.calibration import ProbabilityCalibrator
from football_predictor.models.column_model import KNOWN_SOURCES, ColumnProbabilityModel

BlendMethod = Literal["weights", "stack", "equal"]
BlendStrategy = Literal["oof", "holdout"]

# Below this many validation rows, calibration and blend fitting are unreliable;
# the ensemble then trains on all data and averages members equally.
_MIN_VAL_SAMPLES = 60

# Below this, walking folds leaves too little per fold to be worth the cost
_MIN_OOF_SAMPLES = 300

_MODEL_EXTENSIONS = {"catboost": "cbm", "xgboost": "json", "logistic": "joblib"}
_COLUMN_MEMBER_EXTENSION = "joblib"


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
        include_probability_members: bool = True,
        blend_strategy: BlendStrategy = "oof",
        blend_folds: int = 3,
    ) -> None:
        settings = get_settings()
        self.weights = list(weights) if weights is not None else list(settings.model.ensemble_weights)
        self.blend: BlendMethod = blend
        self.val_fraction = val_fraction
        self.calibrate = calibrate
        self.refit_on_full = refit_on_full
        self.random_state = random_state
        self.include_probability_members = include_probability_members
        self.blend_strategy: BlendStrategy = blend_strategy
        self.blend_folds = blend_folds

        self._models: list[tuple[str, Any]] = []
        self._feature_names: list[str] = []
        self._blender: WeightBlender | StackingBlender | None = None
        self._fitted_on_val = False
        self._blend_fit_samples = 0
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

        With the default ``blend_strategy="oof"`` and no explicit validation
        block, members train on all of X and the calibration maps and blend
        weights are fitted on out-of-fold predictions from temporal folds
        inside it. Supplying ``val_set`` (or ``blend_strategy="holdout"``)
        switches to a single chronological validation block instead: cheaper,
        but the blend is fitted on far fewer rows.

        Args:
            X: Feature matrix in chronological order
            y: Labels (0=Home, 1=Draw, 2=Away)
            feature_names: Feature names
            val_set: Optional explicit (X_val, y_val) block that must come
                chronologically after X. When omitted under the holdout
                strategy, the last ``val_fraction`` of X is held out.
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

        if self.blend_strategy == "oof" and val_set is None and len(y) >= _MIN_OOF_SAMPLES:
            return self._fit_out_of_fold(X, y)

        X_fit, y_fit, X_val, y_val = self._split_for_validation(X, y, val_set)
        return self._fit_holdout(X, y, X_fit, y_fit, X_val, y_val)

    def _fit_holdout(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> "EnsemblePredictor":
        """Fit members on an earlier block, calibrate and blend on a later one."""
        use_val = X_val is not None and len(y_val) >= _MIN_VAL_SAMPLES

        if not use_val:
            X_fit, y_fit, X_val, y_val = X, y, None, None

        val_pair = (X_val, y_val) if use_val else None
        calib_pair = val_pair if (use_val and self.calibrate) else None

        self._models = self._build_models(self._feature_names)
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
            self._blend_fit_samples = int(len(y_val))

            if self.refit_on_full:
                # Refit on every match, keeping the calibrators and blend from
                # the held-out block. Benchmarked as clearly worse (the maps
                # were learned from a model trained on less data, so they
                # mis-sharpen the refitted one) - off by default, kept for
                # experiments on much larger datasets.
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

    def _fit_out_of_fold(self, X: np.ndarray, y: np.ndarray) -> "EnsemblePredictor":
        """
        Fit calibration and blending on out-of-fold predictions.

        A single held-out block is small: on a season or two of matches it is
        a couple of hundred rows, and weights fitted there swing wildly (the
        benchmark showed the best member being given zero weight). Walking
        temporal folds over the training data instead produces several times
        as many honest predictions to fit on, and lets the final members train
        on every match rather than giving a fifth of them up.
        """
        from football_predictor.training.cv_strategy import TemporalCrossValidator

        names = self._feature_names
        template = self._build_models(names)
        if not template:
            raise RuntimeError("No models available. Install catboost, xgboost, or scikit-learn.")

        member_names = [name for name, _ in template]
        collected: dict[str, list[np.ndarray]] = {name: [] for name in member_names}
        collected_y: list[np.ndarray] = []

        cv = TemporalCrossValidator(n_splits=self.blend_folds)
        for train_idx, test_idx in cv.split(X, y):
            for name, model in self._build_models(names):
                model.fit(X[train_idx], y[train_idx], names)
                collected[name].append(_raw_probabilities(model, X[test_idx]))
            collected_y.append(y[test_idx])

        if not collected_y:
            # Not enough data to walk folds; fall back to the holdout path
            X_fit, y_fit, X_val, y_val = self._split_for_validation(X, y, None)
            return self._fit_holdout(X, y, X_fit, y_fit, X_val, y_val)

        oof_y = np.concatenate(collected_y)
        oof_probs = {name: np.vstack(parts) for name, parts in collected.items()}

        # Final members see every match
        self._models = self._build_models(names)
        for name, model in self._models:
            model.fit(X, y, names)
            if self.calibrate:
                model.calibrator = ProbabilityCalibrator(
                    method=getattr(model, "calibration_method", "isotonic")
                ).fit(oof_probs[name], oof_y)

        calibrated = [
            model.calibrator.transform(oof_probs[name])
            if getattr(model, "calibrator", None) is not None
            else oof_probs[name]
            for name, model in self._models
        ]

        self._blender = self._fit_blender(calibrated, oof_y)
        self._fitted_on_val = True
        self._blend_fit_samples = int(len(oof_y))
        self._trained = True
        return self

    def _build_models(self, feature_names: list[str] | None = None) -> list[tuple[str, Any]]:
        """Instantiate whichever member models are installed and applicable."""
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

        # Probability columns (goal model, market prices) as members in their
        # own right, so the blender can weight them against the classifiers
        # instead of leaving them to be diluted as two features among eighty.
        if self.include_probability_members and feature_names:
            for name, columns in KNOWN_SOURCES.items():
                if ColumnProbabilityModel.available(columns, feature_names):
                    models.append((name, ColumnProbabilityModel(columns, name=name)))

        return models

    def _fit_blender(
        self,
        member_probs: list[np.ndarray],
        y_val: np.ndarray,
    ) -> WeightBlender | StackingBlender:
        """
        Fit the configured blend on validation-set member probabilities.

        Fitted weights can always fall back on "everything to the best member",
        so they cannot be beaten by a single member on the validation block.
        A stacking meta-learner has no such guarantee, so it is kept only when
        it actually scores better there.
        """
        if self.blend == "equal":
            return WeightBlender([1.0] * len(member_probs))

        weighted = WeightBlender().fit(member_probs, y_val)

        if self.blend != "stack":
            return weighted

        stacker = StackingBlender()
        stacker.fit(member_probs, y_val)
        if not stacker.fitted:
            return weighted

        stacked_score = ranked_probability_score(y_val, stacker.transform(member_probs))
        weighted_score = ranked_probability_score(y_val, weighted.transform(member_probs))

        if stacked_score < weighted_score:
            return stacker

        self.blend = "weights"  # so persistence reloads the right blender type
        return weighted

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
            "strategy": self.blend_strategy,
            "fitted_on_validation": self._fitted_on_val,
            "blend_fit_samples": self._blend_fit_samples,
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
            if not hasattr(model, "save"):
                continue
            extension = (
                _COLUMN_MEMBER_EXTENSION if name in KNOWN_SOURCES
                else _MODEL_EXTENSIONS.get(name, "pkl")
            )
            model.save(path / f"{name}.{extension}")

        if self._blender is not None:
            self._blender.save(path / "blender.joblib")

        meta = {
            "weights": self.weights,
            "feature_names": self._feature_names,
            "models": [name for name, _ in self._models],
            "blend": self.blend,
            "blend_strategy": self.blend_strategy,
            "calibrate": self.calibrate,
            "fitted_on_validation": self._fitted_on_val,
            "blend_fit_samples": self._blend_fit_samples,
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
        self.blend_strategy = meta.get("blend_strategy", self.blend_strategy)
        self.calibrate = meta.get("calibrate", self.calibrate)
        self._fitted_on_val = meta.get("fitted_on_validation", False)
        self._blend_fit_samples = meta.get("blend_fit_samples", 0)
        self._models = []

        for name in meta.get("models", []):
            if name in KNOWN_SOURCES:
                member_path = path / f"{name}.{_COLUMN_MEMBER_EXTENSION}"
                if member_path.exists():
                    self._models.append(
                        (name, ColumnProbabilityModel(KNOWN_SOURCES[name], name=name).load(member_path))
                    )
                continue

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


def _raw_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    """Uncalibrated probabilities, for members that expose them."""
    if hasattr(model, "predict_proba_raw"):
        return model.predict_proba_raw(X)
    return model.predict_proba(X)
