"""
Main training orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_predictor.config import get_settings
from football_predictor.data.preprocessor import DataPreprocessor
from football_predictor.evaluation.metrics import calculate_all_metrics
from football_predictor.features.aggregator import FeatureAggregator
from football_predictor.models.ensemble import BlendMethod, EnsemblePredictor
from football_predictor.training.cv_strategy import TemporalCrossValidator


class ModelTrainer:
    """
    Orchestrates the complete training pipeline.

    Data is split chronologically into three blocks:

        [------------ train ------------][-- val --][-- test --]

    Members are fitted on train, early-stopped and calibrated on val, the
    ensemble blend is fitted on val, and test is touched only for the final
    metrics. Previously the test set was passed as the early-stopping and
    calibration set, which inflated the reported test scores.
    """

    def __init__(self, output_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self.output_dir = Path(output_dir) if output_dir else settings.models_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_aggregator = FeatureAggregator()
        self.preprocessor = DataPreprocessor()
        self.ensemble: EnsemblePredictor | None = None
        self._feature_names: list[str] = []

    def prepare_data(
        self,
        df: pd.DataFrame,
        standings_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Prepare data with all features."""
        df = self.preprocessor.encode_outcome(df)
        df = self.feature_aggregator.process_matches(df, standings_df)
        return df

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.15,
        standings_df: pd.DataFrame | None = None,
        blend: BlendMethod = "weights",
        refit_on_full: bool = False,
    ) -> dict[str, Any]:
        """
        Train the ensemble model.

        Args:
            df: Raw match data
            test_size: Final chronological holdout, used only for reporting
            val_size: Block before the test set used for early stopping,
                calibration and blend fitting
            standings_df: Optional standings data
            blend: "weights" (RPS-optimal), "stack" (meta-learner) or "equal"
            refit_on_full: Refit members on train+val after the blend is
                fitted (uses all data, keeps the held-out calibration maps)

        Returns:
            Training metrics and results
        """
        df = self.prepare_data(df, standings_df)
        df = df[df["outcome"].notna()].reset_index(drop=True)

        X, feature_names = self.feature_aggregator.get_feature_matrix(df)
        y = df["outcome"].astype(int).values
        self._feature_names = feature_names

        n = len(y)
        test_start = int(n * (1 - test_size))
        val_start = int(test_start * (1 - val_size))

        X_train, y_train = X[:val_start], y[:val_start]
        X_val, y_val = X[val_start:test_start], y[val_start:test_start]
        X_test, y_test = X[test_start:], y[test_start:]

        self.ensemble = EnsemblePredictor(blend=blend, refit_on_full=refit_on_full)
        self.ensemble.fit(
            X_train,
            y_train,
            feature_names,
            val_set=(X_val, y_val) if len(y_val) > 0 else None,
        )

        train_metrics = calculate_all_metrics(y_train, self.ensemble.predict_proba(X_train))
        test_metrics = calculate_all_metrics(y_test, self.ensemble.predict_proba(X_test))

        # Per-member test metrics make it obvious when a member is dragging
        # the ensemble down (the failure mode the equal-weight average hid).
        member_metrics: dict[str, dict[str, float]] = {}
        if len(y_test) > 0:
            for name, probs in self.ensemble.member_probabilities(X_test).items():
                member_metrics[name] = calculate_all_metrics(y_test, probs)

        self.ensemble.save(self.output_dir / "ensemble")

        agg_state = self.feature_aggregator.save_state()
        with open(self.output_dir / "feature_state.json", "w") as f:
            json.dump(agg_state, f, default=str)

        return {
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "test_samples": len(y_test),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "member_test_metrics": member_metrics,
            "blend": self.ensemble.get_blend_info(),
            "feature_importance": self.ensemble.get_feature_importance(),
        }

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        standings_df: pd.DataFrame | None = None,
        blend: BlendMethod = "weights",
    ) -> dict[str, Any]:
        """
        Run temporal cross-validation.

        Each fold trains on all earlier matches and scores the next block, so
        the averages reflect what the model would have achieved in sequence.
        """
        df = self.prepare_data(df, standings_df)
        df = df[df["outcome"].notna()].reset_index(drop=True)

        X, feature_names = self.feature_aggregator.get_feature_matrix(df)
        y = df["outcome"].astype(int).values

        cv = TemporalCrossValidator(n_splits=n_folds)

        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = EnsemblePredictor(blend=blend)
            model.fit(X_train, y_train, feature_names)

            metrics = calculate_all_metrics(y_test, model.predict_proba(X_test))
            metrics["fold"] = fold
            metrics["train_samples"] = len(y_train)
            metrics["test_samples"] = len(y_test)
            fold_metrics.append(metrics)

        if not fold_metrics:
            raise ValueError(
                "Temporal cross-validation produced no folds; "
                "not enough matches for the requested number of splits."
            )

        avg_metrics = {
            key: float(np.mean([m[key] for m in fold_metrics]))
            for key in ("rps", "accuracy", "log_loss", "ece", "f1_macro")
            if key in fold_metrics[0]
        }

        return {
            "fold_metrics": fold_metrics,
            "average_metrics": avg_metrics,
        }

    def load(self, model_dir: str | Path | None = None) -> "ModelTrainer":
        """Load a trained model."""
        path = Path(model_dir) if model_dir else self.output_dir

        self.ensemble = EnsemblePredictor()
        self.ensemble.load(path / "ensemble")

        with open(path / "feature_state.json") as f:
            state = json.load(f)
        self.feature_aggregator.load_state(state)

        return self
