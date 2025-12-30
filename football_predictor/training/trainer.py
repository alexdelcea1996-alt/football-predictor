"""
Main training orchestrator.
"""

from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

from football_predictor.config import get_settings
from football_predictor.features.aggregator import FeatureAggregator
from football_predictor.data.preprocessor import DataPreprocessor
from football_predictor.models.ensemble import EnsemblePredictor
from football_predictor.training.cv_strategy import TemporalCrossValidator
from football_predictor.evaluation.metrics import calculate_all_metrics


class ModelTrainer:
    """
    Orchestrates the complete training pipeline.
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
        # Encode outcomes
        df = self.preprocessor.encode_outcome(df)
        
        # Compute features
        df = self.feature_aggregator.process_matches(df, standings_df)
        
        return df
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        standings_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            df: Raw match data
            test_size: Proportion for temporal test set
            standings_df: Optional standings data
        
        Returns:
            Training metrics and results
        """
        # Prepare data
        df = self.prepare_data(df, standings_df)
        
        # Filter finished matches with valid outcomes
        df = df[df["outcome"].notna()].reset_index(drop=True)
        
        # Temporal split
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Extract features
        X_train, feature_names = self.feature_aggregator.get_feature_matrix(train_df)
        y_train = train_df["outcome"].astype(int).values
        
        X_test, _ = self.feature_aggregator.get_feature_matrix(test_df)
        y_test = test_df["outcome"].astype(int).values
        
        self._feature_names = feature_names
        
        # Train ensemble
        self.ensemble = EnsemblePredictor()
        self.ensemble.fit(X_train, y_train, feature_names, eval_set=(X_test, y_test))
        
        # Evaluate
        train_probs = self.ensemble.predict_proba(X_train)
        test_probs = self.ensemble.predict_proba(X_test)
        
        train_metrics = calculate_all_metrics(y_train, train_probs)
        test_metrics = calculate_all_metrics(y_test, test_probs)
        
        # Save model
        self.ensemble.save(self.output_dir / "ensemble")
        
        # Save feature aggregator state
        agg_state = self.feature_aggregator.save_state()
        with open(self.output_dir / "feature_state.json", "w") as f:
            json.dump(agg_state, f, default=str)
        
        return {
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": self.ensemble.get_feature_importance(),
        }
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        standings_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Run temporal cross-validation."""
        df = self.prepare_data(df, standings_df)
        df = df[df["outcome"].notna()].reset_index(drop=True)
        
        X, feature_names = self.feature_aggregator.get_feature_matrix(df)
        y = df["outcome"].astype(int).values
        
        cv = TemporalCrossValidator(n_splits=n_folds)
        
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = EnsemblePredictor()
            model.fit(X_train, y_train, feature_names)
            
            probs = model.predict_proba(X_test)
            metrics = calculate_all_metrics(y_test, probs)
            metrics["fold"] = fold
            fold_metrics.append(metrics)
        
        # Aggregate
        avg_metrics = {
            "rps": np.mean([m["rps"] for m in fold_metrics]),
            "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
            "log_loss": np.mean([m["log_loss"] for m in fold_metrics]),
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
