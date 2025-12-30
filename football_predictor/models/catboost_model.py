"""
CatBoost model wrapper for football prediction.
"""

from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV

from football_predictor.config import get_settings


class CatBoostModel:
    """
    CatBoost classifier wrapper with probability calibration.
    
    Primary model in the ensemble - handles categorical features well
    and provides automatic feature importance.
    """
    
    def __init__(
        self,
        iterations: int | None = None,
        learning_rate: float | None = None,
        depth: int | None = None,
        l2_leaf_reg: float | None = None,
        random_state: int = 42,
    ) -> None:
        settings = get_settings()
        
        self.iterations = iterations or settings.model.catboost_iterations
        self.learning_rate = learning_rate or settings.model.catboost_learning_rate
        self.depth = depth or settings.model.catboost_depth
        self.l2_leaf_reg = l2_leaf_reg or settings.model.catboost_l2_leaf_reg
        self.random_state = random_state
        
        self.model: CatBoostClassifier | None = None
        self.calibrator: CalibratedClassifierCV | None = None
        self._feature_names: list[str] = []
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        early_stopping_rounds: int | None = None,
        calibrate: bool = True,
    ) -> "CatBoostModel":
        """
        Train the CatBoost model.
        
        Args:
            X: Feature matrix
            y: Target labels (0=Home, 1=Draw, 2=Away)
            feature_names: Names of features
            eval_set: Optional (X_val, y_val) for early stopping
            early_stopping_rounds: Stop if no improvement
            calibrate: Whether to apply probability calibration
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
        )
        
        if eval_set:
            self.model.fit(X, y, eval_set=[eval_set])
        else:
            self.model.fit(X, y)
        
        if calibrate and len(np.unique(y)) >= 3:
            self.calibrator = CalibratedClassifierCV(
                self.model, method="isotonic", cv=3
            )
            self.calibrator.fit(X, y)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Returns:
            Array of shape (n_samples, 3) with [P(Home), P(Draw), P(Away)]
        """
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)
        if self.model is not None:
            return self.model.predict_proba(X)
        raise RuntimeError("Model not trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}
        importance = self.model.get_feature_importance()
        return dict(zip(self._feature_names, importance))
    
    def save(self, path: str | Path) -> None:
        """Save model to file."""
        if self.model:
            self.model.save_model(str(path))
    
    def load(self, path: str | Path) -> "CatBoostModel":
        """Load model from file."""
        self.model = CatBoostClassifier()
        self.model.load_model(str(path))
        return self
