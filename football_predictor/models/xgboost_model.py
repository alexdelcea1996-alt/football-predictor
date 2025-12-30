"""
XGBoost model wrapper for football prediction.
"""

from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from football_predictor.config import get_settings


class XGBoostModel:
    """
    XGBoost classifier wrapper with probability calibration.
    
    Industry-standard gradient boosting with strong regularization.
    """
    
    def __init__(
        self,
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        max_depth: int | None = None,
        reg_lambda: float | None = None,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        settings = get_settings()
        
        self.n_estimators = n_estimators or settings.model.xgboost_n_estimators
        self.learning_rate = learning_rate or settings.model.xgboost_learning_rate
        self.max_depth = max_depth or settings.model.xgboost_max_depth
        self.reg_lambda = reg_lambda or settings.model.xgboost_reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model: xgb.XGBClassifier | None = None
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
    ) -> "XGBoostModel":
        """Train the XGBoost model."""
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
        )
        
        if eval_set:
            self.model.fit(X, y, eval_set=[eval_set], verbose=False)
        else:
            self.model.fit(X, y)
        
        if calibrate and len(np.unique(y)) >= 3:
            self.calibrator = CalibratedClassifierCV(
                self.model, method="isotonic", cv=3
            )
            self.calibrator.fit(X, y)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)
        if self.model is not None:
            return self.model.predict_proba(X)
        raise RuntimeError("Model not trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self._feature_names, importance))
    
    def save(self, path: str | Path) -> None:
        """Save model."""
        if self.model:
            self.model.save_model(str(path))
    
    def load(self, path: str | Path) -> "XGBoostModel":
        """Load model."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        return self
