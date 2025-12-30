"""
Ensemble predictor combining multiple models.
"""

from pathlib import Path
from typing import Any
import json
import numpy as np

from football_predictor.config import get_settings


class EnsemblePredictor:
    """
    Ensemble of available models with soft voting.
    
    Dynamically uses whichever models are available.
    """
    
    def __init__(
        self,
        weights: list[float] | None = None,
    ) -> None:
        settings = get_settings()
        self.weights = weights or settings.model.ensemble_weights
        
        self._models: list[tuple[str, Any]] = []
        self._feature_names: list[str] = []
        self._trained = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "EnsemblePredictor":
        """Train all available models in the ensemble."""
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self._models = []
        
        # Try CatBoost
        try:
            from football_predictor.models.catboost_model import CatBoostModel
            cat = CatBoostModel()
            cat.fit(X, y, feature_names, eval_set)
            self._models.append(("catboost", cat))
        except ImportError:
            pass
        
        # Try XGBoost
        try:
            from football_predictor.models.xgboost_model import XGBoostModel
            xgb = XGBoostModel()
            xgb.fit(X, y, feature_names, eval_set)
            self._models.append(("xgboost", xgb))
        except ImportError:
            pass
        
        # Try Logistic Regression
        try:
            from football_predictor.models.logistic_model import LogisticModel
            log = LogisticModel()
            log.fit(X, y, feature_names)
            self._models.append(("logistic", log))
        except ImportError:
            pass
        
        if not self._models:
            raise RuntimeError("No models available. Install catboost, xgboost, or scikit-learn.")
        
        self._trained = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using weighted average.
        
        Returns:
            Array of shape (n_samples, 3) with [P(Home), P(Draw), P(Away)]
        """
        if not self._trained:
            raise RuntimeError("Ensemble not trained")
        
        # Collect predictions from all models
        all_probs = []
        for name, model in self._models:
            probs = model.predict_proba(X)
            all_probs.append(probs)
        
        # Equal weight averaging
        n_models = len(all_probs)
        ensemble_probs = sum(all_probs) / n_models
        
        # Normalize probabilities
        ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
        
        return ensemble_probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_with_confidence(
        self, X: np.ndarray
    ) -> list[dict[str, Any]]:
        """Predict with detailed probability breakdown."""
        probs = self.predict_proba(X)
        predictions = []
        
        for i, prob in enumerate(probs):
            pred_class = int(np.argmax(prob))
            outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            
            predictions.append({
                "predicted_outcome": outcome_map[pred_class],
                "predicted_class": pred_class,
                "home_win_prob": float(prob[0]),
                "draw_prob": float(prob[1]),
                "away_win_prob": float(prob[2]),
                "confidence": float(max(prob)),
            })
        
        return predictions
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get averaged feature importance from tree models."""
        importance: dict[str, float] = {}
        count = 0
        
        for name, model in self._models:
            if hasattr(model, "get_feature_importance"):
                imp = model.get_feature_importance()
                for f, v in imp.items():
                    importance[f] = importance.get(f, 0) + v
                count += 1
        
        if count > 0:
            for f in importance:
                importance[f] /= count
        
        return dict(sorted(importance.items(), key=lambda x: -x[1]))
    
    def get_models_info(self) -> list[str]:
        """Get list of trained models."""
        return [name for name, _ in self._models]
    
    def save(self, directory: str | Path) -> None:
        """Save all models to a directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self._models:
            if hasattr(model, "save"):
                ext = {"catboost": "cbm", "xgboost": "json", "logistic": "joblib"}
                model.save(path / f"{name}.{ext.get(name, 'pkl')}")
        
        # Save metadata
        meta = {
            "weights": self.weights,
            "feature_names": self._feature_names,
            "models": [name for name, _ in self._models],
        }
        with open(path / "ensemble_meta.json", "w") as f:
            json.dump(meta, f)
    
    def load(self, directory: str | Path) -> "EnsemblePredictor":
        """Load all models from a directory."""
        path = Path(directory)
        
        with open(path / "ensemble_meta.json") as f:
            meta = json.load(f)
        
        self.weights = meta["weights"]
        self._feature_names = meta["feature_names"]
        self._models = []
        
        for name in meta.get("models", []):
            try:
                if name == "catboost" and (path / "catboost.cbm").exists():
                    from football_predictor.models.catboost_model import CatBoostModel
                    model = CatBoostModel().load(path / "catboost.cbm")
                    self._models.append((name, model))
                elif name == "xgboost" and (path / "xgboost.json").exists():
                    from football_predictor.models.xgboost_model import XGBoostModel
                    model = XGBoostModel().load(path / "xgboost.json")
                    self._models.append((name, model))
                elif name == "logistic" and (path / "logistic.joblib").exists():
                    from football_predictor.models.logistic_model import LogisticModel
                    model = LogisticModel().load(path / "logistic.joblib")
                    self._models.append((name, model))
            except ImportError:
                pass
        
        self._trained = True
        return self
