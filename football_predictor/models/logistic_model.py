"""
Logistic Regression model for football prediction.
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from football_predictor.config import get_settings


class LogisticModel:
    """
    Logistic Regression baseline with built-in normalization.
    
    Provides interpretable predictions and well-calibrated probabilities.
    """
    
    def __init__(
        self,
        C: float | None = None,
        random_state: int = 42,
    ) -> None:
        settings = get_settings()
        
        self.C = C or settings.model.logreg_c
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.model: LogisticRegression | None = None
        self._feature_names: list[str] = []
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "LogisticModel":
        """Train the logistic regression model."""
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LogisticRegression(
            C=self.C,
            random_state=self.random_state,
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
        )
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def get_coefficients(self) -> dict[str, np.ndarray]:
        """Get model coefficients for interpretability."""
        if self.model is None:
            return {}
        return {
            "classes": self.model.classes_.tolist(),
            "coefficients": dict(zip(
                self._feature_names,
                self.model.coef_.T.tolist()
            )),
            "intercepts": self.model.intercept_.tolist(),
        }
    
    def save(self, path: str | Path) -> None:
        """Save model and scaler."""
        joblib.dump({"model": self.model, "scaler": self.scaler}, str(path))
    
    def load(self, path: str | Path) -> "LogisticModel":
        """Load model and scaler."""
        data = joblib.load(str(path))
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self
