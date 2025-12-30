"""
SHAP-based model explainability.
"""

from typing import Any
import numpy as np


class SHAPExplainer:
    """
    SHAP-based explainability for ensemble predictions.
    """
    
    def __init__(self, model: Any = None) -> None:
        self.model = model
        self.explainer = None
        self._shap_values = None
    
    def fit(self, X_background: np.ndarray, model: Any = None) -> "SHAPExplainer":
        """
        Initialize SHAP explainer with background data.
        
        Args:
            X_background: Background dataset for SHAP
            model: Model to explain (uses self.model if None)
        """
        import shap
        
        if model is not None:
            self.model = model
        
        if self.model is None:
            raise ValueError("Model required")
        
        # Use TreeExplainer for tree models or KernelExplainer for ensemble
        if hasattr(self.model, "model") and hasattr(self.model.model, "predict_proba"):
            # For base models
            self.explainer = shap.Explainer(
                self.model.predict_proba,
                X_background[:100],  # Use subset for efficiency
            )
        else:
            # For ensemble
            self.explainer = shap.Explainer(
                self.model.predict_proba,
                X_background[:100],
            )
        
        return self
    
    def explain(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate SHAP explanations.
        
        Args:
            X: Features to explain
            feature_names: Names of features
        
        Returns:
            Dict with SHAP values and importance
        """
        import shap
        
        if self.explainer is None:
            raise RuntimeError("Call fit() first")
        
        shap_values = self.explainer(X)
        self._shap_values = shap_values
        
        # Global importance (mean absolute SHAP)
        if hasattr(shap_values, "values"):
            vals = shap_values.values
            if vals.ndim == 3:
                # Multi-class: average across classes
                importance = np.abs(vals).mean(axis=(0, 2))
            else:
                importance = np.abs(vals).mean(axis=0)
        else:
            importance = np.zeros(X.shape[1])
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        feature_importance = dict(sorted(
            zip(feature_names, importance),
            key=lambda x: -x[1]
        ))
        
        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
        }
    
    def explain_prediction(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Explain individual predictions.
        
        Args:
            X: Single sample or batch
            feature_names: Names of features
            top_k: Number of top features to show
        
        Returns:
            List of explanation dicts per sample
        """
        result = self.explain(X, feature_names)
        shap_values = result["shap_values"]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        explanations = []
        n_samples = X.shape[0]
        
        for i in range(n_samples):
            if hasattr(shap_values, "values"):
                vals = shap_values.values[i]
                if vals.ndim == 2:
                    # Multi-class: take highest class contribution
                    vals = vals[:, np.argmax(np.abs(vals).sum(axis=0))]
            else:
                vals = np.zeros(len(feature_names))
            
            # Top contributing features
            sorted_idx = np.argsort(np.abs(vals))[::-1][:top_k]
            top_features = [
                {
                    "feature": feature_names[idx],
                    "value": float(X[i, idx]),
                    "shap_value": float(vals[idx]),
                    "contribution": "positive" if vals[idx] > 0 else "negative",
                }
                for idx in sorted_idx
            ]
            
            explanations.append({
                "sample_idx": i,
                "top_features": top_features,
            })
        
        return explanations
