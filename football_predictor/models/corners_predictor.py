"""
Corners Over/Under prediction model.
"""

from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from football_predictor.features.corners_features import CornersFeatureCalculator


class CornersPredictor:
    """
    Predicts corners Over/Under for football matches.
    
    Primary target: Over/Under 9.5 corners
    """
    
    FEATURE_COLUMNS = [
        "home_corners_for_5", "home_corners_against_5",
        "home_corners_for_10", "home_corners_against_10",
        "away_corners_for_5", "away_corners_against_5",
        "away_corners_for_10", "away_corners_against_10",
        "exp_home_corners_5", "exp_away_corners_5",
        "exp_total_corners_5", "exp_total_corners_10",
        "home_corners_total_std", "away_corners_total_std",
        "league_avg_corners",
    ]
    
    def __init__(self, threshold: float = 9.5) -> None:
        self.threshold = threshold
        self.feature_calculator = CornersFeatureCalculator()
        self.model = None
        self.scaler = StandardScaler()
        self._feature_names: list[str] = []
        self._trained = False
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with corner features."""
        df = self.feature_calculator.process_matches(df)
        
        # Create target variable
        if "total_corners" in df.columns:
            df["over_target"] = (df["total_corners"] > self.threshold).astype(int)
        
        return df
    
    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        fill_na: float = 0.0,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract feature matrix."""
        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        self._feature_names = available
        X = df[available].fillna(fill_na).values
        return X, available
    
    def fit(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """Train the corners predictor."""
        import xgboost as xgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        # Prepare data
        df = self.prepare_data(df)
        
        # Filter rows with valid targets
        df = df[df["over_target"].notna()].reset_index(drop=True)
        
        # Need at least some history
        df = df.dropna(subset=["exp_total_corners_5"]).reset_index(drop=True)
        
        # Temporal split
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train, feature_names = self.get_feature_matrix(train_df)
        y_train = train_df["over_target"].values
        
        X_test, _ = self.get_feature_matrix(test_df)
        y_test = test_df["over_target"].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbosity=0,
        )
        xgb_model.fit(X_train_scaled, y_train)
        
        # Train LogReg
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logreg_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            logreg_model.fit(X_train_scaled, y_train)
        
        # Ensemble
        xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
        logreg_probs = logreg_model.predict_proba(X_test_scaled)[:, 1]
        ensemble_probs = (xgb_probs + logreg_probs) / 2
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, ensemble_preds)
        try:
            auc = roc_auc_score(y_test, ensemble_probs)
        except:
            auc = 0.5
        
        # Store model
        self.model = (xgb_model, logreg_model)
        self._trained = True
        
        # Baseline: always predict majority class
        baseline_acc = max((y_test == 0).mean(), (y_test == 1).mean())
        
        return {
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "accuracy": accuracy,
            "auc": auc,
            "baseline_accuracy": baseline_acc,
            "improvement": accuracy - baseline_acc,
            "over_rate": y_test.mean(),
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for new matches."""
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained")
        
        xgb_model, logreg_model = self.model
        
        # Get features
        X, _ = self.get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        xgb_probs = xgb_model.predict_proba(X_scaled)[:, 1]
        logreg_probs = logreg_model.predict_proba(X_scaled)[:, 1]
        ensemble_probs = (xgb_probs + logreg_probs) / 2
        
        result = df.copy()
        result["over_prob"] = ensemble_probs
        result["under_prob"] = 1 - ensemble_probs
        result["prediction"] = np.where(ensemble_probs > 0.5, "OVER", "UNDER")
        result["confidence"] = np.maximum(ensemble_probs, 1 - ensemble_probs)
        
        return result
    
    def predict_match(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, Any]:
        """Predict corners for a single match."""
        features = self.feature_calculator.get_match_features(home_team, away_team)
        
        # Create single-row DataFrame
        X = np.array([[features.get(f, 0) or 0 for f in self._feature_names]])
        X_scaled = self.scaler.transform(X)
        
        xgb_model, logreg_model = self.model
        xgb_prob = xgb_model.predict_proba(X_scaled)[0, 1]
        logreg_prob = logreg_model.predict_proba(X_scaled)[0, 1]
        over_prob = (xgb_prob + logreg_prob) / 2
        
        exp_total = features.get("exp_total_corners_5") or features.get("exp_total_corners_10")
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "threshold": self.threshold,
            "prediction": "OVER" if over_prob > 0.5 else "UNDER",
            "over_prob": float(over_prob),
            "under_prob": float(1 - over_prob),
            "confidence": float(max(over_prob, 1 - over_prob)),
            "expected_total": float(exp_total) if exp_total else None,
        }
    
    def save(self, path: str | Path) -> None:
        """Save model."""
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self._feature_names,
            "threshold": self.threshold,
            "feature_state": self.feature_calculator.save_state(),
        }, path / "corners_model.joblib")
    
    def load(self, path: str | Path) -> "CornersPredictor":
        """Load model."""
        import joblib
        path = Path(path)
        
        data = joblib.load(path / "corners_model.joblib")
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._feature_names = data["feature_names"]
        self.threshold = data["threshold"]
        self.feature_calculator.load_state(data["feature_state"])
        self._trained = True
        
        return self
