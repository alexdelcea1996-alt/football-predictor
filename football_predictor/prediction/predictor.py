"""
Prediction interface for upcoming fixtures.
"""

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from football_predictor.features.aggregator import FeatureAggregator
from football_predictor.models.ensemble import EnsemblePredictor
from football_predictor.data.preprocessor import DataPreprocessor


class MatchPredictor:
    """
    Prediction interface for upcoming football matches.
    
    Usage:
        predictor = MatchPredictor()
        predictor.load("models/ensemble")
        
        predictions = predictor.predict_fixtures([
            {"date": "2024-01-15", "home_team": "Arsenal", "away_team": "Liverpool"},
        ])
    """
    
    def __init__(self) -> None:
        self.ensemble: EnsemblePredictor | None = None
        self.feature_aggregator = FeatureAggregator()
        self._loaded = False
    
    def load(self, model_dir: str | Path) -> "MatchPredictor":
        """Load trained model and feature state."""
        path = Path(model_dir)
        
        self.ensemble = EnsemblePredictor()
        self.ensemble.load(path / "ensemble" if (path / "ensemble").exists() else path)
        
        state_path = path / "feature_state.json"
        if not state_path.exists():
            state_path = path.parent / "feature_state.json"
        
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.feature_aggregator.load_state(state)
        
        self._loaded = True
        return self
    
    def predict_fixtures(
        self,
        fixtures: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Predict outcomes for a list of fixtures.
        
        Args:
            fixtures: List of dicts with 'date', 'home_team', 'away_team'
        
        Returns:
            List of prediction dicts
        """
        if not self._loaded or self.ensemble is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        predictions = []
        
        for fixture in fixtures:
            # Get features, including market odds when the fixture carries them
            features = self.feature_aggregator.get_features_for_match(
                home_team=fixture["home_team"],
                away_team=fixture["away_team"],
                match_date=fixture["date"],
                match_week=fixture.get("match_week"),
                odds=fixture_odds(fixture),
            )

            # Convert to array. Missing features are NaN, not None: `nan or 0`
            # evaluates to nan (NaN is truthy), which reaches the models as a
            # missing value and breaks the linear ones.
            feature_names = self.feature_aggregator.feature_names
            X = np.array(
                [[as_feature_value(features.get(name)) for name in feature_names]],
                dtype=float,
            )
            
            # Predict
            probs = self.ensemble.predict_proba(X)[0]
            pred_class = int(probs.argmax())
            outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            
            predictions.append({
                "date": fixture["date"],
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "predicted_outcome": outcome_map[pred_class],
                "home_win_prob": round(float(probs[0]), 4),
                "draw_prob": round(float(probs[1]), 4),
                "away_win_prob": round(float(probs[2]), 4),
                "confidence": round(float(max(probs)), 4),
            })
        
        return predictions
    
    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict from DataFrame."""
        fixtures = df.to_dict("records")
        preds = self.predict_fixtures(fixtures)
        return pd.DataFrame(preds)
    
    def to_csv(
        self,
        fixtures: list[dict[str, Any]],
        output_path: str | Path,
    ) -> None:
        """Predict and save to CSV."""
        preds = self.predict_fixtures(fixtures)
        df = pd.DataFrame(preds)
        df.to_csv(output_path, index=False)


def as_feature_value(value: Any) -> float:
    """Feature value as a finite float; anything missing becomes 0.0."""
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def fixture_odds(fixture: dict[str, Any]) -> tuple[float, float, float] | None:
    """Pull decimal odds out of a fixture record when present."""
    values = [fixture.get(key) for key in ("odds_home", "odds_draw", "odds_away")]
    if any(value is None for value in values):
        return None
    try:
        odds = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return odds if all(np.isfinite(o) and o > 1.0 for o in odds) else None
