"""
Feature aggregator that combines all feature sources.

Orchestrates Elo, rolling stats, H2H, and contextual features
into a unified feature vector with leakage prevention.
"""

from typing import Any

import numpy as np
import pandas as pd

from football_predictor.features.elo import EloRatingSystem
from football_predictor.features.rolling_stats import RollingStatsCalculator
from football_predictor.features.head_to_head import HeadToHeadCalculator
from football_predictor.features.contextual import ContextualFeatures


class FeatureAggregator:
    """
    Aggregates all feature sources into match-level feature vectors.
    
    Ensures temporal consistency (no data leakage) by processing
    matches in chronological order.
    """
    
    # Core feature columns that will always be present
    CORE_FEATURES = [
        "home_elo", "away_elo", "elo_diff", "home_elo_momentum", "away_elo_momentum",
        "home_expected_score",
    ]
    
    def __init__(self) -> None:
        self.elo = EloRatingSystem()
        self.rolling = RollingStatsCalculator()
        self.h2h = HeadToHeadCalculator()
        self.context = ContextualFeatures()
        self._feature_names: list[str] = []
    
    def process_matches(
        self,
        df: pd.DataFrame,
        standings_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Process all matches and compute all features.
        
        Args:
            df: Match data with required columns
            standings_df: Optional standings for contextual features
        
        Returns:
            DataFrame with all computed features
        """
        # Ensure sorted by date
        result = df.copy()
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date").reset_index(drop=True)
        
        # Apply each feature calculator
        result = self.elo.process_matches(result)
        result = self.rolling.process_matches(result)
        result = self.h2h.process_matches(result)
        result = self.context.process_matches(result, standings_df)
        
        # Store feature names (exclude non-feature columns)
        non_features = {
            "date", "home_team", "away_team", "home_goals", "away_goals",
            "outcome", "league", "season", "status", "match_id", "round",
            "home_team_id", "away_team_id", "home_xg", "away_xg",
            "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
            "home_possession", "away_possession", "home_corners", "away_corners",
        }
        self._feature_names = [c for c in result.columns if c not in non_features]
        
        return result
    
    def get_features_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: str | pd.Timestamp,
        match_week: int | None = None,
    ) -> dict[str, float | None]:
        """
        Get all features for an upcoming match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Date of match
            match_week: Optional match week
        
        Returns:
            Dictionary of all features
        """
        features: dict[str, float | None] = {}
        
        features.update(self.elo.get_features_for_match(home_team, away_team))
        features.update(self.rolling.get_features_for_match(home_team, away_team))
        features.update(self.h2h.get_features_for_match(home_team, away_team))
        features.update(self.context.get_features_for_match(
            home_team, away_team, match_date, match_week
        ))
        
        return features
    
    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        fill_na: float = 0.0,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract feature matrix from processed DataFrame.
        
        Args:
            df: Processed DataFrame with features
            fill_na: Value to fill missing features
        
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        feature_cols = [c for c in self._feature_names if c in df.columns]
        X = df[feature_cols].fillna(fill_na).values
        return X, feature_cols
    
    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self._feature_names.copy()
    
    def reset(self) -> None:
        """Reset all calculators."""
        self.elo = EloRatingSystem()
        self.rolling = RollingStatsCalculator()
        self.h2h = HeadToHeadCalculator()
        self.context = ContextualFeatures()
    
    def save_state(self) -> dict[str, Any]:
        """Save aggregator state."""
        return {
            "elo": self.elo.save_state(),
            "rolling": self.rolling.save_state(),
            "h2h": self.h2h.save_state(),
            "context": self.context.save_state(),
            "feature_names": self._feature_names,
        }
    
    def load_state(self, state: dict[str, Any]) -> None:
        """Load aggregator state."""
        self.elo.load_state(state["elo"])
        self.rolling.load_state(state["rolling"])
        self.h2h.load_state(state["h2h"])
        self.context.load_state(state["context"])
        self._feature_names = state["feature_names"]
