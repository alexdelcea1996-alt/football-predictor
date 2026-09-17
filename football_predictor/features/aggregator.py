"""
Feature aggregator that combines all feature sources.

Orchestrates Elo, rolling stats, H2H, contextual, market-odds and experience
features into a unified feature vector with leakage prevention.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from football_predictor.features.contextual import ContextualFeatures
from football_predictor.features.elo import EloRatingSystem
from football_predictor.features.head_to_head import HeadToHeadCalculator
from football_predictor.features.odds_features import DevigMethod, OddsFeatures
from football_predictor.features.poisson_features import PoissonFeatures
from football_predictor.features.rolling_stats import RollingStatsCalculator

# Columns that describe the match or its result, never inputs to the model.
# Raw odds are excluded too: their de-vigged probabilities are the features.
NON_FEATURE_COLUMNS = {
    "date", "home_team", "away_team", "home_goals", "away_goals",
    "outcome", "league", "season", "status", "match_id", "round",
    "home_team_id", "away_team_id", "home_xg", "away_xg",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_possession", "away_possession", "home_corners", "away_corners",
    "home_fouls", "away_fouls", "home_yellows", "away_yellows",
    "home_reds", "away_reds",
    "odds_home", "odds_draw", "odds_away", "odds_source",
}

EXPERIENCE_FEATURES = ("home_matches_played", "away_matches_played", "min_matches_played")


class FeatureAggregator:
    """
    Aggregates all feature sources into match-level feature vectors.

    Ensures temporal consistency (no data leakage) by processing matches in
    chronological order: every feature for a match is computed from matches
    that finished before it.
    """

    # Core feature columns that will always be present
    CORE_FEATURES = [
        "home_elo", "away_elo", "elo_diff", "home_elo_momentum", "away_elo_momentum",
        "home_expected_score",
    ]

    def __init__(
        self,
        use_odds: bool = True,
        use_poisson: bool = True,
        devig_method: DevigMethod = "shin",
        drop_degenerate: bool = True,
        poisson_refit_every: int = 40,
    ) -> None:
        self.elo = EloRatingSystem()
        self.rolling = RollingStatsCalculator()
        self.h2h = HeadToHeadCalculator()
        self.context = ContextualFeatures()
        self.odds = OddsFeatures(method=devig_method)
        self.poisson = PoissonFeatures(refit_every=poisson_refit_every)
        self.use_odds = use_odds
        self.use_poisson = use_poisson
        self.drop_degenerate = drop_degenerate

        self._feature_names: list[str] = []
        self._dropped_features: list[str] = []
        self._matches_played: dict[str, int] = defaultdict(int)

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
        result = df.copy()
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date", kind="mergesort").reset_index(drop=True)

        result = self.elo.process_matches(result)
        result = self.rolling.process_matches(result)
        result = self.h2h.process_matches(result)
        result = self.context.process_matches(result, standings_df)
        result = self._add_experience_features(result)

        if self.use_odds:
            result = self.odds.process_matches(result)

        if self.use_poisson:
            result = self.poisson.process_matches(result)

        candidates = [c for c in result.columns if c not in NON_FEATURE_COLUMNS]
        self._dropped_features = (
            self._find_degenerate(result, candidates) if self.drop_degenerate else []
        )
        dropped = set(self._dropped_features)
        self._feature_names = [c for c in candidates if c not in dropped]

        return result

    def _add_experience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count each team's prior matches.

        Rolling and H2H features are empty at the start of a team's history
        and get filled with zeros, which is indistinguishable from "genuinely
        average" unless the model knows how much history stands behind them.
        """
        result = df.copy()
        home_played: list[int] = []
        away_played: list[int] = []

        for _, row in result.iterrows():
            home, away = row["home_team"], row["away_team"]
            home_played.append(self._matches_played[home])
            away_played.append(self._matches_played[away])

            if pd.notna(row.get("home_goals")) and pd.notna(row.get("away_goals")):
                self._matches_played[home] += 1
                self._matches_played[away] += 1

        result["home_matches_played"] = home_played
        result["away_matches_played"] = away_played
        result["min_matches_played"] = np.minimum(home_played, away_played)
        return result

    @staticmethod
    def _find_degenerate(df: pd.DataFrame, columns: list[str]) -> list[str]:
        """
        Identify columns carrying no information.

        Typical case: xG rolling features when the data source has no xG.
        They arrive all-NaN, get filled with zeros, and become pure noise
        for the tree models to split on.
        """
        degenerate = []
        for col in columns:
            series = df[col]
            if series.isna().all() or series.nunique(dropna=True) <= 1:
                degenerate.append(col)
        return degenerate

    def get_features_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: str | pd.Timestamp,
        match_week: int | None = None,
        odds: tuple[float, float, float] | None = None,
    ) -> dict[str, float | None]:
        """
        Get all features for an upcoming match.

        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Date of match
            match_week: Optional match week
            odds: Optional (home, draw, away) decimal odds for the fixture

        Returns:
            Dictionary of all features
        """
        features: dict[str, float | None] = {}

        features.update(self.elo.get_features_for_match(home_team, away_team))
        features.update(self.rolling.get_features_for_match(home_team, away_team))
        features.update(self.h2h.get_features_for_match(home_team, away_team))
        features.update(
            self.context.get_features_for_match(home_team, away_team, match_date, match_week)
        )

        home_played = self._matches_played.get(home_team, 0)
        away_played = self._matches_played.get(away_team, 0)
        features["home_matches_played"] = home_played
        features["away_matches_played"] = away_played
        features["min_matches_played"] = min(home_played, away_played)

        if self.use_odds:
            features.update(self.odds.get_features_for_match(*(odds or (None, None, None))))

        if self.use_poisson:
            features.update(self.poisson.get_features_for_match(home_team, away_team))

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

    def burn_in_mask(self, df: pd.DataFrame, min_matches: int = 5) -> np.ndarray:
        """
        Boolean mask of matches where both teams have enough history.

        Training on a team's first fixtures teaches the model to fit rows
        whose form features are placeholders.
        """
        if "min_matches_played" not in df.columns:
            return np.ones(len(df), dtype=bool)
        return (df["min_matches_played"] >= min_matches).to_numpy()

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self._feature_names.copy()

    @property
    def dropped_features(self) -> list[str]:
        """Features excluded for carrying no information (e.g. absent xG)."""
        return self._dropped_features.copy()

    def reset(self) -> None:
        """Reset all calculators."""
        self.elo = EloRatingSystem()
        self.rolling = RollingStatsCalculator()
        self.h2h = HeadToHeadCalculator()
        self.context = ContextualFeatures()
        self.odds = OddsFeatures(method=self.odds.method)
        self.poisson = PoissonFeatures(refit_every=self.poisson.refit_every)
        self._matches_played = defaultdict(int)

    def save_state(self) -> dict[str, Any]:
        """Save aggregator state."""
        return {
            "elo": self.elo.save_state(),
            "rolling": self.rolling.save_state(),
            "h2h": self.h2h.save_state(),
            "context": self.context.save_state(),
            "odds": self.odds.save_state(),
            "poisson": self.poisson.save_state(),
            "matches_played": dict(self._matches_played),
            "feature_names": self._feature_names,
            "dropped_features": self._dropped_features,
            "use_odds": self.use_odds,
            "use_poisson": self.use_poisson,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load aggregator state."""
        self.elo.load_state(state["elo"])
        self.rolling.load_state(state["rolling"])
        self.h2h.load_state(state["h2h"])
        self.context.load_state(state["context"])
        if "odds" in state:
            self.odds.load_state(state["odds"])
        if "poisson" in state:
            self.poisson.load_state(state["poisson"])
        self._matches_played = defaultdict(int, state.get("matches_played", {}))
        self._feature_names = state["feature_names"]
        self._dropped_features = state.get("dropped_features", [])
        self.use_odds = state.get("use_odds", self.use_odds)
        self.use_poisson = state.get("use_poisson", self.use_poisson)
