"""
Contextual features for football matches.

Computes context-aware features like rest days, league position, and season phase.
"""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


class ContextualFeatures:
    """Calculate contextual features for football matches."""
    
    def __init__(self) -> None:
        self._last_match: dict[str, pd.Timestamp] = {}
        self._standings: dict[str, dict[str, int]] = {}
    
    def update_standings(self, standings_df: pd.DataFrame) -> None:
        """Update standings from DataFrame with [team, position, points]."""
        self._standings.clear()
        for _, row in standings_df.iterrows():
            team = row.get("team")
            if team:
                self._standings[team] = {
                    "position": int(row.get("position", 0)),
                    "points": int(row.get("points", 0)),
                }
    
    def compute_features(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
        match_week: int | None = None,
    ) -> dict[str, float | None]:
        """Compute contextual features for a match."""
        # Rest days
        home_rest = away_rest = None
        if home_team in self._last_match:
            home_rest = (match_date - self._last_match[home_team]).days
        if away_team in self._last_match:
            away_rest = (match_date - self._last_match[away_team]).days
        
        rest_diff = None
        if home_rest is not None and away_rest is not None:
            rest_diff = home_rest - away_rest
        
        # Position & points
        home_pos = self._standings.get(home_team, {}).get("position")
        away_pos = self._standings.get(away_team, {}).get("position")
        home_pts = self._standings.get(home_team, {}).get("points")
        away_pts = self._standings.get(away_team, {}).get("points")
        
        pos_diff = home_pos - away_pos if home_pos and away_pos else None
        pts_diff = home_pts - away_pts if home_pts and away_pts else None
        
        # Season phase
        phase = None
        if match_week:
            if match_week < 13:
                phase = 0  # early
            elif match_week < 27:
                phase = 1  # mid
            else:
                phase = 2  # late
        
        return {
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_diff": rest_diff,
            "home_position": home_pos,
            "away_position": away_pos,
            "position_diff": pos_diff,
            "points_diff": pts_diff,
            "match_week": match_week,
            "season_phase": phase,
            "day_of_week": match_date.dayofweek,
            "is_weekend": 1 if match_date.dayofweek >= 5 else 0,
            "month": match_date.month,
        }
    
    def process_matches(
        self,
        df: pd.DataFrame,
        standings_df: pd.DataFrame | None = None,
        home_col: str = "home_team",
        away_col: str = "away_team",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Process matches and compute contextual features."""
        result = df.copy()
        result[date_col] = pd.to_datetime(result[date_col])
        result = result.sort_values(date_col, kind="mergesort").reset_index(drop=True)
        
        if standings_df is not None:
            self.update_standings(standings_df)
        
        feature_cols = [
            "home_rest_days", "away_rest_days", "rest_diff",
            "home_position", "away_position", "position_diff",
            "points_diff", "match_week", "season_phase",
            "day_of_week", "is_weekend", "month",
        ]
        for col in feature_cols:
            result[col] = np.nan
        
        team_matches: dict[str, int] = defaultdict(int)
        
        for idx, row in result.iterrows():
            home = row[home_col]
            away = row[away_col]
            date = pd.Timestamp(row[date_col])
            week = int((team_matches[home] + team_matches[away]) / 2 + 1)
            
            features = self.compute_features(home, away, date, week)
            for col, val in features.items():
                result.at[idx, col] = val
            
            self._last_match[home] = date
            self._last_match[away] = date
            team_matches[home] += 1
            team_matches[away] += 1
        
        return result
    
    def get_features_for_match(
        self, home_team: str, away_team: str, match_date: str | pd.Timestamp,
        match_week: int | None = None
    ) -> dict[str, float | None]:
        """Get features for an upcoming match."""
        if isinstance(match_date, str):
            match_date = pd.Timestamp(match_date)
        return self.compute_features(home_team, away_team, match_date, match_week)
    
    def reset(self) -> None:
        self._last_match.clear()
        self._standings.clear()
    
    def save_state(self) -> dict[str, Any]:
        return {
            "last_match": {t: d.isoformat() for t, d in self._last_match.items()},
            "standings": self._standings,
        }
    
    def load_state(self, state: dict[str, Any]) -> None:
        self._last_match = {t: pd.Timestamp(d) for t, d in state["last_match"].items()}
        self._standings = state["standings"]
