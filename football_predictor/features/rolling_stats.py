"""
Rolling statistics calculator for football teams.

Computes time-windowed aggregate statistics with:
- Multiple window sizes (5, 10, 20 games)
- Exponential decay weighting
- Separate home/away performance tracking
- Goals, points, win rates, and form calculations
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from football_predictor.config import get_settings


@dataclass
class TeamHistory:
    """Historical match data for a single team."""
    
    matches: list[dict[str, Any]] = field(default_factory=list)
    home_matches: list[dict[str, Any]] = field(default_factory=list)
    away_matches: list[dict[str, Any]] = field(default_factory=list)
    
    def add_match(
        self,
        date: pd.Timestamp,
        goals_for: int,
        goals_against: int,
        is_home: bool,
        xg_for: float | None = None,
        xg_against: float | None = None,
        shots: int | None = None,
        shots_on_target: int | None = None,
        possession: float | None = None,
    ) -> None:
        """Add a match to history."""
        match_data = {
            "date": date,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "is_home": is_home,
            "result": self._get_result(goals_for, goals_against),
            "points": self._get_points(goals_for, goals_against),
            "xg_for": xg_for,
            "xg_against": xg_against,
            "shots": shots,
            "shots_on_target": shots_on_target,
            "possession": possession,
        }
        
        self.matches.append(match_data)
        if is_home:
            self.home_matches.append(match_data)
        else:
            self.away_matches.append(match_data)
    
    @staticmethod
    def _get_result(goals_for: int, goals_against: int) -> str:
        if goals_for > goals_against:
            return "W"
        elif goals_for < goals_against:
            return "L"
        return "D"
    
    @staticmethod
    def _get_points(goals_for: int, goals_against: int) -> int:
        if goals_for > goals_against:
            return 3
        elif goals_for == goals_against:
            return 1
        return 0


class RollingStatsCalculator:
    """
    Calculate rolling statistics for football teams.
    
    Features computed (for each window size):
    - Goals scored/conceded (mean)
    - Goal difference
    - Win/draw/loss rates
    - Points per game
    - Form (points in last N games)
    - xG and xG against (where available)
    - Shots and shot accuracy
    - Possession
    
    All features are computed separately for home and away performance.
    
    Usage:
        calculator = RollingStatsCalculator()
        df_with_features = calculator.process_matches(matches_df)
    """
    
    def __init__(
        self,
        windows: list[int] | None = None,
        decay_factor: float | None = None,
    ) -> None:
        settings = get_settings()
        self.windows = windows or settings.features.rolling_windows
        self.decay_factor = decay_factor or settings.features.decay_factor
        
        self._team_history: dict[str, TeamHistory] = defaultdict(TeamHistory)
    
    def _get_weighted_mean(
        self,
        values: list[float | None],
        window: int,
    ) -> float | None:
        """Calculate weighted mean with exponential decay."""
        valid = [v for v in values[-window:] if v is not None]
        if not valid:
            return None
        
        n = len(valid)
        weights = np.array([self.decay_factor ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        
        return float(np.average(valid, weights=weights))
    
    def _compute_stats_for_window(
        self,
        matches: list[dict[str, Any]],
        window: int,
    ) -> dict[str, float | None]:
        """Compute statistics for a specific window size."""
        recent = matches[-window:] if len(matches) >= window else matches
        
        if not recent:
            return {
                f"goals_for_{window}": None,
                f"goals_against_{window}": None,
                f"goal_diff_{window}": None,
                f"win_rate_{window}": None,
                f"draw_rate_{window}": None,
                f"loss_rate_{window}": None,
                f"ppg_{window}": None,
                f"form_{window}": None,
                f"xg_for_{window}": None,
                f"xg_against_{window}": None,
            }
        
        n = len(recent)
        
        # Basic stats
        goals_for = [m["goals_for"] for m in recent]
        goals_against = [m["goals_against"] for m in recent]
        points = [m["points"] for m in recent]
        results = [m["result"] for m in recent]
        
        # Win/draw/loss rates
        wins = sum(1 for r in results if r == "W")
        draws = sum(1 for r in results if r == "D")
        losses = sum(1 for r in results if r == "L")
        
        # Form (sum of points, not rate)
        form = sum(points)
        
        stats = {
            f"goals_for_{window}": self._get_weighted_mean(goals_for, window),
            f"goals_against_{window}": self._get_weighted_mean(goals_against, window),
            f"goal_diff_{window}": (sum(goals_for) - sum(goals_against)) / n,
            f"win_rate_{window}": wins / n,
            f"draw_rate_{window}": draws / n,
            f"loss_rate_{window}": losses / n,
            f"ppg_{window}": sum(points) / n,
            f"form_{window}": form,
        }
        
        # Optional stats
        xg_for = [m["xg_for"] for m in recent if m["xg_for"] is not None]
        xg_against = [m["xg_against"] for m in recent if m["xg_against"] is not None]
        
        stats[f"xg_for_{window}"] = np.mean(xg_for) if xg_for else None
        stats[f"xg_against_{window}"] = np.mean(xg_against) if xg_against else None
        
        shots = [m["shots"] for m in recent if m["shots"] is not None]
        shots_on_target = [m["shots_on_target"] for m in recent if m["shots_on_target"] is not None]
        
        stats[f"shots_{window}"] = np.mean(shots) if shots else None
        stats[f"shots_on_target_{window}"] = np.mean(shots_on_target) if shots_on_target else None
        
        if shots_on_target and shots:
            stats[f"shot_accuracy_{window}"] = sum(shots_on_target) / sum(shots) if sum(shots) > 0 else None
        else:
            stats[f"shot_accuracy_{window}"] = None
        
        possession = [m["possession"] for m in recent if m["possession"] is not None]
        stats[f"possession_{window}"] = np.mean(possession) if possession else None
        
        return stats
    
    def get_team_stats(
        self,
        team: str,
        context: str = "all",  # "all", "home", "away"
    ) -> dict[str, float | None]:
        """
        Get current rolling statistics for a team.
        
        Args:
            team: Team name
            context: Which matches to consider ("all", "home", "away")
            
        Returns:
            Dictionary of rolling statistics
        """
        history = self._team_history[team]
        
        if context == "home":
            matches = history.home_matches
        elif context == "away":
            matches = history.away_matches
        else:
            matches = history.matches
        
        all_stats: dict[str, float | None] = {}
        
        for window in self.windows:
            stats = self._compute_stats_for_window(matches, window)
            all_stats.update(stats)
        
        return all_stats
    
    def process_matches(
        self,
        df: pd.DataFrame,
        home_col: str = "home_team",
        away_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Process matches and compute rolling features.
        
        IMPORTANT: Features are computed BEFORE processing each match
        to prevent data leakage.
        
        Args:
            df: DataFrame with match data (must be sorted by date)
            
        Returns:
            DataFrame with added rolling statistics features
        """
        result = df.copy()
        result[date_col] = pd.to_datetime(result[date_col])
        result = result.sort_values(date_col).reset_index(drop=True)
        
        # Initialize all feature columns
        feature_cols: list[str] = []
        for window in self.windows:
            for prefix in ["home_", "away_"]:
                for stat in [
                    "goals_for", "goals_against", "goal_diff",
                    "win_rate", "draw_rate", "loss_rate",
                    "ppg", "form", "xg_for", "xg_against",
                    "shots", "shots_on_target", "shot_accuracy", "possession"
                ]:
                    feature_cols.append(f"{prefix}{stat}_{window}")
        
        for col in feature_cols:
            result[col] = np.nan
        
        # Process each match
        for idx, row in result.iterrows():
            home = row[home_col]
            away = row[away_col]
            
            # Get pre-match stats (before this match)
            home_stats = self.get_team_stats(home, "home")
            away_stats = self.get_team_stats(away, "away")
            
            # Add home team stats
            for key, value in home_stats.items():
                col_name = f"home_{key}"
                if col_name in result.columns:
                    result.at[idx, col_name] = value
            
            # Add away team stats
            for key, value in away_stats.items():
                col_name = f"away_{key}"
                if col_name in result.columns:
                    result.at[idx, col_name] = value
            
            # Update history (after computing features)
            home_goals = row.get(home_goals_col)
            away_goals = row.get(away_goals_col)
            
            if pd.notna(home_goals) and pd.notna(away_goals):
                match_date = pd.Timestamp(row[date_col])
                
                # Get optional stats
                home_xg = row.get("home_xg")
                away_xg = row.get("away_xg")
                home_shots = row.get("home_shots")
                away_shots = row.get("away_shots")
                home_sot = row.get("home_shots_on_target")
                away_sot = row.get("away_shots_on_target")
                home_poss = row.get("home_possession")
                away_poss = row.get("away_possession")
                
                self._team_history[home].add_match(
                    date=match_date,
                    goals_for=int(home_goals),
                    goals_against=int(away_goals),
                    is_home=True,
                    xg_for=home_xg if pd.notna(home_xg) else None,
                    xg_against=away_xg if pd.notna(away_xg) else None,
                    shots=int(home_shots) if pd.notna(home_shots) else None,
                    shots_on_target=int(home_sot) if pd.notna(home_sot) else None,
                    possession=float(home_poss) if pd.notna(home_poss) else None,
                )
                
                self._team_history[away].add_match(
                    date=match_date,
                    goals_for=int(away_goals),
                    goals_against=int(home_goals),
                    is_home=False,
                    xg_for=away_xg if pd.notna(away_xg) else None,
                    xg_against=home_xg if pd.notna(home_xg) else None,
                    shots=int(away_shots) if pd.notna(away_shots) else None,
                    shots_on_target=int(away_sot) if pd.notna(away_sot) else None,
                    possession=float(away_poss) if pd.notna(away_poss) else None,
                )
        
        return result
    
    def get_features_for_match(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, float | None]:
        """
        Get rolling stats features for an upcoming match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dictionary with rolling statistics features
        """
        features: dict[str, float | None] = {}
        
        home_stats = self.get_team_stats(home_team, "home")
        away_stats = self.get_team_stats(away_team, "away")
        
        for key, value in home_stats.items():
            features[f"home_{key}"] = value
        
        for key, value in away_stats.items():
            features[f"away_{key}"] = value
        
        return features
    
    def reset(self) -> None:
        """Reset all team history."""
        self._team_history.clear()
    
    def save_state(self) -> dict[str, Any]:
        """Save calculator state for persistence."""
        return {
            "windows": self.windows,
            "decay_factor": self.decay_factor,
            "team_history": {
                team: {
                    "matches": history.matches,
                    "home_matches": history.home_matches,
                    "away_matches": history.away_matches,
                }
                for team, history in self._team_history.items()
            },
        }
    
    def load_state(self, state: dict[str, Any]) -> None:
        """Load calculator state from saved data."""
        self.windows = state["windows"]
        self.decay_factor = state["decay_factor"]
        
        self._team_history.clear()
        for team, data in state["team_history"].items():
            history = TeamHistory()
            history.matches = data["matches"]
            history.home_matches = data["home_matches"]
            history.away_matches = data["away_matches"]
            self._team_history[team] = history
