"""
Head-to-head history features for football matches.

Computes statistics from previous encounters between two teams:
- Win/draw/loss rates in past meetings
- Average goals scored/conceded
- Recent H2H trend
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from football_predictor.config import get_settings


@dataclass
class H2HRecord:
    """Record of a head-to-head match."""
    
    date: pd.Timestamp
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    
    def get_result_for_team(self, team: str) -> str:
        """Get result (W/D/L) from perspective of given team."""
        if team == self.home_team:
            if self.home_goals > self.away_goals:
                return "W"
            elif self.home_goals < self.away_goals:
                return "L"
        elif team == self.away_team:
            if self.away_goals > self.home_goals:
                return "W"
            elif self.away_goals < self.home_goals:
                return "L"
        return "D"
    
    def get_goals_for_team(self, team: str) -> int:
        """Get goals scored by team."""
        if team == self.home_team:
            return self.home_goals
        return self.away_goals
    
    def get_goals_against_team(self, team: str) -> int:
        """Get goals conceded by team."""
        if team == self.home_team:
            return self.away_goals
        return self.home_goals


class HeadToHeadCalculator:
    """
    Calculate head-to-head statistics between teams.
    
    Features:
    - Win/draw/loss rates in past H2H meetings
    - Average goals scored/conceded in H2H
    - H2H goal difference
    - Recent H2H trend (last 3 matches)
    
    Usage:
        h2h = HeadToHeadCalculator()
        df_with_features = h2h.process_matches(matches_df)
    """
    
    def __init__(self, lookback_matches: int | None = None) -> None:
        settings = get_settings()
        self.lookback = lookback_matches or settings.features.h2h_lookback_matches
        
        # Key: frozenset({team1, team2}), Value: list of H2HRecord
        self._history: dict[frozenset[str], list[H2HRecord]] = defaultdict(list)
    
    def _get_matchup_key(self, team1: str, team2: str) -> frozenset[str]:
        """Get a consistent key for the matchup regardless of home/away."""
        return frozenset({team1, team2})
    
    def add_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        date: pd.Timestamp,
    ) -> None:
        """Add a match to H2H history."""
        key = self._get_matchup_key(home_team, away_team)
        record = H2HRecord(
            date=date,
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
        )
        self._history[key].append(record)
        # Keep sorted by date
        self._history[key].sort(key=lambda x: x.date)
    
    def get_h2h_stats(
        self,
        team1: str,
        team2: str,
        perspective_team: str | None = None,
    ) -> dict[str, float | None]:
        """
        Get H2H statistics for a matchup.
        
        Args:
            team1: First team
            team2: Second team
            perspective_team: Team to compute stats from perspective of
                             (defaults to team1)
        
        Returns:
            Dictionary with H2H features
        """
        key = self._get_matchup_key(team1, team2)
        matches = self._history[key][-self.lookback:] if self._history[key] else []
        
        if not matches:
            return {
                "h2h_matches": 0,
                "h2h_win_rate": None,
                "h2h_draw_rate": None,
                "h2h_loss_rate": None,
                "h2h_goals_for_avg": None,
                "h2h_goals_against_avg": None,
                "h2h_goal_diff": None,
                "h2h_last3_wins": None,
            }
        
        team = perspective_team or team1
        
        wins = 0
        draws = 0
        losses = 0
        goals_for = []
        goals_against = []
        
        for match in matches:
            result = match.get_result_for_team(team)
            if result == "W":
                wins += 1
            elif result == "D":
                draws += 1
            else:
                losses += 1
            
            goals_for.append(match.get_goals_for_team(team))
            goals_against.append(match.get_goals_against_team(team))
        
        n = len(matches)
        
        # Last 3 matches trend
        last3 = matches[-3:] if len(matches) >= 3 else matches
        last3_wins = sum(1 for m in last3 if m.get_result_for_team(team) == "W")
        
        return {
            "h2h_matches": n,
            "h2h_win_rate": wins / n,
            "h2h_draw_rate": draws / n,
            "h2h_loss_rate": losses / n,
            "h2h_goals_for_avg": np.mean(goals_for),
            "h2h_goals_against_avg": np.mean(goals_against),
            "h2h_goal_diff": (sum(goals_for) - sum(goals_against)) / n,
            "h2h_last3_wins": last3_wins / len(last3),
        }
    
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
        Process matches and compute H2H features.
        
        Features are computed BEFORE processing each match (no leakage).
        
        Args:
            df: DataFrame with match data (must be sorted by date)
            
        Returns:
            DataFrame with added H2H features
        """
        result = df.copy()
        result[date_col] = pd.to_datetime(result[date_col])
        result = result.sort_values(date_col, kind="mergesort").reset_index(drop=True)
        
        # Initialize feature columns
        h2h_features = [
            "h2h_matches",
            "h2h_home_win_rate", "h2h_home_draw_rate", "h2h_home_loss_rate",
            "h2h_home_goals_for_avg", "h2h_home_goals_against_avg",
            "h2h_home_goal_diff", "h2h_home_last3_wins",
        ]
        
        for col in h2h_features:
            result[col] = np.nan
        
        for idx, row in result.iterrows():
            home = row[home_col]
            away = row[away_col]
            
            # Get pre-match H2H stats from home team perspective
            h2h_stats = self.get_h2h_stats(home, away, perspective_team=home)
            
            result.at[idx, "h2h_matches"] = h2h_stats["h2h_matches"]
            result.at[idx, "h2h_home_win_rate"] = h2h_stats["h2h_win_rate"]
            result.at[idx, "h2h_home_draw_rate"] = h2h_stats["h2h_draw_rate"]
            result.at[idx, "h2h_home_loss_rate"] = h2h_stats["h2h_loss_rate"]
            result.at[idx, "h2h_home_goals_for_avg"] = h2h_stats["h2h_goals_for_avg"]
            result.at[idx, "h2h_home_goals_against_avg"] = h2h_stats["h2h_goals_against_avg"]
            result.at[idx, "h2h_home_goal_diff"] = h2h_stats["h2h_goal_diff"]
            result.at[idx, "h2h_home_last3_wins"] = h2h_stats["h2h_last3_wins"]
            
            # Add match to history (after computing features)
            home_goals = row.get(home_goals_col)
            away_goals = row.get(away_goals_col)
            
            if pd.notna(home_goals) and pd.notna(away_goals):
                self.add_match(
                    home_team=home,
                    away_team=away,
                    home_goals=int(home_goals),
                    away_goals=int(away_goals),
                    date=pd.Timestamp(row[date_col]),
                )
        
        return result
    
    def get_features_for_match(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, float | None]:
        """
        Get H2H features for an upcoming match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dictionary with H2H features
        """
        stats = self.get_h2h_stats(home_team, away_team, perspective_team=home_team)
        
        return {
            "h2h_matches": stats["h2h_matches"],
            "h2h_home_win_rate": stats["h2h_win_rate"],
            "h2h_home_draw_rate": stats["h2h_draw_rate"],
            "h2h_home_loss_rate": stats["h2h_loss_rate"],
            "h2h_home_goals_for_avg": stats["h2h_goals_for_avg"],
            "h2h_home_goals_against_avg": stats["h2h_goals_against_avg"],
            "h2h_home_goal_diff": stats["h2h_goal_diff"],
            "h2h_home_last3_wins": stats["h2h_last3_wins"],
        }
    
    def reset(self) -> None:
        """Reset all H2H history."""
        self._history.clear()
    
    def save_state(self) -> dict[str, Any]:
        """Save calculator state for persistence."""
        return {
            "lookback": self.lookback,
            "history": {
                f"{sorted(list(key))[0]}|{sorted(list(key))[1]}": [
                    {
                        "date": r.date.isoformat(),
                        "home_team": r.home_team,
                        "away_team": r.away_team,
                        "home_goals": r.home_goals,
                        "away_goals": r.away_goals,
                    }
                    for r in records
                ]
                for key, records in self._history.items()
            },
        }
    
    def load_state(self, state: dict[str, Any]) -> None:
        """Load calculator state from saved data."""
        self.lookback = state["lookback"]
        
        self._history.clear()
        for key_str, records in state["history"].items():
            teams = key_str.split("|")
            key = frozenset(teams)
            self._history[key] = [
                H2HRecord(
                    date=pd.Timestamp(r["date"]),
                    home_team=r["home_team"],
                    away_team=r["away_team"],
                    home_goals=r["home_goals"],
                    away_goals=r["away_goals"],
                )
                for r in records
            ]
