"""
Elo Rating System for football teams.

Implements a custom Elo rating system with:
- Adjustable K-factor for different match importance
- Home advantage adjustment
- Momentum tracking (recent rating changes)
- Pre-match expected score calculation
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from football_predictor.config import get_settings


@dataclass
class EloRating:
    """Elo rating for a team."""
    
    rating: float
    matches_played: int = 0
    last_updated: pd.Timestamp | None = None
    history: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    
    def get_momentum(self, lookback: int = 5) -> float:
        """Calculate rating momentum (change over last N matches)."""
        if len(self.history) < 2:
            return 0.0
        
        recent = self.history[-lookback:]
        if len(recent) < 2:
            return 0.0
        
        return recent[-1][1] - recent[0][1]


class EloRatingSystem:
    """
    Elo Rating System for football matches.
    
    Features:
    - K-factor adjustment for match importance
    - Home advantage bonus
    - Momentum calculation
    - Expected score (probability) derivation
    
    Usage:
        elo = EloRatingSystem()
        elo.process_matches(matches_df)
        
        # Get rating for a team
        rating = elo.get_rating("Manchester United")
        
        # Get expected outcome probabilities
        home_prob, draw_prob, away_prob = elo.predict_match("Team A", "Team B")
    """
    
    def __init__(
        self,
        k_factor: float | None = None,
        k_factor_important: float | None = None,
        home_advantage: float | None = None,
        initial_rating: float | None = None,
    ) -> None:
        settings = get_settings()
        
        self.k_factor = k_factor or settings.features.elo_k_factor
        self.k_factor_important = k_factor_important or settings.features.elo_k_factor_important
        self.home_advantage = home_advantage or settings.features.elo_home_advantage
        self.initial_rating = initial_rating or settings.features.elo_initial_rating
        
        self._ratings: dict[str, EloRating] = defaultdict(
            lambda: EloRating(rating=self.initial_rating)
        )
    
    def get_rating(self, team: str) -> float:
        """Get current Elo rating for a team."""
        return self._ratings[team].rating
    
    def get_rating_info(self, team: str) -> EloRating:
        """Get full rating information for a team."""
        return self._ratings[team]
    
    def get_all_ratings(self) -> dict[str, float]:
        """Get all team ratings as a dictionary."""
        return {team: info.rating for team, info in self._ratings.items()}
    
    def get_expected_score(
        self,
        home_team: str,
        away_team: str,
        apply_home_advantage: bool = True,
    ) -> float:
        """
        Calculate expected score for home team (0-1 scale).
        
        Based on the Elo expected score formula:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        
        Args:
            home_team: Home team name
            away_team: Away team name
            apply_home_advantage: Whether to add home advantage bonus
            
        Returns:
            Expected score (probability) for home team
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        if apply_home_advantage:
            home_rating += self.home_advantage
        
        exponent = (away_rating - home_rating) / 400
        expected = 1 / (1 + 10 ** exponent)
        
        return expected
    
    def predict_match(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float, float]:
        """
        Predict match outcome probabilities.
        
        Converts Elo expected score to three-way outcome probabilities.
        Uses a draw adjustment factor based on historical draw rates (~26%).
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Tuple of (home_win_prob, draw_prob, away_win_prob)
        """
        expected_home = self.get_expected_score(home_team, away_team)
        expected_away = 1 - expected_home
        
        # Draw probability adjustment
        # Historical draw rate in top leagues is around 24-28%
        # More likely when teams are evenly matched
        rating_diff = abs(self.get_rating(home_team) - self.get_rating(away_team))
        
        # Base draw probability (decreases as rating difference increases)
        base_draw = 0.26
        draw_decay = rating_diff / 1000  # Decay factor
        draw_prob = max(0.12, base_draw * np.exp(-draw_decay))
        
        # Adjust win probabilities
        remaining = 1 - draw_prob
        home_win = expected_home * remaining
        away_win = expected_away * remaining
        
        # Normalize to ensure sum = 1
        total = home_win + draw_prob + away_win
        return (home_win / total, draw_prob / total, away_win / total)
    
    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        match_date: pd.Timestamp | None = None,
        is_important: bool = False,
    ) -> tuple[float, float]:
        """
        Update Elo ratings after a match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            match_date: Date of the match
            is_important: If True, use higher K-factor
            
        Returns:
            Tuple of (new_home_rating, new_away_rating)
        """
        # Determine actual score (from home team perspective)
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals < away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5
        
        actual_away = 1.0 - actual_home
        
        # Calculate expected scores
        expected_home = self.get_expected_score(home_team, away_team)
        expected_away = 1 - expected_home
        
        # Select K-factor
        k = self.k_factor_important if is_important else self.k_factor
        
        # Goal difference multiplier (rewards dominant wins)
        goal_diff = abs(home_goals - away_goals)
        if goal_diff <= 1:
            mult = 1.0
        elif goal_diff == 2:
            mult = 1.5
        else:
            mult = (11 + goal_diff) / 8  # Scales with margin
        
        # Update ratings
        home_change = k * mult * (actual_home - expected_home)
        away_change = k * mult * (actual_away - expected_away)
        
        new_home = self._ratings[home_team].rating + home_change
        new_away = self._ratings[away_team].rating + away_change
        
        # Update state
        self._ratings[home_team].rating = new_home
        self._ratings[home_team].matches_played += 1
        self._ratings[home_team].last_updated = match_date
        if match_date:
            self._ratings[home_team].history.append((match_date, new_home))
        
        self._ratings[away_team].rating = new_away
        self._ratings[away_team].matches_played += 1
        self._ratings[away_team].last_updated = match_date
        if match_date:
            self._ratings[away_team].history.append((match_date, new_away))
        
        return new_home, new_away
    
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
        Process all matches and compute Elo features.
        
        Args:
            df: DataFrame with match data (must be sorted by date)
            home_col, away_col: Column names for teams
            home_goals_col, away_goals_col: Column names for goals
            date_col: Column name for match date
            
        Returns:
            DataFrame with added Elo features:
            - home_elo, away_elo: Pre-match Elo ratings
            - elo_diff: home_elo - away_elo
            - home_elo_momentum, away_elo_momentum: Recent rating changes
            - home_expected_score: Expected score for home team
        """
        result = df.copy()
        
        # Ensure sorted by date
        result[date_col] = pd.to_datetime(result[date_col])
        result = result.sort_values(date_col, kind="mergesort").reset_index(drop=True)
        
        # Initialize feature columns
        result["home_elo"] = 0.0
        result["away_elo"] = 0.0
        result["elo_diff"] = 0.0
        result["home_elo_momentum"] = 0.0
        result["away_elo_momentum"] = 0.0
        result["home_expected_score"] = 0.0
        
        for idx, row in result.iterrows():
            home = row[home_col]
            away = row[away_col]
            match_date = pd.Timestamp(row[date_col])
            
            # Get pre-match ratings (before this match is processed)
            result.at[idx, "home_elo"] = self.get_rating(home)
            result.at[idx, "away_elo"] = self.get_rating(away)
            result.at[idx, "elo_diff"] = self.get_rating(home) - self.get_rating(away)
            result.at[idx, "home_elo_momentum"] = self.get_rating_info(home).get_momentum()
            result.at[idx, "away_elo_momentum"] = self.get_rating_info(away).get_momentum()
            result.at[idx, "home_expected_score"] = self.get_expected_score(home, away)
            
            # Update ratings if match is finished
            home_goals = row.get(home_goals_col)
            away_goals = row.get(away_goals_col)
            
            if pd.notna(home_goals) and pd.notna(away_goals):
                self.update_ratings(
                    home, away,
                    int(home_goals), int(away_goals),
                    match_date=match_date,
                )
        
        return result
    
    def get_features_for_match(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, float]:
        """
        Get Elo features for an upcoming match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dictionary with Elo-based features
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        home_momentum = self.get_rating_info(home_team).get_momentum()
        away_momentum = self.get_rating_info(away_team).get_momentum()
        expected = self.get_expected_score(home_team, away_team)
        
        return {
            "home_elo": home_rating,
            "away_elo": away_rating,
            "elo_diff": home_rating - away_rating,
            "home_elo_momentum": home_momentum,
            "away_elo_momentum": away_momentum,
            "home_expected_score": expected,
        }
    
    def save_state(self) -> dict[str, Any]:
        """Save rating system state for persistence."""
        return {
            "k_factor": self.k_factor,
            "k_factor_important": self.k_factor_important,
            "home_advantage": self.home_advantage,
            "initial_rating": self.initial_rating,
            "ratings": {
                team: {
                    "rating": info.rating,
                    "matches_played": info.matches_played,
                    "last_updated": info.last_updated.isoformat() if info.last_updated else None,
                }
                for team, info in self._ratings.items()
            },
        }
    
    def load_state(self, state: dict[str, Any]) -> None:
        """Load rating system state from saved data."""
        self.k_factor = state["k_factor"]
        self.k_factor_important = state["k_factor_important"]
        self.home_advantage = state["home_advantage"]
        self.initial_rating = state["initial_rating"]
        
        self._ratings.clear()
        for team, info in state["ratings"].items():
            self._ratings[team] = EloRating(
                rating=info["rating"],
                matches_played=info["matches_played"],
                last_updated=(
                    pd.Timestamp(info["last_updated"]) 
                    if info["last_updated"] else None
                ),
            )
