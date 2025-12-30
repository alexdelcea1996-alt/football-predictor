"""
Corner-specific feature engineering.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TeamCornerHistory:
    """Stores corner history for a team."""
    corners_for: list[int] = field(default_factory=list)
    corners_against: list[int] = field(default_factory=list)
    home_corners_for: list[int] = field(default_factory=list)
    home_corners_against: list[int] = field(default_factory=list)
    away_corners_for: list[int] = field(default_factory=list)
    away_corners_against: list[int] = field(default_factory=list)


class CornersFeatureCalculator:
    """
    Calculates corner-specific features for prediction.
    
    Features computed:
    - Rolling corner averages (5, 10, 20 games)
    - Home/away corner splits
    - Corner consistency (std dev)
    - Expected total corners
    """
    
    def __init__(
        self,
        windows: list[int] | None = None,
        decay_factor: float = 0.95,
    ) -> None:
        self.windows = windows or [5, 10]
        self.decay_factor = decay_factor
        self.team_history: dict[str, TeamCornerHistory] = {}
        self.league_avg_corners: float = 10.0  # Default
    
    def _get_or_create_history(self, team: str) -> TeamCornerHistory:
        if team not in self.team_history:
            self.team_history[team] = TeamCornerHistory()
        return self.team_history[team]
    
    def _rolling_mean(self, values: list[float], window: int) -> float | None:
        if len(values) < 3:
            return None
        recent = values[-window:] if len(values) >= window else values
        return float(np.mean(recent))
    
    def _rolling_std(self, values: list[float], window: int) -> float | None:
        if len(values) < 5:
            return None
        recent = values[-window:] if len(values) >= window else values
        return float(np.std(recent))
    
    def update_history(
        self,
        home_team: str,
        away_team: str,
        home_corners: int,
        away_corners: int,
    ) -> None:
        """Update team histories after a match."""
        home_hist = self._get_or_create_history(home_team)
        away_hist = self._get_or_create_history(away_team)
        
        # Home team
        home_hist.corners_for.append(home_corners)
        home_hist.corners_against.append(away_corners)
        home_hist.home_corners_for.append(home_corners)
        home_hist.home_corners_against.append(away_corners)
        
        # Away team
        away_hist.corners_for.append(away_corners)
        away_hist.corners_against.append(home_corners)
        away_hist.away_corners_for.append(away_corners)
        away_hist.away_corners_against.append(home_corners)
    
    def get_team_features(
        self,
        team: str,
        venue: str = "all",
    ) -> dict[str, float | None]:
        """Get corner features for a team."""
        hist = self._get_or_create_history(team)
        features: dict[str, float | None] = {}
        
        # Select appropriate history based on venue
        if venue == "home":
            corners_for = hist.home_corners_for
            corners_against = hist.home_corners_against
        elif venue == "away":
            corners_for = hist.away_corners_for
            corners_against = hist.away_corners_against
        else:
            corners_for = hist.corners_for
            corners_against = hist.corners_against
        
        for w in self.windows:
            suffix = f"_{w}"
            features[f"corners_for{suffix}"] = self._rolling_mean(corners_for, w)
            features[f"corners_against{suffix}"] = self._rolling_mean(corners_against, w)
            
            # Total corners in matches
            if corners_for and corners_against:
                totals = [f + a for f, a in zip(corners_for, corners_against)]
                features[f"corners_total{suffix}"] = self._rolling_mean(totals, w)
        
        # Consistency
        features["corners_for_std"] = self._rolling_std(corners_for, 10)
        features["corners_total_std"] = None
        if corners_for and corners_against:
            totals = [f + a for f, a in zip(corners_for, corners_against)]
            features["corners_total_std"] = self._rolling_std(totals, 10)
        
        return features
    
    def get_match_features(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, float | None]:
        """Get combined corner features for a match."""
        home_features = self.get_team_features(home_team, "home")
        away_features = self.get_team_features(away_team, "away")
        
        features: dict[str, float | None] = {}
        
        # Prefix features
        for key, val in home_features.items():
            features[f"home_{key}"] = val
        for key, val in away_features.items():
            features[f"away_{key}"] = val
        
        # Combined predictions
        for w in self.windows:
            home_for = home_features.get(f"corners_for_{w}")
            home_against = home_features.get(f"corners_against_{w}")
            away_for = away_features.get(f"corners_for_{w}")
            away_against = away_features.get(f"corners_against_{w}")
            
            # Expected corners for each team
            if home_for is not None and away_against is not None:
                features[f"exp_home_corners_{w}"] = (home_for + away_against) / 2
            else:
                features[f"exp_home_corners_{w}"] = None
            
            if away_for is not None and home_against is not None:
                features[f"exp_away_corners_{w}"] = (away_for + home_against) / 2
            else:
                features[f"exp_away_corners_{w}"] = None
            
            # Expected total
            exp_home = features.get(f"exp_home_corners_{w}")
            exp_away = features.get(f"exp_away_corners_{w}")
            if exp_home is not None and exp_away is not None:
                features[f"exp_total_corners_{w}"] = exp_home + exp_away
            else:
                features[f"exp_total_corners_{w}"] = None
        
        # League average reference
        features["league_avg_corners"] = self.league_avg_corners
        
        return features
    
    def process_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all matches and compute corner features."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # Compute league average
        if "total_corners" in df.columns:
            self.league_avg_corners = df["total_corners"].mean()
        
        # Process each match
        feature_rows = []
        
        for idx, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            
            # Get features BEFORE updating history
            features = self.get_match_features(home, away)
            feature_rows.append(features)
            
            # Update history with this match's result
            if pd.notna(row.get("home_corners")) and pd.notna(row.get("away_corners")):
                self.update_history(
                    home, away,
                    int(row["home_corners"]),
                    int(row["away_corners"])
                )
        
        # Add features to dataframe
        features_df = pd.DataFrame(feature_rows)
        result = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        return result
    
    def reset(self) -> None:
        """Reset all history."""
        self.team_history.clear()
    
    def save_state(self) -> dict[str, Any]:
        """Save calculator state."""
        return {
            "team_history": {
                team: {
                    "corners_for": hist.corners_for,
                    "corners_against": hist.corners_against,
                    "home_corners_for": hist.home_corners_for,
                    "home_corners_against": hist.home_corners_against,
                    "away_corners_for": hist.away_corners_for,
                    "away_corners_against": hist.away_corners_against,
                }
                for team, hist in self.team_history.items()
            },
            "league_avg": self.league_avg_corners,
        }
    
    def load_state(self, state: dict[str, Any]) -> None:
        """Load calculator state."""
        self.league_avg_corners = state.get("league_avg", 10.0)
        self.team_history.clear()
        
        for team, data in state.get("team_history", {}).items():
            hist = TeamCornerHistory()
            hist.corners_for = data.get("corners_for", [])
            hist.corners_against = data.get("corners_against", [])
            hist.home_corners_for = data.get("home_corners_for", [])
            hist.home_corners_against = data.get("home_corners_against", [])
            hist.away_corners_for = data.get("away_corners_for", [])
            hist.away_corners_against = data.get("away_corners_against", [])
            self.team_history[team] = hist
