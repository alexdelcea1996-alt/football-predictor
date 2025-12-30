"""Tests for rolling statistics calculator."""

import pytest
import pandas as pd
import numpy as np

from football_predictor.features.rolling_stats import RollingStatsCalculator


class TestRollingStats:
    """Tests for RollingStatsCalculator."""
    
    def test_empty_history(self):
        """Test stats with no history."""
        calc = RollingStatsCalculator(windows=[5])
        stats = calc.get_team_stats("Unknown Team")
        
        assert stats["goals_for_5"] is None
        assert stats["win_rate_5"] is None
    
    def test_stats_after_matches(self):
        """Test stats are computed after processing matches."""
        calc = RollingStatsCalculator(windows=[3])
        
        matches = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=4),
            "home_team": ["A", "B", "A", "B"],
            "away_team": ["B", "A", "C", "C"],
            "home_goals": [2, 1, 3, 0],
            "away_goals": [1, 2, 0, 1],
        })
        
        result = calc.process_matches(matches)
        
        # First match should have no history
        assert pd.isna(result.iloc[0]["home_goals_for_3"])
        
        # Later matches should have features
        assert len(result) == 4
    
    def test_home_away_split(self):
        """Test home/away stats are computed separately."""
        calc = RollingStatsCalculator(windows=[5])
        
        matches = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=4),
            "home_team": ["A", "A", "B", "A"],
            "away_team": ["B", "C", "A", "D"],
            "home_goals": [2, 3, 1, 2],
            "away_goals": [0, 1, 0, 1],
        })
        
        calc.process_matches(matches)
        
        home_stats = calc.get_team_stats("A", "home")
        away_stats = calc.get_team_stats("A", "away")
        
        # A has 3 home games, 1 away
        assert home_stats["goals_for_5"] is not None
        assert away_stats["goals_for_5"] is not None
    
    def test_weighted_average(self):
        """Test exponential decay weighting."""
        calc = RollingStatsCalculator(windows=[3], decay_factor=0.9)
        
        # Recent matches should have more weight
        result = calc._get_weighted_mean([1.0, 2.0, 3.0], window=3)
        assert result is not None
        assert result > 2.0  # Should be weighted towards recent
    
    def test_reset(self):
        """Test reset clears history."""
        calc = RollingStatsCalculator()
        
        matches = pd.DataFrame({
            "date": ["2024-01-01"],
            "home_team": ["A"],
            "away_team": ["B"],
            "home_goals": [2],
            "away_goals": [1],
        })
        calc.process_matches(matches)
        
        assert calc.get_team_stats("A")["goals_for_5"] is not None
        
        calc.reset()
        
        assert calc.get_team_stats("A")["goals_for_5"] is None
