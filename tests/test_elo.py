"""Tests for Elo rating system."""

import pytest
import pandas as pd

from football_predictor.features.elo import EloRatingSystem


class TestEloRatingSystem:
    """Tests for EloRatingSystem."""
    
    def test_initial_rating(self):
        """Test that new teams get initial rating."""
        elo = EloRatingSystem(initial_rating=1500)
        assert elo.get_rating("Team A") == 1500
        assert elo.get_rating("Team B") == 1500
    
    def test_expected_score_equal_teams(self):
        """Test expected score for equal teams."""
        elo = EloRatingSystem()
        expected = elo.get_expected_score("Team A", "Team B", apply_home_advantage=False)
        assert 0.49 < expected < 0.51  # Should be ~0.5
    
    def test_expected_score_with_home_advantage(self):
        """Test home advantage increases expected score."""
        elo = EloRatingSystem(home_advantage=100)
        expected = elo.get_expected_score("Team A", "Team B", apply_home_advantage=True)
        assert expected > 0.5  # Home team should be favored
    
    def test_rating_update_after_win(self):
        """Test rating increases after win."""
        elo = EloRatingSystem(initial_rating=1500, k_factor=20)
        
        initial_home = elo.get_rating("Home Team")
        initial_away = elo.get_rating("Away Team")
        
        elo.update_ratings("Home Team", "Away Team", home_goals=2, away_goals=0)
        
        assert elo.get_rating("Home Team") > initial_home
        assert elo.get_rating("Away Team") < initial_away
    
    def test_rating_update_after_draw(self):
        """Test ratings after draw between equal teams."""
        elo = EloRatingSystem(initial_rating=1500, k_factor=20, home_advantage=0)
        
        elo.update_ratings("Team A", "Team B", home_goals=1, away_goals=1)
        
        # Equal teams drawing should result in minimal change
        assert abs(elo.get_rating("Team A") - 1500) < 5
        assert abs(elo.get_rating("Team B") - 1500) < 5
    
    def test_predict_match_probabilities_sum_to_one(self):
        """Test that predicted probabilities sum to 1."""
        elo = EloRatingSystem()
        
        home_prob, draw_prob, away_prob = elo.predict_match("Team A", "Team B")
        
        assert abs(home_prob + draw_prob + away_prob - 1.0) < 0.001
    
    def test_process_matches(self):
        """Test processing multiple matches."""
        elo = EloRatingSystem()
        
        matches = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "A", "C"],
            "home_goals": [2, 1, 0],
            "away_goals": [1, 1, 1],
        })
        
        result = elo.process_matches(matches)
        
        assert "home_elo" in result.columns
        assert "away_elo" in result.columns
        assert "elo_diff" in result.columns
        assert len(result) == 3
    
    def test_save_and_load_state(self):
        """Test state persistence."""
        elo1 = EloRatingSystem()
        elo1.update_ratings("Team A", "Team B", 3, 0)
        
        state = elo1.save_state()
        
        elo2 = EloRatingSystem()
        elo2.load_state(state)
        
        assert elo1.get_rating("Team A") == elo2.get_rating("Team A")
        assert elo1.get_rating("Team B") == elo2.get_rating("Team B")
