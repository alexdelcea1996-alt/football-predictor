"""Tests for evaluation metrics."""

import pytest
import numpy as np

from football_predictor.evaluation.metrics import (
    ranked_probability_score,
    calibration_error,
    calculate_all_metrics,
)


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_rps_perfect_prediction(self):
        """Test RPS = 0 for perfect predictions."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1.0, 0.0, 0.0],  # Perfect home win
            [0.0, 1.0, 0.0],  # Perfect draw
            [0.0, 0.0, 1.0],  # Perfect away win
        ])
        
        rps = ranked_probability_score(y_true, y_proba)
        assert rps == 0.0
    
    def test_rps_worst_prediction(self):
        """Test RPS for worst possible predictions."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [0.0, 0.0, 1.0],  # Predicted away, was home
            [1.0, 0.0, 0.0],  # Predicted home, was draw
            [1.0, 0.0, 0.0],  # Predicted home, was away
        ])
        
        rps = ranked_probability_score(y_true, y_proba)
        assert rps > 0.5  # Should be high
    
    def test_rps_uniform_prediction(self):
        """Test RPS for uniform (no information) predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.ones((6, 3)) / 3.0  # Uniform
        
        rps = ranked_probability_score(y_true, y_proba)
        # Uniform should give RPS around 0.15-0.22 depending on outcome distribution
        assert 0.10 < rps < 0.30
    
    def test_rps_rewards_confidence(self):
        """Test that RPS rewards confident correct predictions."""
        y_true = np.array([0, 0])
        
        confident = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
        unsure = np.array([[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]])
        
        rps_confident = ranked_probability_score(y_true, confident)
        rps_unsure = ranked_probability_score(y_true, unsure)
        
        assert rps_confident < rps_unsure
    
    def test_calibration_error(self):
        """Test calibration error calculation."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ])
        
        cal = calibration_error(y_true, y_proba)
        
        assert "ece" in cal
        assert "mce" in cal
        assert 0 <= cal["ece"] <= 1
        assert 0 <= cal["mce"] <= 1
    
    def test_calculate_all_metrics(self):
        """Test all metrics are computed."""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 100)
        y_proba = np.random.dirichlet([1, 1, 1], 100)
        
        metrics = calculate_all_metrics(y_true, y_proba)
        
        assert "rps" in metrics
        assert "accuracy" in metrics
        assert "log_loss" in metrics
        assert "ece" in metrics
        assert "f1_macro" in metrics
        
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["rps"] > 0


class TestRPSDefinition:
    """Locks the standard RPS definition (sum of K-1 CDF terms / (K-1))."""

    def test_known_value(self):
        # p = (0.5, 0.3, 0.2), outcome = Draw
        # CDF_pred = (0.5, 0.8); CDF_true = (0, 1)
        # RPS = (0.25 + 0.04) / 2 = 0.145
        rps = ranked_probability_score(
            np.array([1]), np.array([[0.5, 0.3, 0.2]])
        )
        assert rps == pytest.approx(0.145)

    def test_uniform_forecast_matches_analytic_value(self):
        # Uniform probabilities over an equal mix of outcomes: 2/9
        y_true = np.array([0, 1, 2])
        y_proba = np.ones((3, 3)) / 3.0
        assert ranked_probability_score(y_true, y_proba) == pytest.approx(2 / 9)

    def test_maximum_is_one(self):
        # Certain and completely wrong: home predicted, away happened
        rps = ranked_probability_score(
            np.array([2]), np.array([[1.0, 0.0, 0.0]])
        )
        assert rps == pytest.approx(1.0)

    def test_per_sample_matches_mean(self):
        from football_predictor.evaluation.metrics import (
            ranked_probability_score_per_sample,
        )

        rng = np.random.default_rng(11)
        y_true = rng.integers(0, 3, 50)
        y_proba = rng.dirichlet([1, 1, 1], 50)

        per_sample = ranked_probability_score_per_sample(y_true, y_proba)
        assert len(per_sample) == 50
        assert per_sample.mean() == pytest.approx(
            ranked_probability_score(y_true, y_proba)
        )
