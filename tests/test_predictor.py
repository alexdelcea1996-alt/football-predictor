"""Tests for the fixture prediction interface."""

import numpy as np
import pytest

from football_predictor.data.simulator import simulate_league
from football_predictor.prediction.predictor import (
    MatchPredictor,
    as_feature_value,
    fixture_odds,
)
from football_predictor.training.trainer import ModelTrainer


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    output = tmp_path_factory.mktemp("model")
    df = simulate_league(n_seasons=2, seed=3)
    # Holdout blending keeps the fixture cheap; out-of-fold blending is
    # exercised in tests/test_ensemble.py
    ModelTrainer(output_dir=output).train(df, test_size=0.2, blend_strategy="holdout")
    return output


class TestValueCoercion:
    def test_missing_values_become_zero(self):
        """`nan or 0` is nan, which used to reach the models as a missing value."""
        assert as_feature_value(None) == 0.0
        assert as_feature_value(np.nan) == 0.0
        assert as_feature_value(np.inf) == 0.0
        assert as_feature_value("nonsense") == 0.0

    def test_real_values_pass_through(self):
        assert as_feature_value(2.5) == 2.5
        assert as_feature_value(0) == 0.0

    def test_fixture_odds_extraction(self):
        assert fixture_odds({"odds_home": 2.1, "odds_draw": 3.4, "odds_away": 3.6}) == (
            2.1, 3.4, 3.6
        )
        assert fixture_odds({"odds_home": 2.1}) is None
        assert fixture_odds({"odds_home": 0.5, "odds_draw": 3.4, "odds_away": 3.6}) is None


class TestMatchPredictor:
    def test_predicts_unpriced_fixture(self, trained_model):
        """Regression: unpriced fixtures produced NaN features and crashed."""
        predictor = MatchPredictor().load(trained_model)

        predictions = predictor.predict_fixtures([
            {"date": "2025-08-16", "home_team": "Arsenal", "away_team": "Chelsea"},
        ])

        assert len(predictions) == 1
        probabilities = [
            predictions[0]["home_win_prob"],
            predictions[0]["draw_prob"],
            predictions[0]["away_win_prob"],
        ]
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-3)

    def test_odds_change_the_prediction(self, trained_model):
        predictor = MatchPredictor().load(trained_model)
        base = {"date": "2025-08-16", "home_team": "Luton", "away_team": "Man City"}

        without = predictor.predict_fixtures([base])[0]
        with_odds = predictor.predict_fixtures([
            {**base, "odds_home": 7.5, "odds_draw": 4.8, "odds_away": 1.42}
        ])[0]

        assert with_odds["away_win_prob"] > without["away_win_prob"]

    def test_unknown_teams_do_not_crash(self, trained_model):
        predictor = MatchPredictor().load(trained_model)

        prediction = predictor.predict_fixtures([
            {"date": "2025-08-16", "home_team": "Newly Promoted", "away_team": "Also New"},
        ])[0]

        assert prediction["predicted_outcome"] in {"Home Win", "Draw", "Away Win"}

    def test_requires_loading_first(self):
        with pytest.raises(RuntimeError, match="not loaded"):
            MatchPredictor().predict_fixtures([{"date": "2025-01-01", "home_team": "A", "away_team": "B"}])
