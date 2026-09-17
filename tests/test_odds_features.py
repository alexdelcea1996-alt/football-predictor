"""Tests for bookmaker-odds features."""

import numpy as np
import pandas as pd
import pytest

from football_predictor.features.odds_features import (
    FEATURE_COLUMNS,
    OddsFeatures,
    implied_probabilities,
)


class TestImpliedProbabilities:
    def test_probabilities_sum_to_one(self):
        odds = np.array([[2.10, 3.40, 3.60], [1.30, 5.50, 9.00]])
        for method in ("proportional", "shin"):
            probs = implied_probabilities(odds, method)
            assert np.allclose(probs.sum(axis=1), 1.0)

    def test_margin_is_removed(self):
        """Raw inverse odds sum above 1; de-vigged probabilities must not."""
        odds = np.array([[2.10, 3.40, 3.60]])
        raw_sum = (1 / odds).sum()
        assert raw_sum > 1.0
        assert implied_probabilities(odds, "shin").sum() == pytest.approx(1.0)

    def test_shin_shifts_margin_towards_longshots(self):
        """Shin takes proportionally more margin out of the longshot."""
        odds = np.array([[1.30, 5.50, 9.00]])
        proportional = implied_probabilities(odds, "proportional")[0]
        shin = implied_probabilities(odds, "shin")[0]

        assert shin[0] > proportional[0]   # favourite gets more probability
        assert shin[2] < proportional[2]   # longshot gets less

    def test_ordering_follows_prices(self):
        probs = implied_probabilities(np.array([[1.50, 4.00, 7.00]]), "shin")[0]
        assert probs[0] > probs[1] > probs[2]

    def test_invalid_odds_return_nan(self):
        odds = np.array([[np.nan, 3.0, 3.0], [0.5, 3.0, 3.0], [2.0, 3.0, 4.0]])
        probs = implied_probabilities(odds)

        assert np.isnan(probs[0]).all()
        assert np.isnan(probs[1]).all()   # decimal odds below 1 are invalid
        assert np.isfinite(probs[2]).all()

    def test_fair_book_is_unchanged(self):
        """A book with no margin should pass through essentially untouched."""
        odds = np.array([[3.0, 3.0, 3.0]])
        assert np.allclose(implied_probabilities(odds, "shin")[0], 1 / 3, atol=1e-6)


class TestOddsFeatures:
    @pytest.fixture
    def matches(self):
        return pd.DataFrame({
            "odds_home": [2.10, 1.30, np.nan],
            "odds_draw": [3.40, 5.50, np.nan],
            "odds_away": [3.60, 9.00, np.nan],
        })

    def test_all_features_present(self, matches):
        out = OddsFeatures().process_matches(matches)
        assert all(col in out.columns for col in FEATURE_COLUMNS)

    def test_unpriced_match_is_flagged_not_guessed(self, matches):
        out = OddsFeatures().process_matches(matches)

        assert out["odds_available"].tolist() == [1.0, 1.0, 0.0]
        assert np.isnan(out["odds_prob_home"].iloc[2])

    def test_overround_is_positive_for_real_books(self, matches):
        out = OddsFeatures().process_matches(matches)
        assert (out["odds_overround"].dropna() > 0).all()

    def test_entropy_lower_for_lopsided_match(self, matches):
        out = OddsFeatures().process_matches(matches)
        assert out["odds_entropy"].iloc[1] < out["odds_entropy"].iloc[0]

    def test_missing_odds_columns_produce_empty_features(self):
        out = OddsFeatures().process_matches(pd.DataFrame({"home_team": ["A"]}))

        assert out["odds_available"].iloc[0] == 0.0
        assert np.isnan(out["odds_prob_home"].iloc[0])

    def test_single_fixture_matches_batch(self, matches):
        batch = OddsFeatures().process_matches(matches)
        single = OddsFeatures().get_features_for_match(2.10, 3.40, 3.60)

        assert single["odds_prob_home"] == pytest.approx(batch["odds_prob_home"].iloc[0])
        assert single["odds_available"] == 1.0
