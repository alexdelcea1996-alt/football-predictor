"""Tests for the Dixon-Coles goal model and its features."""

import numpy as np
import pandas as pd
import pytest

from football_predictor.features.poisson_features import FEATURE_COLUMNS, PoissonFeatures
from football_predictor.models.poisson_model import DixonColesModel


@pytest.fixture(scope="module")
def synthetic_league():
    """A league where team strengths are known, so the fit can be checked."""
    rng = np.random.default_rng(5)
    teams = [f"T{i:02d}" for i in range(16)]
    strength = {t: s for t, s in zip(teams, np.linspace(0.6, -0.6, len(teams)))}

    rows = []
    start = pd.Timestamp("2022-08-01")
    for week in range(70):
        order = rng.permutation(teams)
        for home, away in zip(order[:8], order[8:]):
            lam = np.exp(0.25 + strength[home] - 0.5 * strength[away])
            mu = np.exp(strength[away] - 0.5 * strength[home])
            rows.append({
                "date": start + pd.Timedelta(weeks=week),
                "home_team": home,
                "away_team": away,
                "home_goals": rng.poisson(lam),
                "away_goals": rng.poisson(mu),
            })

    return pd.DataFrame(rows), strength


class TestDixonColesModel:
    def test_recovers_team_strength_ordering(self, synthetic_league):
        df, strength = synthetic_league
        model = DixonColesModel().fit(df)

        fitted = np.array([model.attack[t] for t in strength])
        truth = np.array(list(strength.values()))
        correlation = np.corrcoef(fitted, truth)[0, 1]

        assert correlation > 0.8

    def test_probabilities_are_valid(self, synthetic_league):
        df, _ = synthetic_league
        model = DixonColesModel().fit(df)

        probs = model.predict_proba(df.head(50))
        assert probs.shape == (50, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all()

    def test_draw_probability_tracks_actual_draw_rate(self, synthetic_league):
        """The reason for having this model: draws at their natural rate."""
        df, _ = synthetic_league
        model = DixonColesModel().fit(df)

        predicted = model.predict_proba(df)[:, 1].mean()
        actual = (df["home_goals"] == df["away_goals"]).mean()

        assert predicted == pytest.approx(actual, abs=0.05)

    def test_stronger_home_team_gets_higher_win_probability(self, synthetic_league):
        df, strength = synthetic_league
        model = DixonColesModel().fit(df)

        best, worst = "T00", "T15"
        strong_home = model.predict_match(best, worst)
        weak_home = model.predict_match(worst, best)

        assert strong_home[0] > weak_home[0]

    def test_home_advantage_is_positive(self, synthetic_league):
        df, _ = synthetic_league
        assert DixonColesModel().fit(df).home_advantage > 0

    def test_expected_goals_are_plausible(self, synthetic_league):
        df, _ = synthetic_league
        model = DixonColesModel().fit(df)
        lam, mu = model.expected_goals("T00", "T15")

        assert 0.1 < lam < 6.0
        assert 0.1 < mu < 6.0

    def test_unknown_teams_fall_back_to_average(self, synthetic_league):
        df, _ = synthetic_league
        model = DixonColesModel().fit(df)

        probs = model.predict_match("Unknown FC", "Also Unknown")
        assert sum(probs) == pytest.approx(1.0)
        assert all(p > 0 for p in probs)

    def test_score_matrix_sums_to_one(self, synthetic_league):
        df, _ = synthetic_league
        model = DixonColesModel().fit(df)
        assert model.score_matrix("T00", "T15").sum() == pytest.approx(1.0)

    def test_empty_history_raises(self):
        empty = pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
        with pytest.raises(ValueError, match="No completed matches"):
            DixonColesModel().fit(empty)

    def test_state_roundtrip(self, synthetic_league):
        df, _ = synthetic_league
        model = DixonColesModel().fit(df)
        expected = model.predict_match("T00", "T15")

        restored = DixonColesModel().load_state(model.save_state())
        assert restored.predict_match("T00", "T15") == pytest.approx(expected)


class TestPoissonFeatures:
    def test_all_features_present(self, synthetic_league):
        df, _ = synthetic_league
        out = PoissonFeatures(min_matches=100, refit_every=50).process_matches(df)

        assert all(col in out.columns for col in FEATURE_COLUMNS)

    def test_unavailable_before_minimum_history(self, synthetic_league):
        df, _ = synthetic_league
        out = PoissonFeatures(min_matches=200, refit_every=50).process_matches(df)

        assert (out["poisson_available"].iloc[:200] == 0).all()
        assert out["poisson_available"].iloc[-1] == 1.0

    def test_no_look_ahead(self, synthetic_league):
        """Features for early matches must not change when later ones exist."""
        df, _ = synthetic_league
        prefix_len = 400

        full = PoissonFeatures(min_matches=100, refit_every=40).process_matches(df)
        prefix = PoissonFeatures(min_matches=100, refit_every=40).process_matches(
            df.iloc[:prefix_len]
        )

        np.testing.assert_allclose(
            full["poisson_prob_home"].iloc[:prefix_len].to_numpy(dtype=float),
            prefix["poisson_prob_home"].to_numpy(dtype=float),
            rtol=1e-9,
            equal_nan=True,
        )

    def test_missing_columns_produce_empty_features(self):
        out = PoissonFeatures().process_matches(pd.DataFrame({"home_team": ["A"]}))

        assert out["poisson_available"].iloc[0] == 0.0
        assert np.isnan(out["poisson_prob_home"].iloc[0])

    def test_state_roundtrip_keeps_predictions(self, synthetic_league):
        df, _ = synthetic_league
        features = PoissonFeatures(min_matches=100, refit_every=50)
        features.process_matches(df)
        expected = features.get_features_for_match("T00", "T15")

        restored = PoissonFeatures()
        restored.load_state(features.save_state())

        assert restored.get_features_for_match("T00", "T15")["poisson_prob_draw"] == pytest.approx(
            expected["poisson_prob_draw"]
        )
