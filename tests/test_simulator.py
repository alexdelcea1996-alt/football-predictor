"""Tests for the synthetic league simulator."""

import numpy as np
import pytest

from football_predictor.data.simulator import simulate_league


@pytest.fixture(scope="module")
def league():
    return simulate_league(n_seasons=2, seed=11)


class TestSimulateLeague:
    def test_canonical_schema(self, league):
        for column in ("date", "home_team", "away_team", "home_goals", "away_goals"):
            assert column in league.columns

    def test_double_round_robin_per_season(self, league):
        # 20 teams -> 380 fixtures per season
        assert len(league) == 2 * 380

    def test_chronological(self, league):
        assert league["date"].is_monotonic_increasing

    def test_home_advantage_shows_up(self, league):
        home_wins = (league["home_goals"] > league["away_goals"]).mean()
        away_wins = (league["home_goals"] < league["away_goals"]).mean()
        assert home_wins > away_wins

    def test_scoring_is_realistic(self, league):
        total_goals = (league["home_goals"] + league["away_goals"]).mean()
        assert 2.0 < total_goals < 3.6

    def test_draw_rate_is_realistic(self, league):
        draws = (league["home_goals"] == league["away_goals"]).mean()
        assert 0.15 < draws < 0.35

    def test_odds_have_a_margin(self, league):
        booksum = (1 / league["odds_home"] + 1 / league["odds_draw"] + 1 / league["odds_away"])
        assert (booksum > 1.0).all()
        assert booksum.mean() == pytest.approx(1.05, abs=0.02)

    def test_odds_are_informative(self, league):
        """The simulated market must actually predict results, like a real one."""
        favourite_is_home = league["odds_home"] < league["odds_away"]
        home_win = league["home_goals"] > league["away_goals"]
        assert home_win[favourite_is_home].mean() > home_win[~favourite_is_home].mean()

    def test_teams_differ_in_strength(self, league):
        """Otherwise nothing in the data is learnable."""
        points = {}
        for row in league.itertuples():
            points.setdefault(row.home_team, 0)
            points.setdefault(row.away_team, 0)
            if row.home_goals > row.away_goals:
                points[row.home_team] += 3
            elif row.home_goals < row.away_goals:
                points[row.away_team] += 3
            else:
                points[row.home_team] += 1
                points[row.away_team] += 1

        values = np.array(list(points.values()))
        assert values.std() > 8

    def test_no_odds_variant(self):
        league = simulate_league(n_seasons=1, with_odds=False)
        assert "odds_home" not in league.columns

    def test_reproducible(self):
        first = simulate_league(n_seasons=1, seed=7)
        second = simulate_league(n_seasons=1, seed=7)
        assert first.equals(second)
