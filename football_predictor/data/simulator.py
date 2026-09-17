"""
Synthetic league simulator.

Used for tests, demos and pipeline benchmarking when real data is not at hand.
Unlike a naive random generator, teams here have latent attack and defence
strengths that drift between seasons, so the data contains genuine, learnable
signal, and simulated bookmaker odds are derived from the true probabilities
with noise and a margin, the way a real market behaves.

Numbers produced from simulated data validate the pipeline, not the model's
real-world accuracy: for that, download real matches (scripts/fetch_odds_data.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

DEFAULT_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Liverpool", "Luton", "Man City", "Man United", "Newcastle",
    "Nottingham Forest", "Sheffield United", "Tottenham", "West Ham", "Wolves",
]

HOME_ADVANTAGE = 0.26
BASE_SCORING = 0.15


def simulate_league(
    n_seasons: int = 3,
    teams: list[str] | None = None,
    start_date: str = "2021-08-07",
    with_odds: bool = True,
    bookmaker_margin: float = 0.05,
    bookmaker_noise: float = 0.10,
    strength_drift: float = 0.12,
    max_goals: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate several seasons of a double round-robin league.

    Args:
        n_seasons: Number of seasons to simulate
        teams: Team names (default: 20 Premier League names)
        start_date: First matchday
        with_odds: Attach simulated bookmaker odds
        bookmaker_margin: Overround added to the book (0.05 = 105% book)
        bookmaker_noise: How far the market's view drifts from the truth
        strength_drift: Between-season change in team strengths
        max_goals: Scoreline grid used for the true probabilities
        seed: Random seed

    Returns:
        Match DataFrame in the canonical schema, sorted by date
    """
    rng = np.random.default_rng(seed)
    squad = list(teams or DEFAULT_TEAMS)
    n_teams = len(squad)

    attack = rng.normal(0.0, 0.35, n_teams)
    defence = rng.normal(0.0, 0.30, n_teams)

    rows: list[dict] = []
    matchday = pd.Timestamp(start_date)

    for season in range(n_seasons):
        if season > 0:
            attack += rng.normal(0.0, strength_drift, n_teams)
            defence += rng.normal(0.0, strength_drift, n_teams)
            attack -= attack.mean()
            defence -= defence.mean()

        season_label = f"{2021 + season}/{(2022 + season) % 100:02d}"
        fixtures = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(fixtures)

        # Spread the fixture list over weekly rounds
        per_round = max(1, n_teams // 2)
        for start in range(0, len(fixtures), per_round):
            for home, away in fixtures[start : start + per_round]:
                lam = np.exp(BASE_SCORING + HOME_ADVANTAGE + attack[home] - defence[away])
                mu = np.exp(BASE_SCORING + attack[away] - defence[home])

                home_goals = int(rng.poisson(lam))
                away_goals = int(rng.poisson(mu))

                row = {
                    "date": matchday,
                    "home_team": squad[home],
                    "away_team": squad[away],
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "league": "simulated_league",
                    "season": season_label,
                    "status": "FINISHED",
                }

                if with_odds:
                    true_probs = _outcome_probabilities(lam, mu, max_goals)
                    row.update(
                        _simulate_odds(true_probs, rng, bookmaker_margin, bookmaker_noise)
                    )

                rows.append(row)

            matchday += pd.Timedelta(days=7)

        matchday += pd.Timedelta(days=60)  # summer break

    return pd.DataFrame(rows).sort_values("date", kind="mergesort").reset_index(drop=True)


def _outcome_probabilities(lam: float, mu: float, max_goals: int) -> np.ndarray:
    """True 1X2 probabilities from independent Poisson scorelines."""
    goals = np.arange(max_goals + 1)
    home_pmf = np.exp(goals * np.log(lam) - lam - gammaln(goals + 1))
    away_pmf = np.exp(goals * np.log(mu) - mu - gammaln(goals + 1))
    matrix = np.outer(home_pmf, away_pmf)
    matrix /= matrix.sum()

    return np.array([
        np.tril(matrix, -1).sum(),
        np.trace(matrix),
        np.triu(matrix, 1).sum(),
    ])


def _simulate_odds(
    true_probs: np.ndarray,
    rng: np.random.Generator,
    margin: float,
    noise: float,
) -> dict[str, float]:
    """
    Turn true probabilities into a bookmaker's prices.

    The market sees the truth imperfectly (noise on the log-odds) and prices
    in a margin, so the quoted odds imply probabilities summing above 1.
    """
    perturbed = np.exp(np.log(np.clip(true_probs, 1e-6, None)) + rng.normal(0, noise, 3))
    perturbed /= perturbed.sum()

    quoted = perturbed * (1.0 + margin)
    odds = 1.0 / quoted

    return {
        "odds_home": float(np.round(odds[0], 2)),
        "odds_draw": float(np.round(odds[1], 2)),
        "odds_away": float(np.round(odds[2], 2)),
        "odds_source": "simulated",
    }
