"""
Dixon-Coles bivariate Poisson goal model.

Classifiers learn the 1X2 label directly and systematically under-predict
draws, because a draw is never the most likely single outcome for most
fixtures. A goal-based model avoids that: it estimates each team's attack and
defence strength, derives a scoreline distribution, and reads the draw
probability off the diagonal, so draws come out at their natural rate.

Model (Dixon & Coles, 1997):

    log lambda = attack_home + defence_away + home_advantage   (home goals)
    log mu     = attack_away + defence_home                    (away goals)

with a low-score dependence correction (rho) that fixes the independent
Poisson model's known underestimate of 0-0 and 1-1, and exponential time decay
(xi) so older matches count for less.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

_EPS = 1e-10
# Dixon-Coles per-day decay; ~0.0018 gives a half-life of roughly a year
DEFAULT_XI = 0.0018


class DixonColesModel:
    """
    Dixon-Coles goal model producing calibrated 1X2 probabilities.

    Usage:
        model = DixonColesModel().fit(matches_df)
        p_home, p_draw, p_away = model.predict_match("Arsenal", "Chelsea")
    """

    def __init__(
        self,
        xi: float = DEFAULT_XI,
        max_goals: int = 10,
        rho_bounds: tuple[float, float] = (-0.2, 0.2),
        max_iter: int = 200,
    ) -> None:
        self.xi = xi
        self.max_goals = max_goals
        self.rho_bounds = rho_bounds
        self.max_iter = max_iter

        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_advantage: float = 0.25
        self.rho: float = 0.0
        self._fitted = False
        self._last_params: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        df: pd.DataFrame,
        reference_date: pd.Timestamp | None = None,
        warm_start: bool = True,
    ) -> "DixonColesModel":
        """
        Fit attack/defence ratings by weighted maximum likelihood.

        Args:
            df: Matches with date, home_team, away_team, home_goals, away_goals
            reference_date: "Today" for the time-decay weights (default: the
                latest date in df). Matches are weighted exp(-xi * days_before).
            warm_start: Reuse the previous solution as the starting point,
                which makes repeated refits during a backtest much cheaper.
        """
        data = df.dropna(subset=["home_goals", "away_goals", "home_team", "away_team"])
        if data.empty:
            raise ValueError("No completed matches to fit on")

        home_goals = data["home_goals"].to_numpy(dtype=int)
        away_goals = data["away_goals"].to_numpy(dtype=int)

        teams = sorted(set(data["home_team"]) | set(data["away_team"]))
        index = {team: i for i, team in enumerate(teams)}
        home_idx = data["home_team"].map(index).to_numpy()
        away_idx = data["away_team"].map(index).to_numpy()

        weights = self._time_weights(data["date"], reference_date)

        n_teams = len(teams)
        x0 = self._starting_point(teams, n_teams, warm_start)

        # Precompute the constant part of the Poisson log-likelihood
        log_factorials = gammaln(home_goals + 1) + gammaln(away_goals + 1)

        def negative_log_likelihood(params: np.ndarray) -> float:
            attack, defence, home_adv, rho = self._unpack(params, n_teams)

            log_lambda = attack[home_idx] - defence[away_idx] + home_adv
            log_mu = attack[away_idx] - defence[home_idx]
            lam = np.exp(np.clip(log_lambda, -10, 5))
            mu = np.exp(np.clip(log_mu, -10, 5))

            log_lik = (
                home_goals * np.log(lam)
                + away_goals * np.log(mu)
                - lam
                - mu
                - log_factorials
            )
            log_lik = log_lik + np.log(
                np.clip(_tau(home_goals, away_goals, lam, mu, rho), _EPS, None)
            )

            return float(-np.sum(weights * log_lik))

        bounds = (
            [(-3.0, 3.0)] * n_teams      # attack
            + [(-3.0, 3.0)] * n_teams    # defence
            + [(-1.0, 1.0)]              # home advantage
            + [self.rho_bounds]          # rho
        )

        result = minimize(
            negative_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-8},
        )

        attack, defence, home_adv, rho = self._unpack(result.x, n_teams)

        self.teams = teams
        self.attack = dict(zip(teams, attack))
        self.defence = dict(zip(teams, defence))
        self.home_advantage = float(home_adv)
        self.rho = float(rho)
        self._last_params = result.x
        self._fitted = True
        return self

    def _time_weights(
        self,
        dates: pd.Series,
        reference_date: pd.Timestamp | None,
    ) -> np.ndarray:
        """Exponential decay weights; recent matches dominate the fit."""
        if self.xi <= 0:
            return np.ones(len(dates))
        parsed = pd.to_datetime(dates)
        reference = reference_date or parsed.max()
        days_before = (pd.Timestamp(reference) - parsed).dt.days.to_numpy(dtype=float)
        return np.exp(-self.xi * np.clip(days_before, 0, None))

    def _starting_point(self, teams: list[str], n_teams: int, warm_start: bool) -> np.ndarray:
        """Previous solution where possible, otherwise a neutral start."""
        x0 = np.zeros(2 * n_teams + 2)
        x0[-2] = self.home_advantage
        x0[-1] = self.rho

        if warm_start and self._fitted:
            for i, team in enumerate(teams):
                x0[i] = self.attack.get(team, 0.0)
                x0[n_teams + i] = self.defence.get(team, 0.0)

        return x0

    @staticmethod
    def _unpack(params: np.ndarray, n_teams: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Split the parameter vector, centring attack/defence for identifiability."""
        attack = params[:n_teams]
        defence = params[n_teams : 2 * n_teams]
        attack = attack - attack.mean()
        defence = defence - defence.mean()
        return attack, defence, float(params[-2]), float(params[-1])

    # -------------------------------------------------------------- predict

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected goals for (home, away). Unknown teams score league-average."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        attack_home = self.attack.get(home_team, 0.0)
        attack_away = self.attack.get(away_team, 0.0)
        defence_home = self.defence.get(home_team, 0.0)
        defence_away = self.defence.get(away_team, 0.0)

        lam = float(np.exp(np.clip(attack_home - defence_away + self.home_advantage, -10, 5)))
        mu = float(np.exp(np.clip(attack_away - defence_home, -10, 5)))
        return lam, mu

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Joint distribution over scorelines up to max_goals."""
        lam, mu = self.expected_goals(home_team, away_team)
        goals = np.arange(self.max_goals + 1)

        home_probs = np.exp(goals * np.log(lam) - lam - gammaln(goals + 1))
        away_probs = np.exp(goals * np.log(mu) - mu - gammaln(goals + 1))
        matrix = np.outer(home_probs, away_probs)

        # Dixon-Coles dependence correction on the low scorelines
        matrix[0, 0] *= 1.0 - lam * mu * self.rho
        matrix[0, 1] *= 1.0 + lam * self.rho
        matrix[1, 0] *= 1.0 + mu * self.rho
        matrix[1, 1] *= 1.0 - self.rho

        matrix = np.clip(matrix, 0.0, None)
        total = matrix.sum()
        return matrix / total if total > _EPS else matrix

    def predict_match(self, home_team: str, away_team: str) -> tuple[float, float, float]:
        """1X2 probabilities for one fixture."""
        matrix = self.score_matrix(home_team, away_team)
        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())

        total = home_win + draw + away_win
        if total <= _EPS:
            return 1 / 3, 1 / 3, 1 / 3
        return home_win / total, draw / total, away_win / total

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """1X2 probabilities for a DataFrame of fixtures, shape (n, 3)."""
        return np.array([
            self.predict_match(row.home_team, row.away_team)
            for row in df.itertuples()
        ])

    def team_ratings(self) -> pd.DataFrame:
        """Fitted attack/defence ratings, strongest attack first."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return pd.DataFrame(
            {
                "team": self.teams,
                "attack": [self.attack[t] for t in self.teams],
                "defence": [self.defence[t] for t in self.teams],
            }
        ).sort_values("attack", ascending=False).reset_index(drop=True)

    # ----------------------------------------------------------- state I/O

    def save_state(self) -> dict[str, Any]:
        return {
            "xi": self.xi,
            "max_goals": self.max_goals,
            "teams": self.teams,
            "attack": self.attack,
            "defence": self.defence,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
            "fitted": self._fitted,
        }

    def load_state(self, state: dict[str, Any]) -> "DixonColesModel":
        self.xi = state.get("xi", self.xi)
        self.max_goals = state.get("max_goals", self.max_goals)
        self.teams = state.get("teams", [])
        self.attack = state.get("attack", {})
        self.defence = state.get("defence", {})
        self.home_advantage = state.get("home_advantage", 0.25)
        self.rho = state.get("rho", 0.0)
        self._fitted = state.get("fitted", False)
        return self


def _tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Dixon-Coles low-score dependence correction, vectorized over matches."""
    tau = np.ones_like(lam, dtype=float)

    zero_zero = (home_goals == 0) & (away_goals == 0)
    zero_one = (home_goals == 0) & (away_goals == 1)
    one_zero = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)

    tau[zero_zero] = 1.0 - lam[zero_zero] * mu[zero_zero] * rho
    tau[zero_one] = 1.0 + lam[zero_one] * rho
    tau[one_zero] = 1.0 + mu[one_zero] * rho
    tau[one_one] = 1.0 - rho

    return tau
