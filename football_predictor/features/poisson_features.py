"""
Dixon-Coles features for the classifier pipeline.

The goal model is refitted periodically on the matches played *before* the
fixture being described, and its scoreline distribution is turned into
features. The classifiers then get a principled estimate of the draw
probability and of expected goals, which they cannot derive from form
averages alone, while remaining free to disagree with it.

Refitting on every match would be exact but wasteful; refitting every
``refit_every`` matches (with warm starts) gives the same features to within
a few days of staleness at a fraction of the cost.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_predictor.models.poisson_model import DEFAULT_XI, DixonColesModel

FEATURE_COLUMNS = (
    "poisson_prob_home",
    "poisson_prob_draw",
    "poisson_prob_away",
    "poisson_exp_home_goals",
    "poisson_exp_away_goals",
    "poisson_exp_goal_diff",
    "poisson_exp_total_goals",
    "poisson_available",
)

REQUIRED_COLUMNS = ("date", "home_team", "away_team", "home_goals", "away_goals")


class PoissonFeatures:
    """
    Computes Dixon-Coles features for each match, chronologically.

    Args:
        refit_every: Refit the goal model every N completed matches
        min_matches: Matches required before the model is fitted at all
        window: Only fit on the most recent N matches (None = all history;
            time decay already discounts old matches)
        xi: Per-day time-decay rate passed to the goal model
        max_goals: Scoreline grid size
    """

    def __init__(
        self,
        refit_every: int = 40,
        min_matches: int = 150,
        window: int | None = None,
        xi: float = DEFAULT_XI,
        max_goals: int = 10,
    ) -> None:
        self.refit_every = refit_every
        self.min_matches = min_matches
        self.window = window
        self.xi = xi
        self.max_goals = max_goals

        self.model = DixonColesModel(xi=xi, max_goals=max_goals)
        self._history: list[dict[str, Any]] = []
        self._since_fit = 0

    def process_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Dixon-Coles features, computed only from earlier matches."""
        result = df.copy()
        missing = [c for c in REQUIRED_COLUMNS if c not in result.columns]
        if missing:
            for col in FEATURE_COLUMNS:
                result[col] = 0.0 if col == "poisson_available" else np.nan
            return result

        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date", kind="mergesort").reset_index(drop=True)

        rows: list[dict[str, float]] = []

        for _, match in result.iterrows():
            self._maybe_refit(match["date"])
            rows.append(self._features(match["home_team"], match["away_team"]))

            if pd.notna(match.get("home_goals")) and pd.notna(match.get("away_goals")):
                self._history.append({
                    "date": match["date"],
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "home_goals": match["home_goals"],
                    "away_goals": match["away_goals"],
                })
                self._since_fit += 1

        for col in FEATURE_COLUMNS:
            result[col] = [row[col] for row in rows]

        return result

    def _maybe_refit(self, as_of: pd.Timestamp) -> None:
        """Refit when enough new matches have accumulated."""
        if len(self._history) < self.min_matches:
            return
        if self.model.fitted and self._since_fit < self.refit_every:
            return

        history = pd.DataFrame(self._history)
        if self.window is not None and len(history) > self.window:
            history = history.iloc[-self.window :]

        try:
            self.model.fit(history, reference_date=as_of, warm_start=True)
            self._since_fit = 0
        except (ValueError, RuntimeError):
            # Keep the previous fit rather than dropping the features entirely
            pass

    def _features(self, home_team: str, away_team: str) -> dict[str, float]:
        """Feature dict for one fixture using the current fit."""
        if not self.model.fitted:
            return {col: (0.0 if col == "poisson_available" else np.nan)
                    for col in FEATURE_COLUMNS}

        p_home, p_draw, p_away = self.model.predict_match(home_team, away_team)
        lam, mu = self.model.expected_goals(home_team, away_team)

        return {
            "poisson_prob_home": p_home,
            "poisson_prob_draw": p_draw,
            "poisson_prob_away": p_away,
            "poisson_exp_home_goals": lam,
            "poisson_exp_away_goals": mu,
            "poisson_exp_goal_diff": lam - mu,
            "poisson_exp_total_goals": lam + mu,
            "poisson_available": 1.0,
        }

    def get_features_for_match(self, home_team: str, away_team: str) -> dict[str, float]:
        """Features for an upcoming fixture using the most recent fit."""
        return self._features(home_team, away_team)

    def save_state(self) -> dict[str, Any]:
        return {
            "refit_every": self.refit_every,
            "min_matches": self.min_matches,
            "window": self.window,
            "xi": self.xi,
            "max_goals": self.max_goals,
            "model": self.model.save_state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.refit_every = state.get("refit_every", self.refit_every)
        self.min_matches = state.get("min_matches", self.min_matches)
        self.window = state.get("window", self.window)
        self.xi = state.get("xi", self.xi)
        self.max_goals = state.get("max_goals", self.max_goals)
        self.model = DixonColesModel(xi=self.xi, max_goals=self.max_goals)
        if "model" in state:
            self.model.load_state(state["model"])
