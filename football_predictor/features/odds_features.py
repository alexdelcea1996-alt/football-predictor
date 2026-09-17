"""
Bookmaker-odds features.

Odds are the strongest publicly available signal for match outcomes: the
market prices in injuries, line-ups, motivation and travel that a goals-only
feature set never sees. Raw decimal odds cannot be used as probabilities
directly because they include the bookmaker's margin (the "overround"), so
they are converted with an explicit de-vigging method.

All odds here are pre-match prices, so using them introduces no leakage.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq

DevigMethod = Literal["shin", "proportional"]

ODDS_COLUMNS = ("odds_home", "odds_draw", "odds_away")

FEATURE_COLUMNS = (
    "odds_prob_home",
    "odds_prob_draw",
    "odds_prob_away",
    "odds_overround",
    "odds_favourite_prob",
    "odds_home_minus_away",
    "odds_log_home_away",
    "odds_entropy",
    "odds_available",
)

_EPS = 1e-12


def implied_probabilities(
    odds: np.ndarray,
    method: DevigMethod = "shin",
) -> np.ndarray:
    """
    Convert decimal odds to probabilities with the margin removed.

    Args:
        odds: Array of decimal odds, shape (n_samples, n_outcomes)
        method: "shin" removes the margin assuming some informed money
            (it takes proportionally more out of longshots, which is what
            bookmakers actually do); "proportional" divides by the booksum.

    Returns:
        Probabilities of the same shape, each row summing to 1. Rows with
        invalid or missing odds come back as NaN.
    """
    odds = np.asarray(odds, dtype=float)
    out = np.full_like(odds, np.nan)

    valid = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    if not valid.any():
        return out

    inverse = 1.0 / odds[valid]
    booksum = inverse.sum(axis=1, keepdims=True)
    proportional = inverse / booksum

    if method == "proportional":
        out[valid] = proportional
        return out

    shin = np.empty_like(proportional)
    for i, (row, total) in enumerate(zip(inverse, booksum[:, 0])):
        shin[i] = _shin_row(row, total, fallback=proportional[i])
    out[valid] = shin
    return out


def _shin_row(inverse: np.ndarray, booksum: float, fallback: np.ndarray) -> np.ndarray:
    """
    Solve Shin's model for one book.

    p_i(z) = (sqrt(z^2 + 4(1-z) * b_i^2 / B) - z) / (2(1-z)), with z the
    implied share of insider money chosen so the probabilities sum to 1.
    """
    if booksum <= 1.0 + _EPS:
        return fallback

    def total_prob(z: float) -> float:
        return float(_shin_probs(inverse, booksum, z).sum() - 1.0)

    try:
        lo, hi = 0.0, 0.99
        if total_prob(lo) * total_prob(hi) > 0:
            return fallback
        z = brentq(total_prob, lo, hi, xtol=1e-10, maxiter=100)
    except (ValueError, RuntimeError):
        return fallback

    probs = _shin_probs(inverse, booksum, z)
    total = probs.sum()
    if not np.isfinite(total) or total <= _EPS:
        return fallback
    return probs / total


def _shin_probs(inverse: np.ndarray, booksum: float, z: float) -> np.ndarray:
    if z >= 1.0 - _EPS:
        return inverse / booksum
    inner = z**2 + 4.0 * (1.0 - z) * (inverse**2) / booksum
    return (np.sqrt(np.clip(inner, 0.0, None)) - z) / (2.0 * (1.0 - z))


class OddsFeatures:
    """
    Computes market-derived features for each match.

    Features (all NaN, with odds_available=0, when a match has no odds):
        odds_prob_home/draw/away: de-vigged market probabilities
        odds_overround: bookmaker margin, a proxy for market confidence
        odds_favourite_prob: probability of the most likely outcome
        odds_home_minus_away: market's home-vs-away edge
        odds_log_home_away: log ratio, a smoother version of the same
        odds_entropy: how uncertain the market is about this fixture
        odds_available: 1 when priced, 0 otherwise
    """

    def __init__(self, method: DevigMethod = "shin") -> None:
        self.method: DevigMethod = method

    def process_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add odds features to a match DataFrame (no-op columns when unpriced)."""
        result = df.copy()

        if not all(col in result.columns for col in ODDS_COLUMNS):
            for col in FEATURE_COLUMNS:
                result[col] = 0.0 if col == "odds_available" else np.nan
            return result

        odds = result[list(ODDS_COLUMNS)].to_numpy(dtype=float)
        probs = implied_probabilities(odds, self.method)

        booksum = np.sum(1.0 / np.where(odds > 1.0, odds, np.nan), axis=1)
        available = np.isfinite(probs).all(axis=1)

        favourite = np.full(len(probs), np.nan)
        home_minus_away = np.full(len(probs), np.nan)
        log_ratio = np.full(len(probs), np.nan)
        entropy = np.full(len(probs), np.nan)

        if available.any():
            priced = probs[available]
            favourite[available] = priced.max(axis=1)
            home_minus_away[available] = priced[:, 0] - priced[:, 2]
            log_ratio[available] = np.log(
                np.clip(priced[:, 0], _EPS, None) / np.clip(priced[:, 2], _EPS, None)
            )
            entropy[available] = _entropy(priced)

        result["odds_prob_home"] = probs[:, 0]
        result["odds_prob_draw"] = probs[:, 1]
        result["odds_prob_away"] = probs[:, 2]
        result["odds_overround"] = np.where(available, booksum - 1.0, np.nan)
        result["odds_favourite_prob"] = favourite
        result["odds_home_minus_away"] = home_minus_away
        result["odds_log_home_away"] = log_ratio
        result["odds_entropy"] = entropy
        result["odds_available"] = available.astype(float)

        return result

    def get_features_for_match(
        self,
        odds_home: float | None = None,
        odds_draw: float | None = None,
        odds_away: float | None = None,
    ) -> dict[str, float]:
        """Features for a single upcoming fixture, given its market prices."""
        row = pd.DataFrame(
            [{
                "odds_home": odds_home,
                "odds_draw": odds_draw,
                "odds_away": odds_away,
            }]
        )
        processed = self.process_matches(row)
        return {col: float(processed[col].iloc[0]) if pd.notna(processed[col].iloc[0]) else np.nan
                for col in FEATURE_COLUMNS}

    def save_state(self) -> dict[str, Any]:
        return {"method": self.method}

    def load_state(self, state: dict[str, Any]) -> None:
        self.method = state.get("method", "shin")


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per row (natural log) for finite probability rows."""
    safe = np.clip(probs, _EPS, None)
    return -np.sum(safe * np.log(safe), axis=1)
