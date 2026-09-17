"""
football-data.co.uk loader.

Free historical CSVs going back to the 1990s for most European leagues,
including closing bookmaker odds. Odds are the single strongest publicly
available predictor of match outcomes: they price in team news, motivation and
everything else a goals-only feature set cannot see.

CSVs live at https://www.football-data.co.uk/mmz4281/<season>/<div>.csv
(for example .../2324/E0.csv for the 2023/24 Premier League).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from football_predictor.config import get_settings

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Canonical league name -> football-data.co.uk division code
LEAGUE_CODES = {
    "premier_league": "E0",
    "championship": "E1",
    "la_liga": "SP1",
    "segunda": "SP2",
    "bundesliga": "D1",
    "bundesliga_2": "D2",
    "serie_a": "I1",
    "serie_b": "I2",
    "ligue_1": "F1",
    "ligue_2": "F2",
    "eredivisie": "N1",
    "primeira_liga": "P1",
    "scottish_premiership": "SC0",
    "belgian_pro_league": "B1",
    "super_lig": "T1",
    "greek_super_league": "G1",
}

CODE_TO_LEAGUE = {code: name for name, code in LEAGUE_CODES.items()}

# Core column mapping: source -> canonical
_COLUMN_MAP = {
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

# Odds column triplets in preference order. Closing prices first: they are the
# sharpest, and they are known before kick-off, so they are not leakage.
_ODDS_CANDIDATES: list[tuple[str, tuple[str, str, str]]] = [
    ("pinnacle_closing", ("PSCH", "PSCD", "PSCA")),
    ("market_average_closing", ("AvgCH", "AvgCD", "AvgCA")),
    ("market_average", ("AvgH", "AvgD", "AvgA")),
    ("betbrain_average", ("BbAvH", "BbAvD", "BbAvA")),
    ("pinnacle", ("PSH", "PSD", "PSA")),
    ("bet365", ("B365H", "B365D", "B365A")),
    ("william_hill", ("WHH", "WHD", "WHA")),
]

REQUIRED_COLUMNS = ("home_team", "away_team", "home_goals", "away_goals", "date")


def season_code(start_year: int) -> str:
    """2023 -> '2324' (the code used in football-data.co.uk URLs)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2023 -> '2023/24'."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def season_url(league: str, start_year: int) -> str:
    """Build the CSV URL for one league-season."""
    code = LEAGUE_CODES.get(league, league)
    return f"{BASE_URL}/{season_code(start_year)}/{code}.csv"


def parse_csv(
    source: str | Path,
    league: str | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """
    Parse one football-data.co.uk CSV into the canonical match schema.

    Args:
        source: Path or URL of the CSV
        league: Canonical league name (defaults to the file's Div column)
        season: Season label such as "2023/24"

    Returns:
        DataFrame with canonical columns, odds, and only played matches
    """
    raw = pd.read_csv(source, encoding="latin-1", on_bad_lines="skip")
    raw = raw.dropna(how="all", axis=1).dropna(how="all", axis=0)

    df = pd.DataFrame(index=raw.index)

    for src, dest in _COLUMN_MAP.items():
        if src in raw.columns:
            df[dest] = raw[src]

    missing_core = [c for c in ("home_team", "away_team", "home_goals", "away_goals")
                    if c not in df.columns]
    if missing_core:
        raise ValueError(f"{source}: missing required columns {missing_core}")

    df["date"] = _parse_dates(raw)

    if "Div" in raw.columns:
        div = raw["Div"].astype(str).str.strip()
        df["league"] = div.map(CODE_TO_LEAGUE).fillna(div)
    if league is not None:
        df["league"] = league
    if season is not None:
        df["season"] = season

    _attach_odds(raw, df)

    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["status"] = "FINISHED"

    return df.sort_values("date", kind="mergesort").reset_index(drop=True)


def _parse_dates(raw: pd.DataFrame) -> pd.Series:
    """Parse the Date (and optional Time) columns; files mix 2- and 4-digit years."""
    if "Date" not in raw.columns:
        raise ValueError("CSV has no Date column")

    date_str = raw["Date"].astype(str).str.strip()
    if "Time" in raw.columns:
        time_str = raw["Time"].astype(str).str.strip().replace({"nan": ""})
        combined = (date_str + " " + time_str).str.strip()
    else:
        combined = date_str

    return pd.to_datetime(combined, dayfirst=True, format="mixed", errors="coerce")


def _attach_odds(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    """Pick the best available odds triplet and attach it to the canonical frame."""
    for source_name, (h_col, d_col, a_col) in _ODDS_CANDIDATES:
        if not all(c in raw.columns for c in (h_col, d_col, a_col)):
            continue

        home = pd.to_numeric(raw[h_col], errors="coerce")
        draw = pd.to_numeric(raw[d_col], errors="coerce")
        away = pd.to_numeric(raw[a_col], errors="coerce")

        # Decimal odds below 1.0 are data errors, not prices
        valid = (home > 1.0) & (draw > 1.0) & (away > 1.0)
        if not valid.any():
            continue

        df["odds_home"] = home.where(valid)
        df["odds_draw"] = draw.where(valid)
        df["odds_away"] = away.where(valid)
        df["odds_source"] = source_name
        return


def load_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate several already-downloaded CSVs."""
    frames = [parse_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )


def download_seasons(
    leagues: Sequence[str],
    seasons: Sequence[int],
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    skip_errors: bool = True,
) -> pd.DataFrame:
    """
    Download (and cache) several league-seasons.

    Args:
        leagues: Canonical league names, e.g. ["premier_league", "la_liga"]
        seasons: Season start years, e.g. [2019, 2020, 2021]
        cache_dir: Where to keep raw CSVs (default: <data_dir>/raw)
        use_cache: Reuse already-downloaded files
        skip_errors: Continue when one league-season is unavailable

    Returns:
        Canonical match DataFrame across every league-season, sorted by date
    """
    settings = get_settings()
    cache = Path(cache_dir) if cache_dir else Path(settings.data_dir) / "raw"
    cache.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for year in seasons:
        for league in leagues:
            code = LEAGUE_CODES.get(league, league)
            target = cache / f"{code}_{season_code(year)}.csv"

            try:
                if not (use_cache and target.exists()):
                    _download(season_url(league, year), target)
                frames.append(parse_csv(target, league=league, season=season_label(year)))
            except Exception as exc:  # network, 404, malformed file
                if not skip_errors:
                    raise
                print(f"  skipped {league} {season_label(year)}: {exc}")

    if not frames:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )


def _download(url: str, target: Path) -> None:
    """Fetch one CSV to disk."""
    import requests

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)
