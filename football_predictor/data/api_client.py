"""
API clients for football data providers.

Supports Sportmonks (primary) and API-Football (fallback) with robust error handling,
rate limiting, and retry logic.
"""

import time
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel

from football_predictor.config import get_settings


class Match(BaseModel):
    """Standardized match data model."""
    
    match_id: int
    date: datetime
    league: str
    season: str
    round: str | None = None
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    home_goals: int | None = None
    away_goals: int | None = None
    status: str  # "finished", "scheduled", "live", "postponed"
    
    # Optional detailed statistics
    home_xg: float | None = None
    away_xg: float | None = None
    home_shots: int | None = None
    away_shots: int | None = None
    home_shots_on_target: int | None = None
    away_shots_on_target: int | None = None
    home_possession: float | None = None
    away_possession: float | None = None
    home_corners: int | None = None
    away_corners: int | None = None


class BaseAPIClient(ABC):
    """Abstract base class for API clients."""
    
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self._last_request_time: float = 0
        
        settings = get_settings()
        self._requests_per_minute = settings.api.requests_per_minute
        self._request_timeout = settings.api.request_timeout
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        min_interval = 60.0 / self._requests_per_minute
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> dict[str, Any]:
        """Make an API request with retry logic."""
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        headers = self._get_headers()
        
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self._request_timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    raise ConnectionError(f"API request failed after {retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise ConnectionError("API request failed unexpectedly")
    
    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Get API-specific headers."""
        pass
    
    @abstractmethod
    def get_fixtures(
        self,
        league: str,
        season: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[Match]:
        """Get fixtures for a league and season."""
        pass
    
    @abstractmethod
    def get_standings(self, league: str, season: str) -> pd.DataFrame:
        """Get league standings."""
        pass
    
    def get_fixtures_df(
        self,
        league: str,
        season: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Get fixtures as a DataFrame."""
        matches = self.get_fixtures(league, season, start_date, end_date)
        if not matches:
            return pd.DataFrame()
        
        data = [match.model_dump() for match in matches]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df


class SportmonksClient(BaseAPIClient):
    """Sportmonks API client implementation."""
    
    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or settings.api.sportmonks_api_key
        if not key:
            raise ValueError(
                "Sportmonks API key required. Set FP_API_SPORTMONKS_API_KEY environment variable."
            )
        super().__init__(key, settings.api.sportmonks_base_url)
        self._league_ids = settings.data.league_ids_sportmonks
    
    def _get_headers(self) -> dict[str, str]:
        return {"Authorization": self.api_key}
    
    def _get_league_id(self, league: str) -> int:
        """Get Sportmonks league ID for a league name."""
        if league not in self._league_ids:
            raise ValueError(f"Unknown league: {league}. Available: {list(self._league_ids.keys())}")
        return self._league_ids[league]
    
    def get_fixtures(
        self,
        league: str,
        season: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[Match]:
        """Fetch fixtures from Sportmonks API."""
        league_id = self._get_league_id(league)
        
        # First, get season ID
        seasons_data = self._make_request(f"leagues/{league_id}/seasons")
        season_id = None
        for s in seasons_data.get("data", []):
            if s.get("name") == season:
                season_id = s.get("id")
                break
        
        if not season_id:
            raise ValueError(f"Season {season} not found for league {league}")
        
        # Fetch fixtures with statistics
        params = {
            "include": "scores;statistics",
            "per_page": 100,
        }
        if start_date:
            params["filters"] = f"fixtureStartDate:{start_date.isoformat()}"
        
        fixtures_data = self._make_request(f"fixtures/seasons/{season_id}", params)
        
        matches = []
        for fixture in fixtures_data.get("data", []):
            match = self._parse_fixture(fixture, league, season)
            if match and (not end_date or match.date.date() <= end_date):
                matches.append(match)
        
        return matches
    
    def _parse_fixture(self, fixture: dict[str, Any], league: str, season: str) -> Match | None:
        """Parse a Sportmonks fixture into a Match object."""
        try:
            # Parse status
            state = fixture.get("state", {})
            status_map = {
                "FT": "finished",
                "NS": "scheduled",
                "LIVE": "live",
                "PST": "postponed",
            }
            status = status_map.get(state.get("short", ""), "unknown")
            
            # Parse scores
            scores = fixture.get("scores", [])
            home_goals = away_goals = None
            for score in scores:
                if score.get("description") == "CURRENT":
                    if score.get("participant_id") == fixture.get("home_team_id"):
                        home_goals = score.get("score")
                    else:
                        away_goals = score.get("score")
            
            # Parse statistics
            stats = fixture.get("statistics", [])
            home_stats = {}
            away_stats = {}
            for stat in stats:
                participant_id = stat.get("participant_id")
                stat_type = stat.get("type", {}).get("code", "")
                value = stat.get("data", {}).get("value")
                
                if participant_id == fixture.get("home_team_id"):
                    home_stats[stat_type] = value
                else:
                    away_stats[stat_type] = value
            
            return Match(
                match_id=fixture["id"],
                date=datetime.fromisoformat(fixture["starting_at"].replace("Z", "+00:00")),
                league=league,
                season=season,
                round=fixture.get("round", {}).get("name"),
                home_team=fixture.get("name", "").split(" vs ")[0] if " vs " in fixture.get("name", "") else fixture.get("home_team", {}).get("name", "Unknown"),
                away_team=fixture.get("name", "").split(" vs ")[-1] if " vs " in fixture.get("name", "") else fixture.get("away_team", {}).get("name", "Unknown"),
                home_team_id=fixture.get("home_team_id", 0),
                away_team_id=fixture.get("away_team_id", 0),
                home_goals=home_goals,
                away_goals=away_goals,
                status=status,
                home_xg=home_stats.get("expected-goals"),
                away_xg=away_stats.get("expected-goals"),
                home_shots=home_stats.get("shots-total"),
                away_shots=away_stats.get("shots-total"),
                home_shots_on_target=home_stats.get("shots-on-target"),
                away_shots_on_target=away_stats.get("shots-on-target"),
                home_possession=home_stats.get("ball-possession"),
                away_possession=away_stats.get("ball-possession"),
            )
        except (KeyError, ValueError):
            return None
    
    def get_standings(self, league: str, season: str) -> pd.DataFrame:
        """Fetch league standings from Sportmonks."""
        league_id = self._get_league_id(league)
        
        # Get current season standings
        standings_data = self._make_request(
            f"standings/seasons/{league_id}",
            {"include": "participant"}
        )
        
        rows = []
        for standing in standings_data.get("data", []):
            rows.append({
                "position": standing.get("position"),
                "team": standing.get("participant", {}).get("name"),
                "team_id": standing.get("participant_id"),
                "played": standing.get("details", {}).get("games_played"),
                "won": standing.get("details", {}).get("won"),
                "drawn": standing.get("details", {}).get("draw"),
                "lost": standing.get("details", {}).get("lost"),
                "goals_for": standing.get("details", {}).get("goals_scored"),
                "goals_against": standing.get("details", {}).get("goals_conceded"),
                "goal_diff": standing.get("details", {}).get("goal_difference"),
                "points": standing.get("points"),
            })
        
        return pd.DataFrame(rows)


class APIFootballClient(BaseAPIClient):
    """API-Football client implementation (fallback)."""
    
    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or settings.api.api_football_key
        if not key:
            raise ValueError(
                "API-Football key required. Set FP_API_API_FOOTBALL_KEY environment variable."
            )
        super().__init__(key, settings.api.api_football_base_url)
        self._league_ids = settings.data.league_ids_api_football
    
    def _get_headers(self) -> dict[str, str]:
        return {"x-apisports-key": self.api_key}
    
    def _get_league_id(self, league: str) -> int:
        """Get API-Football league ID for a league name."""
        if league not in self._league_ids:
            raise ValueError(f"Unknown league: {league}. Available: {list(self._league_ids.keys())}")
        return self._league_ids[league]
    
    def get_fixtures(
        self,
        league: str,
        season: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[Match]:
        """Fetch fixtures from API-Football."""
        league_id = self._get_league_id(league)
        
        # API-Football uses year for season (e.g., "2023" for 2023/24)
        season_year = season.split("/")[0] if "/" in season else season.split("-")[0]
        
        params: dict[str, Any] = {
            "league": league_id,
            "season": season_year,
        }
        if start_date:
            params["from"] = start_date.isoformat()
        if end_date:
            params["to"] = end_date.isoformat()
        
        response = self._make_request("fixtures", params)
        
        matches = []
        for fixture in response.get("response", []):
            match = self._parse_fixture(fixture, league, season)
            if match:
                matches.append(match)
        
        return matches
    
    def _parse_fixture(self, fixture: dict[str, Any], league: str, season: str) -> Match | None:
        """Parse an API-Football fixture into a Match object."""
        try:
            fixture_info = fixture.get("fixture", {})
            teams = fixture.get("teams", {})
            goals = fixture.get("goals", {})
            
            # Parse status
            status_short = fixture_info.get("status", {}).get("short", "")
            status_map = {
                "FT": "finished",
                "AET": "finished",
                "PEN": "finished",
                "NS": "scheduled",
                "TBD": "scheduled",
                "1H": "live",
                "2H": "live",
                "HT": "live",
                "PST": "postponed",
                "CANC": "cancelled",
            }
            status = status_map.get(status_short, "unknown")
            
            return Match(
                match_id=fixture_info["id"],
                date=datetime.fromisoformat(fixture_info["date"].replace("Z", "+00:00")),
                league=league,
                season=season,
                round=fixture.get("league", {}).get("round"),
                home_team=teams.get("home", {}).get("name", "Unknown"),
                away_team=teams.get("away", {}).get("name", "Unknown"),
                home_team_id=teams.get("home", {}).get("id", 0),
                away_team_id=teams.get("away", {}).get("id", 0),
                home_goals=goals.get("home"),
                away_goals=goals.get("away"),
                status=status,
            )
        except (KeyError, ValueError):
            return None
    
    def get_standings(self, league: str, season: str) -> pd.DataFrame:
        """Fetch league standings from API-Football."""
        league_id = self._get_league_id(league)
        season_year = season.split("/")[0] if "/" in season else season.split("-")[0]
        
        response = self._make_request(
            "standings",
            {"league": league_id, "season": season_year}
        )
        
        rows = []
        standings = response.get("response", [])
        if standings:
            league_standings = standings[0].get("league", {}).get("standings", [[]])[0]
            for standing in league_standings:
                rows.append({
                    "position": standing.get("rank"),
                    "team": standing.get("team", {}).get("name"),
                    "team_id": standing.get("team", {}).get("id"),
                    "played": standing.get("all", {}).get("played"),
                    "won": standing.get("all", {}).get("win"),
                    "drawn": standing.get("all", {}).get("draw"),
                    "lost": standing.get("all", {}).get("lose"),
                    "goals_for": standing.get("all", {}).get("goals", {}).get("for"),
                    "goals_against": standing.get("all", {}).get("goals", {}).get("against"),
                    "goal_diff": standing.get("goalsDiff"),
                    "points": standing.get("points"),
                })
        
        return pd.DataFrame(rows)


class CSVDataClient:
    """
    CSV-based data client for offline/sample data.
    
    Useful when API keys are not available or for testing.
    """
    
    def __init__(self, data_dir: str | None = None) -> None:
        from pathlib import Path
        settings = get_settings()
        self.data_dir = Path(data_dir) if data_dir else settings.data_dir
    
    def load_matches(self, filepath: str) -> pd.DataFrame:
        """
        Load matches from a CSV file.
        
        Expected columns:
        - date: Match date (YYYY-MM-DD)
        - home_team: Home team name
        - away_team: Away team name  
        - home_goals: Home team goals (optional for scheduled matches)
        - away_goals: Away team goals (optional for scheduled matches)
        - league: League name (optional)
        - season: Season string (optional)
        
        Additional columns (optional):
        - home_xg, away_xg: Expected goals
        - home_shots, away_shots: Total shots
        - home_possession, away_possession: Possession %
        """
        from pathlib import Path
        
        path = Path(filepath)
        if not path.is_absolute():
            path = self.data_dir / filepath
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        df = pd.read_csv(path)
        
        # Validate required columns
        required = ["date", "home_team", "away_team"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Parse dates
        df["date"] = pd.to_datetime(df["date"])
        
        # Add default values
        if "league" not in df.columns:
            df["league"] = "unknown"
        if "season" not in df.columns:
            df["season"] = "unknown"
        
        # Determine status
        if "home_goals" in df.columns and "away_goals" in df.columns:
            df["status"] = df.apply(
                lambda x: "finished" if pd.notna(x["home_goals"]) else "scheduled",
                axis=1
            )
        else:
            df["status"] = "scheduled"
        
        return df.sort_values("date").reset_index(drop=True)
    
    def save_matches(self, df: pd.DataFrame, filepath: str) -> None:
        """Save matches to a CSV file."""
        from pathlib import Path
        
        path = Path(filepath)
        if not path.is_absolute():
            path = self.data_dir / filepath
        
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
