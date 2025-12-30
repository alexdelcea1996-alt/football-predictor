"""
API-Football client for corner statistics.

Uses RapidAPI's API-Football to fetch match statistics including corners.
"""

from datetime import date
from typing import Any
import time

import pandas as pd
import requests


class APIFootballCornersClient:
    """
    API-Football client for fetching corner statistics.
    
    Free tier: 100 requests/day via RapidAPI.
    """
    
    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
    
    # League IDs for API-Football
    LEAGUE_IDS = {
        "premier_league": 39,
        "la_liga": 140,
        "bundesliga": 78,
        "serie_a": 135,
        "ligue_1": 61,
    }
    
    def __init__(self, api_key: str) -> None:
        """
        Args:
            api_key: RapidAPI key for API-Football
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        })
    
    def _request(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        """Make API request."""
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("errors"):
            raise ValueError(f"API Error: {data['errors']}")
        
        return data
    
    def get_fixtures(
        self,
        league: str,
        season: int = 2024,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get fixtures for a league."""
        league_id = self.LEAGUE_IDS.get(league, league)
        
        params = {
            "league": league_id,
            "season": season,
        }
        if status:
            params["status"] = status
        
        data = self._request("fixtures", params)
        return data.get("response", [])
    
    def get_fixture_statistics(self, fixture_id: int) -> dict[str, Any]:
        """Get statistics for a specific fixture."""
        data = self._request("fixtures/statistics", {"fixture": fixture_id})
        return data.get("response", [])
    
    def get_corners_from_fixture(self, fixture_id: int) -> dict[str, int] | None:
        """Extract corner counts from fixture statistics."""
        stats = self.get_fixture_statistics(fixture_id)
        
        if not stats or len(stats) < 2:
            return None
        
        home_corners = 0
        away_corners = 0
        
        for team_stats in stats:
            team_name = team_stats.get("team", {}).get("name", "")
            statistics = team_stats.get("statistics", [])
            
            for stat in statistics:
                if stat.get("type") == "Corner Kicks":
                    value = stat.get("value")
                    if value is not None:
                        if team_stats == stats[0]:
                            home_corners = int(value)
                        else:
                            away_corners = int(value)
        
        return {
            "home_corners": home_corners,
            "away_corners": away_corners,
            "total_corners": home_corners + away_corners,
        }
    
    def fetch_corners_dataset(
        self,
        league: str,
        season: int = 2024,
        max_fixtures: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch corners data for multiple matches.
        
        Args:
            league: League name
            season: Season year
            max_fixtures: Maximum fixtures to fetch (API limit consideration)
        
        Returns:
            DataFrame with match details and corner counts
        """
        print(f"Fetching {league} fixtures for {season}...")
        
        # Get finished fixtures
        fixtures = self.get_fixtures(league, season, status="FT")
        
        if not fixtures:
            print(f"  No fixtures found for {league}")
            return pd.DataFrame()
        
        print(f"  Found {len(fixtures)} finished fixtures")
        
        # Limit to avoid API quota issues
        fixtures = fixtures[:max_fixtures]
        
        rows = []
        for i, fixture in enumerate(fixtures):
            fixture_id = fixture.get("fixture", {}).get("id")
            
            if not fixture_id:
                continue
            
            # Get corner stats
            try:
                corners = self.get_corners_from_fixture(fixture_id)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"    Error fetching {fixture_id}: {e}")
                continue
            
            if not corners:
                continue
            
            home = fixture.get("teams", {}).get("home", {})
            away = fixture.get("teams", {}).get("away", {})
            goals = fixture.get("goals", {})
            
            rows.append({
                "fixture_id": fixture_id,
                "date": fixture.get("fixture", {}).get("date"),
                "home_team": home.get("name"),
                "away_team": away.get("name"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "home_corners": corners["home_corners"],
                "away_corners": corners["away_corners"],
                "total_corners": corners["total_corners"],
                "league": league,
                "season": season,
            })
            
            if (i + 1) % 20 == 0:
                print(f"    Processed {i + 1}/{len(fixtures)} fixtures")
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        
        print(f"  Collected {len(df)} matches with corner data")
        return df


class CornersDataFromCSV:
    """
    Load corners data from a pre-existing CSV dataset.
    
    Use this if you have corner data from another source.
    """
    
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
    
    def load(self) -> pd.DataFrame:
        """Load and validate corners data."""
        df = pd.read_csv(self.filepath)
        
        required = ["date", "home_team", "away_team", "home_corners", "away_corners"]
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        df["date"] = pd.to_datetime(df["date"])
        
        if "total_corners" not in df.columns:
            df["total_corners"] = df["home_corners"] + df["away_corners"]
        
        return df.sort_values("date").reset_index(drop=True)
