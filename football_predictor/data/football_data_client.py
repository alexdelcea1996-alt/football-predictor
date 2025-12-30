"""
Football-Data.org API client.

Supports fetching historical matches and standings from football-data.org.
"""

from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

from football_predictor.config import get_settings


class FootballDataClient:
    """
    Football-Data.org API client.
    
    Free tier: 10 requests/minute, covers major European leagues.
    """
    
    BASE_URL = "https://api.football-data.org/v4"
    
    # League codes for football-data.org
    LEAGUE_CODES = {
        "premier_league": "PL",
        "la_liga": "PD",
        "bundesliga": "BL1",
        "serie_a": "SA",
        "ligue_1": "FL1",
        "champions_league": "CL",
        "eredivisie": "DED",
        "primeira_liga": "PPL",
    }
    
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": api_key})
    
    def _request(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        """Make API request."""
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def get_competitions(self) -> list[dict[str, Any]]:
        """Get available competitions."""
        data = self._request("competitions")
        return data.get("competitions", [])
    
    def get_matches(
        self,
        league: str,
        season: int | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch matches for a league.
        
        Args:
            league: League name (e.g., "premier_league")
            season: Season year (e.g., 2023 for 2023/24)
            date_from: Start date filter
            date_to: End date filter
            status: Match status (SCHEDULED, LIVE, FINISHED, etc.)
        
        Returns:
            DataFrame with match data
        """
        code = self.LEAGUE_CODES.get(league, league.upper())
        
        params: dict[str, Any] = {}
        if season:
            params["season"] = season
        if date_from:
            params["dateFrom"] = date_from.isoformat()
        if date_to:
            params["dateTo"] = date_to.isoformat()
        if status:
            params["status"] = status
        
        data = self._request(f"competitions/{code}/matches", params)
        matches = data.get("matches", [])
        
        if not matches:
            return pd.DataFrame()
        
        rows = []
        for m in matches:
            home = m.get("homeTeam", {})
            away = m.get("awayTeam", {})
            score = m.get("score", {})
            full_time = score.get("fullTime", {})
            
            rows.append({
                "match_id": m.get("id"),
                "date": m.get("utcDate"),
                "home_team": home.get("name"),
                "away_team": away.get("name"),
                "home_team_id": home.get("id"),
                "away_team_id": away.get("id"),
                "home_goals": full_time.get("home"),
                "away_goals": full_time.get("away"),
                "status": m.get("status"),
                "matchday": m.get("matchday"),
                "league": league,
                "season": f"{season}/{season+1}" if season else "unknown",
            })
        
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    
    def get_standings(self, league: str, season: int | None = None) -> pd.DataFrame:
        """Fetch league standings."""
        code = self.LEAGUE_CODES.get(league, league.upper())
        
        params = {"season": season} if season else {}
        data = self._request(f"competitions/{code}/standings", params)
        
        standings = data.get("standings", [])
        if not standings:
            return pd.DataFrame()
        
        # Get total standings (not home/away split)
        total = next((s for s in standings if s.get("type") == "TOTAL"), standings[0])
        table = total.get("table", [])
        
        rows = []
        for t in table:
            team = t.get("team", {})
            rows.append({
                "position": t.get("position"),
                "team": team.get("name"),
                "team_id": team.get("id"),
                "played": t.get("playedGames"),
                "won": t.get("won"),
                "drawn": t.get("draw"),
                "lost": t.get("lost"),
                "goals_for": t.get("goalsFor"),
                "goals_against": t.get("goalsAgainst"),
                "goal_diff": t.get("goalDifference"),
                "points": t.get("points"),
            })
        
        return pd.DataFrame(rows)
    
    def get_all_leagues_matches(
        self,
        leagues: list[str] | None = None,
        season: int = 2023,
    ) -> pd.DataFrame:
        """
        Fetch matches from multiple leagues.
        
        Args:
            leagues: List of league names (defaults to top 5)
            season: Season year
        
        Returns:
            Combined DataFrame
        """
        if leagues is None:
            leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
        
        all_matches = []
        for league in leagues:
            try:
                df = self.get_matches(league, season=season, status="FINISHED")
                if not df.empty:
                    all_matches.append(df)
                    print(f"  ✓ {league}: {len(df)} matches")
            except Exception as e:
                print(f"  ✗ {league}: {e}")
        
        if not all_matches:
            return pd.DataFrame()
        
        return pd.concat(all_matches, ignore_index=True).sort_values("date").reset_index(drop=True)
