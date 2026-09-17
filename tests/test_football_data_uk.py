"""Tests for the football-data.co.uk loader."""

import pandas as pd
import pytest

from football_predictor.data.football_data_uk import (
    parse_csv,
    season_code,
    season_label,
    season_url,
)

CSV_HEADER = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,"
    "B365H,B365D,B365A,PSCH,PSCD,PSCA\n"
)
CSV_ROWS = (
    "E0,11/08/2023,20:00,Burnley,Man City,0,3,A,7,17,2,6,4,8,"
    "8.00,5.00,1.36,8.50,5.30,1.38\n"
    "E0,12/08/2023,12:30,Arsenal,Nott'm Forest,2,1,H,15,9,6,3,7,2,"
    "1.30,5.50,9.00,1.28,5.90,10.5\n"
    "E0,19/08/2023,15:00,Chelsea,Liverpool,1,1,D,13,11,4,5,6,5,"
    "2.40,3.60,2.90,2.45,3.70,2.85\n"
)


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "E0.csv"
    path.write_text(CSV_HEADER + CSV_ROWS)
    return path


class TestSeasonHelpers:
    def test_season_code(self):
        assert season_code(2023) == "2324"
        assert season_code(1999) == "9900"

    def test_season_label(self):
        assert season_label(2023) == "2023/24"

    def test_season_url(self):
        assert season_url("premier_league", 2023).endswith("/2324/E0.csv")


class TestParseCsv:
    def test_maps_to_canonical_schema(self, csv_file):
        df = parse_csv(csv_file)

        assert list(df.columns[:5]) == [
            "home_team", "away_team", "home_goals", "away_goals", "home_shots"
        ]
        assert len(df) == 3
        assert df["home_team"].iloc[0] == "Burnley"
        assert df["away_goals"].iloc[0] == 3

    def test_parses_day_first_dates_with_time(self, csv_file):
        df = parse_csv(csv_file)
        assert df["date"].iloc[0] == pd.Timestamp("2023-08-11 20:00")

    def test_sorted_chronologically(self, csv_file):
        df = parse_csv(csv_file)
        assert df["date"].is_monotonic_increasing

    def test_prefers_closing_odds(self, csv_file):
        df = parse_csv(csv_file)
        # PSCH column (closing) rather than B365H (opening)
        assert df["odds_source"].iloc[0] == "pinnacle_closing"
        assert df["odds_home"].iloc[0] == 8.50

    def test_falls_back_when_closing_odds_absent(self, tmp_path):
        header = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
        path = tmp_path / "SP1.csv"
        path.write_text(header + "SP1,20/08/2023,Sevilla,Valencia,1,2,2.20,3.30,3.40\n")

        df = parse_csv(path)
        assert df["odds_source"].iloc[0] == "bet365"
        assert df["odds_home"].iloc[0] == 2.20

    def test_league_is_resolved_from_division_code(self, csv_file):
        assert parse_csv(csv_file)["league"].iloc[0] == "premier_league"

    def test_unplayed_rows_are_dropped(self, tmp_path):
        path = tmp_path / "E0.csv"
        path.write_text(CSV_HEADER + CSV_ROWS + "E0,26/08/2023,15:00,Fulham,Brentford,,,,,,,,,,,,,,,\n")

        assert len(parse_csv(path)) == 3

    def test_missing_required_columns_raise(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("Div,Date,HomeTeam\nE0,11/08/2023,Burnley\n")

        with pytest.raises(ValueError, match="missing required columns"):
            parse_csv(path)
