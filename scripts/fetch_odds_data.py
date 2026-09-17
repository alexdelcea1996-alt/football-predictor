"""
Download historical matches with bookmaker odds from football-data.co.uk.

Free, no API key, and it goes back decades. Odds are the strongest publicly
available predictor of match outcomes, and more seasons help the tree models
considerably: the previous pipeline trained on roughly one and a half seasons.

Examples:
    python scripts/fetch_odds_data.py --seasons 2018-2024
    python scripts/fetch_odds_data.py --leagues premier_league,la_liga --seasons 2020-2024
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table

from football_predictor.data.football_data_uk import LEAGUE_CODES, download_seasons

console = Console()

DEFAULT_LEAGUES = "premier_league,la_liga,bundesliga,serie_a,ligue_1"


def parse_seasons(spec: str) -> list[int]:
    """Parse '2018-2024' or '2019,2021,2023' into season start years."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        start, end = (int(part) for part in spec.split("-", 1))
        return list(range(start, end))
    return [int(part) for part in spec.split(",") if part.strip()]


@click.command()
@click.option("--leagues", default=DEFAULT_LEAGUES, help="Comma-separated league names")
@click.option("--seasons", default="2018-2024", help="Range '2018-2024' or list '2021,2022'")
@click.option("--output", "-o", default="data/matches_with_odds.csv", help="Output CSV")
@click.option("--cache-dir", default=None, help="Where to keep raw downloads")
@click.option("--refresh", is_flag=True, help="Re-download even if cached")
def main(leagues: str, seasons: str, output: str, cache_dir: str | None, refresh: bool) -> None:
    """Fetch historical matches with odds."""
    league_list = [league.strip() for league in leagues.split(",") if league.strip()]
    season_list = parse_seasons(seasons)

    unknown = [league for league in league_list if league not in LEAGUE_CODES]
    if unknown:
        console.print(f"[red]Unknown leagues: {', '.join(unknown)}[/]")
        console.print(f"[dim]Available: {', '.join(sorted(LEAGUE_CODES))}[/]")
        raise SystemExit(1)

    console.print("[bold green]Fetching matches with odds from football-data.co.uk[/]")
    console.print(f"Leagues: {', '.join(league_list)}")
    console.print(f"Seasons: {season_list[0]}/{(season_list[0] + 1) % 100:02d} "
                  f"to {season_list[-1]}/{(season_list[-1] + 1) % 100:02d}\n")

    df = download_seasons(
        leagues=league_list,
        seasons=season_list,
        cache_dir=cache_dir,
        use_cache=not refresh,
    )

    if df.empty:
        console.print("[red]No matches downloaded.[/]")
        raise SystemExit(1)

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)

    table = Table(title=f"{len(df)} matches saved to {target}")
    table.add_column("League")
    table.add_column("Matches", justify="right")
    table.add_column("With odds", justify="right")
    table.add_column("From")
    table.add_column("To")

    has_odds = "odds_home" in df.columns
    for league, group in df.groupby("league"):
        priced = int(group["odds_home"].notna().sum()) if has_odds else 0
        table.add_row(
            str(league),
            str(len(group)),
            f"{priced} ({priced / len(group):.0%})",
            str(group["date"].min())[:10],
            str(group["date"].max())[:10],
        )

    console.print(table)
    console.print(f"\n[green]Next:[/] python scripts/benchmark.py --data {target}")


if __name__ == "__main__":
    main()
