"""
The Odds API client — fetches historical and live UFC/MMA betting odds.

Get a free API key at https://the-odds-api.com
"""

import logging
import time
from typing import Optional

import requests
import pandas as pd

from src.config import ODDS_API_KEY, ODDS_API_BASE_URL, ODDS_SPORT, RAW_DATA_DIR
from src.data.name_utils import canonical_fighter_display_name, normalize_cross_source_name

logger = logging.getLogger(__name__)


class OddsClient:
    """Client for The Odds API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ODDS_API_KEY
        if not self.api_key:
            logger.warning(
                "No Odds API key configured. Set ODDS_API_KEY in .env file. "
                "Get a free key at https://the-odds-api.com"
            )
        self.base_url = ODDS_API_BASE_URL

    @staticmethod
    def _resolve_sport_key(sport_key: Optional[str]) -> str:
        return sport_key or ODDS_SPORT

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make authenticated GET request."""
        if not self.api_key:
            raise ValueError("ODDS_API_KEY not set. Get one at https://the-odds-api.com")
        params = params or {}
        params["apiKey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        attempts = 3
        last_exc: Exception | None = None
        resp = None
        for attempt in range(1, attempts + 1):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                break
            except requests.HTTPError as exc:
                last_exc = exc
                status_code = exc.response.status_code if exc.response is not None else None
                retryable = status_code == 429 or (status_code is not None and status_code >= 500)
                if attempt >= attempts or not retryable:
                    sanitized = str(exc).replace(self.api_key, "***") if self.api_key else str(exc)
                    raise requests.HTTPError(sanitized, response=exc.response) from None
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise

            logger.warning(
                "Odds API request failed for %s (attempt %s/%s): %s",
                endpoint,
                attempt,
                attempts,
                last_exc,
            )
            time.sleep(min(2 ** (attempt - 1), 4))
        else:
            raise last_exc or RuntimeError(f"Odds API request failed for {endpoint}")

        # Log remaining API usage
        remaining = resp.headers.get("x-requests-remaining", "?") if resp is not None else "?"
        used = resp.headers.get("x-requests-used", "?") if resp is not None else "?"
        logger.info(f"Odds API: {used} used, {remaining} remaining")

        return resp.json() if resp is not None else {}

    def list_sports(self, all_sports: bool = False) -> list[dict]:
        """List available sports from The Odds API /v4/sports endpoint."""
        data = self._get(
            "sports",
            params={"all": str(bool(all_sports)).lower()},
        )
        logger.info("Got %s sports from Odds API", len(data))
        return data

    def get_live_odds(
        self,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "decimal",
        sport_key: Optional[str] = None,
    ) -> list[dict]:
        """
        Get live/upcoming MMA odds.

        Returns list of events with odds from multiple bookmakers.
        """
        resolved_sport_key = self._resolve_sport_key(sport_key)
        data = self._get(
            f"sports/{resolved_sport_key}/odds",
            params={
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
            },
        )
        logger.info("Got odds for %s upcoming events from %s", len(data), resolved_sport_key)
        return data

    def get_event_odds(
        self,
        event_id: str,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "decimal",
        sport_key: Optional[str] = None,
    ) -> dict:
        """Get odds for a specific event by ID."""
        resolved_sport_key = self._resolve_sport_key(sport_key)
        return self._get(
            f"sports/{resolved_sport_key}/events/{event_id}/odds",
            params={
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
            },
        )

    def get_historical_odds(
        self,
        date: str,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "decimal",
    ) -> list[dict]:
        """
        Get historical MMA odds for a specific date (ISO 8601 format).

        Note: Historical odds require a paid plan on The Odds API.
        Date format: '2024-01-20T12:00:00Z'
        """
        data = self._get(
            f"sports/{ODDS_SPORT}/odds-history",
            params={
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
                "date": date,
            },
        )
        return data.get("data", []) if isinstance(data, dict) else data

    def get_historical_sport_odds(
        self,
        date: str,
        sport_key: str,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "decimal",
    ) -> dict:
        """
        Get historical odds snapshots for a specific sport key.

        The response includes the snapshot timestamp plus a ``data`` payload of
        events available at or before the requested time.
        """
        return self._get(
            f"historical/sports/{self._resolve_sport_key(sport_key)}/odds",
            params={
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
                "date": date,
            },
        )

    def get_scores(self, days_from: int = 3) -> list[dict]:
        """Get recent MMA event scores/results."""
        return self._get(
            f"sports/{ODDS_SPORT}/scores",
            params={"daysFrom": days_from},
        )

    def odds_to_dataframe(self, odds_data: list[dict]) -> pd.DataFrame:
        """
        Convert odds API response to a flat DataFrame.

        Returns DataFrame with columns:
            event_id, commence_time, fighter_a, fighter_b,
            bookmaker, a_odds, b_odds, a_implied_prob, b_implied_prob
        """
        rows = []
        for event in odds_data:
            event_id = event.get("id", "")
            commence = event.get("commence_time", "")
            raw_home = str(event.get("home_team", "") or "")
            raw_away = str(event.get("away_team", "") or "")
            home = canonical_fighter_display_name(raw_home)
            away = canonical_fighter_display_name(raw_away)
            if str(home).casefold() <= str(away).casefold():
                fighter_a = home
                fighter_b = away
            else:
                fighter_a = away
                fighter_b = home

            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker.get("title", "")
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = {
                        str(o["name"]): o["price"]
                        for o in market.get("outcomes", [])
                        if "name" in o and "price" in o
                    }
                    canonical_outcomes = {
                        canonical_fighter_display_name(o["name"]): o["price"]
                        for o in market.get("outcomes", [])
                        if "name" in o and "price" in o
                    }
                    home_odds = outcomes.get(raw_home, canonical_outcomes.get(home, None))
                    away_odds = outcomes.get(raw_away, canonical_outcomes.get(away, None))

                    if home_odds is not None and away_odds is not None:
                        if fighter_a == home:
                            a_odds = home_odds
                            b_odds = away_odds
                        else:
                            a_odds = away_odds
                            b_odds = home_odds
                        rows.append({
                            "event_id": event_id,
                            "commence_time": commence,
                            "fighter_a": fighter_a,
                            "fighter_b": fighter_b,
                            "bookmaker": bk_name,
                            "a_odds": a_odds,
                            "b_odds": b_odds,
                            "a_implied_prob": 1.0 / a_odds,
                            "b_implied_prob": 1.0 / b_odds,
                        })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
            # Compute fair probabilities (remove vig)
            total_implied = df["a_implied_prob"] + df["b_implied_prob"]
            df["a_fair_prob"] = df["a_implied_prob"] / total_implied
            df["b_fair_prob"] = df["b_implied_prob"] / total_implied

        return df

    def get_consensus_odds(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Average odds across bookmakers for each fight to get consensus line.

        Returns one row per fight with averaged probabilities.
        """
        if odds_df.empty:
            return odds_df

        prepared = odds_df.copy()
        prepared["_pair_key"] = prepared.apply(
            lambda row: "|".join(
                sorted(
                    [
                        normalize_cross_source_name(row.get("fighter_a")),
                        normalize_cross_source_name(row.get("fighter_b")),
                    ]
                )
            ),
            axis=1,
        )
        commence_dates = pd.to_datetime(
            prepared["commence_time"],
            errors="coerce",
            utc=True,
        )
        prepared["_commence_day"] = commence_dates.dt.strftime("%Y-%m-%d").fillna(
            prepared["commence_time"].astype(str)
        )

        grouped = prepared.groupby(["_pair_key", "_commence_day", "fighter_a", "fighter_b"]).agg(
            event_id=("event_id", "first"),
            commence_time=("commence_time", "first"),
            a_odds_avg=("a_odds", "mean"),
            b_odds_avg=("b_odds", "mean"),
            a_fair_prob_avg=("a_fair_prob", "mean"),
            b_fair_prob_avg=("b_fair_prob", "mean"),
            num_bookmakers=("bookmaker", "count"),
        ).reset_index()

        return grouped[
            [
                "event_id",
                "commence_time",
                "fighter_a",
                "fighter_b",
                "a_odds_avg",
                "b_odds_avg",
                "a_fair_prob_avg",
                "b_fair_prob_avg",
                "num_bookmakers",
            ]
        ]


def fetch_and_save_live_odds() -> pd.DataFrame:
    """Convenience function: fetch live odds and save to CSV."""
    client = OddsClient()
    odds = client.get_live_odds()
    df = client.odds_to_dataframe(odds)
    consensus = client.get_consensus_odds(df)

    if not consensus.empty:
        path = RAW_DATA_DIR / "live_odds.csv"
        consensus.to_csv(path, index=False)
        logger.info(f"Saved {len(consensus)} fights with odds to {path}")

    return consensus
