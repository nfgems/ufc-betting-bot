"""Live tennis odds discovery across active ATP and WTA tournament keys."""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from src.config import TENNIS_ODDS_SPORTS_PREFIX
from src.data.odds_client import OddsClient
from src.data.tennis_data import derive_tournament_seed

logger = logging.getLogger(__name__)

GENERIC_TOURNAMENT_LABELS = {
    "",
    "atp",
    "tennis",
    "singles",
    "men singles",
    "mens singles",
    "wta",
    "women singles",
    "womens singles",
}


def _normalize_tournament_label(value: object) -> str:
    text = str(value or "").strip()
    lowered = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return lowered


def _is_generic_tournament_label(value: object) -> bool:
    return _normalize_tournament_label(value) in GENERIC_TOURNAMENT_LABELS


def _display_tournament_name_from_seed(seed: str) -> str:
    words = []
    for token in str(seed or "").split():
        if token == "us":
            words.append("US")
        else:
            words.append(token.capitalize())
    return " ".join(words).strip()


def infer_tournament_name(
    *,
    sport_key: str,
    title: object = None,
    description: object = None,
) -> str:
    """Infer a tournament name from sport metadata without collapsing to generic labels."""
    candidates = []

    for raw_value in [title, description]:
        text = str(raw_value or "").strip()
        if not text:
            continue
        cleaned = re.sub(r"^(?:atp|wta)\s+", "", text, flags=re.IGNORECASE).strip()
        if not cleaned or _is_generic_tournament_label(cleaned):
            continue
        seed = derive_tournament_seed(cleaned)
        if not seed:
            continue
        candidates.append((seed.count(" ") + 1, len(seed), seed))

    sport_key_seed = derive_tournament_seed(sport_key)
    if sport_key_seed:
        candidates.append((sport_key_seed.count(" ") + 1, len(sport_key_seed), sport_key_seed))

    if candidates:
        best_seed = max(candidates)[2]
        display = _display_tournament_name_from_seed(best_seed)
        if display:
            return display

    return str(title or description or sport_key or "").strip()


def parse_tour_from_sport_key(sport_key: str) -> str:
    """Map an Odds API tennis sport key to ATP or WTA."""
    sport_key = str(sport_key or "").lower()
    if sport_key.startswith("tennis_atp_"):
        return "atp"
    if sport_key.startswith("tennis_wta_"):
        return "wta"
    return ""


def filter_active_tennis_sports(
    sports: list[dict],
    sport_prefixes: Optional[list[str]] = None,
) -> list[dict]:
    """Keep active tournament-specific ATP/WTA sport keys."""
    prefixes = tuple((sport_prefixes or TENNIS_ODDS_SPORTS_PREFIX))
    filtered: list[dict] = []

    for sport in sports:
        sport_key = str(sport.get("key", ""))
        if not sport.get("active", False):
            continue
        if not sport_key.startswith(prefixes):
            continue

        tournament_name = infer_tournament_name(
            sport_key=sport_key,
            title=sport.get("title"),
            description=sport.get("description"),
        )
        filtered.append(
            {
                **sport,
                "tour": parse_tour_from_sport_key(sport_key),
                "sport_key": sport_key,
                "tournament_name": tournament_name,
                "tournament_seed": derive_tournament_seed(tournament_name or sport_key),
            }
        )

    return filtered


def discover_active_tennis_sports(client: Optional[OddsClient] = None) -> list[dict]:
    """Discover active ATP and WTA tournament sport keys via /v4/sports."""
    client = client or OddsClient()
    sports = client.list_sports()
    active = filter_active_tennis_sports(sports)
    logger.info("Discovered %s active tennis tournament sport keys", len(active))
    return active


def fetch_live_tennis_odds(
    client: Optional[OddsClient] = None,
    sports: Optional[list[dict]] = None,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "decimal",
) -> pd.DataFrame:
    """Fetch flat live tennis odds rows across active tournament keys."""
    client = client or OddsClient()
    sports = sports or discover_active_tennis_sports(client)
    flat_frames: list[pd.DataFrame] = []

    for sport in sports:
        sport_key = sport["sport_key"]
        try:
            raw_odds = client.get_live_odds(
                regions=regions,
                markets=markets,
                odds_format=odds_format,
                sport_key=sport_key,
            )
        except Exception as exc:
            logger.warning("Failed to fetch %s odds: %s", sport_key, exc)
            continue

        odds_df = client.odds_to_dataframe(raw_odds)
        if odds_df.empty:
            continue

        metadata = pd.DataFrame(
            [
                {
                    "event_id": event.get("id", ""),
                    "sport_key": sport_key,
                    "tour": sport.get("tour", ""),
                    "tournament_name": sport.get("tournament_name", ""),
                    "tournament_seed": sport.get("tournament_seed", ""),
                    "event_title": event.get("sport_title") or sport.get("title") or sport_key,
                }
                for event in raw_odds
            ]
        )
        odds_df = odds_df.merge(metadata, on="event_id", how="left")
        flat_frames.append(odds_df)

    if not flat_frames:
        return pd.DataFrame(
            columns=[
                "event_id",
                "commence_time",
                "fighter_a",
                "fighter_b",
                "bookmaker",
                "a_odds",
                "b_odds",
                "a_implied_prob",
                "b_implied_prob",
                "a_fair_prob",
                "b_fair_prob",
                "sport_key",
                "tour",
                "tournament_name",
                "tournament_seed",
                "event_title",
            ]
        )

    combined = pd.concat(flat_frames, ignore_index=True)
    logger.info("Fetched %s bookmaker rows across %s tennis sport keys", len(combined), len(sports))
    return combined


def fetch_live_tennis_consensus(
    client: Optional[OddsClient] = None,
    sports: Optional[list[dict]] = None,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "decimal",
) -> pd.DataFrame:
    """Merge live tennis odds across active sport keys into one consensus frame."""
    odds_df = fetch_live_tennis_odds(
        client=client,
        sports=sports,
        regions=regions,
        markets=markets,
        odds_format=odds_format,
    )
    if odds_df.empty:
        return odds_df

    # First pass: aggregate bookmaker rows per event/sport_key combination.
    per_event = (
        odds_df.groupby(
            [
                "event_id",
                "fighter_a",
                "fighter_b",
                "sport_key",
                "tour",
                "tournament_name",
                "tournament_seed",
                "event_title",
            ],
            dropna=False,
        )
        .agg(
            commence_time=("commence_time", "first"),
            a_odds_avg=("a_odds", "mean"),
            b_odds_avg=("b_odds", "mean"),
            a_fair_prob_avg=("a_fair_prob", "mean"),
            b_fair_prob_avg=("b_fair_prob", "mean"),
            num_bookmakers=("bookmaker", "count"),
        )
        .reset_index()
    )

    # Second pass: the same physical match can appear under multiple Odds API
    # sport keys (e.g. a tournament-specific key AND a generic tour key), each
    # with a different event_id.  Deduplicate by (fighter_a, fighter_b) and keep
    # the row with the most bookmakers so we get the richest consensus.
    before_dedup = len(per_event)
    per_event = per_event.sort_values("num_bookmakers", ascending=False)
    consensus = per_event.drop_duplicates(
        subset=["fighter_a", "fighter_b"], keep="first",
    ).sort_values(
        ["commence_time", "tour", "fighter_a", "fighter_b"],
    ).reset_index(drop=True)
    dropped = before_dedup - len(consensus)
    if dropped > 0:
        logger.info(
            "Deduplicated %s cross-sport-key duplicate matchups from tennis consensus",
            dropped,
        )
    logger.info("Built tennis consensus odds for %s matches", len(consensus))
    return consensus
