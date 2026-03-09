"""
UFC market discovery on Polymarket — finds active UFC fight markets
and maps them to fighters.
"""

import json
import logging
import re
from typing import Optional

import pandas as pd
import requests

from src.config import POLYMARKET_GAMMA_URL

logger = logging.getLogger(__name__)


def find_ufc_events(limit: int = 200) -> list[dict]:
    """
    Find all active UFC fight events on Polymarket.

    Uses tag_slug=ufc to query the Gamma API directly, which is more
    reliable than the generic tag search.
    """
    all_events = []
    seen_ids = set()
    offset = 0

    while offset < 500:
        try:
            resp = requests.get(
                f"{POLYMARKET_GAMMA_URL}/events",
                params={
                    "tag_slug": "ufc",
                    "limit": limit,
                    "offset": offset,
                    "closed": False,
                    "active": True,
                },
                timeout=30,
            )
            resp.raise_for_status()
            events = resp.json()
            if not events:
                break

            for event in events:
                eid = event.get("id", "")
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    all_events.append(event)

            if len(events) < limit:
                break
            offset += limit
        except Exception as e:
            logger.warning(f"Failed to fetch UFC events (offset={offset}): {e}")
            break

    logger.info(f"Found {len(all_events)} UFC events on Polymarket")
    return all_events


def _parse_json_field(value) -> list:
    """Parse a field that may be a JSON string or already a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def parse_fight_market(market: dict, event: Optional[dict] = None) -> Optional[dict]:
    """
    Parse a Polymarket market dict into a structured fight market.

    Only returns winner markets (where outcomes are fighter names).
    Skips prop markets (KO, submission, rounds, distance, etc.).
    """
    question = market.get("question", "") or market.get("title", "")
    condition_id = market.get("conditionId", "") or market.get("condition_id", "")

    # Parse outcomes — may be a JSON string like '["Fighter A", "Fighter B"]'
    outcomes = _parse_json_field(market.get("outcomes", []))

    # Only keep winner markets: exactly 2 outcomes that are fighter names
    if len(outcomes) != 2:
        return None

    # Skip prop markets (Yes/No, Over/Under)
    prop_values = {"Yes", "No", "Over", "Under"}
    if outcomes[0] in prop_values or outcomes[1] in prop_values:
        return None

    fighter_a = outcomes[0].strip()
    fighter_b = outcomes[1].strip()

    if not fighter_a or not fighter_b:
        return None

    # Parse token IDs — may be a JSON string
    tokens = _parse_json_field(market.get("clobTokenIds", []))
    token_a = tokens[0] if len(tokens) > 0 else ""
    token_b = tokens[1] if len(tokens) > 1 else ""

    # Parse prices — may be a JSON string like '["0.665", "0.335"]'
    prices = _parse_json_field(market.get("outcomePrices", []))
    price_a = float(prices[0]) if len(prices) > 0 else None
    price_b = float(prices[1]) if len(prices) > 1 else None

    return {
        "market_id": market.get("id", ""),
        "condition_id": condition_id,
        "question": question,
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "token_id_yes": token_a,
        "token_id_no": token_b,
        "price_yes": price_a,
        "price_no": price_b,
        "best_bid": market.get("bestBid"),
        "best_ask": market.get("bestAsk"),
        "volume": market.get("volume", 0),
        "liquidity": market.get("liquidityNum", 0) or market.get("liquidity", 0),
        "end_date": market.get("endDate", ""),
        "active": market.get("active", True),
        "closed": market.get("closed", False),
        "neg_risk": (event or {}).get("negRisk", False),
        "tick_size": market.get("minimum_tick_size", "0.01"),
    }


def get_ufc_fight_markets() -> pd.DataFrame:
    """
    Get all active UFC fight winner markets as a DataFrame.

    Returns DataFrame with one row per fight market including:
        fighter names, token IDs, current prices, volume, liquidity.
    """
    events = find_ufc_events()

    markets = []
    for event in events:
        event_markets = event.get("markets", [])
        if not event_markets:
            continue
        for market in event_markets:
            if market.get("closed", False):
                continue
            parsed = parse_fight_market(market, event=event)
            if parsed and parsed["fighter_a"]:
                parsed["event_title"] = event.get("title", "")
                parsed["event_date"] = event.get("eventDate", "") or event.get("endDate", "")
                markets.append(parsed)

    df = pd.DataFrame(markets)
    if not df.empty:
        df = df[df["active"] & ~df["closed"]].reset_index(drop=True)
        logger.info(f"Found {len(df)} active UFC fight markets")
    else:
        logger.info("No active UFC fight markets found")

    return df
