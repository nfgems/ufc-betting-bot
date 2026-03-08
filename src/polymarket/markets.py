"""
UFC market discovery on Polymarket — finds active UFC fight markets
and maps them to fighters.
"""

import logging
import re
from typing import Optional

import pandas as pd

from src.polymarket.client import GammaClient

logger = logging.getLogger(__name__)


def find_ufc_events(limit: int = 100) -> list[dict]:
    """
    Find all active UFC events on Polymarket.

    Returns list of event dicts with associated markets.
    """
    gamma = GammaClient()

    # Search for UFC events using multiple strategies
    all_events = []
    seen_ids = set()

    # Strategy 1: Search by UFC tag
    for tag in ["UFC", "ufc", "MMA", "mma"]:
        try:
            events = gamma.search_events(tag, limit=limit)
            for event in events:
                eid = event.get("id", "")
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    all_events.append(event)
        except Exception as e:
            logger.debug(f"Tag search '{tag}' failed: {e}")

    # Strategy 2: Browse all active events and filter
    offset = 0
    while True:
        try:
            events = gamma.get_events(limit=100, offset=offset, closed=False)
            if not events:
                break
            for event in events:
                title = (event.get("title", "") or "").lower()
                description = (event.get("description", "") or "").lower()
                eid = event.get("id", "")
                if eid in seen_ids:
                    continue
                if any(kw in title or kw in description for kw in ["ufc", "mma", "fight night", "ppv"]):
                    seen_ids.add(eid)
                    all_events.append(event)
            offset += 100
            if offset > 500:  # Safety limit
                break
        except Exception:
            break

    logger.info(f"Found {len(all_events)} UFC events on Polymarket")
    return all_events


def parse_fight_market(market: dict) -> Optional[dict]:
    """
    Parse a Polymarket market dict into a structured fight market.

    Returns dict with:
        - market_id, condition_id, question
        - fighter_a, fighter_b (extracted from market question)
        - token_id_yes, token_id_no
        - current price (implied probability)
    """
    question = market.get("question", "") or market.get("title", "")
    condition_id = market.get("conditionId", "") or market.get("condition_id", "")

    # Extract fighter names from common question formats:
    # "Will Fighter A beat Fighter B?"
    # "Fighter A vs Fighter B"
    # "Fighter A to win?"
    fighters = _extract_fighters(question)
    if not fighters:
        return None

    # Get token IDs for YES/NO outcomes
    tokens = market.get("clobTokenIds", []) or []
    outcomes = market.get("outcomes", []) or market.get("outcomePrices", [])

    token_yes = tokens[0] if len(tokens) > 0 else ""
    token_no = tokens[1] if len(tokens) > 1 else ""

    # Current prices
    best_bid = market.get("bestBid")
    best_ask = market.get("bestAsk")
    outcome_prices = market.get("outcomePrices", "")

    # Parse outcome prices if available (format: "[0.65, 0.35]" or list)
    price_yes = None
    price_no = None
    if isinstance(outcome_prices, str) and outcome_prices:
        try:
            prices = eval(outcome_prices)  # Safe for simple list literals
            if len(prices) >= 2:
                price_yes = float(prices[0])
                price_no = float(prices[1])
        except Exception:
            pass
    elif isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
        price_yes = float(outcome_prices[0])
        price_no = float(outcome_prices[1])

    return {
        "market_id": market.get("id", ""),
        "condition_id": condition_id,
        "question": question,
        "fighter_a": fighters[0],
        "fighter_b": fighters[1] if len(fighters) > 1 else "",
        "token_id_yes": token_yes,
        "token_id_no": token_no,
        "price_yes": price_yes,
        "price_no": price_no,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "volume": market.get("volume", 0),
        "liquidity": market.get("liquidityNum", 0) or market.get("liquidity", 0),
        "end_date": market.get("endDate", ""),
        "active": market.get("active", True),
        "closed": market.get("closed", False),
        "neg_risk": market.get("negRisk", False),
        "tick_size": market.get("minimum_tick_size", "0.01"),
    }


def _extract_fighters(question: str) -> list[str]:
    """
    Extract fighter names from a market question.

    Handles formats:
        "Will Fighter A beat Fighter B?"
        "Fighter A vs Fighter B"
        "Fighter A vs. Fighter B: Who will win?"
        "Fighter A to beat Fighter B"
    """
    q = question.strip()

    # Pattern: "X vs Y" or "X vs. Y"
    match = re.search(r"(.+?)\s+vs\.?\s+(.+?)(?:\s*[\?:,]|$)", q, re.IGNORECASE)
    if match:
        a = _clean_fighter_name(match.group(1))
        b = _clean_fighter_name(match.group(2))
        if a and b:
            return [a, b]

    # Pattern: "Will X beat Y"
    match = re.search(r"[Ww]ill\s+(.+?)\s+beat\s+(.+?)[\?]?$", q)
    if match:
        a = _clean_fighter_name(match.group(1))
        b = _clean_fighter_name(match.group(2))
        if a and b:
            return [a, b]

    # Pattern: "X to beat Y" or "X to win against Y"
    match = re.search(r"(.+?)\s+to\s+(?:beat|win|defeat)\s+(?:against\s+)?(.+?)[\?]?$", q, re.IGNORECASE)
    if match:
        a = _clean_fighter_name(match.group(1))
        b = _clean_fighter_name(match.group(2))
        if a and b:
            return [a, b]

    return []


def _clean_fighter_name(name: str) -> str:
    """Clean up a fighter name extracted from a question."""
    name = name.strip()
    # Remove common prefixes
    for prefix in ["will ", "can ", "does "]:
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
    # Remove trailing punctuation
    name = re.sub(r"[?!.,;:]+$", "", name).strip()
    # Remove "who will win" suffix
    name = re.sub(r"\s*who will win.*$", "", name, flags=re.IGNORECASE).strip()
    return name


def get_ufc_fight_markets() -> pd.DataFrame:
    """
    Get all active UFC fight markets as a DataFrame.

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
            parsed = parse_fight_market(market)
            if parsed and parsed["fighter_a"]:
                parsed["event_title"] = event.get("title", "")
                parsed["event_date"] = event.get("endDate", "")
                markets.append(parsed)

    df = pd.DataFrame(markets)
    if not df.empty:
        df = df[df["active"] & ~df["closed"]].reset_index(drop=True)
        logger.info(f"Found {len(df)} active UFC fight markets")
    else:
        logger.info("No active UFC fight markets found")

    return df
