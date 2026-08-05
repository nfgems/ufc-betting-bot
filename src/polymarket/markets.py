"""
UFC market discovery on Polymarket — finds active UFC fight markets
and maps them to fighters.
"""

import json
import logging
import re
import time
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from src.config import POLYMARKET_GAMMA_URL

logger = logging.getLogger(__name__)
_LIVE_MARKET_START_BUFFER = timedelta(hours=1)
_CURRENT_CARD_MARKET_GRACE = timedelta(hours=12)


def _parse_market_start_time(*values) -> Optional[datetime]:
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    return None


def _market_is_tradeable(start_time) -> bool:
    parsed = _parse_market_start_time(start_time)
    if parsed is None:
        return False
    return datetime.now(timezone.utc) < (parsed - _LIVE_MARKET_START_BUFFER)


def _fight_pair_key(fighter_a: object, fighter_b: object) -> str:
    """Return the shared cross-source identity key for an unordered matchup."""
    from src.data.name_utils import normalize_cross_source_name

    fighters = sorted(
        [
            normalize_cross_source_name(fighter_a),
            normalize_cross_source_name(fighter_b),
        ]
    )
    if not all(fighters):
        return ""
    return "|".join(fighters)


def _index_bout_contexts(bout_contexts: Optional[Iterable[dict]]) -> dict[str, dict]:
    """Index validated UFC card rows by canonical fighter pair.

    The caller is responsible for limiting these rows to confirmed UFC card
    membership.  A later row for the same pair intentionally wins so callers
    can overlay a bookmaker's bout-specific commence time on a card row.
    """
    indexed: dict[str, dict] = {}
    for context in bout_contexts or []:
        if not isinstance(context, dict):
            continue
        for fighter_a_key, fighter_b_key in (
            ("fighter_a", "fighter_b"),
            ("odds_fighter_a", "odds_fighter_b"),
        ):
            pair_key = _fight_pair_key(
                context.get(fighter_a_key),
                context.get(fighter_b_key),
            )
            if pair_key:
                indexed[pair_key] = context
    return indexed


def _current_card_fallback_is_open(
    card_start_time,
    *,
    accepting_orders: object,
) -> bool:
    """Keep a confirmed current-card market briefly when only card time exists.

    Gamma commonly assigns every bout the card's opening time.  Requiring an
    exact card-context match, an explicitly accepting orderbook, and a bounded
    grace period prevents this fallback from reviving unrelated stale markets.
    The prediction/executor path still applies the real bout-time safety gate.
    """
    parsed = _parse_market_start_time(card_start_time)
    if parsed is None or accepting_orders is not True:
        return False
    now = datetime.now(timezone.utc)
    return (
        parsed - _LIVE_MARKET_START_BUFFER
        <= now
        < parsed + _CURRENT_CARD_MARKET_GRACE
    )


def _bout_time_matches_market_card(bout_start_time, market_card_time) -> bool:
    """Reject a context override whose date is unrelated to Gamma's card."""
    bout_time = _parse_market_start_time(bout_start_time)
    card_time = _parse_market_start_time(market_card_time)
    if bout_time is None or card_time is None:
        return False
    return abs((bout_time.date() - card_time.date()).days) <= 1


def _resolve_market_start_time(market: dict, event: Optional[dict] = None) -> str:
    """Pick the actual fight start field, not the market listing timestamp."""
    source_event = event or {}
    return (
        market.get("gameStartTime", "")
        or market.get("startTime", "")
        or source_event.get("startTime", "")
        or source_event.get("eventDate", "")
        or market.get("eventDate", "")
        or ""
    )


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
        events = None
        for attempt in range(1, 4):
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
                break
            except Exception as e:
                if attempt == 3:
                    logger.warning(
                        "Failed to fetch UFC events after 3 attempts (offset=%s): %s",
                        offset,
                        e,
                    )
                    break

                retry_delay = min(2 ** (attempt - 1), 4)
                logger.info(
                    "Failed to fetch UFC events (offset=%s, attempt %s/3): %s; "
                    "retrying in %ss",
                    offset,
                    attempt,
                    e,
                    retry_delay,
                )
                time.sleep(retry_delay)

        if events is None:
            break
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


def _parse_mapping_field(value) -> dict:
    """Parse a dict-like Gamma field that may arrive JSON-encoded."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _first_present(mapping: dict, *keys):
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _parse_probability(value) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= parsed <= 1.0:
        return parsed
    return None


def _midpoint_from_book(best_bid, best_ask) -> Optional[float]:
    bid = _parse_probability(best_bid)
    ask = _parse_probability(best_ask)
    if bid is None or ask is None or bid > ask:
        return None
    return (bid + ask) / 2.0


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
    price_a = _parse_probability(prices[0]) if len(prices) > 0 else None
    price_b = _parse_probability(prices[1]) if len(prices) > 1 else None
    if price_a is not None and price_b is None:
        price_b = 1.0 - price_a
    elif price_a is None and price_b is not None:
        price_a = 1.0 - price_b
    elif price_a is None and price_b is None:
        midpoint = _midpoint_from_book(market.get("bestBid"), market.get("bestAsk"))
        if midpoint is not None:
            price_a = midpoint
            price_b = 1.0 - midpoint
    fee_schedule = _parse_mapping_field(market.get("feeSchedule"))
    fee_details = _parse_mapping_field(
        market.get("feeDetails") or market.get("fee_details")
    )

    return {
        "market_id": market.get("id", ""),
        "condition_id": condition_id,
        "event_id": (event or {}).get("id", ""),
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
        "event_date": _resolve_market_start_time(market, event=event),
        "active": market.get("active", True),
        "closed": market.get("closed", False),
        "accepting_orders": market.get("acceptingOrders"),
        "neg_risk": (event or {}).get("negRisk", market.get("negRisk")),
        "tick_size": (
            market.get("orderPriceMinTickSize")
            or market.get("minimum_tick_size")
        ),
        "fee_schedule": fee_schedule,
        "fees_enabled": market.get("feesEnabled"),
        "fee_rate": _first_present(
            fee_details,
            "r",
            "rate",
            "fee_rate",
            "feeRate",
        )
        or _first_present(
            fee_schedule,
            "taker_fee_rate",
            "takerFeeRate",
            "taker",
            "fee_rate",
            "feeRate",
        ),
        "fee_exponent": _first_present(
            fee_details,
            "e",
            "exponent",
            "fee_exponent",
            "feeExponent",
        ),
        "fee_source": "gamma" if fee_schedule or fee_details else "",
    }


def get_ufc_fight_markets(
    *,
    bout_contexts: Optional[Iterable[dict]] = None,
) -> pd.DataFrame:
    """
    Get all active UFC fight winner markets as a DataFrame.

    ``bout_contexts`` is an optional, backward-compatible list of confirmed
    UFC card rows.  When a matching row contains a bout-specific commence
    time, that timestamp replaces Polymarket's often-generic card start before
    applying the pre-fight safety buffer.

    Returns DataFrame with one row per fight market including:
        fighter names, token IDs, current prices, volume, liquidity.
    """
    events = find_ufc_events()
    bout_context_by_pair = _index_bout_contexts(bout_contexts)

    markets = []
    skipped_untradeable = 0
    overridden_start_times = 0
    current_card_fallbacks = 0
    for event in events:
        event_markets = event.get("markets", [])
        if not event_markets:
            continue
        for market in event_markets:
            if market.get("closed", False):
                continue
            if market.get("acceptingOrders") is False:
                continue
            parsed = parse_fight_market(market, event=event)
            if not parsed or not parsed["fighter_a"]:
                continue
            parsed["event_title"] = event.get("title", "")
            polymarket_event_date = parsed.get("event_date")
            pair_key = _fight_pair_key(parsed.get("fighter_a"), parsed.get("fighter_b"))
            bout_context = bout_context_by_pair.get(pair_key)
            bout_start_time = None
            if bout_context is not None:
                # A card-level ``event_date`` is membership evidence, not a
                # bout clock. Treating an ISO date as midnight would close all
                # later fights before the card begins. Only the bookmaker's
                # bout-specific commence time is an execution timestamp.
                bout_start_time = bout_context.get("commence_time")
                if (
                    market.get("acceptingOrders") is True
                    and _bout_time_matches_market_card(
                        bout_start_time,
                        polymarket_event_date,
                    )
                ):
                    parsed["polymarket_event_date"] = polymarket_event_date
                    parsed["event_date"] = bout_start_time
                    parsed["event_time_source"] = "bout_context"
                    overridden_start_times += 1

            if _market_is_tradeable(parsed.get("event_date")):
                markets.append(parsed)
                continue

            can_use_current_card_fallback = (
                bout_context is not None
                and _parse_market_start_time(bout_start_time) is None
                and _current_card_fallback_is_open(
                    polymarket_event_date,
                    accepting_orders=market.get("acceptingOrders"),
                )
            )
            if can_use_current_card_fallback:
                parsed["event_time_source"] = "current_card_fallback"
                current_card_fallbacks += 1
                markets.append(parsed)
                continue

            skipped_untradeable += 1
            continue

    df = pd.DataFrame(markets)
    if not df.empty:
        df = df[df["active"] & ~df["closed"]].reset_index(drop=True)
        logger.info(
            "Found %s active UFC fight markets (%s skipped outside the pre-fight window; "
            "%s bout-time overrides; %s current-card fallbacks)",
            len(df),
            skipped_untradeable,
            overridden_start_times,
            current_card_fallbacks,
        )
    else:
        logger.info("No active UFC fight markets found")

    return df
