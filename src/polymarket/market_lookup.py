"""Shared Polymarket token metadata lookups for supported winner markets."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from src.polymarket.markets import get_ufc_fight_markets
from src.polymarket.tennis_markets import discover_tennis_markets

logger = logging.getLogger(__name__)


def _market_rows(markets: object) -> list[dict]:
    if markets is None:
        return []
    if isinstance(markets, pd.DataFrame):
        if markets.empty:
            return []
        return list(markets.to_dict("records"))
    if isinstance(markets, Iterable) and not isinstance(markets, (str, bytes, dict)):
        return [row for row in markets if isinstance(row, dict)]
    if isinstance(markets, dict):
        return [markets]
    return []


def _token_entry(row: dict, *, token_field: str, side: str, fighter: str, opponent: str) -> dict | None:
    token_id = str(row.get(token_field, "") or "").strip()
    fighter_name = str(row.get(fighter, "") or "").strip()
    opponent_name = str(row.get(opponent, "") or "").strip()
    if not token_id or not fighter_name or not opponent_name:
        return None
    return {
        "token_id": token_id,
        "side": side,
        "fighter": fighter_name,
        "opponent": opponent_name,
        "market_id": str(row.get("market_id", "") or "").strip(),
        "condition_id": str(row.get("condition_id", "") or "").strip(),
        "question": str(row.get("question", "") or "").strip(),
        "event_date": str(row.get("event_date", "") or "").strip(),
        "sport": str(row.get("sport", "") or "").strip(),
    }


def build_market_token_lookup(markets: object) -> dict[str, dict]:
    """Index winner-market tokens to canonical side/fighter orientation."""
    lookup: dict[str, dict] = {}
    ambiguous_tokens: set[str] = set()

    for row in _market_rows(markets):
        entries = [
            _token_entry(row, token_field="token_id_yes", side="a", fighter="fighter_a", opponent="fighter_b"),
            _token_entry(row, token_field="token_id_no", side="b", fighter="fighter_b", opponent="fighter_a"),
        ]
        for entry in entries:
            if entry is None:
                continue
            token_id = entry["token_id"]
            if token_id in ambiguous_tokens:
                continue
            existing = lookup.get(token_id)
            if existing is None:
                lookup[token_id] = entry
                continue
            comparable_existing = {
                key: existing.get(key, "")
                for key in ("side", "fighter", "opponent", "market_id", "condition_id")
            }
            comparable_entry = {
                key: entry.get(key, "")
                for key in ("side", "fighter", "opponent", "market_id", "condition_id")
            }
            if comparable_existing != comparable_entry:
                lookup.pop(token_id, None)
                ambiguous_tokens.add(token_id)

    if ambiguous_tokens:
        logger.warning(
            "Skipped %d ambiguous Polymarket token mappings while building the market lookup",
            len(ambiguous_tokens),
        )
    return lookup


def load_supported_market_token_lookup() -> dict[str, dict]:
    """Load token metadata for the supported live Polymarket sports."""
    frames: list[pd.DataFrame] = []

    try:
        ufc_markets = get_ufc_fight_markets()
        if not ufc_markets.empty:
            frames.append(ufc_markets)
    except Exception as exc:
        logger.warning("Failed to load UFC Polymarket markets for reconciliation lookup: %s", exc)

    try:
        tennis_markets = discover_tennis_markets()
        if not tennis_markets.empty:
            frames.append(tennis_markets)
    except Exception as exc:
        logger.warning("Failed to load tennis Polymarket markets for reconciliation lookup: %s", exc)

    if not frames:
        return {}
    return build_market_token_lookup(pd.concat(frames, ignore_index=True, sort=False))
