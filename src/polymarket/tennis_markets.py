"""Polymarket tennis market discovery and matching helpers."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import pandas as pd
import requests

from src.config import POLYMARKET_GAMMA_URL
from src.data.tennis_data import (
    collect_tournament_seed_candidates,
    derive_tournament_seed,
    normalize_name_tokens,
    normalize_player_name,
    tournament_seeds_compatible,
)
from src.data.tennis_odds import discover_active_tennis_sports

logger = logging.getLogger(__name__)

PROP_MARKET_PATTERN = re.compile(
    r"\b(?:over|under|set|sets|game|games|aces|double faults|correct score|exact score|handicap|spread|total)\b",
    re.IGNORECASE,
)
MATCHUP_PATTERN = re.compile(r"(?P<a>.+?)\s+(?:vs\.?|v\.?)\s+(?P<b>.+)$", re.IGNORECASE)
GENERIC_PROP_OUTCOMES = {"over", "under", "yes", "no"}
WTA_TOUR_PATTERN = re.compile(r"\b(?:wta|women(?:'s|’s|s)?)\b|\(w\)", re.IGNORECASE)
ATP_TOUR_PATTERN = re.compile(r"\b(?:atp|men(?:'s|’s|s)?)\b|\(m\)", re.IGNORECASE)


def _parse_json_field(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def build_match_key(fighter_a: object, fighter_b: object) -> str:
    """Build an unordered normalized key for a two-player matchup."""
    names = sorted([normalize_player_name(fighter_a), normalize_player_name(fighter_b)])
    return "|".join(names)


def _player_variants(player_name: str) -> set[str]:
    tokens = normalize_name_tokens(player_name)
    if not tokens:
        return set()

    last = tokens[-1]
    variants = {
        " ".join(tokens),
        last,
    }
    if len(tokens) > 1:
        variants.add(f"{tokens[0]} {last}")
        variants.add(f"{tokens[0][0]} {last}")
    return {variant.strip() for variant in variants if variant.strip()}


def _preferred_player_label(player_name: str, outcome_name: object) -> str:
    """Keep the most informative safe label seen for a player."""
    outcome_text = str(outcome_name or "").strip()
    if not outcome_text:
        return player_name

    player_tokens = normalize_name_tokens(player_name)
    outcome_tokens = normalize_name_tokens(outcome_text)
    if len(outcome_tokens) > len(player_tokens):
        return outcome_text
    if len(" ".join(outcome_tokens)) > len(" ".join(player_tokens)):
        return outcome_text
    return player_name


def _match_outcome_to_player(outcome: object, players: tuple[str, str]) -> Optional[str]:
    outcome_norm = normalize_player_name(outcome)
    outcome_tokens = set(normalize_name_tokens(outcome))
    if not outcome_norm:
        return None

    matches: list[tuple[int, str]] = []
    for player in players:
        player_norm = normalize_player_name(player)
        player_token_list = list(normalize_name_tokens(player))
        player_tokens = set(player_token_list)
        variants = _player_variants(player)
        surname = player_token_list[-1] if player_token_list else ""

        score = -1
        if outcome_norm == player_norm:
            score = 100
        elif outcome_norm in variants:
            score = 90
        elif outcome_tokens and player_tokens and player_tokens.issubset(outcome_tokens):
            score = 80 + len(player_tokens)
        elif (
            surname
            and outcome_tokens
            and player_tokens
            and outcome_tokens.issubset(player_tokens)
            and surname in outcome_tokens
        ):
            score = 70 + len(outcome_tokens)

        if score >= 0:
            matches.append((score, player))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    best_score = matches[0][0]
    winners = [player for score, player in matches if score == best_score]
    if len(winners) != 1:
        return None
    return winners[0]


def _outcomes_look_like_generic_prop_market(outcomes: list[object]) -> bool:
    normalized = [str(outcome or "").strip().lower() for outcome in outcomes if str(outcome or "").strip()]
    return bool(normalized) and all(outcome in GENERIC_PROP_OUTCOMES for outcome in normalized)


def normalize_tennis_market_names(
    players: tuple[str, str],
    outcomes: list[str],
    event_title: str,
) -> Optional[tuple[str, str]]:
    """Map market outcomes back to the safest available player labels."""
    if len(outcomes) != 2:
        return None
    if _outcomes_look_like_generic_prop_market(outcomes):
        return None

    matched = []
    for outcome in outcomes:
        player = _match_outcome_to_player(outcome, players)
        if player is None:
            logger.warning(
                "Could not map tennis market outcome '%s' to event '%s'",
                outcome,
                event_title,
            )
            return None
        matched.append(_preferred_player_label(player, outcome))

    if matched[0] == matched[1]:
        logger.warning(
            "Ambiguous tennis market outcomes %s for event '%s'",
            outcomes,
            event_title,
        )
        return None
    return matched[0], matched[1]


def extract_players_from_title(title: str) -> Optional[tuple[str, str]]:
    """Extract player labels from a tennis event title."""
    title = str(title or "").strip()
    if not title:
        return None

    candidate = title.split(":")[-1].strip()
    match = MATCHUP_PATTERN.search(candidate)
    if not match:
        match = MATCHUP_PATTERN.search(title)
    if not match:
        return None

    player_a = match.group("a").strip(" -")
    player_b = match.group("b").strip(" -")
    if not player_a or not player_b:
        return None
    return player_a, player_b


def _event_matches_seed(event: dict, tournament_seed: str) -> bool:
    if not tournament_seed:
        return True

    event_fields = [
        str(event.get(field, ""))
        for field in ["title", "slug", "category", "subcategory"]
        if str(event.get(field, "")).strip()
    ]
    return any(tournament_seeds_compatible(field, tournament_seed) for field in event_fields)


def _infer_tour_from_event(event: dict, market: Optional[dict] = None) -> str:
    candidate_fields = [
        event.get("seriesSlug"),
        event.get("series_slug"),
        event.get("slug"),
        event.get("title"),
        event.get("category"),
        event.get("subcategory"),
    ]
    if market is not None:
        candidate_fields.extend(
            [
                market.get("slug"),
                market.get("question"),
                market.get("title"),
            ]
        )

    haystack = " | ".join(str(value) for value in candidate_fields if str(value or "").strip())
    if not haystack:
        return ""
    if WTA_TOUR_PATTERN.search(haystack):
        return "wta"
    if ATP_TOUR_PATTERN.search(haystack):
        return "atp"
    return ""


def _matchup_seed_candidates(matchup: dict) -> list[str]:
    return collect_tournament_seed_candidates(
        matchup.get("tournament_seed"),
        matchup.get("tournament_name"),
        matchup.get("sport_key"),
        matchup.get("event_title"),
        matchup.get("market_event_title"),
    )


def _match_market_players_to_matchup(
    market_players: tuple[str, str],
    matchup_players: tuple[str, str],
) -> Optional[tuple[str, str]]:
    resolved = []
    for player_ref in market_players:
        player = _match_outcome_to_player(player_ref, matchup_players)
        if player is None:
            return None
        resolved.append(player)

    if resolved[0] == resolved[1]:
        return None
    return resolved[0], resolved[1]


def _resolve_market_for_matchup(matchup: dict, market_rows: list[dict]) -> Optional[tuple[dict, tuple[str, str]]]:
    matchup_players = (
        str(matchup.get("fighter_a") or ""),
        str(matchup.get("fighter_b") or ""),
    )
    matchup_key = build_match_key(*matchup_players)
    matchup_tour = str(matchup.get("tour") or "").strip().lower()
    seed_candidates = _matchup_seed_candidates(matchup)

    matches: list[tuple[int, dict, tuple[str, str]]] = []
    for market in market_rows:
        market_tour = str(market.get("tour") or "").strip().lower()
        if matchup_tour and market_tour and matchup_tour != market_tour:
            continue

        market_seed = market.get("tournament_seed")
        if seed_candidates and market_seed and not any(
            tournament_seeds_compatible(market_seed, seed) for seed in seed_candidates
        ):
            continue

        resolved = _match_market_players_to_matchup(
            (
                str(market.get("fighter_a") or ""),
                str(market.get("fighter_b") or ""),
            ),
            matchup_players,
        )
        if resolved is None:
            continue

        score = 0
        if build_match_key(market.get("fighter_a"), market.get("fighter_b")) == matchup_key:
            score += 100
        if matchup_tour and market_tour == matchup_tour:
            score += 10
        if seed_candidates and market_seed:
            if any(derive_tournament_seed(market_seed) == seed for seed in seed_candidates):
                score += 20
            else:
                score += 5
        matches.append((score, market, resolved))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    best_score = matches[0][0]
    winners = [(market, resolved) for score, market, resolved in matches if score == best_score]
    if len(winners) != 1:
        return None
    return winners[0]


def _gamma_get(endpoint: str, params: Optional[dict] = None) -> dict | list:
    response = requests.get(f"{POLYMARKET_GAMMA_URL}/{endpoint}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def search_tennis_events(query: str, limit_per_type: int = 10) -> list[dict]:
    """Search Gamma public-search for tennis events related to a tournament seed."""
    if not query:
        return []
    data = _gamma_get("public-search", params={"q": query, "limit_per_type": limit_per_type})
    if not isinstance(data, dict):
        return []
    return data.get("events", [])


def get_event_detail(event_id: str) -> dict:
    """Fetch a single Gamma event with embedded markets."""
    return _gamma_get(f"events/{event_id}")


def parse_tennis_market(
    market: dict,
    event: dict,
    tour: str,
    tournament_seed: str,
) -> Optional[dict]:
    """Parse a Polymarket tennis winner market into the shared execution schema."""
    question = market.get("question", "") or market.get("title", "") or event.get("title", "")
    if PROP_MARKET_PATTERN.search(question):
        return None

    event_title = event.get("title", "") or question
    players = extract_players_from_title(event_title)
    if players is None:
        return None

    outcomes = _parse_json_field(market.get("outcomes", []))
    if len(outcomes) != 2:
        return None

    full_names = normalize_tennis_market_names(players, outcomes, event_title)
    if full_names is None:
        return None

    tokens = _parse_json_field(market.get("clobTokenIds", []))
    prices = _parse_json_field(market.get("outcomePrices", []))
    fighter_a, fighter_b = full_names
    resolved_tour = _infer_tour_from_event(event, market) or tour

    return {
        "market_id": market.get("id", ""),
        "condition_id": market.get("conditionId", "") or market.get("condition_id", ""),
        "question": question,
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "token_id_yes": tokens[0] if len(tokens) > 0 else "",
        "token_id_no": tokens[1] if len(tokens) > 1 else "",
        "price_yes": float(prices[0]) if len(prices) > 0 else None,
        "price_no": float(prices[1]) if len(prices) > 1 else None,
        "best_bid": market.get("bestBid"),
        "best_ask": market.get("bestAsk"),
        "volume": market.get("volume", 0),
        "liquidity": market.get("liquidityNum", 0) or market.get("liquidity", 0),
        "tick_size": market.get("minimum_tick_size", "0.01"),
        "active": market.get("active", True),
        "closed": market.get("closed", False),
        "event_title": event_title,
        "event_date": event.get("eventDate", "") or event.get("startDate", "") or event.get("endDate", ""),
        "sport": "tennis",
        "tour": resolved_tour,
        "tournament_seed": tournament_seed,
        "match_key": build_match_key(fighter_a, fighter_b),
    }


def discover_tennis_markets(
    active_tennis_sports: Optional[list[dict]] = None,
    limit_per_seed: int = 50,
) -> pd.DataFrame:
    """Discover active tennis winner markets using active tournament names as search seeds."""
    sports = active_tennis_sports or discover_active_tennis_sports()
    event_ids_seen: set[str] = set()
    market_rows: list[dict] = []

    for sport in sports:
        tournament_seed = sport.get("tournament_seed") or derive_tournament_seed(sport.get("tournament_name"))
        if not tournament_seed:
            continue

        search_terms = [sport.get("tournament_name", ""), tournament_seed]
        for query in [term for term in search_terms if term]:
            for event in search_tennis_events(query, limit_per_type=limit_per_seed):
                event_id = str(event.get("id", ""))
                if not event_id or event_id in event_ids_seen:
                    continue
                if not _event_matches_seed(event, tournament_seed):
                    continue

                try:
                    detail = get_event_detail(event_id)
                except Exception as exc:
                    logger.warning("Failed to fetch Gamma event %s: %s", event_id, exc)
                    continue

                event_ids_seen.add(event_id)
                if detail.get("closed", False):
                    continue

                for market in detail.get("markets", []):
                    if market.get("closed", False):
                        continue
                    parsed = parse_tennis_market(
                        market=market,
                        event=detail,
                        tour=sport.get("tour", ""),
                        tournament_seed=tournament_seed,
                    )
                    if parsed is not None:
                        market_rows.append(parsed)

    if not market_rows:
        return pd.DataFrame(
            columns=[
                "market_id",
                "condition_id",
                "question",
                "fighter_a",
                "fighter_b",
                "token_id_yes",
                "token_id_no",
                "price_yes",
                "price_no",
                "best_bid",
                "best_ask",
                "volume",
                "liquidity",
                "tick_size",
                "active",
                "closed",
                "event_title",
                "event_date",
                "sport",
                "tour",
                "tournament_seed",
                "match_key",
            ]
        )

    markets = pd.DataFrame(market_rows).drop_duplicates(subset=["market_id"]).reset_index(drop=True)
    logger.info("Discovered %s active tennis Polymarket winner markets", len(markets))
    return markets


def match_tennis_markets(matchups: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Align discovered markets to a matchup frame using fighter_a/fighter_b names."""
    if matchups.empty or markets.empty:
        return pd.DataFrame()

    matchup_rows: list[dict] = []
    market_rows = markets.assign(
        match_key=markets.apply(lambda row: build_match_key(row["fighter_a"], row["fighter_b"]), axis=1)
    ).to_dict("records")

    for matchup in matchups.to_dict("records"):
        resolved_market = _resolve_market_for_matchup(matchup, market_rows)
        if resolved_market is None:
            continue
        market, resolved_players = resolved_market

        aligned = dict(matchup)
        aligned["market_id"] = market.get("market_id")
        aligned["condition_id"] = market.get("condition_id")
        aligned["market_question"] = market.get("question")
        aligned["market_event_title"] = market.get("event_title")
        aligned["market_event_date"] = market.get("event_date")
        aligned["market_tour"] = market.get("tour")
        aligned["market_liquidity"] = market.get("liquidity")
        aligned["tick_size"] = market.get("tick_size")
        aligned["sport"] = "tennis"

        market_fighter_a = resolved_players[0]
        if normalize_player_name(matchup.get("fighter_a")) == normalize_player_name(market_fighter_a):
            aligned["token_id_yes"] = market.get("token_id_yes")
            aligned["token_id_no"] = market.get("token_id_no")
            aligned["price_yes"] = market.get("price_yes")
            aligned["price_no"] = market.get("price_no")
        else:
            aligned["token_id_yes"] = market.get("token_id_no")
            aligned["token_id_no"] = market.get("token_id_yes")
            aligned["price_yes"] = market.get("price_no")
            aligned["price_no"] = market.get("price_yes")

        matchup_rows.append(aligned)

    return pd.DataFrame(matchup_rows)
