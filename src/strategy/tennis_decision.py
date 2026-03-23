"""Tennis market-comparison and execution decision helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config import (
    INITIAL_BANKROLL,
    MAX_BET_FRACTION,
    PROCESSED_DATA_DIR,
    TENNIS_CONFIDENCE_PENALTY_THRESHOLD,
    TENNIS_KELLY_FRACTION,
    TENNIS_LOW_CONFIDENCE_EDGE_PENALTY,
    TENNIS_LOW_HISTORY_EDGE_PENALTY,
    TENNIS_MEDIUM_HISTORY_EDGE_PENALTY,
    TENNIS_MIN_BOOKMAKERS,
    TENNIS_MIN_EDGE_THRESHOLD,
    TENNIS_REFERENCE_EDGE_FLOOR,
    TENNIS_SECOND_SOURCE_CONFIRMATION_GAP,
    TENNIS_SECOND_SOURCE_CONTRADICTION_GAP,
    TENNIS_SUSPICIOUS_REFERENCE_EDGE_THRESHOLD,
)
from src.data.tennis_data import normalize_player_name

TENNIS_AUTO_BLOCKLIST_PATH = PROCESSED_DATA_DIR / "tennis" / "auto_blocklist.csv"
TENNIS_LEGACY_BLOCKLIST_PATH = PROCESSED_DATA_DIR / "tennis" / "manual_review_blocklist.csv"

SecondSourceProvider = Callable[[dict[str, object]], Optional[dict[str, object]]]


def _coerce_probability(value: object) -> float:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(prob):
        return float("nan")
    return prob


def _normalize_probability_pair(prob_a: object, prob_b: object) -> tuple[float, float]:
    a = _coerce_probability(prob_a)
    b = _coerce_probability(prob_b)

    if np.isnan(a) and np.isnan(b):
        return float("nan"), float("nan")
    if np.isnan(a) and not np.isnan(b):
        a = 1.0 - b
    if np.isnan(b) and not np.isnan(a):
        b = 1.0 - a
    if np.isnan(a) or np.isnan(b):
        return float("nan"), float("nan")

    total = a + b
    if total <= 0:
        return float("nan"), float("nan")
    return a / total, b / total


def _split_reason_string(value: object) -> list[str]:
    if value is None:
        return []
    reasons: list[str] = []
    for token in str(value).split(","):
        normalized = token.strip()
        if normalized:
            reasons.append(normalized)
    return reasons


def _reason_string(values: list[str]) -> str:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return ",".join(deduped)


def tennis_match_key(fighter_a: object, fighter_b: object) -> str:
    """Return a stable normalized key for a tennis matchup."""
    names = sorted(
        [
            normalize_player_name(fighter_a),
            normalize_player_name(fighter_b),
        ]
    )
    return "|".join(name for name in names if name)


def load_tennis_auto_blocklist(
    path: Path | str = TENNIS_AUTO_BLOCKLIST_PATH,
    *,
    legacy_path: Path | str | None = TENNIS_LEGACY_BLOCKLIST_PATH,
) -> set[str]:
    """Load a normalized hard blocklist for tennis matchups."""
    candidate_paths = [Path(path)]
    if legacy_path is not None:
        legacy = Path(legacy_path)
        if legacy not in candidate_paths:
            candidate_paths.append(legacy)

    keys: set[str] = set()
    for blocklist_path in candidate_paths:
        if not blocklist_path.exists():
            continue

        try:
            frame = pd.read_csv(blocklist_path)
        except Exception:
            continue
        if frame.empty:
            continue

        if "active" in frame.columns:
            active_values = frame["active"].astype(str).str.strip().str.lower()
            frame = frame[~active_values.isin({"0", "false", "no"})].copy()
        if frame.empty:
            continue

        if "match_key" in frame.columns:
            for value in frame["match_key"].dropna():
                normalized = str(value).strip().lower()
                if normalized:
                    keys.add(normalized)
        elif {"fighter_a", "fighter_b"}.issubset(frame.columns):
            for _, row in frame.iterrows():
                match_key = tennis_match_key(row.get("fighter_a"), row.get("fighter_b"))
                if match_key:
                    keys.add(match_key)
    return keys


# Backwards-compatible alias for older imports.
load_tennis_review_blocklist = load_tennis_auto_blocklist


def fractional_kelly_stake(
    probability: float,
    price: float,
    bankroll: float = INITIAL_BANKROLL,
    fraction: float = TENNIS_KELLY_FRACTION,
    max_bet_fraction: float = MAX_BET_FRACTION,
) -> float:
    """Return a capped fractional Kelly stake using Polymarket-style share prices."""
    if np.isnan(probability) or np.isnan(price):
        return 0.0
    if price <= 0.0 or price >= 1.0 or bankroll <= 0.0 or fraction <= 0.0:
        return 0.0

    b = (1.0 - price) / price
    if b <= 0.0:
        return 0.0

    q = 1.0 - probability
    edge_fraction = ((b * probability) - q) / b
    raw_stake = max(0.0, edge_fraction) * bankroll * fraction
    return min(raw_stake, bankroll * max_bet_fraction)


def _history_penalty(min_player_matches: float) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if np.isnan(min_player_matches):
        return 0.0, reasons
    if min_player_matches <= 5:
        reasons.append("low_history_3_to_5")
        return TENNIS_LOW_HISTORY_EDGE_PENALTY, reasons
    if min_player_matches <= 10:
        reasons.append("low_history_6_to_10")
        return TENNIS_MEDIUM_HISTORY_EDGE_PENALTY, reasons
    return 0.0, reasons


def _confidence_penalty(confidence: float) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if np.isnan(confidence):
        return 0.0, reasons
    if confidence < TENNIS_CONFIDENCE_PENALTY_THRESHOLD:
        reasons.append("low_confidence")
        return TENNIS_LOW_CONFIDENCE_EDGE_PENALTY, reasons
    return 0.0, reasons


def annotate_tennis_reference_edges(
    predictions: pd.DataFrame,
    *,
    min_edge: float = TENNIS_MIN_EDGE_THRESHOLD,
    min_bookmakers: int = TENNIS_MIN_BOOKMAKERS,
    reference_edge_floor: float = TENNIS_REFERENCE_EDGE_FLOOR,
    suspicious_edge_threshold: float = TENNIS_SUSPICIOUS_REFERENCE_EDGE_THRESHOLD,
) -> pd.DataFrame:
    """Annotate model-vs-bookmaker consensus edges and uncertainty-aware thresholds."""
    if predictions.empty:
        return predictions.copy()

    rows: list[dict[str, object]] = []
    for row in predictions.to_dict("records"):
        annotated = dict(row)
        model_a, model_b = _normalize_probability_pair(row.get("prob_a"), row.get("prob_b"))
        market_a, market_b = _normalize_probability_pair(row.get("a_fair_prob_avg"), row.get("b_fair_prob_avg"))

        annotated["reference_market_prob_a"] = market_a
        annotated["reference_market_prob_b"] = market_b
        annotated["reference_edge_a"] = model_a - market_a if not np.isnan(model_a) and not np.isnan(market_a) else np.nan
        annotated["reference_edge_b"] = model_b - market_b if not np.isnan(model_b) and not np.isnan(market_b) else np.nan

        if np.isnan(annotated["reference_edge_a"]) and np.isnan(annotated["reference_edge_b"]):
            annotated["decision_side"] = ""
            annotated["decision_fighter"] = ""
            annotated["decision_status"] = "missing_market_reference"
            annotated["decision_reasons"] = "missing_market_reference"
            rows.append(annotated)
            continue

        decision_side = "a" if annotated["reference_edge_a"] >= annotated["reference_edge_b"] else "b"
        decision_fighter = row.get("fighter_a") if decision_side == "a" else row.get("fighter_b")
        decision_model_prob = model_a if decision_side == "a" else model_b
        decision_market_prob = market_a if decision_side == "a" else market_b
        decision_edge = annotated["reference_edge_a"] if decision_side == "a" else annotated["reference_edge_b"]
        confidence = max(model_a, model_b)
        match_counts = [
            _coerce_probability(row.get("a_num_matches")),
            _coerce_probability(row.get("b_num_matches")),
        ]
        valid_match_counts = [value for value in match_counts if not np.isnan(value)]
        min_player_matches = float(min(valid_match_counts)) if valid_match_counts else float("nan")
        num_bookmakers = int(float(row.get("num_bookmakers", 0) or 0))

        uncertainty_penalty = 0.0
        reasons: list[str] = []
        history_penalty, history_reasons = _history_penalty(min_player_matches)
        uncertainty_penalty += history_penalty
        reasons.extend(history_reasons)
        confidence_penalty, confidence_reasons = _confidence_penalty(confidence)
        uncertainty_penalty += confidence_penalty
        reasons.extend(confidence_reasons)

        required_edge = float(min_edge + uncertainty_penalty)
        status = "candidate"

        if decision_edge < reference_edge_floor:
            status = "reference_market_disagrees"
            reasons.append("reference_market_disagrees")
        elif num_bookmakers < int(min_bookmakers):
            status = "insufficient_bookmakers"
            reasons.append("insufficient_bookmakers")
        elif decision_edge > float(suspicious_edge_threshold):
            status = "suspicious_market_mismatch"
            reasons.append("suspicious_market_mismatch")
        elif decision_edge < required_edge:
            status = "insufficient_reference_edge"
            reasons.append("insufficient_reference_edge")
        else:
            status = "reference_qualified"

        annotated.update(
            {
                "decision_side": decision_side,
                "decision_fighter": decision_fighter,
                "decision_model_prob": decision_model_prob,
                "decision_market_prob": decision_market_prob,
                "decision_edge": decision_edge,
                "decision_confidence": confidence,
                "min_player_matches": min_player_matches,
                "uncertainty_penalty": uncertainty_penalty,
                "required_edge": required_edge,
                "decision_status": status,
                "decision_reasons": _reason_string(reasons),
            }
        )
        rows.append(annotated)

    return pd.DataFrame(rows)


def build_tennis_execution_decisions(
    predictions: pd.DataFrame,
    *,
    min_edge: float = TENNIS_MIN_EDGE_THRESHOLD,
    bankroll: float = INITIAL_BANKROLL,
    min_bookmakers: int = TENNIS_MIN_BOOKMAKERS,
    reference_edge_floor: float = TENNIS_REFERENCE_EDGE_FLOOR,
    suspicious_edge_threshold: float = TENNIS_SUSPICIOUS_REFERENCE_EDGE_THRESHOLD,
) -> pd.DataFrame:
    """Convert tennis predictions and execution prices into bet/no-bet decisions."""
    annotated = annotate_tennis_reference_edges(
        predictions,
        min_edge=min_edge,
        min_bookmakers=min_bookmakers,
        reference_edge_floor=reference_edge_floor,
        suspicious_edge_threshold=suspicious_edge_threshold,
    )
    if annotated.empty:
        return annotated

    rows: list[dict[str, object]] = []
    for row in annotated.to_dict("records"):
        decision = dict(row)
        decision_side = str(row.get("decision_side") or "")
        execution_price = _coerce_probability(row.get("price_yes")) if decision_side == "a" else _coerce_probability(row.get("price_no"))
        execution_edge = float("nan")
        execution_status = str(row.get("decision_status") or "")
        stake_usd = 0.0

        if decision_side not in {"a", "b"}:
            execution_status = "missing_decision_side"
        elif np.isnan(execution_price) or not (0.0 < execution_price < 1.0):
            execution_status = "missing_execution_price"
        else:
            execution_model_prob = float(row.get("decision_model_prob", np.nan))
            execution_edge = execution_model_prob - execution_price
            if row.get("decision_status") != "reference_qualified":
                execution_status = str(row.get("decision_status"))
            elif execution_edge < float(row.get("required_edge", min_edge)):
                execution_status = "insufficient_execution_edge"
            else:
                execution_status = "bettable_now"
                stake_usd = fractional_kelly_stake(
                    probability=execution_model_prob,
                    price=execution_price,
                    bankroll=bankroll,
                )
                if stake_usd <= 0.0:
                    execution_status = "zero_kelly"

        decision.update(
            {
                "execution_price": execution_price,
                "execution_edge": execution_edge,
                "execution_decimal_odds": (1.0 / execution_price) if not np.isnan(execution_price) and execution_price > 0.0 else np.nan,
                "execution_status": execution_status,
                "stake_usd": stake_usd,
            }
        )
        rows.append(decision)

    return pd.DataFrame(rows)


def _extract_second_source_probabilities(
    row: dict[str, object],
    *,
    provider: SecondSourceProvider | None = None,
) -> tuple[float, str, str]:
    decision_side = str(row.get("decision_side") or "")
    source_entry = provider(row) if provider is not None else None
    source_entry = dict(source_entry or {})

    source_name = str(
        source_entry.get("source_name")
        or source_entry.get("second_source_name")
        or row.get("second_source_name")
        or row.get("execution_market_source")
        or row.get("market_source")
        or ""
    ).strip()
    source_detail = str(
        source_entry.get("source_detail")
        or source_entry.get("second_source_detail")
        or row.get("second_source_detail")
        or ""
    ).strip()

    side_probability = _coerce_probability(source_entry.get("second_source_side_prob"))
    if np.isnan(side_probability):
        source_prob_a, source_prob_b = _normalize_probability_pair(
            source_entry.get("second_source_prob_a"),
            source_entry.get("second_source_prob_b"),
        )
        if decision_side == "a":
            side_probability = source_prob_a
        elif decision_side == "b":
            side_probability = source_prob_b

    if np.isnan(side_probability):
        side_probability = _coerce_probability(row.get("second_source_side_prob"))
    if np.isnan(side_probability):
        source_prob_a, source_prob_b = _normalize_probability_pair(
            row.get("second_source_prob_a"),
            row.get("second_source_prob_b"),
        )
        if decision_side == "a":
            side_probability = source_prob_a
        elif decision_side == "b":
            side_probability = source_prob_b

    if np.isnan(side_probability):
        execution_price = _coerce_probability(row.get("execution_price"))
        if decision_side in {"a", "b"} and not np.isnan(execution_price) and 0.0 < execution_price < 1.0:
            side_probability = execution_price
            source_name = source_name or "polymarket_execution_market"
            source_detail = source_detail or "execution_market_price"

    return side_probability, source_name, source_detail


def _evaluate_second_source(
    row: dict[str, object],
    *,
    provider: SecondSourceProvider | None = None,
    required: bool,
    confirmation_gap: float = TENNIS_SECOND_SOURCE_CONFIRMATION_GAP,
    contradiction_gap: float = TENNIS_SECOND_SOURCE_CONTRADICTION_GAP,
) -> dict[str, object]:
    side_probability, source_name, source_detail = _extract_second_source_probabilities(row, provider=provider)
    reference_probability = _coerce_probability(row.get("decision_market_prob"))
    gap_vs_reference = (
        abs(side_probability - reference_probability)
        if not np.isnan(side_probability) and not np.isnan(reference_probability)
        else np.nan
    )

    reasons: list[str] = []
    if np.isnan(side_probability):
        status = "unavailable" if required else "not_required"
        if required:
            reasons.append("second_source_unavailable")
    elif np.isnan(reference_probability):
        status = "available_no_reference"
    elif gap_vs_reference <= float(confirmation_gap):
        status = "confirmed"
        reasons.append("second_source_confirmed")
    elif gap_vs_reference >= float(contradiction_gap):
        status = "contradicted"
        reasons.append("second_source_contradicted")
    else:
        status = "unresolved"
        reasons.append("second_source_unresolved")

    return {
        "second_source_required": bool(required),
        "second_source_name": source_name,
        "second_source_detail": source_detail,
        "second_source_side_prob": side_probability,
        "second_source_gap_vs_reference": gap_vs_reference,
        "second_source_status": status,
        "second_source_reasons": _reason_string(reasons),
    }


def apply_tennis_automation_controls(
    decisions: pd.DataFrame,
    *,
    blocklist_path: Path | str = TENNIS_AUTO_BLOCKLIST_PATH,
    legacy_blocklist_path: Path | str | None = TENNIS_LEGACY_BLOCKLIST_PATH,
    second_source_provider: SecondSourceProvider | None = None,
    min_bookmakers: int = TENNIS_MIN_BOOKMAKERS,
    second_source_confirmation_gap: float = TENNIS_SECOND_SOURCE_CONFIRMATION_GAP,
    second_source_contradiction_gap: float = TENNIS_SECOND_SOURCE_CONTRADICTION_GAP,
) -> pd.DataFrame:
    """Annotate tennis decisions with fully automated skip/block controls."""
    if decisions.empty:
        return decisions.copy()

    blocked_match_keys = load_tennis_auto_blocklist(
        blocklist_path,
        legacy_path=legacy_blocklist_path,
    )
    rows: list[dict[str, object]] = []

    for row in decisions.to_dict("records"):
        annotated = dict(row)
        match_key = tennis_match_key(row.get("fighter_a"), row.get("fighter_b"))
        decision_status = str(row.get("decision_status") or "")
        execution_status = str(row.get("execution_status") or "")
        min_player_matches = _coerce_probability(row.get("min_player_matches"))
        num_bookmakers = int(float(row.get("num_bookmakers", 0) or 0))
        thin_market_requires_confirmation = (
            decision_status == "reference_qualified"
            and num_bookmakers <= int(min_bookmakers)
        )

        second_source = _evaluate_second_source(
            row,
            provider=second_source_provider,
            required=thin_market_requires_confirmation,
            confirmation_gap=second_source_confirmation_gap,
            contradiction_gap=second_source_contradiction_gap,
        )

        structured_status = "auto_eligible"
        automation_reasons: list[str] = []

        if match_key and match_key in blocked_match_keys:
            structured_status = "auto_block"
            automation_reasons.append("auto_blocklist")
        elif decision_status == "suspicious_market_mismatch":
            structured_status = "auto_skip"
            automation_reasons.append("suspicious_market_mismatch")
        elif decision_status != "reference_qualified":
            structured_status = "not_eligible"
            if decision_status:
                automation_reasons.append(decision_status)
        elif not np.isnan(min_player_matches) and min_player_matches <= 5:
            structured_status = "auto_skip"
            automation_reasons.append("low_history_auto_skip")
        elif thin_market_requires_confirmation:
            second_source_status = str(second_source.get("second_source_status") or "")
            if second_source_status == "confirmed":
                automation_reasons.append("thin_market_second_source_confirmed")
            else:
                structured_status = "auto_skip"
                automation_reasons.append("thin_market_auto_skip")
                automation_reasons.extend(_split_reason_string(second_source.get("second_source_reasons")))
        elif str(second_source.get("second_source_status") or "") == "contradicted":
            structured_status = "auto_skip"
            automation_reasons.append("second_source_market_disagreement")

        execution_allowed = structured_status == "auto_eligible"
        annotated.update(
            {
                "match_key": match_key,
                "structured_status": structured_status,
                "automation_status": structured_status,
                "execution_allowed": execution_allowed,
                "automation_reasons": _reason_string(automation_reasons),
                "llm_veto_status": "NO_VETO",
                "llm_veto_reason": "not_run",
                "llm_veto_flags": "",
                **second_source,
            }
        )

        if annotated.get("execution_status") == "bettable_now" and not execution_allowed:
            annotated["execution_status"] = structured_status

        resolved_execution_status = str(annotated.get("execution_status") or execution_status or "")
        trade_ready = bool(execution_allowed and resolved_execution_status == "bettable_now")
        if structured_status == "auto_block":
            trade_status = "blocked"
        elif structured_status == "auto_skip":
            trade_status = "auto_skipped"
        elif trade_ready:
            trade_status = "tradeable_now"
        elif execution_allowed and decision_status == "reference_qualified":
            trade_status = "reference_only_eligible"
        else:
            trade_status = "not_tradeable"

        trade_reasons = automation_reasons.copy()
        if not trade_reasons and resolved_execution_status and resolved_execution_status != "bettable_now":
            trade_reasons.append(resolved_execution_status)
        annotated["trade_ready"] = trade_ready
        annotated["trade_status"] = trade_status
        annotated["trade_reasons"] = _reason_string(trade_reasons)

        rows.append(annotated)

    return pd.DataFrame(rows)


def apply_tennis_review_controls(
    decisions: pd.DataFrame,
    *,
    blocklist_path: Path | str = TENNIS_AUTO_BLOCKLIST_PATH,
    review_actions_path: Path | str | None = None,
    min_bookmakers: int = TENNIS_MIN_BOOKMAKERS,
) -> pd.DataFrame:
    """Backwards-compatible alias for the retired manual-review API."""
    _ = review_actions_path
    return apply_tennis_automation_controls(
        decisions,
        blocklist_path=blocklist_path,
        min_bookmakers=min_bookmakers,
    )
