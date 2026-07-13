"""Structured runtime execution audit for UFC betting cycles."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from src.betting_window import bet_window_status
from src.config import (
    BLEND_WEIGHT,
    CONVICTION_MIN_MODEL_PROB,
    CONVICTION_MIN_NO_ODDS_PROB,
    LOGS_DIR,
    MIN_EDGE_THRESHOLD,
    MIN_FIGHTER_FIGHTS,
    MODEL_AGREEMENT_MIN_EDGE,
    REQUIRE_MODEL_AGREEMENT,
)
from src.strategy.value import (
    _coerce_optional_float,
    _coerce_optional_int,
    _coerce_probability,
    _filter_rejection_reason,
    _first_valid_probability,
    _newbie_rule_config,
    calculate_expected_value,
    compute_independent_blend_probs,
    implied_prob_to_decimal_odds,
    newbie_penalty,
    required_edge_for_market,
)

EXECUTION_AUDIT_LOG_PATH = LOGS_DIR / "execution_decision_audit.jsonl"
EXECUTION_AUDIT_LATEST_PATH = LOGS_DIR / "execution_decision_audit_latest.json"
EXECUTION_AUDIT_SCHEMA_VERSION = 1


PATH_LABELS = {
    "S": "Single Trader / value pipeline",
    "C": "Conviction Trader",
    "M": "Model Tracker",
    "G": "Gemini Tracker",
}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_cycle_id(prefix: str = "cycle") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def _clean_json(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _clean_json(value.item())
        except Exception:
            return str(value)
    return value


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _safe_int(value) -> int | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _pct(value) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed * 100:+.1f}%"


def _prob(value) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed * 100:.1f}%"


def _money(value) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "N/A"
    return f"${parsed:.2f}"


def _row_dict(row) -> dict:
    if row is None:
        return {}
    if isinstance(row, pd.Series):
        return row.to_dict()
    if isinstance(row, dict):
        return dict(row)
    getter = getattr(row, "to_dict", None)
    if callable(getter):
        try:
            return dict(getter())
        except Exception:
            return {}
    return {}


def _norm_name(value) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _row_fight_key(row) -> str:
    data = _row_dict(row)
    fighters = sorted(
        [
            _norm_name(data.get("fighter_a") or data.get("fighter") or data.get("bet_on")),
            _norm_name(data.get("fighter_b") or data.get("opponent")),
        ]
    )
    event_date = str(
        data.get("event_date")
        or data.get("commence_time")
        or data.get("market_event_date")
        or ""
    ).strip()
    return "||".join([fighters[0], fighters[1], event_date])


def _fight_id_from_values(fighter_a: str, fighter_b: str, event_date: str) -> str:
    fighters = sorted([_norm_name(fighter_a), _norm_name(fighter_b)])
    return "||".join([fighters[0], fighters[1], str(event_date or "").strip()])


def _row_event_date(row) -> str:
    data = _row_dict(row)
    return str(
        data.get("event_date")
        or data.get("commence_time")
        or data.get("market_event_date")
        or ""
    )


def _row_market_event_date(row) -> str:
    data = _row_dict(row)
    return str(data.get("market_event_date") or data.get("event_date") or "")


def _bet_side_token(row: dict, side: str) -> str:
    return str(row.get("token_id_yes" if side == "a" else "token_id_no") or "")


def _base_fight(row) -> dict:
    data = _row_dict(row)
    return {
        "fighter_a": str(data.get("fighter_a") or data.get("fighter") or ""),
        "fighter_b": str(data.get("fighter_b") or data.get("opponent") or ""),
        "event_date": str(data.get("event_date") or data.get("commence_time") or ""),
        "market_event_date": str(data.get("market_event_date") or ""),
        "card_date": str(data.get("card_date") or ""),
        "event_title": str(data.get("event_title") or ""),
        "weight_class": str(data.get("weight_class") or ""),
    }


def _prediction_match_keys(rows: pd.DataFrame | Iterable[dict]) -> set[str]:
    if rows is None:
        return set()
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return set()
        return {_row_fight_key(row) for _, row in rows.iterrows()}
    return {_row_fight_key(row) for row in rows}


def _append_cycle_payload(payload: dict, *, log_path: Path = EXECUTION_AUDIT_LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_clean_json(payload), default=str, sort_keys=True) + "\n")


def write_latest_cycle_payload(
    payload: dict,
    *,
    latest_path: Path = EXECUTION_AUDIT_LATEST_PATH,
) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(_clean_json(payload), default=str, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def persist_cycle_payload(
    payload: dict,
    *,
    log_path: Path = EXECUTION_AUDIT_LOG_PATH,
    latest_path: Path = EXECUTION_AUDIT_LATEST_PATH,
) -> None:
    _append_cycle_payload(payload, log_path=log_path)
    write_latest_cycle_payload(payload, latest_path=latest_path)


def load_execution_audit_cycles(
    *,
    limit: int = 20,
    offset: int = 0,
    log_path: Path = EXECUTION_AUDIT_LOG_PATH,
) -> list[dict]:
    """Load a page of newest audit cycles without parsing the entire JSONL file."""
    if not log_path.exists():
        return []
    page_limit = max(1, int(limit or 1))
    page_offset = max(0, int(offset or 0))
    records: list[dict] = []
    skipped = 0
    with open(log_path, "rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        pending = b""
        while position > 0 and len(records) < page_limit:
            read_size = min(64 * 1024, position)
            position -= read_size
            handle.seek(position)
            pending = handle.read(read_size) + pending
            lines = pending.split(b"\n")
            pending = lines[0]
            for raw_line in reversed(lines[1:]):
                raw = raw_line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if skipped < page_offset:
                    skipped += 1
                    continue
                records.append(record)
                if len(records) >= page_limit:
                    break
        if position == 0 and pending.strip() and len(records) < page_limit:
            try:
                record = json.loads(pending.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                record = None
            if record is not None:
                if skipped < page_offset:
                    skipped += 1
                else:
                    records.append(record)
    return records


def load_execution_audit_cycle(
    cycle_id: str,
    *,
    log_path: Path = EXECUTION_AUDIT_LOG_PATH,
) -> dict | None:
    """Find one cycle by id, scanning newest records first in bounded pages."""
    target = str(cycle_id or "").strip()
    if not target:
        return None
    offset = 0
    while True:
        page = load_execution_audit_cycles(limit=50, offset=offset, log_path=log_path)
        if not page:
            return None
        found = next((item for item in page if str(item.get("cycle_id") or "") == target), None)
        if found is not None:
            return found
        if len(page) < 50:
            return None
        offset += len(page)


def load_latest_execution_audit(
    *,
    latest_path: Path = EXECUTION_AUDIT_LATEST_PATH,
    log_path: Path = EXECUTION_AUDIT_LOG_PATH,
) -> dict | None:
    if latest_path.exists():
        try:
            return json.loads(latest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    cycles = load_execution_audit_cycles(limit=1, log_path=log_path)
    return cycles[0] if cycles else None


@dataclass
class ExecutionAuditCollector:
    """Collect per-fight execution decisions for one runtime cycle."""

    dry_run: bool
    event_title: str = ""
    cycle_id: str = field(default_factory=make_cycle_id)
    started_at: str = field(default_factory=utcnow_iso)
    metadata: dict = field(default_factory=dict)
    _fights: dict[str, dict] = field(default_factory=dict)

    def _fight_for_row(self, row) -> dict:
        fight_key = _row_fight_key(row)
        if fight_key not in self._fights:
            base = _base_fight(row)
            base["fight_id"] = fight_key
            base["paths"] = {}
            self._fights[fight_key] = base
        else:
            existing = self._fights[fight_key]
            for key, value in _base_fight(row).items():
                if value and not existing.get(key):
                    existing[key] = value
        return self._fights[fight_key]

    def record_path(
        self,
        trader: str,
        row,
        *,
        status: str,
        gate: str,
        explanation: str,
        numbers: Optional[dict] = None,
        stage: str = "decision",
        final: bool = True,
        order: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        fight = self._fight_for_row(row)
        path = fight["paths"].setdefault(
            trader,
            {
                "trader": trader,
                "label": PATH_LABELS.get(trader, trader),
                "status": "pending",
                "gate": "pending",
                "explanation": "No runtime decision recorded yet.",
                "numbers": {},
                "order": {},
                "stages": [],
                "updated_at": self.started_at,
            },
        )
        record = {
            "stage": stage,
            "status": status,
            "gate": gate,
            "explanation": explanation,
            "numbers": _clean_json(numbers or {}),
            "timestamp": utcnow_iso(),
        }
        if order:
            record["order"] = _clean_json(order)
        if extra:
            record.update(_clean_json(extra))
        path["stages"].append(record)
        if final:
            path["status"] = status
            path["gate"] = gate
            path["explanation"] = explanation
            path["numbers"] = _clean_json(numbers or path.get("numbers", {}))
            if order is not None:
                path["order"] = _clean_json(order)
            if extra:
                for key, value in extra.items():
                    if key not in {"stage", "status", "gate", "explanation", "numbers", "timestamp"}:
                        path[key] = _clean_json(value)
            path["updated_at"] = record["timestamp"]

    def bind_executor(self, executor, trader: str) -> None:
        if executor is None:
            return
        executor.decision_audit_trader = trader
        executor.decision_audit_callback = (
            lambda bet, payload, _trader=trader: self.record_executor_result(
                _trader,
                bet,
                payload,
            )
        )

    def record_executor_result(self, trader: str, bet, payload: dict) -> None:
        bet_dict = _row_dict(bet)
        order = dict(payload.get("order") or {})
        status = str(payload.get("status") or order.get("status") or "skipped")
        gate = str(payload.get("gate") or "executor")
        explanation = str(payload.get("explanation") or "Executor did not return an explanation.")
        numbers = _execution_numbers(bet_dict, payload)
        self.record_path(
            trader,
            bet_dict,
            status=status,
            gate=gate,
            explanation=explanation,
            numbers=numbers,
            stage="executor",
            final=True,
            order=_order_summary(order),
        )

    def record_operator_decision(
        self,
        trader: str,
        bet,
        decision,
        *,
        final_if_blocked: bool = True,
    ) -> None:
        verdict = str(getattr(decision, "verdict", "") or "").upper()
        rationale = str(getattr(decision, "rationale", "") or "")
        if verdict == "BLOCK":
            self.record_path(
                trader,
                bet,
                status="blocked",
                gate="llm_operator_block",
                explanation=f"Blocked by LLM Operator: {rationale}",
                numbers=_selection_numbers_from_bet(_row_dict(bet)),
                stage="operator",
                final=final_if_blocked,
                extra={
                    "operator": {
                        "verdict": verdict,
                        "rationale": rationale,
                        "risk_flags": list(getattr(decision, "risk_flags", []) or []),
                        "decision_key": getattr(decision, "decision_key", ""),
                    }
                },
            )
            return
        self.record_path(
            trader,
            bet,
            status="operator_pass",
            gate="llm_operator_pass",
            explanation=f"LLM Operator passed this {PATH_LABELS.get(trader, trader)} candidate.",
            numbers=_selection_numbers_from_bet(_row_dict(bet)),
            stage="operator",
            final=False,
            extra={
                "operator": {
                    "verdict": verdict or "PASS",
                    "rationale": rationale,
                    "risk_flags": list(getattr(decision, "risk_flags", []) or []),
                    "decision_key": getattr(decision, "decision_key", ""),
                }
            },
        )

    def record_tracker_decision(self, trader: str, row, record: dict) -> None:
        status = str(record.get("status") or "skipped")
        gate = f"tracker_{status}"
        explanation = str(record.get("rationale") or record.get("summary") or "")
        final_status = "candidate" if status == "eligible" else _tracker_final_status(status)
        self.record_path(
            trader,
            row,
            status=final_status,
            gate=gate,
            explanation=explanation or f"{PATH_LABELS.get(trader, trader)} recorded {status}.",
            numbers=_tracker_numbers(row, record),
            stage="tracker_selection",
            final=True,
            extra={"tracker_decision": _clean_json(record)},
        )

    def to_payload(self) -> dict:
        fights = list(self._fights.values())
        for fight in fights:
            for path in fight.get("paths", {}).values():
                if str(path.get("status") or "") in {"pending", "candidate", "operator_pass"}:
                    path["status"] = "incomplete"
                    path["gate"] = "audit_incomplete"
                    path["explanation"] = (
                        "The cycle ended without a final executor result for this candidate. "
                        "No completed order decision was recorded."
                    )
                    path["updated_at"] = utcnow_iso()
            for trader in ("S", "C", "M", "G"):
                fight["paths"].setdefault(
                    trader,
                    {
                        "trader": trader,
                        "label": PATH_LABELS[trader],
                        "status": "not_evaluated",
                        "gate": "not_evaluated",
                        "explanation": f"{PATH_LABELS[trader]} was not evaluated for this fight in this cycle.",
                        "numbers": {},
                        "order": {},
                        "stages": [],
                        "updated_at": self.started_at,
                    },
                )
        fights.sort(
            key=lambda fight: (
                str(fight.get("market_event_date") or fight.get("event_date") or ""),
                str(fight.get("fighter_a") or ""),
                str(fight.get("fighter_b") or ""),
            )
        )
        path_counts: dict[str, dict[str, int]] = {}
        for fight in fights:
            for trader, path in fight.get("paths", {}).items():
                path_counts.setdefault(trader, {})
                status = str(path.get("status") or "unknown")
                path_counts[trader][status] = path_counts[trader].get(status, 0) + 1
        return {
            "schema_version": EXECUTION_AUDIT_SCHEMA_VERSION,
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "completed_at": utcnow_iso(),
            "dry_run": self.dry_run,
            "event_title": self.event_title,
            "metadata": _clean_json(self.metadata),
            "fight_count": len(fights),
            "path_counts": path_counts,
            "fights": _clean_json(fights),
        }

    def persist(self) -> dict:
        payload = self.to_payload()
        persist_cycle_payload(payload)
        return payload


def _selection_numbers_from_bet(bet: dict) -> dict:
    side = str(bet.get("bet_side") or "")
    token_id = _bet_side_token(bet, side)
    return {
        "bet_on": bet.get("bet_on"),
        "bet_side": side,
        "model_probability": _safe_float(bet.get("model_prob")),
        "no_odds_probability": _safe_float(bet.get("no_odds_prob")),
        "market_probability": _safe_float(bet.get("market_prob")),
        "blended_probability": _safe_float(bet.get("blended_prob")),
        "edge": _safe_float(bet.get("edge")),
        "required_edge": _safe_float(bet.get("required_edge")),
        "minimum_edge": _safe_float(bet.get("minimum_edge")),
        "a_ufc_fights": _safe_int(bet.get("a_num_fights")),
        "b_ufc_fights": _safe_int(bet.get("b_num_fights")),
        "min_fight_threshold": MIN_FIGHTER_FIGHTS,
        "market_id": str(bet.get("market_id") or ""),
        "condition_id": str(bet.get("condition_id") or ""),
        "token_id": token_id,
        "token_id_yes": str(bet.get("token_id_yes") or ""),
        "token_id_no": str(bet.get("token_id_no") or ""),
        "decimal_odds": _safe_float(bet.get("decimal_odds")),
        "bet_window": bet_window_status(
            bet.get("event_date") or bet.get("commence_time") or bet.get("market_event_date")
        ),
    }


def _execution_numbers(bet: dict, payload: dict) -> dict:
    numbers = _selection_numbers_from_bet(bet)
    numbers.update(_clean_json(payload.get("numbers") or {}))
    order = payload.get("order") or {}
    if order:
        numbers.update(
            {
                "amount": _safe_float(order.get("bet_size_usd") or order.get("amount")),
                "price": _safe_float(order.get("price")),
                "shares": _safe_float(order.get("shares")),
                "ledger_id": order.get("ledger_bet_id") or order.get("ledger_id"),
                "order_id": _extract_order_id(order),
                "order_type": order.get("order_type"),
            }
        )
    return numbers


def _order_summary(order: dict) -> dict:
    if not order:
        return {}
    return _clean_json(
        {
            "status": order.get("status"),
            "order_type": order.get("order_type"),
            "order_id": _extract_order_id(order),
            "ledger_id": order.get("ledger_bet_id") or order.get("ledger_id"),
            "amount": _safe_float(order.get("bet_size_usd") or order.get("amount")),
            "requested_amount": _safe_float(order.get("requested_bet_size_usd")),
            "submitted_amount": _safe_float(order.get("submitted_amount")),
            "price": _safe_float(order.get("price")),
            "shares": _safe_float(order.get("shares")),
            "token_id": order.get("token_id"),
            "dry_run": order.get("dry_run"),
            "error": order.get("error"),
            "matched": order.get("matched"),
            "placement_state": order.get("placement_state"),
            "response": order.get("response") if isinstance(order.get("response"), dict) else None,
        }
    )


def _extract_order_id(order: dict) -> str | None:
    if not isinstance(order, dict):
        return None
    response = order.get("response")
    for source in (order, response if isinstance(response, dict) else {}):
        for key in ("order_id", "orderID", "orderId", "id"):
            value = source.get(key)
            if value:
                return str(value)
    return None


def _tracker_final_status(status: str) -> str:
    if status in {"no_market"}:
        return "no_market_match"
    if status in {"outside_window", "too_close", "event_started", "event_time_unavailable"}:
        return "outside_window"
    if status.startswith("missing_"):
        return "skipped"
    return "skipped"


def _tracker_numbers(row, record: dict) -> dict:
    data = _row_dict(row)
    side = str(record.get("bet_side") or "")
    if not side and record.get("pick"):
        pick = str(record.get("pick") or "")
        if _norm_name(pick) == _norm_name(data.get("fighter_a")):
            side = "a"
        elif _norm_name(pick) == _norm_name(data.get("fighter_b")):
            side = "b"
    return {
        "bet_on": record.get("pick"),
        "bet_side": side,
        "model_probability": _safe_float(record.get("model_prob") or data.get("prob_a" if side == "a" else "prob_b")),
        "market_probability": _safe_float(record.get("market_prob") or data.get("a_market_prob" if side == "a" else "b_market_prob")),
        "edge": _safe_float(record.get("edge")),
        "confidence": _safe_float(record.get("confidence")),
        "market_id": str(record.get("market_id") or data.get("market_id") or ""),
        "token_id": _bet_side_token(data, side),
        "token_id_yes": str(data.get("token_id_yes") or ""),
        "token_id_no": str(data.get("token_id_no") or ""),
        "bet_window": bet_window_status(data.get("event_date") or data.get("commence_time") or data.get("market_event_date")),
    }


def audit_unmatched_markets(
    collector: ExecutionAuditCollector,
    *,
    trader: str,
    predictions: pd.DataFrame,
    matched_predictions: pd.DataFrame,
    label: str,
) -> None:
    if predictions is None or predictions.empty:
        return
    matched_keys = _prediction_match_keys(matched_predictions)
    for _, row in predictions.iterrows():
        if _row_fight_key(row) in matched_keys:
            continue
        collector.record_path(
            trader,
            row,
            status="no_market_match",
            gate="market_match",
            explanation=f"Skipped by {label} because no active Polymarket market matched this fight.",
            numbers={
                "bet_window": bet_window_status(
                    row.get("event_date") or row.get("commence_time") or row.get("market_event_date")
                ),
            },
        )


def audit_single_value_pipeline(
    collector: ExecutionAuditCollector,
    *,
    predictions: pd.DataFrame,
    matched_predictions: pd.DataFrame,
    min_edge: float = MIN_EDGE_THRESHOLD,
    blend_weight: float = BLEND_WEIGHT,
    edge_scaling_base: Optional[float] = None,
) -> None:
    audit_unmatched_markets(
        collector,
        trader="S",
        predictions=predictions,
        matched_predictions=matched_predictions,
        label="Single Trader",
    )
    if matched_predictions is None or matched_predictions.empty:
        return
    for _, row in matched_predictions.iterrows():
        decision = explain_single_value_row(
            row,
            min_edge=min_edge,
            blend_weight=blend_weight,
            edge_scaling_base=edge_scaling_base,
        )
        collector.record_path("S", row, **decision)


def audit_conviction_pipeline(
    collector: ExecutionAuditCollector,
    *,
    predictions: pd.DataFrame,
    matched_predictions: pd.DataFrame,
    require_positive_ev: bool = True,
) -> None:
    audit_unmatched_markets(
        collector,
        trader="C",
        predictions=predictions,
        matched_predictions=matched_predictions,
        label="Conviction Trader",
    )
    if matched_predictions is None or matched_predictions.empty:
        return
    for _, row in matched_predictions.iterrows():
        collector.record_path(
            "C",
            row,
            **explain_conviction_row(row, require_positive_ev=require_positive_ev),
        )


def explain_single_value_row(
    row,
    *,
    min_edge: float = MIN_EDGE_THRESHOLD,
    blend_weight: float = BLEND_WEIGHT,
    edge_scaling_base: Optional[float] = None,
) -> dict:
    data = _row_dict(row)
    model_a = _coerce_probability(data.get("prob_a"))
    model_b = _coerce_probability(data.get("prob_b"))
    market_a = _first_valid_probability(data.get("a_market_prob"), data.get("a_fair_prob_avg"), default=math.nan)
    market_b = _first_valid_probability(data.get("b_market_prob"), data.get("b_fair_prob_avg"), default=math.nan)
    if model_a is None or model_b is None or math.isnan(market_a) or math.isnan(market_b):
        return {
            "status": "skipped",
            "gate": "missing_probability",
            "explanation": "Skipped by Single Trader because model or market probabilities were unavailable.",
            "numbers": _common_prediction_numbers(data, "a"),
        }

    no_odds_a = _coerce_probability(data.get("no_odds_prob_a"))
    no_odds_b = _coerce_probability(data.get("no_odds_prob_b"))
    line_movement = _coerce_optional_float(data.get("line_movement"))
    line_is_sharp = _coerce_optional_int(data.get("line_is_sharp"))
    line_steam_move = _coerce_optional_int(data.get("line_steam_move"))
    a_fights = _coerce_optional_int(data.get("a_num_fights"))
    b_fights = _coerce_optional_int(data.get("b_num_fights"))
    a_org_tier = _coerce_optional_float(data.get("a_pre_ufc_org_tier_best"))
    b_org_tier = _coerce_optional_float(data.get("b_pre_ufc_org_tier_best"))
    newbie_adjustment = newbie_penalty(a_fights, b_fights, org_tier_a=a_org_tier, org_tier_b=b_org_tier)
    blend_a, blend_b = compute_independent_blend_probs(
        model_a,
        market_a,
        no_odds_a,
        model_b,
        market_b,
        no_odds_b,
        blend_weight,
    )
    candidates = [
        _value_candidate(
            data,
            side="a",
            fighter=data.get("fighter_a", "A"),
            model_prob=model_a,
            market_prob=market_a,
            blended_prob=blend_a,
            no_odds_prob=no_odds_a,
            edge_scaling_base=edge_scaling_base,
            newbie_adjustment=newbie_adjustment,
            a_fights=a_fights,
            b_fights=b_fights,
            line_movement=line_movement,
            line_is_sharp=line_is_sharp,
            line_steam_move=line_steam_move,
        ),
        _value_candidate(
            data,
            side="b",
            fighter=data.get("fighter_b", "B"),
            model_prob=model_b,
            market_prob=market_b,
            blended_prob=blend_b,
            no_odds_prob=no_odds_b,
            edge_scaling_base=edge_scaling_base,
            newbie_adjustment=newbie_adjustment,
            a_fights=a_fights,
            b_fights=b_fights,
            line_movement=line_movement,
            line_is_sharp=line_is_sharp,
            line_steam_move=line_steam_move,
        ),
    ]
    candidates.sort(key=lambda item: item["numbers"].get("edge") or -999.0, reverse=True)
    passing = [candidate for candidate in candidates if candidate["passes"] and (candidate["numbers"].get("edge") or 0.0) >= min_edge]
    if passing:
        candidate = passing[0]
        numbers = candidate["numbers"]
        return {
            "status": "candidate",
            "gate": "value_pipeline_pass",
            "explanation": (
                f"Single Trader value candidate on {numbers['bet_on']}: edge {_pct(numbers['edge'])}, "
                f"required {_pct(numbers['required_edge'])}."
            ),
            "numbers": {**numbers, "minimum_edge": min_edge},
        }

    best = candidates[0]
    numbers = {**best["numbers"], "minimum_edge": min_edge}
    edge = numbers.get("edge") or 0.0
    if edge < min_edge:
        return {
            "status": "skipped",
            "gate": "value_min_edge",
            "explanation": (
                f"Skipped by Single Trader because value edge was {_pct(edge)}, "
                f"needs at least {_pct(min_edge)} before later safeguards."
            ),
            "numbers": numbers,
        }

    rejection = best["rejection"] or {}
    gate = _gate_from_rejection("value", rejection.get("reason"))
    explanation = _value_rejection_explanation(numbers, rejection)
    return {
        "status": "skipped",
        "gate": gate,
        "explanation": explanation,
        "numbers": numbers,
    }


def _value_candidate(
    data: dict,
    *,
    side: str,
    fighter: str,
    model_prob: float,
    market_prob: float,
    blended_prob: float,
    no_odds_prob: Optional[float],
    edge_scaling_base: Optional[float],
    newbie_adjustment,
    a_fights: Optional[int],
    b_fights: Optional[int],
    line_movement: Optional[float],
    line_is_sharp: Optional[int],
    line_steam_move: Optional[int],
) -> dict:
    edge = blended_prob - market_prob
    rejection = _filter_rejection_reason(
        blended_prob,
        market_prob,
        edge,
        str(fighter),
        no_odds_prob,
        line_movement=line_movement,
        line_is_sharp=line_is_sharp,
        line_steam_move=line_steam_move,
        bet_side=side,
        a_num_fights=a_fights,
        b_num_fights=b_fights,
        a_org_tier=data.get("a_pre_ufc_org_tier_best"),
        b_org_tier=data.get("b_pre_ufc_org_tier_best"),
        newbie_adjustment=newbie_adjustment,
        edge_scaling_base=edge_scaling_base,
    )
    required_edge = required_edge_for_market(
        market_prob,
        edge_scaling_base=edge_scaling_base,
        newbie_adjustment=newbie_adjustment,
    )
    return {
        "passes": rejection is None,
        "rejection": rejection,
        "numbers": {
            **_common_prediction_numbers(data, side),
            "bet_on": str(fighter),
            "bet_side": side,
            "model_probability": model_prob,
            "no_odds_probability": no_odds_prob,
            "market_probability": market_prob,
            "blended_probability": blended_prob,
            "edge": edge,
            "required_edge": required_edge,
            "no_odds_edge": (
                no_odds_prob - market_prob if no_odds_prob is not None else None
            ),
            "model_agreement_min_edge": (
                MODEL_AGREEMENT_MIN_EDGE if REQUIRE_MODEL_AGREEMENT else None
            ),
            "newbie_rule": _newbie_rule_config(None).name,
            "newbie_extra_edge_required": newbie_adjustment.extra_edge_required,
            "newbie_reason": newbie_adjustment.reason,
            "decimal_odds": implied_prob_to_decimal_odds(market_prob),
            "line_movement": line_movement,
            "line_is_sharp": line_is_sharp,
            "line_steam_move": line_steam_move,
        },
    }


def _common_prediction_numbers(data: dict, side: str) -> dict:
    return {
        "a_ufc_fights": _safe_int(data.get("a_num_fights")),
        "b_ufc_fights": _safe_int(data.get("b_num_fights")),
        "min_fight_threshold": MIN_FIGHTER_FIGHTS,
        "market_id": str(data.get("market_id") or ""),
        "condition_id": str(data.get("condition_id") or ""),
        "token_id": _bet_side_token(data, side),
        "token_id_yes": str(data.get("token_id_yes") or ""),
        "token_id_no": str(data.get("token_id_no") or ""),
        "bet_window": bet_window_status(
            data.get("event_date") or data.get("commence_time") or data.get("market_event_date")
        ),
    }


def _gate_from_rejection(prefix: str, reason: str | None) -> str:
    normalized = str(reason or "").strip().lower()
    mapping = {
        "low experience": "experience",
        "prob too low": "minimum_probability",
        "odds too long": "maximum_odds",
        "no-odds unavailable": "no_odds_missing",
        "no-odds disagrees": "no_odds_edge",
        "sharp money against": "line_movement",
        "line moved against": "line_movement",
        "edge below threshold": "scaled_edge",
    }
    suffix = mapping.get(normalized, "filter")
    return f"{prefix}_{suffix}"


def _value_rejection_explanation(numbers: dict, rejection: dict) -> str:
    reason = str(rejection.get("reason") or "")
    detail = str(rejection.get("detail") or "")
    fighter = str(numbers.get("bet_on") or "the selected side")
    if reason == "No-odds disagrees":
        return (
            f"Skipped by Single Trader because no-odds edge was {_pct(numbers.get('no_odds_edge'))}, "
            f"needs {_pct(numbers.get('model_agreement_min_edge'))}."
        )
    if reason == "Low experience":
        return f"Skipped by value pipeline because {detail}."
    if reason == "Edge below threshold":
        return (
            f"Skipped by Single Trader because {fighter}'s edge was {_pct(numbers.get('edge'))}, "
            f"needs {_pct(numbers.get('required_edge'))}."
        )
    return f"Skipped by Single Trader at {reason or 'strategy filter'}: {detail}"


def explain_conviction_row(row, *, require_positive_ev: bool = True) -> dict:
    data = _row_dict(row)
    model_a = _coerce_probability(data.get("prob_a"))
    model_b = _coerce_probability(data.get("prob_b"))
    market_a = _first_valid_probability(data.get("a_market_prob"), data.get("a_fair_prob_avg"), default=math.nan)
    market_b = _first_valid_probability(data.get("b_market_prob"), data.get("b_fair_prob_avg"), default=math.nan)
    if model_a is None or model_b is None or math.isnan(market_a) or math.isnan(market_b):
        return {
            "status": "skipped",
            "gate": "conviction_missing_probability",
            "explanation": "Skipped by Conviction Trader because model or market probabilities were unavailable.",
            "numbers": _common_prediction_numbers(data, "a"),
        }
    no_odds_a = _coerce_probability(data.get("no_odds_prob_a"))
    no_odds_b = _coerce_probability(data.get("no_odds_prob_b"))
    a_fights = _safe_int(data.get("a_num_fights"))
    b_fights = _safe_int(data.get("b_num_fights"))
    candidates = [
        {
            "side": "a",
            "fighter": str(data.get("fighter_a") or "A"),
            "model": model_a,
            "market": market_a,
            "no_odds": no_odds_a,
            "own_fights": a_fights,
            "opp_fights": b_fights,
        },
        {
            "side": "b",
            "fighter": str(data.get("fighter_b") or "B"),
            "model": model_b,
            "market": market_b,
            "no_odds": no_odds_b,
            "own_fights": b_fights,
            "opp_fights": a_fights,
        },
    ]
    candidates.sort(key=lambda item: item["model"], reverse=True)
    for candidate in candidates:
        result = _explain_conviction_candidate(
            data,
            candidate,
            require_positive_ev=require_positive_ev,
        )
        if result["status"] == "candidate":
            return result
    return _explain_conviction_candidate(
        data,
        candidates[0],
        require_positive_ev=require_positive_ev,
    )


def _explain_conviction_candidate(data: dict, candidate: dict, *, require_positive_ev: bool) -> dict:
    side = candidate["side"]
    model_prob = candidate["model"]
    market_prob = candidate["market"]
    no_odds_prob = candidate["no_odds"]
    fighter = candidate["fighter"]
    numbers = {
        **_common_prediction_numbers(data, side),
        "bet_on": fighter,
        "bet_side": side,
        "model_probability": model_prob,
        "no_odds_probability": no_odds_prob,
        "market_probability": market_prob,
        "blended_probability": model_prob,
        "edge": model_prob - market_prob,
        "required_model_probability": CONVICTION_MIN_MODEL_PROB,
        "required_no_odds_probability": CONVICTION_MIN_NO_ODDS_PROB,
        "required_edge": 0.0,
        "decimal_odds": implied_prob_to_decimal_odds(market_prob),
    }
    if model_prob < CONVICTION_MIN_MODEL_PROB:
        return {
            "status": "skipped",
            "gate": "conviction_model_probability",
            "explanation": (
                f"Skipped by Conviction Trader because {fighter}'s model probability "
                f"was {_prob(model_prob)}, needs {_prob(CONVICTION_MIN_MODEL_PROB)}."
            ),
            "numbers": numbers,
        }
    if no_odds_prob is None or no_odds_prob < CONVICTION_MIN_NO_ODDS_PROB:
        return {
            "status": "skipped",
            "gate": "conviction_no_odds_probability",
            "explanation": (
                f"Skipped by Conviction Trader because no-odds probability was "
                f"{_prob(no_odds_prob)}, needs {_prob(CONVICTION_MIN_NO_ODDS_PROB)}."
            ),
            "numbers": numbers,
        }
    own_fights = candidate.get("own_fights")
    opp_fights = candidate.get("opp_fights")
    if own_fights is not None and own_fights < MIN_FIGHTER_FIGHTS:
        return {
            "status": "skipped",
            "gate": "conviction_experience",
            "explanation": (
                f"Skipped by Conviction Trader because {fighter} has {own_fights} UFC fights, "
                f"minimum is {MIN_FIGHTER_FIGHTS}."
            ),
            "numbers": numbers,
        }
    if opp_fights is not None and opp_fights < MIN_FIGHTER_FIGHTS:
        return {
            "status": "skipped",
            "gate": "conviction_experience",
            "explanation": (
                f"Skipped by Conviction Trader because opponent has {opp_fights} UFC fights, "
                f"minimum is {MIN_FIGHTER_FIGHTS}."
            ),
            "numbers": numbers,
        }
    expected_value = calculate_expected_value(model_prob, numbers["decimal_odds"])
    numbers["expected_value_per_dollar"] = expected_value
    if require_positive_ev and expected_value <= 0:
        return {
            "status": "skipped",
            "gate": "conviction_positive_ev",
            "explanation": (
                f"Skipped by Conviction Trader because expected value was {_pct(expected_value)} "
                f"at market probability {_prob(market_prob)}."
            ),
            "numbers": numbers,
        }
    return {
        "status": "candidate",
        "gate": "conviction_pipeline_pass",
        "explanation": (
            f"Conviction Trader candidate on {fighter}: model {_prob(model_prob)}, "
            f"no-odds {_prob(no_odds_prob)}, market {_prob(market_prob)}."
        ),
        "numbers": numbers,
    }


def summarize_order_for_explanation(trader: str, order: dict) -> str:
    label = PATH_LABELS.get(trader, trader)
    fighter = str(order.get("fighter") or "selection")
    status = str(order.get("status") or "")
    amount = _money(order.get("bet_size_usd") or order.get("amount"))
    price = _safe_float(order.get("price"))
    price_text = f"{price:.4f}" if price is not None else "N/A"
    order_id = _extract_order_id(order)
    ledger_id = order.get("ledger_bet_id") or order.get("ledger_id")
    if status == "dry_run":
        return f"{label} would place {amount} on {fighter} at {price_text} in dry run."
    if status == "placed":
        suffix = []
        if ledger_id is not None:
            suffix.append(f"ledger #{ledger_id}")
        if order_id:
            suffix.append(f"order {order_id}")
        joined = f" ({', '.join(suffix)})" if suffix else ""
        return f"{label} placed {amount} on {fighter} at {price_text}{joined}."
    if status == "failed":
        return f"{label} order failed for {fighter}: {order.get('error') or 'unknown error'}."
    if status == "unknown":
        return f"{label} submitted {fighter}, but the durable order result is unknown."
    return f"{label} executor returned status {status or 'unknown'} for {fighter}."


def order_payload(
    *,
    status: str,
    gate: str,
    explanation: str,
    order: Optional[dict] = None,
    numbers: Optional[dict] = None,
) -> dict:
    return {
        "status": status,
        "gate": gate,
        "explanation": explanation,
        "order": order or {},
        "numbers": numbers or {},
    }
