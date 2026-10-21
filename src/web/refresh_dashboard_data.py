"""Read-only views of the final refresh publication and its live consumer.

The original dashboard needs enriched rows, whereas the combined service saves
an immutable full card and a smaller current-odds card.  This adapter joins only
the same committed generation, exact bout identities and official card dates.
It never acquires data, loads a model, repairs caches or writes accounting.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import unicodedata
from zoneinfo import ZoneInfo


_NY = ZoneInfo("America/New_York")
_ROW_FIELDS = frozenset({
    "fighter_a", "fighter_b", "event_date", "event_title", "weight_class",
    "is_main_event", "is_title_bout", "num_rounds", "location", "is_empty_arena",
    "prob_a", "prob_b", "no_odds_prob_a", "no_odds_prob_b", "missing_feature_count",
    "model_contract", "trade_blocked", "trade_block_reason", "primary_model_available",
})


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _prob(value):
    value = _number(value)
    return value if value is not None and 0 <= value <= 1 else None


def _prob_pair(row, a, b):
    left, right = _prob(row.get(a)), _prob(row.get(b))
    if left is None or right is None or not math.isclose(left + right, 1, abs_tol=1e-6):
        return None, None
    return left, right


def _instant(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None
    except (TypeError, ValueError):
        return None


def _day(value):
    text = str(value or "")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None
    stamp = _instant(text)
    return stamp.astimezone(_NY).date().isoformat() if stamp else None


def _name(value):
    value = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return " ".join(re.sub(r"[^a-z0-9 ]", "", value).split())


def _bout(row):
    a, b = _name(row.get("fighter_a")), _name(row.get("fighter_b"))
    day = _card_day(row)
    return (tuple(sorted((a, b))), day) if a and b and a != b and day else None


def _card_day(row):
    return _day(row.get("official_event_date") or row.get("card_date") or row.get("event_date"))


def _identities_agree(base, observed):
    """Reject explicit identity conflicts even when the spelling/date agrees."""
    identities = defaultdict(set)
    for row in (base, observed):
        provenance = row.get("feature_provenance") or {}
        for side in ("a", "b"):
            name = _name(row.get(f"fighter_{side}"))
            for value in (row.get(f"fighter_{side}_id"), provenance.get(f"fighter_{side}_id")):
                if value is not None and str(value).strip():
                    identity = str(value).strip().casefold().rstrip("/").rsplit("/", 1)[-1]
                    identities[name].add(identity.removeprefix("ufcstats:"))
    return all(len(values) <= 1 for values in identities.values())


def _unique_index(rows):
    grouped = defaultdict(list)
    for row in rows:
        if isinstance(row, dict) and _bout(row):
            grouped[_bout(row)].append(row)
    return {key: values[0] for key, values in grouped.items() if len(values) == 1}


def _age(value, now):
    stamp = _instant(value)
    return (now - stamp).total_seconds() if stamp else None


def _orient(row, a_name):
    """Orient only explicitly observed side values; do not use fuzzy matching."""
    result = deepcopy(row)
    if _name(row.get("fighter_a")) == _name(a_name):
        return result
    for left, right in (("prob_a", "prob_b"), ("no_odds_prob_a", "no_odds_prob_b"),
                        ("a_market_prob", "b_market_prob"), ("a_fair_prob_avg", "b_fair_prob_avg")):
        result[left], result[right] = row.get(right), row.get(left)
    return result


class DashboardData:
    def __init__(self, service, trading=None, *, now=None, stale_after_seconds=None):
        self.service, self.trading = service, trading
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.stale_after_seconds = float(stale_after_seconds or getattr(trading, "max_age_seconds", 600))

    def _snapshot(self):
        publication = pointer = saved = report = status = refresh = {}
        errors = []
        try:
            publication, pointer, saved = self.service.published_pair()
        except Exception:
            errors.append("accepted_predictions_unavailable")
        try:
            refresh = json.loads(Path(self.service.status_path).read_text(encoding="utf-8"))
            if not isinstance(refresh, dict):
                raise ValueError("Invalid status")
        except Exception:
            errors.append("refresh_status_unavailable")
        if self.trading is not None:
            try:
                report = self.trading.report()
                if not isinstance(report, dict):
                    raise ValueError("Invalid report")
            except Exception:
                errors.append("live_predictions_unavailable")
            try:
                status = self.trading.status()
                if not isinstance(status, dict):
                    raise ValueError("Invalid status")
            except Exception:
                errors.append("trading_status_unavailable")
            try:
                self.trading.readiness()
            except Exception:
                errors.append("trading_consumer_not_ready")
        return publication, pointer, saved, report, status, refresh, errors

    def _coherent(self, publication, saved, cycle):
        generation = publication.get("generation_id")
        if not generation or saved.get("accepted_generation") != generation:
            return False
        if cycle.get("accepted_generation") != generation or cycle.get("publication_job_id") != publication.get("job_id"):
            return False
        pin = publication.get("prediction_report") or {}
        binding = cycle.get("snapshot_binding") or {}
        return bool(pin.get("sha256") and binding.get("final_report_sha256") == pin["sha256"])

    def _evaluating(self, snap):
        """Recognize a normal guarded cycle without promoting execution readiness."""
        publication, _pointer, saved, report, status, refresh, errors = snap
        cycle = report.get("cycle") or {}
        if (status.get("status") != "evaluating" or status.get("cycle_in_progress") is not True
                or status.get("process_alive") is not True or status.get("paused") is not False
                or status.get("ready") is not False or status.get("trading_enabled") is not False
                or status.get("effective_live_mode") != "real" or status.get("orders_disabled") is not False
                or report.get("effective_live_mode") != "real" or report.get("paused") is not False
                or report.get("startup_verified") is not True or report.get("accounting_reconciled") is not True
                or report.get("trading_enabled") is not True or report.get("orders_disabled") is not False
                or report.get("blocking_conditions") or status.get("blocking_conditions")
                or cycle.get("status") in {"error", "degraded", "busy"}
                or any(error != "trading_consumer_not_ready" for error in errors)
                or not self._coherent(publication, saved, cycle)):
            return False
        now = self._now()
        ages = [_age(stamp, now) for stamp in (status.get("updated_at_utc"),
                report.get("completed_at_utc"), cycle.get("market_observed_at"))]
        refresh_age = _age(refresh.get("updated_at_utc"), now)
        return (all(age is not None and -30 <= age <= self.stale_after_seconds for age in ages)
                and refresh_age is not None and -30 <= refresh_age <= 180)

    def _freshness(self, snap):
        publication, pointer, saved, report, status, refresh, errors = snap
        cycle = report.get("cycle") or {}
        coherent = self._coherent(publication, saved, cycle)
        now = self._now()
        observed = cycle.get("market_observed_at")
        live_age = _age(observed, now)
        heartbeat_age = _age(status.get("updated_at_utc"), now)
        refresh_age = _age(refresh.get("updated_at_utc"), now)
        refresh_stale = refresh_age is None or refresh_age < -30 or refresh_age > 180
        stale = (not coherent or live_age is None or live_age < -30 or live_age > self.stale_after_seconds
                 or heartbeat_age is None or heartbeat_age < -30 or heartbeat_age > self.stale_after_seconds)
        evaluating = self._evaluating(snap)
        blockers = [condition for source in (report, status, cycle)
                    for condition in (source.get("blocking_conditions") or [])]
        order_recovery = "original_order_unresolved" in blockers
        consumer_failed = (any(source.get("status") in {"error", "degraded", "busy"}
                               for source in (report, status, cycle))
                           or any(condition != "original_order_unresolved" for condition in blockers))
        failed = bool([error for error in errors if not (
            (evaluating or order_recovery) and error == "trading_consumer_not_ready")]
            or any(condition != "original_order_unresolved" for condition in blockers) or consumer_failed)
        # An attempted, failed cycle must not masquerade as initial startup.
        state = ("unavailable" if not saved else "degraded" if not coherent and (consumer_failed or order_recovery) else
                 "waiting_for_consumer" if not coherent else "stale" if stale else "degraded" if failed else
                 "blocked" if order_recovery else "evaluating" if evaluating else "current")
        source = cycle.get("freshness") or {}
        rows = saved.get("predictions") or []
        dataset = pointer.get("total_rows") or (pointer.get("artifacts") or {}).get("fights_cleaned.csv", {}).get("rows")
        cutoff = source.get("processed_snapshot_max_event_date") if coherent else None
        if cutoff is None and rows:
            cutoff = (rows[0].get("feature_provenance") or {}).get("processed_snapshot_max_event_date")
        return {
            "state": state, "is_stale": stale, "live_available": coherent and bool(cycle.get("predictions")),
            "consumer_evaluating": evaluating,
            "consumer_failed": consumer_failed, "order_recovery_required": order_recovery,
            "bookmaker_coverage_available": coherent and _instant(cycle.get("bookmaker_observed_at")) is not None,
            "polymarket_coverage_available": coherent and _instant(observed) is not None,
            "live_generation_matches_publication": coherent,
            "accepted_generation": publication.get("generation_id"),
            "published_at": publication.get("published_at_utc"), "saved_prediction_at": saved.get("decision_at"),
            "data_through": cutoff, "dataset_rows": dataset,
            "last_monitor_completed_at": report.get("completed_at_utc"),
            "bookmaker_observed_at": cycle.get("bookmaker_observed_at") if coherent else None,
            "polymarket_observed_at": observed if coherent else None,
            "official_card_observed_at": source.get("observed_at_utc") if coherent else None,
            "refresh_heartbeat_at": refresh.get("updated_at_utc"),
            "consumer_heartbeat_at": status.get("updated_at_utc"),
            "heartbeat_age_seconds": heartbeat_age, "refresh_heartbeat_age_seconds": refresh_age,
            "live_age_seconds": live_age if coherent else None, "stale_after_seconds": self.stale_after_seconds,
            "next_refresh_at": (refresh.get("schedule") or {}).get("next_due_at_local"),
            "refresh_state": refresh.get("state"), "refresh_error": bool(refresh.get("last_error")),
            "refresh_is_stale": refresh_stale,
            "pending_refresh": bool(refresh.get("pending_job")), "errors": errors,
            "monthly_enabled": False, "retraining_enabled": False,
        }

    def freshness(self):
        return self.predictions()["freshness"]

    def predictions(self, include_global_feature_importance=False):
        snap = self._snapshot()
        publication, _pointer, saved, report, status, _refresh, _errors = snap
        fresh = self._freshness(snap)
        cycle = report.get("cycle") or {}
        base_rows = saved.get("predictions") or []
        counts = Counter(_bout(row) for row in base_rows if isinstance(row, dict))
        live = _unique_index(cycle.get("predictions") or []) if fresh["live_generation_matches_publication"] else {}
        markets = _unique_index(cycle.get("mapped_markets") or []) if fresh["live_generation_matches_publication"] else {}
        rows = []
        for base in base_rows:
            if not isinstance(base, dict):
                continue
            key = _bout(base)
            current = live.get(key) if key and counts[key] == 1 else None
            if current and (not _identities_agree(base, current)
                            or None in _prob_pair(current, "prob_a", "prob_b")
                            or None in _prob_pair(current, "a_market_prob", "b_market_prob")
                            or current.get("market_inputs_available") is not True
                            or not _instant(current.get("prediction_timestamp"))):
                current = None
            row = {field: deepcopy(value) for field, value in base.items() if field in _ROW_FIELDS}
            row.update(card_date=_card_day(base), event_group_date=_card_day(base),
                       accepted_generation=publication.get("generation_id"), saved_prediction_at=saved.get("decision_at"),
                       prediction_source="saved_publication", market_inputs_available=False,
                       a_market_prob=None, b_market_prob=None, a_polymarket_prob=None, b_polymarket_prob=None,
                       bookmaker_observed_at=None, polymarket_observed_at=None)
            features = base.get("model_features") or {}
            row["fighter_context"] = {name: _number(value) for name, value in features.items()
                                      if name.startswith(("a_", "b_"))}
            row["method_stats"] = {name: value for name, value in row["fighter_context"].items()
                                   if "_roll_" in name or name.endswith(("_ko_rate", "_sub_rate", "_dec_rate"))}
            for side in ("a", "b"):
                count = _number(features.get(f"{side}_num_fights"))
                row[f"{side}_num_fights"] = int(count) if count is not None and count >= 0 and count == int(count) else None
            row["feature_highlights"], row["shap_values"] = [], []
            row["fighter_context_source"] = "accepted_saved_fighter_history"
            if current is not None:
                current = _orient(current, base.get("fighter_a"))
                for field in ("prob_a", "prob_b", "no_odds_prob_a", "no_odds_prob_b", "a_market_prob", "b_market_prob",
                              "trade_blocked", "trade_block_reason", "model_contract", "primary_model_available"):
                    if field in current:
                        row[field] = deepcopy(current[field])
                row.update(prediction_source="current_bookmaker_inputs", market_inputs_available=current.get("market_inputs_available") is True,
                           prediction_timestamp=current.get("prediction_timestamp"), bookmaker_observed_at=cycle.get("bookmaker_observed_at"),
                           event_date=current.get("event_date") or row.get("event_date"), bet_window=deepcopy(current.get("bet_window")))
            else:
                row["prediction_timestamp"] = saved.get("decision_at")
            market = markets.get(key) if key and counts[key] == 1 else None
            if market and _identities_agree(base, market):
                direct = _name(market.get("fighter_a")) == _name(base.get("fighter_a"))
                row["a_polymarket_prob"] = _prob(market.get("price_yes" if direct else "price_no"))
                row["b_polymarket_prob"] = _prob(market.get("price_no" if direct else "price_yes"))
                row["polymarket_observed_at"] = cycle.get("market_observed_at")
            row["polymarket_market_available"] = row["a_polymarket_prob"] is not None and row["b_polymarket_prob"] is not None
            row["prediction_is_stale"] = fresh["is_stale"] if current else True
            row["prediction_cache_status"] = fresh["state"] if current else "saved_only"
            self._enrich(row, fresh, status)
            rows.append(row)
        current_count = sum(row["prediction_source"] == "current_bookmaker_inputs" and row["market_inputs_available"] for row in rows)
        fresh.update(prediction_count=len(rows), current_prediction_count=current_count,
                     saved_only_prediction_count=len(rows) - current_count,
                     matched_market_count=sum(row["polymarket_market_available"] for row in rows),
                     bet_window_open_count=sum((row.get("bet_window") or {}).get("open") is True for row in rows),
                     effective_live_mode=status.get("effective_live_mode") or getattr(self.trading, "mode", "off"),
                     trading_enabled=status.get("trading_enabled") is True and not fresh["order_recovery_required"] and not fresh["consumer_failed"] and not any(
                         error in {"trading_status_unavailable", "trading_consumer_not_ready"} for error in _errors),
                     paused=status.get("paused") is True, consumer_status=status.get("status") or "unavailable")
        fresh["warnings"] = []
        if not fresh["bookmaker_coverage_available"]:
            fresh["warnings"].append("Current bookmaker coverage is unavailable; saved predictions are shown")
        elif len(rows) - current_count:
            fresh["warnings"].append(f"{len(rows) - current_count} bouts have no current bookmaker inputs; saved predictions are shown")
        if fresh["order_recovery_required"] or fresh["consumer_failed"]:
            fresh["warnings"].append(self._consumer_message(fresh))
        elif fresh["state"] not in {"current", "evaluating"}:
            fresh["warnings"].append(f"Live consumer data status: {fresh['state']}")
        if fresh["refresh_error"]:
            fresh["warnings"].append("Latest refresh attempt failed; the prior accepted publication remains served")
        if fresh["refresh_is_stale"]:
            fresh["warnings"].append("Refresh scheduler heartbeat is stale or unavailable")
        payload = {
            "predictions": rows, "prediction_count": len(rows), "current_prediction_count": current_count,
            "saved_only_prediction_count": len(rows) - current_count,
            "matched_market_count": sum(row["polymarket_market_available"] for row in rows),
            "bet_window_open_count": sum((row.get("bet_window") or {}).get("open") is True for row in rows),
            "timestamp": fresh["polymarket_observed_at"] or fresh["published_at"] or saved.get("decision_at"),
            "data_timestamp": fresh["polymarket_observed_at"] or fresh["published_at"] or saved.get("decision_at"),
            "timestamp_parse_failed": False, "freshness_age_minutes": None if fresh["live_age_seconds"] is None else round(max(0, fresh["live_age_seconds"]) / 60, 1),
            "stale_after_minutes": self.stale_after_seconds / 60, "is_stale": fresh["is_stale"],
            "cache_status": "current" if fresh["state"] in {"current", "evaluating", "blocked"} else "stale" if saved else "missing",
            "cache_available": bool(saved), "freshness": fresh,
            "market_probability_source": "bookmaker_consensus",
            "market_price_source": "polymarket_indicative_price",
            "message": (f"{current_count} current predictions; {len(rows) - current_count} saved predictions without current bookmaker inputs."
                        if fresh["bookmaker_coverage_available"] else "Current bookmaker coverage is unavailable; saved predictions are shown."),
        }
        if include_global_feature_importance:
            payload["global_feature_importance"] = []  # No attribution artifact is saved by this consumer.
        return payload

    def _enrich(self, row, fresh, status):
        from src.config import MIN_EDGE_THRESHOLD, MIN_FIGHTER_FIGHTS
        from src.strategy.value import compute_independent_blend_probs, dynamic_blend_weight

        a, b = _prob_pair(row, "prob_a", "prob_b")
        ma, mb = _prob_pair(row, "a_market_prob", "b_market_prob")
        na, nb = _prob_pair(row, "no_odds_prob_a", "no_odds_prob_b")
        if not row["market_inputs_available"]:
            ma = mb = None
        row.update(prob_a=a, prob_b=b, a_market_prob=ma, b_market_prob=mb,
                   no_odds_prob_a=na, no_odds_prob_b=nb)
        side = "a" if a is not None and b is not None and a >= b else "b" if b is not None else None
        pa, pb = row.get("a_polymarket_prob"), row.get("b_polymarket_prob")
        blend_a = blend_b = edge_a = edge_b = weight = None
        if None not in (a, b, ma, mb):
            blend_a, blend_b = compute_independent_blend_probs(a, ma, na, b, mb, nb)
            blend_a, blend_b = float(blend_a), float(blend_b)
            edge_a, edge_b = blend_a - ma, blend_b - mb
            weight = float(dynamic_blend_weight(a if side == "a" else b, ma if side == "a" else mb, na if side == "a" else nb))
        selected = lambda left, right: left if side == "a" else right if side == "b" else None
        value_side = "a" if edge_a is not None and edge_b is not None and edge_a >= edge_b else "b" if edge_b is not None else None
        best_edge = edge_a if value_side == "a" else edge_b if value_side == "b" else None
        confidence = max(a, b) if None not in (a, b) else None
        edge = selected(edge_a, edge_b)
        gap = abs(selected(a, b) - selected(ma, mb)) if side and None not in (a, b, ma, mb) else None
        counts = (row.get("a_num_fights"), row.get("b_num_fights"))
        experience = ("low_sample" if any(count is not None and count < MIN_FIGHTER_FIGHTS for count in counts)
                      else "unknown" if any(count is None for count in counts) else "normal")
        row.update(predicted_side=side, predicted_winner=row.get(f"fighter_{side}") if side else None,
                   predicted_prob=selected(a, b), predicted_market_prob=selected(ma, mb),
                   predicted_polymarket_prob=selected(pa, pb), predicted_market_source="bookmaker_consensus" if ma is not None else None,
                   predicted_blended_prob=selected(blend_a, blend_b), predicted_edge=edge,
                   blended_prob_a=blend_a, blended_prob_b=blend_b, edge_a=edge_a, edge_b=edge_b, blend_weight=weight,
                   value_side=value_side, value_fighter=row.get(f"fighter_{value_side}") if value_side else None,
                   best_edge=best_edge, confidence=confidence,
                   confidence_tier="strong_lean" if confidence is not None and confidence >= .68 else "lean" if confidence is not None and confidence >= .57 else "toss_up" if confidence is not None else "unavailable",
                   market_pick=row.get("fighter_a") if ma is not None and mb is not None and ma >= mb else row.get("fighter_b") if mb is not None else None,
                   no_odds_pick=row.get("fighter_a") if na is not None and nb is not None and na >= nb else row.get("fighter_b") if nb is not None else None,
                   market_gap=gap, market_disagreement=gap is not None and gap >= .08,
                   market_disagreement_note=None, experience_flag=experience,
                   experience_counts_available=all(count is not None for count in counts),
                   pick_value_status="potential_value" if edge is not None and edge >= MIN_EDGE_THRESHOLD else "pass" if edge is not None else "unavailable",
                   value_status="potential_value" if best_edge is not None and best_edge >= MIN_EDGE_THRESHOLD else "pass" if best_edge is not None else "unavailable",
                   pick_has_positive_edge=edge is not None and edge >= MIN_EDGE_THRESHOLD,
                   value_has_positive_edge=best_edge is not None and best_edge >= MIN_EDGE_THRESHOLD,
                   trade_candidate_active=False, trade_candidate_traders=[], trade_candidate_cells={})
        state, reason, detail = self._execution_display(row, fresh, status)
        row.update(pick_execution_status=state, value_execution_status=state, pick_is_bettable=False,
                   value_is_bettable=False, pick_filter_reason=reason, value_filter_reason=reason,
                   pick_filter_detail=detail, value_filter_detail=detail,
                   execution_eligibility_confirmed=False, edge_source="bookmaker_consensus_reference")

    @staticmethod
    def _execution_display(row, fresh, status):
        if fresh["order_recovery_required"]:
            return "blocked", "Order recovery required", "New orders are blocked until the exchange outcome of an existing order is confirmed"
        if not row["market_inputs_available"]:
            detail = ("Showing the saved accepted prediction and observed Polymarket quote; "
                      "current bookmaker inputs and order eligibility are unavailable"
                      if row["polymarket_market_available"] else
                      "Showing the saved accepted prediction; current bookmaker inputs, "
                      "Polymarket quote and order eligibility are unavailable")
            reason = "No current bookmaker odds" if fresh["bookmaker_coverage_available"] else "Current bookmaker coverage unavailable"
            return "saved_only", reason, detail
        if row.get("trade_blocked"):
            return "data_quality_blocked", "Data quality blocked", "The current consumer rejected live data quality"
        if row["prediction_is_stale"]:
            return "stale", "Live observations stale", "Last observed values are retained; await a fresh consumer cycle"
        if fresh["state"] == "degraded" or status.get("paused"):
            return "blocked", "Trading paused" if status.get("paused") else "Consumer unavailable", "Live execution requires the guarded consumer to be ready"
        if fresh["consumer_evaluating"]:
            return "evaluating", "Checking current data", "Showing the last completed cycle while the current consumer checks run; execution readiness is not yet confirmed"
        window = row.get("bet_window") or {}
        if not row["polymarket_market_available"]:
            detail = "The consumer has no exact current market binding for this bout"
            if window.get("open") is not True:
                detail += "; " + (window.get("detail") or window.get("reason") or "event time unavailable")
            return "no_market", "No current Polymarket market", detail
        if window.get("open") is not True:
            return "outside_window" if window.get("state") == "too_early" else "too_close" if window.get("state") == "too_late" else "event_time_unavailable", window.get("reason") or "Event time unavailable", window.get("detail")
        return "execution_review", "Awaiting execution decision", "An open window and current prices do not prove stake, account, risk and order checks have passed"

    def saved_predictions(self):
        """A thin immutable publication view, separate from the current overlay."""
        publication, _pointer, saved, *_ = self._snapshot()
        rows = [{key: deepcopy(value) for key, value in row.items() if key in _ROW_FIELDS}
                for row in saved.get("predictions", []) if isinstance(row, dict)]
        return {"accepted_generation": publication.get("generation_id"), "published_at": publication.get("published_at_utc"),
                "decision_at": saved.get("decision_at"), "prediction_count": len(rows), "predictions": rows}

    def upcoming_events(self):
        data = self.predictions()
        groups = defaultdict(list)
        for row in data["predictions"]:
            groups[(row.get("event_title") or "UFC", row.get("card_date"))].append(row)
        return [{"event": title, "date": day, "fight_count": len(rows), "sport": "ufc",
                 "source": "accepted_publication", "current_prediction_count": sum(row["market_inputs_available"] for row in rows),
                 "saved_only_prediction_count": sum(not row["market_inputs_available"] for row in rows),
                 "freshness": data["freshness"]["state"]}
                for (title, day), rows in groups.items()]

    def runtime_status(self):
        snap = self._snapshot()
        _pub, _pointer, saved, report, status, refresh, errors = snap
        fresh = self._freshness(snap)
        ready = (bool(saved) and not errors and status.get("ready") is True and not fresh["refresh_is_stale"]
                 and not fresh["order_recovery_required"] and not fresh["consumer_failed"])
        observing_while_blocked = self._observing_while_blocked(snap, fresh)
        display_errors = [error for error in errors if not (
            (fresh["consumer_evaluating"] or observing_while_blocked) and error == "trading_consumer_not_ready")]
        betting_state = status.get("status") or "unavailable"
        if status.get("process_alive") is False:
            betting_state = "dead"
        elif fresh["order_recovery_required"] and fresh["is_stale"]:
            betting_state = "stale"
        elif betting_state == "blocked" and not observing_while_blocked:
            betting_state = "degraded"
        elif observing_while_blocked:
            betting_state = "blocked"
        return {"ok": ready, "ready": ready, "startup_source": "refresh-combined",
                "consumer_evaluating": fresh["consumer_evaluating"],
                "effective_live_mode": status.get("effective_live_mode") or getattr(self.trading, "mode", "off"),
                "trading_enabled": status.get("trading_enabled") is True and not fresh["order_recovery_required"] and not fresh["consumer_failed"] and not any(
                    error in {"trading_status_unavailable", "trading_consumer_not_ready"} for error in errors),
                "orders_disabled": status.get("orders_disabled", True), "paused": status.get("paused", False),
                "enabled_sports": ["ufc"], "updated_at": status.get("updated_at_utc"),
                "errors": display_errors, "readiness_errors": errors,
                "warnings": (["Live prediction observations are stale"] if fresh["is_stale"] else [])
                    + (["Refresh scheduler heartbeat is stale or unavailable"] if fresh["refresh_is_stale"] else [])
                    + ([self._consumer_message(fresh)] if fresh["order_recovery_required"] else []),
                "components": {
                    "refresh_loop": {"state": "stale" if fresh["refresh_is_stale"] else refresh.get("state") or "unavailable", "updated_at": refresh.get("updated_at_utc"),
                                     "stale_after_seconds": 180, "message": "Final accepted publication served; scheduled refresh owns data updates"},
                    "betting_loop": {"state": betting_state, "updated_at": status.get("updated_at_utc"),
                                     "stale_after_seconds": self.stale_after_seconds, "process_alive": status.get("process_alive"),
                                     "message": self._consumer_message(fresh)},
                    "monitor_loop": {"state": "evaluating" if fresh["consumer_evaluating"] or (observing_while_blocked and status.get("status") == "evaluating") else "running" if ready or observing_while_blocked else "unavailable", "updated_at": report.get("completed_at_utc"),
                                     "stale_after_seconds": self.stale_after_seconds, "message": "Refresh consumer account and market monitoring"},
                    "btc5m_loop": {"state": "disabled", "message": "BTC is disabled"},
                }, "freshness": fresh}

    def _observing_while_blocked(self, snap, fresh):
        """Recognize one verified execution guard without hiding other faults."""
        _pub, _pointer, _saved, report, status, _refresh, errors = snap
        cycle = report.get("cycle") or {}
        known_status = (status.get("status") == "blocked"
                        and status.get("blocking_conditions") == ["original_order_unresolved"]
                        and status.get("startup_verified") is True
                        and status.get("accounting_reconciled") is False)
        evaluating = (status.get("status") == "evaluating" and status.get("cycle_in_progress") is True
                      and status.get("blocking_conditions", []) == [])
        if (fresh["state"] != "blocked" or fresh["is_stale"] or fresh["refresh_is_stale"]
                or fresh["consumer_failed"] or status.get("process_alive") is not True
                or not (known_status or evaluating) or status.get("ready") is not False
                or cycle.get("status") != "blocked"
                or any(source.get("blocking_conditions") != ["original_order_unresolved"]
                       for source in (report, cycle))
                or any(source.get("trading_enabled") is not False or source.get("orders_disabled") is not False
                       for source in (status, report, cycle))
                or any(source.get("effective_live_mode") != "real" or source.get("paused") is not False
                       for source in (status, report))
                or report.get("startup_verified") is not True or report.get("accounting_reconciled") is not False
                or type(report.get("real_submissions")) is not int or report.get("real_submissions") != 0
                or type(cycle.get("total_orders")) is not int or cycle.get("total_orders") != 0
                or type(cycle.get("confirmed_placed_orders")) is not int or cycle.get("confirmed_placed_orders") != 0
                or any(error != "trading_consumer_not_ready" for error in errors)):
            return False
        ages = [_age(stamp, self._now()) for stamp in
                (report.get("completed_at_utc"), cycle.get("bookmaker_observed_at"))]
        return all(age is not None and -30 <= age <= self.stale_after_seconds for age in ages)

    @staticmethod
    def _consumer_message(fresh):
        if fresh["order_recovery_required"]:
            message = "New orders are blocked until the exchange outcome of an existing order is confirmed"
            if fresh["consumer_failed"]:
                return message + "; the latest observation cycle also failed"
            return message + ("; current market observations remain available" if not fresh["is_stale"]
                              else "; current market observations are stale or unavailable")
        if fresh["consumer_failed"]:
            return "Latest consumer cycle failed; new orders are blocked and current market coverage could not be verified"
        if fresh["consumer_evaluating"]:
            return "Checking current data; the last completed cycle is shown"
        if fresh["state"] == "degraded":
            return "Consumer readiness checks have not passed; new orders are blocked"
        return "UFC consumer refreshes current bookmaker and market observations each cycle"

    def tracker_decisions(self, history=False):
        data = self.predictions()
        fights = []
        for row in data["predictions"]:
            cell = {"status": row["pick_execution_status"], "text": row["pick_filter_reason"],
                    "rationale": row["pick_filter_detail"], "confidence": row["confidence"], "bet_placed": None}
            fights.append({key: row.get(key) for key in ("fighter_a", "fighter_b", "event_date", "event_title", "weight_class", "card_date", "event_group_date")} |
                          {code: deepcopy(cell) for code in ("S", "C", "M")})
        return {"fights": fights, "count": len(fights), "timestamp": data["timestamp"], "history": bool(history),
                "source": "current_consumer_availability", "execution_decisions_available": False}

    def activity(self, limit=50, sport="all"):
        if sport not in {"all", "ufc"}:
            return []
        limit = max(1, min(int(limit), 10000))
        publication, _pointer, _saved, report, status, refresh, _errors = self._snapshot()
        entries = []
        def add(stamp, message, source, level="INFO", tag="MONITOR"):
            if _instant(stamp):
                entries.append({"timestamp": stamp, "level": level, "source": source, "message": message,
                                "sport": "ufc", "tag": tag, "color": "red" if level == "ERROR" else "blue"})
        add(publication.get("published_at_utc"), f"Accepted UFC refresh published: generation {publication.get('generation_id')}", "refresh_loop", tag="REFRESH")
        receipts = []
        state_dir = getattr(self.trading, "state_dir", None)
        if state_dir:
            for path in sorted((Path(state_dir) / "cycles").glob("*.json"), reverse=True)[:min(limit, 200)]:
                try:
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(row, dict):
                        receipts.append(row)
                except (OSError, ValueError):
                    continue
        receipts.append(report)
        seen = set()
        for receipt in receipts:
            stamp = receipt.get("completed_at_utc")
            if not stamp or stamp in seen:
                continue
            seen.add(stamp)
            cycle = receipt.get("cycle") or receipt
            count = cycle.get("prediction_count")
            submitted = receipt.get("real_submissions")
            statement = "no order submissions" if submitted == 0 else "submission count unconfirmed" if submitted is None else f"{submitted} submissions"
            add(stamp, f"UFC consumer cycle completed: {count if type(count) is int else 'unavailable'} current predictions; {statement}", "betting_loop", tag="RUN")
        add(refresh.get("updated_at_utc"), f"Refresh scheduler check: {refresh.get('state') or 'unavailable'}; next refresh {(refresh.get('schedule') or {}).get('next_due_at_local') or 'unavailable'}", "refresh_loop")
        if status.get("paused"):
            state_note = "paused"
        elif any("original_order_unresolved" in (source.get("blocking_conditions") or []) for source in (status, report)):
            state_note = "new orders blocked pending confirmation of an existing order's exchange outcome"
        elif status.get("orders_disabled") is True:
            state_note = "orders disabled"
        elif status.get("status") == "evaluating":
            state_note = "checking current data; last completed cycle shown" if self._evaluating(
                (publication, _pointer, _saved, report, status, refresh, _errors)) else "checking current data; execution readiness unconfirmed"
        else:
            state_note = "trading enabled" if status.get("trading_enabled") else "execution readiness unconfirmed"
        add(status.get("updated_at_utc"), f"UFC consumer heartbeat: {status.get('status') or 'unavailable'}; {state_note}", "betting_loop")
        entries.sort(key=lambda row: _instant(row["timestamp"]))
        return entries[-limit:]
