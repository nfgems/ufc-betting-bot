"""Original dashboard views bound to the combined refresh service, with reads only.

The legacy module is loaded privately for each Flask app. Its server, execution,
recovery and cleanup entrypoints are never registered or started here.
"""
from __future__ import annotations

import importlib.util
import itertools
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
from collections import Counter
from copy import deepcopy

from src.web.refresh_dashboard_data import _bout

logger = logging.getLogger(__name__)
_instances = itertools.count()
_LEDGERS = (
    ("S", "Single (Value)", "bet_ledger_single.json", .30),
    ("C", "Conviction", "bet_ledger_conviction.json", None),
    ("M", "Model Tracker", "bet_ledger_model_tracker.json", 1.0),
    ("G", "Legacy G Trader (retired)", "bet_ledger_gemini_tracker.json", None),
    ("B", "Trader B (disabled)", "bet_ledger_trader_b.json", None),
)
_REDACT = frozenset({
    "token_id", "token_id_yes", "token_id_no", "condition_id", "conditionId",
    "asset", "asset_id", "order_id", "orderID", "existing_order_id", "existing_token_id",
    "ledger_id", "ledger_bet_id", "display_ledger_id", "existing_ledger_id", "decision_key",
    "_ledger_path", "_original_id", "ledger_path", "manifest_path", "processed_dir", "response",
    "sources", "private_key", "api_key", "apiKey", "proxyWallet", "proxy_address", "funder_address",
    "wallet_address", "address", "proxy_target", "ip", "transactionHash", "transaction_hash",
    "orderId", "order_ids", "trade_id", "trade_ids", "taker_order_id", "maker_address", "taker_address",
    "owner_address", "txHash", "privateKey", "api_secret", "api_passphrase", "passphrase", "signature", "signedOrder",
    "actual_fill_trade_ids", "actual_fill_trade_components", "actual_cash_debit_evidence",
    "actual_cash_debit_evidence_sha256",
})
_REDACT_NORMALIZED = frozenset(re.sub(r"[^a-z0-9]", "", key.lower()) for key in _REDACT)


class _LazyReadOnlyClient:
    """Create the authenticated read adapter only when an account read needs it."""

    def __init__(self):
        self._client = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        if name.startswith(("create_", "cancel_", "redeem_", "submit_", "post_", "update_", "delete_")):
            raise RuntimeError("Dashboard account writes are unavailable")
        with self._lock:
            if self._client is None:
                from src.polymarket.readonly_account import ReadOnlyClobClient
                self._client = ReadOnlyClobClient()
            client = self._client
        return getattr(client, name)


def _private_legacy_module():
    # Sharing the legacy module's globals between Flask apps would mix adapters
    # and background caches. Importing it defines views, without starting work.
    name = f"src.web._refresh_dashboard_{next(_instances)}"
    path = Path(__file__).with_name("app.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _public_payload(value):
    if isinstance(value, dict):
        return {key: _public_payload(item) for key, item in value.items()
                if re.sub(r"[^a-z0-9]", "", str(key).lower()) not in _REDACT_NORMALIZED}
    if isinstance(value, (list, tuple)):
        return [_public_payload(item) for item in value]
    if isinstance(value, str):
        for name in ("WEB_DASHBOARD_TOKEN", "POLYMARKET_PRIVATE_KEY", "POLYMARKET_FUNDER_ADDRESS",
                     "ODDS_API_KEY", "CLOB_PROXY_URL"):
            secret = os.getenv(name, "")
            if secret:
                if name == "POLYMARKET_FUNDER_ADDRESS":
                    value = re.sub(re.escape(secret), "[redacted]", value, flags=re.IGNORECASE)
                else:
                    value = value.replace(secret, "[redacted]")
    return value


def _execution_matrix(payload, service, trading, logs):
    """Overlay only original audit paths bound to the current accepted cycle."""
    from src.strategy.execution_audit import load_execution_audit_cycles
    output = deepcopy(payload)
    output["execution_decisions_available"] = False
    output["execution_source"] = "current_consumer_availability"
    try:
        publication, _pointer, saved = service.published_pair()
        report = trading.report()
        cycle = report["cycle"]
        generation = publication.get("generation_id")
        if not generation or saved.get("accepted_generation") != generation or cycle.get("accepted_generation") != generation:
            raise ValueError("generation_mismatch")
        if not publication.get("job_id") or cycle.get("publication_job_id") != publication["job_id"]:
            raise ValueError("publication_job_mismatch")
        digest = (publication.get("prediction_report") or {}).get("sha256")
        if not digest or (cycle.get("snapshot_binding") or {}).get("final_report_sha256") != digest:
            raise ValueError("publication_report_mismatch")
        audit_pointer = cycle.get("execution_audit") or {}
        wanted = audit_pointer.get("cycle_id")
        if not wanted or not audit_pointer.get("completed_at") or type(audit_pointer.get("fight_count")) is not int:
            raise ValueError("execution_pointer_unavailable")
        candidates = []
        latest = Path(logs) / "execution_decision_audit_latest.json"
        if latest.is_file() and latest.stat().st_size <= 32 * 1024 * 1024:
            record = json.loads(latest.read_text(encoding="utf-8"))
            if isinstance(record, dict) and record.get("cycle_id") == wanted:
                candidates.append(record)
        candidates.extend(record for record in load_execution_audit_cycles(
            limit=100, log_path=Path(logs) / "execution_decision_audit.jsonl")
            if isinstance(record, dict) and record.get("cycle_id") == wanted)
        if not candidates or any(record != candidates[0] for record in candidates[1:]):
            raise ValueError("execution_cycle_unavailable_or_ambiguous")
        audit = candidates[0]
        if audit.get("completed_at") != audit_pointer["completed_at"] or audit.get("fight_count") != audit_pointer["fight_count"]:
            raise ValueError("execution_pointer_conflict")
        if not isinstance(audit.get("fights"), list) or len(audit["fights"]) != audit_pointer["fight_count"]:
            raise ValueError("execution_fight_count_conflict")
        mode = report.get("effective_live_mode")
        if mode not in {"real", "off"}:
            raise ValueError("execution_mode_unavailable")
        expected_paper = mode != "real"
        if type(audit.get("dry_run")) is not bool or audit["dry_run"] is not expected_paper:
            raise ValueError("execution_mode_conflict")
        metadata = audit.get("metadata") or {}
        if metadata.get("accepted_generation") not in (None, generation):
            raise ValueError("execution_generation_conflict")

        def unique(rows):
            rows = [row for row in rows if isinstance(row, dict)]
            counts = Counter(_bout(row) for row in rows)
            return {_bout(row): row for row in rows if _bout(row) and counts[_bout(row)] == 1}

        published = unique(saved.get("predictions") or [])
        current = unique(cycle.get("predictions") or [])
        audited = unique(audit["fights"])
        display_counts = Counter(_bout(row) for row in output.get("fights", []))
        applied = 0
        for row in output.get("fights", []):
            key = _bout(row)
            if not key or display_counts[key] != 1 or key not in published or key not in current or key not in audited:
                continue
            actual = audited[key]
            for trader in ("S", "C", "M"):
                path = (actual.get("paths") or {}).get(trader)
                if not isinstance(path, dict) or path.get("trader") != trader or not path.get("status") or not path.get("explanation"):
                    continue
                numbers = deepcopy(path.get("numbers")) if isinstance(path.get("numbers"), dict) else {}
                order = deepcopy(path.get("order")) if isinstance(path.get("order"), dict) else {}
                label = {"outside_window": "Waiting for bet window", "too_close": "Entry window closed",
                         "no_market": "No current market", "no_market_match": "No current market"}.get(
                             path["status"], str(path["status"]).replace("_", " ").capitalize())
                row[trader] = {
                    "status": path["status"], "text": label,
                    "rationale": path["explanation"], "explanation": path["explanation"], "gate": path.get("gate"),
                    "numbers": numbers, "edge": numbers.get("edge"), "confidence": numbers.get("confidence"),
                    "bet_placed": None, "order": order, "order_status": order.get("status"),
                    "execution_source": "saved_current_execution_audit", "execution_cycle_id": wanted,
                    "execution_completed_at": audit["completed_at"], "accepted_generation": generation,
                    # Numeric bet_side belongs to this original audit orientation.
                    "execution_fighter_a": actual.get("fighter_a"), "execution_fighter_b": actual.get("fighter_b"),
                }
                applied += 1
        output.update(execution_decisions_available=applied > 0, execution_path_count=applied,
                      execution_source="saved_current_execution_audit", execution_cycle_id=wanted,
                      execution_completed_at=audit["completed_at"], accepted_generation=generation)
    except (KeyError, TypeError, ValueError, OSError, AttributeError):
        output["execution_unavailable_reason"] = "A coherent current execution audit could not be verified"
    return output


def install_dashboard(app, service, trading):
    """Register original public GET views and return their exact route allowlist."""
    from flask import jsonify, request
    from src.polymarket.tracker import ReadOnlyBetLedgerView
    from src.web.refresh_dashboard_data import DashboardData

    data = DashboardData(service, trading)
    legacy = _private_legacy_module()
    logs = Path(os.getenv("UFC_LOGS_DIR") or legacy.LOGS_DIR)
    legacy.LOGS_DIR = logs
    legacy._server_host = "0.0.0.0"
    legacy._clob_client = _LazyReadOnlyClient()
    # Never call set_clob_client: it includes unrelated ledger adoption.
    legacy._load_prediction_payload = data.predictions
    legacy.get_runtime_status = data.runtime_status
    legacy._runtime_status_with_liveness = data.runtime_status
    legacy._kickoff_limit_order_reconcile = lambda *args, **kwargs: None
    legacy.auto_reconcile_sold_positions = lambda *args, **kwargs: 0
    legacy.maybe_prune_alert_store = lambda *args, **kwargs: None
    legacy._maybe_prune_upcoming_event_snapshots = lambda *args, **kwargs: None
    # Absence from an account snapshot does not prove an old trade was lost.
    legacy._profile_activity_loss_rows = lambda *args, **kwargs: []
    # Token overlap alone does not prove that a wallet row belongs to a trader.
    legacy._match_profile_ledger_bets = lambda *args, **kwargs: []

    def ledger_rows(path, owner=None):
        path = Path(path)
        if not path.is_file():
            return []
        if path.stat().st_size > 32 * 1024 * 1024:
            raise RuntimeError("Historical ledger exceeds dashboard read bound")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("bets"), list):
            raise RuntimeError("Historical ledger format is unavailable")
        rows = []
        for row in payload["bets"]:
            if isinstance(row, dict) and row.get("dry_run") is False:
                rows.append({**row, "_ledger_path": str(path), "_original_id": row.get("id"),
                             "trader": owner or row.get("trader") or legacy._trader_label_from_path(str(path)),
                             "sport": "ufc", "accounting_scope": "original_real_ledger_history"})
        return rows

    def ledger(path):
        return ReadOnlyBetLedgerView.from_bets(ledger_rows(path))

    def merged_ledgers():
        rows = []
        for owner, _label, filename, _blend in _LEDGERS:
            rows.extend(ledger_rows(logs / filename, owner))
        rows.sort(key=lambda row: str(row.get("placed_at") or ""))
        return ReadOnlyBetLedgerView.from_bets([{**row, "id": index} for index, row in enumerate(rows, 1)])

    legacy.BetLedger = lambda path: ledger(path)
    legacy.load_all_trader_ledgers = merged_ledgers
    legacy._trader_breakdown_specs = lambda: [(owner, label, logs / filename, blend)
                                             for owner, label, filename, blend in _LEDGERS]

    def open_account_positions():
        raw, source = legacy._load_live_pnl_snapshot()
        if source == "unavailable":
            # Do not present dormant historical rows as observed wallet holdings.
            return {"bets": [], "unmatched_positions": [], "_pnl_source": source,
                    "_pnl_degraded": True, "accounting_scope": "shared_wallet"}
        scoped = legacy._dashboard_live_pnl_from_raw(raw)
        rows = [legacy._aggregate_open_bet_position(position, []) for position in scoped.get("positions", [])]
        for row in rows:
            row["accounting_scope"] = "shared_wallet_owner_unattributed"
        return {"bets": rows, "unmatched_positions": [], "_pnl_source": source,
                "_pnl_degraded": source != "live", "accounting_scope": "shared_wallet"}

    legacy._compute_open_bets_enriched = open_account_positions

    def response(view):
        original = app.make_response(view())
        if original.is_json:
            body = _public_payload(original.get_json())
            cleaned = jsonify(legacy._sanitize_for_json(body))
            cleaned.status_code = original.status_code
            for key, value in original.headers.items():
                if key.lower() not in {"content-length", "content-type"}:
                    cleaned.headers[key] = value
            cleaned.headers["X-Dashboard-Data-Scope"] = "public-read-only"
            cleaned.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            return cleaned
        return original

    def activity_entries():
        try:
            limit = min(10_000, max(1, int(request.args.get("limit", 500))))
        except (TypeError, ValueError):
            limit = 500
        sport = legacy._normalize_sport_filter(request.args.get("sport", "all"))
        old = legacy._read_activity_entries(logs / "bot.log", limit=limit, sport=sport)
        saved = data.activity(limit=limit, sport=sport)
        unique = {(row.get("timestamp"), row.get("source"), row.get("message")): row for row in old + saved}
        return sorted(unique.values(), key=legacy._activity_timestamp_sort_key)[-limit:]

    def activity():
        entries = activity_entries()
        return legacy._json_no_store(entries, extra_headers=legacy._bot_activity_headers(logs / "bot.log", entries))

    def activity_snapshot():
        entries = activity_entries()
        return legacy._json_no_store(legacy._bot_activity_snapshot(logs / "bot.log", entries),
                                    extra_headers=legacy._bot_activity_headers(logs / "bot.log", entries))

    def alerts():
        payload = legacy.api_bot_alerts().get_json()
        current = legacy._runtime_issue_activity_entries(data.runtime_status())
        known = {(row.get("source"), row.get("message")) for row in payload.get("active_entries", [])}
        for row in current:
            if (row.get("source"), row.get("message")) in known:
                continue
            active = {**row, "status": "active", "last_seen": row.get("timestamp"),
                      "fingerprint": hashlib.sha256((str(row.get("source")) + ":" + str(row.get("message"))).encode()).hexdigest(),
                      "evidence_scope": "current_runtime_status"}
            payload.setdefault("entries", []).append(active)
            payload.setdefault("active_entries", []).append(active)
        entries, active = payload.get("entries", []), payload.get("active_entries", [])
        payload["entry_count"] = len(entries)
        for prefix, rows in (("", entries), ("active_", active)):
            payload[prefix + "error_count"] = sum(str(row.get("level", "")).upper() in {"ERROR", "CRITICAL"} for row in rows)
            payload[prefix + "warning_count"] = sum(str(row.get("level", "")).upper() == "WARNING" for row in rows)
        payload["active_count"] = len(active)
        return jsonify(payload)

    def significant_actions():
        entries = legacy.api_significant_actions().get_json()
        for row in data.activity(limit=30, sport=legacy._normalize_sport_filter(request.args.get("sport", "all"))):
            entries.append({**row, "tag": "REFRESH" if "refresh" in str(row.get("source", "")) else "RUN", "color": "purple"})
        unique = {(row.get("timestamp"), row.get("source"), row.get("message")): row for row in entries}
        return jsonify(sorted(unique.values(), key=legacy._activity_timestamp_sort_key)[-30:])

    def summary():
        payload = legacy.api_summary().get_json()
        payload["accounting_scope"] = "shared_wallet_with_original_real_ledger_history_separate"
        historical = merged_ledgers().get_summary()
        payload["original_real_ledger_summary"] = historical
        for key in ("total_bets", "settled_bets", "wins", "losses", "win_rate", "total_wagered"):
            payload[key] = historical[key]
        # The profile aggregate includes redeemable/resolved positions' old
        # consideration. Open exposure follows the same visible-position filter
        # as the current positions table, while aggregate wallet P&L is retained.
        raw, source = legacy._load_live_pnl_snapshot()
        if source in {"live", "stale"}:
            visible = legacy._dashboard_live_pnl_from_raw(
                raw, sport=legacy._normalize_sport_filter(request.args.get("sport", "all")))
            payload["open_invested"] = visible["visible_invested"]
        else:
            payload["open_invested"] = None
        payload["freshness"] = data.freshness()
        payload["roi"] = None
        payload["_roi_unavailable_reason"] = "A complete observed wallet cost basis is not available"
        payload["_counts_scope"] = "original_real_ledger_history"
        if payload.get("_pnl_source") == "unavailable":
            for key in ("realized_pnl", "unrealized_pnl", "open_bets"):
                payload[key] = None
            if payload.get("_profile_source") == "unavailable":
                for key in ("total_pnl", "open_invested", "positions_value"):
                    payload[key] = None
        return jsonify(payload)

    def profile_bets():
        original = app.make_response(legacy.api_profile_bets())
        payload = original.get_json()
        account_summary = payload.get("summary", {})
        account_summary["roi"] = None
        account_summary["accounting_scope"] = "shared_wallet"
        account_summary["original_real_ledger_summary"] = merged_ledgers().get_summary()
        account_summary["_roi_unavailable_reason"] = "A complete observed wallet cost basis is not available"
        if account_summary.get("_pnl_source") == "unavailable":
            for key in ("realized_pnl", "unrealized_pnl", "open_bets", "open_invested"):
                account_summary[key] = None
            if account_summary.get("_profile_source") == "unavailable":
                for key in ("total_pnl", "open_invested", "positions_value"):
                    account_summary[key] = None
        return jsonify(payload), original.status_code

    def predictions_detail():
        return jsonify(data.predictions(include_global_feature_importance=True))

    def predictions():
        return jsonify(data.predictions(include_global_feature_importance=False))

    def runtime():
        return jsonify(data.runtime_status())

    def upcoming():
        return jsonify(data.upcoming_events())

    def tracker_decisions():
        if request.args.get("history", "") == "1":
            return legacy.api_tracker_decisions()
        if hasattr(data, "tracker_decisions"):
            baseline = data.tracker_decisions(history=False)
        else:
            baseline = legacy.api_tracker_decisions().get_json()
        return jsonify(_execution_matrix(baseline, service, trading, logs))

    @app.context_processor
    def dashboard_context():
        try:
            freshness = data.freshness()
            state = trading.status()
        except Exception:
            freshness, state = {"state": "unavailable"}, {}
        text = ("UFC trading is paused." if state.get("paused") else
                "UFC new orders are blocked pending recovery of an existing order." if freshness.get("order_recovery_required") else
                "The latest UFC consumer cycle failed; new orders are blocked." if freshness.get("consumer_failed") else
                "UFC status needs attention. Last verified information is shown below." if freshness.get("is_stale") else
                "UFC consumer is checking current data; the last completed cycle is shown." if freshness.get("consumer_evaluating") else
                "UFC real trading is enabled under the existing entry and risk rules." if freshness.get("trading_enabled") else
                "UFC trading is off or waiting for verified startup.")
        return {"dashboard_freshness": freshness, "dashboard_live_mode_text": text,
                "dashboard_trading_text": text}

    overrides = {
        "/api/summary": summary, "/api/predictions": predictions,
        "/api/predictions-detail": predictions_detail, "/api/upcoming-events": upcoming,
        "/api/runtime-status": runtime, "/api/bot-activity": activity,
        "/api/bot-activity-snapshot": activity_snapshot, "/api/tracker-decisions": tracker_decisions,
        "/api/profile-bets": profile_bets,
        "/api/bot-alerts": alerts, "/api/significant-actions": significant_actions,
    }
    routes = set()
    for rule in legacy.app.url_map.iter_rules():
        path = rule.rule
        if "GET" not in rule.methods or path in {"/healthz", "/readyz"} or path.startswith("/static/"):
            continue
        view = overrides.get(path, legacy.app.view_functions[rule.endpoint])
        # The original route table contains only GET reads and explicit POST
        # mutations. POST routes are deliberately not copied to the live app.
        if "POST" in rule.methods:
            continue
        def public_view(view=view):
            return response(view)
        app.add_url_rule(path, endpoint="refresh_dashboard:" + rule.endpoint, view_func=public_view, methods=["GET"])
        routes.add(path)
    app.add_url_rule("/api/dashboard-status", endpoint="refresh_dashboard:freshness",
                     view_func=lambda: response(lambda: jsonify(data.freshness())), methods=["GET"])
    routes.add("/api/dashboard-status")
    app.extensions["refresh_dashboard"] = {"legacy": legacy, "data": data, "public_get_paths": frozenset(routes)}
    return frozenset(routes)
