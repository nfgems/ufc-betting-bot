"""Immutable final-card/current-consumer presentation without exchange access."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from src.web.refresh_dashboard_data import DashboardData


NOW = datetime(2026, 10, 11, 16, 16, tzinfo=timezone.utc)
OBSERVED = "2026-10-11T12:15:42-04:00"


@pytest.fixture
def view(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("dashboard acquired network data"))
    publication = {"generation_id": "accepted-1", "job_id": "job-1", "published_at_utc": "2026-10-11T00:08:01-04:00",
                   "prediction_report": {"sha256": "fixture-sha", "path": "private-path"}}
    saved = {"accepted_generation": "accepted-1", "decision_at": "2026-10-10T23:36:27-04:00", "predictions": []}
    current, markets = [], []
    for index in range(13):
        a, b = ("Joaquin Buckley", "Mike Malott") if index == 0 else (f"Fighter A {index}", f"Fighter B {index}")
        saved["predictions"].append({"fighter_a": a, "fighter_b": b, "event_date": "2026-10-17", "event_title": "UFC Fight Night",
            "prob_a": .6, "prob_b": .4, "no_odds_prob_a": .65, "no_odds_prob_b": .35,
            "model_features": {"a_age": 31, "b_age": None, "a_num_fights": 12, "b_num_fights": 5,
                               "a_roll_slpm": 3.2, "b_roll_slpm": 4.1, "a_weight": None},
            "feature_provenance": {"processed_snapshot_max_event_date": "2026-10-10", "processed_dir": "private-path"},
            "fighter_a_id": "private-id", "token_id": "private-token"})
        if index < 9:
            current.append({"fighter_a": b, "fighter_b": a, "official_event_date": "2026-10-17", "event_date": "2026-10-17T22:00:00-04:00",
                "prob_a": .3, "prob_b": .7, "no_odds_prob_a": .32, "no_odds_prob_b": .68,
                "a_market_prob": .45, "b_market_prob": .55, "market_inputs_available": True, "trade_blocked": False,
                "prediction_timestamp": OBSERVED,
                "bet_window": {"open": False, "state": "too_early", "reason": "Bet window not open",
                               "detail": "48h before fight", "window_opens_at": "2026-10-16T02:00:00+00:00"}})
        if index < 8:
            markets.append({"fighter_a": b, "fighter_b": a, "event_date": "2026-10-17 21:00:00+00:00",
                            "price_yes": .545, "price_no": .455, "token_id_yes": "private-token"})
    saved["prediction_count"] = len(saved["predictions"])
    report = {"cycle_id": "cycle-1", "completed_at_utc": OBSERVED, "real_submissions": 0,
        "account": {"wallet": "private-wallet"}, "original_accounting": {"path": "private-ledger"},
        "cycle": {"accepted_generation": "accepted-1", "publication_job_id": "job-1",
            "snapshot_binding": {"final_report_sha256": "fixture-sha"}, "predictions": current, "mapped_markets": markets,
            "bookmaker_observed_at": OBSERVED, "market_observed_at": OBSERVED, "prediction_count": 9, "status": "evaluated",
            "freshness": {"processed_snapshot_max_event_date": "2026-10-10", "observed_at_utc": OBSERVED}}}
    status = {"updated_at_utc": OBSERVED, "process_alive": True, "ready": True, "status": "waiting",
              "effective_live_mode": "real", "trading_enabled": True, "orders_disabled": False, "paused": False}
    refresh = {"updated_at_utc": OBSERVED, "state": "waiting", "total_rows": 9999,
               "schedule": {"next_due_at_local": "2026-10-13T11:00:00-04:00"}}
    status_path = tmp_path / "refresh-status.json"
    status_path.write_text(json.dumps(refresh))
    service = SimpleNamespace(status_path=status_path,
                              published_pair=lambda: (publication, {"total_rows": 9503}, saved))
    trading = SimpleNamespace(report=lambda: report, status=lambda: status, readiness=lambda: {"ready": True},
                              state_dir=tmp_path / "consumer", mode="real", max_age_seconds=600)
    return SimpleNamespace(data=DashboardData(service, trading, now=lambda: NOW), publication=publication, saved=saved,
                           report=report, status=status, refresh=refresh, service=service, trading=trading)


def test_full_card_with_exact_current_overlay_and_orientation(view):
    data = view.data.predictions(include_global_feature_importance=True)
    assert (data["prediction_count"], data["current_prediction_count"], data["saved_only_prediction_count"], data["matched_market_count"]) == (13, 9, 4, 8)
    row = data["predictions"][0]
    assert row["fighter_a"] == "Joaquin Buckley" and row["prob_a"] == .7 and row["prob_b"] == .3
    assert row["no_odds_prob_a"] == .68 and row["a_market_prob"] == .55
    assert row["a_polymarket_prob"] == .455 and row["b_polymarket_prob"] == .545
    assert row["predicted_market_prob"] == .55 and row["predicted_polymarket_prob"] == .455
    assert row["predicted_market_source"] == "bookmaker_consensus"
    assert data["global_feature_importance"] == []
    assert data["bet_window_open_count"] == 0 and row["pick_filter_detail"] == "48h before fight"
    assert row["pick_execution_status"] == "outside_window" and row["pick_is_bettable"] is False
    assert row["fighter_context"]["b_age"] is None and row["method_stats"]["a_roll_slpm"] == 3.2
    assert (row["a_num_fights"], row["b_num_fights"]) == (12, 5)
    assert row["feature_highlights"] == [] and row["shap_values"] == []


def evaluating(view):
    view.status.update(status="evaluating", cycle_in_progress=True, ready=False, trading_enabled=False)
    view.report.update(effective_live_mode="real", orders_disabled=False, trading_enabled=True,
                       paused=False, startup_verified=True, accounting_reconciled=True, blocking_conditions=[])
    def guarded():
        raise RuntimeError("Consumer temporarily unready during its guarded evaluation")
    view.trading.readiness = guarded


def test_evaluating_displays_last_verified_cycle_without_promoting_ready_or_trading(view):
    evaluating(view)
    before = deepcopy(view.status)
    data = view.data.predictions()
    fresh = data["freshness"]
    assert fresh["state"] == "evaluating" and fresh["consumer_evaluating"] is True
    assert fresh["trading_enabled"] is False and fresh["errors"] == ["trading_consumer_not_ready"]
    assert data["cache_status"] == "current" and data["current_prediction_count"] == 9
    assert data["predictions"][0]["pick_execution_status"] == "evaluating"
    assert data["predictions"][0]["pick_is_bettable"] is False
    assert not any("degraded" in warning for warning in fresh["warnings"])
    runtime = view.data.runtime_status()
    assert runtime["consumer_evaluating"] is True and runtime["ready"] is False
    assert runtime["trading_enabled"] is False and runtime["orders_disabled"] is False
    assert runtime["components"]["monitor_loop"]["state"] == "evaluating"
    heartbeat = next(row for row in view.data.activity() if "heartbeat" in row["message"])
    assert "checking current data; last completed cycle shown" in heartbeat["message"]
    assert "orders disabled" not in heartbeat["message"]
    assert view.status == before
    with pytest.raises(RuntimeError):
        view.trading.readiness()


@pytest.mark.parametrize("conflict", ["initial", "prior_stale", "heartbeat_stale", "dead", "generation",
                                     "job", "sha", "report_blocker", "status_blocker", "mode",
                                     "paused", "refresh_stale", "missing_progress", "future_report"])
def test_evaluating_does_not_hide_startup_or_current_evidence_failures(view, conflict):
    evaluating(view)
    old = (NOW - timedelta(minutes=30)).isoformat()
    if conflict == "initial": view.report.pop("startup_verified")
    elif conflict == "prior_stale": view.report["completed_at_utc"] = old
    elif conflict == "heartbeat_stale": view.status["updated_at_utc"] = old
    elif conflict == "dead": view.status["process_alive"] = False
    elif conflict == "generation": view.report["cycle"]["accepted_generation"] = "other"
    elif conflict == "job": view.report["cycle"]["publication_job_id"] = "other"
    elif conflict == "sha": view.report["cycle"]["snapshot_binding"]["final_report_sha256"] = "other"
    elif conflict == "report_blocker": view.report["blocking_conditions"] = ["account_read_unavailable"]
    elif conflict == "status_blocker": view.status["blocking_conditions"] = ["unattributed_active_exposure"]
    elif conflict == "mode": view.report["effective_live_mode"] = "off"
    elif conflict == "paused": view.status["paused"] = True
    elif conflict == "refresh_stale":
        view.refresh["updated_at_utc"] = old
        view.service.status_path.write_text(json.dumps(view.refresh))
    elif conflict == "missing_progress": view.status.pop("cycle_in_progress")
    else: view.report["completed_at_utc"] = (NOW + timedelta(minutes=5)).isoformat()
    fresh = view.data.freshness()
    assert fresh["consumer_evaluating"] is False
    assert fresh["trading_enabled"] is False and "trading_consumer_not_ready" in fresh["errors"]
    assert view.data.runtime_status()["ready"] is False
    heartbeat = next(row for row in view.data.activity() if "heartbeat" in row["message"])
    assert "last completed cycle shown" not in heartbeat["message"]
    assert "orders disabled" not in heartbeat["message"]


def test_missing_bookmaker_rows_preserve_saved_model_and_null_market(view):
    row = view.data.predictions()["predictions"][-1]
    assert row["prob_a"] == .6 and row["predicted_prob"] == .6
    for field in ("a_market_prob", "b_market_prob", "a_polymarket_prob", "predicted_market_prob",
                  "predicted_blended_prob", "predicted_edge", "blended_prob_a", "best_edge", "blend_weight"):
        assert row[field] is None
    assert row["prediction_source"] == "saved_publication" and row["market_inputs_available"] is False
    assert row["pick_filter_reason"] == "No current bookmaker odds" and row["pick_is_bettable"] is False


def test_saved_prediction_can_show_current_polymarket_quote_without_bookmaker_inputs(view):
    base = view.saved["predictions"][-1]
    before = deepcopy(base)
    view.report["cycle"]["mapped_markets"].append({
        "fighter_a": base["fighter_b"], "fighter_b": base["fighter_a"],
        "event_date": "2026-10-17T21:00:00Z", "price_yes": .385, "price_no": .625})
    data = view.data.predictions()
    row = data["predictions"][-1]
    assert data["current_prediction_count"] == 9 and data["saved_only_prediction_count"] == 4
    assert data["matched_market_count"] == data["freshness"]["matched_market_count"] == 9
    assert row["a_polymarket_prob"] == row["predicted_polymarket_prob"] == .625
    assert row["b_polymarket_prob"] == .385 and row["polymarket_market_available"] is True
    assert row["polymarket_observed_at"] == OBSERVED
    assert row["prob_a"] == .6 and row["prediction_source"] == "saved_publication"
    assert row["prediction_timestamp"] == view.saved["decision_at"]
    assert row["market_inputs_available"] is False and row["bookmaker_observed_at"] is None
    for field in ("a_market_prob", "b_market_prob", "predicted_market_prob", "predicted_blended_prob",
                  "predicted_edge", "blend_weight", "best_edge"):
        assert row[field] is None
    assert "observed Polymarket quote" in row["pick_filter_detail"]
    assert "current price and order eligibility are unavailable" not in row["pick_filter_detail"]
    assert row["pick_execution_status"] == "saved_only" and row["pick_is_bettable"] is False
    assert row["execution_eligibility_confirmed"] is False and base == before


@pytest.mark.parametrize("field,value", [("accepted_generation", "other"), ("publication_job_id", "other"),
                                          ("snapshot_binding", {"final_report_sha256": "other"})])
def test_new_publication_does_not_mix_prior_consumer(view, field, value):
    view.report["cycle"][field] = value
    data = view.data.predictions()
    assert data["current_prediction_count"] == 0 and data["matched_market_count"] == 0
    assert data["freshness"]["state"] == "waiting_for_consumer"
    assert all(row["prob_a"] == .6 and row["a_market_prob"] is None for row in data["predictions"])


@pytest.mark.parametrize("kind", ["duplicate_live", "wrong_date", "different_name", "invalid_prob", "missing_timestamp"])
def test_ambiguous_or_invalid_rows_never_overlay(view, kind):
    row = view.report["cycle"]["predictions"][0]
    if kind == "duplicate_live":
        view.report["cycle"]["predictions"].append(deepcopy(row))
    elif kind == "wrong_date":
        row["official_event_date"] = "2026-10-18"
    elif kind == "different_name":
        row["fighter_a"] = "A different fighter"
    elif kind == "invalid_prob":
        row["prob_a"] = float("nan")
    else:
        row.pop("prediction_timestamp")
    data = view.data.predictions()
    assert data["current_prediction_count"] == 8
    assert data["predictions"][0]["prob_a"] == .6
    assert data["predictions"][0]["a_market_prob"] is None


def test_duplicate_saved_identity_preserves_rows_but_rejects_current_binding(view):
    view.saved["predictions"].append(deepcopy(view.saved["predictions"][0]))
    data = view.data.predictions()
    assert data["prediction_count"] == 14 and data["current_prediction_count"] == 8 and data["matched_market_count"] == 7


def test_stale_report_does_not_invent_recent_observations(view):
    view.data._now = lambda: NOW + timedelta(minutes=30)
    data = view.data.predictions()
    assert data["cache_status"] == "stale" and data["freshness"]["state"] == "stale"
    assert data["data_timestamp"] == OBSERVED
    assert data["predictions"][0]["a_market_prob"] == .55
    assert data["predictions"][0]["pick_execution_status"] == "stale"


def test_failed_readiness_and_pause_block_bettable_labels(view):
    def unready():
        raise RuntimeError("secret-http-query")
    view.trading.readiness = unready
    view.status["paused"] = True
    view.report["cycle"]["predictions"][0]["bet_window"]["open"] = True
    data = view.data.predictions()
    assert data["freshness"]["state"] == "degraded"
    assert data["predictions"][0]["pick_filter_reason"] == "Trading paused"
    assert view.data.runtime_status()["trading_enabled"] is False
    assert "secret-http-query" not in json.dumps(data)


def test_open_window_and_price_do_not_claim_execution_checks(view):
    view.report["cycle"]["predictions"][0]["bet_window"].update(open=True, state="open")
    data = view.data.predictions()
    assert data["bet_window_open_count"] == 1
    row = data["predictions"][0]
    assert row["pick_execution_status"] == "execution_review"
    assert row["execution_eligibility_confirmed"] is False
    assert row["pick_is_bettable"] is False


def test_public_view_thin_historical_view_and_read_only(view):
    before = deepcopy(view.saved), deepcopy(view.report)
    data = view.data.predictions()
    text = json.dumps(data)
    assert "private-" not in text
    assert "account" not in data and "original_accounting" not in data
    saved = view.data.saved_predictions()
    assert len(saved["predictions"]) == 13 and saved["predictions"][0]["prob_a"] == .6
    assert "a_market_prob" not in saved["predictions"][0]
    assert (view.saved, view.report) == before


def test_freshness_separates_data_monitor_heartbeat_and_next_refresh(view):
    fresh = view.data.freshness()
    assert fresh["data_through"] == "2026-10-10" and fresh["dataset_rows"] == 9503
    assert fresh["published_at"] == "2026-10-11T00:08:01-04:00"
    assert fresh["last_monitor_completed_at"] == OBSERVED and fresh["consumer_heartbeat_at"] == OBSERVED
    assert fresh["next_refresh_at"] == "2026-10-13T11:00:00-04:00"
    assert fresh["warnings"] == ["4 bouts have no current bookmaker inputs; saved predictions are shown"]
    assert fresh["trading_enabled"] is True and fresh["current_prediction_count"] == 9
    assert view.data.upcoming_events() == [{"event": "UFC Fight Night", "date": "2026-10-17", "fight_count": 13,
        "sport": "ufc", "source": "accepted_publication", "current_prediction_count": 9, "saved_only_prediction_count": 4, "freshness": "current"}]


def test_activity_uses_real_saved_observations_and_cycle_receipts(view):
    cycles = view.trading.state_dir / "cycles"
    cycles.mkdir(parents=True)
    (cycles / "older.json").write_text(json.dumps({"completed_at_utc": "2026-10-11T12:10:00-04:00", "prediction_count": 9,
        "real_submissions": None, "account": {"secret": "private-wallet"}}))
    (cycles / "bad.json").write_text("broken")
    entries = view.data.activity(limit=10)
    assert len(entries) == 5
    assert sum("cycle completed" in row["message"] for row in entries) == 2
    assert "submission count unconfirmed" in entries[1]["message"]
    assert not any(row["timestamp"] == NOW.isoformat() for row in entries)
    assert "private-wallet" not in json.dumps(entries)
    assert view.data.activity(limit=1)[0]["timestamp"] == OBSERVED
    assert view.data.activity(sport="btc") == []


def test_missing_report_retains_publication_and_emits_unavailability(view):
    def missing():
        raise FileNotFoundError("private-path")
    view.trading.report = missing
    data = view.data.predictions()
    assert data["prediction_count"] == 13 and data["current_prediction_count"] == 0
    assert data["freshness"]["state"] == "waiting_for_consumer"
    assert data["freshness"]["consumer_failed"] is False
    assert data["freshness"]["bookmaker_coverage_available"] is False
    assert data["freshness"]["polymarket_coverage_available"] is False
    assert data["predictions"][0]["pick_filter_reason"] == "Current bookmaker coverage unavailable"
    assert "13 bouts have no current bookmaker inputs; saved predictions are shown" not in data["freshness"]["warnings"]
    assert data["is_stale"] is True
    assert "private-path" not in json.dumps(data)


def test_tracker_availability_never_invents_prior_bets(view):
    matrix = view.data.tracker_decisions()
    assert matrix["count"] == 13 and matrix["execution_decisions_available"] is False
    assert matrix["fights"][0]["S"]["status"] == "outside_window"
    assert matrix["fights"][0]["M"]["bet_placed"] is None


def test_stale_refresh_scheduler_does_not_hide_healthy_consumer_or_claim_readiness(view):
    view.refresh["updated_at_utc"] = "2026-10-10T12:15:42-04:00"
    view.service.status_path.write_text(json.dumps(view.refresh))
    data = view.data.predictions()
    assert data["freshness"]["state"] == "current"  # These are fresh consumer observations.
    assert data["freshness"]["refresh_is_stale"] is True
    assert "Refresh scheduler heartbeat is stale or unavailable" in data["freshness"]["warnings"]
    runtime = view.data.runtime_status()
    assert runtime["ready"] is False and runtime["components"]["refresh_loop"]["state"] == "stale"
    assert runtime["trading_enabled"] is True  # Actual consumer arming is a separate fact.


@pytest.mark.parametrize("counts,expected,known", [
    ((0, 12), "low_sample", True), ((1, 3), "low_sample", True),
    ((2, 3), "normal", True), ((None, 3), "unknown", False),
    ((None, 0), "low_sample", False), ((-1, 3), "unknown", False),
    ((1.5, 3), "unknown", False), ((True, 3), "unknown", False),
])
def test_observed_experience_is_honest_and_never_changes_trade_guards(view, counts, expected, known):
    features = view.saved["predictions"][0]["model_features"]
    features.update(a_num_fights=counts[0], b_num_fights=counts[1])
    before = deepcopy(view.saved), deepcopy(view.report), deepcopy(view.status)
    row = view.data.predictions()["predictions"][0]
    assert row["experience_flag"] == expected
    assert row["experience_counts_available"] is known
    assert row["execution_eligibility_confirmed"] is False and row["pick_is_bettable"] is False
    assert (view.saved, view.report, view.status) == before


def test_no_market_and_closed_window_are_both_explained(view):
    row = view.data.predictions()["predictions"][8]
    assert row["prediction_source"] == "current_bookmaker_inputs"
    assert row["polymarket_market_available"] is False
    assert row["pick_execution_status"] == "no_market"
    assert row["pick_filter_reason"] == "No current Polymarket market"
    assert "no exact current market binding" in row["pick_filter_detail"]
    assert "48h before fight" in row["pick_filter_detail"]
    assert row["pick_is_bettable"] is False and row["value_is_bettable"] is False


def test_explicit_card_day_controls_utc_crossing_display_and_matching(view):
    base = view.saved["predictions"][0]
    base.update(official_event_date="2026-10-17", event_date="2026-10-17T01:00:00+00:00")
    row = view.data.predictions()["predictions"][0]
    assert row["card_date"] == row["event_group_date"] == "2026-10-17"
    assert row["prediction_source"] == "current_bookmaker_inputs"


@pytest.mark.parametrize("target", ["current", "market", "provenance"])
def test_explicit_identity_conflict_rejects_only_the_conflicting_overlay(view, target):
    base = view.saved["predictions"][0]
    base.update(fighter_a_id="aaaaaaaaaaaaaaaa", fighter_b_id="bbbbbbbbbbbbbbbb")
    current = view.report["cycle"]["predictions"][0]
    current.update(fighter_a_id="bbbbbbbbbbbbbbbb", fighter_b_id="aaaaaaaaaaaaaaaa")
    market = view.report["cycle"]["mapped_markets"][0]
    market.update(fighter_a_id="bbbbbbbbbbbbbbbb", fighter_b_id="aaaaaaaaaaaaaaaa")
    if target == "current": current["fighter_b_id"] = "cccccccccccccccc"
    elif target == "market": market["fighter_b_id"] = "cccccccccccccccc"
    else: current["feature_provenance"] = {"fighter_b_id": "cccccccccccccccc"}
    row = view.data.predictions()["predictions"][0]
    assert row["prediction_source"] == ("current_bookmaker_inputs" if target == "market" else "saved_publication")
    assert row["polymarket_market_available"] is (target != "market")
    assert row["pick_is_bettable"] is False


def test_evaluating_runtime_has_no_error_badge_but_retains_unready_evidence(view):
    evaluating(view)
    runtime = view.data.runtime_status()
    assert runtime["consumer_evaluating"] is True and runtime["errors"] == []
    assert runtime["readiness_errors"] == ["trading_consumer_not_ready"]
    assert runtime["ready"] is runtime["ok"] is runtime["trading_enabled"] is False
    with pytest.raises(RuntimeError): view.trading.readiness()
    view.report["blocking_conditions"] = ["account_read_unavailable"]
    runtime = view.data.runtime_status()
    assert runtime["consumer_evaluating"] is False
    assert runtime["errors"] == ["trading_consumer_not_ready"]


def blocked(view):
    for source in (view.status, view.report, view.report["cycle"]):
        source.update(blocking_conditions=["original_order_unresolved"], trading_enabled=False, orders_disabled=False)
    for source in (view.status, view.report):
        source.update(effective_live_mode="real", paused=False, startup_verified=True, accounting_reconciled=False)
    view.status.update(status="blocked", ready=False)
    view.report["cycle"].update(status="blocked", total_orders=0, confirmed_placed_orders=0)
    for row in view.report["cycle"]["predictions"]:
        row.update(trade_blocked=True, trade_block_reason="original_order_unresolved")
    def unready():
        raise RuntimeError("private-order-id")
    view.trading.readiness = unready


def test_order_recovery_blocks_execution_while_preserving_current_observations(view):
    blocked(view)
    before = deepcopy(view.saved), deepcopy(view.report), deepcopy(view.status)

    data = view.data.predictions()
    fresh = data["freshness"]
    assert fresh["state"] == "blocked" and fresh["order_recovery_required"] is True
    assert fresh["consumer_failed"] is False and fresh["is_stale"] is False
    assert fresh["bookmaker_coverage_available"] is fresh["polymarket_coverage_available"] is True
    assert fresh["trading_enabled"] is False and data["cache_status"] == "current"
    assert (data["current_prediction_count"], data["matched_market_count"]) == (9, 8)
    assert data["predictions"][0]["a_market_prob"] == .55
    assert data["predictions"][0]["a_polymarket_prob"] == .455
    for row in data["predictions"]:
        assert row["pick_filter_reason"] == "Order recovery required"
        assert row["pick_execution_status"] == "blocked" and row["pick_is_bettable"] is False
        assert row["execution_eligibility_confirmed"] is False
    runtime = view.data.runtime_status()
    assert runtime["ready"] is runtime["trading_enabled"] is False
    assert runtime["errors"] == []
    assert runtime["readiness_errors"] == ["trading_consumer_not_ready"]
    assert runtime["components"]["betting_loop"]["state"] == "blocked"
    assert runtime["orders_disabled"] is False
    assert runtime["components"]["monitor_loop"]["state"] == "running"
    assert "exchange outcome of an existing order is confirmed" in runtime["components"]["betting_loop"]["message"]
    assert "current market observations remain available" in runtime["components"]["betting_loop"]["message"]
    heartbeat = next(row for row in view.data.activity() if "heartbeat" in row["message"])
    assert "new orders blocked" in heartbeat["message"]
    assert "waiting_for_consumer" not in json.dumps(data)
    assert "private-order-id" not in json.dumps(data)
    assert (view.saved, view.report, view.status) == before
    view.status["process_alive"] = False
    assert view.data.runtime_status()["components"]["monitor_loop"]["state"] == "unavailable"
    assert view.data.runtime_status()["components"]["betting_loop"]["state"] == "dead"
    assert view.data.runtime_status()["errors"] == ["trading_consumer_not_ready"]
    with pytest.raises(RuntimeError):
        view.trading.readiness()


def test_blocked_consumer_evaluation_keeps_warning_and_last_verified_observations(view):
    blocked(view)
    view.status.update(status="evaluating", cycle_in_progress=True)
    for field in ("blocking_conditions", "startup_verified", "accounting_reconciled"):
        view.status.pop(field)
    runtime = view.data.runtime_status()
    assert runtime["errors"] == [] and runtime["readiness_errors"] == ["trading_consumer_not_ready"]
    assert runtime["ready"] is runtime["trading_enabled"] is False
    assert runtime["components"]["monitor_loop"]["state"] == "evaluating"
    assert runtime["components"]["betting_loop"]["state"] == "blocked"
    assert runtime["freshness"]["order_recovery_required"] is True
    assert view.data.predictions()["current_prediction_count"] == 9
    view.status["cycle_in_progress"] = False
    assert view.data.runtime_status()["errors"] == ["trading_consumer_not_ready"]


def publication_blocked(view):
    reasons = ["current_official_card_changed", "final_publication_freshness_blocked"]
    view.report["cycle"]["freshness"].update(
        official_card_observed=True, official_card_matches_publication=False,
        event_date="2026-10-24", trade_block_reasons=reasons,
        strict_messages=["private-source-path: snapshot missing completed event"])
    for row in view.report["cycle"]["predictions"]:
        row.update(trade_blocked=True, trade_block_reason="; ".join(
            (["original_order_unresolved"] if view.status.get("status") == "blocked" else []) + reasons))
    return list(reasons)


@pytest.mark.parametrize("evaluating_now", [False, True])
def test_publication_blockers_remain_visible_alongside_order_recovery(view, evaluating_now):
    blocked(view)
    reasons = publication_blocked(view)
    if evaluating_now:
        view.status.update(status="evaluating", cycle_in_progress=True)
        for field in ("blocking_conditions", "startup_verified", "accounting_reconciled"):
            view.status.pop(field)
    before = deepcopy(view.saved), deepcopy(view.report), deepcopy(view.status)
    data = view.data.predictions()
    fresh = data["freshness"]
    assert fresh["publication_block_reasons"] == reasons
    assert fresh["publication_block_messages"] == [
        "Official card check is for 2026-10-24; saved predictions are for 2026-10-17",
        "Saved predictions failed the current data freshness checks"]
    assert fresh["state"] == "blocked" and fresh["is_stale"] is False
    assert fresh["trading_enabled"] is False and fresh["consumer_failed"] is False
    assert (data["current_prediction_count"], data["matched_market_count"]) == (9, 8)
    warnings = " ".join(fresh["warnings"])
    assert "exchange outcome of an existing order is unconfirmed" in warnings
    for message in fresh["publication_block_messages"]:
        assert message in warnings
        assert message in data["predictions"][0]["pick_filter_detail"]
    for row in data["predictions"]:
        assert row["pick_filter_reason"] == "Order recovery and data checks"
        assert row["pick_is_bettable"] is row["execution_eligibility_confirmed"] is False
    assert data["predictions"][0]["trade_block_reason"] == "original_order_unresolved; " + "; ".join(reasons)
    runtime = view.data.runtime_status()
    assert runtime["ready"] is runtime["trading_enabled"] is False
    assert runtime["readiness_errors"] == ["trading_consumer_not_ready"]
    assert runtime["components"]["betting_loop"]["state"] == "blocked"
    assert all(message in runtime["components"]["betting_loop"]["message"]
               for message in fresh["publication_block_messages"])
    assert "private-source-path" not in json.dumps(data)
    assert (view.saved, view.report, view.status) == before
    with pytest.raises(RuntimeError):
        view.trading.readiness()


@pytest.mark.parametrize("fault", ["generation", "job", "sha", "market_stale", "heartbeat_stale",
    "official_stale", "official_future", "official_unknown", "old_report", "dead", "failed", "not_observed"])
def test_publication_diagnostics_never_promote_stale_or_unbound_cycles(view, fault):
    blocked(view)
    publication_blocked(view)
    cycle = view.report["cycle"]
    old = (NOW - timedelta(minutes=30)).isoformat()
    if fault == "generation": cycle["accepted_generation"] = "other"
    elif fault == "job": cycle["publication_job_id"] = "other"
    elif fault == "sha": cycle["snapshot_binding"]["final_report_sha256"] = "other"
    elif fault == "market_stale": cycle["market_observed_at"] = old
    elif fault == "heartbeat_stale": view.status["updated_at_utc"] = old
    elif fault == "official_stale": cycle["freshness"]["observed_at_utc"] = old
    elif fault == "official_future": cycle["freshness"]["observed_at_utc"] = (NOW + timedelta(minutes=5)).isoformat()
    elif fault == "official_unknown": cycle["freshness"].pop("observed_at_utc")
    elif fault == "old_report": view.report["completed_at_utc"] = old
    elif fault == "dead": view.status["process_alive"] = False
    elif fault == "failed": view.status["status"] = "degraded"
    else: cycle["freshness"]["official_card_observed"] = False
    data = view.data.predictions()
    assert data["freshness"]["publication_block_reasons"] == []
    assert data["freshness"]["publication_block_messages"] == []
    assert "saved predictions are for" not in " ".join(data["freshness"]["warnings"])
    assert data["freshness"]["order_recovery_required"] is True
    assert data["freshness"]["trading_enabled"] is False


def test_publication_diagnostics_do_not_change_global_controls_or_invent_dates(view):
    before_runtime = view.data.runtime_status()
    reasons = publication_blocked(view)
    view.report["cycle"]["freshness"]["event_date"] = "invalid-private-date"
    view.report["cycle"]["freshness"]["trade_block_reasons"] += ["private-unknown-reason", reasons[0]]
    data = view.data.predictions()
    assert data["freshness"]["publication_block_reasons"] == reasons
    assert data["freshness"]["publication_block_messages"][0] == "Current official card differs from the saved prediction card"
    assert "Current card/data checks block new orders" in " ".join(data["freshness"]["warnings"])
    assert data["predictions"][0]["pick_filter_reason"] == "Card/data checks blocked"
    assert all(row["pick_is_bettable"] is row["execution_eligibility_confirmed"] is False for row in data["predictions"])
    runtime = view.data.runtime_status()
    for field in ("ready", "trading_enabled", "orders_disabled", "paused", "errors", "readiness_errors"):
        assert runtime[field] == before_runtime[field]
    assert "invalid-private-date" not in json.dumps(data["freshness"])
    assert "private-unknown-reason" not in json.dumps(data["freshness"])


@pytest.mark.parametrize("fault", ["extra_blocker", "missing_status_blocker", "missing_report_blocker",
    "missing_cycle_blocker", "armed_status", "orders_disabled", "paused", "startup", "mode", "submitted",
    "unknown_orders", "confirmed_orders", "stale_bookmaker", "old_report", "refresh_stale", "generation"])
def test_order_blocker_does_not_hide_other_runtime_faults(view, fault):
    blocked(view)
    old = (NOW - timedelta(minutes=30)).isoformat()
    if fault == "extra_blocker": view.report["blocking_conditions"].append("account_read_unavailable")
    elif fault == "missing_status_blocker": view.status.pop("blocking_conditions")
    elif fault == "missing_report_blocker": view.report.pop("blocking_conditions")
    elif fault == "missing_cycle_blocker": view.report["cycle"].pop("blocking_conditions")
    elif fault == "armed_status": view.status["trading_enabled"] = True
    elif fault == "orders_disabled": view.report["cycle"]["orders_disabled"] = True
    elif fault == "paused": view.status["paused"] = True
    elif fault == "startup": view.report["startup_verified"] = False
    elif fault == "mode": view.status["effective_live_mode"] = "off"
    elif fault == "submitted": view.report["real_submissions"] = 1
    elif fault == "unknown_orders": view.report["cycle"].pop("total_orders")
    elif fault == "confirmed_orders": view.report["cycle"]["confirmed_placed_orders"] = 1
    elif fault == "stale_bookmaker": view.report["cycle"]["bookmaker_observed_at"] = old
    elif fault == "old_report": view.report["completed_at_utc"] = old
    elif fault == "refresh_stale":
        view.refresh["updated_at_utc"] = old
        view.service.status_path.write_text(json.dumps(view.refresh))
    else: view.report["cycle"]["accepted_generation"] = "different"
    runtime = view.data.runtime_status()
    assert "trading_consumer_not_ready" in runtime["errors"]
    assert runtime["ready"] is runtime["trading_enabled"] is False
    assert runtime["components"]["monitor_loop"]["state"] == "unavailable"
    assert runtime["components"]["betting_loop"]["state"] in {"degraded", "stale"}


@pytest.mark.parametrize("source", ["status", "report", "cycle"])
def test_failed_incoherent_cycle_is_not_presented_as_startup_or_no_odds(view, source):
    view.report["cycle"] = {"status": "evaluated", "predictions": []}
    target = view.report["cycle"] if source == "cycle" else getattr(view, source)
    target.update(status="degraded", blocking_conditions=["market_or_prediction_cycle_unavailable"])
    data = view.data.predictions()
    fresh = data["freshness"]
    assert fresh["state"] == "degraded" and fresh["consumer_failed"] is True
    assert fresh["bookmaker_coverage_available"] is fresh["polymarket_coverage_available"] is False
    assert fresh["polymarket_observed_at"] is fresh["bookmaker_observed_at"] is None
    assert fresh["trading_enabled"] is False
    assert data["prediction_count"] == 13 and data["current_prediction_count"] == 0
    assert data["predictions"][0]["pick_filter_reason"] == "Current bookmaker coverage unavailable"
    assert "no current bookmaker inputs" not in " ".join(fresh["warnings"])
    assert "Latest consumer cycle failed" in " ".join(fresh["warnings"])
    runtime = view.data.runtime_status()
    assert runtime["ready"] is runtime["trading_enabled"] is False
    assert "Latest consumer cycle failed" in runtime["components"]["betting_loop"]["message"]


def test_order_blocker_cannot_promote_stale_or_conflicting_observations(view):
    view.report["cycle"]["blocking_conditions"] = ["original_order_unresolved"]
    view.report["cycle"]["accepted_generation"] = "other"
    data = view.data.predictions()
    assert data["freshness"]["state"] == "degraded"
    assert data["freshness"]["is_stale"] is True and data["freshness"]["trading_enabled"] is False
    assert data["current_prediction_count"] == data["matched_market_count"] == 0
    assert data["cache_status"] == "stale"
    assert "stale or unavailable" in view.data.runtime_status()["components"]["betting_loop"]["message"]


def test_successful_empty_market_observations_are_distinct_from_failed_acquisition(view):
    view.report["cycle"].update(predictions=[], mapped_markets=[], prediction_count=0)
    data = view.data.predictions()
    assert data["freshness"]["state"] == "current"
    assert data["freshness"]["bookmaker_coverage_available"] is True
    assert data["freshness"]["polymarket_coverage_available"] is True
    assert data["current_prediction_count"] == data["matched_market_count"] == 0
    assert data["freshness"]["warnings"] == ["13 bouts have no current bookmaker inputs; saved predictions are shown"]
