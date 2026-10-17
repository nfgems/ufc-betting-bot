from pathlib import Path
import json
import sys
from types import SimpleNamespace

from flask import Flask
import pytest

from src.web.refresh_dashboard import install_dashboard, _public_payload, _execution_matrix


class DashboardFixture:
    def __init__(self, service, trading):
        self.service, self.trading = service, trading

    def freshness(self):
        return {"processed_data_through": "2026-10-10", "prediction_generated_at": "2026-10-11T04:08:00+00:00",
                "next_refresh_at": "2026-10-13T15:00:00+00:00", "state": "waiting"}

    def runtime_status(self):
        return {"ok": True, "ready": True, "effective_live_mode": "real", "trading_enabled": True,
                "components": {}, "updated_at": "2026-10-11T16:15:44+00:00"}

    def predictions(self, include_global_feature_importance=False):
        return {"predictions": [], "prediction_count": 0, "timestamp": "2026-10-11T04:08:00+00:00"}

    def upcoming_events(self):
        return [{"event": "UFC fixture", "date": "2026-10-17", "fight_count": 13, "sport": "ufc"}]

    def activity(self, limit=50, sport="all"):
        return [{"timestamp": "2026-10-11 16:15:44", "level": "INFO", "source": "runtime.refresh_consumer",
                 "message": "Completed actual saved cycle; waiting", "sport": "ufc"}]


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv("UFC_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("WEB_DASHBOARD_TOKEN", "private-fixture-token")
    monkeypatch.setitem(sys.modules, "src.web.refresh_dashboard_data", SimpleNamespace(DashboardData=DashboardFixture))
    source = Path(__file__).resolve().parents[1] / "src" / "web"
    app = Flask(__name__, template_folder=str(source / "templates"), static_folder=str(source / "static"))
    app.config["TESTING"] = True
    public = install_dashboard(app, SimpleNamespace(), SimpleNamespace(status=lambda: {"trading_enabled": True}))
    legacy = app.extensions["refresh_dashboard"]["legacy"]
    legacy._load_live_pnl_snapshot = lambda: ({"positions": [], "closed_positions": []}, "unavailable")
    return app, app.test_client(), legacy, public, tmp_path


@pytest.mark.parametrize("path,marker", [
    ("/", b"Fight Decisions"), ("/ufc", b"Fight Decisions"),
    ("/predictions", b"Predictions"), ("/activity", b"activityList"),
    ("/bet-history", b"Bet History"), ("/execution-breakdown", b"Execution"),
])
def test_original_pages_render_publicly_without_initializing_exchange_client(dashboard, monkeypatch, path, marker):
    _app, client, legacy, public, _logs = dashboard
    from src.polymarket import readonly_account
    monkeypatch.setattr(readonly_account, "ReadOnlyClobClient", lambda: pytest.fail("page initialized account client"))
    response = client.get(path)
    assert path in public
    assert response.status_code == 200 and marker in response.data
    assert response.headers["Cache-Control"].startswith("no-store")
    assert b"type='password'" not in response.data and b'Dashboard token' not in response.data
    assert legacy._clob_client._client is None


@pytest.mark.parametrize("scenario", ["evaluating", "initial", "paused"])
def test_initial_page_context_distinguishes_verified_evaluation_from_startup_and_pause(dashboard, monkeypatch, scenario):
    app, client, _legacy, _public, _logs = dashboard
    data = app.extensions["refresh_dashboard"]["data"]
    monkeypatch.setattr(data, "freshness", lambda: {"state": "evaluating", "consumer_evaluating": scenario != "initial"})
    data.trading.status = lambda: {"status": "evaluating", "ready": False, "trading_enabled": False,
                                  "orders_disabled": False, "paused": scenario == "paused"}
    phrase = (b"UFC consumer is checking current data; the last completed cycle is shown." if scenario == "evaluating" else
              b"UFC trading is paused." if scenario == "paused" else b"UFC trading is off or waiting for verified startup.")
    for path in ("/", "/ufc", "/predictions", "/activity", "/bet-history", "/execution-breakdown"):
        response = client.get(path)
        assert response.status_code == 200 and phrase in response.data
    assert data.trading.status()["ready"] is False and data.trading.status()["trading_enabled"] is False


def test_activity_merges_original_log_and_current_saved_cycle_without_inventing_time(dashboard):
    _app, client, _legacy, _public, logs = dashboard
    (logs / "bot.log").write_text("2026-10-10 12:00:00,123 [INFO] src.bot: Earlier real activity\n")
    feed = client.get("/api/bot-activity-snapshot?limit=10").get_json()
    assert feed["entry_count"] == 2
    assert feed["entries"][0]["message"] == "Earlier real activity"
    assert feed["entries"][-1]["timestamp"] == "2026-10-11 16:15:44"
    assert feed["last_entry"] == "2026-10-11 16:15:44"


def test_original_real_history_retains_dormant_b_and_excludes_paper_without_writes(dashboard):
    _app, client, legacy, _public, logs = dashboard
    rows = [{"id": 1, "fighter": "A", "opponent": "B", "status": "open", "dry_run": False,
             "amount": 5, "shares": 10, "price": .5, "placed_at": "2026-06-01T00:00:00+00:00",
             "token_id": "private-token", "order_id": "original-order"},
            {"id": 2, "status": "open", "dry_run": True, "amount": 99}]
    original = logs / "bet_ledger_trader_b.json"
    original.write_text(json.dumps({"bets": rows}))
    pin = original.read_bytes(), original.stat().st_mtime_ns
    response = client.get("/api/bets")
    assert response.status_code == 200
    assert len(response.json["all"]) == 1
    assert response.json["all"][0]["trader"] == "B"
    assert "token_id" not in response.json["all"][0] and "order_id" not in response.json["all"][0]
    assert "_ledger_path" not in response.json["all"][0]
    legacy._load_live_pnl_snapshot = lambda: ({"positions": [], "closed_positions": []}, "live")
    assert client.get("/api/open-bets-enriched").json["bets"] == []
    assert (original.read_bytes(), original.stat().st_mtime_ns) == pin


def test_original_gets_disable_all_ledger_cleanup_and_never_register_legacy_mutations(dashboard, monkeypatch):
    app, client, legacy, public, _logs = dashboard
    monkeypatch.setattr(legacy, "_compute_limit_orders_display", lambda: [])
    assert client.get("/api/open-limit-orders").status_code == 200
    assert legacy.auto_reconcile_sold_positions(None, set()) == 0
    assert legacy._kickoff_limit_order_reconcile(set()) is None
    assert legacy.maybe_prune_alert_store(None, 0) is None
    assert legacy._maybe_prune_upcoming_event_snapshots(None) is None
    assert legacy._profile_activity_loss_rows({}, active_token_ids=set(), represented_token_ids=set()) == []
    assert legacy._match_profile_ledger_bets(token_id="same-token") == []
    assert client.get("/api/bot-alerts").status_code == 200
    for path in ("/api/reconcile-limit-orders", "/api/settle-auto", "/api/redeem-auto", "/api/refresh-prices"):
        assert path not in public and client.post(path).status_code == 404
    assert all("POST" not in rule.methods for rule in app.url_map.iter_rules())


def test_public_outputs_scrub_secrets_and_keep_private_module_state_per_app(dashboard, monkeypatch):
    app, client, legacy, _public, logs = dashboard
    monkeypatch.setattr(legacy, "_load_market_intel_artifact", lambda: {
        "alerts": [{"message": "private-fixture-token", "token_id": "private"}], "ip": "private-ip"})
    payload = client.get("/api/injury-alerts").json
    assert payload == {"alerts": [{"message": "[redacted]"}]}
    other = Flask("other")
    install_dashboard(other, SimpleNamespace(), SimpleNamespace(status=lambda: {}))
    assert app.extensions["refresh_dashboard"]["legacy"] is not other.extensions["refresh_dashboard"]["legacy"]


@pytest.mark.parametrize("source", ["live", "unavailable"])
def test_wallet_roi_never_uses_mixed_history_and_unknown_account_is_not_zero(dashboard, monkeypatch, source):
    from flask import jsonify
    _app, client, legacy, _public, _logs = dashboard
    payload = {"roi": 11.0, "total_pnl": 22.0, "realized_pnl": 20.0,
               "unrealized_pnl": 2.0, "positions_value": 1.0, "open_invested": 1.0,
               "open_bets": 1, "_pnl_source": source, "_profile_source": "unavailable"}
    monkeypatch.setattr(legacy, "api_summary", lambda: jsonify(payload))
    monkeypatch.setattr(legacy, "_load_live_pnl_snapshot", lambda: ({"positions": [], "closed_positions": []}, source))
    actual = client.get("/api/summary").json
    assert actual["roi"] is None
    assert actual["original_real_ledger_summary"]["total_bets"] == 0
    if source == "live":
        assert actual["total_pnl"] == 22 and actual["positions_value"] == 1
    else:
        assert actual["total_pnl"] is actual["positions_value"] is actual["open_bets"] is None


@pytest.mark.parametrize("confirmed", [False, True])
def test_fee_accounting_is_preserved_in_active_public_reads_without_private_evidence(dashboard, monkeypatch, confirmed):
    from flask import jsonify
    from src.polymarket.tracker import BetLedger
    from src.polymarket.fill_cashflow import fill_binding, observed_cash_debit
    _app, client, legacy, _public, logs = dashboard
    ledger = BetLedger(logs / "bet_ledger_single.json")
    row = ledger.add_bet(fighter="Fixture A", opponent="Fixture B", side="a", amount=1,
        price=.5, shares=2, token_id="private-fee-token", market_id="fixture-market",
        model_prob=.8, market_prob=.5, edge=.3, decimal_odds=2, dry_run=False,
        order_type="filled_limit", placement_state="filled", order_id="private-fee-order")
    ledger.update_bet_fields(row["id"], actual_fill_amount=1, actual_filled_shares=2,
        actual_fill_price=.5, actual_fill_source="clob_trade_history",
        actual_fill_trade_ids=["private-fee-trade"], cur_price=.5)
    row = ledger.get_bets(fresh=True)[0]
    if confirmed:
        fields = observed_cash_debit(row, {"actual_fill_trade_components": [{
            "trade_id":"private-fee-trade", "order_id":"private-fee-order", "token_id":"private-fee-token",
            "role":"taker", "side":"BUY", "shares":"2", "price":".5", "transaction_hash":"private-fee-tx"}]},
            [{"type":"TRADE", "side":"BUY", "asset":"private-fee-token", "transactionHash":"private-fee-tx",
              "size":"2", "price":".5", "usdcSize":"1.02345", "timestamp":1792100000}])
        assert ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=fields).ok
    pin = ledger.path.read_bytes(), ledger.path.stat().st_mtime_ns
    monkeypatch.setattr(legacy, "api_summary", lambda: jsonify({"total_pnl":22,
        "_pnl_source":"live", "_profile_source":"live"}))
    monkeypatch.setattr(legacy, "api_profile_bets", lambda: jsonify({"bets":[], "summary":{
        "total_pnl":22, "_pnl_source":"live", "_profile_source":"live"}}))
    for url in ("/api/summary", "/api/profile-bets"):
        payload = client.get(url).json
        summary = payload if url.endswith("summary") else payload["summary"]
        assert summary["total_pnl"] == 22  # Independent observed wallet total retained.
        accounting = summary["original_real_ledger_summary"]
        assert accounting["cash_cost_basis_complete"] is confirmed
        if confirmed:
            assert accounting["open_invested"] == 1.02345
            assert accounting["total_pnl"] == pytest.approx(-.02345)
        else:
            assert accounting["open_invested"] is accounting["total_pnl"] is None
            assert accounting["open_cash_cost_unknown_count"] == 1
    breakdown = client.get("/api/trader-breakdown").json
    single = next(item for item in breakdown if item["trader"] == "S")
    assert single["cash_cost_basis_complete"] is confirmed
    if not confirmed:
        assert single["total_pnl"] is None
    public_bets = client.get("/api/bets").data.decode()
    assert "private-fee-" not in public_bets
    assert "actual_cash_debit_evidence" not in public_bets
    if confirmed:
        assert '1.02345' in public_bets
    assert (ledger.path.read_bytes(), ledger.path.stat().st_mtime_ns) == pin


def test_unavailable_profile_keeps_history_separate_and_does_not_invent_account_pnl(dashboard, monkeypatch):
    from flask import jsonify
    _app, client, legacy, _public, _logs = dashboard
    monkeypatch.setattr(legacy, "api_profile_bets", lambda: jsonify({"bets": [], "summary": {
        "roi": 0, "total_pnl": 0, "realized_pnl": 0, "unrealized_pnl": 0,
        "open_invested": 0, "positions_value": 0, "_pnl_source": "unavailable", "_profile_source": "unavailable"}}))
    actual = client.get("/api/profile-bets").json["summary"]
    assert actual["roi"] is actual["total_pnl"] is actual["realized_pnl"] is None
    assert actual["original_real_ledger_summary"]["total_bets"] == 0


def test_public_identifier_aliases_are_redacted_and_display_id_is_retained(monkeypatch):
    monkeypatch.setenv("POLYMARKET_FUNDER_ADDRESS", "0xABCdef")
    private = {key: "private" for key in (
        "orderId", "order_ids", "trade_id", "tradeIds", "takerOrderId", "maker_address", "taker_address",
        "ownerAddress", "transaction_hash", "txHash", "privateKey", "api_secret", "apiPassphrase", "signature")}
    assert _public_payload({"id": 3, "nested": [private], "message": "wallet 0xabcdef"}) == {
        "id": 3, "nested": [{}], "message": "wallet [redacted]"}


@pytest.mark.parametrize("source", ["live", "stale"])
def test_resolved_position_cost_is_not_open_exposure_and_counts_follow_declared_ledger_scope(dashboard, monkeypatch, source):
    from flask import jsonify
    _app, client, legacy, _public, logs = dashboard
    real = {"id": 1, "status": "won", "dry_run": False, "amount": 10, "shares": 20,
            "price": .5, "result_pnl": 10, "settled_at": "2026-01-01T00:00:00+00:00"}
    (logs / "bet_ledger_single.json").write_text(json.dumps({"bets": [real, {**real, "id": 2, "dry_run": True}]}))
    raw = {"positions": [{"size": 24.45, "invested": 11.9804, "value": 0,
                          "redeemable": True, "sport": "ufc"}], "closed_positions": [],
           "total_invested": 11.9804, "total_pnl": -11.9804, "num_positions": 1, "num_closed": 749}
    monkeypatch.setattr(legacy, "_load_live_pnl_snapshot", lambda: (raw, source))
    monkeypatch.setattr(legacy, "api_summary", lambda: jsonify({
        "open_bets": 0, "open_invested": 11.9804, "total_pnl": -11.9804,
        "total_bets": 749, "settled_bets": 749, "wins": 600, "losses": 149,
        "win_rate": .8, "total_wagered": 999, "_pnl_source": source, "_profile_source": "unavailable"}))
    result = client.get("/api/summary").json
    assert result["open_bets"] == 0 and result["open_invested"] == 0
    assert result["total_pnl"] == -11.9804
    assert result["_counts_scope"] == "original_real_ledger_history"
    assert {key: result[key] for key in ("total_bets", "settled_bets", "wins", "losses", "win_rate", "total_wagered")} == {
        "total_bets": 1, "settled_bets": 1, "wins": 1, "losses": 0, "win_rate": 1, "total_wagered": 10}


def test_missing_wallet_snapshot_keeps_active_exposure_unknown_even_with_observed_profile_pnl(dashboard, monkeypatch):
    from flask import jsonify
    _app, client, legacy, _public, _logs = dashboard
    monkeypatch.setattr(legacy, "api_summary", lambda: jsonify({
        "open_bets": 3, "open_invested": 11.98, "total_pnl": 100,
        "_pnl_source": "unavailable", "_profile_source": "live"}))
    summary = client.get("/api/summary").json
    assert summary["open_bets"] is summary["open_invested"] is None
    assert summary["total_pnl"] == 100
    monkeypatch.setattr(legacy, "api_profile_bets", lambda: jsonify({"summary": {
        "open_bets": 3, "open_invested": 11.98, "total_pnl": 100,
        "_pnl_source": "unavailable", "_profile_source": "live"}, "bets": []}))
    profile = client.get("/api/profile-bets").json["summary"]
    assert profile["open_bets"] is profile["open_invested"] is None
    assert profile["total_pnl"] == 100


@pytest.fixture
def current_execution(tmp_path):
    names = {"fighter_a": "Melissa Croden", "fighter_b": "Chelsea Chandler", "card_date": "2026-10-17"}
    no_input = {"fighter_a": "Missing Odds", "fighter_b": "Opponent", "card_date": "2026-10-17"}
    cell = {"status": "no_signal", "text": "No conviction signal"}
    baseline = {"fights": [{**names, **{code: dict(cell) for code in ("S", "C", "M")}},
                           {**no_input, **{code: dict(cell) for code in ("S", "C", "M")}}], "count": 2}
    publication = {"generation_id": "accepted", "job_id": "publication", "prediction_report": {"sha256": "report-pin"}}
    saved = {"accepted_generation": "accepted", "predictions": [names, no_input]}
    actual = {**names, "fighter_a": "Chelsea Chandler", "fighter_b": "Melissa Croden", "paths": {
        "C": {"trader": "C", "status": "outside_window", "gate": "bet_window",
              "explanation": "Bet window opens at 2026-10-15T21:00:00+00:00",
              "numbers": {"bet_on": "Melissa Croden", "bet_side": "b", "model_probability": .733516,
                          "market_probability": .70, "edge": .033516}, "order": {}}}}
    audit = {"cycle_id": "exact-cycle", "completed_at": "2026-10-11T17:01:00+00:00", "fight_count": 1,
             "dry_run": False, "fights": [actual]}
    report = {"effective_live_mode": "real", "cycle": {
        "accepted_generation": "accepted", "publication_job_id": "publication", "snapshot_binding": {"final_report_sha256": "report-pin"},
        "predictions": [names], "execution_audit": {key: audit[key] for key in ("cycle_id", "completed_at", "fight_count")}}}
    (tmp_path / "execution_decision_audit_latest.json").write_text(json.dumps(audit))
    service = SimpleNamespace(published_pair=lambda: (publication, {}, saved))
    trading = SimpleNamespace(report=lambda: report)
    return baseline, service, trading, tmp_path, report, audit


def test_current_matrix_preserves_exact_execution_path_and_quote_across_reversed_display_names(current_execution):
    baseline, service, trading, logs, _report, audit = current_execution
    result = _execution_matrix(baseline, service, trading, logs)
    actual = result["fights"][0]["C"]
    assert actual["status"] == "outside_window" and actual["gate"] == "bet_window"
    assert actual["rationale"] == audit["fights"][0]["paths"]["C"]["explanation"]
    assert actual["numbers"] == audit["fights"][0]["paths"]["C"]["numbers"]
    assert actual["execution_fighter_a"] == "Chelsea Chandler" and actual["text"] == "Waiting for bet window"
    assert actual["bet_placed"] is None
    assert result["fights"][1] == baseline["fights"][1] and result["count"] == 2
    assert baseline["fights"][0]["C"]["status"] == "no_signal"


@pytest.mark.parametrize("conflict", ["generation", "job", "report_pin", "pointer_id", "completion", "count", "paper", "missing_mode", "card_date", "duplicate"])
def test_current_matrix_refuses_mismatched_or_ambiguous_execution_evidence(current_execution, conflict):
    baseline, service, trading, logs, report, audit = current_execution
    if conflict == "generation": report["cycle"]["accepted_generation"] = "other"
    elif conflict == "job": report["cycle"]["publication_job_id"] = "other"
    elif conflict == "report_pin": report["cycle"]["snapshot_binding"]["final_report_sha256"] = "other"
    elif conflict == "pointer_id": report["cycle"]["execution_audit"]["cycle_id"] = "other"
    elif conflict == "completion": audit["completed_at"] = "2026-10-10T17:01:00+00:00"
    elif conflict == "count": audit["fight_count"] = 2
    elif conflict == "paper": audit["dry_run"] = True
    elif conflict == "missing_mode": report.pop("effective_live_mode")
    elif conflict == "card_date": audit["fights"][0]["card_date"] = "2026-10-18"
    else:
        audit["fights"].append(dict(audit["fights"][0]))
        audit["fight_count"] = report["cycle"]["execution_audit"]["fight_count"] = 2
    (logs / "execution_decision_audit_latest.json").write_text(json.dumps(audit))
    actual = _execution_matrix(baseline, service, trading, logs)
    assert actual["execution_decisions_available"] is False
    assert actual["fights"] == baseline["fights"]
