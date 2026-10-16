"""Real consumer wiring with offline exchange doubles and isolated accounting."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src import refresh_live, live_control
from src.polymarket import executor as executor_module, tracker
from src.strategy import duo_trader
from src.strategy.bankroll import BankrollManager


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("network forbidden"))
    monkeypatch.setenv("UFC_REFRESH_TRADING_STATE_DIR", str(tmp_path / "controls"))
    monkeypatch.setenv("LIVE_TRADING_MODE", "real")
    monkeypatch.setenv("LIVE_TRADING_ARMED", "1")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMATION", "REAL_TRADING_ENABLED")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "offline-fixture")
    monkeypatch.setenv("ODDS_API_KEY", "offline-fixture")
    monkeypatch.setenv("WEB_DASHBOARD_TOKEN", "offline-fixture")
    monkeypatch.setattr(refresh_live, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(live_control, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(tracker, "LEDGER_PATH", tmp_path / "bet_ledger.json")
    for name, filename in (("SINGLE_LEDGER", "single"), ("CONVICTION_LEDGER", "conviction"),
                           ("MODEL_TRACKER_LEDGER", "model_tracker"), ("LEGACY_G_TRACKER_LEDGER", "gemini_tracker")):
        monkeypatch.setattr(duo_trader, name, tmp_path / f"bet_ledger_{filename}.json")


def intent(path, *, order_id=None, state="unknown", token="alpha-token", paper=False):
    ledger = tracker.BetLedger(path)
    row = ledger.add_bet(fighter="Alpha Fighter", opponent="Beta Fighter", side="a",
        amount=5, price=.5, shares=10, token_id=token, market_id="fixture-market",
        model_prob=.8, market_prob=.5, edge=.3, decimal_odds=2, dry_run=paper,
        order_type="marketable_limit", placement_state=state, order_id=order_id,
        event_date=(datetime.now(timezone.utc) + timedelta(hours=24)).isoformat())
    return row


@pytest.mark.parametrize("gate", ["LIVE_TRADING_ARMED", "LIVE_TRADING_CONFIRMATION", "WEB_DASHBOARD_TOKEN"])
def test_real_controls_reject_before_publication_or_client(monkeypatch, tmp_path, gate):
    model = tmp_path / "accepted.pkl"
    model.write_bytes(b"fixture model")
    monkeypatch.setenv("LIVE_MODEL", str(model))
    monkeypatch.delenv(gate)
    monkeypatch.setattr(refresh_live, "_run_cycle", lambda **kw: pytest.fail("read before controls"))
    with pytest.raises(RuntimeError, match="Real-money live trading blocked"):
        refresh_live.run_refresh_live_cycle(root=tmp_path, baseline=tmp_path, dry_run=False)


def test_live_model_must_match_final_published_primary(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(b"accepted bytes")
    snapshot = SimpleNamespace(snapshot={"models": {"primary": {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}}})
    refresh_live._assert_selected_live_model(snapshot, str(path))
    path.write_bytes(b"different bytes")
    with pytest.raises(RuntimeError, match="final accepted primary"):
        refresh_live._assert_selected_live_model(snapshot, str(path))


def test_enabled_original_unknown_id_recovers_without_replacing_or_refunding(tmp_path):
    path = tmp_path / "bet_ledger_single.json"
    original = intent(path, order_id="original-exchange-order")
    calls = []
    client = SimpleNamespace(get_open_orders=lambda: [], get_order=lambda order_id: calls.append(order_id) or {
        "id": order_id, "status": "CANCELED", "size_matched": 0, "original_size": 10})
    result = refresh_live._maintain_original_accounting(client)
    row = tracker.BetLedger(path).bets[0]
    assert calls == ["original-exchange-order"]
    assert row["id"] == original["id"] and row["order_id"] == "original-exchange-order"
    assert row["status"] == "cancelled"
    assert result["recovered_original_submissions"] == 1
    assert result["manual_positions_adopted"] == 0


def test_unknown_no_id_preserves_uncertainty_and_never_adopts_manual_replacement(tmp_path):
    path = tmp_path / "bet_ledger_single.json"
    original = intent(path, state="pending_submit")
    client = SimpleNamespace(get_open_orders=lambda: [{"id": "manual-order", "asset_id": "alpha-token",
                                                       "price": .5, "original_size": 10}])
    with pytest.raises(RuntimeError, match="ownership/fill remains unresolved"):
        refresh_live._maintain_original_accounting(client)
    row = tracker.BetLedger(path).bets[0]
    assert row["id"] == original["id"] and row["order_id"] is None
    assert row["placement_state"] == "unknown" and row["status"] == "open"


def test_dormant_owner_uncertain_intent_is_read_only_and_blocks(tmp_path):
    path = tmp_path / "bet_ledger_trader_b.json"
    intent(path, order_id="dormant-original", state="unknown")
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="Dormant original owner"):
        refresh_live._maintain_original_accounting(SimpleNamespace())
    assert path.read_bytes() == before


@pytest.mark.parametrize("mode", [None, 0, 1, "false"])
def test_unclassified_original_active_modes_are_blocked(tmp_path, mode):
    path = tmp_path / "bet_ledger_trader_b.json"
    path.write_text(json.dumps({"bets": [{"id": 1, "status": "open", "dry_run": mode}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unclassified real/paper"):
        refresh_live.original_accounting_ledgers()


def test_submitted_original_fill_is_recovered_before_new_orders(monkeypatch, tmp_path):
    path = tmp_path / "bet_ledger_single.json"
    original = intent(path, order_id="submitted-original", state="submitted")
    monkeypatch.setattr(tracker, "auto_settle_from_polymarket", lambda *a, **kw: 0)
    client = SimpleNamespace(get_open_orders=lambda: [], get_order=lambda order_id: {
        "id": order_id, "status": "MATCHED", "size_matched": 7.25, "original_size": 10},
        get_trades=lambda **kw: [{"id": "observed-fill", "status": "CONFIRMED", "asset_id": "alpha-token",
            "taker_order_id": "submitted-original", "price": .5, "size": 7.25}])
    result = refresh_live._maintain_original_accounting(client)
    row = tracker.BetLedger(path).bets[0]
    assert row["id"] == original["id"] and row["order_id"] == "submitted-original"
    assert row["actual_filled_shares"] == 7.25 and row["actual_fill_amount"] == 3.62
    assert row["placement_state"] == "filled"
    assert result["recovered_original_submissions"] == 1


def test_only_enabled_confirmed_real_fills_can_settle(monkeypatch, tmp_path):
    enabled = tmp_path / "bet_ledger_single.json"
    real = intent(enabled, order_id="actual-filled", state="filled")
    tracker.BetLedger(enabled).update_bet_fields(real["id"], actual_filled_shares=7.25, actual_fill_amount=3.625)
    paper = intent(enabled, state="filled", paper=True)
    legacy = intent(enabled, state="filled")  # requested quantity alone does not prove a fill
    dormant = tmp_path / "bet_ledger_trader_b.json"
    b = intent(dormant, state="filled")
    tracker.BetLedger(dormant).update_bet_fields(b["id"], actual_filled_shares=10, actual_fill_amount=5)
    before = dormant.read_bytes()
    seen = []
    def settle(view, **kw):
        rows = view.get_open_bets(fresh=True)
        seen.extend(row["id"] for row in rows)
        for row in rows:
            view.settle_bet(row["id"], False)
        return len(rows)
    monkeypatch.setattr(tracker, "auto_settle_from_polymarket", settle)
    result = refresh_live._maintain_original_accounting(SimpleNamespace())
    rows = tracker.BetLedger(enabled).bets
    assert seen == [real["id"]]
    assert rows[0]["status"] == "lost" and rows[0]["result_pnl"] == -3.62
    assert rows[1]["id"] == paper["id"] and rows[1]["status"] == "open"
    assert rows[2]["id"] == legacy["id"] and rows[2]["status"] == "open"
    assert dormant.read_bytes() == before
    assert result["settled_original_real_positions"] == 1


def market(when):
    return {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter", "event_date": when,
            "token_id_yes": "alpha-token", "token_id_no": "beta-token", "market_id": "fixture-market",
            "condition_id": "fixture-condition", "price_yes": .5, "price_no": .5,
            "tick_size": "0.01", "neg_risk": False, "active": True, "closed": False,
            "accepting_orders": True}


def test_wallet_assertion_never_adopts_or_erases_manual_positions(monkeypatch, tmp_path):
    from src.polymarket import monitor
    monkeypatch.setattr(monitor, "PositionMonitor", lambda **kw: SimpleNamespace(wallet_address="fixture-wallet"))
    monkeypatch.setattr(executor_module, "_fetch_wallet_positions_for_reconciliation",
                        lambda wallet: [{"asset": "alpha-token", "size": 10, "avgPrice": .5}])
    with pytest.raises(RuntimeError, match="no original owner evidence"):
        executor_module.assert_live_wallet_exposure_synced(pd.DataFrame([market("2099-10-17T21:00:00Z")]),
                                                         clob_client=SimpleNamespace())
    assert not (tmp_path / "bet_ledger_single.json").exists()
    path = tmp_path / "bet_ledger_single.json"
    intent(path, order_id="actual-order", state="filled")
    before = path.read_bytes()
    monkeypatch.setattr(executor_module, "_fetch_wallet_positions_for_reconciliation", lambda wallet: [])
    with pytest.raises(RuntimeError, match="settlement/sale evidence required"):
        executor_module.assert_live_wallet_exposure_synced(pd.DataFrame([market("2099-10-17T21:00:00Z")]),
                                                         clob_client=SimpleNamespace())
    assert path.read_bytes() == before


def test_low_cash_real_cycle_uses_original_portfolio_without_submission(monkeypatch, tmp_path):
    from src.data import refresh_service, refresh_trading, odds_client, line_tracker
    from src.polymarket import markets, monitor
    from src import bot
    when = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    day = refresh_live._event_date(when)
    model = tmp_path / "accepted.pkl"
    model.write_bytes(b"offline accepted model")
    monkeypatch.setenv("LIVE_MODEL", str(model))
    consensus = {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter", "commence_time": when,
                 "event_id": "observed-fixture", "a_fair_prob_avg": .8, "b_fair_prob_avg": .2}
    class Consumer:
        def __init__(self, service):
            self.rows = [{**consensus, "event_date": day}]
            self.publication = {"generation_id": "final-original", "job_id": "original-job"}
            self.report = {"event_date": day}
            self.binding = {"accepted_generation": "final-original"}
            self.snapshot = {"models": {"primary": {"bytes": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}}}
        def read(self): return self
        def score(self, fights, **kw):
            return [{**consensus, "event_date": when, "official_event_date": day,
                "prob_a": .8, "prob_b": .2, "no_odds_prob_a": .8, "no_odds_prob_b": .2,
                "fighter_a_id": "0000000000000001", "fighter_b_id": "0000000000000002",
                "model_features": {}, "feature_provenance": {}, "a_num_fights": 10, "b_num_fights": 10,
                "market_inputs_available": True}]
    class Odds:
        def get_live_odds(self): return [{"observed": True}]
        def odds_to_dataframe(self, rows): return rows
        def get_consensus_odds(self, rows): return pd.DataFrame([consensus])
    monkeypatch.setattr(refresh_service, "RefreshService", lambda **kw: kw)
    monkeypatch.setattr(refresh_trading, "PublishedTradingSnapshot", Consumer)
    monkeypatch.setattr(odds_client, "OddsClient", Odds)
    monkeypatch.setattr(markets, "get_ufc_fight_markets", lambda **kw: pd.DataFrame([market(when)]))
    monkeypatch.setattr(bot, "_live_prediction_quality_assessment", lambda *a, **kw: {"reasons": []})
    monkeypatch.setattr(bot, "_withhold_unconfirmed_identity", lambda row: [])
    monkeypatch.setattr(line_tracker, "detect_injury_or_cancellation", lambda *a, **kw: {"severity": "none"})
    monkeypatch.setattr(monitor, "PositionMonitor", lambda **kw: SimpleNamespace(wallet_address="fixture-wallet"))
    monkeypatch.setattr(executor_module, "_fetch_wallet_positions_for_reconciliation", lambda wallet: [])
    client = SimpleNamespace(get_open_orders=lambda: [],
        get_clob_market_info=lambda cid: {"mts": "0.01", "nr": False, "t": [{"t": "alpha-token"}, {"t": "beta-token"}]},
        create_limit_order=lambda **kw: pytest.fail("insufficient cash must never submit"))
    for path in (duo_trader.SINGLE_LEDGER, duo_trader.CONVICTION_LEDGER, duo_trader.MODEL_TRACKER_LEDGER):
        path.write_text(json.dumps({"bets": []}), encoding="utf-8")
    originals = {path: path.read_bytes() for path in tmp_path.glob("bet_ledger*.json")}
    result = refresh_live.run_refresh_live_cycle(root=tmp_path, baseline=tmp_path, dry_run=False,
        clob_client=client, readonly_clob=SimpleNamespace(),
        bankroll_basis=duo_trader.WalletBankrollBasis(.50334, .50334, "confirmed offline account"))
    assert result["trading_enabled"] is True and result["orders_disabled"] is False
    assert result["real_submissions"] == result["total_orders"] == 0
    assert result["current_card"]["bet_window_open_count"] == 1
    assert result["trader_s"]["allocation"] == .50334
    assert all(path.read_bytes() == original for path, original in originals.items())


@pytest.mark.parametrize("placement", ["primary", "near_miss"])
def test_pause_received_after_signing_aborts_post_and_preserves_id(monkeypatch, tmp_path, placement):
    from src.trading_controls import set_paused
    from test_trading_order_recovery import sdk_wrapper, candidate
    wrapper, sdk = sdk_wrapper(tmp_path)
    executor = executor_module.OrderExecutor(BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=wrapper, dry_run=False, skip_wallet_conflict_check=True)
    executor.ledger = tracker.BetLedger(tmp_path / "bet_ledger_single.json")
    executor._check_liquidity = lambda *a, **kw: {"ok": True, "adjusted_size": 25,
        "available_liquidity": 100, "best_ask": .5, "slippage": 0, "reason": ""}
    executor._authoritative_wallet_conflict = lambda **kw: (False, "")
    original_record = executor._record_prepared_order_id
    def pause_after_journal(row, order_id):
        original_record(row, order_id)
        set_paused(tmp_path / "controls", True)
    executor._record_prepared_order_id = pause_after_journal
    bet = candidate()
    if placement == "near_miss":
        bet.update(pd.Series({"model_prob": .51, "blended_prob": .51, "edge": .01}))
    result = (executor._place_near_miss_limit(bet, pd.DataFrame()) if placement == "near_miss"
              else executor._place_bet(bet, pd.DataFrame()))
    assert sdk.posts == [] and len(sdk.created) == 1
    assert result["status"] == "failed"
    row = tracker.BetLedger(executor.ledger.path).bets[0]
    assert row["order_id"].startswith("0x") and row["placement_state"] == "failed"
    assert row["status"] == "cancelled"
    assert executor.bankroll.available_cash == 500
