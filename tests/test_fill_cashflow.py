"""Synthetic entry fee evidence and original-ledger wiring. Network is forbidden."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from types import SimpleNamespace
import pytest

from src.polymarket import client as client_module, tracker
from src.polymarket.executor import OrderExecutor
from src.polymarket.fill_cashflow import (
    CashDebitUnavailable, fill_binding, observed_cash_debit, verified_cash_debit,
    verified_fill_shares,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("network forbidden"))
    monkeypatch.setattr(tracker, "LEDGER_PATH", tmp_path / "unused.json")


def original(path):
    ledger = tracker.BetLedger(path)
    row = ledger.add_bet(fighter="Alpha", opponent="Beta", side="a", amount=1,
        price=.5, shares=2, token_id="token-original", market_id="market",
        model_prob=.8, market_prob=.5, edge=.3, decimal_odds=2, dry_run=False,
        order_type="filled_limit", placement_state="filled", order_id="order-original",
        event_date=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
    ledger.update_bet_fields(row["id"], actual_fill_amount=1, actual_filled_shares=2,
        actual_fill_price=.5, actual_fill_source="clob_trade_history", actual_fill_trade_ids=["trade-original"])
    return ledger, ledger.get_bets(fresh=True)[0]


def trades():
    return [{"id": "trade-original", "status": "CONFIRMED", "side": "BUY",
             "asset_id": "token-original", "taker_order_id": "order-original",
             "size": "2", "price": "0.5", "transaction_hash": "0xoriginal", "fee_rate_bps": "0"}]


def activity():
    return [{"type": "TRADE", "side": "BUY", "asset": "token-original",
             "transactionHash": "0xoriginal", "size": "2", "price": "0.5",
             "usdcSize": "1.02", "timestamp": 1792100000}]


def evidence(row, *, trade_rows=None, activity_rows=None):
    executor = OrderExecutor(SimpleNamespace(), clob_client=SimpleNamespace(
        get_trades=lambda **kw: trades() if trade_rows is None else trade_rows), dry_run=False)
    found, order = executor._lookup_order_from_trade_history(row, row["order_id"])
    assert found
    return observed_cash_debit(row, order, activity() if activity_rows is None else activity_rows)


@pytest.mark.parametrize("won, expected", [(True, .98), (False, -1.02)])
def test_exact_cash_debit_drives_future_settlement_preserving_gross(tmp_path, won, expected):
    ledger, row = original(tmp_path / "ledger.json")
    fields = evidence(row)
    assert fields["actual_additional_charge"] == "0.02"  # trade fee_rate_bps=0 is ignored
    assert ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=fields).ok
    saved = ledger.get_bets(fresh=True)[0]
    assert fill_binding(saved) == fill_binding(row)
    assert verified_cash_debit(saved) == Decimal("1.02")
    assert ledger.get_summary()["open_invested"] == 1.02
    result = ledger.settle_bet(row["id"], won)
    assert result.ok and result.bet["result_pnl"] == expected


def test_missing_activity_leaves_unknown_not_zero_or_nominal_settlement(tmp_path):
    ledger, row = original(tmp_path / "ledger.json")
    with pytest.raises(CashDebitUnavailable, match="activity"):
        evidence(row, activity_rows=[])
    ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row),
        fields={"cash_debit_status": "unknown", "cash_debit_unknown_reason": "missing_activity"})
    before = ledger.path.read_bytes()
    assert ledger.settle_bet(row["id"], False).status == "cash_debit_unknown"
    assert ledger.path.read_bytes() == before
    summary = ledger.get_summary()
    assert summary["open_invested"] is None and summary["unrealized_pnl"] is None
    assert summary["total_pnl"] is None and summary["open_cash_cost_unknown_count"] == 1
    assert not summary["cash_cost_basis_complete"]


def test_unmarked_real_fill_cannot_finalize_gross(tmp_path):
    ledger, row = original(tmp_path / "ledger.json")
    assert "cash_debit_status" not in row
    assert ledger.settle_bet(row["id"], True).status == "cash_debit_unknown"
    assert ledger.get_summary()["open_invested"] is None


@pytest.mark.parametrize("source", [None, "", "clob_order_response", "legacy_reconciliation"])
def test_unmarked_real_actual_fill_requires_cost_regardless_of_source(tmp_path, source):
    ledger, row = original(tmp_path / "ledger.json")
    ledger.update_bet_fields(row["id"], actual_fill_source=source)
    before = ledger.path.read_bytes()
    assert ledger.settle_bet(row["id"], False).status == "cash_debit_unknown"
    assert ledger.get_summary()["open_invested"] is None
    assert ledger.path.read_bytes() == before


def test_unrelated_crypto_fill_keeps_existing_accounting(tmp_path):
    ledger, row = original(tmp_path / "ledger.json")
    ledger.update_bet_fields(row["id"], strategy="btc5m_momentum")
    result = ledger.settle_bet(row["id"], True)
    assert result.ok and result.bet["result_pnl"] == 1
    assert "cash_debit_status" not in result.bet
    assert "result_pnl_exact" not in result.bet


@pytest.mark.parametrize("quantity", ["12.344", "12.346"])
@pytest.mark.parametrize("won", [True, False])
def test_fractional_fill_uses_exact_payout_preserving_rounded_original(tmp_path, quantity, won):
    ledger, row = original(tmp_path / "ledger.json")
    ledger.update_bet_fields(row["id"], order_type="limit", placement_state="resting",
                             shares=13, amount=6.5)
    exact_shares = Decimal(quantity)
    gross = exact_shares * Decimal("0.5")
    debit = gross + Decimal("0.02345")
    converted = ledger.convert_limit_bet_to_position(row["id"], float(exact_shares),
        filled_amount=float(gross), fill_price=.5, fill_source="clob_trade_history",
        fill_trade_ids=["trade-original"])
    assert converted.ok
    row = converted.bet
    original_binding = fill_binding(row)
    assert row["actual_filled_shares"] == round(float(exact_shares), 2)
    assert row["requested_shares"] == 13
    fields = evidence(row, trade_rows=[{**trades()[0], "size": quantity}],
        activity_rows=[{**activity()[0], "size": quantity, "usdcSize": str(debit)}])
    assert fields["actual_filled_shares_exact"] == quantity
    assert ledger.record_cash_debit(row["id"], expected_binding=original_binding, fields=fields).ok
    saved = ledger.get_bets(fresh=True)[0]
    assert fill_binding(saved) == original_binding
    assert verified_fill_shares(saved) == exact_shares
    assert verified_cash_debit(saved) == debit
    ledger.update_current_price(row["id"], .75)
    summary = ledger.get_summary(fresh=True)
    assert summary["unrealized_pnl"] == pytest.approx(float(exact_shares * Decimal(".75") - debit))
    result = ledger.settle_bet(row["id"], won)
    expected_pnl = (exact_shares if won else Decimal(0)) - debit
    assert result.ok
    assert Decimal(result.bet["result_pnl_exact"]) == expected_pnl
    assert ledger.get_summary(fresh=True)["realized_pnl"] == float(expected_pnl)
    assert fill_binding(result.bet) == original_binding
    assert result.bet["actual_fill_amount"] == round(float(gross), 2)


def test_exact_share_provenance_rejects_tampering_and_changed_original(tmp_path):
    ledger, row = original(tmp_path / "ledger.json")
    fields = evidence(row)
    altered = {**fields, "actual_filled_shares_exact": "2.001"}
    before = ledger.path.read_bytes()
    assert ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=altered).status == "invalid_cash_debit"
    assert ledger.path.read_bytes() == before
    ledger.update_bet_fields(row["id"], actual_filled_shares=2.001)
    changed = ledger.path.read_bytes()
    assert ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=fields).status == "fill_changed"
    assert ledger.path.read_bytes() == changed
    bad = {**row, "actual_filled_shares": 2.006}
    with pytest.raises(CashDebitUnavailable, match="original_filled_quantity_mismatch"):
        evidence(bad)


@pytest.mark.parametrize("requested", ["1.99", "NaN", "Infinity", 0])
def test_converted_fill_never_exceeds_valid_original_request(tmp_path, requested):
    _, row = original(tmp_path / "ledger.json")
    row["requested_shares"] = requested
    executor = OrderExecutor(SimpleNamespace(), clob_client=SimpleNamespace(
        get_trades=lambda **kw: trades()), dry_run=False)
    assert executor._lookup_order_from_trade_history(row, row["order_id"]) == (False, None)


def test_legacy_settled_accounting_remains_byte_identical(tmp_path):
    ledger, row = original(tmp_path / "ledger.json")
    ledger.update_bet_fields(row["id"], status="won", settled_at="2026-10-01T00:00:00+00:00",
                            result_pnl=1)
    before = ledger.path.read_bytes()
    assert ledger.get_summary(fresh=True)["realized_pnl"] == 1
    assert ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=evidence(row)).status == "not_open_real"
    assert ledger.path.read_bytes() == before


def test_exact_realized_cost_preserves_subcent_charge_and_observed_zero(tmp_path):
    ledger, row = original(tmp_path / "ledger.json")
    fields = evidence(row, activity_rows=[{**activity()[0], "usdcSize": "1.02345"}])
    ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=fields)
    result = ledger.settle_bet(row["id"], False)
    assert result.bet["result_pnl_exact"] == "-1.02345"
    assert ledger.get_summary()["realized_pnl"] == -1.02345
    zero = evidence(row, activity_rows=[{**activity()[0], "usdcSize": "1.00"}])
    assert Decimal(zero["actual_additional_charge"]) == 0


def test_multiple_distinct_transactions_sum_once_shared_transaction_rejected(tmp_path):
    ledger, row = original(tmp_path / "ledger.json")
    ledger.update_bet_fields(row["id"], actual_fill_trade_ids=["trade-original", "trade-second"])
    row = ledger.get_bets(fresh=True)[0]
    rows = [{**trades()[0], "size": "1"},
            {**trades()[0], "id": "trade-second", "size": "1", "transaction_hash": "0xsecond"}]
    acts = [{**activity()[0], "size": "1", "usdcSize": ".51"},
            {**activity()[0], "size": "1", "usdcSize": ".52", "transactionHash": "0xsecond"}]
    assert evidence(row, trade_rows=rows, activity_rows=acts)["actual_cash_debit"] == "1.03"
    rows[1]["transaction_hash"] = "0xoriginal"
    with pytest.raises(CashDebitUnavailable, match="shared_transaction"):
        evidence(row, trade_rows=rows, activity_rows=acts)
    with pytest.raises(CashDebitUnavailable, match="incomplete"):
        evidence(row, trade_rows=rows[:1], activity_rows=acts)


def test_maker_activity_is_unknown_without_allocation_proof(tmp_path):
    _, row = original(tmp_path / "ledger.json")
    trade = {**trades()[0], "taker_order_id": "another-order",
             "maker_orders": [{"order_id": "order-original", "asset_id": "token-original",
                               "matched_amount": "2", "price": ".5"}]}
    with pytest.raises(CashDebitUnavailable, match="role"):
        evidence(row, trade_rows=[trade])


@pytest.mark.parametrize("bad_fill", [None, {}, {"order_id": "wrong"}])
def test_malformed_or_rehashed_conflicting_provenance_is_unknown(tmp_path, bad_fill):
    ledger, row = original(tmp_path / "ledger.json")
    fields = evidence(row)
    fields["actual_cash_debit_evidence"]["fills"] = [bad_fill]
    fields["actual_cash_debit_evidence_sha256"] = hashlib.sha256(json.dumps(
        fields["actual_cash_debit_evidence"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=fields).status == "invalid_cash_debit"


def test_unknown_pnl_history_does_not_plot_zero_or_crash(capsys):
    tracker._print_pnl_trend([None, .5, None, -.5])
    output = capsys.readouterr().out
    assert "+0.50" in output and "-0.50" in output and "+0.00" not in output


def test_malformed_original_trade_id_is_unavailable(tmp_path):
    ledger, row = original(tmp_path / "ledger.json")
    fields = evidence(row)
    bad = deepcopy(row)
    bad["actual_fill_trade_ids"] = [{}]
    fields["actual_cash_debit_evidence"]["binding"] = fill_binding(bad)
    fields["actual_cash_debit_evidence_sha256"] = hashlib.sha256(json.dumps(
        fields["actual_cash_debit_evidence"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(CashDebitUnavailable): verified_cash_debit({**bad, **fields})


def test_later_activity_recovers_unknown_and_concurrent_settlement_preserves_history(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    ledger, row = original(tmp_path / "ledger.json")
    binding, fields = fill_binding(row), evidence(row)
    ledger.record_cash_debit(row["id"], expected_binding=binding,
        fields={"cash_debit_status": "unknown", "cash_debit_unknown_reason": "missing_activity"})
    assert ledger.settle_bet(row["id"], False).status == "cash_debit_unknown"
    assert tracker.BetLedger(ledger.path).record_cash_debit(row["id"], expected_binding=binding, fields=fields).ok
    def replay():
        return tracker.BetLedger(ledger.path).record_cash_debit(row["id"], expected_binding=binding, fields=fields)
    def settle(): return tracker.BetLedger(ledger.path).settle_bet(row["id"], False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [future.result() for future in [pool.submit(replay), pool.submit(settle), pool.submit(replay)]]
    assert sum(result.bet.get("status") == "lost" for result in results) >= 1
    saved = ledger.get_bets(fresh=True)[0]
    assert saved["status"] == "lost" and saved["result_pnl_exact"] == "-1.02"
    assert fill_binding(saved) == binding
    before = ledger.path.read_bytes()
    assert not replay().ok
    assert ledger.path.read_bytes() == before


@pytest.mark.parametrize("field, value", [("actual_cash_debit", "999"), ("result_pnl_exact", "NaN")])
def test_malformed_fee_aware_settled_summary_is_unknown_without_history_rewrite(tmp_path, field, value):
    ledger, row = original(tmp_path / "ledger.json")
    ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=evidence(row))
    assert ledger.settle_bet(row["id"], False).ok
    document = json.loads(ledger.path.read_bytes())
    document["bets"][0][field] = value
    ledger.path.write_text(json.dumps(document), encoding="utf-8")
    before = ledger.path.read_bytes()
    summary = ledger.get_summary(fresh=True)
    assert not summary["cash_cost_basis_complete"] and summary["settled_cash_cost_unknown_count"] == 1
    assert all(summary[key] is None for key in ("total_wagered", "realized_pnl", "total_pnl", "roi"))
    assert ledger.path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_size", "wrong_price", "nan", "bool", "below_gross", "missing_tx"])
def test_missing_duplicate_or_invalid_cashflow_rejected(tmp_path, mutation):
    _, row = original(tmp_path / "ledger.json")
    rows = activity()
    if mutation == "duplicate": rows.append(deepcopy(rows[0]))
    elif mutation == "wrong_size": rows[0]["size"] = "3"
    elif mutation == "wrong_price": rows[0]["price"] = ".6"
    elif mutation == "nan": rows[0]["usdcSize"] = "NaN"
    elif mutation == "bool": rows[0]["usdcSize"] = True
    elif mutation == "below_gross": rows[0]["usdcSize"] = ".99"
    elif mutation == "missing_tx": rows[0]["transactionHash"] = ""
    with pytest.raises(CashDebitUnavailable): evidence(row, activity_rows=rows)


def test_shared_tx_unrelated_order_cannot_reuse_activity_debit(tmp_path):
    _, row = original(tmp_path / "ledger.json")
    rows = trades()
    other = {**rows[0], "id": "manual-trade", "taker_order_id": "manual-order"}
    with pytest.raises(CashDebitUnavailable, match="shared"):
        evidence(row, trade_rows=rows + [other])


def test_duplicate_trade_id_identical_is_deduplicated_conflict_rejected(tmp_path):
    _, row = original(tmp_path / "ledger.json")
    rows = trades()
    assert evidence(row, trade_rows=rows + deepcopy(rows))["actual_cash_debit"] == "1.02"
    conflicting = {**rows[0], "size": "3"}
    executor = OrderExecutor(SimpleNamespace(), clob_client=SimpleNamespace(
        get_trades=lambda **kw: rows + [conflicting]), dry_run=False)
    assert executor._lookup_order_from_trade_history(row, row["order_id"]) == (False, None)


def test_replay_idempotent_conflict_rejected_stale_and_settled_untouched(tmp_path):
    ledger, row = original(tmp_path / "ledger.json")
    fields = evidence(row)
    assert ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=fields).ok
    before = ledger.path.read_bytes()
    assert ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=fields).ok
    assert ledger.path.read_bytes() == before
    changed = evidence(row, activity_rows=[{**activity()[0], "usdcSize": "1.03"}])
    assert ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=changed).status == "cash_debit_conflict"
    assert ledger.path.read_bytes() == before
    stale = {**fill_binding(row), "order_id": "another"}
    assert ledger.record_cash_debit(row["id"], expected_binding=stale, fields=fields).status == "fill_changed"
    assert ledger.settle_bet(row["id"], False).ok
    settled = ledger.path.read_bytes()
    assert ledger.record_cash_debit(row["id"], expected_binding=fill_binding(row), fields=fields).status == "not_open_real"
    assert ledger.path.read_bytes() == settled


def test_readonly_producer_complete_and_truncated_pages(monkeypatch):
    from src.polymarket.readonly_account import ReadOnlyClobClient
    client = ReadOnlyClobClient(private_key="offline-fixture", funder_address="fixture")
    monkeypatch.setattr(client, "_ensure_client", lambda: None)
    monkeypatch.setattr(client, "_discover_proxy_address", lambda: "fixture")
    monkeypatch.setattr(client_module, "request_data_api_json", lambda *a, **kw: activity())
    assert client.get_trade_activity(start=1792000000, end=1792200000) == activity()
    calls = []
    def full(*args, **kwargs):
        calls.append(kwargs["params"]["offset"])
        return activity() * 500
    monkeypatch.setattr(client_module, "request_data_api_json", full)
    with pytest.raises(ValueError, match="complete-read"):
        client.get_trade_activity(start=1792000000, end=1792200000)
    assert calls == list(range(0, 5000, 500))


def test_actual_maintenance_producer_consumer_wiring_no_get_order(monkeypatch, tmp_path):
    from src import refresh_live, live_control
    from src.strategy import duo_trader
    path = tmp_path / "bet_ledger_single.json"
    ledger, row = original(path)
    monkeypatch.setattr(refresh_live, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(live_control, "LOGS_DIR", tmp_path)
    monkeypatch.setenv("UFC_REFRESH_TRADING_STATE_DIR", str(tmp_path / "controls"))
    for name, suffix in (("SINGLE_LEDGER", "single"), ("CONVICTION_LEDGER", "conviction"), ("MODEL_TRACKER_LEDGER", "model_tracker")):
        monkeypatch.setattr(duo_trader, name, tmp_path / f"bet_ledger_{suffix}.json")
    seen = []
    def settle(view, **kwargs):
        current = view.get_open_bets(fresh=True)[0]
        seen.append(current["actual_cash_debit"])
        return int(view.settle_bet(current["id"], False).ok)
    monkeypatch.setattr(tracker, "auto_settle_from_polymarket", settle)
    clob = SimpleNamespace(get_trades=lambda **kw: trades(),
        get_trade_activity=lambda **kw: activity(),
        get_order=lambda *a: pytest.fail("closed get_order is not required"))
    result = refresh_live._maintain_original_accounting(clob)
    assert seen == ["1.02"] and result["original_open_cash_debits_confirmed"] == 1
    assert result["settled_original_real_positions"] == 1
    assert ledger.get_bets(fresh=True)[0]["result_pnl"] == -1.02
