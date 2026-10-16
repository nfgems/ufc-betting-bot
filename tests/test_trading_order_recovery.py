"""Bounded, offline exchange fixtures; no credentialed client or real orders."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import threading
import time

import pandas as pd
import pytest

from src.polymarket.client import ClobClientWrapper
from src.polymarket.executor import OrderExecutor
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    monkeypatch.setenv("LIVE_TRADING_ARMED", "false")
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not contact a real API")
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)


def executor_for(path, clob=None):
    executor = OrderExecutor(
        BankrollManager(initial_bankroll=500.0, auto_detect_balance=False),
        clob_client=clob or SimpleNamespace(),
        dry_run=False,
        skip_wallet_conflict_check=True,
    )
    executor.ledger = BetLedger(path)
    return executor


def journal(executor, *, order_id=None, placement_state="unknown"):
    bet = executor._journal_live_order_attempt(
        fighter="Alpha", opponent="Beta", side="a", amount=25.0,
        price=0.5, shares=50.0, token_id="123", market_id="market-1",
        model_prob=0.70, market_prob=0.50, edge=0.20, decimal_odds=2.0,
        event_date=(datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
        order_type="marketable_limit",
    )
    if order_id:
        executor._record_prepared_order_id(bet, order_id)
    executor._update_submission_state(bet, placement_state=placement_state)
    return executor._reload_ledger_entry(bet)


def candidate():
    return pd.Series({
        "fighter_a": "Alpha", "fighter_b": "Beta", "bet_on": "Alpha",
        "model_prob": 0.70, "blended_prob": 0.70, "market_prob": 0.50,
        "edge": 0.20, "decimal_odds": 2.0, "bet_side": "a",
        "token_id_yes": "123", "token_id_no": "456", "market_id": "market-1",
        "tick_size": "0.01", "neg_risk": False, "fee_rate": 0.0,
        "fee_exponent": 1.0, "fee_source": "clob", "override_bet_size": 25.0,
        "event_date": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
    })


def sdk_wrapper(tmp_path, *, on_post=None, version=2):
    # Use the runtime's real signing/hash implementation with a public dummy key.
    # No account credentials, SDK HTTP calls, or automatic auth are constructed.
    pytest.importorskip("py_clob_client_v2")
    from py_clob_client_v2.order_builder.builder import OrderBuilder
    from py_clob_client_v2.clob_types import CreateOrderOptions
    from py_clob_client_v2.signer import Signer

    class ExchangeFixture:
        def __init__(self):
            self.builder = OrderBuilder(Signer("11" * 32, 137))
            self.created = []
            self.posts = []

        def create_order(self, args, options):
            order = self.builder.build_order(
                args, CreateOrderOptions(tick_size=options.tick_size, neg_risk=options.neg_risk),
                version=version,
            )
            self.created.append(order)
            return order

        def post_order(self, order, order_type):
            self.posts.append(order)
            return on_post(order) if on_post else {"success": True}

        def create_and_post_order(self, *args):
            raise AssertionError("Journalled execution must not use rebuilding SDK POST helper")

    sdk = ExchangeFixture()
    wrapper = ClobClientWrapper(private_key="11" * 32)
    wrapper._client = sdk
    wrapper._ensure_client = lambda: None
    wrapper._log_geoblock_status = lambda action: None
    wrapper.get_open_orders = lambda: []
    return wrapper, sdk


@pytest.mark.parametrize("version", [1, 2])
def test_signed_id_saved_before_post_and_retained_after_interruption(tmp_path, version):
    executor = executor_for(tmp_path / "ledger.json")
    bet = journal(executor, placement_state="pending_submit")
    witnessed = []
    def interrupt(order):
        from eth_account import Account
        row = BetLedger(executor.ledger.path).bets[0]
        assert row["id"] == bet["id"]
        assert row["order_id"].startswith("0x") and len(row["order_id"]) == 66
        assert Account._recover_hash(row["order_id"], signature=order.signature) == sdk.builder.signer.address()
        witnessed.append(row["order_id"])
        raise SystemExit("interruption after POST received, before response")
    wrapper, sdk = sdk_wrapper(tmp_path, on_post=interrupt, version=version)
    with pytest.raises(SystemExit):
        wrapper.create_limit_order(
            "123", "BUY", 0.5, 50.0, "0.01", False,
            on_order_prepared=lambda oid: executor._record_prepared_order_id(bet, oid),
        )
    restarted = executor_for(executor.ledger.path)
    assert restarted.ledger.bets[0]["order_id"] == witnessed[0]
    assert restarted.ledger.bets[0]["id"] == bet["id"]
    assert len(sdk.created) == len(sdk.posts) == 1
    assert restarted.ledger.get_summary()["open_invested"] == 0


def test_journal_failure_aborts_post(tmp_path):
    wrapper, sdk = sdk_wrapper(tmp_path)
    def cannot_save(order_id):
        raise OSError("durable ledger write failed")
    with pytest.raises(OSError, match="write failed"):
        wrapper.create_limit_order("123", "BUY", 0.5, 50, "0.01", False, on_order_prepared=cannot_save)
    assert len(sdk.created) == 1
    assert sdk.posts == []


def test_real_sdk_transient_post_failure_does_not_repeat_request(tmp_path, monkeypatch):
    pytest.importorskip("py_clob_client_v2")
    import httpx
    from py_clob_client_v2.client import ClobClient
    from py_clob_client_v2.clob_types import ApiCreds, CreateOrderOptions
    import py_clob_client_v2.http_helpers.helpers as helpers

    # Real SDK post_order/_post/helpers.post with an entirely local transport.
    sdk = ClobClient("https://fixture.invalid", 137, key="11" * 32,
                     creds=ApiCreds(api_key="fixture", api_secret="ZmFrZQ==", api_passphrase="fixture"),
                     retry_on_error=False)
    sdk.create_order = lambda args, options: sdk.builder.build_order(
        args, CreateOrderOptions(tick_size=options.tick_size, neg_risk=options.neg_risk), version=2,
    )
    sdk._l2_headers = lambda *args, **kwargs: {}
    calls = []
    class LocalTransport:
        def request(self, **kwargs):
            calls.append(kwargs["method"])
            raise httpx.ReadTimeout("fixture lost response")
    monkeypatch.setattr(helpers, "_http_client", LocalTransport())
    wrapper = ClobClientWrapper(private_key="11" * 32)
    wrapper._client = sdk
    wrapper._ensure_client = lambda: None
    wrapper._log_geoblock_status = lambda action: None
    ids = []
    with pytest.raises(Exception, match="Request exception"):
        wrapper.create_limit_order("123", "BUY", 0.5, 50, "0.01", False, on_order_prepared=ids.append)
    assert calls == ["POST"]
    assert len(ids) == 1


def test_pre_submit_guard_runs_after_id_saved_and_aborts_post(tmp_path):
    wrapper, sdk = sdk_wrapper(tmp_path)
    events = []
    def guard():
        assert events == ["journalled"]
        raise RuntimeError("orders disabled")
    with pytest.raises(RuntimeError, match="orders disabled"):
        wrapper.create_limit_order(
            "123", "BUY", 0.5, 50, "0.01", False,
            on_order_prepared=lambda oid: events.append("journalled"), pre_submit_check=guard,
        )
    assert sdk.posts == []


def test_pre_submit_guard_can_make_reentrant_wrapper_read(tmp_path):
    wrapper, sdk = sdk_wrapper(tmp_path)
    owner_thread = threading.get_ident()
    sdk.get_order = lambda oid: {"id": oid, "status": "LIVE"}
    def guard():
        assert threading.get_ident() == owner_thread
        assert wrapper.get_order("original-order")["id"] == "original-order"
    wrapper.create_limit_order(
        "123", "BUY", 0.5, 50, "0.01", False,
        on_order_prepared=lambda oid: None, pre_submit_check=guard,
    )
    assert len(sdk.posts) == 1


def test_wrapper_keeps_shared_transport_serialized(tmp_path):
    active = 0
    peak = 0
    counter_lock = threading.Lock()
    def post(order):
        nonlocal active, peak
        with counter_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with counter_lock:
            active -= 1
        return {"success": True}
    first, _ = sdk_wrapper(tmp_path, on_post=post)
    second, _ = sdk_wrapper(tmp_path, on_post=post)
    errors = []
    def call(wrapper):
        try:
            wrapper.create_limit_order("123", "BUY", 0.5, 50, "0.01", False, on_order_prepared=lambda oid: None)
        except BaseException as exc:
            errors.append(exc)
    threads = [threading.Thread(target=call, args=(wrapper,)) for wrapper in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == []
    assert peak == 1


def test_ordinary_executor_timeout_reconciles_original_order_before_duplicate_attempt(tmp_path):
    def timeout(order):
        raise TimeoutError("response lost after submit")
    wrapper, sdk = sdk_wrapper(tmp_path, on_post=timeout)
    executor = executor_for(tmp_path / "ledger.json", wrapper)
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True, "adjusted_size": 25.0, "available_liquidity": 100.0,
        "best_ask": 0.50, "slippage": 0.0, "reason": "",
    }
    result = executor._place_bet(candidate(), pd.DataFrame())
    assert result["status"] == "unknown"
    original = BetLedger(executor.ledger.path).bets[0]
    assert original["order_id"]
    reads = []
    wrapper.get_order = lambda oid: reads.append(oid) or {
        "id": oid, "status": "LIVE", "original_size": original["shares"], "size_matched": 0,
    }
    restarted = executor_for(executor.ledger.path, wrapper)
    assert restarted._place_bet(candidate(), pd.DataFrame()) is None
    assert reads == [original["order_id"]]
    assert len(sdk.created) == len(sdk.posts) == 1
    assert restarted.ledger.bets[0]["id"] == original["id"]
    assert restarted.ledger.bets[0]["order_id"] == original["order_id"]
    assert restarted.ledger.bets[0]["placement_state"] == "submitted"
    assert restarted.ledger.get_summary()["open_invested"] == 0
    assert executor.bankroll.available_cash == 500.0


@pytest.mark.parametrize("reported", [None, float("nan"), -1, "invalid"])
def test_filled_status_without_actual_quantity_retains_original_unknown(tmp_path, reported):
    clob = SimpleNamespace(get_open_orders=lambda: [], get_order=lambda oid: {
        "id": oid, "status": "MATCHED", "size_matched": reported,
    })
    executor = executor_for(tmp_path / "ledger.json", clob)
    original = journal(executor, order_id="original-exchange-order")
    recovered = executor._reconcile_unresolved_submission(original)
    assert recovered["id"] == original["id"]
    assert recovered["order_id"] == original["order_id"]
    assert recovered["placement_state"] == "unknown"
    assert recovered["status"] == "open"
    assert executor.ledger.get_summary()["open_invested"] == 0
    assert executor.bankroll.available_cash == 500.0


def test_exact_filled_quantity_recovers_position_and_keeps_original_ids(tmp_path):
    executor = executor_for(tmp_path / "ledger.json", SimpleNamespace(
        get_open_orders=lambda: [], get_order=lambda oid: {
            "id": oid, "status": "MATCHED", "size_matched": "12.5", "original_size": "50",
        },
        get_trades=lambda **kw: [{"id": "observed-fill", "status": "CONFIRMED", "asset_id": "123",
            "taker_order_id": "original-order", "price": .5, "size": 12.5}],
    ))
    original = journal(executor, order_id="original-order")
    recovered = executor._reconcile_unresolved_submission(original)
    assert recovered["id"] == original["id"]
    assert recovered["order_id"] == "original-order"
    assert recovered["order_type"] == "filled_limit"
    assert recovered["placement_state"] == "filled"
    assert recovered["actual_filled_shares"] == 12.5
    assert recovered["actual_fill_amount"] == 6.25
    assert BetLedger(executor.ledger.path).get_summary()["open_invested"] == 6.25
    assert executor.bankroll.available_cash == 500.0  # no invented reservation refund


@pytest.mark.parametrize("order_data", [{}, {"id": "some-manual-order", "status": "CANCELED", "size_matched": 0}])
def test_missing_or_mismatched_response_does_not_release_unknown_order(tmp_path, order_data):
    executor = executor_for(tmp_path / "ledger.json", SimpleNamespace(
        get_open_orders=lambda: [], get_order=lambda oid: order_data,
    ))
    original = journal(executor, order_id="original-order")
    recovered = executor._reconcile_unresolved_submission(original)
    assert recovered["status"] == "open"
    assert recovered["placement_state"] == "unknown"
    assert recovered["order_id"] == "original-order"


def test_explicit_cancel_with_zero_actual_fill_unblocks_but_does_not_refund_cash(tmp_path):
    executor = executor_for(tmp_path / "ledger.json", SimpleNamespace(
        get_open_orders=lambda: [], get_order=lambda oid: {
            "id": oid, "status": "CANCELED", "size_matched": 0,
        },
    ))
    original = journal(executor, order_id="original-order")
    recovered = executor._reconcile_unresolved_submission(original)
    assert recovered["status"] == "cancelled"
    assert recovered["id"] == original["id"]
    assert recovered["order_id"] == "original-order"
    assert executor.bankroll.available_cash == 500.0
    assert executor.ledger.get_summary()["open_invested"] == 0


def test_no_id_cannot_adopt_economically_identical_manual_order(tmp_path):
    executor = executor_for(tmp_path / "ledger.json", SimpleNamespace())
    original = journal(executor, placement_state="pending_submit")
    recovered = executor._reconcile_unresolved_submission(original, open_orders=[{
        "id": "manual-order", "asset_id": "123", "price": 0.5, "original_size": 50,
    }])
    assert recovered["id"] == original["id"]
    assert recovered["order_id"] is None
    assert recovered["placement_state"] == "unknown"
    assert recovered["status"] == "open"


def test_confirmed_taker_trade_history_is_attributed_only_to_original_order(tmp_path):
    def missing(oid):
        raise KeyError(oid)
    executor = executor_for(tmp_path / "ledger.json", SimpleNamespace(
        get_open_orders=lambda: [], get_order=missing,
        get_trades=lambda **kwargs: [
            {"id": "manual", "taker_order_id": "manual-order", "asset_id": "123", "size": 50, "status": "CONFIRMED"},
            {"id": "actual", "taker_order_id": "original-order", "asset_id": "123", "size": 10, "price": .5, "status": "CONFIRMED"},
            {"id": "actual", "taker_order_id": "original-order", "asset_id": "123", "size": 10, "price": .5, "status": "CONFIRMED"},
        ],
    ))
    original = journal(executor, order_id="original-order")
    recovered = executor._reconcile_unresolved_submission(original)
    assert recovered["order_id"] == "original-order"
    assert recovered["actual_filled_shares"] == 10
    assert recovered["actual_fill_amount"] == 5


def test_original_prepared_id_cannot_be_overwritten_by_response(tmp_path):
    executor = executor_for(tmp_path / "ledger.json")
    original = journal(executor, order_id="original-order")
    with pytest.raises(RuntimeError, match="differs"):
        executor._update_submission_state(original, placement_state="submitted", order_id="other-order")
    assert BetLedger(executor.ledger.path).bets[0]["order_id"] == "original-order"


@pytest.mark.parametrize("order_response", [None, {}, {"id": "original-order", "status": "MATCHED", "size_matched": 20}])
@pytest.mark.parametrize("role", ["taker", "maker"])
def test_confirmed_original_fills_use_actual_weighted_trade_prices(tmp_path, order_response, role):
    def fill(trade_id, size, price):
        trade = {"id": trade_id, "status": "CONFIRMED", "asset_id": "123", "price": price, "size": size}
        if role == "taker":
            trade["taker_order_id"] = "original-order"
        else:
            trade["taker_order_id"] = "other-taker"
            trade["maker_orders"] = [{"order_id": "original-order", "asset_id": "123",
                                       "matched_amount": size, "price": price}]
        return trade
    first, second = fill("fill-1", 10, .37), fill("fill-2", 10, .39)
    executor = executor_for(tmp_path / "ledger.json", SimpleNamespace(get_open_orders=lambda: [],
        get_order=lambda oid: order_response, get_trades=lambda **kw: [first, first.copy(), second,
            {"id": "manual-fill", "taker_order_id": "manual-order", "asset_id": "123",
             "size": 50, "price": .5, "status": "CONFIRMED"}]))
    original = journal(executor, order_id="original-order")
    recovered = executor._reconcile_unresolved_submission(original)
    assert recovered["id"] == original["id"] and recovered["order_id"] == "original-order"
    assert recovered["actual_filled_shares"] == 20
    assert recovered["actual_fill_amount"] == 7.60
    assert recovered["actual_fill_price"] == pytest.approx(.38)
    assert recovered["actual_fill_source"] == "clob_trade_history"
    assert recovered["actual_fill_trade_ids"] == ["fill-1", "fill-2"]
    assert recovered["price"] == .5 and recovered["submitted_amount"] == 25
    assert recovered["requested_shares"] == 50
    assert executor.bankroll.available_cash == 500


@pytest.mark.parametrize("missing_evidence", ["unavailable", "no_price", "no_id", "wrong_token", "pending", "unconfirmed", "not_confirmed", "unsuccessful", "contradictory_duplicate"])
def test_actual_fill_economics_missing_or_uncertain_never_uses_limit_price(tmp_path, missing_evidence):
    trade = {"id": "original-fill", "status": "CONFIRMED", "asset_id": "123",
             "taker_order_id": "original-order", "price": .37, "size": 10}
    trades = [trade]
    if missing_evidence == "no_price": trade.pop("price")
    if missing_evidence == "no_id": trade.pop("id")
    if missing_evidence == "wrong_token": trade["asset_id"] = "other-token"
    if missing_evidence == "pending": trade["status"] = "MATCHED"
    if missing_evidence in {"unconfirmed", "not_confirmed", "unsuccessful"}: trade["status"] = missing_evidence
    if missing_evidence == "contradictory_duplicate": trades.append({**trade, "size": 20})
    client = SimpleNamespace(get_open_orders=lambda: [], get_order=lambda oid: {
        "id": oid, "status": "MATCHED", "price": .5, "size_matched": 10, "original_size": 50})
    if missing_evidence != "unavailable": client.get_trades = lambda **kw: trades
    executor = executor_for(tmp_path / "ledger.json", client)
    original = journal(executor, order_id="original-order")
    recovered = executor._reconcile_unresolved_submission(original)
    assert recovered["placement_state"] == "unknown" and recovered["status"] == "open"
    assert recovered.get("actual_fill_amount") is None
    assert recovered["order_id"] == original["order_id"] and recovered["id"] == original["id"]


@pytest.mark.parametrize("unconfirmed_status", ["not_failed", "failure_pending", "FAILED_PENDING", "unknown_failure"])
def test_mixed_confirmed_and_unknown_failure_labels_preserve_original_uncertainty(tmp_path, unconfirmed_status):
    trades = [{"id": "confirmed-fill", "status": "CONFIRMED", "asset_id": "123",
               "taker_order_id": "original-order", "price": .4, "size": 10},
              {"id": "unconfirmed-fill", "status": unconfirmed_status, "asset_id": "123",
               "taker_order_id": "original-order", "price": .4, "size": 5}]
    executor = executor_for(tmp_path / "ledger.json", SimpleNamespace(get_open_orders=lambda: [],
        get_order=lambda oid: None, get_trades=lambda **kw: trades))
    original = journal(executor, order_id="original-order")
    recovered = executor._reconcile_unresolved_submission(original)
    assert recovered["placement_state"] == "unknown" and recovered["status"] == "open"
    assert recovered.get("actual_fill_amount") is None
    assert recovered["id"] == original["id"] and recovered["order_id"] == original["order_id"]
