import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from src import betting_window
from src.polymarket.executor import OrderExecutor, _ledger_entry_blocks_new_order
from src.polymarket.tracker import BetLedger
from src.strategy import duo_trader
from src.strategy.bankroll import BankrollManager


class _StubClob:
    def create_market_order(self, **kwargs):
        return {"orderID": "stub-order"}

    def create_limit_order(self, **kwargs):
        return {"orderID": "stub-order"}


class _SlowMarketOrderClob:
    def __init__(self):
        self.calls = 0
        self.started = threading.Event()
        self._lock = threading.Lock()

    def get_open_orders(self):
        return []

    def create_market_order(self, **kwargs):
        with self._lock:
            self.calls += 1
            call_no = self.calls
            self.started.set()
        time.sleep(0.15)
        return {"orderID": f"slow-order-{call_no}"}

    def create_limit_order(self, **kwargs):
        with self._lock:
            self.calls += 1
            call_no = self.calls
            self.started.set()
        time.sleep(0.15)
        return {"orderID": f"slow-limit-{call_no}"}


def test_ledger_entry_blocks_new_order_respects_run_mode():
    assert _ledger_entry_blocks_new_order({"dry_run": True}, True) is True
    assert _ledger_entry_blocks_new_order({"dry_run": True}, False) is False
    assert _ledger_entry_blocks_new_order({"dry_run": False}, True) is True
    assert _ledger_entry_blocks_new_order({"dry_run": False}, False) is True


def test_live_executor_blocks_missing_event_time_and_records_terminal_audit(tmp_path):
    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=_StubClob(),
        dry_run=False,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    audit = []
    executor.decision_audit_callback = lambda bet, payload: audit.append(payload)
    bet = pd.Series(
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "bet_on": "Alpha",
            "bet_side": "a",
            "model_prob": 0.65,
            "blended_prob": 0.65,
            "market_prob": 0.50,
            "edge": 0.15,
            "decimal_odds": 2.0,
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "market-1",
            "tick_size": "0.01",
            "neg_risk": False,
        }
    )

    result = executor._place_near_miss_limit(bet, pd.DataFrame())

    assert result is None
    assert audit[-1]["status"] == "skipped"
    assert audit[-1]["gate"] == "event_time_unavailable"


def test_wallet_position_lookup_failure_blocks_live_order():
    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=_StubClob(),
        dry_run=False,
    )
    executor._get_live_positions_cached = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("positions unavailable")
    )

    conflict, reason = executor._authoritative_wallet_conflict(
        token_ids={"token-yes"},
        fighter="Alpha",
    )

    assert conflict is True
    assert "could not verify live wallet positions" in reason


def test_dry_run_executor_skips_duplicate_open_dry_run_market(tmp_path):
    ledger = BetLedger(path=tmp_path / "ledger.json")
    ledger.add_bet(
        fighter="Charles Johnson",
        opponent="Bruno Silva",
        side="a",
        amount=20.0,
        price=0.62,
        shares=32.26,
        token_id="token-yes",
        market_id="1510646",
        model_prob=0.675,
        market_prob=0.62,
        edge=0.055,
        decimal_odds=1.6129,
        dry_run=True,
        order_type="market",
    )

    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=object(),
        dry_run=True,
    )
    executor.ledger = ledger

    bet = pd.Series(
        {
            "bet_on": "Charles Johnson",
            "model_prob": 0.676,
            "blended_prob": 0.676,
            "market_prob": 0.615,
            "edge": 0.061,
            "decimal_odds": 1.626,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "market_id": "1510646",
        }
    )

    assert executor._place_bet(bet, pd.DataFrame()) is None
    assert len(ledger.open_bets) == 1


def test_duplicate_check_refreshes_stale_ledger_instance(tmp_path):
    path = tmp_path / "ledger.json"
    stale_ledger = BetLedger(path=path)
    writer_ledger = BetLedger(path=path)
    writer_ledger.add_bet(
        fighter="Charles Johnson",
        opponent="Bruno Silva",
        side="a",
        amount=20.0,
        price=0.62,
        shares=32.26,
        token_id="token-yes",
        market_id="1510646",
        model_prob=0.675,
        market_prob=0.62,
        edge=0.055,
        decimal_odds=1.6129,
        dry_run=True,
        order_type="market",
    )

    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=object(),
        dry_run=True,
    )
    executor.ledger = stale_ledger

    bet = pd.Series(
        {
            "bet_on": "Charles Johnson",
            "model_prob": 0.676,
            "blended_prob": 0.676,
            "market_prob": 0.615,
            "edge": 0.061,
            "decimal_odds": 1.626,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "market_id": "1510646",
        }
    )

    assert executor._place_bet(bet, pd.DataFrame()) is None
    assert len(BetLedger(path=path).open_bets) == 1


def test_real_run_executor_ignores_old_dry_run_duplicate(tmp_path):
    ledger = BetLedger(path=tmp_path / "ledger.json")
    ledger.add_bet(
        fighter="Charles Johnson",
        opponent="Bruno Silva",
        side="a",
        amount=20.0,
        price=0.62,
        shares=32.26,
        token_id="token-yes",
        market_id="1510646",
        model_prob=0.675,
        market_prob=0.62,
        edge=0.055,
        decimal_odds=1.6129,
        dry_run=True,
        order_type="market",
    )

    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=_StubClob(),
        dry_run=False,
    )
    executor.ledger = ledger
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.62,
        "reason": "",
    }

    bet = pd.Series(
        {
            "bet_on": "Charles Johnson",
            "model_prob": 0.676,
            "blended_prob": 0.676,
            "market_prob": 0.62,
            "edge": 0.056,
            "decimal_odds": 1.6129,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "market_id": "1510646",
            "tick_size": "0.01",
            "neg_risk": False,
            "event_date": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is not None
    assert result["status"] == "placed"
    assert result["dry_run"] is False
    assert len(ledger.open_bets) == 2


def test_same_side_tracker_wallet_position_does_not_block_primary_bet(tmp_path):
    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=_StubClob(),
        dry_run=False,
        force_market_order=True,
    )
    executor.ledger = BetLedger(path=tmp_path / "single.json")
    executor._get_live_positions_cached = lambda **kwargs: [
        {"asset": "token-yes", "size": 6.36}
    ]
    executor._authoritative_open_clob_order_conflict = lambda **kwargs: (False, "")
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.62,
        "reason": "",
    }
    bet = pd.Series(
        {
            "fighter_a": "Valter Walker",
            "fighter_b": "Thomas Petersen",
            "bet_on": "Valter Walker",
            "model_prob": 0.766,
            "blended_prob": 0.689,
            "market_prob": 0.622,
            "edge": 0.067,
            "decimal_odds": 1.6077,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "2885013",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 25.0,
            "event_date": (
                datetime.now(timezone.utc) + timedelta(hours=24)
            ).isoformat(),
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is not None
    assert result["status"] == "placed"
    assert len(executor.ledger.get_open_bets()) == 1


def test_executor_blocks_bets_before_bet_window_opens(tmp_path, monkeypatch):
    now = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(betting_window, "_current_utc", lambda: now)

    ledger = BetLedger(path=tmp_path / "ledger.json")
    executor = OrderExecutor(
        bankroll=duo_trader.BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=_StubClob(),
        dry_run=False,
    )
    executor.ledger = ledger
    executor._authoritative_wallet_conflict = lambda **kwargs: (False, "")
    executor._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.62,
        "reason": "",
    }

    bet = pd.Series(
        {
            "bet_on": "Charles Johnson",
            "model_prob": 0.676,
            "blended_prob": 0.676,
            "market_prob": 0.62,
            "edge": 0.056,
            "decimal_odds": 1.6129,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "market_id": "1510646",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 25.0,
            "event_date": (now + timedelta(days=4)).isoformat(),
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is None
    assert ledger.open_bets == []


def test_executor_blocks_near_miss_limit_before_bet_window_opens(tmp_path, monkeypatch):
    now = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(betting_window, "_current_utc", lambda: now)

    ledger = BetLedger(path=tmp_path / "ledger.json")
    executor = OrderExecutor(
        bankroll=duo_trader.BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=_StubClob(),
        dry_run=False,
    )
    executor.ledger = ledger
    executor._authoritative_wallet_conflict = lambda **kwargs: (False, "")

    bet = pd.Series(
        {
            "fighter_a": "Charles Johnson",
            "fighter_b": "Bruno Silva",
            "bet_on": "Charles Johnson",
            "model_prob": 0.64,
            "blended_prob": 0.64,
            "market_prob": 0.62,
            "edge": 0.02,
            "decimal_odds": 1.6129,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "1510646",
            "tick_size": "0.01",
            "neg_risk": False,
            "event_date": (now + timedelta(days=4)).isoformat(),
        }
    )

    result = executor._place_near_miss_limit(bet, pd.DataFrame())

    assert result is None
    assert ledger.open_bets == []


def test_concurrent_market_duplicate_attempts_are_serialized(tmp_path):
    path = tmp_path / "ledger.json"
    clob = _SlowMarketOrderClob()

    executor_a = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor_b = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor_a.ledger = BetLedger(path=path)
    executor_b.ledger = BetLedger(path=path)
    executor_a._authoritative_wallet_conflict = lambda **kwargs: (False, "")
    executor_b._authoritative_wallet_conflict = lambda **kwargs: (False, "")

    liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.62,
        "reason": "",
    }
    executor_a._check_liquidity = liquidity
    executor_b._check_liquidity = liquidity

    bet = pd.Series(
        {
            "bet_on": "Charles Johnson",
            "model_prob": 0.676,
            "blended_prob": 0.676,
            "market_prob": 0.62,
            "edge": 0.056,
            "decimal_odds": 1.6129,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "market_id": "1510646",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 25.0,
            "event_date": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
        }
    )

    results = [None, None]
    errors = []

    def _run(index, executor):
        try:
            results[index] = executor._place_bet(bet, pd.DataFrame())
        except Exception as exc:  # pragma: no cover - failure path asserted below
            errors.append(exc)

    thread_a = threading.Thread(target=_run, args=(0, executor_a))
    thread_b = threading.Thread(target=_run, args=(1, executor_b))

    thread_a.start()
    assert clob.started.wait(timeout=2)
    thread_b.start()

    thread_a.join(timeout=2)
    thread_b.join(timeout=2)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert not errors
    assert clob.calls == 1
    assert sum(result is None for result in results) == 1
    assert sum(result is not None and result["status"] == "placed" for result in results) == 1
    assert sorted(round(executor.bankroll.bankroll, 2) for executor in (executor_a, executor_b)) == [480.0, 500.0]
    assert len(BetLedger(path=path).open_bets) == 1


def test_concurrent_opposite_side_market_attempts_are_serialized(tmp_path):
    path = tmp_path / "ledger.json"
    clob = _SlowMarketOrderClob()

    executor_a = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor_b = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor_a.ledger = BetLedger(path=path)
    executor_b.ledger = BetLedger(path=path)
    executor_a._authoritative_wallet_conflict = lambda **kwargs: (False, "")
    executor_b._authoritative_wallet_conflict = lambda **kwargs: (False, "")

    liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.62,
        "reason": "",
    }
    executor_a._check_liquidity = liquidity
    executor_b._check_liquidity = liquidity

    bet_a = pd.Series(
        {
            "bet_on": "Charles Johnson",
            "model_prob": 0.676,
            "blended_prob": 0.676,
            "market_prob": 0.62,
            "edge": 0.056,
            "decimal_odds": 1.6129,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "1510646",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 25.0,
            "event_date": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
        }
    )
    bet_b = pd.Series(
        {
            "bet_on": "Bruno Silva",
            "model_prob": 0.676,
            "blended_prob": 0.676,
            "market_prob": 0.62,
            "edge": 0.056,
            "decimal_odds": 1.6129,
            "bet_side": "b",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "1510646",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 25.0,
            "event_date": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
        }
    )

    results = [None, None]
    errors = []

    def _run(index, executor, bet):
        try:
            results[index] = executor._place_bet(bet, pd.DataFrame())
        except Exception as exc:  # pragma: no cover - failure path asserted below
            errors.append(exc)

    thread_a = threading.Thread(target=_run, args=(0, executor_a, bet_a))
    thread_b = threading.Thread(target=_run, args=(1, executor_b, bet_b))

    thread_a.start()
    assert clob.started.wait(timeout=2)
    thread_b.start()

    thread_a.join(timeout=2)
    thread_b.join(timeout=2)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert not errors
    assert clob.calls == 1
    assert sum(result is None for result in results) == 1
    assert sum(result is not None and result["status"] == "placed" for result in results) == 1
    assert len(BetLedger(path=path).open_bets) == 1


def test_concurrent_market_duplicate_attempts_are_serialized_across_trader_ledgers(
    tmp_path,
    monkeypatch,
):
    single_path = tmp_path / "bet_ledger_single.json"
    conviction_path = tmp_path / "bet_ledger_conviction.json"
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single_path)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction_path)

    clob = _SlowMarketOrderClob()

    executor_a = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor_b = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor_a.ledger = BetLedger(path=single_path)
    executor_b.ledger = BetLedger(path=conviction_path)
    executor_a._authoritative_wallet_conflict = lambda **kwargs: (False, "")
    executor_b._authoritative_wallet_conflict = lambda **kwargs: (False, "")

    liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.62,
        "reason": "",
    }
    executor_a._check_liquidity = liquidity
    executor_b._check_liquidity = liquidity

    bet = pd.Series(
        {
            "bet_on": "Charles Johnson",
            "model_prob": 0.676,
            "blended_prob": 0.676,
            "market_prob": 0.62,
            "edge": 0.056,
            "decimal_odds": 1.6129,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "market_id": "1510646",
            "tick_size": "0.01",
            "neg_risk": False,
            "override_bet_size": 25.0,
            "event_date": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
        }
    )

    results = [None, None]
    errors = []

    def _run(index, executor):
        try:
            results[index] = executor._place_bet(bet, pd.DataFrame())
        except Exception as exc:  # pragma: no cover - failure path asserted below
            errors.append(exc)

    thread_a = threading.Thread(target=_run, args=(0, executor_a))
    thread_b = threading.Thread(target=_run, args=(1, executor_b))

    thread_a.start()
    assert clob.started.wait(timeout=2)
    thread_b.start()

    thread_a.join(timeout=2)
    thread_b.join(timeout=2)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert not errors
    assert clob.calls == 1
    assert sum(result is None for result in results) == 1
    assert sum(result is not None and result["status"] == "placed" for result in results) == 1
    assert sorted(round(executor.bankroll.bankroll, 2) for executor in (executor_a, executor_b)) == [480.0, 500.0]
    assert len(BetLedger(path=single_path).open_bets) + len(BetLedger(path=conviction_path).open_bets) == 1


def test_concurrent_market_and_near_miss_attempts_are_serialized_across_trader_ledgers(
    tmp_path,
    monkeypatch,
):
    single_path = tmp_path / "bet_ledger_single.json"
    conviction_path = tmp_path / "bet_ledger_conviction.json"
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single_path)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction_path)

    clob = _SlowMarketOrderClob()

    executor_a = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor_b = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=clob,
        dry_run=False,
    )
    executor_a.ledger = BetLedger(path=single_path)
    executor_b.ledger = BetLedger(path=conviction_path)
    executor_a._authoritative_wallet_conflict = lambda **kwargs: (False, "")
    executor_b._authoritative_wallet_conflict = lambda **kwargs: (False, "")

    executor_a._check_liquidity = lambda *args, **kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.62,
        "reason": "",
    }

    market_bet = pd.Series(
        {
            "fighter_a": "Charles Johnson",
            "fighter_b": "Bruno Silva",
            "bet_on": "Charles Johnson",
            "model_prob": 0.676,
            "blended_prob": 0.676,
            "market_prob": 0.62,
            "edge": 0.056,
            "decimal_odds": 1.6129,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "1510646",
            "tick_size": "0.01",
            "neg_risk": False,
            "event_date": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
            "override_bet_size": 25.0,
        }
    )
    near_miss_bet = pd.Series(
        {
            "fighter_a": "Charles Johnson",
            "fighter_b": "Bruno Silva",
            "bet_on": "Charles Johnson",
            "model_prob": 0.64,
            "blended_prob": 0.64,
            "market_prob": 0.62,
            "edge": 0.02,
            "decimal_odds": 1.6129,
            "bet_side": "a",
            "token_id_yes": "token-yes",
            "token_id_no": "token-no",
            "market_id": "1510646",
            "tick_size": "0.01",
            "neg_risk": False,
        }
    )

    results = {"market": None, "near_miss": None}
    errors = []

    def _run_market():
        try:
            results["market"] = executor_a._place_bet(market_bet, pd.DataFrame())
        except Exception as exc:  # pragma: no cover - failure path asserted below
            errors.append(exc)

    def _run_near_miss():
        try:
            results["near_miss"] = executor_b._place_near_miss_limit(
                near_miss_bet,
                pd.DataFrame(),
            )
        except Exception as exc:  # pragma: no cover - failure path asserted below
            errors.append(exc)

    thread_market = threading.Thread(target=_run_market)
    thread_near_miss = threading.Thread(target=_run_near_miss)

    thread_market.start()
    assert clob.started.wait(timeout=2)
    thread_near_miss.start()

    thread_market.join(timeout=2)
    thread_near_miss.join(timeout=2)

    assert not thread_market.is_alive()
    assert not thread_near_miss.is_alive()
    assert not errors
    assert clob.calls == 1
    assert results["market"] is not None
    assert results["market"]["status"] == "placed"
    assert results["near_miss"] is None
    assert len(BetLedger(path=single_path).open_bets) + len(BetLedger(path=conviction_path).open_bets) == 1
