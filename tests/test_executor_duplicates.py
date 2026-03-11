import pandas as pd

from src.polymarket.executor import OrderExecutor, _ledger_entry_blocks_new_order
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager


class _StubClob:
    def create_market_order(self, **kwargs):
        return {"orderID": "stub-order"}


def test_ledger_entry_blocks_new_order_respects_run_mode():
    assert _ledger_entry_blocks_new_order({"dry_run": True}, True) is True
    assert _ledger_entry_blocks_new_order({"dry_run": True}, False) is False
    assert _ledger_entry_blocks_new_order({"dry_run": False}, True) is True
    assert _ledger_entry_blocks_new_order({"dry_run": False}, False) is True


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
        }
    )

    result = executor._place_bet(bet, pd.DataFrame())

    assert result is not None
    assert result["status"] == "placed"
    assert result["dry_run"] is False
    assert len(ledger.open_bets) == 2
