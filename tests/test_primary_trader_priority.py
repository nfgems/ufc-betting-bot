import pytest

from src.polymarket.executor import OrderExecutor
from src.polymarket.tracker import BetLedger
from src.strategy import duo_trader
from src.strategy.bankroll import BankrollManager


class _Clob:
    def __init__(self, orders=None):
        self.orders = list(orders or [])

    def get_open_orders(self):
        return list(self.orders)


def _paths(tmp_path):
    return {
        "S": tmp_path / "single.json",
        "C": tmp_path / "conviction.json",
        "M": tmp_path / "model_tracker.json",
        "G": tmp_path / "legacy_g_tracker.json",
    }


def _seed_bet(
    path,
    *,
    token_id="token-no",
    shares=4.66,
    order_type="filled_limit",
    placement_state="submitted",
    order_id=None,
):
    ledger = BetLedger(path=path)
    ledger.add_bet(
        fighter="Beta",
        opponent="Alpha",
        side="b",
        amount=round(float(shares) * 0.43, 2),
        price=0.43,
        shares=shares,
        token_id=token_id,
        market_id="market-1",
        model_prob=0.55,
        market_prob=0.43,
        edge=0.12,
        decimal_odds=2.3256,
        dry_run=False,
        order_type=order_type,
        placement_state=placement_state,
        order_id=order_id,
    )


def _executor(path, *, clob=None):
    executor = OrderExecutor(
        bankroll=BankrollManager(initial_bankroll=500, auto_detect_balance=False),
        clob_client=clob or _Clob(),
        dry_run=False,
    )
    executor.ledger = BetLedger(path=path)
    return executor


@pytest.mark.parametrize("primary_label", ["S", "C"])
@pytest.mark.parametrize("tracker_label", ["M", "G"])
def test_tracker_position_does_not_block_primary_trader(
    tmp_path,
    monkeypatch,
    primary_label,
    tracker_label,
):
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        duo_trader,
        "get_all_trader_ledgers",
        lambda: list(paths.items()),
    )
    _seed_bet(paths[tracker_label])
    executor = _executor(paths[primary_label])
    executor._get_live_positions_cached = lambda **kwargs: [
        {"asset": "token-no", "size": 4.66}
    ]

    attribution = executor._priority_conflict_attribution(
        {"token-yes", "token-no"}
    )
    assert attribution["token-no"]["tracker_shares"] == pytest.approx(4.66)

    assert executor._authoritative_wallet_conflict(
        token_ids={"token-yes", "token-no"},
        fighter="Alpha",
    ) == (False, "")


def test_unattributed_or_undercovered_position_still_blocks(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        duo_trader,
        "get_all_trader_ledgers",
        lambda: list(paths.items()),
    )
    executor = _executor(paths["S"])
    executor._get_live_positions_cached = lambda **kwargs: [
        {"asset": "token-no", "size": 4.66}
    ]

    assert executor._authoritative_wallet_conflict(
        token_ids={"token-no"},
        fighter="Alpha",
    )[0] is True

    _seed_bet(paths["M"], shares=3.0)
    assert executor._authoritative_wallet_conflict(
        token_ids={"token-no"},
        fighter="Alpha",
    )[0] is True


def test_tracker_order_does_not_block_primary_but_unknown_order_does(
    tmp_path,
    monkeypatch,
):
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        duo_trader,
        "get_all_trader_ledgers",
        lambda: list(paths.items()),
    )
    _seed_bet(
        paths["M"],
        order_type="limit_bid",
        placement_state="submitted",
        order_id="tracker-order",
    )
    clob = _Clob([{"id": "tracker-order", "asset_id": "token-no"}])
    executor = _executor(paths["S"], clob=clob)

    assert executor._authoritative_open_clob_order_conflict(
        token_ids={"token-no"},
        fighter="Alpha",
    ) == (False, "")

    clob.orders = [{"id": "unknown-order", "asset_id": "token-no"}]
    conflict, reason = executor._authoritative_open_clob_order_conflict(
        token_ids={"token-no"},
        fighter="Alpha",
    )
    assert conflict is True
    assert "open CLOB order" in reason


def test_closed_tracker_ledger_row_still_attributes_exact_open_order(
    tmp_path,
    monkeypatch,
):
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        duo_trader,
        "get_all_trader_ledgers",
        lambda: list(paths.items()),
    )
    _seed_bet(
        paths["G"],
        order_type="limit_bid",
        placement_state="submitted",
        order_id="legacy-order",
    )
    ledger = BetLedger(path=paths["G"])
    assert ledger.cancel_bet(1, reason="stale wallet reconciliation").ok

    executor = _executor(
        paths["C"],
        clob=_Clob([{"id": "legacy-order", "asset_id": "token-no"}]),
    )

    assert executor._authoritative_open_clob_order_conflict(
        token_ids={"token-no"},
        fighter="Alpha",
    ) == (False, "")
