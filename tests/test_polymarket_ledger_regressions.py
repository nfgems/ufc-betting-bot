import builtins
from pathlib import Path

import pytest

from src.polymarket import executor as executor_module
from src.polymarket.tracker import BetLedger


def _add_bet(
    ledger: BetLedger,
    *,
    fighter: str = "Fighter A",
    order_type: str | None = None,
    order_id: str | None = None,
) -> None:
    ledger.add_bet(
        fighter=fighter,
        opponent="Opponent",
        side="a",
        amount=10.0,
        price=0.5,
        shares=20.0,
        token_id=f"token-{fighter}",
        market_id=f"market-{fighter}",
        model_prob=0.6,
        market_prob=0.5,
        edge=0.1,
        decimal_odds=2.0,
        dry_run=True,
        order_type=order_type,
        order_id=order_id,
    )


def test_get_bets_default_path_returns_detached_dicts(tmp_path):
    ledger = BetLedger(path=tmp_path / "ledger.json")
    _add_bet(ledger)

    bets = ledger.get_bets()
    bets[0]["fighter"] = "Mutated Fighter"

    assert ledger.bets[0]["fighter"] == "Fighter A"


def test_get_bets_fresh_returns_detached_dicts(tmp_path):
    ledger = BetLedger(path=tmp_path / "ledger.json")
    _add_bet(ledger)

    bets = ledger.get_bets(fresh=True)
    bets[0]["fighter"] = "Mutated Fighter"

    assert ledger.bets[0]["fighter"] == "Fighter A"


@pytest.mark.parametrize(
    ("mutation", "field", "mutated_value", "expected_value"),
    [
        ("settle", "status", "open", "won"),
        ("cancel", "cancel_reason", "mutated", "test_cancel"),
        ("convert", "shares", 999.0, 5.0),
        ("price", "cur_price", 0.99, 0.61),
    ],
)
def test_successful_ledger_mutations_return_detached_bet_snapshots(
    tmp_path,
    mutation,
    field,
    mutated_value,
    expected_value,
):
    ledger = BetLedger(path=tmp_path / "ledger.json")
    if mutation == "convert":
        _add_bet(ledger, order_type="limit_bid", order_id="order-1")
        result = ledger.convert_limit_bet_to_position(
            1,
            5.0,
            cancel_reason="partial_fill",
        )
    else:
        _add_bet(ledger)
        if mutation == "settle":
            result = ledger.settle_bet(1, True)
        elif mutation == "cancel":
            result = ledger.cancel_bet(1, reason="test_cancel")
        else:
            result = ledger.update_current_price(1, 0.61)

    assert result.ok is True

    result.bet[field] = mutated_value

    assert ledger.bets[0][field] == expected_value


def test_coordinated_ledger_paths_falls_back_on_missing_duo_trader_import(monkeypatch, tmp_path):
    ledger_path = tmp_path / "ledger.json"
    real_import = builtins.__import__

    def _missing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.strategy.duo_trader":
            error = ModuleNotFoundError("No module named 'src.strategy.duo_trader'")
            error.name = "src.strategy.duo_trader"
            raise error
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _missing_import)

    assert executor_module._coordinated_ledger_paths(ledger_path) == (ledger_path.resolve(),)


def test_coordinated_ledger_paths_propagates_transitive_module_not_found(monkeypatch, tmp_path):
    ledger_path = tmp_path / "ledger.json"
    real_import = builtins.__import__

    def _broken_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.strategy.duo_trader":
            error = ModuleNotFoundError("No module named 'missing_dependency'")
            error.name = "missing_dependency"
            raise error
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _broken_import)

    with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
        executor_module._coordinated_ledger_paths(ledger_path)


def test_coordinated_ledger_paths_propagates_plain_import_error(monkeypatch, tmp_path):
    ledger_path = tmp_path / "ledger.json"
    real_import = builtins.__import__

    def _broken_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.strategy.duo_trader":
            raise ImportError("cannot import name 'SINGLE_LEDGER' from 'src.strategy.duo_trader'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _broken_import)

    with pytest.raises(ImportError, match="cannot import name 'SINGLE_LEDGER'"):
        executor_module._coordinated_ledger_paths(ledger_path)


def test_coordinated_ledger_paths_does_not_swallow_unexpected_import_failures(monkeypatch, tmp_path):
    ledger_path = tmp_path / "ledger.json"
    real_import = builtins.__import__

    def _broken_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.strategy.duo_trader":
            raise RuntimeError("boom")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _broken_import)

    with pytest.raises(RuntimeError, match="boom"):
        executor_module._coordinated_ledger_paths(ledger_path)


def test_coordinated_open_bets_keeps_current_ledger_fresh_without_reloading_others(
    monkeypatch,
    tmp_path,
):
    current_path = (tmp_path / "current.json").resolve()
    other_path = (tmp_path / "other.json").resolve()

    class _FakeLedger:
        instances: dict[Path, "_FakeLedger"] = {}

        def __init__(self, path):
            self.path = Path(path)
            self.calls: list[bool] = []
            _FakeLedger.instances[self.path.resolve()] = self

        def get_open_bets(self, *, fresh: bool = False):
            self.calls.append(fresh)
            return [{"fighter": self.path.stem, "status": "open"}]

    current_ledger = _FakeLedger(current_path)

    monkeypatch.setattr(
        executor_module,
        "_coordinated_ledger_paths",
        lambda _: (current_path, other_path),
    )
    monkeypatch.setattr(executor_module, "BetLedger", _FakeLedger)

    executor = object.__new__(executor_module.OrderExecutor)
    executor.ledger = current_ledger

    bets = executor._coordinated_open_bets()

    assert current_ledger.calls == [True]
    assert _FakeLedger.instances[other_path].calls == [False]
    assert [bet["_ledger_path"] for bet in bets] == [str(current_path), str(other_path)]


def test_coordinated_ledger_paths_include_existing_tennis_ledger(monkeypatch, tmp_path):
    from src.strategy import duo_trader

    single = tmp_path / "single.json"
    conviction = tmp_path / "conviction.json"
    tennis = tmp_path / "tennis.json"
    single.write_text("[]", encoding="utf-8")
    conviction.write_text("[]", encoding="utf-8")
    tennis.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)
    monkeypatch.setattr(duo_trader, "TENNIS_LEDGER", tennis)

    paths = executor_module._coordinated_ledger_paths(single)

    assert paths == tuple(
        sorted(
            (single.resolve(), conviction.resolve(), tennis.resolve()),
            key=lambda path: str(path),
        )
    )


def test_reconcile_import_positions_uses_provided_ledger_path(monkeypatch, tmp_path):
    from src.strategy import duo_trader

    single = tmp_path / "single.json"
    tennis = tmp_path / "tennis.json"
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)

    imported = executor_module._reconcile_import_positions(
        live_positions=[
            {
                "asset": "token-tennis",
                "size": 4,
                "avgPrice": 0.5,
                "market": "market-tennis",
                "outcome": "Yes",
                "title": "Player One vs. Player Two",
            }
        ],
        market_token_lookup={
            "token-tennis": {
                "side": "a",
                "fighter": "Player One",
                "opponent": "Player Two",
                "market_id": "market-tennis",
                "condition_id": "cond-tennis",
            }
        },
        tracked_tokens=set(),
        import_ledger_path=tennis,
    )

    assert imported == 1
    assert not single.exists()

    tennis_ledger = BetLedger(path=tennis)
    assert len(tennis_ledger.bets) == 1
    assert tennis_ledger.bets[0]["token_id"] == "token-tennis"
    assert tennis_ledger.bets[0]["market_id"] == "market-tennis"
    assert tennis_ledger.bets[0]["condition_id"] == "cond-tennis"
    assert tennis_ledger.bets[0]["side"] == "a"
    assert tennis_ledger.bets[0]["fighter"] == "Player One"
    assert tennis_ledger.bets[0]["opponent"] == "Player Two"


def test_reconcile_import_positions_normalizes_no_token_to_side_b(tmp_path):
    ledger_path = tmp_path / "ledger.json"

    imported = executor_module._reconcile_import_positions(
        live_positions=[
            {
                "asset": "token-no",
                "size": 3,
                "avgPrice": 0.37,
                "title": "Fighter A vs. Fighter B",
            }
        ],
        market_token_lookup={
            "token-no": {
                "side": "b",
                "fighter": "Fighter B",
                "opponent": "Fighter A",
                "market_id": "market-1",
                "condition_id": "cond-1",
            }
        },
        tracked_tokens=set(),
        import_ledger_path=ledger_path,
    )

    assert imported == 1
    ledger = BetLedger(path=ledger_path)
    assert ledger.bets[0]["side"] == "b"
    assert ledger.bets[0]["fighter"] == "Fighter B"
    assert ledger.bets[0]["opponent"] == "Fighter A"
    assert ledger.bets[0]["condition_id"] == "cond-1"


def test_reconcile_import_positions_skips_unmapped_tokens(tmp_path):
    ledger_path = tmp_path / "ledger.json"

    imported = executor_module._reconcile_import_positions(
        live_positions=[
            {
                "asset": "unknown-token",
                "size": 5,
                "avgPrice": 0.44,
                "title": "Unknown A vs. Unknown B",
            }
        ],
        market_token_lookup={},
        tracked_tokens=set(),
        import_ledger_path=ledger_path,
    )

    assert imported == 0
    assert not ledger_path.exists()


def test_cancel_all_stale_limit_bids_includes_tennis_ledger(monkeypatch, tmp_path):
    from src.strategy import duo_trader
    from src.strategy import bankroll as bankroll_module

    single = tmp_path / "single.json"
    conviction = tmp_path / "conviction.json"
    tennis = tmp_path / "tennis.json"
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)
    monkeypatch.setattr(duo_trader, "TENNIS_LEDGER", tennis)

    class _FakeBankroll:
        def __init__(self, *args, **kwargs):
            pass

    captured_paths: list[Path] = []

    class _FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def cancel_stale_limit_bids(self, ledger):
            captured_paths.append(Path(ledger.path).resolve())
            return 0

    monkeypatch.setattr(bankroll_module, "BankrollManager", _FakeBankroll)
    monkeypatch.setattr(executor_module, "OrderExecutor", _FakeExecutor)

    total = executor_module.cancel_all_stale_limit_bids(clob_client=object())

    assert total == 0
    assert captured_paths == [
        single.resolve(),
        conviction.resolve(),
        tennis.resolve(),
    ]


def test_fetch_wallet_positions_for_reconciliation_reuses_cache_after_429(monkeypatch):
    wallet = "0xabc"
    cached_positions = [{"asset": "token-1", "size": "2"}]
    executor_module._WALLET_POSITION_FETCH_CACHE.clear()
    executor_module._WALLET_POSITION_RATE_LIMIT_UNTIL.clear()
    executor_module._WALLET_POSITION_FETCH_CACHE[wallet] = (100.0, cached_positions)

    class _RateLimitedResponse:
        status_code = 429
        headers = {"Retry-After": "15"}

    monkeypatch.setattr(executor_module.time, "monotonic", lambda: 200.0)
    monkeypatch.setattr(executor_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        executor_module.requests,
        "get",
        lambda *args, **kwargs: _RateLimitedResponse(),
    )

    result = executor_module._fetch_wallet_positions_for_reconciliation(wallet)

    assert result == cached_positions
    assert executor_module._WALLET_POSITION_RATE_LIMIT_UNTIL[wallet] == pytest.approx(215.0)


def test_fetch_wallet_positions_for_reconciliation_skips_network_during_cooldown(monkeypatch):
    wallet = "0xabc"
    executor_module._WALLET_POSITION_FETCH_CACHE.clear()
    executor_module._WALLET_POSITION_RATE_LIMIT_UNTIL.clear()
    executor_module._WALLET_POSITION_RATE_LIMIT_UNTIL[wallet] = 130.0

    calls: list[int] = []

    def _unexpected_get(*args, **kwargs):
        calls.append(1)
        raise AssertionError("network should not be called during cooldown")

    monkeypatch.setattr(executor_module.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(executor_module.requests, "get", _unexpected_get)

    assert executor_module._fetch_wallet_positions_for_reconciliation(wallet) is None
    assert calls == []
