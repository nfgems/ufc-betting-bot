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
