"""Runtime parity checks for an immutable confirmation strategy."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.strategy import duo_trader
from src.strategy.runtime_strategy import (
    RUNTIME_CONSTRAINTS,
    build_confirmed_strategy_payload,
    canonical_json_sha256,
    shared_strategy_constants_snapshot,
    validate_confirmed_strategy_payload,
    validate_locked_strategy_config,
)


def _strategy_config() -> dict:
    return {
        "name": "confirmed_production",
        "s_min_edge": 0.03,
        "s_blend_weight": 0.35,
        "s_model_agreement_min_edge": 0.02,
        "c_min_model_prob": 0.70,
        "c_min_no_odds_prob": 0.55,
        "c_share": 0.40,
        "c_bet_fraction": 0.04,
        "c_max_bet_fraction": 0.07,
        "c_min_edge": 0.01,
        "c_max_decimal_odds": 2.5,
        "m_enabled": False,
        "m_min_model_prob": 0.65,
        "m_min_no_odds_prob": 0.50,
        "m_bet_fraction": 0.03,
        "m_share": 0.30,
    }


def test_confirmed_strategy_payload_reproduces_exact_hash_and_safe_constraints():
    config = _strategy_config()
    payload = build_confirmed_strategy_payload(
        config,
        expected_sha256=canonical_json_sha256(config),
    )

    assert validate_confirmed_strategy_payload(payload) == payload
    assert payload["runtime_constraints"] == RUNTIME_CONSTRAINTS


def test_dormant_m_share_does_not_reduce_confirmed_c_allocation():
    config = _strategy_config()
    config["c_share"] = 1.0
    config["m_share"] = 0.30

    assert validate_locked_strategy_config(config)["c_share"] == 1.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("m_enabled", True, "M tracker disabled"),
        ("s_min_edge", float("nan"), "finite"),
        ("c_max_decimal_odds", float("inf"), "finite"),
    ],
)
def test_confirmed_strategy_rejects_unsupported_or_nonfinite_semantics(
    field,
    value,
    message,
):
    config = _strategy_config()
    config[field] = value

    with pytest.raises(ValueError, match=message):
        validate_locked_strategy_config(config)


def test_runtime_constraints_cannot_arm_unevaluated_execution_paths():
    config = _strategy_config()
    payload = build_confirmed_strategy_payload(
        config,
        expected_sha256=canonical_json_sha256(config),
    )
    payload["runtime_constraints"] = {
        "model_tracker_enabled": True,
        "near_miss_orders_enabled": False,
    }

    with pytest.raises(ValueError, match="disable unevaluated M and near-miss"):
        validate_confirmed_strategy_payload(payload)


def test_duo_runtime_uses_exact_confirmed_s_c_parameters_and_forces_m_near_miss_off(
    monkeypatch,
):
    config = _strategy_config()
    calls: dict[str, object] = {}

    class FakeBankroll:
        def __init__(self):
            self.bankroll = 100.0
            self.available_cash = 100.0
            self.total_equity = 100.0
            self.is_stopped = False

        def get_stats(self):
            return {"bankroll": self.bankroll}

    class FakeExecutor:
        def __init__(self):
            self.ledger = SimpleNamespace(open_bets=[])

        def _match_predictions_to_markets(self, predictions, markets):
            return pd.DataFrame()

        def refresh_open_limit_orders(self, **_kwargs):
            return {}

    created: list[SimpleNamespace] = []

    def fake_create(profile, allocation, *args, **kwargs):
        calls.setdefault("create", []).append(
            {"name": profile.name, "allocation": allocation, **kwargs}
        )
        trader = SimpleNamespace(
            name=profile.name,
            bankroll=FakeBankroll(),
            executor=FakeExecutor(),
        )
        created.append(trader)
        return trader

    def fake_value(*args, **kwargs):
        calls["value"] = kwargs
        return pd.DataFrame()

    def fake_conviction(*args, **kwargs):
        calls["conviction"] = kwargs
        return pd.DataFrame()

    class FakeAudit:
        def __init__(self, **_kwargs):
            self.metadata = {}
            self.cycle_id = "cycle"
            self._fights = {}

        def bind_executor(self, *_args):
            return None

        def record_path(self, *_args, **_kwargs):
            return None

        def persist(self):
            return {"cycle_id": "cycle", "fight_count": 0, "completed_at": "now"}

    from src.strategy import execution_audit

    monkeypatch.setattr(execution_audit, "ExecutionAuditCollector", FakeAudit)
    monkeypatch.setattr(
        execution_audit,
        "audit_single_value_pipeline",
        lambda *a, **k: calls.__setitem__("audit_single", k),
    )
    monkeypatch.setattr(
        execution_audit,
        "audit_conviction_pipeline",
        lambda *a, **k: calls.__setitem__("audit_conviction", k),
    )
    monkeypatch.setattr(
        duo_trader,
        "_resolve_total_bankroll",
        lambda **_kwargs: duo_trader.WalletBankrollBasis(100.0, 100.0, "test"),
    )
    monkeypatch.setattr(duo_trader, "_create_trader", fake_create)
    monkeypatch.setattr(duo_trader, "find_value_bets", fake_value)
    monkeypatch.setattr(duo_trader, "find_conviction_bets", fake_conviction)
    monkeypatch.setattr(
        duo_trader,
        "_create_tracker_trader",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("confirmed runtime must not create M")
        ),
    )

    result = duo_trader.run_duo_traders(
        predictions=pd.DataFrame(),
        markets=pd.DataFrame(),
        dry_run=True,
        run_model_tracker=True,
        strategy_config=config,
        confirmed_shared_constants=shared_strategy_constants_snapshot(),
    )

    assert calls["value"] == {
        "min_edge": config["s_min_edge"],
        "blend_weight": config["s_blend_weight"],
        "near_miss_min_edge": 0.0,
        "edge_scaling_base": config["s_min_edge"],
        "model_agreement_min_edge": config["s_model_agreement_min_edge"],
    }
    assert calls["conviction"] == {
        "require_positive_ev": False,
        "min_edge": config["c_min_edge"],
        "max_decimal_odds": config["c_max_decimal_odds"],
        "min_model_prob": config["c_min_model_prob"],
        "min_no_odds_prob": config["c_min_no_odds_prob"],
    }
    assert calls["audit_single"]["model_agreement_min_edge"] == config[
        "s_model_agreement_min_edge"
    ]
    assert calls["audit_conviction"]["min_model_prob"] == config[
        "c_min_model_prob"
    ]
    assert calls["audit_conviction"]["min_no_odds_prob"] == config[
        "c_min_no_odds_prob"
    ]
    assert calls["audit_conviction"]["min_edge"] == config["c_min_edge"]
    assert calls["audit_conviction"]["max_decimal_odds"] == config[
        "c_max_decimal_odds"
    ]
    assert calls["create"][1]["allocation"] == pytest.approx(40.0)
    assert calls["create"][1]["max_bet_fraction"] == config["c_max_bet_fraction"]
    assert result["near_miss_orders_enabled"] is False
    assert result["model_tracker_enabled"] is False
    assert result["confirmed_strategy_config_sha256"] == canonical_json_sha256(config)


def test_real_duo_runtime_rejects_differing_explicit_edge_override():
    with pytest.raises(ValueError, match="min_edge differs"):
        duo_trader.run_duo_traders(
            predictions=pd.DataFrame(),
            markets=pd.DataFrame(),
            dry_run=False,
            min_edge=0.02,
            strategy_config=_strategy_config(),
            confirmed_shared_constants=shared_strategy_constants_snapshot(),
        )


def test_confirmed_runtime_requires_bound_shared_constants():
    with pytest.raises(ValueError, match="bound shared\\s+strategy constants"):
        duo_trader.run_duo_traders(
            predictions=pd.DataFrame(),
            markets=pd.DataFrame(),
            dry_run=True,
            strategy_config=_strategy_config(),
        )


def test_confirmed_runtime_fails_closed_on_shared_constant_drift(monkeypatch):
    """DIR-AUD-P2-019: a post-confirmation KELLY_FRACTION edit cannot trade."""
    config = _strategy_config()
    payload = build_confirmed_strategy_payload(
        config,
        expected_sha256=canonical_json_sha256(config),
    )

    from src import config as app_config

    monkeypatch.setattr(
        app_config, "KELLY_FRACTION", app_config.KELLY_FRACTION * 2
    )

    with pytest.raises(ValueError, match="drifted.*kelly_fraction"):
        duo_trader.run_duo_traders(
            predictions=pd.DataFrame(),
            markets=pd.DataFrame(),
            dry_run=True,
            strategy_config=config,
            confirmed_shared_constants=payload["shared_constants"],
        )


def test_confirmed_runtime_fails_closed_on_min_fighter_fights_drift(monkeypatch):
    config = _strategy_config()
    payload = build_confirmed_strategy_payload(
        config,
        expected_sha256=canonical_json_sha256(config),
    )

    from src import config as app_config

    monkeypatch.setattr(
        app_config, "MIN_FIGHTER_FIGHTS", app_config.MIN_FIGHTER_FIGHTS + 1
    )

    with pytest.raises(ValueError, match="drifted.*min_fighter_fights"):
        duo_trader.run_duo_traders(
            predictions=pd.DataFrame(),
            markets=pd.DataFrame(),
            dry_run=True,
            strategy_config=config,
            confirmed_shared_constants=payload["shared_constants"],
        )


def test_confirmed_real_runtime_fails_before_s_c_when_m_retirement_fails(
    monkeypatch,
):
    config = _strategy_config()
    monkeypatch.setattr(
        duo_trader,
        "ensure_legacy_g_orders_retired",
        lambda **_kwargs: {"maintenance_incomplete": False},
    )
    monkeypatch.setattr(
        duo_trader,
        "retire_model_tracker_open_limit_orders",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("CLOB unavailable")),
    )

    with pytest.raises(RuntimeError, match="Model Tracker order retirement"):
        duo_trader.run_duo_traders(
            predictions=pd.DataFrame(),
            markets=pd.DataFrame(),
            clob=object(),
            dry_run=False,
            min_edge=config["s_min_edge"],
            strategy_config=config,
            confirmed_shared_constants=shared_strategy_constants_snapshot(),
        )


def test_confirmed_real_runtime_fails_before_orders_when_near_miss_remains_open(
    monkeypatch,
):
    config = _strategy_config()

    class FakeAudit:
        def __init__(self, **_kwargs):
            self.metadata = {}
            self.cycle_id = "cycle"
            self._fights = {}

        def bind_executor(self, *_args):
            return None

        def record_path(self, *_args, **_kwargs):
            return None

    class FakeBankroll:
        bankroll = 100.0
        available_cash = 100.0
        total_equity = 100.0
        is_stopped = False

    class FakeExecutor:
        def __init__(self):
            self.ledger = SimpleNamespace(
                open_bets=[{"id": 1, "order_type": "near_miss_limit"}]
            )

        def _match_predictions_to_markets(self, predictions, markets):
            return pd.DataFrame()

        def refresh_open_limit_orders(self, **kwargs):
            assert kwargs["cancel_without_model_view_reason"] == (
                "confirmed strategy disables near-miss orders"
            )
            assert kwargs["cancel_partially_filled"] is True
            return {"maintenance_incomplete": False}

    from src.strategy import execution_audit

    monkeypatch.setattr(execution_audit, "ExecutionAuditCollector", FakeAudit)
    monkeypatch.setattr(execution_audit, "audit_single_value_pipeline", lambda *_a, **_k: None)
    monkeypatch.setattr(
        duo_trader,
        "ensure_legacy_g_orders_retired",
        lambda **_kwargs: {"maintenance_incomplete": False},
    )
    monkeypatch.setattr(
        duo_trader,
        "retire_model_tracker_open_limit_orders",
        lambda **_kwargs: {"maintenance_incomplete": False},
    )
    monkeypatch.setattr(duo_trader, "assert_live_wallet_exposure_synced", lambda **_kwargs: None)
    monkeypatch.setattr(
        duo_trader,
        "_create_trader",
        lambda *_args, **_kwargs: SimpleNamespace(
            name="Single Trader",
            bankroll=FakeBankroll(),
            executor=FakeExecutor(),
        ),
    )
    monkeypatch.setattr(
        duo_trader,
        "find_value_bets",
        lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()),
    )

    with pytest.raises(RuntimeError, match="near-miss orders remain open"):
        duo_trader.run_duo_traders(
            predictions=pd.DataFrame(),
            markets=pd.DataFrame(),
            clob=object(),
            dry_run=False,
            min_edge=config["s_min_edge"],
            bankroll_basis=duo_trader.WalletBankrollBasis(100.0, 100.0, "test"),
            strategy_config=config,
            confirmed_shared_constants=shared_strategy_constants_snapshot(),
        )
