import types

import pytest

from src import bot, live_control


@pytest.fixture(autouse=True)
def _clear_live_env(monkeypatch):
    for env_name in [
        "ODDS_API_KEY",
        "POLYMARKET_PRIVATE_KEY",
        "POLYMARKET_FUNDER_ADDRESS",
        "WEB_DASHBOARD_TOKEN",
        "LIVE_TRADING_MODE",
        "LIVE_TRADING_ARMED",
        "LIVE_TRADING_CONFIRMATION",
        "LIVE_MODEL",
    ]:
        monkeypatch.delenv(env_name, raising=False)


@pytest.fixture
def _seed_live_artifacts(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    models_dir.mkdir()
    logs_dir.mkdir()

    (models_dir / "xgboost_model.pkl").write_bytes(b"primary")
    (models_dir / "xgboost_no_odds_model.pkl").write_bytes(b"no-odds")

    monkeypatch.setattr(live_control, "MODELS_DIR", models_dir)
    monkeypatch.setattr(live_control, "LOGS_DIR", logs_dir)
    return {"models_dir": models_dir, "logs_dir": logs_dir}


def test_evaluate_live_startup_defaults_to_off():
    status = live_control.evaluate_live_startup(startup_source="test")

    assert status["requested_live_mode"] == live_control.LIVE_MODE_OFF
    assert status["effective_live_mode"] == live_control.LIVE_MODE_OFF
    assert status["trading_enabled"] is False
    assert status["ready"] is True
    assert status["errors"] == []


def test_public_bind_detection_matches_non_local_hosts():
    assert live_control.is_public_bind("0.0.0.0") is True
    assert live_control.is_public_bind("203.0.113.10") is True
    assert live_control.is_public_bind("127.0.0.1") is False
    assert live_control.is_public_bind("localhost") is False


def test_real_mode_requires_arming_confirmation_and_public_dashboard_token(_seed_live_artifacts, monkeypatch):
    monkeypatch.setattr(live_control, "LIVE_TRADING_DISABLED", False)
    monkeypatch.setenv("ODDS_API_KEY", "odds-key")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "private-key")

    status = live_control.evaluate_live_startup(
        requested_mode=live_control.LIVE_MODE_REAL,
        host="0.0.0.0",
        startup_source="test",
    )

    assert status["ready"] is False
    assert status["effective_live_mode"] == live_control.LIVE_MODE_OFF
    errors = " ".join(status["errors"])
    assert "LIVE_TRADING_ARMED" in errors
    assert "LIVE_TRADING_CONFIRMATION" in errors
    assert "WEB_DASHBOARD_TOKEN" in errors


def test_real_mode_becomes_ready_when_fully_armed(_seed_live_artifacts, monkeypatch):
    monkeypatch.setattr(live_control, "LIVE_TRADING_DISABLED", False)
    monkeypatch.setenv("ODDS_API_KEY", "odds-key")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "private-key")
    monkeypatch.setenv("WEB_DASHBOARD_TOKEN", "secret-token")
    monkeypatch.setenv("LIVE_TRADING_ARMED", "1")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMATION", live_control.REAL_TRADING_CONFIRM_VALUE)

    status = live_control.evaluate_live_startup(
        requested_mode=live_control.LIVE_MODE_REAL,
        host="0.0.0.0",
        startup_source="test",
    )

    assert status["ready"] is True
    assert status["effective_live_mode"] == live_control.LIVE_MODE_REAL
    assert status["trading_enabled"] is True
    assert any("POLYMARKET_FUNDER_ADDRESS" in warning for warning in status["warnings"])


def test_assert_real_trading_allowed_raises_concrete_message(_seed_live_artifacts, monkeypatch):
    monkeypatch.setattr(live_control, "LIVE_TRADING_DISABLED", False)
    monkeypatch.setenv("ODDS_API_KEY", "odds-key")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "private-key")

    with pytest.raises(RuntimeError) as excinfo:
        live_control.assert_real_trading_allowed(startup_source="test")

    message = str(excinfo.value)
    assert "LIVE_TRADING_ARMED=1" in message
    assert "LIVE_TRADING_CONFIRMATION=REAL_TRADING_ENABLED" in message


def test_polymarket_real_guard_does_not_require_ufc_model_or_odds(tmp_path, monkeypatch):
    monkeypatch.setattr(live_control, "LIVE_TRADING_DISABLED", False)
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "private-key")
    monkeypatch.setenv("LIVE_TRADING_ARMED", "1")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMATION", live_control.REAL_TRADING_CONFIRM_VALUE)

    status = live_control.evaluate_polymarket_live_startup(
        requested_mode=live_control.LIVE_MODE_REAL,
        startup_source="test",
        ledger_paths=(tmp_path / "btc5m_ledger.json",),
    )

    assert status["ready"] is True
    assert status["effective_live_mode"] == live_control.LIVE_MODE_REAL
    assert status["errors"] == []


def test_cmd_duo_live_returns_early_when_real_guard_blocks(monkeypatch):
    monkeypatch.setattr(
        bot,
        "assert_real_trading_allowed",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("blocked by live guard")),
    )

    messages = []
    monkeypatch.setattr(bot.logger, "error", lambda message, *args, **kwargs: messages.append(message))

    result = bot.cmd_duo_live(
        types.SimpleNamespace(
            dry_run=False,
            real=True,
            model="xgboost",
            min_edge=0.02,
        )
    )

    assert result == {"status": "error", "reason": "blocked by live guard"}
    assert messages == ["blocked by live guard"]


def test_real_cmd_duo_live_refuses_unconfirmed_strategy_before_clob_actions(monkeypatch):
    monkeypatch.setattr(bot, "assert_real_trading_allowed", lambda **_kwargs: None)
    monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args: None)
    monkeypatch.setattr(bot, "_resolve_runtime_bundle_summary", lambda **_kwargs: None)
    monkeypatch.setattr(
        "src.model.train.load_model",
        lambda _name: {"feature_cols": [], "feature_importance": {}},
    )
    monkeypatch.setattr(
        "src.polymarket.client.ClobClientWrapper",
        lambda: (_ for _ in ()).throw(AssertionError("CLOB must not be initialized")),
    )
    monkeypatch.setattr(
        "src.strategy.duo_trader.ensure_legacy_g_orders_retired",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("orders must not be touched")
        ),
    )

    result = bot.cmd_duo_live(
        types.SimpleNamespace(
            dry_run=False,
            real=True,
            model="xgboost",
            min_edge=0.02,
        )
    )

    assert result["status"] == "error"
    assert result["total_orders"] == 0
    assert "final-v2 confirmed strategy" in result["reason"]
