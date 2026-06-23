import json
from datetime import datetime, timedelta, timezone

from src.polymarket.btc_5m import BTC5M_ORDER_TYPE, BTC5M_STRATEGY_NAME
from src.web import app as web_app


def _write_ledger(path, bets):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "bets": bets,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _btc5m_bet(
    *,
    bet_id=1,
    profile="late_capture",
    dry_run=False,
    status="open",
    amount=10.0,
    price=0.80,
    shares=12.5,
    side="up",
    result_pnl=None,
):
    now = datetime.now(timezone.utc)
    window_start = now.replace(second=0, microsecond=0)
    window_end = window_start + timedelta(minutes=5)
    return {
        "id": bet_id,
        "fighter": f"BTC 5m {side.title()}",
        "opponent": "BTC 5m Down" if side == "up" else "BTC 5m Up",
        "side": side,
        "amount": amount,
        "price": price,
        "shares": shares,
        "token_id": f"{side}-token",
        "market_id": "btc-market",
        "condition_id": "btc-condition",
        "model_prob": 0.0,
        "market_prob": price,
        "edge": 0.0,
        "decimal_odds": round(1.0 / price, 4),
        "dry_run": dry_run,
        "status": status,
        "placed_at": now.isoformat(),
        "event_date": window_end.isoformat(),
        "market_event_date": window_start.isoformat(),
        "settled_at": now.isoformat() if status in {"won", "lost"} else None,
        "result_pnl": result_pnl,
        "cur_price": None,
        "order_type": BTC5M_ORDER_TYPE,
        "order_id": f"order-{bet_id}",
        "placement_state": "dry_run" if dry_run else "filled",
        "reason": "test btc5m position",
        "strategy": BTC5M_STRATEGY_NAME,
        "profile": profile,
        "strategy_style": "probability_capture",
        "market_slug": "btc-updown-5m-1781986200",
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "btc_window_start_price": 100000.0,
        "btc_window_start_price_full": 100000.0,
        "btc_current_price": 100050.0,
        "btc_current_price_full": 100050.0,
        "btc_move_usd": 50.0,
        "supporting_prob": 0.88,
    }


def _write_signal(path, *, profile="late_capture", action="trade", direction="up"):
    now = datetime.now(timezone.utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": 1,
        "recorded_at": now.isoformat(),
        "strategy": BTC5M_STRATEGY_NAME,
        "profile": profile,
        "strategy_style": "probability_capture",
        "market": {
            "slug": "btc-updown-5m-1781986200",
            "market_id": "btc-market",
            "condition_id": "btc-condition",
            "window_start": now.isoformat(),
            "window_end": (now + timedelta(minutes=5)).isoformat(),
            "seconds_left": 45.0,
            "active": True,
            "closed": False,
            "accepting_orders": True,
        },
        "price": {
            "source": "binance",
            "symbol": "BTCUSDT",
            "price_to_beat": 100000.0,
            "current_price": 100050.0,
            "distance_to_price_to_beat_usd": 50.0,
        },
        "signal": {
            "action": action,
            "reason_code": "test_signal",
            "reason": "test signal reason",
            "direction": direction,
            "supporting_prob": 0.88,
            "entry_price": 0.80,
            "btc_move_usd": 50.0,
        },
        "result": {
            "status": "ok" if action == "trade" else "idle",
            "mode": "real",
            "orders": [
                {
                    "status": "placed",
                    "direction": direction,
                    "amount": 10.0,
                    "price": 0.80,
                    "shares": 12.5,
                    "order_style": "marketable_limit",
                }
            ] if action == "trade" else [],
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_api_btc5m_live_reads_configured_and_paper_ledgers(tmp_path, monkeypatch):
    live_ledger = tmp_path / "configured_btc5m.json"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "btc5m_signal_snapshots.jsonl"

    _write_ledger(live_ledger, [_btc5m_bet()])
    _write_ledger(
        paper_dir / "cheap_side.json",
        [
            _btc5m_bet(
                bet_id=2,
                profile="cheap_side",
                dry_run=True,
                status="won",
                amount=5.0,
                price=0.25,
                shares=20.0,
                side="down",
                result_pnl=15.0,
            )
        ],
    )
    _write_signal(signal_log)

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", live_ledger)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    response = web_app.app.test_client().get("/api/btc5m/live")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    payload = response.get_json()
    assert payload["schema_version"] == 1
    assert payload["summary"]["open_positions"] == 1
    assert payload["summary"]["open_exposure"] == 10.0
    assert payload["summary"]["open_win_pnl"] == 2.5
    assert payload["summary"]["open_loss_pnl"] == -10.0

    live_profile = next(p for p in payload["profiles"] if p["profile"] == "late_capture")
    assert live_profile["mode"] == "live"
    assert live_profile["open_positions"][0]["entry_price"] == 0.8
    assert live_profile["open_positions"][0]["win_pnl"] == 2.5
    assert live_profile["last_signal"]["signal"]["direction"] == "up"

    paper_profile = next(p for p in payload["profiles"] if p["profile"] == "cheap_side")
    assert paper_profile["mode"] == "paper"
    assert paper_profile["stats"]["realized_pnl"] == 15.0


def test_api_btc5m_live_supports_production_ledger_override(tmp_path, monkeypatch):
    configured_missing = tmp_path / "missing_default.json"
    production_ledger = tmp_path / "production_btc5m.json"
    signal_log = tmp_path / "signals.jsonl"
    paper_dir = tmp_path / "paper"

    _write_ledger(
        production_ledger,
        [_btc5m_bet(profile="ml_candidate_v1", amount=7.5, price=0.75, shares=10.0)],
    )
    _write_signal(signal_log, profile="ml_candidate_v1")

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", configured_missing)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv(web_app.BTC5M_MONITOR_LEDGER_ENV, f"production={production_ledger}")

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    profile = next(p for p in payload["profiles"] if p["profile"] == "ml_candidate_v1")
    assert profile["ledger_label"] == "production"
    assert profile["mode"] == "live"
    assert profile["stats"]["open_exposure"] == 7.5
    assert profile["open_positions"][0]["loss_pnl"] == -7.5
    assert "production=" in payload["config"]["monitor_ledger_env_value"]


def test_api_btc5m_live_reads_promoted_profile_ledgers(tmp_path, monkeypatch):
    configured_missing = tmp_path / "missing_default.json"
    live_dir = tmp_path / "btc5m_live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    promoted_ledger = live_dir / "late_capture.json"

    _write_ledger(promoted_ledger, [_btc5m_bet(profile="late_capture", dry_run=True)])
    _write_signal(signal_log, profile="late_capture")

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", configured_missing)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    assert payload["config"]["live_profiles"] == ["late_capture"]
    assert payload["config"]["live_ledger_dir"] == str(live_dir)
    profile = next(p for p in payload["profiles"] if p["profile"] == "late_capture")
    assert profile["ledger_source"] == "live"
    assert profile["ledger_label"] == "live:late_capture"
    assert profile["mode"] == "dry_run"
    assert profile["stats"]["open_exposure"] == 10.0


def test_api_btc5m_live_shows_live_profile_before_first_ledger_write(tmp_path, monkeypatch):
    configured_missing = tmp_path / "missing_default.json"
    live_dir = tmp_path / "btc5m_live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "missing_signals.jsonl"
    ledger_path = live_dir / "late_capture_gap005.json"

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", configured_missing)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture_gap005")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)
    web_app.set_runtime_status(
        {
            "service": "ufc-betting-bot",
            "startup_source": "web",
            "requested_live_mode": "real",
            "requested_live_mode_raw": "real",
            "effective_live_mode": "real",
            "trading_enabled": True,
            "trading_live": True,
            "model_name": "xgboost",
            "host": "0.0.0.0",
            "public_bind": True,
            "ready": True,
            "errors": [],
            "warnings": [],
            "checks": [],
            "components": {
                "btc5m_loop:late_capture_gap005": {
                    "state": "running",
                    "message": "BTC 5m profile late_capture_gap005 last cycle idle.",
                    "profile": "late_capture_gap005",
                    "ledger_path": str(ledger_path),
                    "trading_mode": "real",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        }
    )

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    assert payload["config"]["live_profiles"] == ["late_capture_gap005"]
    assert payload["config"]["signal_log_enabled"] is True
    assert payload["summary"]["profile_count"] == 1
    profile = payload["profiles"][0]
    assert profile["profile"] == "late_capture_gap005"
    assert profile["mode"] == "live"
    assert profile["ledger_exists"] is False
    assert profile["ledger_path"] == str(ledger_path)
    assert profile["runtime"]["state"] == "running"
    assert profile["stats"]["open_exposure"] == 0.0
    alert_codes = {alert["code"] for alert in payload["alerts"]}
    assert "live_ledger_pending_first_write" in alert_codes
    assert "no_btc5m_ledgers" not in alert_codes


def test_btc5m_page_renders():
    response = web_app.app.test_client().get("/btc5m")

    assert response.status_code == 200
    assert b"BTC 5m Live State" in response.data
