import json
from datetime import datetime, timedelta, timezone

import pytest

from src.polymarket.btc_5m import BTC5M_ORDER_TYPE, BTC5M_STRATEGY_NAME
from src.web import app as web_app


PROMOTED_BTC5M_LIVE_PROFILES = [
    "late_capture_gap005",
    "late_capture",
    "late_capture_min86",
    "late_capture_min88",
    "late_capture_min90",
    "late_capture_min92",
    "late_capture_mid_gap005",
    "late_capture_mid_min88",
    "late_capture_full_min88",
    "late_capture_gap005_min88",
    "late_capture_gap010_min88",
    "late_capture_full_min88_liq",
]


@pytest.fixture(autouse=True)
def _disable_btc5m_activity_enrichment(monkeypatch):
    monkeypatch.setattr(web_app, "_btc5m_fetch_trade_activity", lambda: [])
    monkeypatch.setattr(web_app, "BTC5M_LIVE_PROFILES", "")
    monkeypatch.delenv("BTC5M_LEDGER_PATH", raising=False)
    monkeypatch.delenv("BTC5M_LIVE_PROFILES", raising=False)
    monkeypatch.delenv("BTC5M_LIVE_LEDGER_DIR", raising=False)


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
    actual_fill_price=None,
    actual_fill_amount=None,
    actual_filled_shares=None,
):
    now = datetime.now(timezone.utc)
    window_start = now.replace(second=0, microsecond=0)
    window_end = window_start + timedelta(minutes=5)
    bet = {
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
    if actual_fill_price is not None:
        bet["actual_fill_price"] = actual_fill_price
    if actual_fill_amount is not None:
        bet["actual_fill_amount"] = actual_fill_amount
    if actual_filled_shares is not None:
        bet["actual_filled_shares"] = actual_filled_shares
    return bet


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
    assert live_profile["open_positions"][0]["submitted_entry_price"] == 0.8
    assert live_profile["open_positions"][0]["actual_fill_price"] is None
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


def test_api_btc5m_live_shows_submitted_and_actual_fill_prices(tmp_path, monkeypatch):
    live_ledger = tmp_path / "btc5m.json"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"

    _write_ledger(
        live_ledger,
        [
            _btc5m_bet(
                amount=4.99,
                price=0.94,
                shares=5.31,
                actual_fill_price=0.83,
                actual_fill_amount=4.45974,
                actual_filled_shares=5.31,
            )
        ],
    )

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", live_ledger)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    row = next(p for p in payload["profiles"] if p["profile"] == "late_capture")["open_positions"][0]
    assert row["submitted_entry_price"] == 0.94
    assert row["entry_price"] == 0.94
    assert row["actual_fill_price"] == 0.83
    assert row["actual_fill_amount"] == 4.45974
    assert row["risk_if_loss"] == 4.46
    assert row["win_pnl"] == 0.85
    assert payload["summary"]["open_exposure"] == 4.46


def test_api_btc5m_live_returns_full_btc_bet_history(tmp_path, monkeypatch):
    live_ledger = tmp_path / "btc5m.json"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    count = web_app.BTC5M_MONITOR_RECENT_TRADE_LIMIT + 3
    bets = []

    for index in range(count):
        bet = _btc5m_bet(
            bet_id=index + 1,
            status="won",
            amount=5.0,
            price=0.5,
            shares=10.0,
            result_pnl=5.0,
            actual_fill_price=0.49,
            actual_fill_amount=4.9,
            actual_filled_shares=10.0,
        )
        placed_at = start + timedelta(minutes=index)
        bet["placed_at"] = placed_at.isoformat()
        bets.append(bet)
    _write_ledger(live_ledger, bets)

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", live_ledger)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    profile = next(p for p in payload["profiles"] if p["profile"] == "late_capture")
    history = payload["bet_history"]
    assert len(profile["recent_trades"]) == web_app.BTC5M_MONITOR_RECENT_TRADE_LIMIT
    assert len(history["rows"]) == count
    assert history["rows"][0]["id"] == count
    assert history["rows"][0]["submitted_entry_price"] == 0.5
    assert history["rows"][0]["actual_fill_price"] == 0.49
    assert history["summary"]["total_trades"] == count
    assert history["summary"]["filled_trades"] == count
    assert history["summary"]["realized_pnl"] == count * 5.0
    assert history["summary"]["avg_attempted_entry_price"] == 0.5
    assert history["summary"]["avg_actual_fill_price"] == 0.49


def test_api_btc5m_live_enriches_fill_from_polymarket_activity(tmp_path, monkeypatch):
    live_ledger = tmp_path / "btc5m.json"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "103497057124497519444157624423416308759922196631593890511428621723186945106"

    bet = _btc5m_bet(
        amount=4.99,
        price=0.94,
        shares=5.31,
        actual_fill_price=0.83,
        actual_fill_amount=4.4073,
        actual_filled_shares=5.31,
    )
    bet["actual_fill_source"] = "clob_order_response"
    bet["token_id"] = token_id
    bet["market_slug"] = "btc-updown-5m-1782257100"
    bet["condition_id"] = "0xd57ee8c21c001514715d92e9dc627e97ebb93dd3290cdb8fcc169a60b3bcafe6"
    _write_ledger(live_ledger, [bet])

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "slug": "btc-updown-5m-1782257100",
                "size": 5.31,
                "usdcSize": 4.45974,
                "price": 0.83,
                "transactionHash": "0x55176ed",
                "timestamp": 1782257362,
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", live_ledger)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    row = next(p for p in payload["profiles"] if p["profile"] == "late_capture")["open_positions"][0]
    assert row["submitted_entry_price"] == 0.94
    assert row["actual_fill_price"] == 0.83
    assert row["actual_fill_avg_price"] == 0.8399
    assert row["actual_fill_amount"] == 4.45974
    assert row["actual_fill_source"] == "polymarket_activity"
    assert row["actual_fill_tx_hash"] == "0x55176ed"
    assert row["risk_if_loss"] == 4.46


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
    assert not [
        item
        for item in payload["freshness"]["ledgers"]
        if item["label"] == "configured" and item["source"] == "configured"
    ]


def test_api_btc5m_live_keeps_explicit_configured_ledger_with_live_profiles(
    tmp_path, monkeypatch
):
    configured_ledger = tmp_path / "configured_btc5m.json"
    live_dir = tmp_path / "btc5m_live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"

    _write_ledger(configured_ledger, [_btc5m_bet(profile="manual_configured")])

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", configured_ledger)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LEDGER_PATH", str(configured_ledger))
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    configured_freshness = [
        item
        for item in payload["freshness"]["ledgers"]
        if item["label"] == "configured" and item["source"] == "configured"
    ]
    assert len(configured_freshness) == 1
    assert configured_freshness[0]["status"] == "fresh"


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


def test_api_btc5m_live_returns_all_promoted_live_profiles_for_dashboard(tmp_path, monkeypatch):
    configured_missing = tmp_path / "missing_default.json"
    live_dir = tmp_path / "btc5m_live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "missing_signals.jsonl"
    existing_profile = "late_capture_gap005"

    _write_ledger(
        live_dir / f"{existing_profile}.json",
        [_btc5m_bet(profile=existing_profile, amount=5.0, price=0.90, shares=5.55)],
    )

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", configured_missing)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", ",".join(PROMOTED_BTC5M_LIVE_PROFILES))
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)
    web_app.set_runtime_status(
        {
            "service": "ufc-betting-bot",
            "ready": True,
            "components": {},
        }
    )

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    assert payload["config"]["live_profiles"] == PROMOTED_BTC5M_LIVE_PROFILES
    assert payload["summary"]["profile_count"] == len(PROMOTED_BTC5M_LIVE_PROFILES)
    profiles = {profile["profile"]: profile for profile in payload["profiles"]}
    assert set(profiles) == set(PROMOTED_BTC5M_LIVE_PROFILES)

    existing = profiles[existing_profile]
    assert existing["ledger_exists"] is True
    assert existing["mode"] == "live"
    assert existing["stats"]["total_bets"] == 1
    assert existing["stats"]["open_exposure"] == 5.0

    new_profile = profiles["late_capture_full_min88_liq"]
    assert new_profile["ledger_exists"] is False
    assert new_profile["ledger_label"] == "live:late_capture_full_min88_liq"
    assert new_profile["mode"] == "live"
    assert new_profile["stats"]["total_bets"] == 0
    assert new_profile["stats"]["open_exposure"] == 0.0

    live_ledger_labels = {
        item["label"]
        for item in payload["freshness"]["ledgers"]
        if item["source"] == "live"
    }
    assert live_ledger_labels == {
        f"live:{profile_name}" for profile_name in PROMOTED_BTC5M_LIVE_PROFILES
    }
    pending_alerts = [
        alert for alert in payload["alerts"]
        if alert["code"] in {"live_ledger_pending_first_write", "live_ledgers_pending_first_write"}
    ]
    assert len(pending_alerts) == 1
    assert pending_alerts[0]["code"] == "live_ledgers_pending_first_write"
    assert len(pending_alerts[0]["labels"]) == len(PROMOTED_BTC5M_LIVE_PROFILES) - 1


def test_api_btc5m_live_handles_alt_asset_profiles_for_dashboard(tmp_path, monkeypatch):
    configured_missing = tmp_path / "missing_default.json"
    live_dir = tmp_path / "btc5m_live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    eth_profile = "eth_late_capture_gap005"
    hype_profile = "hype_late_capture_gap005_min88"
    eth_bet = _btc5m_bet(profile=eth_profile, amount=10.0, price=0.90, shares=11.11)
    eth_bet["market_slug"] = "eth-updown-5m-1781986200"

    _write_ledger(live_dir / f"{eth_profile}.json", [eth_bet])
    _write_signal(signal_log, profile=hype_profile)

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", configured_missing)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", f"{eth_profile},{hype_profile}")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)
    web_app.set_runtime_status({"service": "ufc-betting-bot", "ready": True, "components": {}})

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    assert payload["config"]["live_profiles"] == [eth_profile, hype_profile]
    assert payload["config"]["live_assets"] == ["ETH", "HYPE"]
    profiles = {profile["profile"]: profile for profile in payload["profiles"]}
    assert profiles[eth_profile]["asset_symbol"] == "ETH"
    assert profiles[eth_profile]["market_slug_prefix"] == "eth-updown-5m"
    assert profiles[eth_profile]["profile_price_source"] == "binance"
    assert profiles[eth_profile]["profile_price_source_fallbacks"] == ["coinbase", "hyperliquid"]
    assert profiles[eth_profile]["stats"]["open_exposure"] == 10.0
    assert profiles[hype_profile]["asset_symbol"] == "HYPE"
    assert profiles[hype_profile]["profile_price_source"] == "hyperliquid"
    assert profiles[hype_profile]["profile_price_source_fallbacks"] == []
    assert profiles[hype_profile]["hyperliquid_coin"] == "@107"
    assert profiles[hype_profile]["coinbase_product_id"] == "HYPE-USD"
    assert profiles[hype_profile]["ledger_exists"] is False
    assert payload["bet_history"]["rows"][0]["asset_symbol"] == "ETH"
    assert payload["recent_signals"][-1]["asset_symbol"] == "HYPE"


def test_api_btc5m_live_returns_configured_signal_tail(tmp_path, monkeypatch):
    assert web_app.BTC5M_MONITOR_SIGNAL_LIMIT == 100

    configured_missing = tmp_path / "missing_default.json"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []

    for index in range(web_app.BTC5M_MONITOR_SIGNAL_LIMIT + 8):
        recorded_at = start + timedelta(seconds=index)
        rows.append(
            json.dumps(
                {
                    "recorded_at": recorded_at.isoformat(),
                    "profile": f"profile-{index}",
                    "market": {},
                    "price": {},
                    "signal": {
                        "action": "skip",
                        "direction": "up",
                        "reason_code": f"reason-{index}",
                    },
                    "result": {"status": "idle"},
                }
            )
        )
    signal_log.parent.mkdir(parents=True, exist_ok=True)
    signal_log.write_text("\n".join(rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", configured_missing)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "")
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    assert payload["config"]["signal_limit"] == 100
    assert len(payload["recent_signals"]) == web_app.BTC5M_MONITOR_SIGNAL_LIMIT
    assert payload["recent_signals"][0]["profile"] == "profile-8"
    assert payload["recent_signals"][0]["signal"]["entry_price"] is None
    assert payload["recent_signals"][-1]["profile"] == f"profile-{web_app.BTC5M_MONITOR_SIGNAL_LIMIT + 7}"


def test_btc5m_page_renders():
    response = web_app.app.test_client().get("/btc5m")

    assert response.status_code == 200
    assert b"Crypto 5m Live State" in response.data
    assert b"Profile Stats" in response.data
    assert b"profileStateFilter" in response.data
    assert b"Entry Ask" in response.data
    assert b"Trade History" in response.data
    assert b"Attempt" in response.data
    assert b"Last 100 snapshots" in response.data
