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
    monkeypatch.setattr(web_app, "_btc5m_fetch_clob_trade_history", lambda: [])
    monkeypatch.setattr(web_app, "_btc5m_official_winning_side_for_slug", lambda market_slug: None)
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
    actual_fill_source="polymarket_activity",
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
    if (
        actual_fill_source
        and (
            actual_fill_price is not None
            or actual_fill_amount is not None
            or actual_filled_shares is not None
        )
    ):
        bet["actual_fill_source"] = actual_fill_source
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


@pytest.mark.parametrize(
    "market_slug",
    [
        "btc-updown-5m-1781986200",
        "eth-updown-5m-1781986200",
        "sol-updown-5m-1781986200",
    ],
)
def test_btc5m_monitor_identifies_crypto_5m_slug_assets(market_slug):
    assert web_app._btc5m_is_bet({"market_slug": market_slug}) is True


def test_btc5m_monitor_excludes_xrp_slug_assets():
    assert web_app._btc5m_is_bet({"market_slug": "xrp-updown-5m-1781986200"}) is False


def test_api_btc5m_live_uses_polymarket_activity_history_and_excludes_xrp(tmp_path, monkeypatch):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"

    def buy(asset, *, condition, token, amount, shares):
        slug = f"{asset.lower()}-updown-5m-1781986200"
        return {
            "type": "TRADE",
            "side": "BUY",
            "outcome": "Up",
            "asset": token,
            "conditionId": condition,
            "slug": slug,
            "eventSlug": slug,
            "size": shares,
            "usdcSize": amount,
            "price": amount / shares,
            "transactionHash": f"0x{asset.lower()}buy",
            "timestamp": 1781986260,
            "title": f"{asset} Up or Down - test",
        }

    def redeem(asset, *, condition, shares):
        slug = f"{asset.lower()}-updown-5m-1781986200"
        return {
            "type": "REDEEM",
            "conditionId": condition,
            "slug": slug,
            "eventSlug": slug,
            "size": shares,
            "usdcSize": shares,
            "transactionHash": f"0x{asset.lower()}redeem",
            "timestamp": 1781986565,
            "title": f"{asset} Up or Down - test",
        }

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            buy("BTC", condition="btc-condition", token="btc-token", amount=9.0, shares=10.0),
            redeem("BTC", condition="btc-condition", shares=10.0),
            buy("ETH", condition="eth-condition", token="eth-token", amount=18.0, shares=20.0),
            redeem("ETH", condition="eth-condition", shares=20.0),
            buy("SOL", condition="sol-condition", token="sol-token", amount=27.0, shares=30.0),
            redeem("SOL", condition="sol-condition", shares=30.0),
            buy("XRP", condition="xrp-condition", token="xrp-token", amount=36.0, shares=40.0),
            redeem("XRP", condition="xrp-condition", shares=40.0),
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv(
        "BTC5M_LIVE_PROFILES",
        "late_capture_gap005,eth_late_capture_gap005,sol_late_capture_gap005",
    )
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    rows = payload["bet_history"]["rows"]
    audit_rows = payload["bet_history"]["unattributed_activity"]["rows"]

    assert rows == []
    assert {row["asset_symbol"] for row in audit_rows} == {"BTC", "ETH", "SOL"}
    assert all(row["actual_fill_source"] == "polymarket_activity" for row in audit_rows)
    assert all(row["history_source"] == "polymarket_activity" for row in audit_rows)
    assert all("xrp" not in str(row.get("market_slug") or "") for row in audit_rows)
    assert payload["bet_history"]["summary"]["total_trades"] == 0
    assert payload["bet_history"]["summary"]["filled_trades"] == 0
    assert payload["bet_history"]["summary"]["wins"] == 0
    assert payload["bet_history"]["summary"]["losses"] == 0
    assert payload["bet_history"]["summary"]["settled_trades"] == 0
    assert payload["bet_history"]["summary"]["realized_pnl"] == 0.0
    assert payload["summary"]["realized_pnl"] == 0.0
    assert payload["bet_history"]["unattributed_activity"]["summary"]["total_trades"] == 3
    assert {row["status"] for row in audit_rows} == {"won"}
    assert {row["settlement_state"] for row in audit_rows} == {"won"}
    assert payload["config"]["allowed_assets"] == ["BTC", "ETH", "SOL"]
    assert payload["summary"]["profile_count"] == 3
    assert {profile["profile"] for profile in payload["profiles"]} == {
        "late_capture_gap005",
        "eth_late_capture_gap005",
        "sol_late_capture_gap005",
    }


def test_api_btc5m_live_excludes_paper_ledgers_when_activity_history_is_available(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    slug = "btc-updown-5m-1781986200"

    _write_ledger(
        paper_dir / "paper_profile.json",
        [
            _btc5m_bet(
                profile="paper_profile",
                dry_run=True,
                status="won",
                amount=5.0,
                price=0.5,
                shares=10.0,
                result_pnl=5.0,
            )
        ],
    )
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": "btc-token",
                "conditionId": "btc-condition",
                "slug": slug,
                "eventSlug": slug,
                "size": 10.0,
                "usdcSize": 9.0,
                "price": 0.9,
                "transactionHash": "0xactivity",
                "timestamp": 1781986260,
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture_gap005")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    assert payload["bet_history"]["summary"]["total_trades"] == 0
    assert payload["bet_history"]["unattributed_activity"]["summary"]["total_trades"] == 1
    assert all(
        row["history_source"] == "polymarket_activity"
        for row in payload["bet_history"]["unattributed_activity"]["rows"]
    )
    paper_profile = next(profile for profile in payload["profiles"] if profile["profile"] == "paper_profile")
    assert paper_profile["stats"]["total_bets"] == 0
    assert paper_profile["stats"]["realized_pnl"] == 0.0


def test_api_btc5m_live_uses_official_resolution_for_both_side_redeems(tmp_path, monkeypatch):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    slug = "btc-updown-5m-1782266700"
    condition = "0xa1d9020b44ef25cf94184ab40d116f8bb4134260ec83e2fb0d48362b6772595b"
    up_token = "41012827539679491555432056939398489163662394337454019537863193444996277256370"
    down_token = "31042859049543281512502743517688352112483733846335868516283955979596592141992"

    monkeypatch.setattr(web_app, "_btc5m_official_winning_side_for_slug", lambda market_slug: "up")
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": up_token,
                "conditionId": condition,
                "slug": slug,
                "eventSlug": slug,
                "size": 21.23,
                "usdcSize": 19.6004,
                "price": 0.9232,
                "transactionHash": "0xup",
                "timestamp": 1782266989,
                "title": "Bitcoin Up or Down - June 23, 10:05PM-10:10PM ET",
            },
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Down",
                "asset": down_token,
                "conditionId": condition,
                "slug": slug,
                "eventSlug": slug,
                "size": 5.55,
                "usdcSize": 4.995,
                "price": 0.9,
                "transactionHash": "0xdown",
                "timestamp": 1782266891,
                "title": "Bitcoin Up or Down - June 23, 10:05PM-10:10PM ET",
            },
            {
                "type": "REDEEM",
                "conditionId": condition,
                "slug": slug,
                "eventSlug": slug,
                "size": 26.91,
                "usdcSize": 26.91,
                "price": 0,
                "transactionHash": "0xredeem",
                "timestamp": 1782267101,
                "title": "Bitcoin Up or Down - June 23, 10:05PM-10:10PM ET",
            },
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture_gap005")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    assert payload["bet_history"]["rows"] == []
    rows = {
        row["side"]: row
        for row in payload["bet_history"]["unattributed_activity"]["rows"]
    }

    assert rows["up"]["status"] == "won"
    assert rows["up"]["settlement_state"] == "won"
    assert rows["up"]["realized_pnl"] is None
    assert rows["down"]["status"] == "lost"
    assert rows["down"]["settlement_state"] == "lost"
    assert rows["down"]["realized_pnl"] is None
    assert payload["bet_history"]["summary"]["wins"] == 0
    assert payload["bet_history"]["summary"]["losses"] == 0
    assert payload["bet_history"]["summary"]["settled_trades"] == 0
    assert payload["bet_history"]["summary"]["realized_pnl"] == 0.0


def test_api_btc5m_live_does_not_infer_ambiguous_redeem_winner_without_official_resolution(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    slug = "btc-updown-5m-1782266700"
    condition = "btc-condition"

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": "up-token",
                "conditionId": condition,
                "slug": slug,
                "eventSlug": slug,
                "size": 10.0,
                "usdcSize": 9.4,
                "price": 0.94,
                "transactionHash": "0xup",
                "timestamp": 1782266891,
            },
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Down",
                "asset": "down-token",
                "conditionId": condition,
                "slug": slug,
                "eventSlug": slug,
                "size": 10.0,
                "usdcSize": 9.4,
                "price": 0.94,
                "transactionHash": "0xdown",
                "timestamp": 1782266892,
            },
            {
                "type": "REDEEM",
                "conditionId": condition,
                "slug": slug,
                "eventSlug": slug,
                "size": 10.0,
                "usdcSize": 10.0,
                "transactionHash": "0xredeem",
                "timestamp": 1782267101,
            },
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture_gap005")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    assert payload["bet_history"]["rows"] == []
    rows = {
        row["side"]: row
        for row in payload["bet_history"]["unattributed_activity"]["rows"]
    }

    assert rows["up"]["status"] == "pending"
    assert rows["down"]["status"] == "pending"
    assert rows["up"]["realized_pnl"] is None
    assert rows["down"]["realized_pnl"] is None
    assert payload["bet_history"]["summary"]["wins"] == 0
    assert payload["bet_history"]["summary"]["losses"] == 0
    assert payload["bet_history"]["summary"]["settled_trades"] == 0
    assert payload["bet_history"]["summary"]["realized_pnl"] == 0.0


def test_api_btc5m_live_does_not_mark_past_activity_loss_without_resolution(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    slug = "btc-updown-5m-1782266700"

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": "up-token",
                "conditionId": "btc-condition",
                "slug": slug,
                "eventSlug": slug,
                "size": 10.0,
                "usdcSize": 9.4,
                "price": 0.94,
                "transactionHash": "0xpending",
                "timestamp": 1782266891,
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture_gap005")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    assert payload["bet_history"]["rows"] == []
    row = payload["bet_history"]["unattributed_activity"]["rows"][0]

    assert row["status"] == "pending"
    assert row["settlement_state"] == "awaiting_settlement"
    assert row["realized_pnl"] is None
    assert payload["bet_history"]["summary"]["losses"] == 0
    assert payload["bet_history"]["summary"]["settled_trades"] == 0
    assert payload["bet_history"]["summary"]["realized_pnl"] == 0.0
    assert payload["summary"]["realized_pnl"] == 0.0


def test_api_btc5m_live_uses_official_resolution_without_redeem_activity(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    slug = "btc-updown-5m-1782266700"
    condition = "btc-condition"

    monkeypatch.setattr(web_app, "_btc5m_official_winning_side_for_slug", lambda market_slug: "up")
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": "up-token",
                "conditionId": condition,
                "slug": slug,
                "eventSlug": slug,
                "size": 10.0,
                "usdcSize": 9.4,
                "price": 0.94,
                "transactionHash": "0xup",
                "timestamp": 1782266891,
            },
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Down",
                "asset": "down-token",
                "conditionId": condition,
                "slug": slug,
                "eventSlug": slug,
                "size": 5.0,
                "usdcSize": 4.7,
                "price": 0.94,
                "transactionHash": "0xdown",
                "timestamp": 1782266892,
            },
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture_gap005")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    assert payload["bet_history"]["rows"] == []
    rows = {
        row["side"]: row
        for row in payload["bet_history"]["unattributed_activity"]["rows"]
    }

    assert rows["up"]["status"] == "won"
    assert rows["up"]["settlement_state"] == "won"
    assert rows["up"]["realized_pnl"] is None
    assert rows["down"]["status"] == "lost"
    assert rows["down"]["settlement_state"] == "lost"
    assert rows["down"]["realized_pnl"] is None
    assert payload["bet_history"]["summary"]["wins"] == 0
    assert payload["bet_history"]["summary"]["losses"] == 0
    assert payload["bet_history"]["summary"]["settled_trades"] == 0
    assert payload["bet_history"]["summary"]["realized_pnl"] == 0.0


def test_api_btc5m_live_keeps_recently_closed_activity_pending_until_settlement_delay(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    fixed_now = datetime(2026, 6, 26, 21, 5, 30, tzinfo=timezone.utc)
    window_start = datetime(2026, 6, 26, 21, 0, 0, tzinfo=timezone.utc)
    slug = f"btc-updown-5m-{int(window_start.timestamp())}"
    condition = "btc-condition"

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    monkeypatch.setattr(web_app, "datetime", FixedDateTime)
    monkeypatch.setattr(web_app, "BTC5M_MONITOR_SETTLEMENT_DELAY_SECONDS", 120.0)
    monkeypatch.setattr(web_app, "_btc5m_official_winning_side_for_slug", lambda market_slug: "up")
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Down",
                "asset": "down-token",
                "conditionId": condition,
                "slug": slug,
                "eventSlug": slug,
                "size": 10.0,
                "usdcSize": 9.4,
                "price": 0.94,
                "transactionHash": "0xrecent",
                "timestamp": int((window_start + timedelta(minutes=4, seconds=30)).timestamp()),
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture_gap005")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    assert payload["bet_history"]["rows"] == []
    row = payload["bet_history"]["unattributed_activity"]["rows"][0]

    assert row["status"] == "pending"
    assert row["settlement_state"] == "awaiting_settlement"
    assert row["realized_pnl"] is None
    assert payload["bet_history"]["summary"]["wins"] == 0
    assert payload["bet_history"]["summary"]["losses"] == 0
    assert payload["bet_history"]["summary"]["settled_trades"] == 0
    assert payload["bet_history"]["summary"]["realized_pnl"] == 0.0


def test_api_btc5m_live_reads_configured_and_paper_ledgers(tmp_path, monkeypatch):
    live_ledger = tmp_path / "configured_btc5m.json"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "btc5m_signal_snapshots.jsonl"

    _write_ledger(
        live_ledger,
        [
            _btc5m_bet(
                actual_fill_price=0.80,
                actual_fill_amount=10.0,
                actual_filled_shares=12.5,
            )
        ],
    )
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
    assert live_profile["open_positions"][0]["actual_fill_price"] == 0.8
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
        [
            _btc5m_bet(
                profile="ml_candidate_v1",
                amount=7.5,
                price=0.75,
                shares=10.0,
                actual_fill_price=0.75,
                actual_fill_amount=7.5,
                actual_filled_shares=10.0,
            )
        ],
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


@pytest.mark.parametrize("asset_slug", ["btc", "eth", "sol"])
def test_api_btc5m_live_excludes_unfilled_submitted_orders_from_trade_history(
    tmp_path,
    monkeypatch,
    asset_slug,
):
    live_ledger = tmp_path / "btc5m.json"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"

    bets = []
    for index in range(2):
        bet = _btc5m_bet(
            bet_id=index + 1,
            status="won",
            amount=50.0,
            price=0.94,
            shares=53.19,
            result_pnl=3.19,
        )
        bet["market_slug"] = f"{asset_slug}-updown-5m-1781986200"
        bet["placement_state"] = "submitted"
        bet["order_id"] = f"resting-order-{index + 1}"
        bets.append(bet)
    _write_ledger(live_ledger, bets)

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", live_ledger)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    profile = next(p for p in payload["profiles"] if p["profile"] == "late_capture")
    assert profile["recent_trades"] == []
    assert profile["stats"]["realized_pnl"] == 0.0
    assert profile["unconfirmed_order_count"] == 2
    assert len(profile["unconfirmed_orders"]) == 2
    assert payload["bet_history"]["rows"] == []
    assert payload["bet_history"]["summary"]["realized_pnl"] == 0.0
    assert payload["summary"]["realized_pnl"] == 0.0
    assert payload["summary"]["unconfirmed_live_orders"] == 2
    assert {
        alert["code"] for alert in payload["alerts"]
    } >= {"unconfirmed_live_crypto_settlements"}


@pytest.mark.parametrize("asset_slug", ["btc", "eth", "sol"])
@pytest.mark.parametrize("actual_fill_source", ["", "clob_order_response"])
def test_api_btc5m_live_excludes_ledger_only_fill_numbers_without_polymarket_source(
    tmp_path,
    monkeypatch,
    asset_slug,
    actual_fill_source,
):
    live_ledger = tmp_path / "btc5m.json"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"

    bet = _btc5m_bet(
        status="won",
        amount=50.0,
        price=0.94,
        shares=53.19,
        result_pnl=3.19,
        actual_fill_price=0.94,
        actual_fill_amount=50.0,
        actual_filled_shares=53.19,
        actual_fill_source=actual_fill_source,
    )
    bet["market_slug"] = f"{asset_slug}-updown-5m-1781986200"
    bet["placement_state"] = "filled"
    _write_ledger(live_ledger, [bet])

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", live_ledger)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    profile = next(p for p in payload["profiles"] if p["profile"] == "late_capture")
    assert profile["recent_trades"] == []
    assert profile["stats"]["realized_pnl"] == 0.0
    assert profile["unconfirmed_order_count"] == 1
    assert payload["bet_history"]["rows"] == []
    assert payload["summary"]["realized_pnl"] == 0.0


def test_api_btc5m_live_attributes_unique_market_activity_to_ledger_order(tmp_path, monkeypatch):
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
    bet["order_id"] = "unique-order-1"
    _write_ledger(live_ledger, [bet])
    monkeypatch.setattr(web_app, "_btc5m_official_winning_side_for_slug", lambda market_slug: "up")

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "conditionId": "0xd57ee8c21c001514715d92e9dc627e97ebb93dd3290cdb8fcc169a60b3bcafe6",
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

    row = next(row for row in payload["bet_history"]["rows"] if row["actual_fill_tx_hash"] == "0x55176ed")
    assert row["profile"] == "late_capture"
    assert row["profile_attribution_source"] == "ledger_market_match"
    assert row["order_id"] == "unique-order-1"
    assert row["ledger_bet_id"] == 1
    assert row["submitted_entry_price"] == 0.94
    assert row["actual_fill_price"] == 0.83
    assert row["actual_fill_avg_price"] == 0.8399
    assert row["actual_fill_amount"] == 4.45974
    assert row["actual_fill_source"] == "polymarket_activity"
    assert row["actual_fill_tx_hash"] == "0x55176ed"
    assert row["risk_if_loss"] == 4.46
    assert row["status"] == "won"
    assert row["realized_pnl"] == 0.85
    assert payload["bet_history"]["unattributed_activity"]["rows"] == []
    assert payload["bet_history"]["summary"]["total_trades"] == 1
    assert payload["bet_history"]["summary"]["realized_pnl"] == 0.85


def test_api_btc5m_live_matches_activity_by_fill_hash_across_profiles(tmp_path, monkeypatch):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "shared-up-token"
    market_slug = "btc-updown-5m-1782257100"
    condition_id = "0xd57ee8c21c001514715d92e9dc627e97ebb93dd3290cdb8fcc169a60b3bcafe6"

    bet_one = _btc5m_bet(
        profile="late_capture",
        amount=9.3,
        price=0.93,
        shares=10.0,
        actual_fill_price=0.93,
        actual_fill_amount=9.3,
        actual_filled_shares=10.0,
    )
    bet_one.update(
        {
            "token_id": token_id,
            "market_slug": market_slug,
            "condition_id": condition_id,
            "actual_fill_source": "clob_order_response",
            "actual_fill_tx_hash": "0xaaa",
            "actual_fill_tx_hashes": ["0xaaa"],
        }
    )
    bet_two = _btc5m_bet(
        bet_id=2,
        profile="late_capture_min88",
        amount=9.4,
        price=0.94,
        shares=10.0,
        actual_fill_price=0.94,
        actual_fill_amount=9.4,
        actual_filled_shares=10.0,
    )
    bet_two.update(
        {
            "token_id": token_id,
            "market_slug": market_slug,
            "condition_id": condition_id,
            "actual_fill_source": "clob_order_response",
            "actual_fill_tx_hash": "0xbbb",
            "actual_fill_tx_hashes": ["0xbbb"],
        }
    )
    _write_ledger(live_dir / "late_capture.json", [bet_one])
    _write_ledger(live_dir / "late_capture_min88.json", [bet_two])

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "conditionId": condition_id,
                "slug": market_slug,
                "size": 5.0,
                "usdcSize": 4.7,
                "price": 0.94,
                "transactionHash": "0xaaa",
                "timestamp": 1782257362,
            },
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "slug": market_slug,
                "size": 10.6,
                "usdcSize": 10.03,
                "price": 0.9462,
                "transactionHash": "0xbbb",
                "timestamp": 1782257363,
            },
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture,late_capture_min88")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    rows = {
        row["profile"]: row
        for row in payload["bet_history"]["rows"]
        if row["profile"] in {"late_capture", "late_capture_min88"}
    }

    assert rows["late_capture"]["actual_fill_source"] == "polymarket_activity"
    assert rows["late_capture"]["actual_fill_amount"] == 4.7
    assert rows["late_capture"]["actual_filled_shares"] == 5.0
    assert rows["late_capture"]["risk_if_loss"] == 4.7
    assert rows["late_capture_min88"]["actual_fill_source"] == "polymarket_activity"
    assert rows["late_capture_min88"]["actual_fill_amount"] == 10.03
    assert rows["late_capture_min88"]["actual_filled_shares"] == 10.6
    assert rows["late_capture_min88"]["risk_if_loss"] == 10.03
    assert not rows["late_capture"].get("actual_fill_ambiguous")
    assert not rows["late_capture_min88"].get("actual_fill_ambiguous")


def test_api_btc5m_live_does_not_duplicate_ambiguous_market_activity(tmp_path, monkeypatch):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "shared-up-token"
    market_slug = "btc-updown-5m-1782257100"
    condition_id = "0xd57ee8c21c001514715d92e9dc627e97ebb93dd3290cdb8fcc169a60b3bcafe6"

    bet_one = _btc5m_bet(
        profile="late_capture",
        amount=5.0,
        price=0.94,
        shares=5.31,
        actual_fill_price=0.94,
        actual_fill_amount=4.7,
        actual_filled_shares=5.0,
        actual_fill_source="clob_order_response",
    )
    bet_one.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    bet_two = _btc5m_bet(
        bet_id=2,
        profile="late_capture_min88",
        amount=10.0,
        price=0.94,
        shares=10.63,
        actual_fill_price=0.9462,
        actual_fill_amount=10.03,
        actual_filled_shares=10.6,
        actual_fill_source="clob_order_response",
    )
    bet_two.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    _write_ledger(live_dir / "late_capture.json", [bet_one])
    _write_ledger(live_dir / "late_capture_min88.json", [bet_two])

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "slug": market_slug,
                "size": 21.26,
                "usdcSize": 20.03,
                "price": 0.9421,
                "transactionHash": "0xaggregate",
                "timestamp": 1782257362,
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture,late_capture_min88")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    rows = [
        row for row in payload["bet_history"]["unattributed_activity"]["rows"]
        if row["actual_fill_tx_hash"] == "0xaggregate"
    ]

    assert payload["bet_history"]["rows"] == []
    assert len(rows) == 1
    assert rows[0]["profile"] is None
    assert rows[0]["profile_attribution_source"] == "unattributed"
    assert rows[0]["actual_fill_amount"] == 20.03
    assert rows[0]["actual_filled_shares"] == 21.26
    assert rows[0]["risk_if_loss"] == 20.03


def test_api_btc5m_live_attributes_activity_with_clob_maker_order_id(tmp_path, monkeypatch):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "shared-up-token"
    market_slug = "btc-updown-5m-1782257100"
    condition_id = "0xd57ee8c21c001514715d92e9dc627e97ebb93dd3290cdb8fcc169a60b3bcafe6"

    bet_one = _btc5m_bet(profile="late_capture", bet_id=1, amount=5.0, shares=5.3)
    bet_one.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    bet_two = _btc5m_bet(profile="late_capture_min88", bet_id=2, amount=10.0, shares=10.6)
    bet_two.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    _write_ledger(live_dir / "late_capture.json", [bet_one])
    _write_ledger(live_dir / "late_capture_min88.json", [bet_two])

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "slug": market_slug,
                "size": 10.6,
                "usdcSize": 10.03,
                "price": 0.9462,
                "transactionHash": "0xmaker",
                "timestamp": 1782257363,
            }
        ],
    )
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_clob_trade_history",
        lambda: [
            {
                "transaction_hash": "0xmaker",
                "status": "CONFIRMED",
                "maker_orders": [
                    {
                        "order_id": "order-2",
                        "asset_id": token_id,
                        "matched_amount": "10.6",
                        "price": "0.9462",
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture,late_capture_min88")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    row = next(row for row in payload["bet_history"]["rows"] if row["actual_fill_tx_hash"] == "0xmaker")

    assert row["profile"] == "late_capture_min88"
    assert row["profile_attribution_source"] == "clob_order_history"
    assert row["order_id"] == "order-2"
    assert row["clob_order_ids"] == ["order-2"]
    assert row["ledger_bet_id"] == 2
    profile = next(item for item in payload["profiles"] if item["profile"] == "late_capture_min88")
    assert profile["stats"]["total_bets"] == 1
    assert profile["unconfirmed_order_count"] == 0
    assert profile["recent_trades"][0]["order_id"] == "order-2"


def test_api_btc5m_live_assigns_duplicate_activity_rows_to_clob_orders(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "duplicate-clob-up-token"
    market_slug = "btc-updown-5m-1782257300"
    condition_id = "0xdedede8c21c001514715d92e9dc627e97ebb93dd3290cdb8fcc169a60b3bc"

    bet_one = _btc5m_bet(profile="late_capture", bet_id=1, amount=10.0, shares=10.6)
    bet_one.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    bet_two = _btc5m_bet(profile="late_capture_min88", bet_id=2, amount=10.0, shares=10.6)
    bet_two.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    _write_ledger(live_dir / "late_capture.json", [bet_one])
    _write_ledger(live_dir / "late_capture_min88.json", [bet_two])
    monkeypatch.setattr(web_app, "_btc5m_official_winning_side_for_slug", lambda market_slug: "up")

    duplicate_activity = {
        "type": "TRADE",
        "side": "BUY",
        "outcome": "Up",
        "asset": token_id,
        "conditionId": condition_id,
        "slug": market_slug,
        "eventSlug": market_slug,
        "size": 10.6,
        "usdcSize": 10.03,
        "price": 0.9462,
        "transactionHash": "0xduplicate-maker",
        "timestamp": 1782257363,
    }
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [dict(duplicate_activity), dict(duplicate_activity)],
    )
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_clob_trade_history",
        lambda: [
            {
                "transaction_hash": "0xduplicate-maker",
                "status": "CONFIRMED",
                "maker_orders": [
                    {
                        "order_id": "order-1",
                        "asset_id": token_id,
                        "matched_amount": "10.6",
                        "price": "0.9462",
                    },
                    {
                        "order_id": "order-2",
                        "asset_id": token_id,
                        "matched_amount": "10.6",
                        "price": "0.9462",
                    },
                ],
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture,late_capture_min88")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    rows = [
        row for row in payload["bet_history"]["rows"]
        if row["actual_fill_tx_hash"] == "0xduplicate-maker"
    ]

    assert len(rows) == 2
    assert {row["id"] for row in rows} and len({row["id"] for row in rows}) == 2
    assert {row["profile"] for row in rows} == {"late_capture", "late_capture_min88"}
    assert {row["order_id"] for row in rows} == {"order-1", "order-2"}
    assert all(row["profile_attribution_source"] == "clob_order_history" for row in rows)
    assert all(row["actual_fill_source"] == "clob_trade_history" for row in rows)
    assert all(row["actual_fill_amount"] == 10.02972 for row in rows)
    assert payload["bet_history"]["summary"]["total_trades"] == 2
    profile_stats = {item["profile"]: item["stats"] for item in payload["profiles"]}
    assert profile_stats["late_capture"]["total_bets"] == 1
    assert profile_stats["late_capture_min88"]["total_bets"] == 1


def test_api_btc5m_live_keeps_unmatched_duplicate_activity_row(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "extra-duplicate-clob-up-token"
    market_slug = "btc-updown-5m-1782257350"
    condition_id = "0xdfdfdf8c21c001514715d92e9dc627e97ebb93dd3290cdb8fcc169a60b3bc"

    bet = _btc5m_bet(profile="late_capture", bet_id=1, amount=10.0, shares=10.6)
    bet.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    _write_ledger(live_dir / "late_capture.json", [bet])
    monkeypatch.setattr(web_app, "_btc5m_official_winning_side_for_slug", lambda market_slug: "up")

    duplicate_activity = {
        "type": "TRADE",
        "side": "BUY",
        "outcome": "Up",
        "asset": token_id,
        "conditionId": condition_id,
        "slug": market_slug,
        "eventSlug": market_slug,
        "size": 10.6,
        "usdcSize": 10.03,
        "price": 0.9462,
        "transactionHash": "0xextra-duplicate-maker",
        "timestamp": 1782257363,
    }
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [dict(duplicate_activity), dict(duplicate_activity)],
    )
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_clob_trade_history",
        lambda: [
            {
                "transaction_hash": "0xextra-duplicate-maker",
                "status": "CONFIRMED",
                "maker_orders": [
                    {
                        "order_id": "order-1",
                        "asset_id": token_id,
                        "matched_amount": "10.6",
                        "price": "0.9462",
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    rows = [
        row for row in payload["bet_history"]["rows"]
        if row["actual_fill_tx_hash"] == "0xextra-duplicate-maker"
    ]
    audit_rows = [
        row for row in payload["bet_history"]["unattributed_activity"]["rows"]
        if row["actual_fill_tx_hash"] == "0xextra-duplicate-maker"
    ]

    assert len(rows) == 1
    assert len(audit_rows) == 1
    attributed = rows[0]
    unattributed = audit_rows[0]
    assert attributed["actual_fill_source"] == "clob_trade_history"
    assert unattributed["actual_fill_source"] == "polymarket_activity"
    assert attributed["profile"] == "late_capture"
    assert attributed["order_id"] == "order-1"
    assert unattributed["profile"] is None
    assert unattributed["profile_attribution_source"] == "unattributed"
    assert payload["bet_history"]["summary"]["total_trades"] == 1


def test_api_btc5m_live_attributes_activity_with_clob_taker_order_id(tmp_path, monkeypatch):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "taker-up-token"
    market_slug = "btc-updown-5m-1782257400"
    condition_id = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    bet = _btc5m_bet(profile="late_capture", bet_id=1, amount=7.0, shares=7.45)
    bet.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    _write_ledger(live_dir / "late_capture.json", [bet])

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "slug": market_slug,
                "size": 7.45,
                "usdcSize": 6.99555,
                "price": 0.939,
                "transactionHash": "0xtaker",
                "timestamp": 1782257463,
            }
        ],
    )
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_clob_trade_history",
        lambda: [
            {
                "transaction_hash": "0xtaker",
                "status": "CONFIRMED",
                "taker_order_id": "order-1",
                "asset_id": token_id,
                "side": "BUY",
                "size": "7.45",
                "price": "0.939",
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    row = next(row for row in payload["bet_history"]["rows"] if row["actual_fill_tx_hash"] == "0xtaker")

    assert row["profile"] == "late_capture"
    assert row["profile_attribution_source"] == "clob_order_history"
    assert row["order_id"] == "order-1"
    assert row["clob_fill_roles"] == ["taker"]
    assert row["ledger_bet_id"] == 1
    profile = next(item for item in payload["profiles"] if item["profile"] == "late_capture")
    assert profile["unconfirmed_order_count"] == 0


def test_api_btc5m_live_aggregates_partial_activity_fills_by_clob_order_id(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "partial-up-token"
    market_slug = "btc-updown-5m-1782257400"
    condition_id = "0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"

    bet = _btc5m_bet(profile="late_capture", bet_id=1, amount=50.0, price=0.94, shares=53.19)
    bet.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    _write_ledger(live_dir / "late_capture.json", [bet])
    monkeypatch.setattr(web_app, "_btc5m_official_winning_side_for_slug", lambda market_slug: "up")

    partials = [
        ("0xpart1", 4.0, 3.76),
        ("0xpart2", 3.0, 2.82),
        ("0xpart3", 3.0, 2.82),
    ]
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "conditionId": condition_id,
                "slug": market_slug,
                "eventSlug": market_slug,
                "size": shares,
                "usdcSize": amount,
                "price": 0.94,
                "transactionHash": tx_hash,
                "timestamp": 1782257463 + index,
            }
            for index, (tx_hash, shares, amount) in enumerate(partials)
        ],
    )
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_clob_trade_history",
        lambda: [
            {
                "transaction_hash": tx_hash,
                "status": "CONFIRMED",
                "taker_order_id": "order-1",
                "asset_id": token_id,
                "side": "BUY",
                "size": str(shares),
                "price": "0.94",
            }
            for tx_hash, shares, _amount in partials
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    rows = [row for row in payload["bet_history"]["rows"] if row["order_id"] == "order-1"]

    assert len(rows) == 1
    row = rows[0]
    assert row["profile"] == "late_capture"
    assert row["profile_attribution_source"] == "clob_order_history"
    assert row["ledger_bet_id"] == 1
    assert row["actual_fill_source"] == "clob_trade_history"
    assert row["actual_fill_amount"] == 9.4
    assert row["actual_filled_shares"] == 10.0
    assert row["actual_fill_avg_price"] == 0.94
    assert row["submitted_amount"] == 50.0
    assert row["submitted_shares"] == 53.19
    assert row["actual_fill_status"] == "partial"
    assert row["activity_fill_count"] == 3
    assert row["aggregated_partial_fills"] is True
    assert row["actual_fill_tx_hashes"] == ["0xpart1", "0xpart2", "0xpart3"]
    assert row["status"] == "won"
    assert row["realized_pnl"] == 0.6
    assert payload["bet_history"]["summary"]["total_trades"] == 1
    assert payload["bet_history"]["summary"]["filled_trades"] == 1

    profile = next(item for item in payload["profiles"] if item["profile"] == "late_capture")
    assert profile["stats"]["total_bets"] == 1
    assert profile["stats"]["wins"] == 1
    assert profile["stats"]["realized_pnl"] == 0.6
    assert profile["unconfirmed_order_count"] == 0


def test_api_btc5m_live_does_not_aggregate_unattributed_activity_rows_by_transaction_hash(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "duplicate-activity-up-token"
    market_slug = "btc-updown-5m-1782257500"
    condition_id = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"

    monkeypatch.setattr(web_app, "_btc5m_official_winning_side_for_slug", lambda market_slug: "up")
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "conditionId": condition_id,
                "slug": market_slug,
                "eventSlug": market_slug,
                "size": 4.0,
                "usdcSize": 3.76,
                "price": 0.94,
                "transactionHash": "0xduplicate-tx",
                "timestamp": 1782257563,
            },
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "conditionId": condition_id,
                "slug": market_slug,
                "eventSlug": market_slug,
                "size": 6.0,
                "usdcSize": 5.64,
                "price": 0.94,
                "transactionHash": "0xduplicate-tx",
                "timestamp": 1782257563,
            },
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    rows = [
        row for row in payload["bet_history"]["unattributed_activity"]["rows"]
        if row["actual_fill_tx_hash"] == "0xduplicate-tx"
    ]

    assert payload["bet_history"]["rows"] == []
    assert len(rows) == 2
    assert {row["actual_fill_amount"] for row in rows} == {3.76, 5.64}
    assert {row["actual_filled_shares"] for row in rows} == {4.0, 6.0}
    assert all(row["profile"] is None for row in rows)
    assert all(row["profile_attribution_source"] == "unattributed" for row in rows)
    assert payload["bet_history"]["summary"]["total_trades"] == 0
    assert payload["bet_history"]["unattributed_activity"]["summary"]["total_trades"] == 2


def test_api_btc5m_live_splits_multi_profile_clob_aggregate(tmp_path, monkeypatch):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "aggregate-up-token"
    market_slug = "btc-updown-5m-1782257700"
    condition_id = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    bet_one = _btc5m_bet(profile="late_capture", bet_id=1, amount=5.0, shares=5.0)
    bet_one.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    bet_two = _btc5m_bet(profile="late_capture_min88", bet_id=2, amount=6.0, shares=6.0)
    bet_two.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    _write_ledger(live_dir / "late_capture.json", [bet_one])
    _write_ledger(live_dir / "late_capture_min88.json", [bet_two])

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "slug": market_slug,
                "size": 11.0,
                "usdcSize": 10.34,
                "price": 0.94,
                "transactionHash": "0xaggregate-clob",
                "timestamp": 1782257763,
            }
        ],
    )
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_clob_trade_history",
        lambda: [
            {
                "transaction_hash": "0xaggregate-clob",
                "status": "CONFIRMED",
                "maker_orders": [
                    {
                        "order_id": "order-1",
                        "asset_id": token_id,
                        "side": "BUY",
                        "matched_amount": "5",
                        "price": "0.94",
                    },
                    {
                        "order_id": "order-2",
                        "asset_id": token_id,
                        "side": "BUY",
                        "matched_amount": "6",
                        "price": "0.94",
                    },
                ],
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture,late_capture_min88")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    rows = [
        row for row in payload["bet_history"]["rows"]
        if row["actual_fill_tx_hash"] == "0xaggregate-clob"
    ]

    assert len(rows) == 2
    assert {row["profile"] for row in rows} == {"late_capture", "late_capture_min88"}
    assert {row["order_id"] for row in rows} == {"order-1", "order-2"}
    assert {row["actual_fill_amount"] for row in rows} == {4.7, 5.64}
    assert all(row["actual_fill_source"] == "clob_trade_history" for row in rows)
    assert all(row["profile_attribution_source"] == "clob_order_history" for row in rows)
    assert payload["summary"]["profile_count"] == 2


def test_api_btc5m_live_splits_same_profile_multi_order_clob_aggregate(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "same-profile-aggregate-up-token"
    market_slug = "btc-updown-5m-1782257800"
    condition_id = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

    bet_one = _btc5m_bet(profile="late_capture", bet_id=1, amount=5.0, shares=5.0)
    bet_one.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    bet_two = _btc5m_bet(profile="late_capture", bet_id=2, amount=6.0, shares=6.0)
    bet_two.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    _write_ledger(live_dir / "late_capture.json", [bet_one, bet_two])

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "conditionId": condition_id,
                "slug": market_slug,
                "eventSlug": market_slug,
                "size": 11.0,
                "usdcSize": 10.34,
                "price": 0.94,
                "transactionHash": "0xsame-profile-aggregate",
                "timestamp": 1782257863,
            }
        ],
    )
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_clob_trade_history",
        lambda: [
            {
                "transaction_hash": "0xsame-profile-aggregate",
                "status": "CONFIRMED",
                "maker_orders": [
                    {
                        "order_id": "order-1",
                        "asset_id": token_id,
                        "side": "BUY",
                        "matched_amount": "5",
                        "price": "0.94",
                    },
                    {
                        "order_id": "order-2",
                        "asset_id": token_id,
                        "side": "BUY",
                        "matched_amount": "6",
                        "price": "0.94",
                    },
                ],
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    rows = [
        row for row in payload["bet_history"]["rows"]
        if row["actual_fill_tx_hash"] == "0xsame-profile-aggregate"
    ]

    assert len(rows) == 2
    assert {row["profile"] for row in rows} == {"late_capture"}
    assert {row["order_id"] for row in rows} == {"order-1", "order-2"}
    assert {row["actual_fill_amount"] for row in rows} == {4.7, 5.64}
    assert all(row["actual_fill_source"] == "clob_trade_history" for row in rows)
    assert all(row["profile_attribution_source"] == "clob_order_history" for row in rows)
    profile = next(item for item in payload["profiles"] if item["profile"] == "late_capture")
    assert profile["stats"]["total_bets"] == 2
    assert profile["unconfirmed_order_count"] == 0


def test_api_btc5m_live_does_not_guess_duplicate_local_order_id_attribution(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "live"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "duplicate-order-up-token"
    market_slug = "btc-updown-5m-1782257900"
    condition_id = "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

    bet_one = _btc5m_bet(profile="late_capture", bet_id=1, amount=5.0, shares=5.0)
    bet_one.update(
        {
            "order_id": "shared-order",
            "token_id": token_id,
            "market_slug": market_slug,
            "condition_id": condition_id,
        }
    )
    bet_two = _btc5m_bet(profile="late_capture_min88", bet_id=2, amount=6.0, shares=6.0)
    bet_two.update(
        {
            "order_id": "shared-order",
            "token_id": token_id,
            "market_slug": market_slug,
            "condition_id": condition_id,
        }
    )
    _write_ledger(live_dir / "late_capture.json", [bet_one])
    _write_ledger(live_dir / "late_capture_min88.json", [bet_two])

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "conditionId": condition_id,
                "slug": market_slug,
                "eventSlug": market_slug,
                "size": 5.0,
                "usdcSize": 4.7,
                "price": 0.94,
                "transactionHash": "0xduplicate-order",
                "timestamp": 1782257963,
            }
        ],
    )
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_clob_trade_history",
        lambda: [
            {
                "transaction_hash": "0xduplicate-order",
                "status": "CONFIRMED",
                "taker_order_id": "shared-order",
                "asset_id": token_id,
                "side": "BUY",
                "size": "5",
                "price": "0.94",
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture,late_capture_min88")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    assert payload["bet_history"]["rows"] == []
    row = next(
        row for row in payload["bet_history"]["unattributed_activity"]["rows"]
        if row["actual_fill_tx_hash"] == "0xduplicate-order"
    )

    assert row["profile"] is None
    assert row["profile_attribution_source"] == "unattributed"


def test_api_btc5m_live_does_not_create_profile_card_for_historical_clob_profile(
    tmp_path,
    monkeypatch,
):
    live_dir = tmp_path / "logs" / "btc5m_live"
    backup_dir = tmp_path / "logs" / "btc5m_live_backups" / "backup-1"
    paper_dir = tmp_path / "paper"
    signal_log = tmp_path / "signals.jsonl"
    token_id = "historical-up-token"
    market_slug = "btc-updown-5m-1782258000"
    condition_id = "0xcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"

    _write_ledger(live_dir / "late_capture.json", [])
    historical_bet = _btc5m_bet(profile="late_capture_min90", bet_id=90, amount=4.0, shares=4.5)
    historical_bet.update({"token_id": token_id, "market_slug": market_slug, "condition_id": condition_id})
    historical_bet["order_id"] = "order-90"
    _write_ledger(backup_dir / "late_capture_min90.json", [historical_bet])

    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_trade_activity",
        lambda: [
            {
                "type": "TRADE",
                "side": "BUY",
                "outcome": "Up",
                "asset": token_id,
                "conditionId": condition_id,
                "slug": market_slug,
                "size": 4.5,
                "usdcSize": 4.23,
                "price": 0.94,
                "transactionHash": "0xhistorical",
                "timestamp": 1782258063,
            }
        ],
    )
    monkeypatch.setattr(
        web_app,
        "_btc5m_fetch_clob_trade_history",
        lambda: [
            {
                "transaction_hash": "0xhistorical",
                "status": "CONFIRMED",
                "maker_orders": [
                    {
                        "order_id": "order-90",
                        "asset_id": token_id,
                        "side": "BUY",
                        "matched_amount": "4.5",
                        "price": "0.94",
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", tmp_path / "missing_configured.json")
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", "late_capture")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()
    assert payload["bet_history"]["rows"] == []
    row = next(
        row for row in payload["bet_history"]["unattributed_activity"]["rows"]
        if row["actual_fill_tx_hash"] == "0xhistorical"
    )

    assert row["profile"] == "late_capture_min90"
    assert row["profile_attribution_source"] == "clob_order_history"
    assert payload["summary"]["profile_count"] == 1
    assert [profile["profile"] for profile in payload["profiles"]] == ["late_capture"]


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
        [
            _btc5m_bet(
                profile=existing_profile,
                amount=5.0,
                price=0.90,
                shares=5.55,
                actual_fill_price=0.90,
                actual_fill_amount=5.0,
                actual_filled_shares=5.55,
            )
        ],
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
    sol_profile = "sol_late_capture_gap005_min88"
    eth_bet = _btc5m_bet(
        profile=eth_profile,
        amount=10.0,
        price=0.90,
        shares=11.11,
        actual_fill_price=0.90,
        actual_fill_amount=10.0,
        actual_filled_shares=11.11,
    )
    eth_bet["market_slug"] = "eth-updown-5m-1781986200"

    _write_ledger(live_dir / f"{eth_profile}.json", [eth_bet])
    _write_signal(signal_log, profile=sol_profile)

    monkeypatch.setattr(web_app, "BTC5M_LEDGER_PATH", configured_missing)
    monkeypatch.setattr(web_app, "BTC5M_PAPER_LEDGER_DIR", paper_dir)
    monkeypatch.setattr(web_app, "BTC5M_SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setenv("BTC5M_LIVE_PROFILES", f"{eth_profile},{sol_profile}")
    monkeypatch.setenv("BTC5M_LIVE_LEDGER_DIR", str(live_dir))
    monkeypatch.delenv(web_app.BTC5M_MONITOR_LEDGER_ENV, raising=False)
    web_app.set_runtime_status({"service": "ufc-betting-bot", "ready": True, "components": {}})

    payload = web_app.app.test_client().get("/api/btc5m/live").get_json()

    assert payload["config"]["live_profiles"] == [eth_profile, sol_profile]
    assert payload["config"]["live_assets"] == ["ETH", "SOL"]
    profiles = {profile["profile"]: profile for profile in payload["profiles"]}
    assert profiles[eth_profile]["asset_symbol"] == "ETH"
    assert profiles[eth_profile]["market_slug_prefix"] == "eth-updown-5m"
    assert profiles[eth_profile]["profile_price_source"] == "binance"
    assert profiles[eth_profile]["profile_price_source_fallbacks"] == ["coinbase", "hyperliquid"]
    assert profiles[eth_profile]["stats"]["open_exposure"] == 10.0
    assert profiles[sol_profile]["asset_symbol"] == "SOL"
    assert profiles[sol_profile]["profile_price_source"] == "binance"
    assert profiles[sol_profile]["profile_price_source_fallbacks"] == ["coinbase", "hyperliquid"]
    assert profiles[sol_profile]["hyperliquid_coin"] == "SOL"
    assert profiles[sol_profile]["coinbase_product_id"] == "SOL-USD"
    assert profiles[sol_profile]["ledger_exists"] is False
    assert payload["bet_history"]["rows"][0]["asset_symbol"] == "ETH"
    assert payload["recent_signals"][-1]["asset_symbol"] == "SOL"


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
