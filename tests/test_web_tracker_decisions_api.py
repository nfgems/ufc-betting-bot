from types import SimpleNamespace

import pytest

from src.web import app as web_app


@pytest.fixture(autouse=True)
def _reset_dashboard_host(monkeypatch):
    monkeypatch.setattr(web_app, "_server_host", "127.0.0.1")


def test_api_tracker_decisions_uses_card_day_keys_and_ignores_tracker_only_rows(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "_load_prediction_payload",
        lambda include_global_feature_importance=False: {
            "predictions": [
                {
                    "fighter_a": "Dominick Reyes",
                    "fighter_b": "Johnny Walker",
                    "event_date": "2026-04-12T01:45:00+00:00",
                    "weight_class": "Light Heavyweight",
                }
            ]
        },
    )
    monkeypatch.setattr("src.strategy.llm_operator.load_decision_log", lambda: [])
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_tracker_decision_log",
        lambda: [
            {
                "type": "decision",
                "timestamp": "2026-04-06T21:10:24.545782+00:00",
                "trader": "M",
                "decision_id": "M_1",
                "fighter_a": "Dominick Reyes",
                "fighter_b": "Johnny Walker",
                "event_date": "2026-04-12T01:45:00+00:00",
                "market_event_date": "2026-04-11 17:00:00+00",
                "event_title": "2026-05-17",
                "status": "outside_window",
                "summary": "Outside tracker window",
                "rationale": "Model Tracker skipped this fight because it is outside the tracker window.",
            },
            {
                "type": "decision",
                "timestamp": "2026-04-06T21:10:24.861192+00:00",
                "trader": "G",
                "decision_id": "G_1",
                "fighter_a": "Dominick Reyes",
                "fighter_b": "Johnny Walker",
                "event_date": "2026-04-12T01:45:00+00:00",
                "market_event_date": "2026-04-11 17:00:00+00",
                "event_title": "2026-05-17",
                "status": "outside_window",
                "summary": "Outside tracker window",
                "rationale": "Gemini Tracker skipped this fight because it is outside the tracker window.",
            },
            {
                "type": "decision",
                "timestamp": "2026-04-06T21:10:24.561629+00:00",
                "trader": "M",
                "decision_id": "M_2",
                "fighter_a": "Ronda Rousey",
                "fighter_b": "Gina Carano",
                "event_date": "2026-05-17T00:00:00+00:00",
                "market_event_date": "2026-05-17 01:00:00+00",
                "event_title": "2026-05-17",
                "status": "outside_window",
                "summary": "Outside tracker window",
                "rationale": "Model Tracker skipped this fight because it is outside the tracker window.",
            },
        ],
    )
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: SimpleNamespace(bets=[]),
    )

    client = web_app.app.test_client()
    response = client.get("/api/tracker-decisions")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1

    fight = payload["fights"][0]
    assert fight["fighter_a"] == "Dominick Reyes"
    assert fight["fighter_b"] == "Johnny Walker"
    assert fight["event_group_date"] == "2026-04-11"
    assert fight["M"]["status"] == "outside_window"
    assert fight["G"]["status"] == "outside_window"


def test_api_tracker_decisions_merges_ledger_bets_onto_tracker_card_day(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "_load_prediction_payload",
        lambda include_global_feature_importance=False: {
            "predictions": [
                {
                    "fighter_a": "Jiri Prochazka",
                    "fighter_b": "Carlos Ulberg",
                    "event_date": "2026-04-12T04:00:00+00:00",
                    "market_event_date": "2026-04-11 17:00:00+00",
                    "weight_class": "Light Heavyweight",
                }
            ]
        },
    )
    monkeypatch.setattr("src.strategy.llm_operator.load_decision_log", lambda: [])
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_tracker_decision_log",
        lambda: [
            {
                "type": "decision",
                "timestamp": "2026-04-08T21:39:08.389937+00:00",
                "trader": "M",
                "decision_id": "M_1",
                "fighter_a": "Jiri Prochazka",
                "fighter_b": "Carlos Ulberg",
                "event_date": "2026-04-12T04:00:00+00:00",
                "market_event_date": "2026-04-11 17:00:00+00",
                "status": "outside_window",
                "summary": "Outside tracker window",
                "rationale": "Model Tracker skipped this fight because it is outside the tracker window.",
            },
            {
                "type": "decision",
                "timestamp": "2026-04-08T21:39:09.106666+00:00",
                "trader": "G",
                "decision_id": "G_1",
                "fighter_a": "Jiri Prochazka",
                "fighter_b": "Carlos Ulberg",
                "event_date": "2026-04-12T04:00:00+00:00",
                "market_event_date": "2026-04-11 17:00:00+00",
                "status": "outside_window",
                "summary": "Outside tracker window",
                "rationale": "Gemini Tracker skipped this fight because it is outside the tracker window.",
            },
        ],
    )
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: SimpleNamespace(
            bets=[
                {
                    "fighter": "Carlos Ulberg",
                    "opponent": "Jiri Prochazka",
                    "side": "a",
                    "edge": 0.026,
                    "reason": "Model edge",
                    "event_date": "2026-04-12T04:00:00+00:00",
                    "placed_at": "2026-04-08T21:39:07+00:00",
                    "_ledger_path": "bet_ledger_single.json",
                }
            ]
        ),
    )

    client = web_app.app.test_client()
    response = client.get("/api/tracker-decisions")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1

    fight = payload["fights"][0]
    assert fight["fighter_a"] == "Jiri Prochazka"
    assert fight["fighter_b"] == "Carlos Ulberg"
    assert fight["event_group_date"] == "2026-04-11"
    assert fight["S"]["status"] == "bet"
    assert fight["S"]["text"] == "Carlos Ulberg"
    assert fight["M"]["status"] == "outside_window"
    assert fight["G"]["status"] == "outside_window"
