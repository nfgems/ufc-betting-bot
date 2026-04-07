import json
from types import SimpleNamespace

from src.strategy import llm_operator
from src.web import app as web_app


def test_api_operator_decisions_returns_ufc_entries_with_no_store(monkeypatch):
    monkeypatch.setattr(
        llm_operator,
        "load_decision_log",
        lambda: [
            {
                "verdict": "PASS",
                "timestamp": "2026-03-23T21:55:14.235844+00:00",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "bet_on": "Fighter A",
            }
        ],
    )

    client = web_app.app.test_client()

    response = client.get("/api/operator-decisions")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    payload = response.get_json()
    assert payload["count"] == 1
    assert payload["decisions"][0]["sport"] == "ufc"


def test_api_operator_decisions_can_limit_results(monkeypatch):
    monkeypatch.setattr(
        llm_operator,
        "load_decision_log",
        lambda: [
            {
                "verdict": "PASS",
                "timestamp": "2026-03-24T02:41:57.517679+00:00",
                "fighter_a": "New Fighter",
                "fighter_b": "New Opponent",
            },
            {
                "verdict": "BLOCK",
                "timestamp": "2026-03-23T02:41:57.517679+00:00",
                "fighter_a": "Old Fighter",
                "fighter_b": "Old Opponent",
            },
        ],
    )

    client = web_app.app.test_client()

    response = client.get("/api/operator-decisions?limit=1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    assert payload["total_count"] == 2
    assert [decision["fighter_a"] for decision in payload["decisions"]] == ["New Fighter"]


def test_api_tracker_decisions_merges_operator_and_tracker_logs(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "_load_prediction_payload",
        lambda include_global_feature_importance=False: {
            "predictions": [
                {
                    "fighter_a": "Fighter A",
                    "fighter_b": "Fighter B",
                    "event_date": "2027-04-12T23:00:00Z",
                    "weight_class": "Lightweight",
                }
            ]
        },
    )
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: SimpleNamespace(bets=[]),
    )
    monkeypatch.setattr(
        llm_operator,
        "load_decision_log",
        lambda: [
            {
                "verdict": "BLOCK",
                "timestamp": "2026-03-23T21:55:14.235844+00:00",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "bet_on": "Fighter A",
                "rationale": "Operator vetoed the fight",
                "event_date": "2027-04-12T23:00:00Z",
            }
        ],
    )
    monkeypatch.setattr(
        llm_operator,
        "load_tracker_decision_log",
        lambda: [
            {
                "type": "decision",
                "decision_id": "m-1",
                "trader": "M",
                "timestamp": "2026-03-23T21:55:14.235844+00:00",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "pick": "Fighter A",
                "status": "eligible",
                "summary": "Pick: Fighter A",
                "rationale": "Model picked Fighter A",
                "event_date": "2027-04-12T23:00:00Z",
                "market_event_date": "2027-04-12T23:00:00Z",
                "sources": [],
            },
            {
                "type": "outcome",
                "decision_id": "m-1",
                "trader": "M",
                "timestamp": "2026-03-23T21:56:14.235844+00:00",
                "bet_placed": True,
                "order_status": "placed",
                "order_type": "market",
            },
            {
                "type": "decision",
                "decision_id": "g-1",
                "trader": "G",
                "timestamp": "2026-03-23T21:57:14.235844+00:00",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "status": "outside_window",
                "summary": "Outside tracker window",
                "rationale": "Too far from the event",
                "event_date": "2027-04-12T23:00:00Z",
                "market_event_date": "2027-04-12T23:00:00Z",
                "sources": [],
            },
        ],
    )

    client = web_app.app.test_client()

    response = client.get("/api/tracker-decisions")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    fight = payload["fights"][0]
    assert fight["fighter_a"] == "Fighter A"
    assert fight["S"]["status"] == "blocked"
    assert fight["S"]["rationale"] == "Operator vetoed the fight"
    assert fight["M"]["text"] == "Fighter A"
    assert fight["M"]["bet_placed"] is True
    assert fight["G"]["status"] == "outside_window"
